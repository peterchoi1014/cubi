use serde_json::Value;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

pub(crate) struct OpenAiFixture {
    pub(crate) base_url: String,
    pub(crate) requests: Arc<Mutex<Vec<Value>>>,
    shutdown: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl OpenAiFixture {
    pub(crate) async fn spawn(responses: Vec<Value>) -> Self {
        assert!(!responses.is_empty(), "fixture needs at least one response");
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let (shutdown, mut shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let mut response_index = 0usize;
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => {
                        let (mut socket, _) = accepted.unwrap();
                        let request = read_json_request(&mut socket).await;
                        recorded.lock().unwrap().push(request);
                        let response = responses
                            .get(response_index)
                            .unwrap_or_else(|| responses.last().unwrap());
                        response_index = response_index.saturating_add(1);
                        let body = response.to_string();
                        let wire = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        socket.write_all(wire.as_bytes()).await.unwrap();
                    }
                }
            }
        });
        Self {
            base_url: format!("http://{address}/v1"),
            requests,
            shutdown,
            task,
        }
    }

    pub(crate) async fn shutdown(self) {
        let _ = self.shutdown.send(());
        self.task.await.unwrap();
    }
}

async fn read_json_request(socket: &mut tokio::net::TcpStream) -> Value {
    let mut request = Vec::new();
    let (header_end, content_length) = loop {
        let mut chunk = [0_u8; 4096];
        let count = socket.read(&mut chunk).await.unwrap();
        assert!(count > 0, "connection closed before request headers");
        request.extend_from_slice(&chunk[..count]);
        if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            let header_end = index + 4;
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            break (header_end, content_length);
        }
    };
    while request.len() < header_end + content_length {
        let mut chunk = [0_u8; 4096];
        let count = socket.read(&mut chunk).await.unwrap();
        assert!(count > 0, "connection closed before request body");
        request.extend_from_slice(&chunk[..count]);
    }
    serde_json::from_slice(&request[header_end..header_end + content_length]).unwrap()
}
