# Installing Cubi

## Prerequisites

### Required

- **Rust** 1.85 or later — [rustup.rs](https://rustup.rs/)
- A local LLM backend — one of:
  - **[Ollama](https://ollama.ai/)** (recommended)
  - **[llama.cpp](https://github.com/ggerganov/llama.cpp)** `llama-server`
  - **[LM Studio](https://lmstudio.ai/)**
  - Any other OpenAI-compatible local server

### System Requirements

- macOS (Apple Silicon or Intel) or Linux
- 8 GB+ RAM (16 GB+ recommended for larger models)
- 10 GB+ free disk space for model weights

## 1. Install your backend

### Ollama (recommended)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start the daemon (leave running in another terminal)
ollama serve
```

### llama-server, LM Studio, etc.

Follow each project's docs. Cubi only needs a reachable OpenAI-compatible
`/v1/chat/completions` endpoint — see [Using a non-Ollama backend](#using-a-non-ollama-backend)
below.

## 2. Pull a model

```bash
# Recommended default — most reliable tool-calling (~5.2 GB)
ollama pull qwen3:8b
```

Other options:

| Model | Size | Speed | Quality | Use case |
| --- | --- | --- | --- | --- |
| `qwen3:8b` | 5.2 GB | ⚡ | ⭐⭐⭐⭐⭐ | **Default** — most reliable tool-calling |
| `qwen3:4b` | 2.6 GB | ⚡⚡ | ⭐⭐⭐⭐ | Tool-capable, fits smaller machines |
| `devstral` | 14 GB | ⚡ | ⭐⭐⭐⭐⭐ | Best for code/agent workflows, 131K context |
| `qwen2.5:3b` | 1.9 GB | ⚡⚡⚡ | ⭐⭐⭐ | Smaller tool-capable model |
| `phi4-mini` | 2.5 GB | ⚡⚡ | ⭐⭐⭐ | Microsoft's tool-capable mini |
| `mistral:7b` | 4.1 GB | ⚡ | ⭐⭐⭐⭐ | High-quality general-purpose |
| `llama3.1:8b` | 4.7 GB | ⚡ | ⭐⭐⭐⭐ | General-purpose tool-capable |

Pull any with `ollama pull <model-name>` and list installed with `ollama list`.

## 3. Install Cubi

### From source

```bash
git clone https://github.com/peterchoi1014/cubi.git
cd cubi
cargo install --path .
# Binary lands at $CARGO_HOME/bin/cubi (usually ~/.cargo/bin/cubi)
```

Or, for a local build without installing globally:

```bash
cargo build --release
# Binary: ./target/release/cubi
```

### From crates.io

Once the crate is published:

```bash
cargo install cubi
```

The `tree-sitter` Cargo feature is **on by default** and adds ~4.5 MB to the
release binary for the Rust/Python/JS/TS parsers used by the `repo_map`
tool and `/repomap` command. To build the smallest possible binary (with a
regex-based outline fallback instead), disable it:

```bash
cargo install --no-default-features cubi
```

## 4. Verify

```bash
ollama serve &       # if not already running
cubi
```

Then inside the REPL:

```
You: /doctor
```

`/doctor` probes Ollama reachability, lists models, and confirms the config
directory is writable.

## Choosing a model

The CLI uses `qwen3:8b` by default. Override at startup with `CUBI_MODEL`, or
switch interactively with `/model`:

```bash
CUBI_MODEL=qwen3:4b cubi
```

## Optional features

Cubi ships with optional capabilities behind cargo feature flags so the
default binary stays under 8 MB. Enable them at install time:

```bash
# Headless-browser tool family (browser_open / browser_eval /
# browser_screenshot / browser_text / browser_close). Pulls in
# chromiumoxide (~+4 MB to the release binary) and requires a
# Chromium or Chrome binary on PATH at runtime.
cargo install --features browser --path .
```

If the browser binary isn't on PATH, point Cubi at it with
`CHROME=/path/to/chromium`. `cubi doctor` reports whether the launch
probe succeeds.

## Using a non-Ollama backend

Cubi's OpenAI-compatible client speaks to any local server that exposes
`/v1/chat/completions`. Set two environment variables before launching:

```bash
# llama.cpp's llama-server on its default port
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8080/v1

# LM Studio on its default port
export OPENAI_API_KEY=lm-studio
export OPENAI_BASE_URL=http://localhost:1234/v1
```

Then `cubi` and `/doctor` will probe and list models from that server.

### Large long-context models (e.g. GLM-5.2)

Cubi recognizes the [GLM-5.2](https://z.ai/blog/glm-5.2) family (Z.ai / Zhipu,
MIT-licensed) and maps it to its **1M-token context window** for token
accounting and compaction. It is a tool-capable agentic coding model, so it is
not flagged by the small-model tool-calling warning. Point cubi at any
OpenAI-compatible server hosting it and select it with `CUBI_MODEL`:

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
CUBI_MODEL=glm-5.2 cubi          # Ollama tag, HF repo id (zai-org/GLM-5.2),
                                 # and provider-prefixed forms all resolve
```

GLM-5.2 is a flagship model — expect large weights and substantial RAM/VRAM
requirements; run a quantized build on consumer hardware.

### Tool-calling caveats

- **LM Studio**'s public chat-completions docs page omits `tools` from the
  supported-parameters table, but the field *is* forwarded to llama.cpp and
  works for tool-capable models (qwen3, llama3.1+, mistral-small, devstral).
  If an older LM Studio build rejects `tools`, upgrade.
- **llama-server** requires the model's chat template to advertise `tools`
  support; check `llama-server --help` for `--chat-template` if calls aren't
  parsed as `tool_calls`.
- **Qwen3 `<think>...</think>` blocks** are stripped from assistant content by
  default. Set `CUBI_KEEP_THINKING=1` if you need them in raw form.

## Shell completions

```bash
cubi completions bash   # or zsh, fish
```

Pipe the output to the file your shell expects, then `source` it.

## Troubleshooting

If anything in the install pipeline fails, run `cubi doctor` for a structured
diagnosis, or see [`docs/troubleshooting.md`](docs/troubleshooting.md).
