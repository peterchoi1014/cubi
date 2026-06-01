//! Tamper-evident receipts log (`--receipts <path>`).
//!
//! Every tool call and key lifecycle event in a session produces an
//! append-only JSONL entry whose `this_hash` chains over `prev_hash` ||
//! the canonical serialization of the rest of the entry. A separate
//! sidecar directory at `<path>.payloads/<sha>.json` stores the full
//! tool args/results so the receipts file itself stays small and
//! grep-able.
//!
//! Optional Ed25519 signing kicks in when a key has been provisioned
//! with `cubi keys init`; without one, the chain is still verifiable
//! against payload tampering, just not against an attacker who can
//! rewrite the whole file.
//!
//! Verification is implemented as a pure function over the on-disk
//! files so the binary can re-walk a chain produced by an older
//! version without sharing in-memory state.

use anyhow::{Context, Result, anyhow};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One entry in the receipts log. Each variant maps onto a specific
/// `event` string in the JSONL output; per-variant metadata (e.g. tool
/// name, ok flag) is hoisted into top-level columns so awk/jq users
/// can filter without parsing the payload sidecar.
#[derive(Debug, Clone)]
pub enum ReceiptEvent {
    SessionStart { model: String, cwd: PathBuf },
    UserMessage,
    ToolCall { name: String },
    ToolResult { name: String, ok: bool },
    AssistantMessage,
    SessionEnd,
}

impl ReceiptEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::SessionStart { .. } => "session_start",
            Self::UserMessage => "user_message",
            Self::ToolCall { .. } => "tool_call",
            Self::ToolResult { .. } => "tool_result",
            Self::AssistantMessage => "assistant_message",
            Self::SessionEnd => "session_end",
        }
    }
}

/// Mutable inner state guarded by the writer's mutex.
#[derive(Debug)]
struct Inner {
    seq: u64,
    prev_hash: String,
}

/// Append-only writer for a hash-chained receipts log.
///
/// Cheap to share across tasks: wrap in `Arc` and clone the `Arc` —
/// the internal `Mutex<Inner>` serializes writes so chain entries
/// can't interleave even if two tool calls finish concurrently.
#[derive(Debug)]
pub struct ReceiptsWriter {
    path: PathBuf,
    payload_dir: PathBuf,
    signing_key: Option<SigningKey>,
    state: Mutex<Inner>,
}

impl ReceiptsWriter {
    /// Open (or create) a receipts log at `path`. The sidecar payload
    /// directory `<path>.payloads/` is created on demand on the first
    /// write so a probe followed by no events leaves no detritus.
    pub fn open(path: &Path, signing_key: Option<SigningKey>) -> Result<Self> {
        let path = path.to_path_buf();
        let payload_dir = sidecar_dir(&path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create receipts parent dir {}", parent.display())
                })?;
            }
        }
        // Recover the chain tip if the file already exists so resuming
        // appends to an in-progress log instead of forking the chain.
        let (seq, prev_hash) = recover_tail(&path)?;
        Ok(Self {
            path,
            payload_dir,
            signing_key,
            state: Mutex::new(Inner { seq, prev_hash }),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Touch-create the receipts file (and parent / payload dirs) so
    /// the caller can surface IO failures at startup rather than on
    /// the first event mid-session.
    pub fn probe(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        if !self.payload_dir.exists() {
            std::fs::create_dir_all(&self.payload_dir)?;
        }
        Ok(())
    }

    /// Append one event. `payload` is the full data blob (tool args,
    /// tool result, message text wrapped in an object, etc.); the
    /// JSONL line carries only its SHA-256 hash plus event metadata,
    /// and the payload is written to
    /// `<path>.payloads/<payload_sha256>.json`.
    ///
    /// Best-effort: filesystem failures are surfaced to the caller so
    /// the integration layer can log a single warning and continue
    /// without aborting the session.
    pub fn write(&self, event: ReceiptEvent, payload: &Value) -> Result<()> {
        let payload_bytes = canonical_bytes(payload);
        let payload_hash = sha256_hex(&payload_bytes);

        // Persist the payload sidecar first. Pretty-print so a human
        // poking at the file can read it without re-formatting.
        if !self.payload_dir.exists() {
            std::fs::create_dir_all(&self.payload_dir).with_context(|| {
                format!(
                    "failed to create receipts payload dir {}",
                    self.payload_dir.display()
                )
            })?;
        }
        let payload_path = self.payload_dir.join(format!("{payload_hash}.json"));
        if !payload_path.exists() {
            let pretty = serde_json::to_string_pretty(payload)
                .context("failed to serialize receipt payload")?;
            std::fs::write(&payload_path, pretty).with_context(|| {
                format!("failed to write receipt payload {}", payload_path.display())
            })?;
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("receipts writer mutex poisoned"))?;
        state.seq += 1;

        // Build the base record (everything except this_hash + sig).
        let mut record = serde_json::Map::new();
        record.insert("seq".into(), json!(state.seq));
        record.insert("ts".into(), json!(now_rfc3339()));
        record.insert("event".into(), json!(event.kind()));
        match &event {
            ReceiptEvent::SessionStart { model, cwd } => {
                record.insert("model".into(), json!(model));
                record.insert("cwd".into(), json!(cwd.display().to_string()));
            }
            ReceiptEvent::ToolCall { name } => {
                record.insert("name".into(), json!(name));
            }
            ReceiptEvent::ToolResult { name, ok } => {
                record.insert("name".into(), json!(name));
                record.insert("ok".into(), json!(ok));
            }
            ReceiptEvent::UserMessage
            | ReceiptEvent::AssistantMessage
            | ReceiptEvent::SessionEnd => {}
        }
        record.insert("payload_sha256".into(), json!(payload_hash));
        record.insert("prev_hash".into(), json!(state.prev_hash));

        let record_value = Value::Object(record);
        let chain_input = chain_input_bytes(&state.prev_hash, &record_value);
        let this_hash = sha256_hex(&chain_input);

        let mut record_map = match record_value {
            Value::Object(m) => m,
            _ => unreachable!(),
        };
        record_map.insert("this_hash".into(), json!(this_hash));

        if let Some(sk) = &self.signing_key {
            let sig = sk.sign(this_hash.as_bytes());
            record_map.insert("sig".into(), json!(B64.encode(sig.to_bytes())));
        }

        let line = serde_json::to_string(&Value::Object(record_map))
            .context("failed to serialize receipt line")?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("failed to open receipts file {}", self.path.display()))?;
        writeln!(file, "{line}")
            .with_context(|| format!("failed to write receipts line to {}", self.path.display()))?;

        state.prev_hash = this_hash;
        Ok(())
    }
}

/// Canonical sorted-key JSON serialization. Objects iterate in
/// lexicographic key order via `BTreeMap`; arrays preserve their input
/// order; scalars use `serde_json`'s default rendering. Documented in
/// `DEVELOPMENT.md` so external verifiers (Python, Go) can match
/// byte-for-byte.
pub fn canonical_bytes(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_canonical(&mut out, value);
    out
}

fn write_canonical(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<&String, &Value> = map.iter().collect();
            out.push(b'{');
            for (i, (k, v)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                // serde_json escapes strings consistently — fine for keys.
                serde_json::to_writer(&mut *out, k).expect("string serialization is infallible");
                out.push(b':');
                write_canonical(out, v);
            }
            out.push(b'}');
        }
        Value::Array(items) => {
            out.push(b'[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(out, v);
            }
            out.push(b']');
        }
        other => {
            serde_json::to_writer(&mut *out, other).expect("scalar serialization is infallible");
        }
    }
}

/// Bytes hashed to compute `this_hash`: the previous chain hash
/// concatenated with the canonical serialization of the record
/// *without* its `this_hash` or `sig` fields.
fn chain_input_bytes(prev_hash: &str, record: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(prev_hash.as_bytes());
    out.extend(canonical_bytes(record));
    out
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

fn sidecar_dir(path: &Path) -> PathBuf {
    // `<path>.payloads` regardless of the receipts file's extension —
    // keeps the rule simple and grep-friendly.
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".payloads");
    match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(name),
        _ => PathBuf::from(name),
    }
}

/// Re-read the last well-formed line of an existing receipts file so a
/// new writer continues the chain instead of restarting at seq=1.
fn recover_tail(path: &Path) -> Result<(u64, String)> {
    if !path.exists() {
        return Ok((0, String::new()));
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read receipts file {}", path.display()))?;
    let last = raw.lines().rfind(|l| !l.trim().is_empty());
    let Some(line) = last else {
        return Ok((0, String::new()));
    };
    let value: Value = serde_json::from_str(line)
        .with_context(|| format!("receipts tail line is not valid JSON in {}", path.display()))?;
    let seq = value
        .get("seq")
        .and_then(|s| s.as_u64())
        .ok_or_else(|| anyhow!("receipts tail line missing seq"))?;
    let this_hash = value
        .get("this_hash")
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow!("receipts tail line missing this_hash"))?
        .to_string();
    Ok((seq, this_hash))
}

/// RFC3339 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`). Hand-formatted to
/// avoid a date crate; shared with `event_sink::now_rfc3339`.
fn now_rfc3339() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (y, m, d, hour, minute, second) = crate::sessions::civil_from_unix(now);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

// ---------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------

#[derive(Debug)]
pub struct VerifyOptions {
    /// Re-hash every payload sidecar to ensure it still matches the
    /// `payload_sha256` claim in the JSONL entry. On by default.
    pub verify_payloads: bool,
    /// Optional pinned verification key. When `None`, `sig` fields are
    /// ignored (the chain is still walked). When `Some(_)`, every
    /// entry MUST carry a valid signature.
    pub pub_key: Option<VerifyingKey>,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            verify_payloads: true,
            pub_key: None,
        }
    }
}

#[derive(Debug)]
pub struct VerifyReport {
    pub entries: u64,
    pub signed: u64,
}

#[derive(Debug)]
pub enum VerifyError {
    /// I/O failure (missing file, can't read payload sidecar).
    Io { message: String },
    /// Chain hash, signature, or payload-hash mismatch at a known seq.
    Tamper { seq: u64, reason: String },
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { message } => write!(f, "{message}"),
            Self::Tamper { seq, reason } => write!(f, "{reason} at seq={seq}"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// Walk a receipts file top-to-bottom, recomputing every `this_hash`
/// and (when enabled) re-hashing every payload sidecar. Returns the
/// number of entries verified, or a typed error pinpointing the seq
/// at which tampering was detected.
pub fn verify_file(
    path: &Path,
    opts: &VerifyOptions,
) -> std::result::Result<VerifyReport, VerifyError> {
    let raw = std::fs::read_to_string(path).map_err(|e| VerifyError::Io {
        message: format!("failed to read receipts file {}: {}", path.display(), e),
    })?;
    let payload_dir = sidecar_dir(path);

    let mut prev_hash = String::new();
    let mut expected_seq: u64 = 0;
    let mut signed_count: u64 = 0;
    let mut entries: u64 = 0;

    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|e| VerifyError::Tamper {
            seq: expected_seq + 1,
            reason: format!("invalid JSON on line {}: {}", lineno + 1, e),
        })?;
        let map = value.as_object().ok_or_else(|| VerifyError::Tamper {
            seq: expected_seq + 1,
            reason: format!("non-object entry on line {}", lineno + 1),
        })?;
        let seq = map
            .get("seq")
            .and_then(|s| s.as_u64())
            .ok_or_else(|| VerifyError::Tamper {
                seq: expected_seq + 1,
                reason: "missing seq".into(),
            })?;
        expected_seq += 1;
        if seq != expected_seq {
            return Err(VerifyError::Tamper {
                seq,
                reason: format!("seq out of order (expected {expected_seq}, got {seq})"),
            });
        }
        let this_hash = map
            .get("this_hash")
            .and_then(|s| s.as_str())
            .ok_or_else(|| VerifyError::Tamper {
                seq,
                reason: "missing this_hash".into(),
            })?
            .to_string();
        let entry_prev = map
            .get("prev_hash")
            .and_then(|s| s.as_str())
            .ok_or_else(|| VerifyError::Tamper {
                seq,
                reason: "missing prev_hash".into(),
            })?;
        if entry_prev != prev_hash {
            return Err(VerifyError::Tamper {
                seq,
                reason: "chain broken".into(),
            });
        }

        // Recompute this_hash over the record minus this_hash + sig.
        let mut without = map.clone();
        without.remove("this_hash");
        without.remove("sig");
        let chain_input = chain_input_bytes(&prev_hash, &Value::Object(without));
        let computed = sha256_hex(&chain_input);
        if computed != this_hash {
            return Err(VerifyError::Tamper {
                seq,
                reason: "chain broken".into(),
            });
        }

        // Optional payload-sha verification.
        let claimed_payload = map
            .get("payload_sha256")
            .and_then(|s| s.as_str())
            .ok_or_else(|| VerifyError::Tamper {
                seq,
                reason: "missing payload_sha256".into(),
            })?;
        if opts.verify_payloads {
            let payload_path = payload_dir.join(format!("{claimed_payload}.json"));
            let payload_raw =
                std::fs::read_to_string(&payload_path).map_err(|e| VerifyError::Tamper {
                    seq,
                    reason: format!(
                        "payload sidecar missing or unreadable ({}): {}",
                        payload_path.display(),
                        e
                    ),
                })?;
            let parsed: Value =
                serde_json::from_str(&payload_raw).map_err(|e| VerifyError::Tamper {
                    seq,
                    reason: format!("payload sidecar is not valid JSON: {e}"),
                })?;
            let recomputed = sha256_hex(&canonical_bytes(&parsed));
            if recomputed != claimed_payload {
                return Err(VerifyError::Tamper {
                    seq,
                    reason: "payload hash mismatch".into(),
                });
            }
        }

        // Optional signature verification.
        if let Some(pk) = &opts.pub_key {
            let sig_b64 =
                map.get("sig")
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| VerifyError::Tamper {
                        seq,
                        reason: "signature missing".into(),
                    })?;
            let sig_bytes = B64.decode(sig_b64).map_err(|e| VerifyError::Tamper {
                seq,
                reason: format!("signature is not valid base64: {e}"),
            })?;
            let sig_bytes: [u8; 64] =
                sig_bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| VerifyError::Tamper {
                        seq,
                        reason: "signature has wrong length".into(),
                    })?;
            let sig = Signature::from_bytes(&sig_bytes);
            pk.verify(this_hash.as_bytes(), &sig)
                .map_err(|_| VerifyError::Tamper {
                    seq,
                    reason: "signature invalid".into(),
                })?;
            signed_count += 1;
        } else if map.contains_key("sig") {
            signed_count += 1;
        }

        prev_hash = this_hash;
        entries += 1;
    }

    Ok(VerifyReport {
        entries,
        signed: signed_count,
    })
}

// ---------------------------------------------------------------------
// Keypair management
// ---------------------------------------------------------------------

/// Default key directory: `~/.cubi/keys/`. Returns `None` when the
/// home directory itself cannot be resolved.
pub fn keys_dir() -> Option<PathBuf> {
    crate::sessions::home_dir().map(|h| h.join(".cubi").join("keys"))
}

pub fn default_priv_path() -> Option<PathBuf> {
    keys_dir().map(|d| d.join("ed25519.priv"))
}

#[allow(dead_code)]
pub fn default_pub_path() -> Option<PathBuf> {
    keys_dir().map(|d| d.join("ed25519.pub"))
}

/// Load the default signing key if one has been provisioned. Returns
/// `Ok(None)` when no key exists — callers fall back to unsigned
/// receipts in that case rather than erroring.
pub fn load_default_signing_key() -> Result<Option<SigningKey>> {
    let Some(priv_path) = default_priv_path() else {
        return Ok(None);
    };
    if !priv_path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&priv_path)
        .with_context(|| format!("failed to read {}", priv_path.display()))?;
    let seed = B64.decode(raw.trim()).with_context(|| {
        format!(
            "{} is not valid base64 (expected raw 32-byte Ed25519 seed)",
            priv_path.display()
        )
    })?;
    let seed: [u8; 32] = seed.as_slice().try_into().map_err(|_| {
        anyhow!(
            "{} must contain a 32-byte Ed25519 seed (got {} bytes)",
            priv_path.display(),
            seed.len()
        )
    })?;
    Ok(Some(SigningKey::from_bytes(&seed)))
}

/// Generate and persist a fresh Ed25519 keypair. Refuses to overwrite
/// existing material unless `force` is set. Returns the path to the
/// public-key file and its `ssh-ed25519`-formatted single-line content
/// so the caller can echo it for the user to publish.
pub fn init_keypair(force: bool) -> Result<(PathBuf, String)> {
    let dir = keys_dir().ok_or_else(|| anyhow!("could not resolve home directory"))?;
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let priv_path = dir.join("ed25519.priv");
    let pub_path = dir.join("ed25519.pub");
    if !force && (priv_path.exists() || pub_path.exists()) {
        return Err(anyhow!(
            "key material already exists at {} — re-run with --force to overwrite",
            dir.display()
        ));
    }

    let mut seed = [0u8; 32];
    getrandom::getrandom(&mut seed).map_err(|e| anyhow!("failed to read OS randomness: {e}"))?;
    let sk = SigningKey::from_bytes(&seed);
    let vk = sk.verifying_key();

    std::fs::write(&priv_path, B64.encode(seed))
        .with_context(|| format!("failed to write {}", priv_path.display()))?;
    // Lock down the private key to 0600 on Unix; on Windows we rely on
    // the user profile ACL, same as the rest of `~/.cubi`.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&priv_path)?.permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(&priv_path, perms)?;
    }

    let ssh_line = format_ssh_ed25519(&vk);
    std::fs::write(&pub_path, format!("{ssh_line}\n"))
        .with_context(|| format!("failed to write {}", pub_path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&pub_path)?.permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&pub_path, perms)?;
    }
    Ok((pub_path, ssh_line))
}

/// Read a verifying key from disk. Accepts the `ssh-ed25519 <b64>
/// [comment]` form written by `init_keypair`, and falls back to a
/// bare 32-byte base64 string for callers that hand-export the raw
/// material.
pub fn load_verifying_key(path: &Path) -> Result<VerifyingKey> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read public key {}", path.display()))?;
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("ssh-ed25519") {
        let token = rest
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow!("malformed ssh-ed25519 line in {}", path.display()))?;
        let wire = B64.decode(token).with_context(|| {
            format!("ssh-ed25519 key is not valid base64 in {}", path.display())
        })?;
        let pk_bytes = decode_ssh_ed25519(&wire)?;
        return VerifyingKey::from_bytes(&pk_bytes).map_err(|e| anyhow!(e.to_string()));
    }
    let bytes = B64
        .decode(trimmed)
        .with_context(|| format!("public key is not valid base64 in {}", path.display()))?;
    let arr: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("public key must be 32 bytes (got {})", bytes.len()))?;
    VerifyingKey::from_bytes(&arr).map_err(|e| anyhow!(e.to_string()))
}

/// SSH wire format for an Ed25519 public key:
/// `[u32:11]["ssh-ed25519"][u32:32][32-byte key]`. We emit the same
/// line shape as `ssh-keygen -y` so downstream tooling can ingest
/// the `.pub` file unmodified.
fn format_ssh_ed25519(vk: &VerifyingKey) -> String {
    let key_bytes = vk.to_bytes();
    let mut wire = Vec::with_capacity(4 + 11 + 4 + 32);
    wire.extend_from_slice(&(11u32).to_be_bytes());
    wire.extend_from_slice(b"ssh-ed25519");
    wire.extend_from_slice(&(32u32).to_be_bytes());
    wire.extend_from_slice(&key_bytes);
    format!("ssh-ed25519 {} cubi-receipts", B64.encode(&wire))
}

fn decode_ssh_ed25519(wire: &[u8]) -> Result<[u8; 32]> {
    let mut cur = 0usize;
    let read_u32 = |buf: &[u8], cur: &mut usize| -> Result<u32> {
        if *cur + 4 > buf.len() {
            return Err(anyhow!("truncated ssh wire format"));
        }
        let v = u32::from_be_bytes([buf[*cur], buf[*cur + 1], buf[*cur + 2], buf[*cur + 3]]);
        *cur += 4;
        Ok(v)
    };
    let name_len = read_u32(wire, &mut cur)? as usize;
    if cur + name_len > wire.len() || &wire[cur..cur + name_len] != b"ssh-ed25519" {
        return Err(anyhow!("not an ssh-ed25519 key"));
    }
    cur += name_len;
    let key_len = read_u32(wire, &mut cur)? as usize;
    if key_len != 32 || cur + 32 > wire.len() {
        return Err(anyhow!("ssh-ed25519 key has wrong length"));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&wire[cur..cur + 32]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cubi-receipts-{label}-{nanos}.jsonl"))
    }

    #[test]
    fn canonical_sorts_object_keys() {
        let v = json!({"b": 1, "a": [3, {"y": 2, "x": 1}]});
        let bytes = canonical_bytes(&v);
        assert_eq!(
            std::str::from_utf8(&bytes).unwrap(),
            r#"{"a":[3,{"x":1,"y":2}],"b":1}"#
        );
    }

    #[test]
    fn writer_chains_hashes_across_entries() {
        let path = tmp_path("chain");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(sidecar_dir(&path));
        let w = ReceiptsWriter::open(&path, None).unwrap();
        w.write(
            ReceiptEvent::SessionStart {
                model: "qwen3:4b".into(),
                cwd: PathBuf::from("/tmp"),
            },
            &json!({"model": "qwen3:4b"}),
        )
        .unwrap();
        w.write(
            ReceiptEvent::ToolCall {
                name: "bash".into(),
            },
            &json!({"command": "true"}),
        )
        .unwrap();
        let report = verify_file(&path, &VerifyOptions::default()).unwrap();
        assert_eq!(report.entries, 2);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(sidecar_dir(&path));
    }

    #[test]
    fn verify_detects_payload_tamper() {
        let path = tmp_path("payload-tamper");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(sidecar_dir(&path));
        let w = ReceiptsWriter::open(&path, None).unwrap();
        w.write(
            ReceiptEvent::ToolCall {
                name: "bash".into(),
            },
            &json!({"command": "echo hi"}),
        )
        .unwrap();
        // Overwrite the only payload sidecar with garbage.
        let dir = sidecar_dir(&path);
        let mut entries = std::fs::read_dir(&dir).unwrap();
        let entry = entries.next().unwrap().unwrap();
        std::fs::write(entry.path(), r#"{"command": "rm -rf /"}"#).unwrap();
        let err = verify_file(&path, &VerifyOptions::default()).unwrap_err();
        match err {
            VerifyError::Tamper { seq, .. } => assert_eq!(seq, 1),
            other => panic!("expected tamper, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_detects_chain_tamper() {
        let path = tmp_path("chain-tamper");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(sidecar_dir(&path));
        let w = ReceiptsWriter::open(&path, None).unwrap();
        w.write(ReceiptEvent::UserMessage, &json!({"text": "hi"}))
            .unwrap();
        w.write(ReceiptEvent::AssistantMessage, &json!({"text": "hello"}))
            .unwrap();
        // Replace one this_hash with the zero digest.
        let raw = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = raw.lines().map(|l| l.to_string()).collect();
        let mut v: Value = serde_json::from_str(&lines[1]).unwrap();
        v["this_hash"] = json!("0000000000000000000000000000000000000000000000000000000000000000");
        lines[1] = serde_json::to_string(&v).unwrap();
        std::fs::write(&path, lines.join("\n")).unwrap();
        let err = verify_file(&path, &VerifyOptions::default()).unwrap_err();
        match err {
            VerifyError::Tamper { seq, .. } => assert_eq!(seq, 2),
            other => panic!("expected tamper, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(sidecar_dir(&path));
    }

    #[test]
    fn signing_round_trip_verifies() {
        let path = tmp_path("sign");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(sidecar_dir(&path));
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).unwrap();
        let sk = SigningKey::from_bytes(&seed);
        let vk = sk.verifying_key();
        let w = ReceiptsWriter::open(&path, Some(sk)).unwrap();
        w.write(ReceiptEvent::UserMessage, &json!({"text": "hi"}))
            .unwrap();
        w.write(
            ReceiptEvent::ToolCall {
                name: "bash".into(),
            },
            &json!({"command": "true"}),
        )
        .unwrap();
        let opts = VerifyOptions {
            verify_payloads: true,
            pub_key: Some(vk),
        };
        let report = verify_file(&path, &opts).unwrap();
        assert_eq!(report.entries, 2);
        assert_eq!(report.signed, 2);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(sidecar_dir(&path));
    }

    #[test]
    fn signature_bit_flip_is_detected() {
        let path = tmp_path("sig-flip");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(sidecar_dir(&path));
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).unwrap();
        let sk = SigningKey::from_bytes(&seed);
        let vk = sk.verifying_key();
        let w = ReceiptsWriter::open(&path, Some(sk)).unwrap();
        w.write(ReceiptEvent::UserMessage, &json!({"text": "hi"}))
            .unwrap();
        // Flip one bit in the base64 signature.
        let raw = std::fs::read_to_string(&path).unwrap();
        let mut v: Value = serde_json::from_str(raw.trim()).unwrap();
        let sig_b64 = v["sig"].as_str().unwrap().to_string();
        let mut bytes = B64.decode(&sig_b64).unwrap();
        bytes[0] ^= 0x01;
        v["sig"] = json!(B64.encode(&bytes));
        std::fs::write(&path, serde_json::to_string(&v).unwrap()).unwrap();
        let opts = VerifyOptions {
            verify_payloads: true,
            pub_key: Some(vk),
        };
        let err = verify_file(&path, &opts).unwrap_err();
        match err {
            VerifyError::Tamper { seq, reason } => {
                assert_eq!(seq, 1);
                assert!(reason.contains("signature"), "got: {reason}");
            }
            other => panic!("expected tamper, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(sidecar_dir(&path));
    }

    #[test]
    fn ssh_format_round_trips() {
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).unwrap();
        let sk = SigningKey::from_bytes(&seed);
        let vk = sk.verifying_key();
        let line = format_ssh_ed25519(&vk);
        assert!(line.starts_with("ssh-ed25519 "));
        let tmp = tmp_path("sshpub");
        std::fs::write(&tmp, format!("{line}\n")).unwrap();
        let loaded = load_verifying_key(&tmp).unwrap();
        assert_eq!(loaded.to_bytes(), vk.to_bytes());
        let _ = std::fs::remove_file(&tmp);
    }
}
