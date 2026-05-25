# AI Chat CLI

A powerful command-line AI chat application built with Rust, featuring local AI model inference through Ollama and distributed task execution capabilities via Repartir.

<div align="center">

![Rust](https://img.shields.io/badge/rust-1.92.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)

</div>

## ✨ Features

- 🤖 **Local AI Models** - Run AI models completely offline using Ollama
- 💬 **Interactive Chat** - Beautiful colored terminal interface with conversation history
- ⚡ **Streaming + native tool-calling agent loop** - Tokens stream live; the
  model can call built-in or MCP tools, see the results, and keep going (up
  to 12 round-trips per turn)
- 🧰 **Built-in tools** - shell (`bash`, `shell` for cross-platform), filesystem
  (`read_file`, `write_file`, `edit_file`, `list_files`, `search_glob`, `grep`),
  git (`worktree`), web (`web_fetch`, `web_search`), long-lived `bash` REPL
  (`repl_start`/`eval`/`close`), Jupyter `notebook` (`list`/`read`/`insert`/
  `replace`/`delete`), LSP-backed code intel (`lsp` for hover/definition/
  references via your language server), time (`sleep`, `schedule`,
  `prevent_sleep`), structured output (`brief`, `synthetic_output`),
  inter-agent messaging (`send_message`, `recv_messages`, `remote_trigger`),
  OS notifications (`notify`), plus a `think` no-op and an `agent_run`
  meta-tool for spawning focused subagents
- 🔌 **MCP (Model Context Protocol) support** - Load external tool servers from
  `~/.ai-chat-cli/mcp.json` and call them with `/mcp-tools`, `/mcp-call`,
  `/mcp-reload`; list and render MCP prompts with `/mcp-prompts`
- 🌿 **Git workflow** - `/diff`, `/commit`, `/review`, `/worktree`, `/branch`,
  `/tag`, `/files` shell out to your installed `git` and respect plan mode
- 🛡️ **Project trust + plan mode + admin policy** - Tools refuse to write/exec
  outside trusted directories; `/plan` toggles a read-only mode that blocks
  every write/exec path. Manage trust with `/trust` (current dir) and
  `/add-dir <path>` (additional dirs). Admins can push a read-only
  `policy.json` (`/etc/ai-chat-cli/policy.json` or `~/.ai-chat-cli/policy.json`)
  whose tool deny-list overrides any user allow; inspect with `/policy`
- 🧠 **Project memory + persistent memdir** - Auto-injected `AICHAT.md` per
  project (`/memory`, `/memory-reload`, `/init`) plus cross-session notes in
  `~/.ai-chat-cli/memdir/` (`/memdir`, `/memdir-add`, `/memdir-rm`, `/memdir-clear`)
- ✅ **Todos** - `/todos`, `/todo-add`, `/todo-done`, `/todo-rm`, `/todo-clear`
  with on-disk per-project persistence
- 💾 **Conversation Management** - Save and load chat sessions as JSON files
  (`/save`, `/load`); export to Markdown (`/export`); every turn is
  auto-checkpointed and recoverable via `/sessions` / `/resume`; trim or
  summarize context with `/rewind` (also rolls back any `edit_file` /
  `write_file` mutations from the rewound turns) and `/compact`
- 🧩 **Plugins + Skills** - Drop reusable Markdown skill packs at
  `~/.ai-chat-cli/skills/` and namespaced command bundles at
  `~/.ai-chat-cli/plugins/<name>/commands/*.md` (invoked as
  `/<plugin>:<command>`); reload both with `/reload-plugins`
- 🎨 **Themes + output styles** - `/theme auto|light|dark` and
  `/output-style concise|markdown|explanatory` persist in
  `~/.ai-chat-cli/config.json`; the chosen output style is injected as a
  system steering prompt on every turn
- 🪝 **Hooks** - `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`,
  `SessionStart`, `Notification`; manage with `/hooks list|add|rm`
- 🔁 **Cross-machine settings sync** - Wrap `~/.ai-chat-cli/` in git with
  `/settings-sync init <remote>`, then `/settings-sync push` / `pull`
  to move config, memdir, skills, and plugins between machines
- 📊 **Opt-in telemetry** - Set `telemetry = true` in
  `~/.ai-chat-cli/config.json` (or `AICHAT_TELEMETRY=1`) to log every tool
  call as one JSON line in `~/.ai-chat-cli/telemetry.log`
- 💡 **Tip-of-the-day** - A short tip is shown on startup (TTY only); see
  more with `/tip` and supplement the built-in pool by dropping plaintext
  files in `~/.ai-chat-cli/tips/`
- 📦 **Batch Processing** - Process multiple prompts from text files
- 🔄 **Model Switching** - Switch between different AI models on the fly
- 🎨 **Colored Output** - Syntax-highlighted responses with emoji indicators
- ⌨️ **Readline Support** - Command history with up/down arrow navigation
- ⚡ **Built with Rust** - Fast, safe, and memory-efficient

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Basic Chat](#basic-chat)
  - [Commands](#commands)
  - [Batch Processing](#batch-processing)
  - [Conversation Management](#conversation-management)
- [Available Models](#available-models)
- [Architecture](#architecture)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Prerequisites

### Required

- **Rust** 1.92.0 or later ([Install Rust](https://rustup.rs/))
- **Ollama** - Local AI runtime ([Install Ollama](https://ollama.ai/))

### System Requirements

- macOS (Apple Silicon or Intel) or Linux
- 8GB+ RAM (16GB+ recommended for larger models)
- 10GB+ free disk space for AI models

## 📥 Installation

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### 2. Download an AI Model

```bash
# Recommended: Fast and lightweight (1.3GB)
ollama pull llama3.2:1b

# Or choose another model:
ollama pull llama3.2:3b    # Medium size (2GB)
ollama pull phi3:mini      # Microsoft's small model (2.3GB)
ollama pull mistral:7b     # High quality (4.1GB)
```

### 3. Clone and Build

```bash
# Clone the repository
git clone https://github.com/peterchoi1014/ai-chat-cli.git
cd ai-chat-cli

# Build in release mode
cargo build --release

# Binary will be available at: ./target/release/ai-chat-cli
```

## 🚀 Quick Start

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, run the CLI
cargo run --release

# Or run the compiled binary directly
./target/release/ai-chat-cli
```

### Choosing a model

By default the CLI uses `llama3.2:1b`. Override at startup with the
`AI_CHAT_CLI_MODEL` environment variable, or switch interactively with the
`/model` command:

```bash
# Pick a different default just for this session
AI_CHAT_CLI_MODEL=mistral:7b cargo run --release
```

You should see:

```
============================================================
  AI Chat CLI - Powered by Repartir
============================================================

Type /help to list all available slash commands.

Start chatting! (Ctrl+C to interrupt, /quit to exit)

You: 
```

## 💡 Usage

### Basic Chat

Simply type your message and press Enter:

```
You: What is Rust programming language?
AI: Rust is a systems programming language that focuses on safety, 
    speed, and concurrency. It achieves memory safety without using 
    garbage collection...

You: Can you write a hello world program?
AI: Sure! Here's a simple hello world in Rust:

    fn main() {
        println!("Hello, World!");
    }

You: Thanks!
AI: You're welcome! Feel free to ask if you have more questions.
```

### Commands

The slash-command surface is a single source of truth defined in
[`src/commands.rs`](src/commands.rs); `/help` lists everything at runtime. The
groups below mirror that registry.

#### General

| Command | Description |
| --- | --- |
| `/help` | Show all available commands |
| `/status` | Show session status (model, trust, plan mode, counts) |
| `/version` | Show version |
| `/quit` (alias `/exit`) | Exit the chat |

#### Model & history

| Command | Description |
| --- | --- |
| `/model [name]` | Show or switch the active Ollama model |
| `/history` | Show conversation history |
| `/clear` | Clear conversation history |
| `/rewind [n]` | Remove the last `n` exchanges (default 1) |
| `/compact` | Summarize old turns to reduce context length |

#### Conversation persistence

| Command | Description |
| --- | --- |
| `/save [-f] <f.json>` | Save conversation (`-f` overwrites) |
| `/load <f.json>` | Load conversation |
| `/export [-f] <f.md>` | Export conversation as Markdown |
| `/batch <f>` | Process a file of prompts (one per line) |
| `/sessions` | List auto-saved sessions for this project |
| `/resume [id]` | Resume the latest (or named) auto-saved session |

#### Project memory & todos

| Command | Description |
| --- | --- |
| `/init` | Create a starter `AICHAT.md` |
| `/memory` | Show project memory (`AICHAT.md`) |
| `/memory-reload` | Re-read `AICHAT.md` from disk |
| `/memdir` | List cross-session persistent memories |
| `/memdir-add <text>` | Add a persistent memory |
| `/memdir-rm <n>` | Remove memory by index |
| `/memdir-clear` | Clear all persistent memories |
| `/todos` | List todos |
| `/todo-add <text>` | Add a todo |
| `/todo-done <n>` | Mark todo `n` as done |
| `/todo-rm <n>` | Remove todo `n` |
| `/todo-clear` | Clear all todos |
| `/ask <q>` | Record a single-turn clarifying question |

#### Plan mode & trust

| Command | Description |
| --- | --- |
| `/plan` | Toggle plan mode (read-only; refuses write/exec tools) |
| `/trust [revoke]` | Trust the current project directory (or undo) |
| `/add-dir <path>` | Trust an additional directory for write/exec tools |

#### Git workflow

| Command | Description |
| --- | --- |
| `/diff [path]` | Show `git diff` for the working tree |
| `/commit [-a] <msg>` | Run `git commit` (`-a` stages tracked files first) |
| `/commit-push-pr [-a] <msg>` | Commit, push, and print a GitHub PR URL |
| `/undo [hard]` | Undo the latest commit (or hard reset HEAD~1) |
| `/review` | Ask the model to review the current `git diff` |
| `/security-review` | Ask the model to security-review the current `git diff` |
| `/pr_comments [pr#]` | Show PR review comments via `gh pr view --comments` |
| `/autofix-pr [pr#]` | Fetch PR review comments and ask the model to propose fixes |
| `/worktree [list \| add <path> [branch] \| remove <path>]` | Manage git worktrees (`add` auto-trusts the new path) |
| `/branch [list \| create <name> \| switch <name>]` | List, create, or switch git branches |
| `/tag [list \| <name> \| create <name> [-m <msg>]]` | List or create git tags |
| `/files` | List files tracked by git in this project |
| `/init-verifiers` | Detect project verifier commands and save to `.aichat-verifiers.json` |

#### MCP (Model Context Protocol)

| Command | Description |
| --- | --- |
| `/mcp` | Show overall MCP status (servers, tools, resources) |
| `/mcp-tools` | List available MCP tools |
| `/mcp-call <tool> <json-args>` | Call an MCP tool |
| `/mcp-reload` | Reload MCP configuration from `~/.ai-chat-cli/mcp.json` |
| `/mcp-resources [server]` | List MCP resources |
| `/mcp-read <uri>` | Read an MCP resource by URI |
| `/plugin` | List plugins discovered in `~/.ai-chat-cli/plugins/` |
| `/reload-plugins` | Rescan the plugins / skills directory |
| `/skills [list \| run <name>]` | List or run reusable Markdown skills |
| `/hooks [list \| add <event> <cmd> \| rm <n>]` | Manage lifecycle hooks |

#### Agent control & theming

| Command | Description |
| --- | --- |
| `/agents` | List background / sub-agent sessions |
| `/tasks` | Alias for `/todos` |
| `/teleport <path>` | Change `cwd` to a (preferably trusted) directory |
| `/passes [n]` | Show or set the agent-loop max passes (1..=12) |
| `/effort [low \| medium \| high]` | Set agent effort (maps to pass budget) |
| `/theme [auto \| light \| dark]` | Show or set the colored-output theme |
| `/color [on \| off]` | Toggle colored output for this session |
| `/output-style [concise \| markdown \| explanatory]` | Set the assistant output style |
| `/statusline` | Show the contents of the status line |
| `/keybindings` | Show the active rustyline keybindings |
| `/vim [on \| off]` | Toggle vim-style readline editing |

#### Auth & privacy

| Command | Description |
| --- | --- |
| `/login [provider]` | Print how to record an API key (env-var based) |
| `/logout [provider]` | Forget the in-process API key for a provider |
| `/oauth-refresh` | Show OAuth status (no OAuth backend yet) |
| `/privacy-settings [telemetry on \| off]` | Show or set local privacy preferences |
| `/sandbox-toggle` | Alias for `/plan` (strict-sandbox mode) |
| `/reset-limits` | Clear in-process rate-limit / retry backoff state |

#### Diagnostics & transparency

| Command | Description |
| --- | --- |
| `/doctor` | Run environment health checks (Ollama, model, config dir, `git`) |
| `/env` | Show resolved runtime info (version, model, cwd, plan mode, etc.) |
| `/config` | Print the current `~/.ai-chat-cli/config.json` |
| `/permissions` | List trusted directories and gated built-in tools |
| `/tool-allow <name>` / `/tool-deny <name>` | Per-tool allow / deny in the trust store |
| `/stats` / `/usage` | Show session statistics |
| `/cost` | Show estimated session cost (always $0 for local Ollama) |
| `/perf-issue [summary]` | Print a pre-filled GitHub perf-issue URL |
| `/heapdump` | Print process resident-set / heap info if available |
| `/debug-tool-call [on \| off]` | Toggle verbose tool-call debug logging |
| `/bug [summary]` | Print a pre-filled GitHub issue URL with environment details |
| `/issue [title]` | Print a pre-filled GitHub feature-request URL |
| `/feedback [text]` | Print the feedback URL |

#### Lifecycle & sharing

| Command | Description |
| --- | --- |
| `/upgrade` | Print upgrade instructions |
| `/install` | Print install instructions |
| `/install-github-app` / `/install-slack-app` | Placeholders (no app shipped) |
| `/share <file.md>` | Export this conversation to a shareable Markdown file |
| `/copy` | Copy the last assistant message to the system clipboard |
| `/release-notes` | Print release notes for the current version |
| `/stickers` | Print a friendly ASCII sticker sheet |
| `/settings-sync init <remote> \| push [msg] \| pull \| status` | Git-backed sync of `~/.ai-chat-cli/` across machines |
| `/policy` | Show the read-only admin policy overlay (deny-list + source path) |
| `/tip` | Print a quick tip about using ai-chat-cli |
| `/mcp-prompts [server[:name]]` | List MCP prompts, or fetch a specific one |

#### Examples

```
You: /model
Current model: llama3.2:1b

You: /model mistral:7b
✓ Switched to model: mistral:7b

You: /history

Conversation History:
------------------------------------------------------------
You [1]: What is Rust?
AI [2]: Rust is a systems programming language...
------------------------------------------------------------

You: /quit
Goodbye!
```

### Batch Processing

Process multiple prompts from a text file:

#### 1. Create a prompts file

Create `prompts.txt`:
```
What is Rust?
Write a hello world program in Python
Explain how recursion works
What is the difference between a vector and an array?
```

#### 2. Run batch processing

```
You: /batch prompts.txt

📋 Processing 4 prompts...

▶ [1/4] What is Rust?
✓ Rust is a systems programming language...

▶ [2/4] Write a hello world program in Python
✓ print("Hello, World!")

▶ [3/4] Explain how recursion works
✓ Recursion is a programming technique...

▶ [4/4] What is the difference between a vector and an array?
✓ A vector is a dynamic array...

✓ Batch processing complete
```

### Conversation Management

#### Save a conversation

```
You: Hello, my name is Alice
AI: Nice to meet you, Alice! How can I help you today?

You: Tell me about machine learning
AI: Machine learning is a subset of artificial intelligence...

You: /save my_chat.json
✓ Conversation saved to my_chat.json
```

#### Load a conversation

```
You: /load my_chat.json
✓ Conversation loaded from my_chat.json

You: What's my name?
AI: Your name is Alice.
```

The loaded conversation maintains full context, so the AI remembers previous interactions.

## 🤖 Available Models

Popular models you can use with Ollama:

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `llama3.2:1b` | 1.3GB | ⚡⚡⚡ | ⭐⭐ | Quick responses, limited hardware |
| `llama3.2:3b` | 2GB | ⚡⚡ | ⭐⭐⭐ | Balanced performance |
| `phi3:mini` | 2.3GB | ⚡⚡ | ⭐⭐⭐ | Microsoft's efficient model |
| `mistral:7b` | 4.1GB | ⚡ | ⭐⭐⭐⭐ | High-quality responses |
| `llama3:8b` | 4.7GB | ⚡ | ⭐⭐⭐⭐ | Latest Llama model |

Install any model with:
```bash
ollama pull <model-name>
```

List installed models:
```bash
ollama list
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLI Interface                      │
│              (Colored Terminal UI)                   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│                  AI Executor                         │
│         (Task Management & Coordination)             │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│                Ollama Client                         │
│         (HTTP API Communication)                     │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              Ollama Service                          │
│         (Local AI Model Inference)                   │
└─────────────────────────────────────────────────────┘
```

### Project Structure

```
ai-chat-cli/
├── src/
│   ├── main.rs            # Application entry point
│   ├── cli.rs             # Terminal UI, command dispatch, agent-loop driver
│   ├── commands.rs        # Slash-command registry (single source of truth)
│   ├── agent_loop.rs      # Streaming tool-calling loop + `agent_run` meta-tool
│   ├── executor.rs        # AI executor & model switching
│   ├── llm.rs             # Provider abstraction
│   ├── ollama.rs          # Ollama HTTP client (streaming + tool calls)
│   ├── builtin_tools.rs   # bash, fs, web, repl, notebook, worktree, lsp, think
│   ├── lsp_client.rs      # JSON-RPC client used by the `lsp` builtin tool
│   ├── mcp_client.rs      # MCP transport (stdio + HTTP)
│   ├── mcp_config.rs      # `~/.ai-chat-cli/mcp.json` loader
│   ├── mcp_manager.rs     # MCP server lifecycle & tool routing
│   ├── permissions.rs     # Project trust store + plan-mode gates
│   ├── project_memory.rs  # AICHAT.md discovery & loading
│   ├── memdir.rs          # Cross-session persistent memory store
│   ├── sessions.rs        # Per-project auto-saved session checkpoints
│   ├── todos.rs           # Per-project todo list
│   ├── hooks.rs           # Lifecycle hook registry (PreToolUse, etc.)
│   ├── file_mentions.rs   # `@file` mentions + user-defined Markdown commands
│   ├── git_cmds.rs        # Shell-out helpers for the git slash commands
│   ├── onboarding.rs      # First-run setup
│   └── distributed.rs     # (Reserved) distributed task plumbing
├── Cargo.toml             # Dependencies
├── ROADMAP.md             # Architectural roadmap & shipped/open items
└── README.md              # This file
```

### Key Components

- **CLI** (`cli.rs`) — terminal UI, slash-command dispatch, drives the
  streaming agent loop and persists sessions.
- **Slash-command registry** (`commands.rs`) — every command's trigger, usage,
  help line, and `Cmd` tag in one table so `/help`, the welcome banner, and
  dispatch can't drift apart.
- **Agent loop** (`agent_loop.rs`) — runs the tool-calling loop (≤12
  round-trips per turn) and exposes the `agent_run` meta-tool for subagents.
- **Built-in tools** (`builtin_tools.rs`) — shell, filesystem, web, REPL,
  Jupyter notebook, git worktree, and LSP code-intel implementations.
- **MCP layer** (`mcp_client.rs`, `mcp_config.rs`, `mcp_manager.rs`) — loads
  external MCP servers and exposes their tools to the model.
- **Permissions** (`permissions.rs`) — trust store consulted by every
  write/exec tool; also enforces `/plan` mode.
- **Memory & sessions** (`project_memory.rs`, `memdir.rs`, `sessions.rs`,
  `todos.rs`) — AICHAT.md auto-injection, cross-session memdir, per-project
  session checkpoints, and todos.
- **Executor & Ollama client** (`executor.rs`, `llm.rs`, `ollama.rs`) — model
  switching and streaming chat-completion calls with tool support.

## 🛠️ Development

### Build from source

```bash
# Debug build (faster compilation)
cargo build

# Release build (optimized)
cargo build --release
```

### Run tests

```bash
cargo test
```

### Check code

```bash
# Check for errors
cargo check

# Run clippy linter
cargo clippy

# Format code
cargo fmt
```

### Dependencies

Main dependencies (see `Cargo.toml` for the complete list):

- `tokio` — async runtime (rt-multi-thread, macros, process, io-util, time)
- `reqwest` — HTTP client for Ollama and MCP (with `stream` feature)
- `futures-util` — stream combinators for token-by-token streaming
- `serde` / `serde_json` — JSON serialization
- `colored` — terminal colors
- `rustyline` — readline-style input with history
- `anyhow` — error handling
- `dirs` — cross-platform home/config directory resolution
- `uuid` — request IDs

## 🐛 Troubleshooting

### Ollama connection failed

**Error**: `Failed to send request to Ollama`

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

### Model not found

**Error**: `Model 'llama3.2:1b' not found`

**Solution**:
```bash
# List installed models
ollama list

# Pull the required model
ollama pull llama3.2:1b
```

### Slow responses

**Solutions**:
1. Use a smaller model: `llama3.2:1b` instead of `mistral:7b`
2. Close other applications to free up RAM
3. Use GPU acceleration if available (Ollama automatic)

### Command not recognized

Make sure you're using the correct prefix:
- ✅ `/help` (correct)
- ❌ `help` (incorrect - missing slash)

### File save/load errors

**Error**: `Permission denied` or `No such file or directory`

**Solution**:
```bash
# Use absolute path
/save /Users/username/chats/conversation.json

# Or ensure current directory is writable
ls -la
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow Rust style guidelines (`cargo fmt`)
- Add tests for new features
- Update documentation
- Keep commits atomic and well-described

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Ollama](https://ollama.ai/)** - Local AI model runtime
- **[Repartir](https://github.com/paiml/repartir)** - Distributed computing framework
- **Rust Community** - For amazing tools and libraries

## 📚 Resources

- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Rust Book](https://doc.rust-lang.org/book/)
- [Repartir Documentation](https://paiml.github.io/repartir/)

## 🗺️ Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the full plan (built-in tools, slash
commands, subsystems, and implementation priorities derived from an
architectural review of similar tools).

Highlights still to come (see [`ROADMAP.md`](ROADMAP.md) for the full list):

- [ ] RAG (Retrieval Augmented Generation) support
- [ ] Multi-modal support (images, audio)
- [ ] Web interface
- [ ] Distributed inference across remote workers
- [ ] Conversation search and tagging
- [ ] Export to additional formats (PDF)
- [ ] Plugin / skills system for extensibility
- [ ] Cross-platform shell tool (`pwsh` on Windows)

---

<div align="center">

**Built with ❤️ using Rust and Ollama**

[Report Bug](https://github.com/peterchoi1014/ai-chat-cli/issues) · [Request Feature](https://github.com/peterchoi1014/ai-chat-cli/issues)

</div>
