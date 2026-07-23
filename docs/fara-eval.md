# Evaluating Microsoft Fara1.5 as a local model for Cubi

Feasibility notes on wiring [`microsoft/Fara1.5-27B`](https://huggingface.co/microsoft/Fara1.5-27B)
into Cubi as a local model. Short version: **Fara1.5 is not a drop-in
replacement for Cubi's default model, and it is not directly Ollama-loadable.**
It is a specialized browser *computer-use agent* (CUA), not a general-purpose
chat/tool-calling model, and adopting it would be a feature project, not a
config change.

## What Fara1.5 actually is

Fara1.5 is Microsoft Research's family of browser computer-use agents (4B / 9B /
27B), built on Qwen3.5 via supervised fine-tuning. It:

- takes **pixel-level browser screenshots** (plus conversation history) as input,
- runs an **Observe → Think → Act** loop, and
- emits **coordinate-level GUI actions** — click at (x, y), type, scroll, visit
  URL, web search, context-management ops — in its own action schema.

Reported results are strong for its niche: ~72% on Online-Mind2Web and 88.6% on
WebVoyager for the 27B, beating OpenAI Operator and Gemini 2.5 Computer Use.
But the niche is *browser navigation by looking at the screen*, not answering
questions, editing code, or driving Cubi's built-in shell/git/MCP tools.

Sources:
- <https://www.microsoft.com/en-us/research/articles/fara1-5-computer-use-agent/>
- <https://github.com/microsoft/fara>
- <https://huggingface.co/microsoft/Fara1.5-27B>

## How Cubi runs models today

- **Backends** (`src/llm.rs`): Cubi talks to either Ollama (`/api/chat`) or any
  OpenAI-compatible endpoint (`/v1/chat/completions`). `create_provider()`
  (`src/llm.rs:936`) selects the OpenAI path whenever `OPENAI_API_KEY` /
  `CUBI_API_KEY` is set, with the endpoint from `OPENAI_BASE_URL` /
  `CUBI_BASE_URL`. So *transport-wise*, a vLLM-served Fara endpoint is already
  reachable.
- **Messages are text-only.** The core `Message` type carries `content: String`
  (`src/ollama.rs:58`), and the OpenAI request message is likewise
  `content: String` (`src/llm.rs:265`). There is no multimodal/image-parts
  content anywhere in the message model. The README roadmap still lists
  "Multi-modal support (images, audio)" as a *TODO*.
- **The browser tools don't feed pixels back.** With the `browser` feature,
  Cubi drives Chromium via chromiumoxide (`src/browser_tool.rs`) using
  `browser_open` / `browser_eval` / `browser_screenshot` / `browser_text` /
  `browser_close`. It navigates by **JS eval and CSS selectors**, not by mouse
  coordinates, and `browser_screenshot` writes a PNG to disk
  (`src/builtin_tools.rs:2426`) — the image is never returned to the model.

## The three hard mismatches

1. **Vision input.** Fara *requires* screenshots as model input; Cubi has no way
   to put an image into a message. Adding it touches the core `Message` type
   (`src/ollama.rs`), the OpenAI serializer `OaiRequestMessage`
   (`src/llm.rs:261`), the Ollama serializer, and session persistence
   (`src/sessions.rs`). This is the roadmap's "multi-modal support" item.

2. **Serving / "local".** There is **no GGUF for Fara1.5**, so it is *not*
   loadable in Ollama — Microsoft recommends self-hosting via **vLLM**, which
   exposes an OpenAI-compatible endpoint but needs a real GPU (the 27B most of
   all). Only the older **Fara-7B** ships GGUF variants for Ollama/LM Studio.
   Cubi's headline "local model" story is Ollama-centric; Fara1.5 would only run
   through the OpenAI-compatible path against a GPU-backed vLLM server.

3. **Action space / agent loop.** Fara emits coordinate-based GUI actions in its
   own schema and expects the CUA Observe–Think–Act harness. Cubi's agent loop
   (`src/agent_loop.rs`) is a generic native-tool-calling loop over shell/fs/git/
   MCP tools, and its browser tools are selector/JS-based. Fara would not know
   how to call Cubi's tools, and Cubi cannot execute Fara's pixel actions.

## Recommendation

- **As Cubi's general default model: no.** Fara1.5 is specialized for browser
  GUI navigation. For the normal agent loop, a general instruct model with
  native tool-calling (today's `qwen3:8b` default) remains the right choice.
- **Pointing Cubi at a vLLM Fara endpoint today "just to try it":** possible via
  `OPENAI_BASE_URL`/`OPENAI_API_KEY`, but it will underperform — without image
  input and the CUA action harness, Fara is being used far outside its design,
  and Cubi can't feed it the screenshots it depends on.
- **If the real goal is browser computer-use in Cubi**, that is a genuine
  feature, and Fara is a reasonable model choice for it. Minimum scope:
  1. multimodal message content (image parts through `Message` →
     `OaiRequestMessage` → session persistence);
  2. return `browser_screenshot` output to the model as an image instead of a
     path;
  3. a coordinate-level browser action layer wired to chromiumoxide input
     events (mouse move/click, key input, scroll) to match Fara's action space;
  4. a CUA-style Observe–Think–Act loop (either a new mode or a specialized
     agent) distinct from the general tool-calling loop.
  For local feasibility, target the smaller **Fara-9B/4B** (or the GGUF
  **Fara-7B** so Ollama works) before the 27B, which needs substantial GPU VRAM.

## Decision

We are chasing **"a better local generalist model"**, so Fara1.5 does not apply
(it is a specialized browser CUA, per the mismatches above). The generalist
upgrade landed alongside this note: the default model was bumped from
`qwen3:8b` to **`qwen3.5:9b`** — the direct Qwen3 successor on Ollama, same
weight class, 256K native context (up from 32K), with improved native
tool-calling. See `src/main.rs` (`DEFAULT_MODEL`), the context-window registry
in `src/llm.rs`, and the updated onboarding/help/README/INSTALL copy. The
`qwen3.5:4b` variant is the small-machine fallback. The nightly `cubi bench`
regression baseline stays pinned to `qwen3:8b` in CI on purpose.

If the **"browser computer-use as a Cubi feature"** path is ever revived, Fara
becomes relevant again: scope the multimodal + browser-action-layer work above
as its own milestone, prototype against Fara-9B on a GPU box (or Fara-7B GGUF
locally), and treat the 27B as the quality ceiling rather than the starting
point.
