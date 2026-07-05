# Security Policy

## Supported versions

Cubi is developed on `main` and shipped as tagged releases. Security fixes land
on `main` and in the next release; only the **latest release** is supported.

| Version | Supported |
| --- | --- |
| latest release / `main` | ✅ |
| older releases | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately through GitHub's **Report a vulnerability** flow:

1. Go to <https://github.com/peterchoi1014/cubi/security/advisories/new>
   (repository **Security** tab → **Report a vulnerability**).
2. Include: affected version/commit, OS, a description, reproduction steps, and
   the impact.

If private advisories are unavailable to you, open a minimal public issue that
says only "security report — please enable private contact" (no details) and a
maintainer will follow up.

This is a small open-source project maintained on a best-effort basis; there is
no guaranteed response SLA, but reports are taken seriously and triaged as soon
as practical. Please allow reasonable time for a fix before public disclosure.

## Scope

Cubi runs AI models that can invoke tools which **read/write files and execute
shell commands** on your machine. Its safety model is:

- **Per-directory trust** — write/exec tool paths are gated by a trust prompt
  per directory.
- **Plan mode** (`/plan`) — a read-only mode that refuses mutating tools.
- **Admin policy** — an optional policy file can ship a tool deny-list.
- **Tamper-evident receipts** (`--receipts`) — a hash-chained (optionally
  Ed25519-signed) audit log of tool calls and lifecycle events.

Vulnerabilities of particular interest include: **trust/permission bypasses**
(writing or executing outside a trusted directory), **plan-mode escapes**,
**policy deny-list bypasses**, **receipt-chain forgery** that verifies as
intact, path traversal, and injection that escalates a tool's intended scope.

Note that a model deciding to run a destructive command **within an
already-trusted directory that the user approved** is expected behavior, not a
vulnerability — trust is the security boundary.

Secrets (API keys, OAuth tokens) are read from environment variables / files
under `~/.cubi/` and are never intended to be logged; a leak of a secret into
logs, receipts, or session files **is** in scope.
