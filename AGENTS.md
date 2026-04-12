## Mandatory rule for this repo
These instructions apply to both Codex and Claude.

Project source of truth for coding behavior:
- This `AGENTS.md` is authoritative for repository coding and development tasks.
- Local skill files can be kept as reference material, but this file defines enforceable rules.

Core development rules:
- For any code you add or significantly modify, include concise English comments that explain the purpose of the code block.
- Do not introduce new `Result<T, String>` APIs in reusable Rust code. Prefer typed errors, and only convert to string at UI or transport boundaries when necessary.

Comment requirements:
- Comments must be written in English.
- Comments should explain "why" or "what this block does".
- Keep comments short and practical.
- Do not add obvious comments for trivial single-line assignments.
- Each source file should keep a 3-line header comment at the top:
  - `// =========================================`
  - `// =========================================`
  - `// <repo-relative-file-path>`
- The third line must match the file location, for example:
  - `// src/app/editor_window.rs`
  - `// src/ui/video_preview.rs`
  - `// crates/video-engine/src/video.rs`

Crate boundary rule:
- Never edit any file under `crates/gpui-0.2.2/`. Treat it as read-only.
- If a missing feature blocks progress, work around it in higher layers (`src/`, `crates/motionloom/`, or `crates/gpui-video-renderer/`) and discuss with maintainer before considering crate changes.

## Mandatory preview rendering policy
- macOS preview GStreamer pipeline must use `NV12`.
- On macOS, video preview output must render via `paint_surface` with `NV12`.
- On Windows and Linux, preview pipeline/output must use `BGRA` and render via `paint_image`.
- Image media rendered as video content must use `BGRA` (do not route image rendering through `NV12` surface path).

## ACP policy separation
- This file defines coding/development behavior.
- ACP runtime chat behavior is defined separately by `anica/src/api/AGENTS.md`.
- That ACP-specific file does not replace or weaken the repository coding/development rules in this file.

## ACP tool-call language policy
- User-facing chat replies can follow the user's language (for example Chinese, Japanese, English).
- ACP tool-call arguments must use canonical English values.
- Before any ACP tool call, normalize/translate non-English user intent into English argument values.
- Do not pass non-English routing keywords or enum-like values into ACP tool arguments.

## MotionLoom DSL Stability Contract
Hard rule: do not introduce breaking MotionLoom DSL changes without explicit owner approval in the same task.

### Baseline compatibility (must keep)
- `Pass.kernel` is optional by default.
- `Pass.effect` is required.
- `Pass.in` and `Pass.out` are required.
- `Graph.scope` accepted values remain `layer | clip | scene`.
- Existing template scripts must remain parseable unless owner explicitly requests migration.

### Proposal-first rule (must ask before editing)
Before changing any parser/schema behavior, stop and ask for approval if the change affects:
- required vs optional fields
- accepted enum/string values
- default value semantics
- template DSL output format
- effect naming or kernel routing behavior

### If a DSL change is approved
- Update parser + runtime + templates + ACP docs in one task.
- Add/adjust tests for backward compatibility and new behavior.
- Include a short migration note with before/after examples.

### Forbidden without explicit approval
- Making optional DSL fields required (example: forcing `Pass.kernel`).
- Renaming/removing active effects or scopes.
- Silent behavior changes that alter existing user scripts.
