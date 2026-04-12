# ACP Overall Index

This is the only `index.md` under `anica/docs/acp`.
All subfolders should use `catalog.md` (not `index.md`) to avoid naming confusion.

## Agent Read Order (Mandatory)
1. Read this file first.
2. Pick one section from the table below.
3. Open that section's `catalog.md` (or direct doc if no catalog exists).
4. Open only the single best-matching document.
5. If still ambiguous, ask one clarifying question.

## Agent Response Rules
- Prefer documented facts from ACP docs over assumptions.
- If no match exists, say: `Not documented yet`.
- Explain known limitations clearly and separate them from bugs/data corruption.
- Keep answers in Markdown.
- Keep answers concise by default; expand only when user asks.

## Tool Argument Language Rules
- User-visible replies may follow the user language.
- ACP tool-call arguments must use canonical English values.
- When user intent is not in English, translate/normalize to English before tool calls.
- Avoid non-English routing keywords and enum-like values in ACP tool arguments.

## Source Of Truth
- Runtime behavior and request/response schema are defined by implementation in `anica/src/api/*` and `anica/src/api/transport_acp.rs`.
- ACP docs in `anica/docs/acp/*` are routing/explanation docs for users and agents.
- If docs conflict with implementation, follow implementation and update docs.

## Section Map

| Section | Use When | Entry |
|---|---|---|
| Limitations | User reports behavior that looks like a bug but may be expected. | `anica/docs/acp/limitation/catalog.md` |
| Recovery | User asks autosave, recovery draft behavior, unsaved changes, or crash-recovery policy. | `anica/docs/acp/recovery/catalog.md` |
| Export | User asks export mode, preset, codec, or ffmpeg-related export guidance. | `anica/docs/acp/export/catalog.md` |
| Tools | User asks what ACP tools exist, how to call them, params, or failure cases. | `anica/docs/acp/tools/catalog.md` |
| User Guide | User asks CLI setup/install/login steps for external command tools (for example Codex CLI, Gemini CLI, or Claude CLI). | `anica/docs/acp/userguide/catalog.md` |
| Intent Phrases | User asks intent routing phrase dictionaries (multi-language trigger words). | `anica/docs/acp/intent-phrases/catalog.md` |
| MotionLoom | User asks MotionLoom DSL syntax, scope (`adjustment/clip/fusion`), process categories, kernel selection, or effect authoring workflow. | `anica/docs/acp/motionloom/catalog.md` |

## Naming Convention (Keep This)
- Top level router: `anica/docs/acp/index.md` (only one).
- Subfolder router: `anica/docs/acp/<topic>/catalog.md`.
- Recovery docs: `REC-xxxx.md` (example: `REC-0001.md`).
- Limitation docs: `LIM-xxxx.md` (example: `LIM-0001.md`).
- Export docs: `EXP-xxxx.md` (example: `EXP-0001.md`).
- Tool docs: `TOOL-xxxx.md` (example: `TOOL-0001.md`).
- User guide docs: `UG-xxxx.md` (example: `UG-0001.md`).
- Intent docs: `INTENT-xxxx.md` (example: `INTENT-0001.md`).
- MotionLoom docs: `ML-xxxx.md` (example: `ML-0001.md`).
