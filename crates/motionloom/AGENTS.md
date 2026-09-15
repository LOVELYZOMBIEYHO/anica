## MotionLoom Crate Rules

These instructions apply to MotionLoom parser, renderer, examples, tests, and
documentation in this crate. They are intended for both Codex and other LLM
coding agents.

## Documentation Source of Truth

- `README.md` is the user-facing overview for humans and LLMs.
- `PUBLIC_API.md` defines the intended public API layers and stability policy.
- `src/lib.rs` and `src/api.rs` provide the docs.rs/rustdoc entry points.
- This `AGENTS.md` is only for coding-agent workflow rules. Do not duplicate
  full user documentation here; link or update the source documents above.
- Keep `motionloom::api` as the recommended stable integration surface.
- Keep `motionloom::experimental` public for advanced/editor APIs that may
  change faster than the stable surface.
- Keep crate-root re-exports for compatibility unless there is an explicit
  migration plan and all Anica usages are updated.

## Skill Routing

- Reusable task workflows live under `skills/<skill-name>/SKILL.md`.
- Before taking task actions in this crate, inspect the available skill names
  and frontmatter descriptions under `skills/*/SKILL.md`.
- If the user names a skill, or the request clearly matches a skill's
  description, read that `SKILL.md` completely before acting and follow it for
  the current task.
- When a selected skill links supporting files such as `references/`, read only
  the files required for the current workflow, but read each selected file
  completely.
- Select the smallest set of skills that covers the request. Do not load every
  skill or combine unrelated workflows by default.
- Skills supplement this `AGENTS.md` and the parent repository instructions;
  they do not override mandatory repository rules. Explicit user instructions
  take precedence over optional skill guidance.
- When adding, renaming, or removing a skill, update the registry below in the
  same change.

### Skill Registry

- `motionloom-image-to-meshasset`: build and iteratively fit a universal
  `MeshAsset` from one or more image references with the image-analysis,
  mesh-evaluation, and fingerprint-safe proposal APIs. Read
  `skills/motionloom-image-to-meshasset/SKILL.md`.

## DSL Authoring Rules

- Keep MotionLoom scripts parseable by the current parser. Do not invent syntax
  because it looks natural in XML/JSX.
- `curve(...)` points must use numeric keyframe values:
  `curve("time:value[:ease], time:value[:ease]")`.
- Do not put function calls such as `random(...)`, `sin(...)`, `cos(...)`, or
  `floor(...)` inside the value field of a `curve(...)` point.
- Use standalone expressions for procedural motion, for example:
  `x="random(-26,26,floor($time.sec*2)+17) + 28*sin($time.sec*0.8)"`.
- Use `curve(...)` for deterministic smooth numeric interpolation only.
- Do not use `$index` in scene expressions unless the parser/runtime explicitly
  supports it. Prefer `Repeat` attributes such as `xStep`, `yStep`,
  `rotationStep`, and `opacityStep` for per-instance variation.
- Do not animate string attributes such as `Text.value` or `Path.d` with
  `curve(...)`. Use fixed strings/paths, opacity, transform, trim, or numeric
  properties instead.
- Do not duplicate attributes on one node. For example, do not write both
  `x="600"` and `x={curve(...)}` on the same `<Group>`.
- Keep `<Present ... />` as the final direct child of `<Graph>`.

## Parser Changes

- Do not broaden parser behavior just to accept one generated example. First
  rewrite the example into the existing DSL style.
- If a new DSL feature is genuinely needed, propose it first with:
  - intended syntax,
  - parser/runtime impact,
  - backward compatibility risk,
  - at least one before/after example.
- Any approved DSL change must update parser tests, renderer behavior, README,
  and examples together.
