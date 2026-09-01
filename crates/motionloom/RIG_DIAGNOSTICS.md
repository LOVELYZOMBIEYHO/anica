# Rig Evaluation and Pose Comparison

MotionLoom exposes rig diagnostics as an additive, read-only API. It does not
add a DSL tag, modify a `ModelProfile`, or write calibration into a scene.
Rust, CLI, WASM/JS, Action Editor hosts, and Anica can consume the same JSON.

## Evaluation

Use `SceneRenderer::evaluate_rig` with a `RigEvaluationRequest`. The sample can
be an absolute frame, seconds, or a normalized Action phase. Action-phase mode
resolves namespaced `ActionLibrary` imports through the production Scene path.

```json
{
  "actorId": "s77_actor",
  "sample": {
    "kind": "actionPhase",
    "actionId": "walk.standard_walk_loop",
    "phase": 0.5
  },
  "detail": "body",
  "includeScreenProjection": true,
  "includeMatrices": false
}
```

The returned `RigEvaluationReport` uses `schemaVersion: "1.0"` and records:

- document, model SHA-256 when bytes are locally available, skin/clip counts;
- semantic ModelProfile and Action fingerprints;
- active and inactive Action layers, local time, normalized phase, blend
  weight, mask, root-motion mode, and the actual pose driver;
- canonical bone mapping, target GLB node, parent, driver, declared/effective
  axis channels, and why an axis is bypassed;
- model rest, retargeted, post-constraint, post-contact/final Scene, and screen
  projection stages when available;
- selected ground, contact correction, foot lock, root correction, per-foot
  contact window, and before/after positions.

Hosts can discover the envelope through `rig_diagnostics_schema_json()` or the
WASM export `motionloom_rig_diagnostics_schema_json()`.

Unavailable data remains absent and is reflected by `capabilities`; it is not
invented from a rendered image.

## Comparison

`compare_humanoid_poses` aligns canonical bones and reports true quaternion
angular error, parent-relative local angular error, model-global angular error,
joint position error, child endpoint error normalized by body height, and the
first divergent stage. Quaternion `q` and `-q` are treated as the same
rotation. Thresholds and phase tolerance are configurable through
`RigComparisonOptions`.

The deterministic classifier checks evidence in this order:

1. resolved model asset hash;
2. canonical mapping coverage/count;
3. semantic ModelProfile fingerprint;
4. semantic Action fingerprint;
5. Action phase and active blend stack;
6. effective axis/driver;
7. retarget, constraint, contact, and camera-only divergence.

Every root-cause category includes confidence and evidence. The classifier is
intended to give an LLM concrete facts, not replace visual/artistic review.

## Calibration proposals

`propose_rig_calibration` returns suggestions only. It never applies them. In
particular, a baked humanoid reference reports `doNotChangeBoneAxis`, because
semantic `BoneAxisMap` channels do not control that active driver. Timing,
layering, asset, and mapping mismatches must be resolved before per-bone tuning.

## CLI

```text
cargo run -p motionloom --example rig_scene_evaluate -- scene.motionloom actor 120 report.json
cargo run -p motionloom --example rig_scene_evaluate -- scene.motionloom actor time:4.5 report.json
cargo run -p motionloom --example rig_scene_evaluate -- scene.motionloom actor phase:walk.standard_walk_loop:0.5 report.json
cargo run -p motionloom --example rig_compare -- reference.json candidate.json comparison.json
cargo run -p motionloom --example rig_calibrate -- comparison.json proposal.json
```

`rig_compare` returns a non-zero exit status for warning/error differences, so
it can be used as a CI regression gate.

## WASM/JS

Use `WasmSceneRenderer.evaluate_rig_json(requestJson)` for production Scene
evaluation. `motionloom_compare_rigs_json` and
`motionloom_propose_rig_calibration_json` operate only on JSON and do not need a
GPU. The low-level `WasmPoseDiagnostics` API remains available for isolated
World evaluator tests, but it cannot certify Scene contact or camera stages.

## Deliberate exclusions

- No new MotionLoom DSL syntax.
- No automatic writes to ModelProfile, BoneAxisMap, Action, or showcase files.
- No Vision LLM call inside the rendering crate.
- No Action Editor UI in this implementation phase.
