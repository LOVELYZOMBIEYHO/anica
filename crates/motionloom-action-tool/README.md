# MotionLoom Action Tool

Offline FBX / GLB / glTF animation authoring tool. Its only deliverable is a
standalone `.motionloom` `<Action>` fragment; no source animation file is needed
at playback time. A character mesh and its target `ModelProfile` are still needed
to display that Action.

## Boundary

```text
FBX ── offline ufbx evaluator ─┐
FBX ── legacy native reader ───┤
FBX ── explicit Blender bake ──┼── AnimationSource ── canonical sampling ── Action DSL
GLB / glTF ── existing loader ─┘
```

All FBX code, ufbx, the `draco-io/fbx-reader` feature, and Blender process management
are in **this crate only**. The dependency direction is
`motionloom-action-tool -> motionloom`, never the reverse. Neither the renderer
nor its WASM build gains an FBX asset type, parser, tag, or dependency feature.
The existing DSL version and tags are unchanged.

## Inspect

Run from the `anica` repository root:

```sh
cargo run -p motionloom-action-tool -- inspect \
  "../temp-save-3d-action/Stand To Roll.fbx"
```

The report lists the chosen backend, available clip names/durations, canonical
bone mappings, and non-canonical nodes. Animation-only FBX files without meshes
or skins are supported. Inspect does not write an Action.

## Convert (source profile only)

This legacy path does not calibrate to Character1. For target-bound candidates,
use the command below instead. Source-only conversion is not fidelity-certified;
target mode can produce the separate source/native/WASM evidence described in
[Target fidelity status](TARGET_FIDELITY.md).

```sh
cargo run -p motionloom-action-tool -- convert \
  "../temp-save-3d-action/Stand To Roll.fbx" \
  --source-profile fbx_humanoid \
  --action-id stand_to_roll \
  --fps 30 \
  --fbx-backend native \
  --output "../temp-save-3d-action/stand_to_roll.motionloom"
```

Omit `--clip` to use the first clip. Input paths containing spaces must be quoted.
Existing output files are not overwritten unless `--force` is supplied.
Every generated fragment is wrapped and validated by the existing MotionLoom
parser before the file is opened for writing.

Optional authoring controls:

| Option | Default | Meaning |
| --- | --- | --- |
| `--fps` | `30` | Fixed-rate output sampling, including the final time. |
| `--key-reduction-tolerance` | `0` | Zero preserves every sampled Pose. A positive number bounds sampled-channel error in degrees and translation error in millimetres. |
| `--detect-contacts` | off | Add candidate `Contact` intervals for near-ground, slow-moving feet. Review these before enabling runtime foot locking. |
| `--fbx-backend` | `auto` | For target conversion, `auto` uses offline ufbx evaluation. Source-only compatibility commands and explicit `native` use the legacy reader; `blender` explicitly launches Blender. |
| `--force` | off | Replace an existing output file. |

The fixed-rate output uses existing `Pose`, `Bone`, and optional `Contact` tags.
Source curve handles are baked into sampled poses, not exported as new DSL tags.
Reduction bounds apply at sampled times, not to a continuous or anatomical error
metric. Contact detection assumes a flat floor and cannot infer stairs, props,
intent, or reliable locks from every acrobatic clip.

## Convert for Character1 (experimental candidate)

Run from `anica`. Keep this output separate from the old Action and the published
library until the acceptance checks in [TARGET_FIDELITY.md](TARGET_FIDELITY.md) pass.

```sh
cargo run -p motionloom-action-tool -- convert \
  "../temp-save-3d-action/Stand To Roll.fbx" \
  --source-profile fbx_humanoid --fbx-backend auto \
  --target-model ../motionloom-example/assets/sample_assets/characters/character1/character1.glb \
  --target-profile ../anica-landing-page/public/motionloom-actions/character1-scene.motionloom \
  --target-profile-id character1_profile --target-height 1.82 \
  --action-id stand_to_roll --fps 30 \
  --report ../temp-save-3d-action/stand_to_roll_target.report.json \
  --validation-bundle ../temp-save-3d-action/stand_to_roll.validation.json \
  --output ../temp-save-3d-action/stand_to_roll_target.motionloom
```

Use `--force` only to replace an existing candidate/report. The tool protects the
input, model and profile from output-path collisions even with `--force`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--target-model` | absent | Target GLB mesh and rest hierarchy. |
| `--target-profile` | absent | Existing DSL file containing `ModelProfile` declarations. |
| `--target-profile-id` | absent | Exact profile used by the playback scene. All three target options are required together. |
| `--target-height` | `1.82` | Height in metres of the normalized playback model. Must match playback. |
| `--motion-scale` | `proportional` | Scale root travel by target/source rest hip-to-foot height; `preserve` retains source metres at the specified target height. |
| `--max-position-mm` | `1` | Tool-side reconstruction tolerance at evaluated times. |
| `--max-rotation-deg` | `0.1` | Tool-side rotation reconstruction tolerance at evaluated times. |
| `--report` | absent | JSON hashes, metrics, checks and explicit limitations. |
| `--strict-fidelity` | off | Conversion-time fail-closed gate. Full certification additionally requires separate hash-matched WASM evidence and `certify`. |
| `--time-grid` | `subframes` | Fidelity-first float-second times. `milliseconds` emits Action Editor-safe unique times but can exceed strict angular tolerances. |
| `--validation-bundle` | absent | Hash-bound native snapshots consumed by the separate WASM parity runner. |

The JSON report also contains a 120 Hz `joint_trajectory_audit` for feet, toes,
hands, head and hips. It reports model-global joint-centre ranges and low/slow
samples before scene contacts. Those values help locate contact phases, but are
not a collision, skin-surface, foot-lock or no-penetration certificate.

Target mode uses 30 fps (or the requested rate) as a starting grid and inserts
subframe poses when necessary. It rejects legacy channel-based reduction and
automatic contact detection rather than applying them without target validation.
The existing parser validates the exported DSL. The result is bound to the
recorded target/profile, not a universally calibrated humanoid Action.

The complete evidence chain is deliberately two-stage. Build a Node WASM package,
run `scripts/verify-wasm.cjs`, then combine the matching reports without replacing
either input:

```sh
wasm-pack build crates/motionloom --dev --target nodejs \
  --out-dir /tmp/motionloom-pose-diagnostics-wasm -- --offline
node crates/motionloom-action-tool/scripts/verify-wasm.cjs \
  /tmp/motionloom-pose-diagnostics-wasm/motionloom.js \
  ../motionloom-example/assets/sample_assets/characters/character1/character1.glb \
  ../temp-save-3d-action/stand_to_roll.validation.json \
  ../temp-save-3d-action/stand_to_roll.wasm-parity.json
cargo run -p motionloom-action-tool -- certify \
  ../temp-save-3d-action/stand_to_roll_target.report.json \
  ../temp-save-3d-action/stand_to_roll.wasm-parity.json \
  ../temp-save-3d-action/stand_to_roll.certified.json
```

## FBX support

The default evaluator uses ufbx at 120 Hz so FBX cubic curves, Euler rotation
order, pivots and animation layers are evaluated rather than approximated. The
legacy `native` backend remains available for audits: `audit-source` showed that
the supplied file matched at authored keys but deviated between keys, which is
why it is no longer the production default.

- ASCII and binary FBX, including compressed arrays.
- Clip enumeration, source hierarchy, animation-only files, translation units
  converted to metres, and source-world axis metadata.
- Bone-local axes are preserved for the source profile. World-axis conversion
  must not invert the meaning of a local knee flexion.
- Euler rotation orders, pre/post rotation, linear/step curves and unweighted
  cubic slopes. Curves are internally baked at up to 1/120-second intervals
  before quaternion conversion, preserving the intermediate samples of full turns.
- `fbx_humanoid` body, shoulders, hands, toes, and 30 finger mappings.
  Extra intermediate spine nodes are folded into their mapped descendant;
  terminal helper nodes have no independent canonical channel.
- Rest-relative sampling, quaternion interpolation, angular unwrapping, optional
  error-bounded reduction and contact candidates.
- Hips pitch/roll and residual joint axes use existing `rotationX/Y/Z` channels
  where the current semantic profile does not cover them. They require target
  axis calibration; this is not a promise of universal retargeting to every rig.

Native mode rejects unsupported pivot/offset stacks, reflected/non-uniform
scales, special inheritance, unbaked constraints, multiple animation layers,
weighted/mixed-interpolation curves, and unaligned/sparse XYZ source keys instead of
silently dropping those semantics. Use an explicitly selected offline evaluator
or bake those constructs before importing.

The older glTF loader exposes CUBICSPLINE key values but not source tangents.
Such glTF inputs produce a diagnostic; pre-bake their curves when sub-key fidelity
matters. This task deliberately does not modify that loader in `motionloom`.

## Optional Blender evaluator

Blender is **not required by the default command** and is never launched
automatically after an import error. To opt in explicitly:

```sh
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender" \
cargo run -p motionloom-action-tool -- convert input.fbx \
  --fbx-backend blender --action-id imported --output imported.motionloom
```

The tool starts an isolated factory-default background process with auto-execution
disabled, imports the FBX, and bakes to a temporary GLB. It does not modify an open
Blender document. Temporary data is removed on return; only the requested Action
is retained. A crash is returned as an error, not treated as successful conversion.

The installed Blender crashed during local testing, so this integration is
optional and not a prerequisite for the native acceptance tests. Do not rely on
it on a machine where the standalone background Blender invocation is unstable.

## Tests and local acceptance

```sh
cargo test -p motionloom-action-tool
cargo check -p motionloom --target wasm32-unknown-unknown
cargo tree -p motionloom --target wasm32-unknown-unknown -e features -i draco-io
```

The last command may show the pre-existing glTF/Draco features, but must not show
`fbx-reader` or `compression`. The tool's own dependency tree does include them.

Tests use project-authored ASCII and generated binary fixtures (no redistributed
third-party motion). They cover FBX/GLB parity, no-mesh input, units/axis separation,
knee flexion, full turns, cubic evaluation, pre/post rotations, reduction,
contact output, invalid inputs, and DSL round trips. The locally supplied
`Stand To Roll.fbx` converts natively to 72 poses at 30 fps (2.366667 seconds),
with 51 mapped bones. It is not checked into this crate as a test asset.

### Legacy source-only Character1 visual audit (2026-08-27)

The generated Action was rendered with the existing native GPU frame renderer
and the Action Editor's Character1 profile at frames 0, 18, 35, 53 and 71.
Rendering succeeded without Blender. This is a **decode/render smoke test, not
a visual-fidelity acceptance pass**: limb poses and floor contact still differ
from a convincing standing-to-roll motion. Frames 18 and 71 were also rendered
with a larger floor and without actor grounding; visible suspension remained,
so floor extent and that grounding setting alone do not explain the result.

The current source profile maps rest-relative Euler components to semantic
channels. It does not solve source-to-target joint-frame alignment, target
proportions or pelvis/ground placement. A bone-name match is not sufficient for
that operation, especially when raw residual axes coexist with semantic axes.
Target-aware offline retargeting and source/target pose comparisons remain
necessary before calling this particular converted Action production-ready.
Do not compensate by modifying the original FBX, adding arbitrary per-frame
offsets, or changing the MotionLoom runtime as part of this importer task.

Parser success is not a claim that the result is production-quality on every
character. Preview the generated Action on the intended target, review its
rest/axis calibration and contacts, then refine it in Action Editor. Source
asset permissions also still need checking before publishing a converted library.

## Source layout

- `src/source.rs`: format-neutral data and structural validation.
- `src/fbx_source.rs`: native FBX decoding, compatibility guards and curve baking.
- `src/ufbx_source.rs`: production FBX evaluation for target conversion, isolated to this tool.
- `src/target_reference.rs`: independent source-evaluator comparison.
- `src/gltf_source.rs`: adapter for the existing glTF loader.
- `src/blender_source.rs`: explicit offline subprocess integration.
- `src/lib.rs`: mapping, sampling, reduction, contacts, DSL output and validation.
- `src/target.rs`: target rest alignment, pelvis-space conversion, adaptive sampling,
  read-only joint trajectory audits and candidate fidelity reports.
- `src/main.rs`: inspect/convert command-line interface.
- `tests/`: redistributable synthetic regression inputs.

`draco-io` is Apache-2.0; it is an authoring-tool dependency, not an FBX SDK added
to MotionLoom. FBX conversion itself does not require a Web UI, showcase or
runtime FBX change. The optional read-only pose diagnostic API is built into the
native/WASM MotionLoom packages solely for parity evidence.
