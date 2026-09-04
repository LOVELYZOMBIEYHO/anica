# MotionLoom Main Public API

This document lists the recommended public API surface for applications and
open-source users integrating MotionLoom as a standalone Rust crate.

MotionLoom exposes more model structs and compatibility re-exports at the crate
root than this list. Those lower-level types are useful for advanced tooling and
for existing applications such as Anica, but new code should start with
`motionloom::api`.

The crate root is intentionally broader for compatibility. `motionloom::api` is
the curated stable surface. `motionloom::experimental` is public and usable, but
may change faster than the main API.

## API Layers

MotionLoom has three useful integration layers:

1. **Stable APIs** are re-exported from `motionloom::api`.
2. **Easy root-document APIs** accept a full MotionLoom script string and route
   it internally.
3. **Typed scene/process APIs** accept already parsed graphs and are better for
   applications that manage parse/cache/export state themselves.

Use root-document APIs for CLI tools and simple standalone integrations. Use
typed APIs when the host already knows whether the document is a scene graph,
process graph, or app-layer effect.

## Core Parsing

### 2D Skeleton authoring

`builtin_proportion_profiles`, `validate_skeleton`, `auto_correct_skeleton`, and
`build_skeleton_overlay` form the stable host API for profile-driven 2D rigs.
The overlay result is renderer-independent and is intended for editor gizmos;
it is not burned into exported frames.

### `parse_graph_script`

```rust
parse_graph_script(script) -> Result<GraphScript, GraphParseError>
```

Parses the main MotionLoom graph format, including scene graphs and
scene/process composition graphs.

Use this when the caller expects a scene/composition graph and wants typed
control over rendering.

`GraphAssetSource`, `MaterialAssetNode`, `PrimitiveAssetNode`, `PrimitiveGeometry`,
`PrimitiveModifierNode`, `PrimitiveMeshBuildNode`, `PrimitiveLodNode`,
`PrimitiveCollisionNode`, `TerrainAssetNode`, `VegetationAssetNode`,
`VegetationKind`, `VegetationLod`, and `CompoundAssetNode` expose the typed asset
representation used by generated geometry. External GLB, individual
primitives, and compound primitive assets remain distinct through parsing and
asset resolution. A resolved PrimitiveAsset retains its referenced PBR
MaterialAsset so native and WASM world renderers consume the same self-contained
material definition after CompoundAsset expansion.
The advanced PrimitiveAsset block is additive: compact self-closing assets
deserialize with empty modifiers and default build/LOD policies. Native and
WASM renderers consume the same generated triangle mesh and stable cache key.
`TerrainAssetNode` retains its resolved height map, optional RGBA blend map,
and up to four resolved PBR layer definitions. Terrain is an additive
`GraphAssetSource::Terrain` variant and therefore does not change existing
PrimitiveAsset, external ModelAsset, or CompoundAsset behavior.
`VegetationAssetNode` retains its resolved kind-specific MaterialAssets and
bounded generation, LOD, wind, and collision settings. Vegetation is an
additive `GraphAssetSource::Vegetation` variant and does not change existing
DSL assets. As with any new public Rust enum variant, downstream exhaustive
matches over `GraphAssetSource` must add the Vegetation case or a wildcard.
`MaterialAssetNode` also carries transmissive PBR controls (`transmission`,
`ior`, optical `thickness`, attenuation, depth-write policy, and sort priority)
without coupling the visual material to PrimitiveAsset collision.

### `parse_process_graph_script`

```rust
parse_process_graph_script(script) -> Result<ProcessGraph, GraphParseError>
```

Parses process-only graphs used for effects, Layer FX, and effect runtime
evaluation.

Process-only graphs that reference external inputs such as
`from="input:clip0"` need a host application to provide the source clip.

### `parse_motionloom_document`

```rust
parse_motionloom_document(script) -> Result<MotionLoomDocument, GraphParseError>
```

Parses a root MotionLoom document and classifies it as scene, process, world, or
mixed graph.

Use this when building tools that accept arbitrary MotionLoom DSL input.

## Single Frame Rendering

### `render_scene_graph_frame`

```rust
render_scene_graph_frame(&graph, frame, SceneRenderProfile::Gpu)
```

Renders one scene/composition frame to an `image::RgbaImage`.

This is the simple one-shot API. It creates renderer state internally, so it is
best for single frame exports, tests, or simple examples.

For multiple frames, prefer `SceneRenderer`.

### `SceneRenderer::new`

```rust
let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await?;
```

Creates a reusable scene renderer. Prefer this for preview, playback, PNG
sequence generation, or any integration that renders many frames.

### `SceneRenderer::render_frame`

```rust
renderer.render_frame(&graph, frame).await?
```

Renders one frame using the reusable renderer. This avoids rebuilding internal
state for every frame.

## Export APIs

### Root Document PNG Sequence

```rust
render_motionloom_document_to_png_sequence_with_progress(
    script,
    asset_root,
    output_dir,
    progress_every_frames,
    callback,
)
```

Exports a full MotionLoom script to a PNG frame sequence.

This is the easiest PNG sequence API. It inspects the root document and routes
to the correct renderer internally. It does not require FFmpeg.

### Root Document Video

```rust
render_motionloom_document_to_video_with_progress(
    ffmpeg_bin,
    script,
    asset_root,
    output_path,
    profile,
    progress_every_frames,
    callback,
)
```

Exports a full MotionLoom script to video using FFmpeg.

This API is best for CLI tools or applications that accept arbitrary MotionLoom
documents. The caller supplies the FFmpeg binary path; MotionLoom does not bundle
FFmpeg.

### Typed Scene PNG Sequence

```rust
render_scene_graph_to_png_sequence_with_progress(
    &graph,
    output_dir,
    progress_every_frames,
    callback,
)
```

Exports an already parsed scene/composition graph to PNG frames.

Use this when the caller has already parsed a `GraphScript` with
`parse_graph_script`. It avoids the extra root-document inspect/parse step.

### Typed Scene Video

```rust
render_scene_graph_to_video_with_progress(
    ffmpeg_bin,
    &graph,
    output_path,
    profile,
    progress_every_frames,
    callback,
)
```

Exports an already parsed scene/composition graph to video.

Use this when the caller already knows the graph is scene/composition content.

## Scene 3D Lighting Contract

The public Scene DSL owns 3D lighting; hosts do not need to construct or expose
legacy World implementation types. A `CompositeGroup space="3d"` accepts:

- `EnvironmentLight` for equirectangular HDR/EXR/LDR backgrounds and IBL
- `DirectionalLight`, `PointLight`, `SpotLight`, and `RectAreaLight`
- `AmbientOcclusion` and `ContactShadow`
- `ColorManagement` with `aces`, `reinhard`, or `none` tone mapping
- `AtmosphereFog` with `linear`, `exp`, or height-aware distance attenuation;
  optional `boundsMin`/`boundsMax` and `edgeFeather` confine the medium to a
  world-space box without changing unbounded scenes
- optional `Camera3D` depth-of-field optics (`focusTarget`, `focusDistance`,
  `focalLength`, `fStop`, and `maxBlur`)

The renderer caches decoded environments across frames, uploads linear
RGBA16F mip levels, and evaluates animated light values per frame. The same
contract is used by `SceneRenderer`, GPU-texture rendering, native export, and
WASM WebGPU rendering. Existing scenes with no authored light retain the
legacy studio-light fallback. Existing cameras without `depthOfField="true"`
skip the depth-aware post pass and retain their previous output path.

## Process / Layer FX APIs

### `compile_runtime_program`

```rust
let runtime = compile_runtime_program(graph)?;
```

Compiles a process graph into a runtime program for evaluating effect parameters
over time.

This is the key API for Layer FX-style integrations.

### `RuntimeProgram::evaluate_frame`

```rust
runtime.evaluate_frame(frame)
```

Evaluates process effect parameters for one frame.

### `RuntimeProgram::evaluate_at_time_sec`

```rust
runtime.evaluate_at_time_sec(time_norm, time_sec)
```

Evaluates process effect parameters at an explicit timeline time.

### `RuntimeProgram::unsupported_kernels`

```rust
runtime.unsupported_kernels()
```

Returns kernels that the runtime could not execute natively.

Hosts should report or skip unsupported effects instead of silently pretending
they ran.

## Process Catalog APIs

### `process_effects`

```rust
process_effects()
```

Returns the built-in process effect catalog.

Use this as the source of truth for UI pickers, LLM tooling, and effect
discovery.

### `process_effect_for_id`

```rust
process_effect_for_id("tone_map")
```

Looks up one effect definition.

### `process_effects_for_category`

```rust
process_effects_for_category(category)
```

Lists effects by category.

### `kernel_source_by_name`

```rust
kernel_source_by_name("tone_map.wgsl")
```

Returns embedded WGSL source for a known kernel.

### `is_known_process_kernel`

```rust
is_known_process_kernel("tone_map.wgsl")
```

Checks whether a process kernel is bundled with MotionLoom.

## Preview and GPU Integration

### `SceneRenderer::render_frame_to_wgpu_texture`

```rust
renderer.render_frame_to_wgpu_texture(&graph, frame).await?
```

Renders a frame to a MotionLoom-owned `wgpu::Texture`.

Use this when the host wants GPU output without managing the target texture
itself. The preferred output type name is `GpuFrameTexture`;
`SceneGpuTexture` remains available for source compatibility. Mixed Scene
frames containing `<CompositeGroup space="3d">` stay on the shared wgpu device
and are sampled by the Scene compositor without a CPU readback/re-upload.

### `SceneRenderer::render_frame_to_wgpu_target_texture`

```rust
renderer
    .render_frame_to_wgpu_target_texture(&graph, frame, target, width, height)
    .await?
```

Renders a frame into a caller-owned `wgpu::Texture`.

This is the preferred path for high-performance host integration because the
host controls texture allocation and reuse. The target texture must belong to
the same `wgpu::Device` used by the renderer.

### `SceneRenderer::render_frame_to_preview_surface`

```rust
renderer
    .render_frame_to_preview_surface(&graph, frame, options)
    .await?
```

Renders a frame to the best preview surface requested by the host.

This is the higher-level preview abstraction. It can return a GPU texture,
platform surface, or CPU BGRA fallback depending on platform support and
options.

### `WgpuPreviewEngine`

```rust
let mut engine = WgpuPreviewEngine::new_with_cpu_fallback().await;
let preview = engine
    .render_preview_surface_with_cpu_fallback(&graph, frame, options)
    .await?;

let mut graph_cache = WgpuPreviewGraphCache::default();
let preview = engine
    .render_script_preview_surface_with_cpu_fallback(
        &mut graph_cache,
        script,
        script_hash,
        frame,
        Some((640, 360)),
        options,
    )
    .await?;
```

Reusable live-preview lifecycle for host applications and examples. It keeps GPU
and CPU `SceneRenderer` instances alive, shares quality scaling through
`WgpuPreviewQuality`, caches parsed script graphs through
`WgpuPreviewGraphCache`, allocates reusable wgpu target textures, and exposes
both high-level preview-surface rendering and caller-owned target-texture
rendering. Window creation, event loops, keyboard controls, and final
presentation remain the responsibility of the host.

Interactive hosts can preload incrementally instead of blocking the event loop
for every representative frame:

```rust
let mut session = WgpuPreviewEngine::begin_preload_graph_resources(&graph);
while !session.is_finished() {
    let progress = engine
        .preload_graph_resources_step(&graph, &mut session)
        .await?;
    present_loading_progress(progress.completed_frames, progress.total_frames);
}
let report = session.report();
```

The existing `preload_graph_resources` convenience method remains available
and runs the same bounded session to completion. `Scene3DFrameProfile` exposes
texture decode time/count, decoded bytes, cache hits, retained GPU texture
resources, asset resolution, geometry preparation, and submission timings.
Hosts should use these counters to distinguish Cargo compilation, cold asset
preparation, and steady-state frame cost.

### Preview host protocol

```rust
use motionloom::{PreviewCommand, PreviewEvent};
```

`PreviewCommand` and `PreviewEvent` are serde-compatible controller/viewer
messages for external preview hosts and future embedded viewers. Current
commands cover `LoadScript`, `SetFrame`, `SetQuality`, `SetOverride`,
`ClearOverride`, `SetAssetRoots`, `SetWindowBounds`, `SetWindowVisible`,
`SetInteractionTarget`, and `SetInteractionTargets`;
events cover `Ready`, `Rendered`, `WindowBounds`, `Error`, `PickResult`, and
`TransformDrag`. `TransformDragEnd` marks release/commit boundaries for editor
controllers that write keyframes after a native preview drag.
`SetWindowBounds` lets an editor controller align a native external preview host
over its own preview panel, while `SetWindowVisible` lets it hide the companion
viewer when the editor loses focus. `SetInteractionTargets` lets the controller
provide editable node bounds, graph size, and current transform values so an
external viewer can hit-test and emit drag edits from its own GPU surface. The
single-node `SetInteractionTarget` command is retained for simpler controllers.
The protocol is transport-neutral: the
standalone preview example uses newline-delimited JSON over local TCP, while an
embedded host can reuse the same types with in-process channels.

## Diagnostics

### `inspect_root_graph`

```rust
inspect_root_graph(script)
```

Inspects a root document and reports whether it contains scene, process, world,
or mixed content.

### `inspect_gpu_compatibility`

```rust
inspect_gpu_compatibility(script)
```

Performs a static GPU compatibility inspection. This is useful before choosing a
preview or export path.

## Experimental / Advanced APIs

The following APIs remain public under `motionloom::experimental`, but should
not be treated as the main MotionLoom integration path yet:

- `parse_world_graph_script`
- `render_world_frame`
- `WorldFrameRenderer`
- `render_world_graph_to_video_with_progress`
- `render_world_graph_to_png_sequence_with_progress`
- GLB metadata and diagnostics helpers
- Animation-only GLB inspection data and byte/path loaders used by external
  authoring tools
- Anica/editor-oriented helpers such as `experimental::effects`,
  `experimental::keyframe`, `experimental::transitions`, and
  `experimental::clip`
- Text layout preparation helpers under `experimental::text`

Scene/model AST structs such as `RectNode`, `TexNode`, and `PassNode` remain
visible through the crate root for compatibility and advanced tooling. They are
not the recommended starting point for new integrations.

Editor-oriented scene DSL structs are also re-exported at the crate root for
tooling that needs to inspect or generate UI-editable scene graphs:

- `AnimationTargetNode` and `AnimationKeyNode` for UI-editable keyframes with
  `time` or `frame` timing.
- `SkeletonNode`, `SkeletonBoneNode`, `ActionNode`, `ActionPoseNode`,
  `ActionBoneNode`, `ApplyActionNode`, and IK data inside actions for rigs.
- Unified Scene 3D models accept optional `profile`, `rig`, `retarget`, and
  `<Play>` data. A `kind="3d"` ModelProfile plus Action/ApplyAction lowers to
  the existing GPU skinning backend. `AnimationTarget` also accepts typed
  `bones.<canonical-bone>.<component>` paths, and 3D Actions can carry
  analytic two-bone IK, blending, speed, loop, and body-mask metadata.
- Scene DSL exposes external motion through a low-level `AnimationAsset`
  wrapped by an executable `Action`; `ApplyAction` accepts only Action ids.
  It also exposes relative 3D `Anchor` placement, timed `ApplyAction`
  destination/facing controls,
  cross-model two-bone `Constraint`, and discrete Scene `activeCamera`
  switching. These features lower into the existing Scene 3D renderer and do
  not restore the removed public `<World>` authoring root.
- `Camera3D.hiddenBones` accepts typed `model_id:canonical_bone` selectors for
  first-person coverage. Selecting the canonical `hips` root hides the whole
  actor color pass for that camera while preserving its shadow pass.
  Visibility is local to the selected camera's beauty
  and CPU view passes; the actor remains complete for animation, collision,
  other cameras, and shadow casting. A camera `Anchor node="head"` samples the
  final animated, collision-corrected humanoid pose, allowing one actor to
  remain continuous across first-person, third-person, and close-up cuts.
- Static GLB stages use the additive `Environment` specialization of `Model`,
  with a declared coordinate profile and nested semantic `Surface` and `Anchor`
  nodes. Surfaces can carry asset-space centroid/bounds/normal evidence; anchors
  can use normalized surface UV coordinates. `Camera3D` position/target, Model
  position, `ApplyAction.destination`, `face`, `takeoff`, `contact`, `landing`,
  and `ground` can share the resulting ids. Runtime grounding raycasts the
  transformed walkable triangles. External Actions may contain semantic
  `Marker` nodes, allowing root motion to follow authored action phases instead
  of fixed percentages. `Camera3D.up`, `roll`, and `horizonLock` make imported
  environments camera-safe. `ApplyAction.groundOffset` separates a target rig's
  root origin from the physical surface while preserving all older action
  syntax.
- `ApplyAction.ground` also accepts the Model id of a `TerrainAsset` whose
  collision mode is `solid`. This additive provider raycasts the generated
  terrain collision triangles; existing semantic Surface ids retain priority
  and unchanged behavior. Position-animated humanoids over solid terrain
  should declare `Model collision="kinematic"`.
- `ContactSurfaceNode` and `ApplyAction.contactTargets` add semantic support
  planes without changing `ApplyAction.ground`. The first runtime profile is
  `kind="seat"`: an Action `Contact` may use `effector="pelvis"`,
  `target="seat"`, and `mode="surface"`. Primitive top planes follow Model
  translation, rotation, and scale; random-access native and WASM frames use
  the same stateless resolver. Missing fields remain optional and older Graphs
  deserialize with an empty contact-surface registry.
- `CharacterNode` and `PartNode` for bone-attached or dense vector artwork.
- `PuppetNode`, `PinNode`, `MeshTopologyNode`, `VertexNode`, `TriangleNode`,
  `EdgeNode`, and `RegionNode` for AE-style pin deformation and optional manual
  topology.
- `FaceJawNode`, `MaskNode`, `CameraNode`, and `SceneLayerNode` for higher-level
  scene helpers.

These model structs are useful for editor/LLM tooling, but they follow the DSL
runtime and may evolve faster than `motionloom::api`. If a host only needs to
render or export scripts, prefer parsing/rendering through `motionloom::api`
instead of constructing AST structs directly.

For editor property panels, use `docs/acp/motionloom/scene-ui-schema.json` as
the machine-readable metadata source for Scene Camera and Layer3D property
labels, groups, value types, and animatability. Do not infer UI schema directly
from the AST structs.

Frame-key UI integrations can use:

- `extract_editable_animation_timeline(script)` to parse `.motionloom` text into
  `EditableAnimationTimeline`.
- `upsert_editable_animation_target(script, target)` to update one
  node/property channel.
- `replace_editable_animation_targets(script, targets)` to replace the full
  editor keyframe set.
- `editable_animation_target(script, node, property)` and
  `remove_editable_animation_target(...)` for channel-local editing that keeps
  unrelated source formatting.

These helpers re-parse generated DSL after write-back, so UI saves fail fast
instead of emitting invalid MotionLoom text.

Action-authoring integrations can use the experimental editor surface:

- `extract_editable_action_document(script)` returns editable Action, Pose,
  Bone channel, Contact, ApplyAction binding, Skeleton, and Model target data.
- `apply_action_edit(script, command)` applies one typed `ActionEditCommand`
  such as `SetBoneChannel`, `AddPose`, `MirrorPose`, `UpsertContact`, or
  `SetBinding`.
- `motionloom_editable_actions_json(script)` and
  `motionloom_apply_action_edit(script, command_json)` expose the same contract
  to WASM browser hosts.
- `inspect_humanoid_action_compatibility(action, profile)` reports missing
  required body mappings and optional mappings actually referenced by the
  Action. It is read-only and does not change retargeting or playback.

External clip-backed Actions are inspectable and retain editable Contact and
ApplyAction metadata, but authored Pose commands reject them. This prevents a
visual editor from pretending it can rewrite animation data stored inside an
external GLB.

`ActionLibraryNode` is an additive Graph declaration for selectively importing
authored Actions from a standalone MotionLoom file. Imported ids use
`library.action` namespaces and the renderer caches the hydrated graph. Native
resolvers support project-relative paths and URLs; WASM hosts register the
library bytes through the existing asset resolver before rendering. No
`ActionAsset` type is introduced and existing Action APIs are unchanged.

The workspace `motionloom-action-tool` binary is the supported offline bridge
from an animated glTF/GLB clip to a standalone canonical `<Action>`. Its
dependency points towards `motionloom`; neither the stable renderer API nor
the WASM runtime depends on the converter. Imported source animation therefore
never becomes a runtime or deployment requirement.

Animation capability discovery is available through
`animation_property_descriptor`, `animation_properties_for_node_kind`,
`animation_property_schema_json`, and `inspect_animation_targets`. They share
the typed registry used by parser validation and rendering; hosts should not
maintain a separate property whitelist.

For an LLM authoring loop, use `motionloom_analyze_script_json` or
`motionloom_analyze_script_for_target_json`. These APIs always return a
machine-readable `MotionLoomAuthoringReport`, including parse and compile
status, source-addressed errors and warnings, effective graph facts, and
recommended repairs. Invalid authored DSL is represented by
`status: "unrenderable"` rather than a thrown WASM exception.

## Shot Validation

Shot quality validation is an additive runtime API; it does not add tags or
attributes to the MotionLoom DSL. Use a persistent renderer for render-derived
checks:

```rust
let options = motionloom::api::ShotValidationOptions::cinematic();
let report = renderer.validate_shots(&graph, options).await?;
```

`SceneRenderer::validate_shots` samples an inclusive frame range and currently
computes linear-light average luminance, dark/highlight ratios, clipped-pixel
ratio, and a 64-bin luminance histogram from actual rendered RGBA frames. On
WASM, the renderer also consumes the exact Action Editor joint projection for
framing checks when a 3D humanoid snapshot is available.

Backends and hosts feed additional typed `ShotValidationFrameObservation`
values to `validate_shots_with_observations`. Supported observations cover
projected joints, ID/depth visibility samples, penetration depth, camera
clearance/path sweeps, and external composition scores. `observedChecks`
distinguishes an evaluated empty result from missing data. Enabled checks with
no observations are reported as `unavailable`, never as a fabricated pass.

Browser hosts use `WasmSceneRenderer.validate_shots_json(options, observations)`
or the render-independent `motionloom_analyze_shot_observations_json`. Native
CI can run:

```text
cargo run -p motionloom --example shot_validation -- scene.motionloom strict gpu
```

The report is versioned JSON suitable for Action Editor, Anica, CI, or a later
Vision LLM review stage. A Vision LLM remains an orchestration concern above
MotionLoom and is not invoked by this API.

Use `motionloom_dsl_schema_json()` to retrieve the complete registered DSL
catalog before generation. Its `requiredAttributes` field makes mandatory
contracts such as `RigidBody.id`, `target`, `dimension`, and `type` explicit.
Use `motionloom_showcase_schema_json(script)` only for the smaller syntax slice
demonstrated by one example.

`motionloom_showcase_schema_json` serves a separate purpose: it extracts the
language slice demonstrated by one example for dataset learning and generates
the `schema.json` stored beside that showcase's `main.motionloom`.

Environment asset inspection is available through
`inspect_glb_environment_path`, `inspect_glb_environment_bytes`, and
`inspect_glb_environment_json`. Reports contain renderer-space bounds,
coordinate-profile evidence, transformed walkable triangle Surface proposals,
Anchor proposals, confidence, diagnostics, and a starter DSL fragment. The WASM wrapper
`motionloom_inspect_glb_environment_json(asset_label, bytes)` operates on bytes
that a browser host has already fetched; inspection never requires a public
`World` authoring API.

Humanoid importers may opt into `inspect_glb_humanoid_profile_bytes` or
`inspect_glb_humanoid_profile_json`. These additive APIs prioritize VRM 1.0 or
0.x humanoid metadata, then validate known Mixamo-compatible names and joint
hierarchy, and finally reuse the existing geometry/name heuristic. Their WASM
wrapper is `motionloom_inspect_glb_humanoid_profile_json(asset_label, bytes)`.
The existing `inspect_glb_skeleton_*` report shape and `humanoid_v1` DSL remain
unchanged for compatibility.

Low-level kernel resolution helpers such as `default_kernel_for_effect` and
`resolve_pass_kernel` are also kept for compatibility. Prefer the process
catalog APIs for effect discovery.

Rust types and functions whose names contain `World` are legacy compatibility
APIs for Anica internal tools and design/debug surfaces. They are not a current
DSL authoring surface: `<World>` is invalid. New integrations should parse a
unified `<Scene>` and place true-3D content in a `space="3d"` track.

`Model.scaleMode` defaults to `none`: rendering, physics and picking preserve
the same glTF origin and authored units. `normalize_height` is explicit.
Dynamic RigidBody transforms are physics-owned; a transform AnimationTarget on
the same Model is an authoring error. The deterministic backend uses quaternion
orientation, resolved Collider3D data, multi-point manifolds, swept CCD,
rotational impulses, rolling friction and fixed-step island sleeping.

## Recommended API Choice

| Use case | Recommended API |
| --- | --- |
| Parse scene/composition DSL | `parse_graph_script` |
| Render one PNG/image frame | `render_scene_graph_frame` |
| Render many frames interactively | `SceneRenderer::new` + `SceneRenderer::render_frame` |
| Export arbitrary script to PNG sequence | `render_motionloom_document_to_png_sequence_with_progress` |
| Export arbitrary script to video | `render_motionloom_document_to_video_with_progress` |
| Export parsed scene graph to PNG sequence | `render_scene_graph_to_png_sequence_with_progress` |
| Export parsed scene graph to video | `render_scene_graph_to_video_with_progress` |
| Build Layer FX runtime | `parse_process_graph_script` + `compile_runtime_program` |
| Discover available process effects | `process_effects` |
| Analyze LLM-authored DSL and suggest repairs | `motionloom_analyze_script_json` |
| Analyze for a specific renderer | `motionloom_analyze_script_for_target_json` |
| Validate rendered shot quality | `SceneRenderer::validate_shots` |
| Merge renderer/editor observations | `SceneRenderer::validate_shots_with_observations` |
| Evaluate the final humanoid rig at one Scene frame | `SceneRenderer::evaluate_rig_frame` |
| Compare two versioned humanoid reports | `compare_humanoid_poses` |
| Generate read-only calibration suggestions | `propose_rig_calibration` |
| Generate one example's learning schema | `motionloom_showcase_schema_json` |
| GPU preview texture | `SceneRenderer::render_frame_to_wgpu_texture` |
| Host-owned zero-copy target | `SceneRenderer::render_frame_to_wgpu_target_texture` |
| Cross-platform preview abstraction | `SceneRenderer::render_frame_to_preview_surface` |

## Stable Rig Evaluation and Comparison

`SceneRenderer::evaluate_rig_frame` samples the same 3D path used for the
rendered frame. Its versioned `RigEvaluationReport` records asset, profile and
Action provenance; active/inactive Action layers and normalized phase; each
canonical bone's driver, stage transforms and effective axis declarations;
ground/contact settings; and final screen projection. Missing stages remain
explicitly unavailable instead of being estimated.

`compare_humanoid_poses` compares two reports by canonical bone and returns
angular/endpoint errors, the first divergent stage, a rule-based root-cause
category and evidence. `propose_rig_calibration` is deliberately read-only. It
will not recommend `BoneAxisMap` changes when a baked-reference driver bypasses
those semantic axes.

Native command-line workflows are available without changing the DSL:

```text
cargo run -p motionloom --example rig_scene_evaluate -- scene.motionloom actor 120 report.json
cargo run -p motionloom --example rig_scene_evaluate -- scene.motionloom actor phase:standard_walk_loop:0.5 report.json
cargo run -p motionloom --example rig_compare -- reference.json candidate.json comparison.json
cargo run -p motionloom --example rig_calibrate -- comparison.json proposal.json
```

Browser hosts use `WasmSceneRenderer.evaluate_rig_json` for frame, time, or
Action-phase requests, or `evaluate_rig_frame_json` as a convenience. Pure JSON
comparison and calibration entry points are `motionloom_compare_rigs_json` and
`motionloom_propose_rig_calibration_json`. These APIs are additive and do not
add tags or attributes to MotionLoom DSL.

## Experimental Low-level Pose Diagnostics

Offline authoring and parity tests may call
`motionloom::experimental::diagnose_world_actor_pose`. It evaluates existing
Action DSL through the runtime's CPU pose evaluator and returns read-only,
column-major model-global joint matrices. It does not fetch assets, submit GPU
work, mutate playback, or include scene contact/skin deformation.

WASM exposes the additive `WasmPoseDiagnostics` handle with the same evaluation
stage. Prefer the stable Scene report when contact, constraints or camera
projection matter. Existing rendering and editor APIs are unchanged, and the
DSL remains the authored source of truth.

## Scene visual style inspection

`motionloom::api::resolve_scene_render_style(&graph, scene_id)` returns resolved
Scene-owned style/quality settings and explicit-node override evidence. The DSL
remains the source of truth; this JSON is inspection output, not an editing API.
WASM hosts use `motionloom_render_style_json(script, scene_id)`; CLI hosts can
use the `render_style_report` example. See [RENDER_STYLE.md](RENDER_STYLE.md) for
supported settings, precedence, fallback behavior and Rust AST migration notes.
