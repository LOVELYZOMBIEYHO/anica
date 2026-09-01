# MotionLoom

MotionLoom is an agent-native motion graphics and compositing engine for Rust
and WebAssembly. It turns a portable text DSL into deterministic 2D, 2.5D,
true-3D, and GPU effect compositions.

MotionLoom powers motion graphics in [Anica](https://github.com/LOVELYZOMBIEYHO/anica),
but the crate is designed to be embedded independently in editors, renderers,
automation systems, and LLM authoring tools.

- Declarative timeline and scene DSL
- Unified 2D, 2.5D, 3D, and Process render graph
- WebGPU rendering with CPU fallback paths
- Resolution-independent vector shapes and resolution-aware text
- GLB/glTF models, PBR materials, HDR/EXR lighting, atmospheric fog, camera DoF, and shadows
- AnimationTarget keyframes, reusable external humanoid actions, puppet deformation, and IK
- Deterministic swept humanoid collision, action-aware traversal, and contact correction
- Unified deterministic 2D/3D rigid bodies with static, dynamic, and kinematic modes
- Static analysis and machine-readable repair reports for LLM workflows
- Native Rust and browser WASM integration

[Documentation](https://docs.rs/motionloom) ·
[Interactive demo](https://lovelyzombieyho.github.io/anica-landing-page/motionloom/) ·
[Examples](https://github.com/LOVELYZOMBIEYHO/motionloom-example)

## Status

MotionLoom is under active development. Version `0.1.x` keeps existing DSL
documents working where practical, while the recommended public Rust surface is
being consolidated under `motionloom::api`.

The public authoring root is `<Scene>`. True-3D content is placed in a
`space="3d"` Scene track; `<World>` is not part of the current public DSL.

First-person and editorial camera cuts reuse `Camera3D`, `Anchor`, and the
Scene `activeCamera` animation property. A first-person camera may declare
`hiddenBones={["hero:head"]}` to hide its owner's head only from that camera's
view while retaining the complete animated body and shadow for later shots.

`AtmosphereFog` remains global when bounds are omitted. Add matching
`boundsMin`/`boundsMax` values and optional `edgeFeather` to confine fog to a
world-space box, such as the exterior beyond a station entrance. Native WGPU
and browser WebGPU use the same bounded ray integration.

## Install

```toml
[dependencies]
motionloom = "0.1"
```

MotionLoom requires Rust 1.85 or newer. `wgpu` is part of the core renderer and
is enabled by default.

Video export requires an FFmpeg binary supplied by the host. Parsing,
single-frame rendering, preview, and PNG sequence export do not require FFmpeg.

## Quick Start

A MotionLoom document is ordinary text and can be stored, generated, reviewed,
or transferred without a project-specific binary container.

```xml
<Graph fps={30} duration="3s" size={[1280,720]}>
  <Background color="#090B12" />

  <Scene id="hello_motionloom">
    <Timeline>
      <Track id="title" space="screen" compositeOrder="10">
        <Sequence from="0s" duration="3s" out="hold">
          <Layer>
            <Circle x="640" y="310" radius="110" color="#B6FF35" />
            <Text id="headline"
                  x="640" y="520"
                  value="MOTIONLOOM"
                  fontSize="72"
                  align="center"
                  color="#F7F8FA" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <AnimationTarget node="headline" property="opacity">
    <Key time="0s" value="0" />
    <Key time="0.7s" value="1" ease="ease_out" />
  </AnimationTarget>

  <Present from="hello_motionloom" />
</Graph>
```

Parse it through the curated API:

```rust
use motionloom::api::{compile_render_pass_dag, parse_graph_script};

let source = std::fs::read_to_string("intro.motionloom")?;
let graph = parse_graph_script(&source)?;
let render_graph = compile_render_pass_dag(&graph)?;

println!("{} render passes", render_graph.nodes.len());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `SceneRenderer` for retained multi-frame preview, or the scene export
helpers for image sequences and video output.

Rigid bodies use one explicit tag in both dimensions:

```xml
<Physics gravity={[0,-9.81,0]} fixedStep="1/120s" iterations="4" />
<RigidBody id="crate_body" target="crate"
           dimension="3d" type="dynamic" shape="box"
           size={[1,1,1]} mass="2" friction="0.6"
           rollingFriction="0.08" restitution="0.25"
           restitutionThreshold="0.5" sleep="true" />
```

`dimension` and `type` are required. There is no separate `RigidBody2D` API;
this keeps schema discovery and LLM repair rules identical across 2D and 3D.
2D bodies may declare local `gravity`; 3D bodies use their Scene's `<Physics>`
gravity so every object shares one deterministic world force.
3D dynamics use quaternion orientation, shape-derived inertia, rotational
contact impulses and sustained linear-plus-angular sleep tests. Use
`rollingFriction` to stop residual ground spin without excessive global
`angularDamping`.

Reusable generated 3D geometry uses typed assets:

```xml
<Assets>
  <ImageAsset id="stone_color" src="stone.jpg" colorSpace="srgb" />
  <MaterialAsset id="stone" shading="pbr" baseColorTexture="stone_color"
                 metallic="0" roughness="0.84" mapping="triplanar"
                 textureScale={[0.3,0.3]} variationAmount={[0.2,0.15]} />
  <PrimitiveAsset id="ball" shape="sphere" radius="0.5"
                  segments="32" color="#50E3E6"
                  collision="solid" collider="auto" />
  <PrimitiveAsset id="step" shape="box" size={[4,0.3,0.9]}
                  material="stone" bevelRadius="0.025" bevelSegments="3"
                  collision="solid" collider="box" />
</Assets>
<Model id="ball_model" asset="ball" position={[0,4,0]} />
<RigidBody id="ball_body" target="ball_model"
           dimension="3d" type="dynamic" shape="auto" />
```

`PrimitiveAsset` supports `box`, `sphere`, `capsule`, `plane`, `cylinder`,
`cone`, and `wedge`. It shares the normal Model PBR, lighting, shadow, bounds, cache, and
physics paths. The former encoded `motionloom:box` ModelAsset source has been
removed.

Collision is disabled by default. `collision="solid|sensor"` enables it;
omitted `collider` means `auto`, while an explicit collider shape and size may
intentionally differ from the visual primitive. `CompoundAsset` groups
transformed PrimitiveAsset instances into one reusable visual/collision asset.
`MaterialAsset shading="pbr"` exposes the existing glTF material pipeline to
typed primitives, including base-color, metallic/roughness, normal, occlusion
and emissive texture slots. `color` remains a multiplicative tint. Box bevels
are visual-only and preserve authored bounds; auto collision still uses the
unbeveled canonical box. `materialSeed` adds deterministic per-instance UV
variation without moving collision surfaces.

Heightfield ground is an additive typed asset and uses the same Model/PBR path:

```xml
<ImageAsset id="height" src="forest-height.png" colorSpace="linear-srgb" />
<ImageAsset id="blend" src="forest-splat.png" colorSpace="linear-srgb" />
<TerrainAsset id="forest" heightMap="height" size={[40,40]}
              heightScale="4" heightOffset="-1"
              layers={["soil","moss","stone","leaves"]} blendMap="blend"
              chunks={[4,4]} lod="auto" collision="solid" />
<Model id="ground" asset="forest" />
```

`TerrainAsset` converts a grayscale height map into a smooth triangle
heightfield. `lod="full|half|quarter|auto"` controls source sampling; `auto`
caps either height-map dimension at approximately 257 retained samples.
`collision="solid"` feeds the existing environment grounding and kinematic
controller queries. An optional RGBA `blendMap` combines up to four PBR
`MaterialAsset` ids in R, G, B, and A order at load time, so native and WASM
renderers receive the same retained mesh and textures. Heightfields remain
2.5D; use a GLB Environment for caves, overhangs, and vertical interiors.
`chunks={[x,z]}` partitions the retained terrain into independent GPU draws;
off-camera terrain chunks are conservatively frustum-culled without changing
the collision mesh.

The Model id of a solid terrain is also a direct humanoid ground provider:

```xml
<Model id="terrain_ground" asset="forest" />
<Model id="hero" asset="hero_asset" collision="kinematic" />
<ApplyAction target="hero" action="walk" ground="terrain_ground"
             contactCorrection="auto" footLock="auto" />
```

This is additive: `ground="floor"` continues to resolve an existing
`<Surface id="floor" ... />` exactly as before. A Terrain Model binding uses
the generated solid collision triangles for slopes and uneven height, and the
same deterministic CPU query feeds native WGPU and WASM random-access frames.
The authoring analyzer warns when a position-animated humanoid uses an Action
in a scene with solid terrain but omits `collision="kinematic"`.

Terrain migration is not required. Before this addition, existing
`ModelAsset`, `PrimitiveAsset`, and `CompoundAsset` ground scenes render as
authored. After it, those declarations remain unchanged; authors opt in only
by declaring a new `TerrainAsset` and referencing it from a normal `Model`.

Procedural vegetation is also an additive typed asset:

```xml
<VegetationAsset id="forest_tree" kind="tree" height="7"
                 trunkMaterial="bark" foliageMaterial="leaves"
                 density="28" branchLevels="3" seed="12"
                 lod="auto" wind="true" collision="solid" />
<VegetationAsset id="forest_fern" kind="fern" height="0.8"
                 material="fern_leaves" density="18" seed="77"
                 lod="auto" wind="true" />
<VegetationAsset id="old_stump" kind="deadwood" height="1.2"
                 trunkMaterial="old_bark" branchLevels="1" seed="18" />
```

V1 supports `tree`, `shrub`, `grass`, `flower`, `fern`, and `deadwood`.
Trees and shrubs use `trunkMaterial` plus `foliageMaterial`; grass and ferns
use `material`; flowers use `material` and an optional `stemMaterial`; deadwood
uses `trunkMaterial`. `density` controls the bounded amount of foliage, blades,
stems, or fronds inside one asset, not world scattering. `branchLevels` is
limited to tree, shrub, and deadwood. Geometry and atlas-cell choices are
deterministic for a given `seed`.

`lod="full|half|quarter|auto"` adjusts procedural detail. Auto LOD is selected
from camera distance relative to authored height. Wind is a lightweight GPU
vertex deformation shared by native and WASM rendering; it does not rebuild
meshes. `collision="solid"` is intentionally limited to tree and deadwood and
uses a coarse trunk cylinder rather than foliage triangles. Repeated Models of
the same resolved asset reuse retained mesh and texture resources. V1 does not
include a scatter system, biome simulation, or runtime-growing plants.

Vegetation migration is not required. Existing ModelAsset, PrimitiveAsset,
CompoundAsset, and TerrainAsset declarations remain unchanged. Authors opt in
by declaring a VegetationAsset and placing it through a normal Model node.

A `CompoundAsset` can also be a native rigid hierarchy without becoming a GLB:

```xml
<CompoundAsset id="hero" rig="hero_rig">
  <Instance asset="torso" bone="chest" />
  <Instance asset="limb" bone="upper_arm_l" position={[0,-0.2,0]} />
</CompoundAsset>
<Skeleton id="hero_rig" profile="motionloom_humanoid_v1" space="3d">
  <Bone id="root" role="root" position={[0,0,0]} />
  <Bone id="chest" role="chest" parent="root" position={[0,1.25,0]} />
  <Bone id="upper_arm_l" role="upper_arm" side="left" parent="chest"
        position={[-0.25,0.1,0]} />
</Skeleton>
```

`space="3d"` keeps the existing `Skeleton`, `Bone`, `Action`, and
`ApplyAction` vocabulary. Bone `position` and `rotation` are local XYZ values;
the legacy `x`, `y`, and scalar `rotation` attributes remain unchanged for 2D
rigs. Each bone-bound `Instance` inherits the complete parent hierarchy.
Authored Action `rotationX/Y/Z` values are additive to the rest pose, while
`forward/bend`, `turn/twist`, and `side` map to canonical X, Y, and Z axes.
Visual parts normally keep collision disabled and share one kinematic capsule
`RigidBody` on the parent Model. Set `continuousCollision="true"` on that
feet-rooted capsule when a timeline animates the parent position. MotionLoom
then sweeps the native rig from its authored start to the current target and
expands the primitive children from the collision-resolved root, so direct
timeline motion cannot tunnel through solid PrimitiveAsset walls.

Authored and imported humanoid Actions share one contact pipeline:

```xml
<Action id="walk" skeleton="humanoid_v1" duration="1.066s">
  <Pose t="0s">...</Pose>
  <Pose t="1.066s">...</Pose>
  <Contact id="left_plant" effector="foot_l" target="ground"
           from="0" to="0.36" mode="lock" />
</Action>
<ApplyAction target="hero" action="walk" loop="true"
             rootMotion="in_place" ground="floor"
             contactCorrection="auto" footLock="auto" />
```

`Contact` times are normalized Action phases. Canonical Pose Actions and
external clips both publish those phases to the Scene solver. Ground
correction owns vertical root placement, while `footLock="auto"` reconstructs
a deterministic world-space support target and applies two-bone IK. This
keeps random-access rendering and sequential preview identical without a
mutable animation-history cache. `ApplyAction` blend-in and blend-out apply to
the authored timeline window; a looping Action does not fade back to bind pose
at every internal cycle seam. Existing scripts remain unchanged unless they
explicitly opt into contact correction and foot locking.
`ground` accepts either a semantic Surface id or the Model id of a
`TerrainAsset collision="solid"`; Surface bindings remain fully compatible.

Non-ground contacts use an additive semantic `ContactSurface`. This avoids a
full rigid-body simulation while allowing the same Action to sit on differently
sized props and differently proportioned humanoids:

```xml
<ContactSurface id="bench_seat" source="bench_seat_model" kind="seat"
                plane="top" forward={[0,0,1]} bounds={[2.8,0.72]}
                margin="0.02" />
<Action id="sit" skeleton="humanoid_v1" duration="2s">
  <Pose t="0s">...</Pose>
  <Contact id="pelvis_seat" effector="pelvis" target="seat"
           from="0.62" to="1" mode="surface" weight="1" />
</Action>
<ApplyAction target="hero" action="sit" contactCorrection="auto"
             contactTargets={{ seat: "bench_seat" }} />
```

`plane="top"` derives the plane from a PrimitiveAsset Model; explicit
`position`, `normal`, and `forward` are available for imported or compound
props. The renderer resolves the plane in world space, clamps correction to its
bounds, and estimates a scale-aware pelvis contact offset. Contacts beginning
at phase zero remain active across direct seek and Action transitions. Existing
`ground` and Actions without `contactTargets` keep their previous behavior.

Each authored Bone key may choose `interpolation="linear|hold|ease|bezier"`.
The default remains `linear`. A Bezier key may additionally provide numeric
`inTangent` and `outTangent` values; these shape the normalized transition to
the next pose without changing the canonical Bone values themselves. The 2D,
native-rig, and imported-GLB Action paths use the same interpolation contract.

External animation is an offline authoring input, not a runtime dependency.
The separate `motionloom-action-tool` workspace crate can inspect an animated
glTF/GLB and export a standalone `humanoid_v1` `<Action>`. The generated file
contains only MotionLoom Pose/Bone data and can be edited in the Action Editor,
committed to an Action Library, or generated by an LLM without loading the
source clip at render time.

Large authored Actions can remain in a standalone MotionLoom document and be
selected into a Graph with an `ActionLibrary` declaration:

```xml
<ActionLibrary id="performance" src="actions/performance.motionloom"
               actions={["formal_bow","stand_up"]} />
<ApplyAction target="hero" action="performance.formal_bow" />
```

The external file uses `<ActionLibrary> ... <Action> ... </ActionLibrary>` as
its root. Imported ids are always namespaced by the declaration id, only the
listed Actions enter the executable graph, and the parsed result is retained
across preview frames. Relative paths resolve from the main `.motionloom`
project root. Browser hosts preload the file under the unchanged `src` key.
This is additive: inline `Action` and AnimationAsset-backed `Action` keep their
existing behavior.

Migration is optional. Before, authors pasted the complete `<Action>` block
directly under `<Graph>` and referenced `action="formal_bow"`. After moving
that unchanged block under an external `<ActionLibrary>` root, declare the
selection in the Graph and reference `action="performance.formal_bow"`.

`humanoid_v1` full body conformance requires the 22 canonical body bones,
including separate `chest` and `upper_chest` joints. Its 30 named finger bones
are canonical but optional: an Action that does not key fingers remains fully
usable on a hand rig without them. Tooling can call
`inspect_humanoid_action_compatibility` to distinguish a complete profile from
degraded playback when an Action actually references an unmapped optional
bone. Existing profile DSL remains parseable; this check is diagnostic and
does not silently alter playback.

Transmissive surfaces use the same tag instead of a glass-specific node:

```xml
<MaterialAsset id="glass" shading="pbr" baseColor="#E8F7FA"
               roughness="0.08" specular="1" transmission="0.94"
               ior="1.52" thickness="0.012"
               attenuationColor="#B7DDE2" attenuationDistance="6"
               depthWrite="auto" doubleSided="true" />
```

`transmission` describes light passing through a solid surface and is distinct
from `alphaMode="blend"`, which describes partial coverage such as smoke or a
fade. Opaque and mask draws fill depth first; blend and transmissive draws are
then sorted far-to-near. `depthWrite="auto"` therefore writes depth for opaque
materials and disables it for transparent ones. `true|false` and integer
`sortPriority` are expert overrides. Visual transmission never enables or
changes PrimitiveAsset collision.
The transmissive pass samples one renderer-owned opaque-scene snapshot for IOR
normal refraction and thickness attenuation. That snapshot is reused by every
glass draw and across frames; adding panes does not allocate another target.

World-space atmospheric instances extend the existing `Repeat` tag inside a
3D CompositeGroup; there is no rain- or particle-specific node:

```xml
<Repeat id="rain" mode="volume" count="120" seed="77"
        boundsMin={[-12,3,-18]} boundsMax={[12,13,-3]}
        velocity={[-0.7,-18,0.4]} lifetime="0.65s"
        phase="random" respawn="random" scaleRange={[0.8,1.25]}>
  <Model asset="rain_streak" castShadow="false" receiveShadow="false" />
</Repeat>
```

`mode="volume"` requires `CompositeGroup space="3d"` and exactly one
self-closing Model template. Positions, phases, respawn cycles, and scale
variation are seed-stable for deterministic preview, export, and replay. The
same feature works for rain, snow, dust, embers, and debris. Existing 2D
linear/grid/scatter Repeat behavior is unchanged.

Primitive resources are retained independently: geometry and PBR material
identity exclude `materialSeed`, decoded `ImageAsset` pixels are shared by
source revision, and GPU textures are shared across different primitive
shapes. Seed variation is passed as per-instance shader data. A staircase may
therefore vary every tread without decoding or uploading the same stone image
for every step. File metadata and in-memory resolver byte revisions invalidate
only the affected texture; geometry and collision caches remain intact.

## Architecture

MotionLoom compiles authored content into one validated render-pass DAG:

```text
MotionLoom DSL
    ├── Scene timeline: 2D / 2.5D / 3D
    ├── Assets and Scene texture dependencies
    └── Process passes and effects
                ↓
       Validated render-pass DAG
                ↓
    WebGPU preview / image / video export
```

Explicit dependencies allow the engine and authoring tools to inspect ordering,
texture flow, effect scope, unresolved references, and cycles before rendering.

## Agent Authoring

MotionLoom supports a structured authoring loop rather than treating generated
text as an opaque program:

```text
Example retrieval → Syntax discovery → DSL authoring
                  → Static analysis and repair → Render verification
```

`motionloom_analyze_script_json()` returns parse and compile status,
source-addressed diagnostics, effective behavior, and recommended repairs.
Per-showcase `schema.json` files describe the syntax demonstrated by individual
examples. The complete protocol is documented in
[LLM_AUTHORING.md](LLM_AUTHORING.md).

## Visual Action Authoring

MotionLoom exposes typed editor helpers for authored humanoid motion without
making a browser UI reimplement the DSL parser. `extract_editable_action_document`
returns Actions, poses, canonical Bone channels, contacts, ApplyAction bindings,
skeletons, and model targets. `apply_action_edit` applies a typed command,
preserves unrelated source text, and re-parses the resulting graph before it is
returned.

Browser hosts use the matching WASM functions
`motionloom_editable_actions_json` and `motionloom_apply_action_edit`. The Anica
landing-page playground exposes them through the **Action** option directly
below **Puppet Warp**, while keeping the DSL textarea as the source of truth.

## Rig Diagnostics

Humanoid parity checks use a non-mutating API, not extra DSL. Call
`SceneRenderer::evaluate_rig_frame` to capture the exact rendered pose and
`compare_humanoid_poses` to compare two actors or documents by Action phase and
canonical bone. Reports include provenance, active layers, pose drivers,
effective axis maps, contact settings, stage transforms and screen projection.
The same report is available in browsers through
`WasmSceneRenderer.evaluate_rig_json`; CLI examples are documented in
[PUBLIC_API.md](PUBLIC_API.md), with the report contract and comparison rules in
[RIG_DIAGNOSTICS.md](RIG_DIAGNOSTICS.md).

## Public API

New integrations should start with:

- `motionloom::api` — curated integration surface
- `motionloom::prelude` — small convenience import set
- `motionloom::experimental` — advanced or lower-stability editor helpers

The crate root retains broader re-exports for compatibility with existing
hosts. See [PUBLIC_API.md](PUBLIC_API.md) for parsing, rendering, GPU texture,
export, compatibility inspection, and authoring-analysis APIs.

## Platform Support

| Target | Primary path |
| --- | --- |
| macOS | Rust + wgpu/Metal |
| Windows | Rust + wgpu/DX12 |
| Linux | Rust + wgpu/Vulkan where available |
| Browser | WASM + WebGPU, with supported fallback paths |

Actual GPU availability depends on the host adapter, browser, and enabled
graphics backend. Hosts that require broad compatibility should expose a clear
fallback or renderer-capability report.

## Documentation

- [LLM Authoring Guide](LLM_AUTHORING.md) — DSL rules and agent repair protocol
- [Public API](PUBLIC_API.md) — supported Rust integration surface
- [Changelog](CHANGELOG.md) — release history
- [Path DSL benchmark](benchmarks/path-dsl/README.md) — benchmark methodology
- [MotionLoom examples](https://github.com/LOVELYZOMBIEYHO/motionloom-example) — portable core and showcase documents

## Contributing

Issues and focused pull requests are welcome through the
[Anica repository](https://github.com/LOVELYZOMBIEYHO/anica). Please include a
minimal `.motionloom` reproduction for parser or renderer issues and identify
the native or WASM target used for testing.

## License

MotionLoom is available under the [Apache License 2.0](../../LICENSE).
