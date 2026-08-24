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
- GLB/glTF models, PBR materials, HDR/EXR image-based lighting, shadows, and cameras
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

`PrimitiveAsset` supports `box`, `sphere`, `plane`, `cylinder`, `cone`, and
`wedge`. It shares the normal Model PBR, lighting, shadow, bounds, cache, and
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
