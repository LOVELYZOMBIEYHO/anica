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
- GLB/glTF models, PBR materials, live Scene-to-material textures, and cameras
- AnimationTarget keyframes, reusable external humanoid actions, puppet deformation, and IK
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
