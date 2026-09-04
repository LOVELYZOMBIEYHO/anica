# Scene RenderStyle (V1)

RenderStyle is an opt-in Graph resource referenced by `Scene`. It does not
reintroduce the removed `World` DSL tag. The original DSL remains the source of
truth; resolver JSON is read-only evidence, not a second authoring format.

```xml
<RenderStyle id="anime_bright">
  <SurfaceStyle shading="toon" shadingSteps="3" diffuseWrap="0.15"
                rimLight="0.18" specular="0.1" />
  <LightingStyle preset="soft_sunlight" ambientIntensity="0.65" />
  <PostStyle toneMapping="aces" exposure="1.0" saturation="1.15"
             contrast="1.04" bloomThreshold="0.9" bloomIntensity="0.08" />
</RenderStyle>
<RenderQuality id="web" preset="web_high">
  <Shadows resolution="2048" filtering="pcf" />
  <AntiAliasing mode="fxaa" />
</RenderQuality>
<Scene id="forest" renderStyle="anime_bright" renderQuality="web">
  <Timeline>
    <!-- Existing Track / Sequence / CompositeGroup space="3d" hierarchy. -->
  </Timeline>
</Scene>
```

## Contract and supported knobs

Style definitions remain static literal resources. A Scene may switch its
`renderStyle` reference with discrete `AnimationTarget` keys; individual style
parameters are not animation properties and styles never interpolate. Animate
existing light, ColorManagement, camera and Effect nodes for continuous
changes. All identifiers are case-sensitive.

```xml
<AnimationTarget node="forest" property="renderStyle">
  <Key time="0s" value="physical" />
  <Key time="3s" value="toon" />
</AnimationTarget>
```

| Child | Supported attributes |
| --- | --- |
| SurfaceStyle | shading: physical/stylized/toon/clay; shadingSteps: integer 2–16; diffuseWrap: 0–1; rimLight: 0–4; rimPower: 0.1–32; specular: 0–4; roughnessBias: −1–1; saturation: 0–3; outline: none |
| LightingStyle | preset: neutral/soft_sunlight/cinematic/overcast/night; ambientIntensity: 0–10; ambientColor: #RRGGBB; shadowStyle: hard/soft |
| PostStyle | toneMapping: none/reinhard/aces; exposure: 0–32 (existing linear multiplier, **not EV**); saturation: 0–3; contrast: 0–3; whiteBalance: 1000–40000 K; bloomThreshold: 0–32; bloomIntensity: 0–4 |
| RenderQuality | id; optional preset: web_low/web_high/desktop_high/cinematic |
| Resolution | scale: 0.25–2; scales the 3D render island only |
| Shadows | resolution: integer 128–4096; filtering: hard/pcf |
| AmbientOcclusion (inside RenderQuality) | quality: off/low/medium/high |
| AntiAliasing | mode: none/fxaa |

Unknown children, unknown attributes, invalid references, duplicate declarations,
non-finite values and unsupported modes fail before GPU submission. Illustration,
screen/geometry outlines, LUT assets, style inheritance and local style volumes
are **not V1 features** and are not accepted as no-op settings.

## Ownership and precedence

- No `renderStyle` and no `renderQuality`: no compiled style payload; legacy path.
- Style applies to each 3D CompositeGroup owned by its Scene, including groups
  inside nested timelines, sequences and precomposes. It does not recolor SVG UI.
- Shared Primitive/Compound/GLB material resources are not rewritten.
- Surface saturation/specular/roughness bias are explicitly global modifiers of
  material values. Unlit materials retain their unlit behavior. Clay overrides
  lit base color/metallic/roughness for a deliberately material-free study.
- Explicit ColorManagement owns its **whole group**, including its existing
  parser defaults. It overrides style toneMapping/exposure/contrast/whiteBalance.
  Style saturation remains independent. This avoids pretending omitted values
  can be distinguished from defaults in the existing AST.
- Explicit lights or an EnvironmentLight suppress the preset's implicit key.
  Ambient color/intensity remain documented multipliers on ambient illumination.
- DoF, FOV and focus distance remain Camera3D-owned.
- Style Bloom is lowered into an ordinary Process effect on the 3D group,
  before explicitly authored group effects. Explicit Bloom effects compose;
  they do not magically cancel the style Bloom. Set style intensity to zero to
  avoid double Bloom. It uses the existing Bloom pipeline/color convention.
- RenderQuality overrides quality defaults, never scene geometry or camera.
  FXAA shares the optics pass; it filters in-focus edges, while out-of-focus
  regions continue through the existing depth-aware DoF blur.

## Quality reality and portability

Native WGPU and WASM WebGPU share the same WGSL, uniforms and resolver. Shadow
maps resize only when the quality changes. Light count remains the existing
four-light limit. The implementation does not add GI, ray tracing or SSAO.

Existing AO is analytic, not screen-space geometry-aware SSAO. `off` disables it;
low/medium/high currently use that existing algorithm and produce an explicit
`RENDER_STYLE_FALLBACK` warning. No quality name pretends to add missing samples.

Presets resolve to concrete values:

| Preset | Island scale | Shadow map | AA | AO |
| --- | --- | --- | --- | --- |
| web_low | 0.75 | 512 | none | off |
| web_high | 1 | 1536 | fxaa | existing |
| desktop_high | 1 | 2048 | fxaa | existing |
| cinematic | 1.5 | 4096 | fxaa | existing |

Explicit quality child settings override these preset values. Host render-size
and device limits still apply. This is not automatic FPS-adaptive quality.
The CPU-only preview is not a reference implementation of these 3D shader modes.

## Inspection and host integration

Rust: `motionloom::api::resolve_scene_render_style(&graph, scene_id)`.
WASM: `motionloom_render_style_json(script, scene_id)`.
CLI: `cargo run -p motionloom --example render_style_report -- file.motionloom scene_id`.
The existing authoring report also includes `renderStyles` and fallback warnings.
No host needs to rewrite the DSL or introduce JSON controls.

The report contains concrete style/quality defaults, references, fallback
messages and per-island override evidence. `finalExpression` is the authored
override expression, **not a claim about a sampled animation frame**. At render
time the existing animation evaluator supplies the actual value.

```json
{
  "sceneId": "forest",
  "styleId": "anime_bright",
  "shading": "toon",
  "shadingSteps": 3,
  "overrides": [{
    "islandId": "forest_island",
    "property": "post.exposure",
    "styleValue": 1.0,
    "finalExpression": "1.2",
    "source": "ColorManagement:grade"
  }],
  "fallbacks": []
}
```

## Compatibility and migration

Before: `<Scene id="main">...` remains valid with no visual migration.
After: add Graph-level definitions and optional Scene references.
Existing Effect/ApplyEffect and Camera3D syntax is unchanged.
Existing serialized graphs deserialize through serde defaults.
Public AST struct-literal consumers must add the new optional/default fields;
this is a **Rust AST source-compatibility change**, not a DSL breaking change.
The recommended stable integration remains parsing DSL through `motionloom::api`.

## Showcase and validation

S80 `main.motionloom` switches physical/stylized/toon/clay/bright-anime at
three-second cuts over one 15-second Scene. It shares the same procedural
forest, owl, bench, lights and camera throughout. The standalone
`physical.motionloom`, `stylized.motionloom`, `toon.motionloom` and
`clay.motionloom` are controlled comparisons. Separate files avoid the existing
cross-Scene transition/composition limitations of the WASM preview; the main
showcase changes one Scene's style reference instead.
Scene labels remain ordinary 2D overlays.
No external or non-CC0 asset is needed; S1–S79 are not edited.

Run semantic tests with `cargo test -p motionloom --test render_style`.
Run the actual GPU comparison with
`cargo test -p motionloom --test render_style -- --include-ignored`.
GPU tests assert distinct modes, legacy-neutral output, and unchanged output
dimensions under island scaling. Browser compilation alone does not certify
browser pixel parity; a real WebGPU browser smoke run is required separately.

### Verification record

- Native library suite: 525 tests passed.
- RenderStyle suite: 9 tests passed, including GPU rendering and parsing S1–S79.
- Anica binary check and Landing Page production build passed.
- Landing Page release WASM rebuilt (12,545,473 bytes).
- S80 native physical/toon/clay/anime frames inspected.
- S80 also rendered successfully in an actual in-app WebGPU browser using the
  release WASM and host-provided font bytes (1280×720 RGBA).
- Automated ChromeDriver execution was blocked by its HTTP 404 startup failure.
  `wasm-pack test` also builds an unrelated existing Naga-only test that cannot
  compile for WASM; the targeted browser test itself compiles successfully.
- Cross-browser pixel parity is not certified by this smoke test.

The native legacy-default comparison is pixel-exact on the tested host. AO
quality levels remain the explicitly reported analytic-AO fallback described
above; this release does not claim new SSAO or illustration/outline support.
