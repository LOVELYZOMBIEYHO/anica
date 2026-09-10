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
<Scene id="forest" renderStyle="anime_bright">
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
| SurfaceStyle | shading: physical/stylized/toon/clay/cel; shadingSteps: integer 2–16; diffuseWrap: 0–1; rimLight: 0–4; rimPower: 0.1–32; specular: 0–4; roughnessBias: −1–1; saturation: 0–3; outline: none; shadowThreshold: 0–1; shadowFeather: 0.001–0.5; shadowColor: #RRGGBB |
| OutlineStyle | enabled: true/false; method: geometry; width: 0–12 output pixels; color: #RRGGBB; distanceMode: screen |
| LightingStyle | preset: neutral/soft_sunlight/cinematic/overcast/night; ambientIntensity: 0–10; ambientColor: #RRGGBB; shadowStyle: hard/soft |
| PostStyle | toneMapping: none/reinhard/aces; exposure: 0–32 (existing linear multiplier, **not EV**); saturation: 0–3; contrast: 0–3; whiteBalance: 1000–40000 K; bloomThreshold: 0–32; bloomIntensity: 0–4 |

Unknown children, unknown attributes, invalid references, duplicate declarations,
non-finite values and unsupported modes fail before GPU submission. Illustration,
screen-space edge-detection outlines, LUT assets, style inheritance and local style volumes
are **not V1 features** and are not accepted as no-op settings.

## Ownership and precedence

- No `renderStyle`: no compiled style payload; the immediate renderer uses its defaults.
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

## Immediate renderer behavior

Native WGPU and WASM WebGPU share the same WGSL and style resolver. The immediate
renderer uses one fixed fast path: the Graph render size, a 1536 shadow map,
existing analytic ambient occlusion, and no quality-driven anti-aliasing pass.
Light count remains the existing four-light limit. The implementation does not
add GI, ray tracing or SSAO. The CPU-only preview is not a reference
implementation of these 3D shader modes.

Output dimensions belong to Graph `renderSize`, for example
`<Graph ... size={[1280,720]} renderSize={[2560,1440]}>`. A future host render
API may choose slower export settings without turning them into scene semantics.

## Inspection and host integration

Rust: `motionloom::api::resolve_scene_render_style(&graph, scene_id)`.
WASM: `motionloom_render_style_json(script, scene_id)`.
CLI: `cargo run -p motionloom --example render_style_report -- file.motionloom scene_id`.
The existing authoring report also includes `renderStyles`.
No host needs to rewrite the DSL or introduce JSON controls.

The report contains concrete style defaults, references and per-island override
evidence. `finalExpression` is the authored
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
  }]
}
```

## Breaking migration

The quality resource and Scene quality reference have been removed without a
compatibility layer. Old DSL and serialized graph JSON now fail parsing.

Before:

```xml
<RenderQuality id="preview" preset="cinematic">
  <Resolution scale="2" />
  <Shadows resolution="4096" filtering="pcf" />
</RenderQuality>
<Scene id="main" renderStyle="cinema" renderQuality="preview">...</Scene>
```

After:

```xml
<Graph fps="24" duration="10s" size={[1280,720]} renderSize={[2560,1440]}>
  <Scene id="main" renderStyle="cinema">...</Scene>
</Graph>
```

Use `renderSize` only when the output itself needs more pixels. It does not
promise better lighting, shadows, materials or sampling. Existing
Effect/ApplyEffect and Camera3D syntax is unchanged.

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
dimensions. Browser compilation alone does not certify
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

The native legacy-default comparison is pixel-exact on the tested host. That
original V1 verification does not cover the Cel extension below.

## Cel Shading extension

`cel` is the English DSL name for cel shading (賽璐璐). It is a distinct surface
mode, not an image-posterization effect. All settings remain in the authored DSL.

```xml
<RenderStyle id="cel_shading">
  <SurfaceStyle shading="cel" shadingSteps="3"
                shadowThreshold="0.5" shadowFeather="0.025"
                shadowColor="#75658F" specular="0.05" />
  <OutlineStyle enabled="true" method="geometry" width="2"
                color="#000000" distanceMode="screen" />
</RenderStyle>
<Scene id="main" renderStyle="cel_shading">
  <!-- Existing timeline and 3D groups. -->
</Scene>
```

Cel defaults to a 1.5-pixel black outline. Other shading modes default to no
outline; explicitly adding OutlineStyle enables it independently. Width zero
or enabled=false disables it. Legacy SurfaceStyle outline=none is retained;
combining it with OutlineStyle enabled=true is rejected.

The first authored light controls the bands; subsequent lights add attenuated
smooth fill. Ambient illumination and existing shadows still apply. This is
art-directed lighting, not energy-conserving PBR. Color grading and extreme
ambient exposure can reduce the visible contrast between bands.

Optional Model material-slot bindings avoid changing shared materials or GLBs:

```xml
<Model id="hero" asset="character">
  <MaterialBinding material="*" outlineWidth="2" />
  <MaterialBinding material="Face" celRole="face" outlineWidth="0.5"
                   celShadowColor="#B87E87" celControlMap="face_control" />
  <MaterialBinding material="Hair" celRole="hair" hairHighlight="0.3"
                   celControlMap="hair_control" />
</Model>
```

Exact, case-sensitive slot names override the wildcard. Missing slots and
duplicates produce rendering errors. outlineWidth accepts 0–12 pixels,
celShadowColor accepts #RRGGBB, hairHighlight accepts 0–2. Roles are body/skin/hair/face.
Skin currently uses the same band calculation, with separately authored shadow
color; hair adds a tangent-aligned highlight band. Correct UVs/tangents and model
design matter: this does not automatically turn an arbitrary GLB into an anime
character. celControlMap references an ordinary ImageAsset; missing IDs fail.
Use linear-srgb for data textures. No GLB file is modified.

Control-map channels: R multiplies outline width (0 disables, 1 full width);
G stores a normalized face shadow-threshold/SDF ramp; B multiplies the hair
highlight. A is reserved. Without a map, R and B default to one. The face role
requires a map, uses G instead of normal-derived bands, mirrors U with key-light
side, and keeps the backlit face in shadow. It assumes model bind-space +Z
forward and +X right, deformed by the same skin matrices as the face. Assets
with a different facing convention need corresponding authored model/UV setup.
This is a normalized artist-authored threshold map, not a geometric distance
field generated from a face mesh. The S84 map is an illustrative procedural ramp,
not a production anime face texture.

The outline pass reuses skinned geometry, instance buffers and textures.
Completed opaque depth occludes outlines; no CPU skinning or per-frame mesh
regeneration is added. Width scales with island resolution to remain in final
output pixels. Additional outlines add a GPU pass and draw submissions; they
are not free. The material-width override cannot re-enable a globally disabled
outline.

### Current boundaries

- Opaque geometry only; blended/transmissive materials do not receive outlines.
  Opaque materials ignore texture alpha, including solid-color fallback textures.
  Masked materials discard alpha below max(alphaCutoff, 0.99); actor fades below
  0.99 suppress outlines. This is not a complete alpha-card/hair silhouette solution.
- A separate cached outline-normal channel welds coincident positions with
  identical skin weights within each material/mesh chunk. Lighting normals stay
  untouched. Different material boundaries, disconnected shells and unusual
  topology can still require asset-side cleanup.
- Width maps sample the vertex UV at mip zero; fine mask detail needs enough
  geometry. Face maps require authored UVs and the documented facing convention.
- Hair highlights are a basic tangent band, not a dedicated anisotropic hair BRDF.
- CPU preview is not the reference for this GPU feature.

S84 is a new 20-second comparison: Physical, Toon, Cel with outlines, Cel without
outlines, then camera distance/occlusion with an animated CC0 Character1.
No existing showcase is modified for this extension.

Verification is recorded in S84's README. Browser timing is render/readback wall
time, not GPU utilization or sustained FPS. The report's overrides list includes
per-model Cel MaterialBinding settings as read-only evidence.
