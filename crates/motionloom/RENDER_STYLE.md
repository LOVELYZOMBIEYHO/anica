# Scene RenderStyle (V1)

## Anti-aliasing

```xml
<RenderStyle id="clean_motion">
  <SurfaceStyle shading="filmic_physical_v1" />
  <AntiAliasingStyle method="taa" quality="high" fallback="smaa" sharpness="0.15" />
</RenderStyle>
```

`method` accepts `auto`, `off`, `fxaa`, `smaa`, `msaa`, `taa`, or `ssaa`.
`quality` accepts `low`, `medium`, `high`, or `ultra`; `sharpness` is in `[0,1]`.
Safe fallbacks are `auto`, `off`, `fxaa`, and `smaa`. Preview profiles may
downgrade an expensive request. Native and Web immediate preview currently run
true FXAA, spatial morphology AA, and depth/normal/velocity-rejected TAA.

| Method | Low | Medium | High | Ultra |
| --- | --- | --- | --- | --- |
| FXAA | fast | balanced | precise | precise |
| SMAA | basic | edge + blend | wider edge search | wider edge search |
| MSAA intent | 2x | 2x | 4x | 8x |
| TAA | no jitter | 2 phases | 4 phases | 8 phases |
| SSAA intent | 1.25x | 1.5x | 2x | 4x |

FXAA and spatial morphology AA use discrete shader thresholds and search cost,
so Low, Medium and High produce measurably different pixels; their documented
High and Ultra modes intentionally match. TAA uses a bounded sub-pixel pattern,
stable-grid history, and de-jittered physical velocity so its phase count does
not become whole-frame camera shake.
`msaa` and `ssaa` are accepted portable intents but currently report and use
their fallback in immediate preview. Fallback quality is retained, so spatial
fallbacks still follow the Low/Medium/High shader tiers, but they are not
presented as native 2x/4x/8x or supersampled rendering. Weaver resolves edge sampling through its
per-pixel sample job. A RenderStyle without `AntiAliasingStyle` resolves to `off`.
Only a Scene without any RenderStyle retains the host preview AA policy.

Migration example:

```xml
<!-- Before: AA is now intentionally off. -->
<RenderStyle id="old_style">
  <SurfaceStyle shading="physical" />
</RenderStyle>

<!-- After: opt into an explicit, portable AA policy. -->
<RenderStyle id="old_style_with_aa">
  <SurfaceStyle shading="physical" />
  <AntiAliasingStyle method="taa" quality="high" fallback="smaa" sharpness="0.1" />
</RenderStyle>
```

## Filmic physical and lens presets

```xml
<RenderStyle id="filmic_scene">
  <SurfaceStyle shading="filmic_physical_v1" />
  <DepthOfFieldStyle preset="filmic_bokeh_v1" aperture="0.00032" maxBlur="0.0025" />
  <LightingStyle ambientIntensity="0" />
  <PostStyle toneMapping="filmic_aces_v1" exposure="0.97" whiteBalance="6500" bloomIntensity="0" />
</RenderStyle>
```

Enable the camera with `depthOfField="true" focusDistance="16.2"`.
The bokeh preset uses signed `(focus - viewDepth) * aperture` CoC and a fixed
41-sample golden-angle disk. Style `aperture` and `maxBlur`
are static normalized-UV controls in [0, 0.1], defaulting to 0.00032 / 0.0025.
They are not f-numbers or pixels. Camera focalLength/fStop/maxBlur do not control
this preset; camera focusDistance/focusTarget/focusOffset still do. `quality` is
accepted but does not alter its fixed 41 taps. Set aperture to zero or disable
camera DoF to remove blur. At 4K, maxBlur 0.0025 and the outer tap scale of 0.4
produce a maximum radius of approximately 3.84 pixels, versus the earlier
cinematic preset's 43.2-pixel cap with camera maxBlur=2 percentHeight.

`filmic_physical_v1` keeps glTF factors linear, converts authored light colors
using RGB and exact sRGB, uses correlated Smith visibility and Lambert diffuse,
and shadows only the selected shadow owner. Environment irradiance remains active
at zero ambient fill. `filmic_aces_v1` uses filmic ACES input/output matrices,
an RRT/ODT fit, a 1/0.6 scale and exact sRGB output.
These opt-in modes preserve existing physical/aces/cinematic_bokeh_v1 behavior.

The material/light preset provides a standards-oriented glTF physical path; environment
prefiltering, shadows, fog and rasterization remain MotionLoom implementations.
The bokeh kernel retains premultiplied alpha for compositor compatibility. Depth
comes from the reversed depth buffer. The filmic ACES curve is intended for
physically shaded scenes; legacy
display-locked materials still use their old inverse display curve.


## Ink preset and universal controls

`SurfaceStyle shading="ink_wash_soft_v1"` selects a complete, versioned GPU
preset. Its dedicated `world/render/shaders/presets/ink_wash_soft_v1.wgsl`
uses a physically lit scene as input, depth-aware wash filtering, five soft ink
densities, one-sided contour deposition, world-anchored pigment variation and
stationary paper grain. This is stylized ink rendering, not fluid simulation.
The preset preserves material chroma and applies ink density primarily to
luminance. For reference-like pastel colour, prefer a light universal tone blend
(A light toneStrength and modest saturation preserve the authored palette); strong duotone blends deliberately
replace more of the original palette. Far contours are faint and highlights
receive less contour ink.
It needs no temporal history, external textures, new render targets or quality
DSL. Native and WASM GPU paths embed the same WGSL.

```xml
<RenderStyle id="courtyard_ink">
  <SurfaceStyle shading="ink_wash_soft_v1" />
  <ColorStyle tint="#DDE5D7" tintStrength="0.15" saturation="0.75" />
  <ToneStyle exposure="1.0" contrast="0.95" shadowColor="#293B38"
             highlightColor="#F1EBDD" toneStrength="0.65" />
</RenderStyle>
```

`ColorStyle` and `ToneStyle` are optional, static, strict children available to
**every** shading mode. They are applied by the shared resolve shader, not by
individual presets. Future presets must return display RGB through that same
finish function. Unsupported shading names are errors, never silent fallbacks.

| Parameter | Range / format | Neutral default |
| --- | --- | --- |
| ColorStyle.tint | #RRGGBB | #FFFFFF |
| ColorStyle.tintStrength | 0–1 | 0 |
| ColorStyle.saturation | 0–3 | 1 |
| ToneStyle.exposure | 0–32, linear-light multiplier, not EV | 1 |
| ToneStyle.contrast | 0–3, display-space pivot 0.5 | 1 |
| ToneStyle.shadowColor | #RRGGBB | #000000 |
| ToneStyle.highlightColor | #RRGGBB | #FFFFFF |
| ToneStyle.toneStrength | 0–1 | 0 |

Order: scene lighting and transparent surfaces → optional froxel volume
composite → temporal resolve and DoF → display tone mapping → preset
→ universal exposure → contrast → luminance-based shadow/highlight gradient
blend → multiplicative tint blend → saturation → clamp. Exposure uses the
renderer's gamma-2.2 working conversion. Tone colours interpolate by display
luminance; tint multiplies RGB and blends by tintStrength, preserving black.
Saturation is last, so zero guarantees monochrome even with coloured tone/tint.
Controls operate on straight RGB and preserve premultiplied alpha.

These controls affect the entire rendered 3D island, including its environment
background, but not separate SVG/screen UI. Explicit `ColorManagement` and legacy
`PostStyle` keep their previous precedence; universal controls run afterwards
and compose with them, not replace them. Later Process effects, including style
bloom, run after this island resolve. They can intentionally alter the result.

### Additive migration

Before: `<SurfaceStyle shading="physical" />` with optional legacy children.
After: the same source remains valid and unchanged; optionally add ColorStyle
and ToneStyle or choose ink_wash_soft_v1. Omitted/new empty controls are neutral.

`SurfaceStyle shading="pbr_npr_soft_v1"` keeps the filmic PBR material and
lighting path, then applies restrained painterly tone simplification, bilateral
colour softening and soft normal/depth accents. It preserves metallic and
specular highlights and deliberately does not draw a hard outline. Universal
`ColorStyle` and `ToneStyle` controls run after the preset.
Old serialized graphs and resolved reports default to disabled neutral universal
controls. Older engine versions do not understand the new tags/preset and must
be rebuilt before loading such a document. CPU-only previews are not a reference
implementation of the GPU styles.

See [the self-contained template](examples/ink_wash.motionloom). Preset-specific
shader constants stay internal;
there is no new RenderStyle `preset` attribute.

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
| SurfaceStyle | shading: physical/stylized/toon/clay/cel/ink_wash_soft_v1/pbr_npr_soft_v1; shadingSteps: integer 2–16; diffuseWrap: 0–1; rimLight: 0–4; rimPower: 0.1–32; specular: 0–4; roughnessBias: −1–1; saturation: 0–3; outline: none; shadowThreshold: 0–1; shadowFeather: 0.001–0.5; shadowColor: #RRGGBB |
| OutlineStyle | enabled: true/false; method: geometry; width: 0–12 output pixels; color: #RRGGBB; distanceMode: screen |
| LightingStyle | preset: neutral/soft_sunlight/cinematic/overcast/night; ambientIntensity: 0–10; ambientColor: #RRGGBB; shadowStyle: hard/soft |
| PostStyle | toneMapping: none/reinhard/aces; exposure: 0–32 (existing linear multiplier, **not EV**); saturation: 0–3; contrast: 0–3; whiteBalance: 1000–40000 K; bloomThreshold: 0–32; bloomIntensity: 0–4 |

ColorStyle and ToneStyle are documented above and share all shading modes.
Unknown children, unknown attributes, invalid references, duplicate declarations,
non-finite values and unsupported modes fail before GPU submission. Illustration,
custom screen-space OutlineStyle methods, LUT assets, style inheritance and local style volumes
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

## Validation and compatibility

S80 `main.motionloom` switches physical/stylized/toon/clay/bright-anime at
three-second cuts over one 15-second Scene. It shares the same procedural
forest, owl, bench, lights and camera throughout. The standalone
`physical.motionloom`, `stylized.motionloom`, `toon.motionloom` and
`clay.motionloom` are controlled comparisons. Separate files avoid the existing
cross-Scene transition/composition limitations of the WASM preview; the main
an example changes one Scene's style reference instead.
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
No existing scene document is modified for this extension.

Verification is recorded in S84's README. Browser timing is render/readback wall
time, not GPU utilization or sustained FPS. The report's overrides list includes
per-model Cel MaterialBinding settings as read-only evidence.
