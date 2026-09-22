# Immediate preview: material foundation

## Host quality contract

`ImmediatePreviewSettings` separates editor performance from authored
`RenderStyle` and offline output. Hosts can select `Portable`, `Balanced`,
`Cinematic`, or `Ultra`; each profile maps to concrete shadow-map,
texture-filtering, light-count, DoF-sampling, HDR, screen-space lighting, and
antialiasing budgets. The parsed graph and requested output dimensions are
never rewritten by selecting a profile.

```rust
use motionloom::{
    ImmediatePreviewProfile, ImmediatePreviewSettings, WgpuPreviewEngine,
};

let mut preview = WgpuPreviewEngine::new_with_cpu_fallback().await;
preview.set_settings(ImmediatePreviewSettings {
    profile: ImmediatePreviewProfile::Cinematic,
    target_fps: 30.0,
    dynamic_resolution: true,
    min_resolution_scale: 0.5,
});
```

`WgpuPreviewEngine::capabilities()` reports the actual immediate features,
including whether the host uses the browser WebGPU tier, its bounded SSR/SSGI
sample limits, and the deliberate absence of local reflection probes.
`last_frame_metrics()` combines GPU timing,
Scene CPU timing, triangle/light counts, retained-cache counts, shadow-map
resolution, and estimated render-target memory.

The quality degradation order is resolution (respecting the configured floor),
optical samples, then profile-controlled shadow and filtering cost. Authored
materials, animation, camera, colors, and style identity remain unchanged.

The improvements are renderer internals shared by native and WASM WebGPU.
Anti-aliasing is additionally an explicit RenderStyle policy:

```xml
<AntiAliasingStyle method="taa" quality="high" fallback="smaa" sharpness="0.15" />
```

Scenes without a RenderStyle retain the host profile policy. A referenced
RenderStyle without `AntiAliasingStyle` deliberately resolves to `off`; this is
the compatibility break that makes AA cost explicit. Immediate preview supports
`off`, FXAA, compact spatial morphology AA, and history-rejected TAA. Portable
`msaa` and `ssaa` intents use the requested safe fallback until a backend offers
their required multisample targets or internal-resolution path. The last-frame
profile reports requested/effective methods and whether a fallback occurred.

- Material textures receive retained mip chains and 8x anisotropic filtering.
  Color filtering uses linear light with alpha-weighted mip generation; normal
  mip vectors are normalized; metallic, roughness and AO remain linear data.
  Texture caching includes the semantic role, including when a GLB reuses one
  image for multiple material slots.
- Material AO is sampled separately and attenuates indirect diffuse and specular
  light. It no longer stains base color or direct lighting. Primitive materials,
  GLB materials and blended terrain carry independent AO; missing AO is neutral.
- The 3D intermediate target and transmission snapshot use RGBA16Float. Tone
  mapping and output gamma run once after transparency and camera depth of field.
  Exposure, white balance and contrast remain before blending. SceneCompositor
  receives and blends linear-premultiplied RGBA16F; display readback is encoded
  only after the HDR composition is complete.
- The primary opaque geometry pass uses MRT to retain material normal, velocity,
  roughness, metallic, AO and a temporal reactive mask while it writes shaded
  HDR. Per-object motion comes from previous model transforms and previous bone
  palettes. This removes the former extra geometry submission and fullscreen
  normal-reconstruction pass.
- Temporal resolve reprojects the prior display frame. Host-controlled unstyled
  scenes retain the profile's stable projection. An explicit `taa` style uses a
  bounded Halton phase count selected by `quality`; depth, normal, velocity and
  reactive evidence reject invalid history so thin geometry does not accumulate
  the former long trails. History is
  rejected after a non-sequential seek, camera cut, size change or
  preview-profile change. Motion vectors use
  interpolated current and previous clip positions instead of a quantized
  fragment pixel centre; history confidence and motion blur use de-jittered
  physical velocity.
- Material-aware SSR uses profile-bounded ray marching, binary
  hit refinement, back-face rejection, and roughness-dependent filtering. The
  environment map remains the stable off-screen reflection fallback. This is
  not a local reflection-probe system.
- Cinematic and Ultra add bounded diffuse screen-space GI. It samples visible
  neighbouring radiance with normal, range, metallic, and AO rejection, then
  keeps environment IBL as the non-screen fallback. It is a realtime indirect
  bounce approximation, not path-traced or unrestricted multi-bounce GI.
- Shadow filtering uses a rotated Poisson kernel and slope-aware bias instead
  of an axis-aligned 3x3 kernel. The same algorithm is compiled for native and
  browser WebGPU.
- Native and WASM consume the same graph, atmosphere plan, material data, and
  WGSL. Browser defaults to Portable and caps the expensive Cinematic/Ultra
  screen-space loops at 20 SSR steps and 4 GI taps; native caps them at 40 and
  8. This changes implementation quality and cost, not authored semantics or
  requested output dimensions.

These changes improve handling of authored surface detail and bright highlights;
they do not generate rust, dirt, realistic skin, geometry or new illumination.
Raster transmission remains screen-space refraction, while Weaver is the
offline physical-lighting path; matching IOR and attenuation semantics does not
make the two solvers pixel-identical. Atmosphere and lighting share resolved
scene semantics, but sampling, temporal history, and occlusion can still produce
different images. Weaver remains the final-quality reference.
There is no mandatory offline rendering step. Mips are built on texture-cache
misses. Typical square texture mip storage adds about one third; HDR and
MRT G-buffer targets add frame-sized GPU memory, and the final resolve adds one
fullscreen pass even when depth of field is disabled.
Immediate frame rate still depends on device, render size and scene complexity.

## Validation observation

A representative 1280x720 frame was captured before and after on native Metal.
Its predominantly flat toon materials have no decoded detail
textures, so the visual change is small. This is a regression comparison, not
evidence that the renderer can automatically reproduce a textured horror image.

Twelve warmed debug submissions averaged 47.2 ms before and 47.8 ms after in the
initial comparison; a final-build capture averaged 49.9 ms. These are CPU
submission timings, not GPU frame timings or
playback FPS; they do not establish browser or release performance. First-run
shader compilation also incurs startup cost and is excluded from this average.

GPU regression fixtures isolate material AO from direct light and verify that
high emission retains energy through defocus, including partial alpha coverage.
CPU mip tests cover linear-light filtering, transparent colors, normal vectors
and odd image edges. WASM compilation checks the shared code; browser runtime
performance and pixel equivalence require separate testing on target devices.

Validation on 2026-09-12 covers the shared WGSL and Rust renderer contract. The
S90 Cinematic preview is also used as a native Metal smoke test for the MRT and
temporal pipeline.

Reproduce the native GPU fixtures with:

```sh
cargo test -p motionloom --test immediate_materials -- --ignored --test-threads=1
```
