# Immediate preview: material foundation

## Host quality contract

`ImmediatePreviewSettings` separates editor performance from authored
`RenderStyle` and offline output. Hosts can select `Portable`, `Balanced`, or
`Cinematic`; each profile maps to concrete shadow-map, texture-filtering,
light-count, DoF-sampling, HDR, and antialiasing budgets. The parsed graph is
never rewritten.

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

`WgpuPreviewEngine::capabilities()` reports the actual immediate features.
`last_frame_metrics()` combines GPU timing,
Scene CPU timing, triangle/light counts, retained-cache counts, shadow-map
resolution, and estimated render-target memory.

The quality degradation order is resolution (respecting the configured floor),
optical samples, then profile-controlled shadow and filtering cost. Authored
materials, animation, camera, colors, and style identity remain unchanged.

The improvements are renderer internals shared by native and WASM WebGPU.
Existing MotionLoom scripts need no additional tags or quality settings.

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
  Exposure, white balance and contrast remain before blending. The public Scene
  compositor still receives RGBA8; Scene Process bloom is not HDR bloom.
- The primary opaque geometry pass uses MRT to retain material normal, velocity,
  roughness, metallic, AO and a temporal reactive mask while it writes shaded
  HDR. Per-object motion comes from previous model transforms and previous bone
  palettes. This removes the former extra geometry submission and fullscreen
  normal-reconstruction pass.
- Temporal resolve reprojects the prior display frame. The default `Balanced`
  profiles use a stable projection, because moving an alpha-tested grass sample
  every frame is both visually distracting and unnecessary editor work. The
  eight-sample Halton implementation remains internal but is not enabled until
  a coverage-aware resolve can guarantee stable thin geometry. History is
  rejected after a non-sequential seek, camera cut, size change or
  preview-profile change. Motion vectors use
  interpolated current and previous clip positions instead of a quantized
  fragment pixel centre; history confidence and motion blur use de-jittered
  physical velocity.
- Material-aware SSR ray marches resolved depth using the G-buffer normal and
  attenuates reflection by roughness and metallic response. It is reserved for
  `Cinematic`; the default `Balanced` profile keeps TAA and velocity motion blur
  but omits this relatively expensive ray march.

These changes improve handling of authored surface detail and bright highlights;
they do not generate rust, dirt, realistic skin, geometry or new illumination.
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
