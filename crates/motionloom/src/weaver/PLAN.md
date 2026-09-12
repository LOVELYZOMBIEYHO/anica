# Weaver implementation and acceptance plan

All Weaver-specific code, documentation, configurations and tests live under
`src/weaver/`. The only external integration is Cargo feature/dependency wiring,
module/API registration, and access to the existing frame-lowering bridge.

## Architecture now on disk

```text
weaver/
  api.rs                  public job/progress/cancellation API
  config/                 typed job, presets, validation
  scene/                  evaluated scene bridge and snapshot
  geometry/               world-space mesh packing and BVH
  camera/                 frame-matched camera and physical lens
  lighting/               HDR environment and importance distribution
  backend/wgpu/shaders/   path integration, BSDF, visibility, medium sampling
  jobs/                   preparation, GPU scheduling, checkpoint/resume
  color/                  display transform and universal grading
  output/                 raw HDR and auxiliary/display output
  denoise/                optional native auxiliary-guided denoising
  tests/                  CPU/GPU physics and S89 acceptance fixtures
```

Shader BSDF/integrator/volume helpers currently share one kernel source to keep
their buffer contract consistent. Split shader sources into modules once their
sampling interfaces stabilize; do not create empty placeholder folders.

## Acceptance gates

1. **Scene bridge and basic tracer:** implemented and S89 baseline rendered.
   Reuses 219,607 evaluated S89 triangles and original embedded textures.
2. **Sampling correctness:** parameter validation, HDR CDF normalization and
   physical camera tests pass; GPU Lambertian-energy and dispatch-invariance pass.
   Disk cancellation/resume is byte-exact against a fresh render. A reproducible
   4K pixel caught undefined environment longitude at a pole; the guarded
   atan2 mapping passes the same pixel at 64 samples.
3. **S89 appearance:** compare frame 0 and frame 240 with raw and denoised output;
   inspect columns, railing, lantern emission, stone roughness, background and DoF.
   The held-camera direct/indirect comparison and three native-4K 64/256-spp
   crop experiments are recorded in [tests/s89/VALIDATION.md](tests/s89/VALIDATION.md).
   Offline light/lens overrides preserve the immediate-preview source scene.
4. **4K output:** explicit low-sample 4K validation first; then measure convergence
   and time for Ultra. A large image or denoised low-spp image is not Ultra acceptance.
   Native 3840x2160/frame 240/16-spp execution completed in 902.06 seconds on M2,
   with raw and denoised images visually inspected. Nearly all pixels reached
   the sample cap; higher-sample convergence remains open.
5. **Production closure:** transmission/absorption, complete light support,
   normal/ray-footprint filtering, material energy tests and artifact reduction.
6. **Job robustness:** verify cancellation/resume and corruption handling,
   performance profiling, device-specific memory budgeting and frame caches.
7. **Animation:** fixed eight-second S89 camera sequence; inspect temporal stability,
   then add shutter sampling/animated geometry and temporal denoising.

The 4K/4096-spp/16-bounce Ultra preset is an available *budget*, not evidence that
the scene has converged or that the renderer has achieved production parity.
Remaining gates must stay visible in status reports rather than being silently
replaced by a lower-quality run.

## S89 visual controls

Keep the source scene and its eight-second camera animation. Override style with
`courtyard_filmic_physical` in the job so the showcase's two-second style cuts do
not contaminate physical comparisons. Start with the existing PBR maps; improve
geometry or maps only after identifying their limitation in raw/AOV images.

For a fog baseline, opt out explicitly. For physical mist, specify a bounded
medium in the job and tune extinction/albedo/anisotropy against the scene scale.
Keep HDR radiance and camera exposure separate from final display grading.

No CLI binary, separate crate, HTTP service, new DSL tags or automatic browser
preview routing is introduced in this implementation.
