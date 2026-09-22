# MotionLoom Weaver

Weaver is MotionLoom's opt-in native offline path tracer. It lives entirely in
`src/weaver/`; the crate's `weaver` feature registers its API and the scene bridge.
There is no new crate, CLI executable, DSL syntax, or browser preview dependency.

This is a working initial renderer, **not a claim of production/Cycles parity**.
See [PLAN.md](PLAN.md) for the remaining acceptance gates.

## Rust API

Enable `motionloom`'s `weaver` Cargo feature. The recommended entry point is
`motionloom::api::weaver`. The host supplies an executor, output location, and
optional native denoiser. The synchronous GPU batches inside the async job should
run on a dedicated render worker, not on a UI event loop.

```rust,no_run
use motionloom::api::weaver::{
    render, CancellationToken, QualityPreset, RenderJob,
};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let mut job = RenderJob::new("scene.motionloom", QualityPreset::Ultra);
job.scene_id = "main_scene".into();
job.render_style = "filmic_scene".into();
job.frame = 240; // Evaluated 8-second camera; style override wins over style cuts.
job.output = ".render-output/ultra".into();
// Optional: a trusted host-installed Open Image Denoise shared library.
// job.denoiser_library = Some("/absolute/path/to/library".into());
let cancel = CancellationToken::default();
let report = render(&job, &cancel, |p| {
    println!("tile {}/{}", p.completed_tiles, p.total_tiles);
}).await?;
println!("{}: {}", report.status, report.output.display());
# Ok(())
# }
```

`RenderJob::new` resolves a quality preset; it does not choose a scene ID or
RenderStyle for a generic caller. Set `job.scene_id = "auto"` (or leave it empty)
to let Weaver pick the first scene that owns a 3D composite, and
`job.render_style = "auto"` to pick the first RenderStyle Weaver can represent;
when none is representable, `auto` lowers neutral physical defaults instead of
rejecting the document. An explicit id still wins and is validated as before.
Modify typed fields afterward. Serde can store
the fully resolved job; unknown JSON fields and unsupported versions are rejected.
Relative assets resolve against the source document directory.

Offline lighting can be calibrated independently of the immediate preview:

```rust,ignore
job.lighting.environment_intensity = Some(0.65);
job.lighting.light_intensities.insert("sun".into(), 2.5);
job.lighting.light_intensities.insert("sky_fill".into(), 0.0);
job.lighting.exposure = Some(0.97);
job.sun_angular_diameter_degrees = 1.5;
job.lens.f_stop = 1.4;
```

These are explicit job overrides, not a new DSL or universal quality guarantee.
Light IDs must exist. Omitted overrides preserve authored values. A wider sun
softens penumbrae; it does not increase the total directional-light energy.
The physical lens conversion depends on the authored camera FOV and sensor width.
Even a low f-stop remains wide-angle when the scene uses a wide FOV; focus and
scene scale must be considered together.

## Progressive preview

`weaver::preview::PreviewSession` keeps the evaluated scene, GPU geometry and
tile films alive so a host can accumulate samples and display the current image.
It reuses the same lowering, camera and display transforms as `render`, so a
preview differs from a final frame only by resolution and sample count. The final
render path and its checkpoints are unchanged.

```sh
cargo run -p motionloom --release --features weaver --example weaver_preview -- \
  ../motionloom-example/showcase/s-000096/main.motionloom --frame 0 --size 640x360 --samples 2
```

Both `weaver_preview` and `weaver_frame` default scene and style to `auto`, so a
plain document works without authored ids; override with `--scene-id` / `--style`.

Controls: Esc quit, Space pause, R reset, D toggle denoise, Left/Right change
frame, Up/Down change samples per update. Debug knobs: `WEAVER_PREVIEW_DEBUG`,
`WEAVER_READ_DEBUG` and `WEAVER_PREVIEW_BOUNCES`.

Denoising is on by default and runs entirely on the GPU: a WGSL a-trous wavelet
filter guided by the film AOVs (albedo, normal, depth, variance). `PreviewSession::denoise_rgba`
and the final `denoised/` outputs use the same pass, so no host denoiser library
is required. A host-provided OIDN library remains an optional override through
`job.denoiser_library`.

Sessions clamp paths to a look-development budget (total 8, diffuse 4, glossy 6)
by default; final render jobs keep the authored budget. `PreviewSession::set_bounce_budget`
or `WEAVER_PREVIEW_BOUNCES` restores parity when needed, and changing the
estimator resets the accumulated samples.

Preview update cost is dominated by the path integrator (~31 µs per sample-pixel
at the authored 16-bounce budget on an M2 after the binned SAH build), so a small
progressive first image is expected to take seconds. Kernel efficiency and cache
reuse are tracked as remaining work, not assumed.

## One-off frame

```sh
cargo run -p motionloom --release --features weaver --example weaver_frame -- \
  path/to/main.motionloom --frame 486 --size 640x360 --samples 32 --out .render-output/frame
```

Use `--composite-scene` to request the authored 2D+3D frame instead of the
default 3D-only beauty. A scene with no 3D island takes the existing strict GPU
raster path. The current mixed path traces camera-compatible 3D islands and
retains authored 2D runs independently. Raster runs enter the shared-device
compositor as linear-premultiplied RGBA16F textures; scene-linear work runs
before the display transform and Screen/Lens work runs in display-linear space
after it. It rejects 2D below or between 3D islands until ordered per-island
textures are available, rather than exporting a frame with incorrect order.

Transmission remains strict by default. `--transmission-stopgap` is an explicit
opaque/alpha PBR fallback for documents that need migration time; it is not
physical glass or refraction. Composite jobs additionally emit `coverage.exr`
and `motion.exr`. Coverage is populated; sequence motion contains camera motion
in output-pixel units. Object/deformation motion is not yet represented.

Flags: `--scene-id`, `--style` (both default `auto`), `--frame`, `--size WxH`,
`--samples` (omit for Adaptive Ultra), `--out`, `--f-stop`, `--focus`. Without
`--out`, renders land in the workspace-root `.render-output/weaver` (found by
walking up for the `anica`/`motionloom-example` marker), never inside `anica/`.

## Quality presets

| Setting | Production | Ultra | Reference |
|---|---:|---:|---:|
| Minimum samples | 128 | 256 | 512 |
| Maximum samples | 1024 | 4096 | 16384 |
| Relative noise threshold | 0.01 | 0.003 | 0.001 |
| Total scattering depth | 12 | 16 | 24 |
| Diffuse depth | 6 | 8 | 12 |
| Glossy depth | 8 | 12 | 16 |
| Russian roulette begins | 5 | 6 | 8 |

Defaults: 3840x2160; f/4; 36 mm sensor width; focus distance 16.2 scene units;
9 aperture blades; 0.53-degree directional-light angular diameter. Lens dimensions
assume one scene unit is one meter. Sun diameter applies to directional lights.
World lights keep existing scene intensity units, not certified photometric units.

The estimator uses luminance standard error with a 0.01 dark-pixel floor. It is
Weaver-specific and is not numerically interchangeable with another renderer's
threshold. Zero threshold disables adaptive stopping. A pixel that reaches its
sample cap without convergence is reported as `sample_limit_reached`.

## Implemented path

- Evaluated MotionLoom model transforms, camera keyframes, embedded GLB materials.
- Binned SAH BVH (median splits were measured ~1.5x slower on the S96 landscape
  because scene-spanning sky-dome triangles overlapped both children);
  secondary-ray visibility independent of camera culling.
- Terrain heightfields and procedural vegetation, exported through the same
  offline geometry path as GLB meshes.
- Buffer offsets travel through f32 uniforms as raw u32 bit patterns
  (`bitcast`), so multi-million-triangle scenes keep exact material, light and
  environment addresses past f32's 2^24 integer limit.
- GPU a-trous denoiser (WGSL) driven by the accumulated albedo/normal/depth/
  variance film planes; final frames also write `denoised/` by default.
- Bilinear linearized base-color textures; data normal/metallic/roughness maps.
- Diffuse and GGX reflection; alpha mask/blend; emissive surfaces.
- Directional disks and point lights; emissive-triangle area sampling with MIS.
- Float HDR/EXR environments and sRGB PNG/JPEG environments; solid-angle CDF/MIS.
- Thin-lens DoF and polygonal aperture sampling.
- Bounded homogeneous volume, exponential free flights, HG phase sampling,
  direct/environment illumination, and shadow transmittance.
- Float32 accumulation, per-pixel adaptive sampling and auxiliary passes.
- 128x128 GPU film tiles, deterministic per-pixel/sample random sequences.
- Cancellation between dispatches and resumable film checkpoints.
- Raw EXR and PNG; optional auxiliary-guided HDR denoising through the OIDN C ABI.
- Authored filmic/ACES/Reinhard display transform; universal ColorStyle/ToneStyle.

## Explicit limitations

`CompositeScene` supports pure 2D and mixed scenes whose 2D runs are ordered
above one or more camera-compatible 3D islands. Multiple compatible islands are
currently merged into one physical trace; distinct island textures, 2D below or
between islands, and depth-aware cross-island interleaving are rejected until
the layered executor is complete. Partial group opacity, cel/ink surface
presets, bloom, outlines, orthographic projection, spot lights, and physical
transmission BSDFs are not implemented. Several are rejected with typed errors.
The geometry snapshot API still rejects terrain/vegetation; the Weaver offline
path accepts them. Non-neutral white balance is applied at the display stage.

RectAreaLight is sampled across its authored physical width and height. This
produces broad glossy reflections and convergent penumbrae rather than treating
the source as a point-light brightness multiplier.

Non-neutral surface specular/roughness-bias/saturation overrides are currently
rejected; the physical BSDF uses the imported glTF material values. A physical
override can explicitly use specular=1 and roughnessBias=0. Universal
ColorStyle/ToneStyle grading is supported separately at the display stage.
`render_style = "auto"` sanitizes a non-representable authored style by keeping
its lighting and tone intent and dropping only these surface/outline/post fields.
Preview AO/contact-shadow strengths and shadow-strength hacks are not applied;
physical ray visibility determines occlusion.

`AtmosphereFog` lowers once to `AtmosphereMediumPlan`. Weaver consumes that plan
directly: `density` is extinction per world unit, `scatteringColor` is linear
single-scattering albedo, and `anisotropy`, height falloff, bounds, edge feather,
light shafts, and water caustics retain the authored meaning used by the live
preview. No RenderJob atmosphere override is required.
Normal maps cannot add silhouette detail. Set `job.texture_mips = true` to
build box-filtered mip chains (sRGB levels average in linear space) and select
the level from the ray footprint, which removes minification aliasing at 4K.
With it off, textures keep the base-level bilinear path. Environment background
blur is not implemented.
Shutter motion blur, object/deformation motion vectors, heterogeneous volumes,
caustic-specific sampling and hardware BVH traversal remain future milestones.
Animated sequences retain the parsed graph, refit the SAH BVH, update resident
GPU buffers and can opt into conservative temporal denoising. `transmission` is
reserved in the budget contract; materials using it are rejected unless the
job explicitly enables the documented non-refractive stopgap.

The built-in a-trous denoiser runs on the selected GPU. An optional OIDN path
uses a CPU device and whole-frame buffers and requires a compatible host-provided
library. No other application is discovered, launched or required by Weaver.
Native library initialization occurs only when configured.

## Output and resume

Each job gets a hash subdirectory containing `resolved-job.json`, `report.json`,
`beauty.exr`, `scene-composite.exr`, `display-master.exr`, `display.png`,
`albedo.exr`, `normal.exr`, `depth.exr`,
`variance.exr`, `sample-count.exr`, `coverage.exr`, `motion.exr`, and
`checkpoints/`. `beauty.exr` remains pure 3D; `scene-composite.exr` preserves
scene-linear HDR before Screen/Lens composition; `display-master.exr` is the
complete display-linear RGBA16F result; and `display.png` is encoded once from
that master. The `denoised/` directory contains the equivalent beauty and two
master EXRs plus display PNG. Auxiliary passes remain at the job root.
Denoising does not alter those passes.
Depth is the mean first surface distance, not a deep compositing pass.

Re-run the identical job to resume. The hash covers settings, scene script,
resolved geometry/materials, camera/light uniforms, textures and shader source.
The lock rejects concurrent writers. A process crash can leave `render.lock`;
the host must confirm the job is no longer running before removing that one file.
An ordinary cancellation releases it automatically. Checkpoint writes use a
temporary file and rename; power-loss durability is not guaranteed.

The film/output memory estimate is checked against `memory_budget_mib`. Scene
buffers are additionally checked against GPU storage limits. The budget is a
guard estimate, not a guarantee on total process RSS while assets decode.

`region: Some([x,y,width,height])` renders a crop while preserving the full-frame
camera, pixel coordinates and random sequences. This is useful for reproducing
individual noisy or invalid samples without paying for the entire 4K frame.
`render_sequence` remains the low-level inclusive frame API. For deliverables,
`render_master_sequence` writes canonical `display-master` and optional
`scene-composite` EXR sequences, resumes signature-matched frames, and records a
durable `sequence-manifest.json`. It derives a BT.709 ProRes 4444 XQ master and
H.264 review MP4 from those EXRs, plus a 48 kHz stereo audio master when the DSL
contains audio clips. Final `ffprobe` acceptance checks frame count, dimensions,
profile, alpha pixel format, color tags, and audio layout. Acceleration structures
are not yet cached across animated frames.

Run the native sequence entry point with:

```sh
cargo run --release -p motionloom --features weaver --example weaver_sequence -- \
  scene.motionloom --size 1920x1080 --samples 32 \
  --out .render-output/weaver --transmission-stopgap
```

Omitting `--frames` exports the complete authored timeline. Use an explicit
`--frames START:END` only for a crop, diagnostic render, or encoder test.

## Tests

From the `anica` directory:

```sh
cargo test -p motionloom --features weaver --lib weaver::tests
cargo test -p motionloom --features weaver --lib weaver::tests::physics::gpu_lambertian_energy_and_batch_invariance -- --ignored --nocapture
```

Render tests, cost controls and assets are documented alongside the relevant
test module. High-cost tests are always explicit.
