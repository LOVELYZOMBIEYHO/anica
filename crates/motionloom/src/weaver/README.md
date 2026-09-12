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
    render, CancellationToken, QualityPreset, RenderJob, Volume,
};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let mut job = RenderJob::new("scene.motionloom", QualityPreset::Ultra);
job.scene_id = "main_scene".into();
job.render_style = "filmic_scene".into();
job.frame = 240; // Evaluated 8-second camera; style override wins over style cuts.
job.output = ".render-output/ultra".into();
job.volume = Some(Volume {
    bounds_min: [-200.0, -10.0, -200.0],
    bounds_max: [200.0, 90.0, -16.0],
    extinction: 0.012,
    albedo: [0.9, 0.94, 0.98],
    anisotropy: 0.25,
    max_bounces: 4,
});
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
RenderStyle for a generic caller. Modify typed fields afterward. Serde can store
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
- Median BVH; secondary-ray visibility independent of camera culling.
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

Currently supported scene composition is one active 3D CompositeGroup reached
through Timeline/Track/Sequence/Group. General 2D composition, scene layers,
partial group opacity, cel/ink surface presets, bloom, outlines, non-neutral white
balance, orthographic projection, spot lights, and transmission BSDFs
are not implemented. Several are rejected with typed errors. Terrain/vegetation
remain subject to the shared geometry extractor's capabilities.

RectAreaLight is sampled across its authored physical width and height. This
produces broad glossy reflections and convergent penumbrae rather than treating
the source as a point-light brightness multiplier.

Non-neutral surface specular/roughness-bias/saturation overrides are currently
rejected; the physical BSDF uses the imported glTF material values. A physical
override can explicitly use specular=1 and roughnessBias=0. Universal
ColorStyle/ToneStyle grading is supported separately at the display stage.
Preview AO/contact-shadow strengths and shadow-strength hacks are not applied;
physical ray visibility determines occlusion.

Legacy linear screen fog has no automatic physical conversion: specify `volume`
or explicitly opt into `allow_legacy_fog_omission` for an un-fogged baseline.
Normal maps cannot add silhouette detail. Texture filtering currently has no
ray-footprint mip selection. Environment background blur is not implemented.
Animated geometry/shutter motion blur, temporal denoising, BVH reuse across
frames, heterogeneous volumes, caustic-specific sampling and hardware BVH traversal
remain future milestones. `transmission` is reserved in the budget contract;
materials using it are rejected rather than rendered as opaque plastic.

The optional denoiser currently uses a CPU device and whole-frame buffers; it
requires a compatible host-provided library. No other application is discovered,
launched or required by Weaver. The test environment can explicitly supply an
existing OIDN library. Native library initialization occurs only when configured.

## Output and resume

Each job gets a hash subdirectory containing `resolved-job.json`, `report.json`,
`beauty.exr`, `display.png`, `albedo.exr`, `normal.exr`, `depth.exr`,
`variance.exr`, `sample-count.exr`, and `checkpoints/`. Denoised files are separate.
The `denoised/` directory contains only beauty EXR and display PNG; auxiliary
passes remain at the job root. Denoising does not alter those original passes.
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
`render_sequence` exports an inclusive frame range through the same job API;
each returned report identifies its frame's output directory. It does not yet
cache acceleration structures across frames or encode a video.

## Tests

From the `anica` directory:

```sh
cargo test -p motionloom --features weaver --lib weaver::tests
cargo test -p motionloom --features weaver --lib weaver::tests::physics::gpu_lambertian_energy_and_batch_invariance -- --ignored --nocapture
```

Render tests, cost controls and assets are documented alongside the relevant
test module. High-cost tests are always explicit.
