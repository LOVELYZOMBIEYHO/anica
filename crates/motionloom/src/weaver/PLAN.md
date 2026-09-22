# Weaver implementation and acceptance plan

All Weaver-specific code, documentation, configurations and tests live under
`src/weaver/`. The only external integration is Cargo feature/dependency wiring,
module/API registration, and access to the existing frame-lowering bridge.

## SceneCompositor phase status

1. **Formal contract — implemented.** `SceneCompositionPlan` remains the
   versioned, serializable, renderer-independent IR and fixes the working format
   to linear-premultiplied RGBA16F.
2. **RGBA8 plate removal — implemented.** Mixed snapshots retain independent
   evaluated 2D runs; no `display_plate` or saved-PNG overlay path remains.
3. **Raster boundary — implemented.** Raster straight-alpha sRGB output is
   decoded and premultiplied once when each run enters the compositor. It is not
   flattened or quantized again before final encoding.
4. **Shared GPU execution — implemented.** Weaver and SceneCompositor use the
   same `SceneGpuContext`, device and queue. Canonical layers upload as RGBA16F
   textures and source-over executes on that device. Weaver's tiled film remains
   CPU-readable because EXR/AOV/checkpoint writers require it.
5. **Ordered composition — implemented for the accepted mixed layout.** The 3D
   beauty is the base and evaluated image-plane runs retain authored order.
   Camera-compatible 3D islands currently merge into one physical trace.
6. **Colour stages — implemented.** Scene-linear layers execute before the
   display transform; Screen/Lens runs execute in display-linear space after it.
7. **Output split — implemented.** `beauty.exr` remains pure Weaver 3D;
   `scene-composite.exr` preserves pre-display HDR; `display-master.exr` retains
   the complete post-display RGBA16F result; and `display.png` is encoded once
   from that master. Denoised output follows the same contract while root AOV
   files remain untouched.
8. **S74 acceptance — implemented on Apple M2.** Frame 0 at authored 1920x1080
   completed through the RGBA16F path. The hard lower-screen rectangle was
   migrated to a feathered ground-mist gradient so it no longer creates an
   opaque-looking horizontal obstruction.

## Master sequence and video phase status

9. **Master profile — implemented.** `MasterSequenceSettings` fixes the source
   to compositor-complete RGBA16F display masters and an explicit SDR BT.709
   delivery target. HDR is not exposed until a validated luminance transform
   exists.
10. **Resumable scheduler — implemented.** Inclusive frame ranges render with
    bounded one-frame memory, copy masters atomically, and skip only frames whose
    source/job signature and expected EXRs still match.
11. **Sequence manifest — implemented.** `sequence-manifest.json` records source
    and sequence hashes, rational FPS, resolution, color/alpha contract, every
    frame, output paths, state, and the final probe evidence.
12. **Professional video master — implemented.** The display-master EXRs encode
    to BT.709 ProRes 4444 XQ with retained alpha and 24-bit PCM when audio exists.
13. **Audio and review output — implemented.** Authored audio is mixed to a
    48 kHz stereo float WAV, trimmed to the selected frame range, then muxed as
    PCM in the master and AAC in the H.264 review MP4.
14. **Acceptance — implemented.** The exporter checks first/middle/last EXRs and
    uses `ffprobe` to enforce frame count, resolution, codec/profile, alpha pixel
    format, BT.709 tags, and 48 kHz stereo audio. It also decodes a middle frame
    and rejects effectively black encoder output. S74 frame 0 completed on M2
    and a second run resumed the EXR without retracing.

Coverage, depth, normal and albedo AOVs are populated. Sequence renders also
populate camera motion in output-pixel units from the previous physical camera.
Per-object deformation motion and independent 3D-island textures remain future
work rather than being presented as completed behaviour.

## Sequence performance phase status

15. **Phase profiling — implemented.** Every frame report and sequence manifest
    records parse, scene evaluation, geometry/BVH, GPU setup, path trace,
    composition/output and denoise timings, plus the frame-delta classification.
16. **Persistent sequence session — implemented.** Source text and the parsed
    graph remain live across frames and are invalidated by path, size or modified
    time. The second S74 frame measured 0.00009 s parser time.
17. **Static GPU cache — implemented.** One device, queue and path-tracing
    pipeline remain resident. Exact scenes reuse buffers; animated scenes update
    existing allocations when capacity permits.
18. **Frame deltas — implemented.** Reports distinguish first frame,
    camera/uniform-only, resident scene-buffer update, full scene rebuild and
    pure-2D work.
19. **BVH reuse/refit — implemented.** Stable triangle order retains SAH tree
    topology and refits bounds for animated geometry. Topology/count changes
    safely fall back to a full build.
20. **GPU batching/residency — implemented.** Active tiles dispatch in one pass
    and read back through one collective submit/wait per sampling round. Static
    textures are not uploaded again when unchanged.
21. **Motion and temporal denoise — implemented with an explicit boundary.**
    Camera motion vectors feed an opt-in history pass with normal/albedo rejection
    and neighbourhood clamping. Independent denoise remains the default;
    `--temporal-denoise` selects temporal mode. Object/deformation motion is not
    fabricated and temporal ranges rerender until history sidecars exist.
22. **Acceptance and bounded storage — implemented.** Completed canonical EXRs
    are atomic; duplicate job AOV/checkpoint directories are removed by default,
    while interrupted frames retain recovery data. `checkpoint_retention=all`
    preserves diagnostic jobs. A two-frame S74 1920x1080 Metal run completed on
    Apple M2 with 24 MiB of retained masters rather than tens of GiB of duplicate
    checkpoints.

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
preview routing is introduced in this implementation. The opt-in
`weaver::preview::PreviewSession` API and its `weaver_preview` example host are
the only additions; they reuse the existing lowering and leave `render` and its
checkpoints unchanged.
