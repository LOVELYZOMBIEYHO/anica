# S89 Weaver acceptance

Assets remain at `motionloom-example/showcase/s-000089`; tests resolve the workspace
from `CARGO_MANIFEST_DIR`, independently of the working directory. They are not
portable asset fixtures for an installed crates.io package.

From `anica`, run a bounded smoke render:

```sh
cargo test -p motionloom --features weaver --lib weaver::tests::s89::s89_gpu_baseline -- --ignored --nocapture
```

Explicitly raise cost, choose the held camera, enable physical mist and provide
your installed native denoiser library if desired:

```sh
WEAVER_TEST_WIDTH=640 WEAVER_TEST_SAMPLES=64 WEAVER_TEST_FRAME=240 \
WEAVER_TEST_VOLUME=1 \
cargo test -p motionloom --features weaver --lib weaver::tests::s89::s89_gpu_baseline -- --ignored --nocapture
```

`WEAVER_DENOISER_LIBRARY` optionally supplies an absolute shared-library path.
`WEAVER_TEST_REGION=x,y,width,height` preserves full-frame ray coordinates while
rendering only the requested crop, useful for numerical regression checks.
Weaver does not search another application's installation or require its renderer.
`WEAVER_TEST_WIDTH=3840` produces native 3840x2160 output, not upscaled output.
The test fixes min/max samples to `WEAVER_TEST_SAMPLES` for timing comparisons;
use the typed Ultra preset in the host API for adaptive 256..4096 samples.

Additional controlled-experiment overrides:

- `WEAVER_TEST_HEIGHT` requests an exact height; for example width 480 and height
  272. If omitted, height remains width * 9 / 16 for existing benchmark commands.
- `WEAVER_TEST_BOUNCES=1` isolates camera-visible emission and direct lighting;
  compare against `16` with every other input fixed and volume disabled.
- `WEAVER_TEST_BATCH` changes dispatch size, not the per-pixel random sequence.
- `WEAVER_TEST_FSTOP` and `WEAVER_TEST_FOCUS` override physical lens settings.
- `WEAVER_TEST_CALIBRATED=1` tests environment 0.65, sun 2.5, sky fill 0,
  sun diameter 1.5 degrees, f/1.4 and focus 16.2; source scene is unchanged.
- The ignored `compare_linear_renders` test accepts absolute directories through
  `WEAVER_COMPARE_A`, `WEAVER_COMPARE_B`, `WEAVER_COMPARE_OUTPUT`. It writes a
  side-by-side PNG, amplified difference, linear RGB metrics and estimated
  standard errors from the raw variance/sample-count passes.

Output: workspace `.render-output/weaver-s89/<job-hash>/`.
Tests assert dimensions, sample budgets, finite radiance and non-black output.
High-sample and 4K renders require an explicit invocation and are not part of
ordinary `cargo test` runs.

## Numerical and resume regression checks

```sh
cargo test -p motionloom --features weaver --lib weaver::tests::s89::environment_pole_remains_finite -- --ignored --nocapture
cargo test -p motionloom --features weaver --lib weaver::tests::s89::cancelled_job_resumes_without_changing_samples -- --ignored --nocapture
```

The pole fixture renders pixel `(748,505)` of the 4K held camera at 64 samples.
The original failure occurred at sample 6: rounding produced a polar environment
direction with zero horizontal components, making `atan2(0,0)` undefined on Metal.
Longitude now has a defined pole fallback. Invalid samples still fail the job;
they are not silently discarded or replaced with black.

The GPU physics test also compares a crop with its exact full-frame pixel and
compares different dispatch batch sizes. Resume checks compare checkpoint bytes
against a fresh render, not just visually similar PNGs.

## Recorded baseline

- Apple M2, 219,607 triangles.
- 320x180, 8 spp: initial GPU bridge/kernel smoke test passed.
- 640x360, 64 spp, 16-bounce budget: raw EXR and auxiliary-guided denoising passed;
  approximately 116 seconds including scene preparation/output on this machine.
- This is a low-sample engineering baseline, not final Ultra convergence.
- Final implementation changes need a new recorded verification below; previous
  baseline timings are not promises about subsequent kernels or other hardware.

## Native 4K baseline (2026-09-11)

- Job `9fe9e0d538d94214`: 3840x2160, frame 240, 16 spp, 16 total bounces,
  physical bounded mist, thin-lens DoF, host-provided HDR denoising.
- Apple M2; report elapsed time 902.06 seconds, including output and denoising.
- All 510 tiles completed. Report: 514 converged pixels and 8,293,886 pixels at
  the sample limit. Status is correctly `sample_limit_reached`, not `converged`.
- Raw and denoised PNGs visually inspected: floor joints, column highlights and
  railing shadows are present. Raw dark regions/mist are noisy; denoising smooths
  distant and dark detail. Low-poly mountain/flower silhouettes remain visible.
- This verifies native 4K execution/output, not 4096-spp Ultra convergence or
  production-renderer parity. Next visual gate is higher-sample crop convergence
  before committing to an expensive full-frame Ultra run.
- The subsequent output-only optimization writes denoised beauty/PNG without
  duplicating auxiliary EXRs or cloning the full film. Original baseline files
  are retained; the sampler is unchanged.
