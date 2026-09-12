# S89 appearance validation — 2026-09-11

This experiment separates renderer correctness from art direction and output
resolution. The original S89 geometry, materials, camera animation and preview
DSL are unchanged. Paths below are relative to workspace
`.render-output/weaver-s89/`.

## 1. Direct versus indirect illumination

Same held camera (frame 240), 640x360, 64 samples, f/4, original lighting;
no volume and no denoiser. Only path budgets differ.

| Render | Directory | Mean linear RGB |
|---|---|---:|
| One scattering level / direct | `72bd620371c2ddec` | 0.175745 |
| 16 scattering levels | `f81f052f607691d2` | 0.197630 |

Mean RGB increases approximately 12.45%. Column bases, vase backs and floor
shadows receive additional light. This is evidence of indirect contribution,
not proof that every BSDF is physically calibrated. The multi-bounce image has
higher variance. Comparison images/metrics are under
`validation/direct-vs-indirect/` (A/direct left, B/multi-bounce right).

## 2. Native 4K crops, without denoising

Each crop is 128x128 pixels from the full 3840x2160 camera, not an enlarged
low-resolution render. Same seed, f/4, original lighting, no volume, 16-level
budget. Fixed 64 versus 256 samples.

| Crop origin | Region | 64 spp RMS standard error | 256 spp RMS standard error |
|---|---|---:|---:|
| (540,1390) | Vase | 0.02302 | 0.01069 |
| (970,1410) | Column base | 0.01921 | 0.00961 |
| (1850,1750) | Floor joint | 0.03221 | 0.01651 |

The statistic is sqrt(mean(sample luminance variance / sample count)), in linear
radiance units. It estimates uncertainty, not error against a ground-truth
renderer. All three decrease approximately twofold when samples quadruple;
mean RGB changes less than 0.2%. Residual speckles remain, and almost all pixels
still reach the strict 0.003 noise-threshold sample cap.

| Region | 64 spp directory | 256 spp directory |
|---|---|---|
| Vase | `45d969a0dfc7117c` | `bc03a686890e4d45` |
| Column | `aa7bd1f9f00ac163` | `dbc0602caa139c4e` |
| Floor | `ebeb20b4675001a1` | `d0e3f50330a6e2d8` |

Comparison outputs are under `validation/{vase,column,floor}-64-vs-256/`.
Visual inspection confirms that stone joints, wood detail and vase reflections
exist in raw output. Denoising is not responsible for creating those details.

## 3. Offline light and lens calibration candidate

`d3ed2675c806a0d6`: 640x360, 128 spp, no denoising, same framing.

- Preserve the 57-degree vertical FOV: 18.65 mm at 36 mm sensor width, 16:9.
- One scene unit is provisionally treated as one meter, not a certified asset
  measurement. Focus remains 16.2 units; f/4 becomes f/1.4.
- Environment lighting 0.1 becomes 0.65, matching visible background intensity.
- Sun intensity 3.75 becomes 2.5; artificial directional sky fill 0.45 becomes 0.
- Sun angular diameter 0.53 becomes 1.5 degrees for softer penumbrae. This is an
  artistic extended source, not a claim about the real Sun's angular diameter.
- No mist during this comparison; geometry and original PBR maps are unchanged.

Backlit surfaces are more readable and shadows less severe, but low-poly
silhouettes and noisy indirect reflections remain. A wide-angle physical lens
does not imply portrait-like background blur. The PNG environment is still LDR.

## 4. Full-frame output

The native 3840x2160/128-spp run (`011693b8b877feff`) was stopped at the user's
request after 63/510 completed tiles because of its runtime. Checkpoints are
retained. The user requested exact 480x272 instead; the replacement retains the
calibrated lighting/lens, 128 samples and 16-level path budget, with raw and
separately denoised output. This is a quality-validation candidate, not a claim
of Ultra convergence or production-renderer parity. Record its completed report
and visual inspection here before acceptance.

Supporting regression checks pass: optional lighting JSON compatibility and
parameter validation; GPU Lambertian energy, dispatch/crop invariance; and an
S89 single-dispatch fixture confirming all 16,384 crop pixels receive exactly
four samples. The render tests verify finite radiance and sample budgets.

## Remaining acceptance work

- Reduce indirect/reflection variance and validate material energy at grazing
  angles; increasing samples alone is expensive.
- Calibrate against an independently rendered reference with matched inputs.
- Replace simplified asset silhouettes only in a separately authorized asset pass.
- Add missing transmission, complete light types and ray-footprint filtering.
- Validate final-frame convergence and temporal stability independently of image size.
