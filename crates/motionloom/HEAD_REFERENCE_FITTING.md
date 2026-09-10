# Head reference fitting (authoring API 1.0)

Import `motionloom::api::head_fitting`. This opt-in, filesystem-free module
accepts annotations, not images or natural-language instructions. The existing
HeadAsset DSL is extended in place with semantic face measurements; the fitted
ordinary DSL remains the complete playback artifact.

```rust,ignore
let request: HeadReferenceSet = serde_json::from_str(json)?;
let validation = validate_head_reference_set(&request);
let before = evaluate_head_reference_fit(source, &request)?;
let proposal = fit_head_asset_to_references(source, &request, || cancel_flag())?;
let next_source = apply_head_fit_proposal(source, &proposal)?;
```

The four functions have `_json` wrappers with typed `HeadFitError` failures.
WASM exports `validateHeadReferenceSet`, `evaluateHeadReferenceFit`,
`fitHeadAssetToReferences`, `applyHeadFitProposal`. JSON fitting is synchronous;
run it in a worker for responsiveness. Rust cancellation returns the best completed
candidate. No external files are read by the core; `imageId` is an opaque host ID.
Reports contain `schemaVersion` and `measurement`. Rejections use typed errors;
reference validation includes structured diagnostics.

## Coordinates and contract

See `examples/head_reference_fit/synthetic.json` for the full request and
`HeadReferenceSet` / `FitOptions` for the serde contract. Unknown JSON fields,
unknown views and unsupported projections fail instead of guessing orientation.
Pixels use original image dimensions, origin top-left, Y down. Omit invisible or
unknown landmarks; `[0,0]` is a real image corner, not a missing-value sentinel.
The centerline is directed from top toward chin. Its first point sets horizontal
zero; rotation is removed using this axis. Top/chin separation along the axis
sets head height. Image translation, uniform scale and in-plane roll cancel.
All views must share the same anatomical vertical axis and head-height baseline.
Pitch, yaw, perspective, artistic distortions and hair must be corrected or
excluded by the host before submission.

The model faces +Z; +X is anatomical left; +Y is up.

| View | Camera position | Normalized horizontal axis |
|---|---|---|
| front | +Z, looking toward -Z | +X (viewer right = anatomical left) |
| left | +X, looking toward -X | -Z (nose points viewer left) |
| right | -X, looking toward +X | +Z (nose points viewer right) |
| back | -Z, looking toward +Z | -X |

Images are never implicitly flipped. `left` / `right` always mean anatomical
sides. Geometry symmetry does not merge side observations. `symmetry: "x"`
rejects unmirrored off-axis features; `"none"` permits existing asymmetry.

Optional annotation fields default to `confidence: 1`, `weight: 1`,
`enabled: true`; confidence is 0..1, weight is 0..100. Zero-weight/disabled
observations do not affect fit or consistency. Active contours must be simple,
closed `head_outline` polygons with 3..2048 points and nonzero area, plus
open semantic curves for eyelids, iris, nose and lips. The request accepts up
to 32 contours per view, at most 8 views, 128 landmarks/view, 256 mesh
segments/rings and 200 iterations. Annotations outside the analysis window are
rejected, not clipped. Common landmark height disagreement above 0.06 head
height blocks fitting. Missing side views produce a warning.

## Measurements and search

Mesh generation is the existing runtime generator, including HeadMorph and
modifiers. The CPU rasterizer takes the union of projected triangles, preserving
concavities; it does not use a projected vertex cloud as a silhouette. Masks use
128×128 samples over [-1,1]² (cell width 0.015625 head height). Bidirectional
mean nearest-boundary distance and mask IoU are reproducible approximations.
Contour error in pixels equals normalized error × aligned reference height.
Landmarks return normalized endpoints and pixel/normalized residuals.

`head_top` and `chin` use mesh vertical extrema. `nose_tip` uses an initial nasal
region surface extremum and tracks that vertex through the search; it is labelled
**low confidence**, weighted by 0.35 and unavailable in back view. It is not the
HeadFeature center and is not guaranteed to be the anatomical tip. With
FaceLayout, eye corners, iris centers, nose base, mouth corners and lip centers
are measured from the same face coordinate system. Semantic curves are compared
with bidirectional point-to-curve distance, so complete eyelid shape, iris
placement and lip arcs constrain the fit. A fresh evaluation initializes
correspondence from its own source; a proposal compares both states with the
same original correspondence.

Fixed-order bounded coordinate descent tunes width/depth (relative to authored
HeadShape height), forehead/cheek/jaw/chin, then FaceLayout eye height,
spacing, aperture width/opening/tilt, optional iris radius/pupil radius, nose
height/projection/length, mouth/lip dimensions, and facial-cage socket width, height,
and depth. `allowedParameters` and
`lockedParameters` use names such as `Eye.opening`; locks win.
Explicit HeadFeature fields and HeadMorph remain unchanged by fitting. A
HeadFeature can additionally carry an `offset={[x,y,z]}` vector for a smooth
local cage deformation.
The parameter bounds are the `PARAMETERS` table in the module. HeadShape.size Y
and HeadMorph.headHeight remain unchanged; default `preserveHeight` additionally
rejects measured surface-height drift above 0.5%. Setting it false only relaxes
this drift check; it does not add a global scale parameter.

The data objective is a confidence/weight-normalized sum of contour distance +
0.1×(1−IoU) and landmark distances, plus a penalty for mesh edges longer than
0.3 head height. Search adds 0.002×squared parameter displacement from source.
It rejects collapsed heads and analysis-window overflow. This is a basic quality
guard, **not** a complete self-intersection or anatomical quality detector.
The report's objective is the data-plus-geometry measure; parameter regularization
is only used for candidate selection. Search never accepts worse data objective
than the initial source. Stop reasons distinguish cancellation, iteration limit,
convergence, no improvement and encountering parameter boundaries.

Errors/stagnation do not establish that the shape model is incapable. Examine
annotation agreement, raster precision, restricted/locked parameters, unsupported
landmarks and held-out views before diagnosing representational limits. The
generated head remains a CEL-ready head mesh with attached component-local
sclera, optional iris/pupil, eyeliner, and texture geometry. Semantic curves are
fitting constraints and relief controls. Hair, teeth, and texture synthesis remain
outside this API.

`topology="facialCage"` uses these semantic edits to regenerate the welded cage.
`topology="explicit"` is comparison-only: fitting returns a typed error rather
than rewriting arbitrary `Vertex` positions.

## Reviewed output and CLI

Source fingerprints use SHA-256 over exact UTF-8 bytes. Changes carry original and
new numeric values. The editor patches only unique target child attribute spans,
skipping comments/quoted text; existing Action-only source editors cannot locate
HeadAsset nodes, so a dedicated scanner is used. Text outside the target is
unchanged. Apply verifies fingerprint, replays allowed changes, checks before/after
values and candidate text equality, then reparses, compiles and authoring-validates.
Input scenes with existing authoring errors must be repaired first.

From `anica`:

```sh
cargo run -p motionloom --example fit_head_references -- \
  crates/motionloom/examples/head_reference_fit/baseline.motionloom \
  crates/motionloom/examples/head_reference_fit/synthetic.json \
  /tmp/head-fit-synthetic --gpu
```

Outputs: validation/report JSON, playable candidate DSL, CPU comparison SVG,
and four separate review DSL scenes. `--gpu` additionally renders front, left,
back and held-out three-quarter PNGs with the existing GPU renderer. Review
scenes use fixed clay lighting and perspective cameras for visual inspection;
they are distinct from orthographic CPU fitting measurements. The host can draw
its own original images under the normalized overlay and control opacity without
putting references into candidate DSL.

## Supplied turnaround

`examples/head_reference_fit/user-turnaround.json` records conservative manual
annotations from the 1448×1086 user image. The left-facing profile is provisionally
classified anatomical left under the convention above. Nose observations have
confidence 0.35; alignment top/head-height and side centerline are estimates because
hair obscures the cranium. Back has no visible supported facial landmark. Hair
outlines are deliberately omitted. These sparse annotations exercise fitting but
cannot constrain a full head, validate likeness or establish true orthography.
Do not interpret an improved nose residual as recreation of this character.

Use the existing S86 script as input and write the candidate to a separate folder:

```sh
cargo run -p motionloom --example fit_head_references -- \
  ../motionloom-example/showcase/s-000086/main.motionloom \
  crates/motionloom/examples/head_reference_fit/user-turnaround.json \
  /tmp/head-fit-turnaround --gpu
```


## Verification (2026-09-05)

- `cargo test -p motionloom --lib head_ --offline`: 13 passed, including 11 new
  authoring/projection tests and the existing head parser/mesh tests.
- Native and actual Node WASM execution: synthetic objective
  `0.06293539185108006 → 0.0005656687024256341` on both platforms, with a
  comparison tolerance of `2e-6`. `check-wasm.cjs` also checks stale-source rejection.
- GPU review uses the existing renderer and requires host Metal/WebGPU access.
  A sandbox without a GPU adapter correctly returns a GPU error; no CPU image is
  substituted as an official render.

To reproduce cross-platform parity after generating the native CLI report:

```sh
CARGO_INCREMENTAL=0 cargo rustc -p motionloom --lib \
  --target wasm32-unknown-unknown --offline -- -C debuginfo=0 -C opt-level=1
wasm-bindgen target/wasm32-unknown-unknown/debug/motionloom.wasm \
  --target nodejs --out-dir /tmp/motionloom-head-fit-wasm
node crates/motionloom/examples/head_reference_fit/check-wasm.cjs \
  /tmp/motionloom-head-fit-wasm/motionloom.js \
  crates/motionloom/examples/head_reference_fit/baseline.motionloom \
  crates/motionloom/examples/head_reference_fit/synthetic.json \
  /tmp/head-fit-synthetic/report.json
```

Views without measurable observations report `error: null`, not a perfect zero.
Nasal cage tracking is disabled when modifiers/topology prevent reliable vertex
correspondence. `symmetry: "x"` fitting rejects modifiers whose symmetry cannot
be established; evaluation still measures their generated geometry. This first
version does not diagnose every possible cross-view contradiction: it checks
common landmark heights and front/back silhouette widths, while subtler conflicts
remain visible in per-view residuals. No whole-body, rig, texture, new modeling DSL
or unknown-perspective solver is included.

The checked-in `examples/head_reference_fit/output-synthetic` and
`output-turnaround` folders contain reproducible CLI outputs and all four GPU
views. The supplied-image run improves its sparse landmark objective from
`0.14369752753111517` to `0.138843759533827` (~3.4%), stopping at its 12-iteration
limit. Native/WASM results agree within `2e-6`; the original S86 fingerprint
still matches. Visual inspection of the held-out three-quarter view shows an
excessively pointed nasal projection, shallow eye/mouth relief and no represented
hair. It does not resemble the illustrated character sufficiently. These are
observed limitations of this candidate and its sparse constraints, not proof that
all possible HeadAsset parameters fail. Additional reliable surface annotations,
more expressive geometry and an improved nasal correspondence require separate
 assessment before claiming character reconstruction.

## Semantic face constraints (schema 1.0 extension)

The same HeadAsset API now accepts more than a silhouette. A view may contain
one closed head_outline plus open semantic curves for upper/lower eyelids,
iris_left/right, nose_profile, upper_lip and lower_lip. Curves are scored with
bidirectional point-to-curve distance and are returned in ViewFit.curves.

Eye components expose width, opening, and tilt. Optional Iris children expose
position, shape, geometry scale, radius, and pupilRadius. The bounded fitter edits
radius and pupilRadius; it preserves authored Iris position, shape, and scale.
Nose.position controls the nose independently; Mouth owns opening,
upperLip, and lowerLip. These parameters are used by
the analytic candidates and are available to the bounded fitter through
allowedParameters and lockedParameters. Supported semantic landmarks include
eye inner/outer corners, iris centers, nose_base, mouth corners and lip centers.

HeadFeature also accepts an optional offset vector. It applies a smooth local
vector displacement to the continuous head cage, which lets an author correct
forehead, cheek, jaw and other face planes without attaching disconnected
geometry. The renderer still produces the same CEL-ready continuous surface.

The API keeps schemaVersion 1.0 during development. Flat FaceLayout attributes
and the removed flat Eye iris/lid fields are breaking parser changes.
