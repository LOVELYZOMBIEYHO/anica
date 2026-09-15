# MeshAsset image-reference fitting API 1.0

MotionLoom provides a filesystem-free authoring loop for fitting any explicit
`MeshAsset` against one through eight image references. Reference analysis,
evaluation reports and proposals are sidecar data. No reference metadata or
provider-specific instruction enters the DSL; the accepted `.motionloom` source
remains the playback artifact.

The API is intentionally split into three stages:

```rust,ignore
use motionloom::api::mesh_reference::*;

let analysis = analyze_image_reference(image_bytes, &analysis_request)?;
let evaluation = evaluate_mesh_asset_reference(source, &reference_set).await?;
let candidate = apply_mesh_asset_proposal(source, &proposal)?;
```

JSON wrappers share the same computation. WASM exports
`analyzeImageReference`, `evaluateMeshAssetReference` and
`applyMeshAssetProposal`.

## Host boundary

Core accepts image bytes and never fetches a URL. A Codex, Claude, Drive,
browser, desktop or other host resolves its attachment into bytes and assigns an
opaque `imageId`. This keeps credentials, network policy and provider names out
of MotionLoom. Use the SHA-256 `contentHash` in the analysis to prove which bytes
were measured.

Images are limited to 40 million pixels. Coordinates use original image pixels,
top-left origin and Y down. Images are never mirrored implicitly.

## Image analysis

`AnalyzeImageReferenceRequest` supports `alpha`, `backgroundColor`, `guided`,
`mask` and conservative `auto` segmentation. Guided mode accepts an analysis
region plus foreground/background points. Analysis returns:

- a deterministic run-length encoded foreground mask;
- connected regions and bounds;
- ordered outer and hole contours;
- general geometric landmarks (`topmost`, `bottommost`, `leftmost`,
  `rightmost`);
- optional host-authored semantic labels and bindings;
- depth hints and confidence diagnostics.

The built-in analyzer deliberately does not invent semantic object parts or
metric depth from one RGB image. An LLM may label a returned region, bind a
landmark to a mesh vertex/edge/face, or add host depth evidence. Unknown depth
remains unknown.

Native example:

```sh
cargo run -p motionloom --example analyze_image_reference -- \
  reference.png request.json /tmp/reference-analysis
```

It writes `analysis.json`, `mask.png` and `overlay.png`, while stdout stays a
compact JSON summary suitable for an agent tool call.

## Mesh evaluation

`evaluate_mesh_asset_reference` parses the source, resolves a named `Model`
whose asset is a `MeshAsset`, and uses the evaluated runtime transform and
camera at each requested frame. Geometry is projected into the original
reference resolution and rasterized with a depth buffer. Each view reports:

- mask IoU;
- bidirectional mean, p95 and maximum boundary distance in pixels;
- landmark residuals;
- ranked per-control-vertex screen residuals;
- reference/candidate coverage and aggregate error.

Camera policy is `frozen`. Calibrate camera and object placement before fitting
geometry; do not optimize both at once. Reports carry source, topology, camera
and image fingerprints so an LLM cannot safely reuse stale evidence.

Native forms:

```sh
cargo run -p motionloom --example evaluate_mesh_reference -- \
  scene.motionloom references.json /tmp/mesh-evaluation

cargo run -p motionloom --example evaluate_mesh_reference -- \
  scene.motionloom --analysis analysis.json ASSET_ID MODEL_ID 0 \
  /tmp/mesh-evaluation
```

The output directory contains `evaluation.json` plus overlay and difference
PNGs. Overlay colors are white for agreement, green for missing candidate area,
and magenta for candidate overflow.

## Safe proposals

V1 proposals only move existing authored control vertices. They cannot add or
delete vertices/faces, alter UVs, or change topology. This keeps multiview
correspondence and landmark bindings stable.

`api::mesh_authoring` provides a separate `MeshTopologyProposal` path for cases
where the accepted cage is insufficient. It runs versioned GeometryRecipe
operations, validates the complete cage, returns old-to-new correspondence, and
invalidates affected feature bindings before a new baseline is accepted. See
[Mesh authoring API](MESH_AUTHORING.md).

Every proposal contains:

- exact source SHA-256;
- topology signature over ordered face indices;
- unique target `MeshAsset` ID;
- per-vertex before and after positions;
- evidence views and confidence;
- maximum movement relative to the source bounds;
- boundary/component policy.

Apply verifies the fingerprints and before values, refuses pinned or nonfinite
vertices, patches only literal `Vertex position` spans, reparses the complete
source, and atomically returns a candidate. Failure never returns a partially
modified source.

Before patching, the candidate is checked for invalid indices, repeated face
vertices, degenerate faces, open or non-manifold edges, disconnected face
components, inconsistent winding, collapsed closed volume and non-adjacent
triangle self-intersection. Self-intersection uses an AABB sweep broad phase
before segment/triangle narrow-phase tests. Coplanar overlaps remain a known V1
limitation and should receive an additional visual review.

```sh
cargo run -p motionloom --example apply_mesh_proposal -- \
  scene.motionloom proposal.json candidate.motionloom
```

## Recommended LLM loop

1. Resolve attachment or Drive content to bytes in the host.
2. Analyze every view and inspect masks/overlays.
3. Have the LLM label regions and execute a GeometryRecipe for a low-resolution,
   valid `MeshAsset`.
4. Freeze cameras and evaluate every available view.
5. Change a small, coherent vertex group using the ranked residuals.
6. Apply the fingerprinted proposal.
7. Evaluate the candidate and accept only cross-view improvement.
8. Repeat until thresholds, stagnation, ambiguity or the movement budget stops
   the process; subdivide only after the control cage is accepted.

One image proves only one projection. Call a result `camera_match` for a single
view, `multiview_fit` for consistent multiple views, and never claim full
reconstruction when the back or depth is unobserved.

## Current limits

- Built-in segmentation is deterministic color/alpha/guided geometry analysis,
  not a bundled neural segmentation model.
- Semantic part recognition remains a host/LLM responsibility.
- Monocular depth is a hint, not metric evidence.
- Evaluation uses authored `MeshAsset` control faces so vertex residuals remain
  directly editable; always visually review the final subdivided render.
- V1 proposals do not edit topology or UVs.
- Mesh authoring topology proposals are a separate fingerprint-safe operation;
  rebind every invalidated feature after using one.
- Evaluation is an authoring operation and should run in a worker in browser
  hosts; it is not part of immediate playback.
