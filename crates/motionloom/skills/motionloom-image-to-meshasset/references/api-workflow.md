# Mesh reference API workflow

MotionLoom exposes one backward-compatible mesh-reference contract with
`schemaVersion: "1.0"`. The semantic-feature, objective, quality-gate, and full
fingerprint fields described here extend that contract; they are not a separate
V2 schema. Suffixes such as `references-v2.json` are authoring-run names only.

The workflow can attempt any raster reference that the image decoder accepts,
but input suitability determines the result:

- one clear view supports a `camera_match` or shallow 2.5D result;
- two consistent orthogonal views support a `partial_multiview_fit`;
- front, side, and back views support a `multiview_fit`;
- transparent, isolated, evenly lit subjects are easier to segment;
- heavy occlusion, reflections, blur, cropped silhouettes, inconsistent lenses,
  or unrelated views can prevent convergence;
- one image does not reveal trustworthy unseen geometry or metric depth.

Always keep reference bytes outside the DSL. Give each byte-exact image an
opaque `imageId` and retain the returned `contentHash`.

## Phase 0: GeometryRecipe and MeshAuthoringSession

Query `meshAuthoringSchema` first so the LLM uses supported operations. Describe
the initial cage as a generic `GeometryRecipe`, then execute it through the Rust
API or WASM `executeGeometryRecipe`. Example:

```json
{
  "schemaVersion": "1.0",
  "id": "initial-control-cage",
  "subdivision": 0,
  "operations": [
    {
      "type": "createLoop",
      "spec": {
        "id": "section-a",
        "center": [-1, 0, 0],
        "radiusA": 0.2,
        "radiusB": 0.15,
        "segments": 8,
        "normalAxis": "x",
        "rotationDegrees": 0
      }
    },
    {
      "type": "createLoop",
      "spec": {
        "id": "section-b",
        "center": [1, 0, 0],
        "radiusA": 0.1,
        "radiusB": 0.08,
        "segments": 8,
        "normalAxis": "x",
        "rotationDegrees": 0
      }
    },
    {
      "type": "loftLoops",
      "id": "body-volume",
      "loops": ["section-a", "section-b"],
      "capStart": true,
      "capEnd": true
    }
  ]
}
```

The result includes the cage, semantic regions, operation results, topology
report, topology signature, and correspondence. Insert the emitted `MeshAsset`
into a neutral evaluation scene and create a `MeshAuthoringSession`. Revision 0
is the accepted baseline. All later candidates remain immutable until explicitly
accepted or rejected.

Do not implement one-off geometry code in Python, Blender, `bpy`, or `bmesh`.
Recipes may describe any subject, but the API must remain object-independent.

## Phase A: analyzeImageReference

Run the analysis example once for every byte-exact reference:

```sh
cargo run -p motionloom --example analyze_image_reference -- \
  reference.png request.json /tmp/reference-analysis
```

The output directory contains:

- `analysis.json`: reusable analysis contract, including `contentHash`;
- `mask.png`: segmented foreground;
- `internal-edges.png`: image gradients inside the foreground;
- `overlay.png`: mask, contour, landmark, and feature inspection image.

Inspect the PNG artifacts before fitting. Correct a bad mask or misplaced hint
instead of compensating for it by deforming the mesh.

### Complete analysis request

```json
{
  "schemaVersion": "1.0",
  "imageId": "attachment-front-sha-prefix",
  "view": "front",
  "segmentation": {
    "mode": "guided",
    "analysisRegion": [20, 20, 900, 900],
    "foregroundPoints": [[450, 450]],
    "backgroundPoints": [[25, 25], [915, 25]],
    "backgroundColor": null,
    "threshold": 42,
    "suppliedMask": null,
    "minComponentPixels": 64,
    "maxContourPoints": 512
  },
  "requestedLandmarks": [
    "topmost",
    "bottommost",
    "leftmost",
    "rightmost"
  ],
  "analysisProfile": "human_head_v1",
  "featureHints": [
    {
      "id": "nose_tip",
      "kind": "point",
      "points": [[452, 326]],
      "semanticLabel": "nose_tip",
      "binding": {"type": "vertex", "vertex": 624},
      "confidence": 0.95,
      "snapRadius": 5
    },
    {
      "id": "mouth_line",
      "kind": "polyline",
      "points": [[420, 370], [426, 372], [432, 371]],
      "semanticLabel": "mouth_line",
      "binding": {
        "type": "vertexChain",
        "vertices": [520, 528, 536],
        "closed": false
      },
      "confidence": 0.82,
      "snapRadius": 4
    }
  ],
  "depthHints": [
    {
      "relation": "inFrontOf",
      "subject": "nose_tip",
      "object": "mouth_line",
      "evidence": "consistentSideView",
      "value": null,
      "confidence": 0.75
    }
  ]
}
```

`analysisProfile` is an optional diagnostic profile. The current
`human_head_v1` profile requires host- or LLM-supplied feature hints; MotionLoom
does not invent facial semantics.

### Segmentation modes

- `alpha`: use the supplied image alpha channel.
- `mask`: use `suppliedMask`, encoded as `MaskData` RLE.
- `guided`: use ROI plus foreground/background seed points.
- `backgroundColor`: use an RGB `backgroundColor`, for example `[240,240,240]`.
- `auto`: initial fallback only; inspect its artifacts carefully.

`analysisRegion` is `[x, y, width, height]`. Pixel coordinates use image space
with the origin at the upper-left. `threshold`, seed points, component size, and
contour-point limit are segmentation controls, not mesh-fitting controls.

A supplied mask has this form:

```json
{
  "width": 1024,
  "height": 1024,
  "startsForeground": false,
  "runs": [120, 18, 100, 22]
}
```

### Feature kinds and bindings

Feature kinds are `point`, `polyline`, `closedContour`, and `region`.
`closedContour` and `region` require at least three points; `point` requires
exactly one.

Available bindings are:

```json
{"type":"vertex","vertex":18}
{"type":"edge","vertices":[18,19],"t":0.4}
{"type":"face","face":7,"barycentric":[0.2,0.3,0.5]}
{"type":"vertexChain","vertices":[18,19,20],"closed":false}
{"type":"nearestContour"}
```

Use `nearestContour` only while analyzing an unbound hint. Bind it to actual
mesh topology before expecting an internal feature residual. A face binding is
evaluated with three barycentric weights.

The analyzer snaps each hint toward a nearby internal edge within
`snapRadius`. It lowers confidence when no credible edge is found. Feature IDs
must be unique within a view and are also used by depth relations.

Supported ordinal depth relations are:

- `inFrontOf` or `closerThan`;
- `behind` or `fartherThan`.

Both `subject` and `object` must name bound features. Unknown or zero-confidence
depth hints are retained as uncertainty but do not add an objective penalty.

## Author and freeze the MeshAsset

Create ordinary `MeshAsset`, `Vertex`, and `Face` DSL. Do not add fitting tags to
the DSL. Keep subdivision off while evaluating the editable control cage.

Create one fixed scene frame per reference and finish camera/object calibration
before modifying geometry. During fitting keep all of these unchanged:

- reference bytes and dimensions;
- camera and selected frame;
- object transform;
- render dimensions;
- metric weights and quality gates.

Use `cameraPolicy: "frozen"`, a neutral material, flat background, and no DoF,
TAA jitter, motion blur, wind, or stylized outline.

## Phase B: evaluateMeshAssetReference

Evaluate a complete multiview set:

```sh
cargo run -p motionloom --example evaluate_mesh_reference -- \
  scene.motionloom references.json /tmp/mesh-evaluation
```

For one existing analysis:

```sh
cargo run -p motionloom --example evaluate_mesh_reference -- \
  scene.motionloom --analysis /tmp/reference-analysis/analysis.json \
  object_mesh object_model 0 /tmp/mesh-evaluation
```

### Complete multiview request

Replace each `{}` analysis value below with the full corresponding
`analysis.json` object:

```json
{
  "schemaVersion": "1.0",
  "targetAssetId": "object_mesh",
  "targetModelId": "object_model",
  "references": [
    {"id": "front", "frame": 0, "analysis": {}},
    {"id": "right", "frame": 48, "analysis": {}},
    {"id": "back", "frame": 96, "analysis": {}},
    {"id": "left", "frame": 144, "analysis": {}}
  ],
  "options": {
    "cameraPolicy": "frozen",
    "allowBoundary": false,
    "allowMultipleComponents": false,
    "maxVertexResiduals": 512,
    "metricWeights": {
      "silhouette": 0.25,
      "boundary": 0.20,
      "landmarks": 0.25,
      "features": 0.20,
      "depth": 0.10
    },
    "qualityGates": {
      "minimumMaskIou": 0.90,
      "maximumP95EdgeDistancePx": 16,
      "maximumLandmarkMeanDistancePx": 6,
      "maximumFeatureMeanDistancePx": 6,
      "maximumDepthViolations": 0
    }
  }
}
```

Weights are normalized over active components. Feature and depth weights become
inactive when no evaluable feature or confident two-feature depth relation is
present. Keep the same weights through an accepted/rejected proposal sequence.

### Evaluation report

`evaluation.json` includes aggregate `objective`, `passesQualityGates`, and one
entry per view. Inspect:

- `maskIou`;
- `meanEdgeDistancePx`, `p95EdgeDistancePx`, and
  `maximumEdgeDistancePx`;
- `landmarks` and `features`, including candidate points and residual distances;
- `depthViolations`;
- `objectiveComponents` for silhouette, boundary, landmarks, features, depth;
- ranked visible `vertexResiduals`;
- per-view and aggregate quality-gate results.

The output directory also includes `{view}-overlay.png` and
`{view}-difference.png`. In an overlay, white agrees, green is missing candidate
area, and magenta is candidate overflow.

Copy all six fingerprints from the current evaluation before constructing a
proposal:

- `sourceFingerprint` binds exact source text;
- `topologySignature` binds vertex/face topology;
- `cameraFingerprint` binds all evaluated camera snapshots;
- `referenceSetFingerprint` binds image hashes, analyses, features, and depth;
- `metricProfileFingerprint` binds weights and gates;
- `evaluationFingerprint` binds the preceding five values together.

Any source, topology, camera, reference, or metric change makes the proposal
context stale.

## Phase C: applyMeshAssetProposal

Apply one bounded proposal to the exact source that produced its evaluation:

```sh
cargo run -p motionloom --example apply_mesh_proposal -- \
  accepted.motionloom proposal.json candidate.motionloom
```

### Complete fingerprint-safe proposal

```json
{
  "schemaVersion": "1.0",
  "sourceFingerprint": "FROM_CURRENT_EVALUATION",
  "topologySignature": "FROM_CURRENT_EVALUATION",
  "cameraFingerprint": "FROM_CURRENT_EVALUATION",
  "referenceSetFingerprint": "FROM_CURRENT_EVALUATION",
  "metricProfileFingerprint": "FROM_CURRENT_EVALUATION",
  "evaluationFingerprint": "FROM_CURRENT_EVALUATION",
  "targetAssetId": "object_mesh",
  "reason": "both profile views place the nose tip farther forward",
  "changes": [
    {
      "vertex": 624,
      "before": [0.0, -0.1188374, 0.8993575],
      "after": [0.0, -0.1188374, 0.9093575],
      "confidence": 0.84,
      "evidenceViews": ["left", "right"]
    }
  ],
  "validation": {
    "allowBoundary": false,
    "allowMultipleComponents": false,
    "maxMoveRelativeToBounds": 0.02,
    "maxChangedVertices": 16,
    "maxEdgeLengthRatio": 1.25,
    "maxLaplacianDeltaRelativeToBounds": 0.04
  }
}
```

All four evaluation-context fingerprints are optional only for compatibility
with older schema-1.0 requests. If any one is supplied, all four must be
supplied. New LLM workflows must always supply all six fingerprints shown
above.

Every `before` value must exactly match the current source. The apply operation
is atomic and rejects:

- stale or inconsistent fingerprints;
- duplicate, missing, pinned, or out-of-range vertices;
- non-finite coordinates or moves above the relative bound;
- batches above `maxChangedVertices`;
- edge stretch above `maxEdgeLengthRatio`;
- local deformation spikes above `maxLaplacianDeltaRelativeToBounds`;
- boundary, component, winding, non-manifold, degenerate-face, collapsed-volume,
  or detected self-intersection failures.

The returned candidate includes a new `sourceFingerprint`; its topology
signature normally remains stable because schema 1.0 proposals modify positions
only.

## Phase D: applyMeshTopologyProposal

Use this phase only when moving the existing cage cannot represent required
geometry. A topology proposal uses GeometryRecipe operations on semantic regions:

```json
{
  "schemaVersion": "1.0",
  "sourceFingerprint": "FROM_CURRENT_EVALUATION",
  "topologySignature": "FROM_CURRENT_EVALUATION",
  "cameraFingerprint": "FROM_CURRENT_EVALUATION",
  "referenceSetFingerprint": "FROM_CURRENT_EVALUATION",
  "metricProfileFingerprint": "FROM_CURRENT_EVALUATION",
  "evaluationFingerprint": "FROM_CURRENT_EVALUATION",
  "targetAssetId": "object_mesh",
  "reason": "the observed appendage needs its own connected extrusion",
  "operations": [
    {
      "type": "extrudeRegion",
      "id": "appendage-extrusion",
      "region": "appendage-root",
      "vector": [0, 0.18, 0],
      "segments": 1
    }
  ],
  "regions": [
    {"id": "appendage-root", "vertices": [], "faces": [42]}
  ],
  "validation": {
    "allowBoundary": false,
    "allowMultipleComponents": false,
    "maxMoveRelativeToBounds": 0.02,
    "maxChangedVertices": 16,
    "maxEdgeLengthRatio": 1.25,
    "maxLaplacianDeltaRelativeToBounds": 0.04
  },
  "evidenceViews": ["front", "left"],
  "confidence": 0.82
}
```

Call the Rust function `apply_mesh_topology_proposal` or WASM
`applyMeshTopologyProposal`. The operation validates the complete candidate and
atomically rewrites only the named MeshAsset. It returns old-to-new vertex and
face mappings, created and removed handles, invalidated regions, and
`invalidatedFeatureIds`.

Rebind every invalidated evaluated feature before using it again. Evaluate the
new candidate with the same views; the camera, reference bytes, and metric
profile remain frozen. The source, topology, and evaluation fingerprints will
change. Use only the accepted candidate's new fingerprints in later proposals.

## Accept, reject, and repeat

Immediately evaluate `candidate.motionloom` against the same frozen
`references.json`:

```sh
cargo run -p motionloom --example evaluate_mesh_reference -- \
  candidate.motionloom references.json /tmp/candidate-evaluation
```

Accept only if aggregate objective improves, topology remains valid, important
features improve or remain stable, and no primary view regresses beyond the
run's recorded tolerance. Confirm with overlays. Otherwise discard the
candidate and continue from the previous accepted source.

For the next proposal, use only the new accepted evaluation fingerprints and
the exact accepted vertex values. Never reuse or edit a stale proposal.

Repeat `evaluate -> propose -> apply -> evaluate`. Re-run analysis only when the
image, mask, landmarks, semantic hints, or bindings change. Stop when gates
pass, three iterations give negligible progress, views conflict, the movement
budget is exhausted, or topology must change.

Position proposals cannot add or delete vertices or faces. Use a topology
proposal for supported local topology changes. When the operation vocabulary is
still insufficient, preserve the accepted source, revise the GeometryRecipe,
rebind affected features, and establish a new baseline before resuming. Enable
subdivision only after fitting converges, and visually inspect the subdivided
delivery render.

Use `decide_candidate_revision` for the aggregate and per-view acceptance rule,
`stop_reason` for the recorded stopping condition, and `artifact_manifest` for
the standard authoring output layout.

## Applicability limits

Any single raster image can be submitted to `analyzeImageReference`, and the
workflow can attempt a `MeshAsset` fit when segmentation succeeds. This does not
mean every image can yield an accurate complete 3D object.

Good candidates include isolated products, props, logos with depth, vehicles,
heads, buildings, and objects with multiple consistent views. Difficult or
ambiguous candidates include transparent hair, smoke, liquids, crowds, severe
motion blur, texture-only detail, repeating occlusion, and inconsistent images
of different instances.

For a single view, produce a camera-matched 2.5D result and mark unseen geometry
as uncertain. For inconsistent multiview inputs, report the conflict rather than
distorting one MeshAsset until every independent crop appears to fit.
