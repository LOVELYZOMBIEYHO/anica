---
name: motionloom-image-to-meshasset
description: Build and iteratively fit a universal MotionLoom MeshAsset from image references with GeometryRecipe, the Rust mesh_authoring API, analyzeImageReference, evaluateMeshAssetReference, and fingerprint-safe position or topology proposals. Use for camera-matched 2.5D objects or multiview 3D control cages without object-specific generators or external modeling APIs.
---

# MotionLoom Image to MeshAsset

Create a low-resolution explicit control cage, measure it against every supplied image, and improve it through small evidence-backed proposals. Never claim that one photograph proves unseen geometry.

## Mandatory execution contract

When a task requires reconstructing or fitting a high-quality model from image references, use `MeshAsset` and follow this skill completely.

You must actually execute:

- `executeGeometryRecipe`
- `analyzeImageReference`
- `evaluateMeshAssetReference`
- `applyMeshAssetProposal`

Do not modify the model through visual guesswork alone.

For every modification after the initial recipe:

1. Use the `sourceFingerprint` and `topologySignature` from the current evaluation.
2. Use `applyMeshAssetProposal` for position-only changes or
   `applyMeshTopologyProposal` for a small coherent topology operation.
3. After applying a proposal, re-evaluate every reference view.
4. Accept a proposal only when no primary view regresses beyond the configured tolerance.
5. If a regression occurs, preserve or restore the last accepted source.
6. Keep the cameras, render dimensions, and reference images unchanged throughout fitting.
7. The final model must be a `MeshAsset`; do not replace it with a `HeadAsset`, GLB, or another model format.
8. Enable subdivision only after the fitting process has converged.

## Resolve images outside MotionLoom

Use the host's attachment or Drive connector to obtain original image bytes. Keep credentials and remote URLs out of MotionLoom core and DSL. Assign each byte-exact image an opaque `imageId`; retain its returned `contentHash` throughout the run.

Do not generate missing reference views unless the user explicitly asks for synthetic design exploration. A generated view is not ground truth.

## Classify the intended result

- One view: label the result `camera_match`; prefer shallow or 2.5D geometry.
- Two consistent orthogonal views: label it `partial_multiview_fit`.
- Front, side, back, and optional held-out view: label it `multiview_fit`.
- Do not label any result `full_reconstruction` when important surfaces remain unobserved.

## Analyze each reference

Call `analyzeImageReference` with image bytes and a JSON request. Prefer:

1. supplied alpha or mask;
2. guided segmentation with an LLM-selected ROI and foreground/background points;
3. background-color segmentation;
4. `auto` only for an initial attempt.

Inspect `mask.png` and `overlay.png`. Correct the ROI or hints when the mask includes text, shadows, adjacent objects, or misses pale edges. Do not proceed merely because the API returned a mask.

Use returned geometric landmarks as measurements. Add semantic labels such as `blade_tip` or `wheel_center` only when visually justified. Treat all monocular depth as a low-confidence hint.

See [references/api-workflow.md](references/api-workflow.md) for commands and JSON shapes.

## Author the initial MeshAsset with GeometryRecipe

Use the smallest control cage that explains the visible silhouette and volume:

- keep faces connected and consistently wound;
- prefer quads for later Catmull-Clark subdivision;
- use shared indices at seams that must remain welded;
- add depth only when a side view, occlusion, or strong shape prior supports it;
- keep important silhouette vertices explicit;
- avoid textures, micro-detail, and high subdivision during fitting.

Express construction as a versioned `GeometryRecipe` and execute it through
`motionloom::api::mesh_authoring`. Use general operations such as cross-section
loops, loft, surface, sweep, extrusion, thickening, mirror, weld, cap, region
transform, and UV projection.

Keep stable operation IDs and semantic region IDs. Use revision-scoped handles
when selecting vertices or faces; a handle is valid only for its revision ID and
topology signature. Emit the returned cage as ordinary `MeshAsset`, `Vertex`,
and `Face` DSL, then parse and validate it before evaluation. Do not invent
reference-fitting tags.

## Freeze the comparison setup

Create one fixed camera/frame per reference view. Calibrate camera and object placement before geometry fitting, then keep `cameraPolicy="frozen"`. Use a neutral material, flat background, no depth of field, temporal jitter, motion blur, animation, wind, or stylized outline.

Do not change the camera and geometry in the same iteration.

## Evaluate every view

Call `evaluateMeshAssetReference`. Read:

- mask IoU;
- mean, p95, and maximum edge distance;
- landmark residuals;
- candidate overflow and missing area in overlay artifacts;
- ranked per-vertex screen residuals;
- source, topology, camera, and reference fingerprints.

White overlay pixels agree, green pixels are missing candidate area, and magenta pixels are candidate overflow. Diagnose all views together.

## Propose a bounded edit

Choose a small coherent vertex set. Infer a 3D movement only where two or more view residuals constrain it. For a single view, move within the camera plane unless depth evidence exists.

Construct `MeshAssetProposal` using the exact `before` values and fingerprints from the current source/evaluation. Include evidence views and confidence. Do not move pinned vertices or exceed the configured fraction of object bounds.

Call `applyMeshAssetProposal`. Never patch the DSL manually after creating the proposal: a stale source, changed topology, invalid winding, non-manifold edge, degenerate face, collapsed volume, or detected self-intersection must abort the whole edit.

## Propose a topology edit when the cage is insufficient

Use a `MeshTopologyProposal` only when position edits cannot express an observed
silhouette or surface. Include the same six current fingerprints, the exact
target asset, a small list of GeometryRecipe operations, semantic regions,
evidence views, confidence, and topology validation options.

Call `applyMeshTopologyProposal`. Never add or remove `Vertex` or `Face` lines
manually. Inspect its complete topology report and `TopologyCorrespondence`.
Treat every returned `invalidatedFeatureId` as unusable until it is rebound to
the candidate's new vertices, edges, faces, or vertex chains. Re-run analysis
with updated bindings when necessary, establish a new frozen evaluation
baseline, and do not reuse a pre-topology proposal.

## Accept or reject

Evaluate the returned candidate with the same references and frozen cameras.

Accept only when:

- aggregate error improves;
- no fitted view regresses beyond tolerance;
- important landmarks improve or remain stable;
- topology validation remains clean;
- a held-out view remains plausible;
- the visual overlay confirms the numerical result.

Otherwise reject the candidate revision and keep the previous accepted source.

## Repeat with limits

Repeat analyze only when the mask, landmarks, labels, or source image changed. Normally repeat evaluate → propose → apply → evaluate.

Stop when thresholds are met, three iterations make negligible progress, views
conflict, the movement budget is exhausted, or the available topology operations
still cannot express the subject. Use `stop_reason` to record the cause. If a
topology proposal succeeds, rebind invalidated features and continue from its
new accepted baseline.

Apply subdivision only after accepting the control cage, then render the subdivided result for visual review because V1 evaluation measures the authored editable cage.

## Deliver

Use `artifact_manifest` as the standard layout. Return the accepted
`.motionloom` source, the unsubdivided control cage, GeometryRecipe, session and
revision history, cropped references, analysis JSON, evaluation reports,
overlay/difference images, accepted and rejected proposals, fit summary, final
confidence label, and unresolved ambiguity. Never overwrite the user's original
source without explicit permission.
