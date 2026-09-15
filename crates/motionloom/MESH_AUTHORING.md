<!-- ========================================= -->
<!-- ========================================= -->
<!-- crates/motionloom/MESH_AUTHORING.md -->

# Mesh authoring API

`motionloom::api::mesh_authoring` is the standard, deterministic path for an
LLM or editor to construct and revise a universal `MeshAsset`. It is
filesystem-free and contains no object-specific generators.

## Workflow

1. Classify the reconstruction and record missing-view assumptions.
2. Call `analyze_references` with distinct, unchanged image bytes and image IDs.
3. Describe the initial cage as a versioned `GeometryRecipe`.
4. Run `execute_geometry_recipe`; each operation reports created handles and
   semantic regions.
5. Emit the result with `mesh_asset_element` inside the caller's scene.
6. Create a `MeshAuthoringSession`; revision 0 becomes the accepted source.
7. Validate the cage with the returned `MeshTopologyReport`.
8. Call `evaluate_session_revision` to establish the frozen camera, reference,
   and metric fingerprints.
9. Use `apply_position_proposal_to_session` for bounded edits to existing
   vertices.
10. Use `apply_topology_proposal_to_session` when the cage lacks the required
    loops, surfaces, volume, or UV layout.
11. Re-evaluate the candidate with the same `MeshReferenceSet`.
12. Call `decide_candidate_revision`; it accepts only an aggregate improvement
    within the per-view regression tolerance.
13. Use `stop_reason` and preserve the accepted source plus the files named by
    `artifact_manifest`.

Candidates are immutable. A rejected candidate never replaces the accepted
revision. Topology proposals require all six current fingerprints and return a
`TopologyCorrespondence`. Any vertex-, edge-, face-, or vertex-chain feature
listed in `invalidatedFeatureIds` must be rebound before the next evaluation.

## GeometryRecipe operations

The first schema version supports:

- cross-section loops and multi-loop lofts;
- point-grid surfaces and path sweeps;
- face-region extrusion and surface thickening;
- semantic region definition and region transforms;
- mirror, weld, and cap operations;
- planar, cylindrical, spherical, and frozen-reference-camera UV projection.

Every operation has a unique ID. Regions contain revision-local vertex and face
indices. `revision_handles` adds the revision ID and topology signature so a
host can reject stale LLM selections before mutation.

Recipes and proposals are capped at 30,000 vertices and 30,000 faces. The
existing topology validator rejects out-of-range indices, degenerate faces,
inconsistent winding, non-manifold edges, disallowed open boundaries,
disallowed disconnected components, and self-intersections.

## Rust and JSON

```rust
use motionloom::api::mesh_authoring::{
    GeometryRecipe, execute_geometry_recipe, mesh_asset_element,
};

let recipe: GeometryRecipe = serde_json::from_str(recipe_json)?;
let result = execute_geometry_recipe(&recipe)?;
let mesh_dsl = mesh_asset_element("subject", "neutral", &result.cage);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`execute_geometry_recipe_json` and `apply_mesh_topology_proposal_json` provide
string-based host adapters. WASM exports `meshAuthoringSchema`,
`executeGeometryRecipe`, and `applyMeshTopologyProposal`. Image analysis and
render-based evaluation remain the existing `analyzeImageReference` and
`evaluateMeshAssetReference` exports.

The native CLI can inspect the machine-readable schema or build an asset:

```text
cargo run -p motionloom --example mesh_authoring -- schema
cargo run -p motionloom --example mesh_authoring -- \
  build recipe.json mesh.motionloom subject neutral
cargo run -p motionloom --example mesh_authoring -- \
  apply-topology accepted.motionloom analyses.json proposal.json \
  evaluation.json candidate.motionloom
```

The `build` output is a standalone `MeshAsset` element intended for insertion
into an existing `<Assets>` block.

## Boundary of responsibility

The API creates geometry, validates it, evaluates references, and manages
revisions. The host owns image cropping, durable artifact storage, proposal
review, and scene presentation. The API never fetches image URLs and does not
call Blender, `bpy`, `bmesh`, or an external model generator.
