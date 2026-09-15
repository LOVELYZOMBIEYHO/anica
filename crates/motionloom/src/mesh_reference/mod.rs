// =========================================
// =========================================
// crates/motionloom/src/mesh_reference/mod.rs

//! Filesystem-free image-reference analysis and safe MeshAsset fitting tools.

mod analysis;
mod evaluation;
mod proposal;
mod schema;
mod topology;

pub use analysis::{
    analysis_internal_edge_png, analysis_mask_png, analysis_overlay_png, analyze_image_reference,
    analyze_image_reference_json, decode_mask,
};
pub(crate) use evaluation::evaluation_context_fingerprint;
pub use evaluation::{
    evaluate_mesh_asset_reference, evaluate_mesh_asset_reference_json,
    mesh_evaluation_difference_png, mesh_evaluation_overlay_png,
};
pub use proposal::{
    apply_mesh_asset_proposal, apply_mesh_asset_proposal_json, mesh_source_fingerprint,
    mesh_topology_signature,
};
pub(crate) use proposal::{mesh_asset, rewrite_mesh_asset_cage};
pub use schema::*;
pub use topology::validate_mesh_topology;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MeshReferenceError {
    #[error("invalid image reference: {0}")]
    Image(String),
    #[error("invalid mesh reference request: {0}")]
    Request(String),
    #[error("invalid MeshAsset candidate: {0}")]
    Candidate(String),
    #[error("source or proposal mismatch: {0}")]
    Source(String),
    #[error("mesh evaluation failed: {0}")]
    Evaluation(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub const MESH_REFERENCE_MEASUREMENT: &str = "triangle depth mask in the active runtime camera; mask IoU; bidirectional boundary, semantic feature, landmark, and ordinal-depth residuals in reference pixels; authored MeshAsset vertex residuals; no implicit image mirroring";

/// Compact machine-readable catalog for LLM tool adapters and host validation.
pub fn mesh_reference_schema_json() -> String {
    serde_json::json!({
        "schemaVersion": MESH_REFERENCE_SCHEMA_VERSION,
        "functions": {
            "analyzeImageReference": {
                "binaryArgument": "imageBytes",
                "jsonArgument": "AnalyzeImageReferenceRequest",
                "result": "ImageReferenceAnalysis"
            },
            "evaluateMeshAssetReference": {
                "arguments": ["source", "MeshReferenceSet"],
                "result": "MeshReferenceEvaluation",
                "async": true
            },
            "applyMeshAssetProposal": {
                "arguments": ["source", "MeshAssetProposal"],
                "result": "ApplyMeshProposalResult"
            }
        },
        "segmentationModes": ["alpha", "backgroundColor", "guided", "mask", "auto"],
        "landmarkBindings": ["vertex", "edge", "face", "nearestContour"],
        "featureKinds": ["point", "polyline", "closedContour", "region"],
        "featureBindings": ["vertex", "edge", "face", "vertexChain", "nearestContour"],
        "proposalPolicy": "existing vertex positions only; topology and UVs remain fixed",
        "measurement": MESH_REFERENCE_MEASUREMENT
    })
    .to_string()
}

#[cfg(test)]
mod tests;
