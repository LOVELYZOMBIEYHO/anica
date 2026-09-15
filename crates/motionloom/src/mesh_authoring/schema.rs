// =========================================
// =========================================
// crates/motionloom/src/mesh_authoring/schema.rs

use crate::ControlCageNode;
use crate::mesh_reference::{ImageReferenceAnalysis, MeshReferenceEvaluation, MeshTopologyReport};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MESH_AUTHORING_SCHEMA_VERSION: &str = "1.0";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Axis {
    #[default]
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReconstructionClass {
    CameraMatch,
    #[default]
    PartialMultiviewFit,
    MultiviewFit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LoopSpec {
    pub id: String,
    pub center: [f32; 3],
    pub radius_a: f32,
    pub radius_b: f32,
    pub segments: usize,
    #[serde(default)]
    pub normal_axis: Axis,
    #[serde(default)]
    pub rotation_degrees: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum UvProjection {
    Planar {
        u_axis: Axis,
        v_axis: Axis,
        #[serde(default = "unit2")]
        scale: [f32; 2],
        #[serde(default)]
        offset: [f32; 2],
    },
    Cylindrical {
        axis: Axis,
        #[serde(default = "unit2")]
        scale: [f32; 2],
        #[serde(default)]
        offset: [f32; 2],
    },
    Spherical {
        #[serde(default = "unit2")]
        scale: [f32; 2],
        #[serde(default)]
        offset: [f32; 2],
    },
    ReferenceCamera {
        origin: [f32; 3],
        right: [f32; 3],
        up: [f32; 3],
        forward: [f32; 3],
        focal: f32,
        center: [f32; 2],
        image_size: [u32; 2],
    },
}

fn unit2() -> [f32; 2] {
    [1.0, 1.0]
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum GeometryOperation {
    CreateLoop {
        spec: LoopSpec,
    },
    DefineRegion {
        id: String,
        #[serde(default)]
        vertices: Vec<u64>,
        #[serde(default)]
        faces: Vec<u64>,
    },
    LoftLoops {
        id: String,
        loops: Vec<String>,
        #[serde(default)]
        cap_start: bool,
        #[serde(default)]
        cap_end: bool,
    },
    CreateSurface {
        id: String,
        rows: Vec<Vec<[f32; 3]>>,
        #[serde(default)]
        close_columns: bool,
    },
    SweepProfile {
        id: String,
        path: Vec<[f32; 3]>,
        radius: f32,
        segments: usize,
    },
    ExtrudeRegion {
        id: String,
        region: String,
        vector: [f32; 3],
        #[serde(default = "one_segment")]
        segments: usize,
    },
    ThickenSurface {
        id: String,
        region: String,
        thickness: f32,
    },
    TransformRegion {
        id: String,
        region: String,
        #[serde(default)]
        translate: [f32; 3],
        #[serde(default = "unit3")]
        scale: [f32; 3],
    },
    MirrorRegion {
        id: String,
        region: String,
        axis: Axis,
        #[serde(default)]
        weld_center: bool,
        #[serde(default = "default_weld_tolerance")]
        tolerance: f32,
    },
    WeldVertices {
        id: String,
        #[serde(default)]
        region: Option<String>,
        #[serde(default = "default_weld_tolerance")]
        tolerance: f32,
    },
    CapBoundary {
        id: String,
        loop_id: String,
    },
    GenerateUv {
        id: String,
        #[serde(default)]
        region: Option<String>,
        projection: UvProjection,
    },
}

fn one_segment() -> usize {
    1
}

fn unit3() -> [f32; 3] {
    [1.0; 3]
}

fn default_weld_tolerance() -> f32 {
    1e-5
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GeometryRecipe {
    pub schema_version: String,
    pub id: String,
    #[serde(default)]
    pub subdivision: u32,
    pub operations: Vec<GeometryOperation>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SemanticRegion {
    pub id: String,
    #[serde(default)]
    pub vertices: Vec<u64>,
    #[serde(default)]
    pub faces: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MeshHandleKind {
    Vertex,
    Face,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshHandle {
    pub revision_id: String,
    pub topology_signature: String,
    pub kind: MeshHandleKind,
    pub index: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TopologyCorrespondence {
    #[serde(default)]
    pub old_to_new_vertices: BTreeMap<u64, Vec<u64>>,
    #[serde(default)]
    pub old_to_new_faces: BTreeMap<u64, Vec<u64>>,
    #[serde(default)]
    pub created_vertices: Vec<u64>,
    #[serde(default)]
    pub created_faces: Vec<u64>,
    #[serde(default)]
    pub removed_vertices: Vec<u64>,
    #[serde(default)]
    pub removed_faces: Vec<u64>,
    #[serde(default)]
    pub invalidated_regions: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OperationResult {
    pub id: String,
    pub operation: String,
    pub created_vertices: Vec<u64>,
    pub created_faces: Vec<u64>,
    pub affected_regions: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshAuthoringResult {
    pub schema_version: String,
    pub recipe_id: String,
    pub cage: ControlCageNode,
    pub regions: BTreeMap<String, SemanticRegion>,
    pub correspondence: TopologyCorrespondence,
    pub operations: Vec<OperationResult>,
    pub topology: MeshTopologyReport,
    pub topology_signature: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RevisionStatus {
    #[default]
    Draft,
    Candidate,
    Accepted,
    Rejected,
    Superseded,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshRevision {
    pub id: String,
    pub parent: Option<String>,
    pub status: RevisionStatus,
    pub source: String,
    pub source_fingerprint: String,
    pub topology_signature: String,
    #[serde(default)]
    pub evaluation: Option<MeshReferenceEvaluation>,
    #[serde(default)]
    pub correspondence: TopologyCorrespondence,
    #[serde(default)]
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FrozenEvaluationSetup {
    pub camera_fingerprint: String,
    pub reference_set_fingerprint: String,
    pub metric_profile_fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshAuthoringSession {
    pub schema_version: String,
    pub session_id: String,
    pub classification: ReconstructionClass,
    pub target_asset_id: String,
    pub target_model_id: String,
    pub accepted_revision: String,
    pub revisions: Vec<MeshRevision>,
    #[serde(default)]
    pub analyses: Vec<ImageReferenceAnalysis>,
    #[serde(default)]
    pub frozen_evaluation: Option<FrozenEvaluationSetup>,
    #[serde(default)]
    pub assumptions: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceInput {
    pub image_bytes: Vec<u8>,
    pub request: crate::mesh_reference::AnalyzeImageReferenceRequest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshTopologyProposal {
    pub schema_version: String,
    pub source_fingerprint: String,
    pub topology_signature: String,
    pub camera_fingerprint: String,
    pub reference_set_fingerprint: String,
    pub metric_profile_fingerprint: String,
    pub evaluation_fingerprint: String,
    pub target_asset_id: String,
    pub reason: String,
    pub operations: Vec<GeometryOperation>,
    #[serde(default)]
    pub regions: Vec<SemanticRegion>,
    #[serde(default)]
    pub validation: crate::mesh_reference::MeshProposalValidationOptions,
    #[serde(default)]
    pub evidence_views: Vec<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApplyTopologyProposalResult {
    pub schema_version: String,
    pub source: String,
    pub source_fingerprint: String,
    pub topology_signature: String,
    pub topology: MeshTopologyReport,
    pub correspondence: TopologyCorrespondence,
    pub invalidated_feature_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionPositionProposalResult {
    pub revision_id: String,
    pub applied: crate::mesh_reference::ApplyMeshProposalResult,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionTopologyProposalResult {
    pub revision_id: String,
    pub applied: ApplyTopologyProposalResult,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    QualityGatesPassed,
    NegligibleImprovement,
    ContradictoryViews,
    MovementBudgetExhausted,
    TopologyInsufficient,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StopConditions {
    pub negligible_improvement_threshold: f32,
    pub negligible_iteration_limit: usize,
    pub recent_objectives: Vec<f32>,
    pub contradictory_views: bool,
    pub movement_budget_exhausted: bool,
    pub topology_insufficient: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RevisionDecision {
    pub revision_id: String,
    pub accepted: bool,
    pub objective_before: f32,
    pub objective_after: f32,
    pub maximum_primary_view_regression: f32,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshAuthoringArtifactManifest {
    pub session: String,
    pub recipe: String,
    pub accepted_source: String,
    pub control_cage: String,
    pub references: String,
    pub analyses: String,
    pub evaluations: String,
    pub proposals: String,
    pub overlays: String,
    pub differences: String,
    pub fit_summary: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshAuthoringCapabilities {
    pub schema_version: String,
    pub operations: Vec<String>,
    pub reference_functions: Vec<String>,
    pub proposal_functions: Vec<String>,
    pub maximum_vertices: usize,
    pub maximum_faces: usize,
}
