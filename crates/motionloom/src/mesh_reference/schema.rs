// =========================================
// =========================================
// crates/motionloom/src/mesh_reference/schema.rs

use serde::{Deserialize, Serialize};

pub const MESH_REFERENCE_SCHEMA_VERSION: &str = "1.0";

fn one() -> f32 {
    1.0
}

fn default_threshold() -> f32 {
    42.0
}

fn default_min_component_pixels() -> u32 {
    64
}

fn default_max_contour_points() -> usize {
    512
}

fn default_snap_radius() -> u32 {
    8
}

fn default_max_changed_vertices() -> usize {
    64
}

fn default_max_edge_length_ratio() -> f32 {
    1.5
}

fn default_max_laplacian_delta() -> f32 {
    0.12
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshReferenceDiagnostic {
    pub severity: String,
    pub code: String,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SegmentationMode {
    Alpha,
    BackgroundColor,
    Guided,
    Mask,
    #[default]
    Auto,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct SegmentationOptions {
    pub mode: SegmentationMode,
    pub analysis_region: Option<[u32; 4]>,
    pub foreground_points: Vec<[u32; 2]>,
    pub background_points: Vec<[u32; 2]>,
    pub background_color: Option<[u8; 3]>,
    pub threshold: f32,
    pub supplied_mask: Option<MaskData>,
    pub min_component_pixels: u32,
    pub max_contour_points: usize,
}

impl Default for SegmentationOptions {
    fn default() -> Self {
        Self {
            mode: SegmentationMode::Auto,
            analysis_region: None,
            foreground_points: vec![],
            background_points: vec![],
            background_color: None,
            threshold: default_threshold(),
            supplied_mask: None,
            min_component_pixels: default_min_component_pixels(),
            max_contour_points: default_max_contour_points(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnalyzeImageReferenceRequest {
    pub schema_version: String,
    pub image_id: String,
    pub view: String,
    #[serde(default)]
    pub segmentation: SegmentationOptions,
    #[serde(default)]
    pub requested_landmarks: Vec<String>,
    #[serde(default)]
    pub analysis_profile: Option<String>,
    #[serde(default)]
    pub feature_hints: Vec<ReferenceFeatureHint>,
    #[serde(default)]
    pub depth_hints: Vec<DepthHint>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MaskData {
    pub width: u32,
    pub height: u32,
    pub starts_foreground: bool,
    pub runs: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceContour {
    pub id: String,
    pub closed: bool,
    pub hole: bool,
    pub points: Vec<[f32; 2]>,
    #[serde(default = "one")]
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceRegion {
    pub id: String,
    pub bounds: [u32; 4],
    pub area: u32,
    pub parent: Option<String>,
    pub semantic_label: Option<String>,
    #[serde(default = "one")]
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum LandmarkBinding {
    Vertex { vertex: usize },
    Edge { vertices: [usize; 2], t: f32 },
    Face { face: usize, barycentric: [f32; 3] },
    NearestContour,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceLandmark {
    pub id: String,
    pub pixel: [f32; 2],
    pub semantic_label: Option<String>,
    pub binding: Option<LandmarkBinding>,
    #[serde(default = "one")]
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReferenceFeatureKind {
    #[default]
    Point,
    Polyline,
    ClosedContour,
    Region,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum FeatureBinding {
    Vertex { vertex: usize },
    Edge { vertices: [usize; 2], t: f32 },
    Face { face: usize, barycentric: [f32; 3] },
    VertexChain { vertices: Vec<usize>, closed: bool },
    NearestContour,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceFeatureHint {
    pub id: String,
    #[serde(default)]
    pub kind: ReferenceFeatureKind,
    pub points: Vec<[f32; 2]>,
    pub semantic_label: Option<String>,
    pub binding: Option<FeatureBinding>,
    #[serde(default = "one")]
    pub confidence: f32,
    #[serde(default = "default_snap_radius")]
    pub snap_radius: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReferenceFeature {
    pub id: String,
    pub kind: ReferenceFeatureKind,
    pub points: Vec<[f32; 2]>,
    pub semantic_label: Option<String>,
    pub binding: Option<FeatureBinding>,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DepthHint {
    pub relation: String,
    pub subject: String,
    pub object: Option<String>,
    pub evidence: String,
    pub value: Option<f32>,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImageReferenceAnalysis {
    pub schema_version: String,
    pub image_id: String,
    pub content_hash: String,
    pub image_size: [u32; 2],
    pub view: String,
    pub foreground_mask: MaskData,
    pub contours: Vec<ReferenceContour>,
    pub regions: Vec<ReferenceRegion>,
    pub landmarks: Vec<ReferenceLandmark>,
    #[serde(default)]
    pub features: Vec<ReferenceFeature>,
    #[serde(default)]
    pub internal_edge_mask: Option<MaskData>,
    pub depth_hints: Vec<DepthHint>,
    pub confidence: f32,
    pub diagnostics: Vec<MeshReferenceDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshReferenceView {
    pub id: String,
    pub frame: u32,
    pub analysis: ImageReferenceAnalysis,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshReferenceOptions {
    pub camera_policy: String,
    pub allow_boundary: bool,
    pub allow_multiple_components: bool,
    pub max_vertex_residuals: usize,
    pub metric_weights: MeshMetricWeights,
    pub quality_gates: MeshQualityGates,
}

impl Default for MeshReferenceOptions {
    fn default() -> Self {
        Self {
            camera_policy: "frozen".into(),
            allow_boundary: true,
            allow_multiple_components: false,
            max_vertex_residuals: 256,
            metric_weights: MeshMetricWeights::default(),
            quality_gates: MeshQualityGates::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshMetricWeights {
    pub silhouette: f32,
    pub boundary: f32,
    pub landmarks: f32,
    pub features: f32,
    pub depth: f32,
}

impl Default for MeshMetricWeights {
    fn default() -> Self {
        Self {
            silhouette: 0.25,
            boundary: 0.20,
            landmarks: 0.25,
            features: 0.20,
            depth: 0.10,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshQualityGates {
    pub minimum_mask_iou: f32,
    pub maximum_p95_edge_distance_px: f32,
    pub maximum_landmark_mean_distance_px: f32,
    pub maximum_feature_mean_distance_px: f32,
    pub maximum_depth_violations: usize,
}

impl Default for MeshQualityGates {
    fn default() -> Self {
        Self {
            minimum_mask_iou: 0.90,
            maximum_p95_edge_distance_px: 16.0,
            maximum_landmark_mean_distance_px: 6.0,
            maximum_feature_mean_distance_px: 6.0,
            maximum_depth_violations: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshReferenceSet {
    pub schema_version: String,
    pub target_asset_id: String,
    pub target_model_id: String,
    pub references: Vec<MeshReferenceView>,
    #[serde(default)]
    pub options: MeshReferenceOptions,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LandmarkResidual {
    pub id: String,
    pub reference: [f32; 2],
    pub candidate: Option<[f32; 2]>,
    pub delta: Option<[f32; 2]>,
    pub distance_px: Option<f32>,
    pub confidence: f32,
    pub method: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VertexResidual {
    pub vertex: usize,
    pub screen_point: [f32; 2],
    pub screen_delta: [f32; 2],
    pub distance_px: f32,
    pub visibility: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeatureResidual {
    pub id: String,
    pub kind: ReferenceFeatureKind,
    pub reference_points: Vec<[f32; 2]>,
    pub candidate_points: Vec<[f32; 2]>,
    pub mean_distance_px: Option<f32>,
    pub p95_distance_px: Option<f32>,
    pub maximum_distance_px: Option<f32>,
    pub confidence: f32,
    pub method: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshObjectiveComponents {
    pub silhouette: f32,
    pub boundary: f32,
    pub landmarks: f32,
    pub features: f32,
    pub depth: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshReferenceViewEvaluation {
    pub id: String,
    pub view: String,
    pub frame: u32,
    pub mask_iou: f32,
    pub mean_edge_distance_px: f32,
    pub p95_edge_distance_px: f32,
    pub maximum_edge_distance_px: f32,
    pub reference_pixels: u32,
    pub candidate_pixels: u32,
    pub candidate_mask: MaskData,
    pub landmarks: Vec<LandmarkResidual>,
    pub features: Vec<FeatureResidual>,
    pub depth_violations: usize,
    pub objective_components: MeshObjectiveComponents,
    pub passes_quality_gates: bool,
    pub vertex_residuals: Vec<VertexResidual>,
    pub error: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshReferenceEvaluation {
    pub schema_version: String,
    pub measurement: String,
    pub source_fingerprint: String,
    pub topology_signature: String,
    pub camera_fingerprint: String,
    pub reference_set_fingerprint: String,
    pub metric_profile_fingerprint: String,
    pub evaluation_fingerprint: String,
    pub target_asset_id: String,
    pub target_model_id: String,
    pub objective: f32,
    pub passes_quality_gates: bool,
    pub views: Vec<MeshReferenceViewEvaluation>,
    pub diagnostics: Vec<MeshReferenceDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshVertexChange {
    pub vertex: usize,
    pub before: [f32; 3],
    pub after: [f32; 3],
    #[serde(default = "one")]
    pub confidence: f32,
    #[serde(default)]
    pub evidence_views: Vec<String>,
}

fn default_max_move() -> f32 {
    0.08
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshProposalValidationOptions {
    pub allow_boundary: bool,
    pub allow_multiple_components: bool,
    pub max_move_relative_to_bounds: f32,
    pub max_changed_vertices: usize,
    pub max_edge_length_ratio: f32,
    pub max_laplacian_delta_relative_to_bounds: f32,
}

impl Default for MeshProposalValidationOptions {
    fn default() -> Self {
        Self {
            allow_boundary: true,
            allow_multiple_components: false,
            max_move_relative_to_bounds: default_max_move(),
            max_changed_vertices: default_max_changed_vertices(),
            max_edge_length_ratio: default_max_edge_length_ratio(),
            max_laplacian_delta_relative_to_bounds: default_max_laplacian_delta(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshAssetProposal {
    pub schema_version: String,
    pub source_fingerprint: String,
    pub topology_signature: String,
    #[serde(default)]
    pub camera_fingerprint: Option<String>,
    #[serde(default)]
    pub reference_set_fingerprint: Option<String>,
    #[serde(default)]
    pub metric_profile_fingerprint: Option<String>,
    #[serde(default)]
    pub evaluation_fingerprint: Option<String>,
    pub target_asset_id: String,
    pub reason: String,
    pub changes: Vec<MeshVertexChange>,
    #[serde(default)]
    pub validation: MeshProposalValidationOptions,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeshTopologyReport {
    pub valid: bool,
    pub vertices: usize,
    pub faces: usize,
    pub open_edges: usize,
    pub non_manifold_edges: usize,
    pub connected_components: usize,
    pub degenerate_faces: Vec<usize>,
    pub inconsistent_winding_edges: usize,
    pub self_intersections: Vec<[usize; 2]>,
    pub signed_volume: Option<f32>,
    pub diagnostics: Vec<MeshReferenceDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApplyMeshProposalResult {
    pub schema_version: String,
    pub source: String,
    pub source_fingerprint: String,
    pub topology_signature: String,
    pub applied_changes: usize,
    pub topology: MeshTopologyReport,
}
