// =========================================
// =========================================
// crates/motionloom/src/rig_diagnostics.rs

//! Stable, read-only humanoid pose diagnostics shared by Rust, CLI, and WASM hosts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const RIG_DIAGNOSTICS_SCHEMA_VERSION: &str = "1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum RigReportDetail {
    Summary,
    #[default]
    Body,
    Full,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum RigSamplePoint {
    Frame {
        frame: u32,
    },
    Time {
        #[serde(rename = "timeSec", alias = "time_sec")]
        time_sec: f32,
    },
    ActionPhase {
        #[serde(rename = "actionId", alias = "action_id")]
        action_id: String,
        phase: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigEvaluationRequest {
    pub actor_id: String,
    pub sample: RigSamplePoint,
    #[serde(default)]
    pub detail: RigReportDetail,
    #[serde(default)]
    pub include_screen_projection: bool,
    #[serde(default)]
    pub include_matrices: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigUnits {
    pub rotation: String,
    pub quaternion_order: String,
    pub position: String,
    pub matrix_layout: String,
}

impl Default for RigUnits {
    fn default() -> Self {
        Self {
            rotation: "degrees".into(),
            quaternion_order: "xyzw".into(),
            position: "motionloomSceneUnits".into(),
            matrix_layout: "columnMajor".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RigEvaluationCapabilities {
    pub model_global_pose: bool,
    pub action_execution: bool,
    pub retarget_driver: bool,
    pub axis_effectiveness: bool,
    pub post_constraints: bool,
    pub post_contact: bool,
    pub screen_projection: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RigAssetProvenance {
    pub id: Option<String>,
    pub resolved_source: Option<String>,
    pub sha256: Option<String>,
    pub skin_joint_count: Option<usize>,
    pub animation_clip_count: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RigProfileProvenance {
    pub id: Option<String>,
    pub preset: Option<String>,
    pub fingerprint: Option<String>,
    pub mapping_count: usize,
    pub axis_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RigActionProvenance {
    pub id: Option<String>,
    pub fingerprint: Option<String>,
    pub duration_sec: Option<f32>,
    pub pose_count: usize,
    pub contact_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RigProvenance {
    pub document: Option<String>,
    pub actor_id: String,
    pub model_asset: RigAssetProvenance,
    pub profile: RigProfileProvenance,
    pub action: RigActionProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ActionDriver {
    EmbeddedGlbClip,
    ExternalGlbAction,
    LegacySemanticAxes,
    BakedHumanoidReference,
    AdditiveAction,
    NativeRigAction,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppliedActionTrace {
    pub action_id: String,
    pub target: String,
    pub authored_start_sec: f32,
    pub authored_duration_sec: Option<f32>,
    pub active: bool,
    pub inactive_reason: Option<String>,
    pub looped: bool,
    pub local_time_sec: Option<f32>,
    pub normalized_phase: Option<f32>,
    pub speed: f32,
    pub blend_weight: f32,
    pub mode: String,
    pub mask: Vec<String>,
    pub root_motion: Option<String>,
    pub driver: ActionDriver,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ActionExecutionTrace {
    pub selected_controller_action: Option<String>,
    pub active_actions: Vec<AppliedActionTrace>,
    pub inactive_actions: Vec<AppliedActionTrace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BonePoseStage {
    ModelRest,
    ProfileCalibratedRest,
    ActionSource,
    Retargeted,
    Layered,
    PostIk,
    PostConstraint,
    PostContact,
    FinalScene,
    ScreenProjected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BoneDriver {
    BindPose,
    RestPoseCalibration,
    LegacyBoneAxis,
    BakedReferenceRetarget,
    EmbeddedClip,
    ExternalClipRetarget,
    OverrideAction,
    AdditiveAction,
    TwoBoneIk,
    ContactCorrection,
    FootLock,
    SceneConstraint,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ScreenProjection {
    pub x: f32,
    pub y: f32,
    pub depth: f32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoneStageTransform {
    pub stage: BonePoseStage,
    pub space: String,
    pub position: Option<[f32; 3]>,
    pub rotation_quaternion: Option<[f32; 4]>,
    pub matrix: Option<[f32; 16]>,
    pub screen: Option<ScreenProjection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AxisEffectiveness {
    pub declared: BTreeMap<String, String>,
    pub effective: BTreeMap<String, bool>,
    pub applied_at_stage: Option<BonePoseStage>,
    pub bypassed_by: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RigDiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigDiagnostic {
    pub severity: RigDiagnosticSeverity,
    pub code: String,
    pub message: String,
    pub bone: Option<String>,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoneEvaluation {
    pub canonical_bone: String,
    pub target_node: Option<String>,
    pub node_index: Option<usize>,
    pub parent_bone: Option<String>,
    pub mapped: bool,
    pub driver: BoneDriver,
    pub stages: Vec<BoneStageTransform>,
    pub axis: AxisEffectiveness,
    pub diagnostics: Vec<RigDiagnostic>,
}

impl BoneEvaluation {
    pub fn stage(&self, stage: BonePoseStage) -> Option<&BoneStageTransform> {
        self.stages.iter().find(|sample| sample.stage == stage)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct FootContactTrace {
    pub contact_active: bool,
    pub authored_contact_window: Option<[f32; 2]>,
    pub position_before: Option<[f32; 3]>,
    pub position_after: Option<[f32; 3]>,
    pub correction_distance: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContactEvaluation {
    pub available: bool,
    pub selected_ground: Option<String>,
    pub ground_kind: Option<String>,
    pub controller_action: Option<String>,
    pub contact_correction_enabled: bool,
    pub foot_lock_enabled: bool,
    pub root_before: Option<[f32; 3]>,
    pub root_after: Option<[f32; 3]>,
    pub root_correction: Option<[f32; 3]>,
    pub feet: BTreeMap<String, FootContactTrace>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigEvaluationReport {
    pub schema_version: String,
    pub engine_version: String,
    pub units: RigUnits,
    pub sample: RigSamplePoint,
    pub frame: u32,
    pub fps: f32,
    pub time_sec: f32,
    pub actor_id: String,
    pub body_height: Option<f32>,
    pub provenance: RigProvenance,
    pub capabilities: RigEvaluationCapabilities,
    pub action_execution: ActionExecutionTrace,
    pub contact_evaluation: ContactEvaluation,
    pub bones: Vec<BoneEvaluation>,
    pub diagnostics: Vec<RigDiagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum RigAlignmentMode {
    Frame,
    Time,
    #[default]
    ActionPhase,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RigAlignment {
    pub mode: RigAlignmentMode,
    pub action_id: Option<String>,
    pub reference_frame: u32,
    pub candidate_frame: u32,
    pub reference_phase: Option<f32>,
    pub candidate_phase: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigComparisonOptions {
    #[serde(default)]
    pub alignment_mode: RigAlignmentMode,
    pub action_id: Option<String>,
    #[serde(default = "default_phase_tolerance")]
    pub phase_tolerance: f32,
    #[serde(default = "default_rotation_warning")]
    pub rotation_warning_deg: f32,
    #[serde(default = "default_rotation_error")]
    pub rotation_error_deg: f32,
    #[serde(default = "default_endpoint_warning")]
    pub endpoint_warning_body_ratio: f32,
    #[serde(default = "default_endpoint_error")]
    pub endpoint_error_body_ratio: f32,
}

fn default_rotation_warning() -> f32 {
    5.0
}
fn default_phase_tolerance() -> f32 {
    0.02
}
fn default_rotation_error() -> f32 {
    15.0
}
fn default_endpoint_warning() -> f32 {
    0.02
}
fn default_endpoint_error() -> f32 {
    0.05
}

impl Default for RigComparisonOptions {
    fn default() -> Self {
        Self {
            alignment_mode: RigAlignmentMode::ActionPhase,
            action_id: None,
            phase_tolerance: default_phase_tolerance(),
            rotation_warning_deg: default_rotation_warning(),
            rotation_error_deg: default_rotation_error(),
            endpoint_warning_body_ratio: default_endpoint_warning(),
            endpoint_error_body_ratio: default_endpoint_error(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RigComparisonStatus {
    Match,
    Warning,
    Error,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RigDifferenceCause {
    AssetMismatch,
    ProfileMismatch,
    MappingMismatch,
    ActionContentMismatch,
    ActionTimingMismatch,
    BlendStackMismatch,
    RetargetMismatch,
    AxisCalibrationMismatch,
    ConstraintMismatch,
    ContactSolverMismatch,
    RootMotionMismatch,
    CameraOnlyDifference,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoneDifference {
    pub bone: String,
    pub status: RigComparisonStatus,
    pub local_angular_error_deg: Option<f32>,
    pub global_angular_error_deg: Option<f32>,
    pub joint_position_error: Option<f32>,
    pub endpoint_error: Option<f32>,
    pub endpoint_error_body_ratio: Option<f32>,
    pub first_divergence: Option<BonePoseStage>,
    pub reference_driver: Option<BoneDriver>,
    pub candidate_driver: Option<BoneDriver>,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RigComparisonSummary {
    pub compared_bones: usize,
    pub matching_bones: usize,
    pub warning_bones: usize,
    pub error_bones: usize,
    pub missing_bones: usize,
    pub mean_angular_error_deg: f32,
    pub max_angular_error_deg: f32,
    pub mean_endpoint_error_body_ratio: f32,
    pub first_divergence: Option<BonePoseStage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootCauseAssessment {
    pub category: RigDifferenceCause,
    pub confidence: f32,
    pub first_divergence_stage: Option<BonePoseStage>,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigRecommendation {
    pub priority: u32,
    pub kind: String,
    pub target: Option<String>,
    pub message: String,
    pub suggested_attribute: Option<String>,
    pub current: Option<String>,
    pub proposed: Option<String>,
    pub confidence: f32,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigComparisonReport {
    pub schema_version: String,
    pub engine_version: String,
    pub status: RigComparisonStatus,
    pub alignment: RigAlignment,
    pub summary: RigComparisonSummary,
    pub root_cause: RootCauseAssessment,
    pub bones: Vec<BoneDifference>,
    pub recommendations: Vec<RigRecommendation>,
    pub diagnostics: Vec<RigDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigCalibrationSuggestion {
    pub applicable: bool,
    pub kind: String,
    pub bone: Option<String>,
    pub attribute: Option<String>,
    pub current: Option<String>,
    pub proposed: Option<String>,
    pub expected_improvement: Option<String>,
    pub reason: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigCalibrationProposal {
    pub schema_version: String,
    pub read_only: bool,
    pub source_comparison_status: RigComparisonStatus,
    pub suggestions: Vec<RigCalibrationSuggestion>,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn fingerprint_serializable<T: Serialize>(value: &T) -> Option<String> {
    serde_json::to_vec(value)
        .ok()
        .map(|bytes| format!("sha256:{}", sha256_hex(&bytes)))
}

pub(crate) fn fingerprint_scene_action(action: &crate::scene::dsl::ActionNode) -> Option<String> {
    let mut semantic = action.clone();
    semantic.id.clear();
    fingerprint_serializable(&semantic)
}

pub(crate) fn fingerprint_scene_profile(
    profile: &crate::scene::dsl::ModelProfileNode,
) -> Option<String> {
    let mut semantic = profile.clone();
    semantic.id.clear();
    semantic.model = None;
    fingerprint_serializable(&semantic)
}

pub fn rig_evaluation_report_json(report: &RigEvaluationReport) -> String {
    serde_json::to_string_pretty(report).expect("rig evaluation reports are serializable")
}

pub fn rig_comparison_report_json(report: &RigComparisonReport) -> String {
    serde_json::to_string_pretty(report).expect("rig comparison reports are serializable")
}

/// Return the versioned machine-readable envelope used by Rust, CLI and WASM
/// hosts. Concrete DTOs remain authoritative for every nested property.
pub fn rig_diagnostics_schema_json() -> String {
    serde_json::to_string_pretty(&serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://motionloom.dev/schema/rig-diagnostics-v1.json",
        "schemaVersion": RIG_DIAGNOSTICS_SCHEMA_VERSION,
        "evaluationRequest": {
            "type": "object",
            "required": ["actorId", "sample"],
            "properties": {
                "actorId": { "type": "string", "minLength": 1 },
                "sample": {
                    "oneOf": [
                        { "type": "object", "required": ["kind", "frame"], "properties": { "kind": { "const": "frame" }, "frame": { "type": "integer", "minimum": 0 } } },
                        { "type": "object", "required": ["kind", "timeSec"], "properties": { "kind": { "const": "time" }, "timeSec": { "type": "number", "minimum": 0 } } },
                        { "type": "object", "required": ["kind", "actionId", "phase"], "properties": { "kind": { "const": "actionPhase" }, "actionId": { "type": "string", "minLength": 1 }, "phase": { "type": "number", "minimum": 0, "maximum": 1 } } }
                    ]
                },
                "detail": { "enum": ["summary", "body", "full"], "default": "body" },
                "includeScreenProjection": { "type": "boolean", "default": false },
                "includeMatrices": { "type": "boolean", "default": false }
            }
        },
        "evaluationReport": {
            "type": "object",
            "required": ["schemaVersion", "engineVersion", "sample", "frame", "fps", "timeSec", "actorId", "provenance", "capabilities", "actionExecution", "contactEvaluation", "bones", "diagnostics"]
        },
        "comparisonOptions": {
            "type": "object",
            "properties": {
                "alignmentMode": { "enum": ["frame", "time", "actionPhase"], "default": "actionPhase" },
                "actionId": { "type": ["string", "null"] },
                "phaseTolerance": { "type": "number", "minimum": 0, "default": 0.02 },
                "rotationWarningDeg": { "type": "number", "minimum": 0, "default": 5.0 },
                "rotationErrorDeg": { "type": "number", "minimum": 0, "default": 15.0 },
                "endpointWarningBodyRatio": { "type": "number", "minimum": 0, "default": 0.02 },
                "endpointErrorBodyRatio": { "type": "number", "minimum": 0, "default": 0.05 }
            }
        },
        "comparisonReport": {
            "type": "object",
            "required": ["schemaVersion", "engineVersion", "status", "alignment", "summary", "rootCause", "bones", "recommendations", "diagnostics"]
        },
        "calibrationProposal": {
            "type": "object",
            "required": ["schemaVersion", "readOnly", "sourceComparisonStatus", "suggestions"],
            "properties": { "readOnly": { "const": true } }
        }
    }))
    .expect("rig diagnostic schema is serializable")
}

pub fn matrix_position(matrix: [f32; 16]) -> [f32; 3] {
    [matrix[12], matrix[13], matrix[14]]
}

pub fn matrix_rotation_quaternion(matrix: [f32; 16]) -> [f32; 4] {
    let x = [matrix[0], matrix[1], matrix[2]];
    let y = [matrix[4], matrix[5], matrix[6]];
    let z = [matrix[8], matrix[9], matrix[10]];
    let normalize = |value: [f32; 3]| {
        let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2])
            .sqrt()
            .max(f32::EPSILON);
        [value[0] / length, value[1] / length, value[2] / length]
    };
    let x = normalize(x);
    let y = normalize(y);
    let z = normalize(z);
    let m00 = x[0];
    let m01 = y[0];
    let m02 = z[0];
    let m10 = x[1];
    let m11 = y[1];
    let m12 = z[1];
    let m20 = x[2];
    let m21 = y[2];
    let m22 = z[2];
    let trace = m00 + m11 + m22;
    let quaternion = if trace > 0.0 {
        let scale = (trace + 1.0).sqrt() * 2.0;
        [
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
            0.25 * scale,
        ]
    } else if m00 > m11 && m00 > m22 {
        let scale = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
            (m21 - m12) / scale,
        ]
    } else if m11 > m22 {
        let scale = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
            (m02 - m20) / scale,
        ]
    } else {
        let scale = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
            (m10 - m01) / scale,
        ]
    };
    normalize_quaternion(quaternion)
}

fn normalize_quaternion(value: [f32; 4]) -> [f32; 4] {
    let length = value.iter().map(|part| part * part).sum::<f32>().sqrt();
    if length <= f32::EPSILON || !length.is_finite() {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        value.map(|part| part / length)
    }
}

pub fn quaternion_angular_error_deg(a: [f32; 4], b: [f32; 4]) -> f32 {
    let a = normalize_quaternion(a);
    let b = normalize_quaternion(b);
    let dot = a
        .iter()
        .zip(b)
        .map(|(left, right)| left * right)
        .sum::<f32>()
        .abs()
        .clamp(0.0, 1.0);
    if dot >= 1.0 - 1.0e-6 {
        0.0
    } else {
        (2.0 * dot.acos()).to_degrees()
    }
}

fn vec3_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn quaternion_conjugate(value: [f32; 4]) -> [f32; 4] {
    [-value[0], -value[1], -value[2], value[3]]
}

fn quaternion_multiply(left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
    normalize_quaternion([
        left[3] * right[0] + left[0] * right[3] + left[1] * right[2] - left[2] * right[1],
        left[3] * right[1] - left[0] * right[2] + left[1] * right[3] + left[2] * right[0],
        left[3] * right[2] + left[0] * right[1] - left[1] * right[0] + left[2] * right[3],
        left[3] * right[3] - left[0] * right[0] - left[1] * right[1] - left[2] * right[2],
    ])
}

fn comparable_rotation(bone: &BoneEvaluation) -> Option<[f32; 4]> {
    comparable_stage(bone).and_then(|stage| stage.rotation_quaternion)
}

fn local_rotation(
    bone: &BoneEvaluation,
    bones: &BTreeMap<&str, &BoneEvaluation>,
) -> Option<[f32; 4]> {
    let rotation = comparable_rotation(bone)?;
    let Some(parent) = bone
        .parent_bone
        .as_deref()
        .and_then(|parent| bones.get(parent))
    else {
        return Some(rotation);
    };
    let parent_rotation = comparable_rotation(parent)?;
    Some(quaternion_multiply(
        quaternion_conjugate(parent_rotation),
        rotation,
    ))
}

fn endpoint_error(
    bone_name: &str,
    reference_bones: &BTreeMap<&str, &BoneEvaluation>,
    candidate_bones: &BTreeMap<&str, &BoneEvaluation>,
) -> Option<f32> {
    let child_errors = reference_bones
        .values()
        .filter(|bone| bone.parent_bone.as_deref() == Some(bone_name))
        .filter_map(|left_child| {
            let right_child = candidate_bones.get(left_child.canonical_bone.as_str())?;
            comparable_stage(left_child)
                .and_then(|stage| stage.position)
                .zip(comparable_stage(right_child).and_then(|stage| stage.position))
                .map(|(left, right)| vec3_distance(left, right))
        })
        .collect::<Vec<_>>();
    child_errors.into_iter().reduce(f32::max)
}

fn selected_action_phase(report: &RigEvaluationReport, action_id: Option<&str>) -> Option<f32> {
    report
        .action_execution
        .active_actions
        .iter()
        .find(|action| action_id.is_none_or(|id| action.action_id == id))
        .and_then(|action| action.normalized_phase)
}

fn comparable_stage(bone: &BoneEvaluation) -> Option<&BoneStageTransform> {
    [
        BonePoseStage::FinalScene,
        BonePoseStage::PostContact,
        BonePoseStage::PostConstraint,
        BonePoseStage::Retargeted,
        BonePoseStage::ModelRest,
    ]
    .into_iter()
    .find_map(|stage| bone.stage(stage))
}

fn first_divergence(
    reference: &BoneEvaluation,
    candidate: &BoneEvaluation,
    rotation_warning: f32,
    position_warning: f32,
) -> Option<BonePoseStage> {
    for stage in [
        BonePoseStage::ModelRest,
        BonePoseStage::ProfileCalibratedRest,
        BonePoseStage::ActionSource,
        BonePoseStage::Retargeted,
        BonePoseStage::Layered,
        BonePoseStage::PostIk,
        BonePoseStage::PostConstraint,
        BonePoseStage::PostContact,
        BonePoseStage::FinalScene,
        BonePoseStage::ScreenProjected,
    ] {
        let (Some(left), Some(right)) = (reference.stage(stage), candidate.stage(stage)) else {
            continue;
        };
        let rotation_differs = left
            .rotation_quaternion
            .zip(right.rotation_quaternion)
            .is_some_and(|(a, b)| quaternion_angular_error_deg(a, b) > rotation_warning);
        let position_differs = left
            .position
            .zip(right.position)
            .is_some_and(|(a, b)| vec3_distance(a, b) > position_warning);
        if rotation_differs || position_differs {
            return Some(stage);
        }
    }
    None
}

pub fn compare_humanoid_poses(
    reference: &RigEvaluationReport,
    candidate: &RigEvaluationReport,
    options: RigComparisonOptions,
) -> RigComparisonReport {
    let reference_bones = reference
        .bones
        .iter()
        .map(|bone| (bone.canonical_bone.as_str(), bone))
        .collect::<BTreeMap<_, _>>();
    let candidate_bones = candidate
        .bones
        .iter()
        .map(|bone| (bone.canonical_bone.as_str(), bone))
        .collect::<BTreeMap<_, _>>();
    let names = reference_bones
        .keys()
        .chain(candidate_bones.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    let body_height = reference
        .body_height
        .or(candidate.body_height)
        .unwrap_or(1.0)
        .max(0.0001);
    let mut bones = Vec::with_capacity(names.len());
    for name in names {
        let (Some(left), Some(right)) = (reference_bones.get(name), candidate_bones.get(name))
        else {
            bones.push(BoneDifference {
                bone: name.into(),
                status: RigComparisonStatus::Missing,
                local_angular_error_deg: None,
                global_angular_error_deg: None,
                joint_position_error: None,
                endpoint_error: None,
                endpoint_error_body_ratio: None,
                first_divergence: Some(BonePoseStage::ModelRest),
                reference_driver: reference_bones.get(name).map(|bone| bone.driver),
                candidate_driver: candidate_bones.get(name).map(|bone| bone.driver),
                evidence: vec!["The canonical bone is absent from one report.".into()],
            });
            continue;
        };
        let left_stage = comparable_stage(left);
        let right_stage = comparable_stage(right);
        let angular = left_stage
            .and_then(|stage| stage.rotation_quaternion)
            .zip(right_stage.and_then(|stage| stage.rotation_quaternion))
            .map(|(a, b)| quaternion_angular_error_deg(a, b));
        let local_angular = local_rotation(left, &reference_bones)
            .zip(local_rotation(right, &candidate_bones))
            .map(|(a, b)| quaternion_angular_error_deg(a, b));
        let position = left_stage
            .and_then(|stage| stage.position)
            .zip(right_stage.and_then(|stage| stage.position))
            .map(|(a, b)| vec3_distance(a, b));
        let endpoint = endpoint_error(name, &reference_bones, &candidate_bones).or(position);
        let ratio = endpoint.map(|value| value / body_height);
        let status = if angular.is_some_and(|value| value >= options.rotation_error_deg)
            || ratio.is_some_and(|value| value >= options.endpoint_error_body_ratio)
        {
            RigComparisonStatus::Error
        } else if angular.is_some_and(|value| value >= options.rotation_warning_deg)
            || ratio.is_some_and(|value| value >= options.endpoint_warning_body_ratio)
        {
            RigComparisonStatus::Warning
        } else {
            RigComparisonStatus::Match
        };
        let divergence = first_divergence(
            left,
            right,
            options.rotation_warning_deg,
            options.endpoint_warning_body_ratio * body_height,
        );
        let mut evidence = Vec::new();
        if left.driver != right.driver {
            evidence.push(format!(
                "Bone drivers differ: {:?} vs {:?}.",
                left.driver, right.driver
            ));
        }
        bones.push(BoneDifference {
            bone: name.into(),
            status,
            local_angular_error_deg: local_angular,
            global_angular_error_deg: angular,
            joint_position_error: position,
            endpoint_error: endpoint,
            endpoint_error_body_ratio: ratio,
            first_divergence: divergence,
            reference_driver: Some(left.driver),
            candidate_driver: Some(right.driver),
            evidence,
        });
    }

    let angular_values = bones
        .iter()
        .filter_map(|bone| bone.global_angular_error_deg)
        .collect::<Vec<_>>();
    let endpoint_values = bones
        .iter()
        .filter_map(|bone| bone.endpoint_error_body_ratio)
        .collect::<Vec<_>>();
    let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len().max(1) as f32;
    let first_stage = bones.iter().filter_map(|bone| bone.first_divergence).min();
    let summary = RigComparisonSummary {
        compared_bones: bones.len(),
        matching_bones: bones
            .iter()
            .filter(|bone| bone.status == RigComparisonStatus::Match)
            .count(),
        warning_bones: bones
            .iter()
            .filter(|bone| bone.status == RigComparisonStatus::Warning)
            .count(),
        error_bones: bones
            .iter()
            .filter(|bone| bone.status == RigComparisonStatus::Error)
            .count(),
        missing_bones: bones
            .iter()
            .filter(|bone| bone.status == RigComparisonStatus::Missing)
            .count(),
        mean_angular_error_deg: mean(&angular_values),
        max_angular_error_deg: angular_values.iter().copied().fold(0.0, f32::max),
        mean_endpoint_error_body_ratio: mean(&endpoint_values),
        first_divergence: first_stage,
    };
    let status = if summary.error_bones > 0 || summary.missing_bones > 0 {
        RigComparisonStatus::Error
    } else if summary.warning_bones > 0 {
        RigComparisonStatus::Warning
    } else {
        RigComparisonStatus::Match
    };
    let action_id = options.action_id.as_deref();
    let reference_phase = selected_action_phase(reference, action_id);
    let candidate_phase = selected_action_phase(candidate, action_id);
    let alignment = RigAlignment {
        mode: options.alignment_mode,
        action_id: options.action_id.clone(),
        reference_frame: reference.frame,
        candidate_frame: candidate.frame,
        reference_phase,
        candidate_phase,
    };
    let root_cause = classify_root_cause(
        reference,
        candidate,
        &alignment,
        &summary,
        &bones,
        options.phase_tolerance.max(0.0),
    );
    let recommendations =
        recommendations_for(&root_cause, &bones, reference_phase, candidate_phase);
    RigComparisonReport {
        schema_version: RIG_DIAGNOSTICS_SCHEMA_VERSION.into(),
        engine_version: env!("CARGO_PKG_VERSION").into(),
        status,
        alignment,
        summary,
        root_cause,
        bones,
        recommendations,
        diagnostics: Vec::new(),
    }
}

fn classify_root_cause(
    reference: &RigEvaluationReport,
    candidate: &RigEvaluationReport,
    alignment: &RigAlignment,
    summary: &RigComparisonSummary,
    bones: &[BoneDifference],
    phase_tolerance: f32,
) -> RootCauseAssessment {
    let mut evidence = Vec::new();
    let asset_hash_mismatch = reference
        .provenance
        .model_asset
        .sha256
        .as_ref()
        .zip(candidate.provenance.model_asset.sha256.as_ref())
        .is_some_and(|(left, right)| left != right);
    let mapping_mismatch = reference.provenance.profile.mapping_count
        != candidate.provenance.profile.mapping_count
        || reference
            .bones
            .iter()
            .map(|bone| bone.canonical_bone.as_str())
            .collect::<BTreeSet<_>>()
            != candidate
                .bones
                .iter()
                .map(|bone| bone.canonical_bone.as_str())
                .collect::<BTreeSet<_>>();
    let active_stack = |report: &RigEvaluationReport| {
        report
            .action_execution
            .active_actions
            .iter()
            .map(|action| {
                (
                    action.action_id.clone(),
                    action.mode.clone(),
                    action.mask.clone(),
                    (action.blend_weight * 10_000.0).round() as i32,
                )
            })
            .collect::<Vec<_>>()
    };
    let axis_mismatch = reference.bones.iter().any(|left| {
        candidate
            .bones
            .iter()
            .find(|right| right.canonical_bone == left.canonical_bone)
            .is_some_and(|right| left.axis != right.axis)
    });
    let driver_mismatch = bones
        .iter()
        .any(|bone| bone.reference_driver != bone.candidate_driver);
    let contact_mismatch = reference.contact_evaluation != candidate.contact_evaluation
        || reference.capabilities.post_contact != candidate.capabilities.post_contact;
    let model_pose_matches =
        summary.error_bones == 0 && summary.warning_bones == 0 && summary.missing_bones == 0;
    let screen_differs = reference.bones.iter().any(|left| {
        candidate
            .bones
            .iter()
            .find(|right| right.canonical_bone == left.canonical_bone)
            .and_then(|right| {
                left.stage(BonePoseStage::ScreenProjected)
                    .and_then(|stage| stage.screen.as_ref())
                    .zip(
                        right
                            .stage(BonePoseStage::ScreenProjected)
                            .and_then(|stage| stage.screen.as_ref()),
                    )
            })
            .is_some_and(|(left, right)| {
                (left.x - right.x).abs() > 0.5
                    || (left.y - right.y).abs() > 0.5
                    || (left.depth - right.depth).abs() > 0.001
            })
    });
    let (category, confidence) = if asset_hash_mismatch {
        evidence.push("Resolved model SHA-256 values differ.".into());
        (RigDifferenceCause::AssetMismatch, 1.0)
    } else if mapping_mismatch {
        evidence.push(format!(
            "Canonical coverage/profile mapping counts differ: reference={} candidate={}.",
            reference.provenance.profile.mapping_count, candidate.provenance.profile.mapping_count
        ));
        (RigDifferenceCause::MappingMismatch, 0.99)
    } else if reference.provenance.profile.fingerprint != candidate.provenance.profile.fingerprint {
        evidence.push("ModelProfile fingerprints differ.".into());
        (RigDifferenceCause::ProfileMismatch, 0.99)
    } else if reference.provenance.action.fingerprint != candidate.provenance.action.fingerprint {
        evidence.push("Action fingerprints differ.".into());
        (RigDifferenceCause::ActionContentMismatch, 0.99)
    } else if alignment
        .reference_phase
        .zip(alignment.candidate_phase)
        .is_some_and(|(left, right)| (left - right).abs() > phase_tolerance)
    {
        evidence.push("The active Action phases differ after selecting the same Action id.".into());
        (RigDifferenceCause::ActionTimingMismatch, 0.98)
    } else if active_stack(reference) != active_stack(candidate) {
        evidence.push("The active Action ids, modes, masks, or blend weights differ.".into());
        (RigDifferenceCause::BlendStackMismatch, 0.96)
    } else if axis_mismatch {
        evidence.push("Declared or effective BoneAxis channels differ.".into());
        (RigDifferenceCause::AxisCalibrationMismatch, 0.95)
    } else if driver_mismatch {
        evidence
            .push("At least one canonical bone is controlled by a different pose driver.".into());
        (RigDifferenceCause::RetargetMismatch, 0.94)
    } else if contact_mismatch {
        evidence.push("Scene contact/ground solver inputs or outputs differ.".into());
        (RigDifferenceCause::ContactSolverMismatch, 0.93)
    } else if bones
        .iter()
        .any(|bone| bone.first_divergence == Some(BonePoseStage::Retargeted))
    {
        evidence.push("The first measurable bone divergence appears at retargeting.".into());
        (RigDifferenceCause::RetargetMismatch, 0.92)
    } else if bones
        .iter()
        .any(|bone| bone.first_divergence == Some(BonePoseStage::PostContact))
    {
        evidence.push("The first measurable bone divergence appears after scene contact.".into());
        (RigDifferenceCause::ContactSolverMismatch, 0.92)
    } else if model_pose_matches && screen_differs {
        evidence.push("Model-space bones match, but screen projections differ.".into());
        (RigDifferenceCause::CameraOnlyDifference, 0.75)
    } else if model_pose_matches {
        evidence.push(
            "All comparable bone transforms and screen projections are within tolerance.".into(),
        );
        (RigDifferenceCause::Unknown, 1.0)
    } else {
        evidence.push("Available stages do not isolate one deterministic cause.".into());
        (RigDifferenceCause::Unknown, 0.35)
    };
    RootCauseAssessment {
        category,
        confidence,
        first_divergence_stage: summary.first_divergence,
        evidence,
    }
}

fn recommendations_for(
    cause: &RootCauseAssessment,
    bones: &[BoneDifference],
    reference_phase: Option<f32>,
    candidate_phase: Option<f32>,
) -> Vec<RigRecommendation> {
    let mut recommendations = Vec::new();
    if cause.category == RigDifferenceCause::MappingMismatch {
        recommendations.push(RigRecommendation {
            priority: 1,
            kind: "alignModelProfileMapping".into(),
            target: None,
            message: "Align the explicit canonical retarget mapping before tuning bone axes."
                .into(),
            suggested_attribute: Some("ModelProfile.Retarget".into()),
            current: None,
            proposed: Some("matchReferenceCanonicalMapping".into()),
            confidence: cause.confidence,
            evidence: cause.evidence.clone(),
        });
    } else if cause.category == RigDifferenceCause::ProfileMismatch {
        recommendations.push(RigRecommendation {
            priority: 1,
            kind: "alignModelProfile".into(),
            target: None,
            message: "Compare semantic ModelProfile contents before editing individual bones."
                .into(),
            suggested_attribute: Some("ModelProfile".into()),
            current: None,
            proposed: Some("matchReferenceProfile".into()),
            confidence: cause.confidence,
            evidence: cause.evidence.clone(),
        });
    }
    if cause.category == RigDifferenceCause::ActionTimingMismatch {
        recommendations.push(RigRecommendation {
            priority: 1,
            kind: "alignActionPhase".into(),
            target: None,
            message: "Align the candidate Action phase before changing rig calibration.".into(),
            suggested_attribute: Some("at/speed".into()),
            current: candidate_phase.map(|value| format!("phase:{value:.6}")),
            proposed: reference_phase.map(|value| format!("phase:{value:.6}")),
            confidence: cause.confidence,
            evidence: cause.evidence.clone(),
        });
    }
    for bone in bones
        .iter()
        .filter(|bone| {
            matches!(
                bone.status,
                RigComparisonStatus::Error | RigComparisonStatus::Warning
            )
        })
        .take(8)
    {
        recommendations.push(RigRecommendation {
            priority: recommendations.len() as u32 + 1,
            kind: "inspectFirstDivergence".into(),
            target: Some(bone.bone.clone()),
            message: format!(
                "Inspect '{}' at {:?} before editing its axis map.",
                bone.bone, bone.first_divergence
            ),
            suggested_attribute: None,
            current: bone
                .global_angular_error_deg
                .map(|value| format!("{value:.3}deg")),
            proposed: Some("withinTolerance".into()),
            confidence: 0.85,
            evidence: bone.evidence.clone(),
        });
    }
    recommendations
}

pub fn propose_rig_calibration(comparison: &RigComparisonReport) -> RigCalibrationProposal {
    let timing_or_stack = matches!(
        comparison.root_cause.category,
        RigDifferenceCause::ActionTimingMismatch | RigDifferenceCause::BlendStackMismatch
    );
    let suggestions = comparison
        .bones
        .iter()
        .filter(|bone| {
            matches!(
                bone.status,
                RigComparisonStatus::Error | RigComparisonStatus::Warning
            )
        })
        .take(12)
        .map(|bone| {
            let bypassed = matches!(
                bone.candidate_driver,
                Some(BoneDriver::BakedReferenceRetarget)
            );
            RigCalibrationSuggestion {
                applicable: !timing_or_stack && !bypassed,
                kind: if bypassed {
                    "doNotChangeBoneAxis"
                } else {
                    "reviewBoneCalibration"
                }
                .into(),
                bone: Some(bone.bone.clone()),
                attribute: (!bypassed).then(|| "BoneAxisMap".into()),
                current: bone
                    .global_angular_error_deg
                    .map(|value| format!("error:{value:.3}deg")),
                proposed: None,
                expected_improvement: None,
                reason: if timing_or_stack {
                    "Resolve Action timing/layering before proposing rig calibration.".into()
                } else if bypassed {
                    "BoneAxisMap does not control the active baked-reference retarget path.".into()
                } else {
                    "The bone differs after timing and asset provenance checks.".into()
                },
                confidence: if bypassed || timing_or_stack {
                    1.0
                } else {
                    0.65
                },
            }
        })
        .collect();
    RigCalibrationProposal {
        schema_version: RIG_DIAGNOSTICS_SCHEMA_VERSION.into(),
        read_only: true,
        source_comparison_status: comparison.status,
        suggestions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(driver: BoneDriver, phase: f32) -> RigEvaluationReport {
        RigEvaluationReport {
            schema_version: RIG_DIAGNOSTICS_SCHEMA_VERSION.into(),
            engine_version: "test".into(),
            units: RigUnits::default(),
            sample: RigSamplePoint::ActionPhase {
                action_id: "walk".into(),
                phase,
            },
            frame: (phase * 30.0).round() as u32,
            fps: 30.0,
            time_sec: phase,
            actor_id: "actor".into(),
            body_height: Some(2.0),
            provenance: RigProvenance {
                document: Some("test.motionloom".into()),
                actor_id: "actor".into(),
                model_asset: RigAssetProvenance {
                    sha256: Some("model".into()),
                    ..RigAssetProvenance::default()
                },
                profile: RigProfileProvenance {
                    fingerprint: Some("profile".into()),
                    mapping_count: 1,
                    ..RigProfileProvenance::default()
                },
                action: RigActionProvenance {
                    id: Some("walk".into()),
                    fingerprint: Some("action".into()),
                    ..RigActionProvenance::default()
                },
            },
            capabilities: RigEvaluationCapabilities {
                model_global_pose: true,
                action_execution: true,
                retarget_driver: true,
                axis_effectiveness: true,
                post_constraints: true,
                post_contact: true,
                screen_projection: true,
            },
            action_execution: ActionExecutionTrace {
                selected_controller_action: Some("walk".into()),
                active_actions: vec![AppliedActionTrace {
                    action_id: "walk".into(),
                    target: "actor".into(),
                    authored_start_sec: 0.0,
                    authored_duration_sec: Some(1.0),
                    active: true,
                    inactive_reason: None,
                    looped: true,
                    local_time_sec: Some(phase),
                    normalized_phase: Some(phase),
                    speed: 1.0,
                    blend_weight: 1.0,
                    mode: "override".into(),
                    mask: Vec::new(),
                    root_motion: None,
                    driver: ActionDriver::BakedHumanoidReference,
                }],
                inactive_actions: Vec::new(),
            },
            contact_evaluation: ContactEvaluation {
                available: true,
                ..ContactEvaluation::default()
            },
            bones: vec![BoneEvaluation {
                canonical_bone: "hips".into(),
                target_node: Some("Hips".into()),
                node_index: Some(0),
                parent_bone: None,
                mapped: true,
                driver,
                stages: vec![
                    BoneStageTransform {
                        stage: BonePoseStage::ModelRest,
                        space: "modelGlobal".into(),
                        position: Some([0.0, 1.0, 0.0]),
                        rotation_quaternion: Some([0.0, 0.0, 0.0, 1.0]),
                        matrix: None,
                        screen: None,
                    },
                    BoneStageTransform {
                        stage: BonePoseStage::FinalScene,
                        space: "modelGlobal".into(),
                        position: Some([0.0, 1.0, 0.0]),
                        rotation_quaternion: Some([0.0, 0.0, 0.0, 1.0]),
                        matrix: None,
                        screen: None,
                    },
                    BoneStageTransform {
                        stage: BonePoseStage::ScreenProjected,
                        space: "authoredPixels".into(),
                        position: None,
                        rotation_quaternion: None,
                        matrix: None,
                        screen: Some(ScreenProjection {
                            x: 320.0,
                            y: 180.0,
                            depth: 4.0,
                            width: 640,
                            height: 360,
                        }),
                    },
                ],
                axis: AxisEffectiveness::default(),
                diagnostics: Vec::new(),
            }],
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn quaternion_sign_does_not_change_angular_error() {
        let value = [0.1, -0.2, 0.3, 0.9];
        let opposite = value.map(|part| -part);
        assert!(quaternion_angular_error_deg(value, opposite) < 0.001);
    }

    #[test]
    fn matrix_identity_extracts_identity_rotation() {
        let matrix = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0,
        ];
        assert_eq!(matrix_position(matrix), [2.0, 3.0, 4.0]);
        assert!(
            quaternion_angular_error_deg(matrix_rotation_quaternion(matrix), [0.0, 0.0, 0.0, 1.0])
                < 0.001
        );
    }

    #[test]
    fn identical_reports_are_a_match_without_camera_claim() {
        let report = report(BoneDriver::LegacyBoneAxis, 0.5);
        let comparison = compare_humanoid_poses(&report, &report, RigComparisonOptions::default());
        assert_eq!(comparison.status, RigComparisonStatus::Match);
        assert_eq!(comparison.root_cause.category, RigDifferenceCause::Unknown);
        assert_eq!(comparison.summary.max_angular_error_deg, 0.0);
    }

    #[test]
    fn screen_only_change_is_classified_as_camera_only() {
        let reference = report(BoneDriver::LegacyBoneAxis, 0.5);
        let mut candidate = reference.clone();
        candidate.bones[0]
            .stages
            .iter_mut()
            .find(|stage| stage.stage == BonePoseStage::ScreenProjected)
            .and_then(|stage| stage.screen.as_mut())
            .expect("screen projection")
            .x += 40.0;
        let comparison =
            compare_humanoid_poses(&reference, &candidate, RigComparisonOptions::default());
        assert_eq!(comparison.status, RigComparisonStatus::Match);
        assert_eq!(
            comparison.root_cause.category,
            RigDifferenceCause::CameraOnlyDifference
        );
    }

    #[test]
    fn phase_mismatch_precedes_calibration_advice() {
        let reference = report(BoneDriver::LegacyBoneAxis, 0.25);
        let candidate = report(BoneDriver::LegacyBoneAxis, 0.75);
        let comparison =
            compare_humanoid_poses(&reference, &candidate, RigComparisonOptions::default());
        assert_eq!(
            comparison.root_cause.category,
            RigDifferenceCause::ActionTimingMismatch
        );
    }

    #[test]
    fn mapping_count_mismatch_is_reported_before_bone_tuning() {
        let reference = report(BoneDriver::BakedReferenceRetarget, 0.5);
        let mut candidate = reference.clone();
        candidate.provenance.profile.mapping_count = 12;
        let comparison =
            compare_humanoid_poses(&reference, &candidate, RigComparisonOptions::default());
        assert_eq!(
            comparison.root_cause.category,
            RigDifferenceCause::MappingMismatch
        );
        assert_eq!(
            comparison.recommendations[0].kind,
            "alignModelProfileMapping"
        );
    }

    #[test]
    fn baked_reference_driver_blocks_axis_calibration_proposal() {
        let reference = report(BoneDriver::BakedReferenceRetarget, 0.5);
        let mut candidate = reference.clone();
        candidate.bones[0]
            .stages
            .iter_mut()
            .find(|stage| stage.stage == BonePoseStage::FinalScene)
            .expect("final stage")
            .rotation_quaternion = Some([0.0, 0.258_819, 0.0, 0.965_926]);
        let comparison =
            compare_humanoid_poses(&reference, &candidate, RigComparisonOptions::default());
        let proposal = propose_rig_calibration(&comparison);
        assert!(!proposal.suggestions[0].applicable);
        assert_eq!(proposal.suggestions[0].kind, "doNotChangeBoneAxis");
    }

    #[test]
    fn report_json_contains_only_finite_test_values() {
        let json = rig_evaluation_report_json(&report(BoneDriver::LegacyBoneAxis, 0.5));
        assert!(!json.contains("NaN"));
        assert!(!json.contains("Infinity"));
        let decoded: RigEvaluationReport = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(decoded.actor_id, "actor");
    }

    #[test]
    fn diagnostic_schema_is_valid_json_and_uses_camel_case_request_fields() {
        let schema = rig_diagnostics_schema_json();
        let value: serde_json::Value = serde_json::from_str(&schema).expect("valid JSON schema");
        assert_eq!(value["schemaVersion"], RIG_DIAGNOSTICS_SCHEMA_VERSION);
        assert!(schema.contains("actionId"));
        assert!(schema.contains("includeScreenProjection"));
    }
}
