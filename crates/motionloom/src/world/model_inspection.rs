// =========================================
// =========================================
// crates/motionloom/src/world/model_inspection.rs

//! Automatic GLB skeleton inspection and humanoid profile proposals.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use super::gltf_loader::{
    GlbLoadError, GlbMeshData, GlbNodeData, load_glb_animation_data, load_glb_mesh_data,
    parse_glb_animation_data, parse_glb_mesh_data, parse_gltf_json_value,
};
use super::{WorldAction, WorldModelProfile};

const REPORT_VERSION: u32 = 1;
const EPSILON: f32 = 1.0e-5;

#[derive(Debug, Error)]
pub enum ModelInspectionError {
    #[error(transparent)]
    Glb(#[from] GlbLoadError),
    #[error("GLB {asset} has no skin or skeleton joints")]
    MissingSkeleton { asset: String },
    #[error("failed to serialize skeleton inspection report: {source}")]
    Serialize {
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlbSkeletonInspectionReport {
    pub version: u32,
    pub asset: String,
    pub skin_joint_count: usize,
    pub animation_clip_count: usize,
    pub rest_pose: RestPoseProposal,
    pub body_basis: BodyBasisProposal,
    pub mappings: Vec<HumanoidBoneProposal>,
    pub axes: Vec<BoneAxisProposal>,
    pub unmapped_joints: Vec<String>,
    pub overall_confidence: f32,
    pub manual_review_required: bool,
    pub diagnostics: Vec<ModelInspectionDiagnostic>,
    pub profile_dsl: String,
}

/// Additive rig-family detection returned by the opt-in humanoid profile
/// inspector. The legacy skeleton inspection JSON remains unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectedHumanoidRig {
    pub family: String,
    pub label: String,
    pub confidence: f32,
    pub mapping_source: String,
    pub matched_bone_count: usize,
    pub core_bone_count: usize,
    pub evidence: Vec<String>,
}

/// Rig-aware profile proposal. Flattening preserves the familiar skeleton
/// report shape while adding one `detectedRig` field for newer hosts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlbHumanoidProfileInspectionReport {
    pub detected_rig: DetectedHumanoidRig,
    #[serde(flatten)]
    pub skeleton: GlbSkeletonInspectionReport,
}

#[derive(Debug, Clone)]
struct PreferredBoneMapping {
    node_index: usize,
    confidence: f32,
    evidence: String,
}

/// Semantic proposal for a static 3D environment. This deliberately reuses
/// GLB node names instead of inventing world coordinates, giving LLM authors a
/// stable vocabulary for grounding, destinations and camera placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlbEnvironmentInspectionReport {
    pub version: u32,
    pub asset: String,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub node_count: usize,
    pub node_names: Vec<String>,
    pub mesh_count: usize,
    pub coordinate_profile: EnvironmentCoordinateProfile,
    pub surfaces: Vec<EnvironmentSurfaceProposal>,
    pub anchors: Vec<EnvironmentAnchorProposal>,
    pub overall_confidence: f32,
    pub manual_review_required: bool,
    pub diagnostics: Vec<EnvironmentInspectionDiagnostic>,
    pub environment_dsl: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentCoordinateProfile {
    pub up: [f32; 3],
    pub forward: [f32; 3],
    pub handedness: String,
    pub unit_scale: f32,
    pub normalization_origin: [f32; 3],
    pub normalization_scale: f32,
    pub confidence: f32,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentSurfaceProposal {
    pub id: String,
    pub node: Option<String>,
    pub kind: String,
    pub height: f32,
    #[serde(default = "default_environment_up")]
    pub normal: [f32; 3],
    #[serde(default)]
    pub centroid: [f32; 3],
    #[serde(default)]
    pub bounds_min: [f32; 3],
    #[serde(default)]
    pub bounds_max: [f32; 3],
    #[serde(default)]
    pub area: f32,
    #[serde(default)]
    pub source: String,
    pub confidence: f32,
    pub evidence: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EnvironmentWalkableTriangle {
    pub points: [[f32; 3]; 3],
    pub normal: [f32; 3],
}

#[allow(dead_code)]
pub(crate) fn environment_walkable_triangles(
    mesh: &GlbMeshData,
) -> Vec<EnvironmentWalkableTriangle> {
    environment_collision_triangles(mesh)
        .into_iter()
        .filter(|triangle| triangle.normal[1] >= 0.55)
        .collect()
}

/// Preserve every non-degenerate environment triangle for deterministic
/// character collision. Ground inspection still filters this list separately.
pub(crate) fn environment_collision_triangles(
    mesh: &GlbMeshData,
) -> Vec<EnvironmentWalkableTriangle> {
    let matrices = world_matrices(&mesh.nodes);
    mesh.triangles
        .iter()
        .filter_map(|triangle| {
            let matrix = triangle
                .mesh_node
                .and_then(|index| matrices.get(index))
                .copied()
                .unwrap_or_else(identity);
            let mut points = [[0.0; 3]; 3];
            for (slot, index) in points.iter_mut().zip(triangle.indices) {
                *slot = transform_point(matrix, *mesh.positions.get(index as usize)?);
            }
            let cross = cross3(sub3(points[1], points[0]), sub3(points[2], points[0]));
            let mut normal = normalize3(cross)?;
            if normal[1] < 0.0 {
                normal = scale3(normal, -1.0);
            }
            Some(EnvironmentWalkableTriangle { points, normal })
        })
        .collect()
}

fn default_environment_up() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentAnchorProposal {
    pub id: String,
    pub node: String,
    pub position: [f32; 3],
    pub role: String,
    pub confidence: f32,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentInspectionDiagnostic {
    pub severity: String,
    pub code: String,
    pub message: String,
    pub recommendation: String,
}

pub fn inspect_glb_environment_path(
    path: impl AsRef<Path>,
) -> Result<GlbEnvironmentInspectionReport, ModelInspectionError> {
    let path = path.as_ref();
    let mesh = load_glb_mesh_data(path)?;
    Ok(build_environment_report(&mesh, &path.to_string_lossy()))
}

pub fn inspect_glb_environment_bytes(
    bytes: &[u8],
    asset_label: impl Into<String>,
) -> Result<GlbEnvironmentInspectionReport, ModelInspectionError> {
    let asset = asset_label.into();
    let mesh = parse_glb_mesh_data(Path::new(&asset), bytes)?;
    Ok(build_environment_report(&mesh, &asset))
}

pub fn inspect_glb_environment_json(
    bytes: &[u8],
    asset_label: impl Into<String>,
) -> Result<String, ModelInspectionError> {
    let report = inspect_glb_environment_bytes(bytes, asset_label)?;
    serde_json::to_string_pretty(&report)
        .map_err(|source| ModelInspectionError::Serialize { source })
}

fn build_environment_report(mesh: &GlbMeshData, asset: &str) -> GlbEnvironmentInspectionReport {
    let matrices = world_matrices(&mesh.nodes);
    // Loader bounds are primitive-local. Static environment placement and
    // camera proposals must use the same global node transforms as the GPU
    // renderer, otherwise an imported scene with a rotated/translated root
    // can appear nowhere near the reported coordinates.
    let (bounds_min, bounds_max) = transformed_environment_bounds(mesh, &matrices)
        .unwrap_or((mesh.bounds_min, mesh.bounds_max));
    let coordinate_profile = infer_environment_coordinate_profile(bounds_min, bounds_max);
    let mut surfaces = Vec::new();
    let mut anchors = Vec::new();
    for node in &mesh.nodes {
        let Some(name) = node.name.as_deref() else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        let position = matrices
            .get(node.index)
            .map(|matrix| [matrix[12], matrix[13], matrix[14]])
            .unwrap_or(node.translation);
        let surface_kind = if ["floor", "ground", "roof", "road", "street", "platform"]
            .iter()
            .any(|token| lower.contains(token))
        {
            Some(("ground", 0.92))
        } else if ["rail", "fence", "wall", "obstacle", "ledge"]
            .iter()
            .any(|token| lower.contains(token))
        {
            Some(("obstacle", 0.84))
        } else {
            None
        };
        if let Some((kind, confidence)) = surface_kind {
            surfaces.push(EnvironmentSurfaceProposal {
                id: semantic_id(name),
                node: Some(name.to_string()),
                kind: kind.to_string(),
                height: position[1],
                normal: coordinate_profile.up,
                centroid: position,
                bounds_min: position,
                bounds_max: position,
                area: 0.0,
                source: "named_node".to_string(),
                confidence,
                evidence: format!("GLB node name '{name}' contains a {kind} semantic token"),
            });
        }
        let role = if lower.contains("takeoff") {
            Some(("takeoff", 0.98))
        } else if lower.contains("landing") {
            Some(("landing", 0.98))
        } else if lower.contains("contact") || lower.contains("vault") {
            Some(("contact", 0.94))
        } else if lower.contains("camera") || lower.starts_with("cam_") {
            Some(("camera", 0.9))
        } else if lower.contains("anchor") || lower.starts_with("ml_") {
            Some(("generic", 0.78))
        } else {
            None
        };
        if let Some((role, confidence)) = role {
            anchors.push(EnvironmentAnchorProposal {
                id: semantic_id(name),
                node: name.to_string(),
                position,
                role: role.to_string(),
                confidence,
                evidence: format!("GLB node name '{name}' indicates a {role} marker"),
            });
        }
    }
    let mut diagnostics = Vec::new();
    if surfaces.is_empty() {
        surfaces = detect_horizontal_environment_surfaces(mesh, &matrices, bounds_min, bounds_max);
    }
    if surfaces.is_empty() {
        surfaces.push(EnvironmentSurfaceProposal {
            id: "auto_ground".to_string(),
            node: None,
            kind: "ground".to_string(),
            height: bounds_min[1],
            normal: coordinate_profile.up,
            centroid: [
                (bounds_min[0] + bounds_max[0]) * 0.5,
                bounds_min[1],
                (bounds_min[2] + bounds_max[2]) * 0.5,
            ],
            bounds_min,
            bounds_max: [bounds_max[0], bounds_min[1], bounds_max[2]],
            area: 0.0,
            source: "bounds_fallback".to_string(),
            confidence: 0.35,
            evidence: "No horizontal geometry candidate was found; using the mesh lower bound"
                .to_string(),
        });
        diagnostics.push(EnvironmentInspectionDiagnostic {
            severity: "warning".to_string(),
            code: "ENVIRONMENT_GROUND_INFERRED_FROM_BOUNDS".to_string(),
            message: "No named or geometry-derived horizontal ground was found.".to_string(),
            recommendation: "Add a GLB empty named ML_Ground or author an explicit Surface."
                .to_string(),
        });
    } else if surfaces.iter().all(|surface| surface.node.is_none()) {
        diagnostics.push(EnvironmentInspectionDiagnostic {
            severity: "info".to_string(),
            code: "ENVIRONMENT_SURFACES_INFERRED_FROM_GEOMETRY".to_string(),
            message: format!(
                "Detected {} horizontal surface candidate(s) from transformed triangles.",
                surfaces.len()
            ),
            recommendation:
                "Review the candidate bounds, then reference the chosen Surface from anchors and actions."
                    .to_string(),
        });
    }
    if anchors.is_empty() {
        diagnostics.push(EnvironmentInspectionDiagnostic {
            severity: "warning".to_string(),
            code: "ENVIRONMENT_HAS_NO_SEMANTIC_ANCHORS".to_string(),
            message: "No takeoff, landing, contact, camera, or ML_ anchor nodes were found."
                .to_string(),
            recommendation:
                "Add named GLB empties such as ML_Takeoff, ML_Landing and ML_CameraWide."
                    .to_string(),
        });
    }
    let surface_score =
        surfaces.iter().map(|item| item.confidence).sum::<f32>() / surfaces.len().max(1) as f32;
    let anchor_score = if anchors.is_empty() {
        0.35
    } else {
        anchors.iter().map(|item| item.confidence).sum::<f32>() / anchors.len() as f32
    };
    let overall_confidence = (surface_score * 0.6 + anchor_score * 0.4).clamp(0.0, 1.0);
    let environment_dsl = build_environment_dsl(&surfaces, &anchors);
    GlbEnvironmentInspectionReport {
        version: REPORT_VERSION,
        asset: asset.to_string(),
        bounds_min,
        bounds_max,
        node_count: mesh.nodes.len(),
        node_names: mesh
            .nodes
            .iter()
            .filter_map(|node| node.name.clone())
            .collect(),
        mesh_count: mesh.mesh_names.len(),
        coordinate_profile,
        surfaces,
        anchors,
        overall_confidence,
        manual_review_required: overall_confidence < 0.75,
        diagnostics,
        environment_dsl,
    }
}

#[derive(Debug, Clone)]
struct HorizontalSurfaceAccumulator {
    height: f32,
    weighted_centroid: [f32; 3],
    weighted_normal: [f32; 3],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    area: f32,
    triangles: usize,
}

fn infer_environment_coordinate_profile(
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
) -> EnvironmentCoordinateProfile {
    let extent_x = (bounds_max[0] - bounds_min[0]).abs();
    let extent_z = (bounds_max[2] - bounds_min[2]).abs();
    let height = (bounds_max[1] - bounds_min[1]).abs().max(0.001);
    let forward = if extent_x >= extent_z {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    EnvironmentCoordinateProfile {
        up: [0.0, 1.0, 0.0],
        forward,
        handedness: "right".to_string(),
        unit_scale: 1.0,
        normalization_origin: [
            (bounds_min[0] + bounds_max[0]) * 0.5,
            bounds_min[1],
            (bounds_min[2] + bounds_max[2]) * 0.5,
        ],
        normalization_scale: 1.0 / height,
        confidence: 0.88,
        evidence: vec![
            "glTF 2.0 defines a right-handed +Y-up coordinate system".to_string(),
            format!(
                "horizontal forward proposal follows the longest X/Z extent ({extent_x:.3} × {extent_z:.3})"
            ),
            "forward sign remains reviewable when no semantic marker is present".to_string(),
        ],
    }
}

fn detect_horizontal_environment_surfaces(
    mesh: &GlbMeshData,
    matrices: &[[f32; 16]],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
) -> Vec<EnvironmentSurfaceProposal> {
    let height_span = (bounds_max[1] - bounds_min[1]).abs().max(0.001);
    let height_tolerance = (height_span * 0.006).clamp(0.015, 0.12);
    let mut groups = Vec::<HorizontalSurfaceAccumulator>::new();
    for triangle in &mesh.triangles {
        let matrix = triangle
            .mesh_node
            .and_then(|index| matrices.get(index))
            .copied()
            .unwrap_or_else(identity);
        let mut points = [[0.0; 3]; 3];
        let mut valid = true;
        for (slot, index) in points.iter_mut().zip(triangle.indices) {
            let Some(position) = mesh.positions.get(index as usize).copied() else {
                valid = false;
                break;
            };
            *slot = transform_point(matrix, position);
        }
        if !valid {
            continue;
        }
        let ab = sub3(points[1], points[0]);
        let ac = sub3(points[2], points[0]);
        let cross = cross3(ab, ac);
        let twice_area = length3(cross);
        if twice_area <= 0.00001 {
            continue;
        }
        let mut normal = scale3(cross, 1.0 / twice_area);
        if normal[1] < 0.0 {
            normal = scale3(normal, -1.0);
        }
        if normal[1] < 0.94 {
            continue;
        }
        let area = twice_area * 0.5;
        let centroid = scale3(add3(add3(points[0], points[1]), points[2]), 1.0 / 3.0);
        let group_index = groups
            .iter()
            .position(|group| (group.height - centroid[1]).abs() <= height_tolerance);
        let index = group_index.unwrap_or_else(|| {
            groups.push(HorizontalSurfaceAccumulator {
                height: centroid[1],
                weighted_centroid: [0.0; 3],
                weighted_normal: [0.0; 3],
                bounds_min: [f32::INFINITY; 3],
                bounds_max: [f32::NEG_INFINITY; 3],
                area: 0.0,
                triangles: 0,
            });
            groups.len() - 1
        });
        let group = &mut groups[index];
        group.height = (group.height * group.area + centroid[1] * area) / (group.area + area);
        group.weighted_centroid = add3(group.weighted_centroid, scale3(centroid, area));
        group.weighted_normal = add3(group.weighted_normal, scale3(normal, area));
        group.area += area;
        group.triangles += 1;
        for point in points {
            for axis in 0..3 {
                group.bounds_min[axis] = group.bounds_min[axis].min(point[axis]);
                group.bounds_max[axis] = group.bounds_max[axis].max(point[axis]);
            }
        }
    }
    groups.sort_by(|a, b| b.area.total_cmp(&a.area));
    let largest_area = groups.first().map_or(0.0, |group| group.area);
    groups
        .into_iter()
        .filter(|group| group.area >= (largest_area * 0.015).max(0.02))
        .take(12)
        .enumerate()
        .map(|(index, group)| {
            let centroid = scale3(group.weighted_centroid, 1.0 / group.area.max(0.00001));
            let normal = normalize3(group.weighted_normal).unwrap_or([0.0, 1.0, 0.0]);
            let confidence =
                (0.58 + (group.area / largest_area.max(0.001)).min(1.0) * 0.28).clamp(0.0, 0.9);
            EnvironmentSurfaceProposal {
                id: format!("auto_surface_{:02}", index + 1),
                node: None,
                kind: "ground".to_string(),
                height: group.height,
                normal,
                centroid,
                bounds_min: group.bounds_min,
                bounds_max: group.bounds_max,
                area: group.area,
                source: "horizontal_triangles".to_string(),
                confidence,
                evidence: format!(
                    "{} transformed triangle(s), area {:.3}, normal [{:.3},{:.3},{:.3}]",
                    group.triangles, group.area, normal[0], normal[1], normal[2]
                ),
            }
        })
        .collect()
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(value: [f32; 3], scale: f32) -> [f32; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length3(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn normalize3(value: [f32; 3]) -> Option<[f32; 3]> {
    let length = length3(value);
    (length > 0.000001).then(|| scale3(value, 1.0 / length))
}

pub(crate) fn transformed_environment_bounds(
    mesh: &GlbMeshData,
    matrices: &[[f32; 16]],
) -> Option<([f32; 3], [f32; 3])> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut found = false;
    for triangle in &mesh.triangles {
        let matrix = triangle
            .mesh_node
            .and_then(|index| matrices.get(index))
            .copied()
            .unwrap_or_else(identity);
        for index in triangle.indices {
            let Some(position) = mesh.positions.get(index as usize).copied() else {
                continue;
            };
            let position = transform_point(matrix, position);
            for axis in 0..3 {
                min[axis] = min[axis].min(position[axis]);
                max[axis] = max[axis].max(position[axis]);
            }
            found = true;
        }
    }
    found.then_some((min, max))
}

fn semantic_id(name: &str) -> String {
    name.trim_start_matches("ML_")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn build_environment_dsl(
    surfaces: &[EnvironmentSurfaceProposal],
    anchors: &[EnvironmentAnchorProposal],
) -> String {
    let mut output = String::from(
        "<Environment id=\"environment\" asset=\"environment_asset\" static=\"true\" collision=\"mesh\">\n",
    );
    for surface in surfaces {
        if let Some(node) = surface.node.as_deref() {
            output.push_str(&format!(
                "  <Surface id=\"{}\" node=\"{}\" kind=\"{}\" height=\"{}\" />\n",
                surface.id, node, surface.kind, surface.height
            ));
        } else {
            output.push_str(&format!(
                "  <Surface id=\"{}\" kind=\"{}\" space=\"asset\" height=\"{}\" normal={{{:?}}} centroid={{{:?}}} boundsMin={{{:?}}} boundsMax={{{:?}}} />\n",
                surface.id,
                surface.kind,
                surface.height,
                surface.normal,
                surface.centroid,
                surface.bounds_min,
                surface.bounds_max
            ));
        }
    }
    for anchor in anchors {
        output.push_str(&format!(
            "  <Anchor id=\"{}\" node=\"{}\" offset={{[0,0,0]}} />\n",
            anchor.id, anchor.node
        ));
    }
    output.push_str("</Environment>");
    output
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RestPoseProposal {
    pub kind: String,
    pub confidence: f32,
    pub left_arm_side_degrees: Option<f32>,
    pub right_arm_side_degrees: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BodyBasisProposal {
    pub up: [f32; 3],
    pub right: [f32; 3],
    pub forward: [f32; 3],
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanoidBoneProposal {
    pub canonical_bone: String,
    pub source_joint: String,
    pub node_index: usize,
    pub confidence: f32,
    pub evidence: Vec<String>,
    pub alternatives: Vec<JointAlternative>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JointAlternative {
    pub source_joint: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoneAxisProposal {
    pub canonical_bone: String,
    pub source_joint: String,
    pub forward: Option<SemanticAxisProposal>,
    pub side: Option<SemanticAxisProposal>,
    pub twist: Option<SemanticAxisProposal>,
    pub bend: Option<SemanticAxisProposal>,
    pub turn: Option<SemanticAxisProposal>,
    pub rest_forward: Option<f32>,
    pub rest_side: Option<f32>,
    pub confidence: f32,
    pub manual_review_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticAxisProposal {
    pub raw: String,
    pub confidence: f32,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInspectionDiagnostic {
    pub severity: String,
    pub code: String,
    pub message: String,
    pub recommendation: String,
    pub canonical_bone: Option<String>,
}

/// Static compatibility result for one portable Action and one declared model
/// profile. This never changes playback; editors use it to explain degradation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct HumanoidActionCompatibilityReport {
    pub action_id: String,
    pub profile_id: String,
    pub full_fidelity: bool,
    pub missing_required_bones: Vec<String>,
    pub missing_action_bones: Vec<String>,
}

/// Compare Action bone usage with the canonical mappings declared by a model
/// profile. Optional bones only become fidelity requirements when keyed.
pub fn inspect_humanoid_action_compatibility(
    action: &WorldAction,
    profile: &WorldModelProfile,
) -> HumanoidActionCompatibilityReport {
    let mapped = profile
        .retarget
        .as_ref()
        .map(|retarget| {
            retarget
                .maps
                .iter()
                .map(|mapping| mapping.to.as_str())
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    let used = action
        .poses
        .iter()
        .flat_map(|pose| pose.bones.iter().map(|bone| bone.id.as_str()))
        .chain(
            action
                .iks
                .iter()
                .flat_map(|ik| [ik.root.as_str(), ik.mid.as_str(), ik.end.as_str()]),
        )
        .collect::<HashSet<_>>();
    let mut missing_required_bones = HUMANOID_BONES
        .iter()
        .filter(|bone| is_core_bone(bone.id) && !mapped.contains(bone.id))
        .map(|bone| bone.id.to_string())
        .collect::<Vec<_>>();
    let mut missing_action_bones = used
        .into_iter()
        .filter(|bone| !mapped.contains(*bone))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    missing_required_bones.sort();
    missing_action_bones.sort();
    HumanoidActionCompatibilityReport {
        action_id: action.id.clone(),
        profile_id: profile.id.clone(),
        full_fidelity: missing_required_bones.is_empty() && missing_action_bones.is_empty(),
        missing_required_bones,
        missing_action_bones,
    }
}

#[derive(Clone, Copy)]
struct CanonicalBoneSpec {
    id: &'static str,
    parent: Option<&'static str>,
    aliases: &'static [&'static str],
    side: Option<Side>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

#[derive(Clone)]
struct Candidate {
    node_index: usize,
    name: String,
    score: f32,
    evidence: Vec<String>,
}

#[derive(Clone, Copy)]
struct BodyBasis {
    up: [f32; 3],
    right: [f32; 3],
    forward: [f32; 3],
    confidence: f32,
}

macro_rules! finger_bone {
    ($id:literal, $parent:literal, $segment:literal, $side:expr) => {
        CanonicalBoneSpec {
            id: $id,
            parent: Some($parent),
            aliases: &[$segment],
            side: Some($side),
        }
    };
}

const HUMANOID_BONES: &[CanonicalBoneSpec] = &[
    CanonicalBoneSpec {
        id: "hips",
        parent: None,
        aliases: &["hips", "hip", "pelvis", "root"],
        side: None,
    },
    CanonicalBoneSpec {
        id: "spine",
        parent: Some("hips"),
        aliases: &["spine", "spine01", "waist"],
        side: None,
    },
    CanonicalBoneSpec {
        id: "chest",
        parent: Some("spine"),
        aliases: &["chest", "spine1", "spine02", "torso"],
        side: None,
    },
    CanonicalBoneSpec {
        id: "upper_chest",
        parent: Some("chest"),
        aliases: &["upperchest", "spine2", "spine03", "spine3"],
        side: None,
    },
    CanonicalBoneSpec {
        id: "neck",
        parent: Some("upper_chest"),
        aliases: &["neck"],
        side: None,
    },
    CanonicalBoneSpec {
        id: "head",
        parent: Some("neck"),
        aliases: &["head"],
        side: None,
    },
    CanonicalBoneSpec {
        id: "shoulder_l",
        parent: Some("upper_chest"),
        aliases: &["shoulder", "clavicle", "collar"],
        side: Some(Side::Left),
    },
    CanonicalBoneSpec {
        id: "upper_arm_l",
        parent: Some("shoulder_l"),
        aliases: &["upperarm", "arm"],
        side: Some(Side::Left),
    },
    CanonicalBoneSpec {
        id: "forearm_l",
        parent: Some("upper_arm_l"),
        aliases: &["forearm", "lowerarm", "elbow"],
        side: Some(Side::Left),
    },
    CanonicalBoneSpec {
        id: "hand_l",
        parent: Some("forearm_l"),
        aliases: &["hand", "wrist"],
        side: Some(Side::Left),
    },
    CanonicalBoneSpec {
        id: "shoulder_r",
        parent: Some("upper_chest"),
        aliases: &["shoulder", "clavicle", "collar"],
        side: Some(Side::Right),
    },
    CanonicalBoneSpec {
        id: "upper_arm_r",
        parent: Some("shoulder_r"),
        aliases: &["upperarm", "arm"],
        side: Some(Side::Right),
    },
    CanonicalBoneSpec {
        id: "forearm_r",
        parent: Some("upper_arm_r"),
        aliases: &["forearm", "lowerarm", "elbow"],
        side: Some(Side::Right),
    },
    CanonicalBoneSpec {
        id: "hand_r",
        parent: Some("forearm_r"),
        aliases: &["hand", "wrist"],
        side: Some(Side::Right),
    },
    CanonicalBoneSpec {
        id: "upper_leg_l",
        parent: Some("hips"),
        aliases: &["upperleg", "upleg", "thigh", "leg"],
        side: Some(Side::Left),
    },
    CanonicalBoneSpec {
        id: "lower_leg_l",
        parent: Some("upper_leg_l"),
        aliases: &["lowerleg", "leg2", "shin", "calf", "knee", "leg"],
        side: Some(Side::Left),
    },
    CanonicalBoneSpec {
        id: "foot_l",
        parent: Some("lower_leg_l"),
        aliases: &["foot", "ankle"],
        side: Some(Side::Left),
    },
    CanonicalBoneSpec {
        id: "toe_l",
        parent: Some("foot_l"),
        aliases: &["toe", "toes", "ball"],
        side: Some(Side::Left),
    },
    CanonicalBoneSpec {
        id: "upper_leg_r",
        parent: Some("hips"),
        aliases: &["upperleg", "upleg", "thigh", "leg"],
        side: Some(Side::Right),
    },
    CanonicalBoneSpec {
        id: "lower_leg_r",
        parent: Some("upper_leg_r"),
        aliases: &["lowerleg", "leg2", "shin", "calf", "knee", "leg"],
        side: Some(Side::Right),
    },
    CanonicalBoneSpec {
        id: "foot_r",
        parent: Some("lower_leg_r"),
        aliases: &["foot", "ankle"],
        side: Some(Side::Right),
    },
    CanonicalBoneSpec {
        id: "toe_r",
        parent: Some("foot_r"),
        aliases: &["toe", "toes", "ball"],
        side: Some(Side::Right),
    },
    finger_bone!("thumb_1_l", "hand_l", "thumb1", Side::Left),
    finger_bone!("thumb_2_l", "thumb_1_l", "thumb2", Side::Left),
    finger_bone!("thumb_3_l", "thumb_2_l", "thumb3", Side::Left),
    finger_bone!("index_1_l", "hand_l", "index1", Side::Left),
    finger_bone!("index_2_l", "index_1_l", "index2", Side::Left),
    finger_bone!("index_3_l", "index_2_l", "index3", Side::Left),
    finger_bone!("middle_1_l", "hand_l", "middle1", Side::Left),
    finger_bone!("middle_2_l", "middle_1_l", "middle2", Side::Left),
    finger_bone!("middle_3_l", "middle_2_l", "middle3", Side::Left),
    finger_bone!("ring_1_l", "hand_l", "ring1", Side::Left),
    finger_bone!("ring_2_l", "ring_1_l", "ring2", Side::Left),
    finger_bone!("ring_3_l", "ring_2_l", "ring3", Side::Left),
    finger_bone!("pinky_1_l", "hand_l", "pinky1", Side::Left),
    finger_bone!("pinky_2_l", "pinky_1_l", "pinky2", Side::Left),
    finger_bone!("pinky_3_l", "pinky_2_l", "pinky3", Side::Left),
    finger_bone!("thumb_1_r", "hand_r", "thumb1", Side::Right),
    finger_bone!("thumb_2_r", "thumb_1_r", "thumb2", Side::Right),
    finger_bone!("thumb_3_r", "thumb_2_r", "thumb3", Side::Right),
    finger_bone!("index_1_r", "hand_r", "index1", Side::Right),
    finger_bone!("index_2_r", "index_1_r", "index2", Side::Right),
    finger_bone!("index_3_r", "index_2_r", "index3", Side::Right),
    finger_bone!("middle_1_r", "hand_r", "middle1", Side::Right),
    finger_bone!("middle_2_r", "middle_1_r", "middle2", Side::Right),
    finger_bone!("middle_3_r", "middle_2_r", "middle3", Side::Right),
    finger_bone!("ring_1_r", "hand_r", "ring1", Side::Right),
    finger_bone!("ring_2_r", "ring_1_r", "ring2", Side::Right),
    finger_bone!("ring_3_r", "ring_2_r", "ring3", Side::Right),
    finger_bone!("pinky_1_r", "hand_r", "pinky1", Side::Right),
    finger_bone!("pinky_2_r", "pinky_1_r", "pinky2", Side::Right),
    finger_bone!("pinky_3_r", "pinky_2_r", "pinky3", Side::Right),
];

fn detect_humanoid_rig(
    document: &Value,
    nodes: &[GlbNodeData],
) -> (DetectedHumanoidRig, HashMap<String, PreferredBoneMapping>) {
    if let Some(preferred) = vrm1_mappings(document, nodes).filter(|map| !map.is_empty()) {
        return detected_rig(
            "vrm_1",
            "VRM 1.0",
            1.0,
            "metadata",
            vec!["VRMC_vrm humanoid metadata".to_string()],
            preferred,
        );
    }
    if let Some(preferred) = vrm0_mappings(document, nodes).filter(|map| !map.is_empty()) {
        return detected_rig(
            "vrm_0",
            "VRM 0.x",
            1.0,
            "metadata",
            vec!["VRM humanoid metadata".to_string()],
            preferred,
        );
    }

    let preferred = known_humanoid_name_mappings(nodes);
    let core_count = preferred.keys().filter(|bone| is_core_bone(bone)).count();
    let generator = document
        .get("asset")
        .and_then(|asset| asset.get("generator"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let ready_player_me = generator.contains("ready player me")
        || generator.contains("readyplayerme")
        || document.get("extras").is_some_and(|extras| {
            extras
                .to_string()
                .to_ascii_lowercase()
                .contains("ready player me")
        });
    let mixamo_namespace = nodes
        .iter()
        .filter_map(|node| node.name.as_deref())
        .any(|name| name.to_ascii_lowercase().contains("mixamorig"));
    let hierarchy_score = known_mapping_hierarchy_score(&preferred, nodes);
    let enough_known_bones = core_count >= 12 && hierarchy_score >= 0.72;

    if ready_player_me && enough_known_bones {
        return detected_rig(
            "ready_player_me",
            "Ready Player Me",
            0.97,
            "preset",
            vec![
                format!("asset generator: {}", generator),
                format!("known humanoid hierarchy {:.0}%", hierarchy_score * 100.0),
            ],
            preferred,
        );
    }
    if mixamo_namespace && enough_known_bones {
        return detected_rig(
            "mixamo",
            "Mixamo",
            0.98,
            "preset",
            vec![
                "mixamorig namespace".to_string(),
                format!("known humanoid hierarchy {:.0}%", hierarchy_score * 100.0),
            ],
            preferred,
        );
    }
    if enough_known_bones {
        return detected_rig(
            "mixamo_compatible",
            "Mixamo-compatible humanoid",
            0.90,
            "preset",
            vec![format!(
                "known humanoid names and hierarchy {:.0}%",
                hierarchy_score * 100.0
            )],
            preferred,
        );
    }

    (
        DetectedHumanoidRig {
            family: "generic_humanoid".to_string(),
            label: "Generic humanoid".to_string(),
            confidence: 0.5,
            mapping_source: "heuristic".to_string(),
            matched_bone_count: 0,
            core_bone_count: 0,
            evidence: vec![
                "No declared or known rig signature; using geometry and names".to_string(),
            ],
        },
        HashMap::new(),
    )
}

fn detected_rig(
    family: &str,
    label: &str,
    confidence: f32,
    mapping_source: &str,
    evidence: Vec<String>,
    preferred: HashMap<String, PreferredBoneMapping>,
) -> (DetectedHumanoidRig, HashMap<String, PreferredBoneMapping>) {
    let core_bone_count = preferred.keys().filter(|bone| is_core_bone(bone)).count();
    (
        DetectedHumanoidRig {
            family: family.to_string(),
            label: label.to_string(),
            confidence,
            mapping_source: mapping_source.to_string(),
            matched_bone_count: preferred.len(),
            core_bone_count,
            evidence,
        },
        preferred,
    )
}

fn vrm1_mappings(
    document: &Value,
    nodes: &[GlbNodeData],
) -> Option<HashMap<String, PreferredBoneMapping>> {
    let bones = document
        .pointer("/extensions/VRMC_vrm/humanoid/humanBones")?
        .as_object()?;
    let mut out = HashMap::new();
    for (vrm_bone, declaration) in bones {
        let Some(canonical) = canonical_vrm_bone(vrm_bone, true) else {
            continue;
        };
        let Some(node_index) = declaration.get("node").and_then(Value::as_u64) else {
            continue;
        };
        insert_declared_mapping(
            &mut out,
            nodes,
            canonical,
            node_index as usize,
            1.0,
            "VRM 1.0 humanoid metadata",
        );
    }
    Some(out)
}

fn vrm0_mappings(
    document: &Value,
    nodes: &[GlbNodeData],
) -> Option<HashMap<String, PreferredBoneMapping>> {
    let bones = document
        .pointer("/extensions/VRM/humanoid/humanBones")?
        .as_array()?;
    let mut out = HashMap::new();
    for declaration in bones {
        let Some(vrm_bone) = declaration.get("bone").and_then(Value::as_str) else {
            continue;
        };
        let Some(canonical) = canonical_vrm_bone(vrm_bone, false) else {
            continue;
        };
        let Some(node_index) = declaration.get("node").and_then(Value::as_u64) else {
            continue;
        };
        insert_declared_mapping(
            &mut out,
            nodes,
            canonical,
            node_index as usize,
            1.0,
            "VRM 0.x humanoid metadata",
        );
    }
    Some(out)
}

fn insert_declared_mapping(
    out: &mut HashMap<String, PreferredBoneMapping>,
    nodes: &[GlbNodeData],
    canonical: &str,
    node_index: usize,
    confidence: f32,
    evidence: &str,
) {
    if nodes
        .get(node_index)
        .and_then(|node| node.name.as_ref())
        .is_some()
    {
        out.insert(
            canonical.to_string(),
            PreferredBoneMapping {
                node_index,
                confidence,
                evidence: evidence.to_string(),
            },
        );
    }
}

fn known_humanoid_name_mappings(nodes: &[GlbNodeData]) -> HashMap<String, PreferredBoneMapping> {
    let mut out = HashMap::new();
    for node in nodes {
        let Some(name) = node.name.as_deref() else {
            continue;
        };
        let Some(canonical) = canonical_known_rig_bone(name) else {
            continue;
        };
        out.entry(canonical.to_string())
            .or_insert_with(|| PreferredBoneMapping {
                node_index: node.index,
                confidence: 0.98,
                evidence: format!("known humanoid rig joint '{name}'"),
            });
    }
    out
}

fn canonical_known_rig_bone(raw: &str) -> Option<String> {
    let suffix = raw.rsplit([':', '|']).next().unwrap_or(raw);
    let compact = suffix
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    let canonical = match compact.as_str() {
        "hips" | "pelvis" => "hips",
        "spine" | "spine01" => "spine",
        "spine1" | "spine02" | "chest" => "chest",
        "spine2" | "spine03" | "upperchest" => "upper_chest",
        "neck" | "neck01" => "neck",
        "head" => "head",
        "leftshoulder" | "claviclel" => "shoulder_l",
        "leftarm" | "leftupperarm" | "upperarml" => "upper_arm_l",
        "leftforearm" | "leftlowerarm" | "lowerarml" => "forearm_l",
        "lefthand" | "handl" => "hand_l",
        "rightshoulder" | "clavicler" => "shoulder_r",
        "rightarm" | "rightupperarm" | "upperarmr" => "upper_arm_r",
        "rightforearm" | "rightlowerarm" | "lowerarmr" => "forearm_r",
        "righthand" | "handr" => "hand_r",
        "leftupleg" | "leftupperleg" | "thighl" => "upper_leg_l",
        "leftleg" | "leftlowerleg" | "calfl" => "lower_leg_l",
        "leftfoot" | "footl" => "foot_l",
        "lefttoebase" | "lefttoes" | "balll" => "toe_l",
        "rightupleg" | "rightupperleg" | "thighr" => "upper_leg_r",
        "rightleg" | "rightlowerleg" | "calfr" => "lower_leg_r",
        "rightfoot" | "footr" => "foot_r",
        "righttoebase" | "righttoes" | "ballr" => "toe_r",
        _ => return canonical_known_finger(&compact),
    };
    Some(canonical.to_string())
}

fn canonical_known_finger(compact: &str) -> Option<String> {
    let (side, rest) = compact
        .strip_prefix("lefthand")
        .or_else(|| compact.strip_prefix("left"))
        .map(|rest| ("l", rest))
        .or_else(|| {
            compact
                .strip_prefix("righthand")
                .or_else(|| compact.strip_prefix("right"))
                .map(|rest| ("r", rest))
        })?;
    let (finger, segment) = ["thumb", "index", "middle", "ring", "pinky", "little"]
        .into_iter()
        .find_map(|finger| {
            let segment = rest.strip_prefix(finger)?;
            Some((if finger == "little" { "pinky" } else { finger }, segment))
        })?;
    let segment = match segment {
        "1" | "proximal" => 1,
        "2" | "intermediate" => 2,
        "3" | "distal" => 3,
        _ => return None,
    };
    Some(format!("{finger}_{segment}_{side}"))
}

fn canonical_vrm_bone(raw: &str, vrm1: bool) -> Option<&'static str> {
    Some(match raw {
        "hips" => "hips",
        "spine" => "spine",
        "chest" => "chest",
        "upperChest" => "upper_chest",
        "neck" => "neck",
        "head" => "head",
        "leftShoulder" => "shoulder_l",
        "leftUpperArm" => "upper_arm_l",
        "leftLowerArm" => "forearm_l",
        "leftHand" => "hand_l",
        "rightShoulder" => "shoulder_r",
        "rightUpperArm" => "upper_arm_r",
        "rightLowerArm" => "forearm_r",
        "rightHand" => "hand_r",
        "leftUpperLeg" => "upper_leg_l",
        "leftLowerLeg" => "lower_leg_l",
        "leftFoot" => "foot_l",
        "leftToes" => "toe_l",
        "rightUpperLeg" => "upper_leg_r",
        "rightLowerLeg" => "lower_leg_r",
        "rightFoot" => "foot_r",
        "rightToes" => "toe_r",
        "leftThumbMetacarpal" if vrm1 => "thumb_1_l",
        "leftThumbProximal" => {
            if vrm1 {
                "thumb_2_l"
            } else {
                "thumb_1_l"
            }
        }
        "leftThumbIntermediate" => "thumb_2_l",
        "leftThumbDistal" => "thumb_3_l",
        "leftIndexProximal" => "index_1_l",
        "leftIndexIntermediate" => "index_2_l",
        "leftIndexDistal" => "index_3_l",
        "leftMiddleProximal" => "middle_1_l",
        "leftMiddleIntermediate" => "middle_2_l",
        "leftMiddleDistal" => "middle_3_l",
        "leftRingProximal" => "ring_1_l",
        "leftRingIntermediate" => "ring_2_l",
        "leftRingDistal" => "ring_3_l",
        "leftLittleProximal" => "pinky_1_l",
        "leftLittleIntermediate" => "pinky_2_l",
        "leftLittleDistal" => "pinky_3_l",
        "rightThumbMetacarpal" if vrm1 => "thumb_1_r",
        "rightThumbProximal" => {
            if vrm1 {
                "thumb_2_r"
            } else {
                "thumb_1_r"
            }
        }
        "rightThumbIntermediate" => "thumb_2_r",
        "rightThumbDistal" => "thumb_3_r",
        "rightIndexProximal" => "index_1_r",
        "rightIndexIntermediate" => "index_2_r",
        "rightIndexDistal" => "index_3_r",
        "rightMiddleProximal" => "middle_1_r",
        "rightMiddleIntermediate" => "middle_2_r",
        "rightMiddleDistal" => "middle_3_r",
        "rightRingProximal" => "ring_1_r",
        "rightRingIntermediate" => "ring_2_r",
        "rightRingDistal" => "ring_3_r",
        "rightLittleProximal" => "pinky_1_r",
        "rightLittleIntermediate" => "pinky_2_r",
        "rightLittleDistal" => "pinky_3_r",
        _ => return None,
    })
}

fn known_mapping_hierarchy_score(
    preferred: &HashMap<String, PreferredBoneMapping>,
    nodes: &[GlbNodeData],
) -> f32 {
    let mut checked = 0usize;
    let mut valid = 0usize;
    for spec in HUMANOID_BONES {
        let Some(parent) = spec.parent else {
            continue;
        };
        let (Some(child), Some(parent)) = (preferred.get(spec.id), preferred.get(parent)) else {
            continue;
        };
        checked += 1;
        let mut cursor = nodes.get(child.node_index).and_then(|node| node.parent);
        while let Some(index) = cursor {
            if index == parent.node_index {
                valid += 1;
                break;
            }
            cursor = nodes.get(index).and_then(|node| node.parent);
        }
    }
    if checked == 0 {
        0.0
    } else {
        valid as f32 / checked as f32
    }
}

/// Inspect a local GLB and propose a `humanoid_v1` profile without changing the model.
pub fn inspect_glb_skeleton_path(
    path: impl AsRef<Path>,
) -> Result<GlbSkeletonInspectionReport, ModelInspectionError> {
    let path = path.as_ref();
    let mesh = load_glb_animation_data(path)?;
    inspect_mesh_skeleton(&mesh, &path.to_string_lossy())
}

/// Inspect GLB bytes. This is the zero-copy entry point used by browser hosts.
pub fn inspect_glb_skeleton_bytes(
    bytes: &[u8],
    asset_label: impl AsRef<str>,
) -> Result<GlbSkeletonInspectionReport, ModelInspectionError> {
    let label = asset_label.as_ref();
    let source_path = PathBuf::from(label);
    let mesh = parse_glb_animation_data(&source_path, bytes)?;
    inspect_mesh_skeleton(&mesh, label)
}

/// Inspect a browser/local GLB using declared VRM humanoid metadata first,
/// known Mixamo-compatible signatures second, and the existing geometry/name
/// heuristic for every unresolved bone. This is a new opt-in API; the legacy
/// skeleton inspection contract above is intentionally untouched.
pub fn inspect_glb_humanoid_profile_bytes(
    bytes: &[u8],
    asset_label: impl AsRef<str>,
) -> Result<GlbHumanoidProfileInspectionReport, ModelInspectionError> {
    let label = asset_label.as_ref();
    let source_path = PathBuf::from(label);
    let document = parse_gltf_json_value(&source_path, bytes)?;
    let mesh = parse_glb_animation_data(&source_path, bytes)?;
    let (detected_rig, preferred) = detect_humanoid_rig(&document, &mesh.nodes);
    let skeleton = inspect_mesh_skeleton_with_preferred(&mesh, label, &preferred)?;
    Ok(GlbHumanoidProfileInspectionReport {
        detected_rig,
        skeleton,
    })
}

/// Stable JSON wrapper for [`inspect_glb_humanoid_profile_bytes`].
pub fn inspect_glb_humanoid_profile_json(
    bytes: &[u8],
    asset_label: impl AsRef<str>,
) -> Result<String, ModelInspectionError> {
    let report = inspect_glb_humanoid_profile_bytes(bytes, asset_label)?;
    serde_json::to_string_pretty(&report)
        .map_err(|source| ModelInspectionError::Serialize { source })
}

/// Return the skeleton inspection report as stable, machine-readable JSON.
pub fn inspect_glb_skeleton_json(
    bytes: &[u8],
    asset_label: impl AsRef<str>,
) -> Result<String, ModelInspectionError> {
    let report = inspect_glb_skeleton_bytes(bytes, asset_label)?;
    serde_json::to_string_pretty(&report)
        .map_err(|source| ModelInspectionError::Serialize { source })
}

fn inspect_mesh_skeleton(
    mesh: &GlbMeshData,
    asset_label: &str,
) -> Result<GlbSkeletonInspectionReport, ModelInspectionError> {
    inspect_mesh_skeleton_with_preferred(mesh, asset_label, &HashMap::new())
}

fn inspect_mesh_skeleton_with_preferred(
    mesh: &GlbMeshData,
    asset_label: &str,
    preferred: &HashMap<String, PreferredBoneMapping>,
) -> Result<GlbSkeletonInspectionReport, ModelInspectionError> {
    let skin = mesh
        .skin
        .as_ref()
        .ok_or_else(|| ModelInspectionError::MissingSkeleton {
            asset: asset_label.to_string(),
        })?;
    if skin.joints.is_empty() {
        return Err(ModelInspectionError::MissingSkeleton {
            asset: asset_label.to_string(),
        });
    }

    let joint_indices = skin
        .joints
        .iter()
        .map(|joint| joint.node_index)
        .collect::<HashSet<_>>();
    let world_matrices = world_matrices(&mesh.nodes);
    let mut mapped_nodes = HashSet::new();
    let mut resolved = BTreeMap::<String, usize>::new();
    let mut proposals = Vec::new();
    let mut diagnostics = Vec::new();

    for spec in HUMANOID_BONES {
        let mut candidates = mesh
            .nodes
            .iter()
            .filter(|node| {
                joint_indices.contains(&node.index) && !mapped_nodes.contains(&node.index)
            })
            .filter_map(|node| score_candidate(node, spec, &resolved, &mesh.nodes))
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| right.score.total_cmp(&left.score));

        let declared = preferred.get(spec.id).and_then(|mapping| {
            if !joint_indices.contains(&mapping.node_index)
                || mapped_nodes.contains(&mapping.node_index)
            {
                return None;
            }
            let node = mesh.nodes.get(mapping.node_index)?;
            Some(Candidate {
                node_index: mapping.node_index,
                name: node.name.clone()?,
                score: mapping.confidence,
                evidence: vec![mapping.evidence.clone()],
            })
        });
        let Some(best) = declared.clone().or_else(|| candidates.first().cloned()) else {
            diagnostics.push(ModelInspectionDiagnostic {
                severity: if is_core_bone(spec.id) { "error" } else { "warning" }.to_string(),
                code: "humanoid_bone_unmapped".to_string(),
                message: format!("No GLB joint could be mapped to canonical bone '{}'.", spec.id),
                recommendation: "Add an explicit <Retarget><Map ... /></Retarget> entry after inspecting the joint hierarchy.".to_string(),
                canonical_bone: Some(spec.id.to_string()),
            });
            continue;
        };

        if declared.is_none() && best.score < 0.42 {
            diagnostics.push(ModelInspectionDiagnostic {
                severity: "warning".to_string(),
                code: "humanoid_bone_low_confidence".to_string(),
                message: format!("Mapping '{}' to '{}' has low confidence ({:.2}).", best.name, spec.id, best.score),
                recommendation: "Confirm this joint visually before using the generated profile for production retargeting.".to_string(),
                canonical_bone: Some(spec.id.to_string()),
            });
        }
        let second_score = candidates
            .iter()
            .filter(|candidate| candidate.node_index != best.node_index)
            .next()
            .map_or(0.0, |candidate| candidate.score);
        if declared.is_none() && best.score - second_score < 0.12 && second_score > 0.35 {
            diagnostics.push(ModelInspectionDiagnostic {
                severity: "warning".to_string(),
                code: "humanoid_bone_ambiguous".to_string(),
                message: format!("Mapping for '{}' is ambiguous between '{}' and another joint.", spec.id, best.name),
                recommendation: "Use the alternatives array and the skeleton overlay to select the correct source joint.".to_string(),
                canonical_bone: Some(spec.id.to_string()),
            });
        }

        mapped_nodes.insert(best.node_index);
        resolved.insert(spec.id.to_string(), best.node_index);
        proposals.push(HumanoidBoneProposal {
            canonical_bone: spec.id.to_string(),
            source_joint: best.name,
            node_index: best.node_index,
            confidence: round_confidence(best.score),
            evidence: best.evidence,
            alternatives: candidates
                .iter()
                .filter(|candidate| candidate.node_index != best.node_index)
                .take(3)
                .map(|candidate| JointAlternative {
                    source_joint: candidate.name.clone(),
                    confidence: round_confidence(candidate.score),
                })
                .collect(),
        });
    }

    let basis = infer_body_basis(&resolved, &world_matrices, &mut diagnostics);
    let rest_pose = infer_rest_pose(&resolved, &world_matrices, basis);
    let axes = propose_axes(
        &resolved,
        &proposals,
        &mesh.nodes,
        &world_matrices,
        basis,
        &mut diagnostics,
    );
    let mapping_confidence = mean(proposals.iter().map(|proposal| proposal.confidence));
    let axis_confidence = mean(axes.iter().map(|proposal| proposal.confidence));
    let overall_confidence = round_confidence(
        mapping_confidence * 0.58 + axis_confidence * 0.30 + rest_pose.confidence * 0.12,
    );
    let manual_review_required = overall_confidence < 0.82
        || diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == "error")
        || axes.iter().any(|axis| axis.manual_review_required);

    if manual_review_required {
        diagnostics.push(ModelInspectionDiagnostic {
            severity: "info".to_string(),
            code: "profile_review_required".to_string(),
            message: "The generated profile is a proposal, not a silent retargeting guarantee.".to_string(),
            recommendation: "Preview small semantic +20 degree actions for side, forward, bend and twist before authoring the final action.".to_string(),
            canonical_bone: None,
        });
    }

    let mapped_names = proposals
        .iter()
        .map(|proposal| proposal.source_joint.as_str())
        .collect::<HashSet<_>>();
    let unmapped_joints = skin
        .joints
        .iter()
        .filter_map(|joint| joint.name.as_deref())
        .filter(|name| !mapped_names.contains(name))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let profile_dsl = build_profile_dsl(asset_label, &proposals, &axes);

    Ok(GlbSkeletonInspectionReport {
        version: REPORT_VERSION,
        asset: asset_label.to_string(),
        skin_joint_count: skin.joints.len(),
        animation_clip_count: mesh.animations.len(),
        rest_pose,
        body_basis: BodyBasisProposal {
            up: round_vector(basis.up),
            right: round_vector(basis.right),
            forward: round_vector(basis.forward),
            confidence: round_confidence(basis.confidence),
        },
        mappings: proposals,
        axes,
        unmapped_joints,
        overall_confidence,
        manual_review_required,
        diagnostics,
        profile_dsl,
    })
}

fn score_candidate(
    node: &GlbNodeData,
    spec: &CanonicalBoneSpec,
    resolved: &BTreeMap<String, usize>,
    nodes: &[GlbNodeData],
) -> Option<Candidate> {
    let name = node.name.clone()?;
    let normalized = normalize_name(&name);
    let compact = normalized.replace(' ', "");
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();
    let mut score = 0.0_f32;
    let mut evidence = Vec::new();

    let alias_score = spec.aliases.iter().fold(0.0_f32, |best, alias| {
        let alias_compact = alias.replace(' ', "");
        let current: f32 = if compact == alias_compact {
            0.82
        } else if tokens.iter().any(|token| *token == *alias) {
            0.74
        } else if compact.contains(&alias_compact) {
            0.62
        } else {
            0.0
        };
        best.max(current)
    });
    if alias_score == 0.0 {
        return None;
    }
    score += alias_score;
    evidence.push(format!("name matches {} role", spec.id));

    if [
        "twist", "ribon", "ribbon", "helper", "ctrl", "control", "hair", "skirt",
    ]
    .iter()
    .any(|marker| tokens.contains(marker) || compact.contains(marker))
    {
        score -= 0.28;
        evidence.push("auxiliary/helper joint name lowers confidence".to_string());
    }

    match (spec.side, detect_side(&tokens, &compact)) {
        (Some(expected), Some(actual)) if expected == actual => {
            score += 0.14;
            evidence.push("left/right name marker matches".to_string());
        }
        (Some(_), Some(_)) => return None,
        (Some(_), None) => score -= 0.08,
        (None, Some(_)) => score -= 0.12,
        _ => {}
    }

    if let Some(parent) = spec.parent.and_then(|parent| resolved.get(parent)) {
        if is_descendant(node.index, *parent, nodes) {
            score += 0.08;
            evidence.push(format!(
                "joint hierarchy descends from {}",
                spec.parent.unwrap_or("root")
            ));
        } else {
            score -= 0.18;
        }
    }
    Some(Candidate {
        node_index: node.index,
        name,
        score: score.clamp(0.0, 1.0),
        evidence,
    })
}

fn normalize_name(name: &str) -> String {
    let mut output = String::with_capacity(name.len());
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
        } else if !output.ends_with(' ') {
            output.push(' ');
        }
    }
    output
        .trim()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn detect_side(tokens: &[&str], compact: &str) -> Option<Side> {
    if tokens.iter().any(|token| matches!(*token, "left" | "l")) || compact.contains("left") {
        return Some(Side::Left);
    }
    if tokens.iter().any(|token| matches!(*token, "right" | "r")) || compact.contains("right") {
        return Some(Side::Right);
    }
    None
}

fn is_descendant(mut node_index: usize, ancestor: usize, nodes: &[GlbNodeData]) -> bool {
    for _ in 0..nodes.len() {
        let Some(parent) = nodes.get(node_index).and_then(|node| node.parent) else {
            return false;
        };
        if parent == ancestor {
            return true;
        }
        node_index = parent;
    }
    false
}

fn is_core_bone(id: &str) -> bool {
    !id.starts_with("thumb_")
        && !id.starts_with("index_")
        && !id.starts_with("middle_")
        && !id.starts_with("ring_")
        && !id.starts_with("pinky_")
}

fn infer_body_basis(
    resolved: &BTreeMap<String, usize>,
    matrices: &[[f32; 16]],
    diagnostics: &mut Vec<ModelInspectionDiagnostic>,
) -> BodyBasis {
    let hips = resolved
        .get("hips")
        .and_then(|index| matrices.get(*index))
        .map(matrix_translation);
    let head = resolved
        .get("head")
        .and_then(|index| matrices.get(*index))
        .map(matrix_translation);
    let left = resolved
        .get("shoulder_l")
        .or_else(|| resolved.get("upper_arm_l"))
        .and_then(|index| matrices.get(*index))
        .map(matrix_translation);
    let right = resolved
        .get("shoulder_r")
        .or_else(|| resolved.get("upper_arm_r"))
        .and_then(|index| matrices.get(*index))
        .map(matrix_translation);

    let up = match (hips, head) {
        (Some(hips), Some(head)) => normalize(sub(head, hips)).unwrap_or([0.0, 1.0, 0.0]),
        _ => [0.0, 1.0, 0.0],
    };
    let right_axis = match (left, right) {
        (Some(left), Some(right)) => reject(sub(right, left), up)
            .and_then(normalize)
            .unwrap_or([1.0, 0.0, 0.0]),
        _ => [1.0, 0.0, 0.0],
    };
    let lateral_forward = normalize(cross(right_axis, up)).unwrap_or([0.0, 0.0, -1.0]);
    let toe_forward = [
        direction_between("foot_l", "toe_l", resolved, matrices),
        direction_between("foot_r", "toe_r", resolved, matrices),
    ]
    .into_iter()
    .flatten()
    .filter_map(|direction| reject(direction, up).and_then(normalize))
    .reduce(|left, right| add(left, right))
    .and_then(normalize);
    let (forward, facing_confidence) = if let Some(toe_forward) = toe_forward {
        let sign = if dot(lateral_forward, toe_forward) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        (scale(lateral_forward, sign), 0.96)
    } else {
        diagnostics.push(ModelInspectionDiagnostic {
            severity: "warning".to_string(),
            code: "body_forward_ambiguous".to_string(),
            message: "The skeleton does not expose a decisive foot-to-toe facing direction."
                .to_string(),
            recommendation: "Preview a +20 degree upper-arm forward action and flip the proposed forward sign if the hand moves behind the character.".to_string(),
            canonical_bone: None,
        });
        (lateral_forward, 0.58)
    };
    let structural_confidence =
        if hips.is_some() && head.is_some() && left.is_some() && right.is_some() {
            0.96
        } else {
            0.55
        };
    BodyBasis {
        up,
        right: right_axis,
        forward,
        confidence: structural_confidence * facing_confidence,
    }
}

fn infer_rest_pose(
    resolved: &BTreeMap<String, usize>,
    matrices: &[[f32; 16]],
    basis: BodyBasis,
) -> RestPoseProposal {
    let left = arm_side_angle(
        "upper_arm_l",
        "forearm_l",
        Side::Left,
        resolved,
        matrices,
        basis,
    );
    let right = arm_side_angle(
        "upper_arm_r",
        "forearm_r",
        Side::Right,
        resolved,
        matrices,
        basis,
    );
    let values = [left, right].into_iter().flatten().collect::<Vec<_>>();
    let average = if values.is_empty() {
        0.0
    } else {
        values.iter().copied().sum::<f32>() / values.len() as f32
    };
    let kind = if values.is_empty() {
        "unknown"
    } else if average < 20.0 {
        "arms_down"
    } else if average > 70.0 {
        "t_pose"
    } else if average > 25.0 {
        "a_pose"
    } else {
        "custom"
    };
    RestPoseProposal {
        kind: kind.to_string(),
        confidence: round_confidence(if values.len() == 2 {
            0.92
        } else if values.len() == 1 {
            0.62
        } else {
            0.20
        }),
        left_arm_side_degrees: left.map(round_degrees),
        right_arm_side_degrees: right.map(round_degrees),
    }
}

fn arm_side_angle(
    upper: &str,
    lower: &str,
    side: Side,
    resolved: &BTreeMap<String, usize>,
    matrices: &[[f32; 16]],
    basis: BodyBasis,
) -> Option<f32> {
    let direction = direction_between(upper, lower, resolved, matrices)?;
    let outward = if side == Side::Left {
        scale(basis.right, -1.0)
    } else {
        basis.right
    };
    let down = scale(basis.up, -1.0);
    Some(
        dot(direction, outward)
            .atan2(dot(direction, down))
            .to_degrees()
            .abs(),
    )
}

fn propose_axes(
    resolved: &BTreeMap<String, usize>,
    mappings: &[HumanoidBoneProposal],
    nodes: &[GlbNodeData],
    matrices: &[[f32; 16]],
    basis: BodyBasis,
    diagnostics: &mut Vec<ModelInspectionDiagnostic>,
) -> Vec<BoneAxisProposal> {
    let mapping_confidence = mappings
        .iter()
        .map(|mapping| (mapping.canonical_bone.as_str(), mapping.confidence))
        .collect::<HashMap<_, _>>();
    let mut proposals = Vec::new();
    for spec in HUMANOID_BONES {
        let Some(node_index) = resolved.get(spec.id).copied() else {
            continue;
        };
        let Some(matrix) = matrices.get(node_index) else {
            continue;
        };
        let axes = matrix_axes(matrix);
        let long = canonical_child(spec.id)
            .and_then(|child| direction_between(spec.id, child, resolved, matrices))
            .or_else(|| matrix_axis(matrix, 1))
            .unwrap_or([0.0, 1.0, 0.0]);
        let source_joint = mappings
            .iter()
            .find(|mapping| mapping.canonical_bone == spec.id)
            .map(|mapping| mapping.source_joint.clone())
            .unwrap_or_else(|| spec.id.to_string());
        let map_conf = *mapping_confidence.get(spec.id).unwrap_or(&0.4);
        let twist = if is_limb(spec.id) {
            Some(axis_alignment(axes, long, "limb long axis"))
        } else {
            None
        };
        let outward = match spec.side {
            Some(Side::Left) => scale(basis.right, -1.0),
            Some(Side::Right) => basis.right,
            None => basis.right,
        };
        let down = scale(basis.up, -1.0);
        let side = if is_upper_limb(spec.id) {
            axis_for_rotation_plane(
                axes,
                down,
                outward,
                "rotates an arms-down limb toward character outward",
            )
        } else {
            None
        };
        let forward = if is_upper_limb(spec.id) {
            canonical_child(spec.id).and_then(|child| {
                axis_for_probed_motion(
                    node_index,
                    *resolved.get(child)?,
                    nodes,
                    matrices,
                    long,
                    basis.forward,
                    "moves limb toward character forward",
                )
            })
        } else {
            None
        };
        let bend = if is_bending_joint(spec.id) {
            canonical_child(spec.id).and_then(|child| {
                axis_for_probed_motion(
                    node_index,
                    *resolved.get(child)?,
                    nodes,
                    matrices,
                    long,
                    basis.forward,
                    "flexes the child segment toward character forward",
                )
            })
        } else if matches!(spec.id, "spine" | "chest" | "upper_chest" | "neck" | "head") {
            Some(axis_alignment(axes, basis.right, "body bend axis"))
        } else {
            None
        };
        let turn = if matches!(
            spec.id,
            "hips" | "spine" | "chest" | "upper_chest" | "neck" | "head"
        ) {
            Some(axis_alignment(axes, basis.up, "body vertical turn axis"))
        } else {
            None
        };
        if forward.is_none()
            && side.is_none()
            && twist.is_none()
            && bend.is_none()
            && turn.is_none()
        {
            continue;
        }
        let rest_offsets = if spec.id.starts_with("upper_arm") {
            canonical_child(spec.id).and_then(|child| {
                solve_arm_rest_offsets(
                    node_index,
                    *resolved.get(child)?,
                    nodes,
                    basis,
                    forward.as_ref()?,
                    side.as_ref()?,
                )
            })
        } else {
            None
        };
        let (rest_forward, rest_side) = rest_offsets
            .map(|(forward, side)| (Some(round_degrees(forward)), Some(round_degrees(side))))
            .unwrap_or((None, None));
        let semantic_confidence = mean(
            [
                forward.as_ref().map(|axis| axis.confidence),
                side.as_ref().map(|axis| axis.confidence),
                twist.as_ref().map(|axis| axis.confidence),
                bend.as_ref().map(|axis| axis.confidence),
                turn.as_ref().map(|axis| axis.confidence),
            ]
            .into_iter()
            .flatten(),
        );
        let confidence = round_confidence(map_conf * 0.45 + semantic_confidence * 0.55);
        let manual_review_required = confidence < 0.72
            || [
                forward.as_ref(),
                side.as_ref(),
                twist.as_ref(),
                bend.as_ref(),
                turn.as_ref(),
            ]
            .into_iter()
            .flatten()
            .any(|axis| axis.confidence < 0.75);
        if manual_review_required {
            diagnostics.push(ModelInspectionDiagnostic {
                severity: "warning".to_string(), code: "bone_axis_review_required".to_string(),
                message: format!("Semantic axis proposal for '{}' is not geometrically decisive ({confidence:.2}).", spec.id),
                recommendation: format!("Preview {} with a +20 degree semantic action and flip the sign if motion is reversed.", spec.id),
                canonical_bone: Some(spec.id.to_string()),
            });
        }
        proposals.push(BoneAxisProposal {
            canonical_bone: spec.id.to_string(),
            source_joint,
            forward,
            side,
            twist,
            bend,
            turn,
            rest_forward,
            rest_side,
            confidence,
            manual_review_required,
        });
    }
    proposals
}

fn axis_alignment(axes: [[f32; 3]; 3], desired: [f32; 3], evidence: &str) -> SemanticAxisProposal {
    choose_axis(axes.map(|axis| dot(axis, desired)), evidence)
}

fn axis_for_rotation_plane(
    axes: [[f32; 3]; 3],
    from: [f32; 3],
    to: [f32; 3],
    evidence: &str,
) -> Option<SemanticAxisProposal> {
    let rotation_axis = normalize(cross(from, to))?;
    Some(axis_alignment(axes, rotation_axis, evidence))
}

fn axis_for_probed_motion(
    node_index: usize,
    child_index: usize,
    nodes: &[GlbNodeData],
    matrices: &[[f32; 16]],
    long: [f32; 3],
    desired_motion: [f32; 3],
    evidence: &str,
) -> Option<SemanticAxisProposal> {
    const PROBE_DEGREES: f32 = 10.0;
    let desired = reject(desired_motion, long)
        .and_then(normalize)
        .unwrap_or(desired_motion);
    let base_child = matrix_translation(matrices.get(child_index)?);
    let mut scores = [0.0; 3];
    for (axis, score) in scores.iter_mut().enumerate() {
        let probed = world_matrices_with_rotation_probe(nodes, node_index, axis, PROBE_DEGREES);
        let motion = sub(matrix_translation(probed.get(child_index)?), base_child);
        *score = normalize(motion).map_or(0.0, |motion| dot(motion, desired));
    }
    Some(choose_axis(
        scores,
        &format!("{evidence}; measured with a +{PROBE_DEGREES:.0} degree renderer-order probe"),
    ))
}

fn solve_arm_rest_offsets(
    node_index: usize,
    child_index: usize,
    nodes: &[GlbNodeData],
    basis: BodyBasis,
    forward_axis: &SemanticAxisProposal,
    side_axis: &SemanticAxisProposal,
) -> Option<(f32, f32)> {
    let (forward_index, forward_sign) = raw_axis_components(&forward_axis.raw)?;
    let (side_index, side_sign) = raw_axis_components(&side_axis.raw)?;
    if forward_index == side_index {
        return None;
    }
    let target = scale(basis.up, -1.0);
    let mut best = (0.0, 0.0, f32::NEG_INFINITY);
    for forward in (-60..=60).step_by(5) {
        for side in (-120..=30).step_by(5) {
            let score = rest_direction_score(
                nodes,
                node_index,
                child_index,
                target,
                (forward_index, forward_sign, forward as f32),
                (side_index, side_sign, side as f32),
            )?;
            if score > best.2 {
                best = (forward as f32, side as f32, score);
            }
        }
    }
    for step in [1.0, 0.2] {
        let center = best;
        for forward_step in -5..=5 {
            for side_step in -5..=5 {
                let forward = center.0 + forward_step as f32 * step;
                let side = center.1 + side_step as f32 * step;
                let score = rest_direction_score(
                    nodes,
                    node_index,
                    child_index,
                    target,
                    (forward_index, forward_sign, forward),
                    (side_index, side_sign, side),
                )?;
                if score > best.2 {
                    best = (forward, side, score);
                }
            }
        }
    }
    Some((best.0, best.1))
}

fn rest_direction_score(
    nodes: &[GlbNodeData],
    node_index: usize,
    child_index: usize,
    target: [f32; 3],
    forward: (usize, f32, f32),
    side: (usize, f32, f32),
) -> Option<f32> {
    let mut rotation = [0.0; 3];
    rotation[forward.0] += forward.1 * forward.2;
    rotation[side.0] += side.1 * side.2;
    let matrices = world_matrices_with_rotation_override(nodes, node_index, rotation);
    let start = matrix_translation(matrices.get(node_index)?);
    let end = matrix_translation(matrices.get(child_index)?);
    let direction = normalize(sub(end, start))?;
    let regularization = (forward.2.abs() + side.2.abs()) * 0.00001;
    Some(dot(direction, target) - regularization)
}

fn raw_axis_components(raw: &str) -> Option<(usize, f32)> {
    let (axis, sign) = raw.split_once(':')?;
    let index = match axis {
        "rotationX" => 0,
        "rotationY" => 1,
        "rotationZ" => 2,
        _ => return None,
    };
    Some((index, sign.parse::<f32>().ok()?.signum()))
}

fn choose_axis(scores: [f32; 3], evidence: &str) -> SemanticAxisProposal {
    let mut ranked = [(0_usize, scores[0]), (1, scores[1]), (2, scores[2])];
    ranked.sort_by(|left, right| right.1.abs().total_cmp(&left.1.abs()));
    let (index, score) = ranked[0];
    let confidence =
        (score.abs() * 0.78 + (score.abs() - ranked[1].1.abs()).max(0.0) * 0.22).clamp(0.0, 1.0);
    SemanticAxisProposal {
        raw: format!(
            "rotation{}:{}",
            ['X', 'Y', 'Z'][index],
            if score >= 0.0 { 1 } else { -1 }
        ),
        confidence: round_confidence(confidence),
        evidence: evidence.to_string(),
    }
}

fn canonical_child(id: &str) -> Option<&'static str> {
    Some(match id {
        "hips" => "spine",
        "spine" => "chest",
        "chest" => "upper_chest",
        "upper_chest" => "neck",
        "neck" => "head",
        "shoulder_l" => "upper_arm_l",
        "upper_arm_l" => "forearm_l",
        "forearm_l" => "hand_l",
        "shoulder_r" => "upper_arm_r",
        "upper_arm_r" => "forearm_r",
        "forearm_r" => "hand_r",
        "upper_leg_l" => "lower_leg_l",
        "lower_leg_l" => "foot_l",
        "foot_l" => "toe_l",
        "upper_leg_r" => "lower_leg_r",
        "lower_leg_r" => "foot_r",
        "foot_r" => "toe_r",
        "thumb_1_l" => "thumb_2_l",
        "thumb_2_l" => "thumb_3_l",
        "index_1_l" => "index_2_l",
        "index_2_l" => "index_3_l",
        "middle_1_l" => "middle_2_l",
        "middle_2_l" => "middle_3_l",
        "ring_1_l" => "ring_2_l",
        "ring_2_l" => "ring_3_l",
        "pinky_1_l" => "pinky_2_l",
        "pinky_2_l" => "pinky_3_l",
        "thumb_1_r" => "thumb_2_r",
        "thumb_2_r" => "thumb_3_r",
        "index_1_r" => "index_2_r",
        "index_2_r" => "index_3_r",
        "middle_1_r" => "middle_2_r",
        "middle_2_r" => "middle_3_r",
        "ring_1_r" => "ring_2_r",
        "ring_2_r" => "ring_3_r",
        "pinky_1_r" => "pinky_2_r",
        "pinky_2_r" => "pinky_3_r",
        _ => return None,
    })
}

fn is_limb(id: &str) -> bool {
    id.contains("arm")
        || id.contains("leg")
        || id.contains("hand")
        || id.contains("foot")
        || id.starts_with("shoulder")
        || id.starts_with("thumb_")
        || id.starts_with("index_")
        || id.starts_with("middle_")
        || id.starts_with("ring_")
        || id.starts_with("pinky_")
}
fn is_upper_limb(id: &str) -> bool {
    id.starts_with("upper_arm") || id.starts_with("upper_leg")
}
fn is_bending_joint(id: &str) -> bool {
    id.starts_with("forearm")
        || id.starts_with("lower_leg")
        || id.starts_with("thumb_")
        || id.starts_with("index_")
        || id.starts_with("middle_")
        || id.starts_with("ring_")
        || id.starts_with("pinky_")
}

fn build_profile_dsl(
    asset_label: &str,
    mappings: &[HumanoidBoneProposal],
    axes: &[BoneAxisProposal],
) -> String {
    let mut output = format!(
        "<ModelProfile id=\"auto_humanoid_profile\" kind=\"3d\" model=\"{}\" preset=\"humanoid_v1\">\n  <Retarget preset=\"humanoid_v1\">\n",
        escape_xml(asset_label)
    );
    for mapping in mappings {
        output.push_str(&format!(
            "    <Map from=\"{}\" to=\"{}\" />\n",
            escape_xml(&mapping.source_joint),
            mapping.canonical_bone
        ));
    }
    output.push_str("  </Retarget>\n  <BoneAxisMap>\n");
    for axis in axes {
        let mut attributes = Vec::new();
        if let Some(value) = &axis.forward {
            attributes.push(format!("forward=\"{}\"", value.raw));
        }
        if let Some(value) = &axis.side {
            attributes.push(format!("side=\"{}\"", value.raw));
        }
        if let Some(value) = &axis.twist {
            attributes.push(format!("twist=\"{}\"", value.raw));
        }
        if let Some(value) = &axis.bend {
            attributes.push(format!("bend=\"{}\"", value.raw));
        }
        if let Some(value) = &axis.turn {
            attributes.push(format!("turn=\"{}\"", value.raw));
        }
        if let Some(value) = axis.rest_forward {
            attributes.push(format!("restForward=\"{value:.2}\""));
        }
        if let Some(value) = axis.rest_side {
            attributes.push(format!("restSide=\"{value:.2}\""));
        }
        if !attributes.is_empty() {
            output.push_str(&format!(
                "    <Axis bone=\"{}\" {} />\n",
                axis.canonical_bone,
                attributes.join(" ")
            ));
        }
    }
    output.push_str("  </BoneAxisMap>\n</ModelProfile>");
    output
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub(crate) fn world_matrices(nodes: &[GlbNodeData]) -> Vec<[f32; 16]> {
    world_matrices_from_locals(nodes, &node_local_matrices(nodes))
}

fn world_matrices_with_rotation_probe(
    nodes: &[GlbNodeData],
    node_index: usize,
    axis: usize,
    degrees: f32,
) -> Vec<[f32; 16]> {
    let mut rotation = [0.0; 3];
    rotation[axis.min(2)] = degrees;
    world_matrices_with_rotation_override(nodes, node_index, rotation)
}

fn world_matrices_with_rotation_override(
    nodes: &[GlbNodeData],
    node_index: usize,
    rotation_degrees: [f32; 3],
) -> Vec<[f32; 16]> {
    let mut locals = node_local_matrices(nodes);
    if let Some(local) = locals.get_mut(node_index) {
        let rotation = multiply(
            multiply(
                rotation_matrix(2, rotation_degrees[2].to_radians()),
                rotation_matrix(1, rotation_degrees[1].to_radians()),
            ),
            rotation_matrix(0, rotation_degrees[0].to_radians()),
        );
        *local = multiply(*local, rotation);
    }
    world_matrices_from_locals(nodes, &locals)
}

fn node_local_matrices(nodes: &[GlbNodeData]) -> Vec<[f32; 16]> {
    nodes
        .iter()
        .map(|node| {
            node.matrix
                .unwrap_or_else(|| trs_matrix(node.translation, node.rotation, node.scale))
        })
        .collect()
}

fn world_matrices_from_locals(nodes: &[GlbNodeData], locals: &[[f32; 16]]) -> Vec<[f32; 16]> {
    let mut output = vec![[0.0; 16]; nodes.len()];
    let mut computed = vec![false; nodes.len()];
    let mut visiting = vec![false; nodes.len()];
    for index in 0..nodes.len() {
        compute_world_matrix(
            index,
            nodes,
            locals,
            &mut output,
            &mut computed,
            &mut visiting,
        );
    }
    output
}

fn compute_world_matrix(
    index: usize,
    nodes: &[GlbNodeData],
    locals: &[[f32; 16]],
    output: &mut [[f32; 16]],
    computed: &mut [bool],
    visiting: &mut [bool],
) -> [f32; 16] {
    if computed[index] {
        return output[index];
    }
    if visiting[index] {
        return identity();
    }
    visiting[index] = true;
    let local = locals.get(index).copied().unwrap_or_else(identity);
    let world = nodes[index]
        .parent
        .filter(|parent| *parent < nodes.len())
        .map(|parent| {
            multiply(
                compute_world_matrix(parent, nodes, locals, output, computed, visiting),
                local,
            )
        })
        .unwrap_or(local);
    visiting[index] = false;
    computed[index] = true;
    output[index] = world;
    world
}

fn rotation_matrix(axis: usize, angle: f32) -> [f32; 16] {
    let (sin, cos) = angle.sin_cos();
    match axis {
        0 => [
            1.0, 0.0, 0.0, 0.0, 0.0, cos, sin, 0.0, 0.0, -sin, cos, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        1 => [
            cos, 0.0, -sin, 0.0, 0.0, 1.0, 0.0, 0.0, sin, 0.0, cos, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        _ => [
            cos, sin, 0.0, 0.0, -sin, cos, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    }
}

fn trs_matrix(t: [f32; 3], q: [f32; 4], s: [f32; 3]) -> [f32; 16] {
    let [x, y, z, w] = q;
    let (xx, yy, zz, xy, xz, yz, wx, wy, wz) = (
        x * x,
        y * y,
        z * z,
        x * y,
        x * z,
        y * z,
        w * x,
        w * y,
        w * z,
    );
    [
        (1.0 - 2.0 * (yy + zz)) * s[0],
        (2.0 * (xy + wz)) * s[0],
        (2.0 * (xz - wy)) * s[0],
        0.0,
        (2.0 * (xy - wz)) * s[1],
        (1.0 - 2.0 * (xx + zz)) * s[1],
        (2.0 * (yz + wx)) * s[1],
        0.0,
        (2.0 * (xz + wy)) * s[2],
        (2.0 * (yz - wx)) * s[2],
        (1.0 - 2.0 * (xx + yy)) * s[2],
        0.0,
        t[0],
        t[1],
        t[2],
        1.0,
    ]
}

fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    out
}

fn identity() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}
fn matrix_translation(matrix: &[f32; 16]) -> [f32; 3] {
    [matrix[12], matrix[13], matrix[14]]
}
fn transform_point(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
    ]
}
fn matrix_axes(matrix: &[f32; 16]) -> [[f32; 3]; 3] {
    [
        matrix_axis(matrix, 0).unwrap_or([1.0, 0.0, 0.0]),
        matrix_axis(matrix, 1).unwrap_or([0.0, 1.0, 0.0]),
        matrix_axis(matrix, 2).unwrap_or([0.0, 0.0, 1.0]),
    ]
}
fn matrix_axis(matrix: &[f32; 16], axis: usize) -> Option<[f32; 3]> {
    normalize([matrix[axis * 4], matrix[axis * 4 + 1], matrix[axis * 4 + 2]])
}
fn direction_between(
    a: &str,
    b: &str,
    resolved: &BTreeMap<String, usize>,
    matrices: &[[f32; 16]],
) -> Option<[f32; 3]> {
    let a = matrix_translation(matrices.get(*resolved.get(a)?)?);
    let b = matrix_translation(matrices.get(*resolved.get(b)?)?);
    normalize(sub(b, a))
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f32; 3], value: f32) -> [f32; 3] {
    [a[0] * value, a[1] * value, a[2] * value]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}
fn normalize(a: [f32; 3]) -> Option<[f32; 3]> {
    let length = length(a);
    (length > EPSILON).then(|| scale(a, 1.0 / length))
}
fn reject(a: [f32; 3], normal: [f32; 3]) -> Option<[f32; 3]> {
    let normal = normalize(normal)?;
    Some(sub(a, scale(normal, dot(a, normal))))
}
fn mean(values: impl IntoIterator<Item = f32>) -> f32 {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}
fn round_confidence(value: f32) -> f32 {
    (value.clamp(0.0, 1.0) * 1000.0).round() / 1000.0
}
fn round_degrees(value: f32) -> f32 {
    (value * 100.0).round() / 100.0
}
fn round_vector(value: [f32; 3]) -> [f32; 3] {
    [
        round_degrees(value[0]),
        round_degrees(value[1]),
        round_degrees(value[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_node(index: usize, parent: Option<usize>, translation: [f32; 3]) -> GlbNodeData {
        GlbNodeData {
            index,
            name: Some(format!("node_{index}")),
            parent,
            children: Vec::new(),
            mesh: None,
            skin: None,
            translation,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            matrix: None,
        }
    }

    fn named_test_node(index: usize, name: &str, parent: Option<usize>) -> GlbNodeData {
        let mut node = test_node(index, parent, [0.0, index as f32 * 0.1, 0.0]);
        node.name = Some(name.to_string());
        node
    }

    fn mixamo_test_nodes(prefix: &str) -> Vec<GlbNodeData> {
        let names_and_parents = [
            ("Hips", None),
            ("Spine", Some(0)),
            ("Spine1", Some(1)),
            ("Spine2", Some(2)),
            ("Neck", Some(3)),
            ("Head", Some(4)),
            ("LeftShoulder", Some(3)),
            ("LeftArm", Some(6)),
            ("LeftForeArm", Some(7)),
            ("LeftHand", Some(8)),
            ("RightShoulder", Some(3)),
            ("RightArm", Some(10)),
            ("RightForeArm", Some(11)),
            ("RightHand", Some(12)),
            ("LeftUpLeg", Some(0)),
            ("LeftLeg", Some(14)),
            ("LeftFoot", Some(15)),
            ("LeftToeBase", Some(16)),
            ("RightUpLeg", Some(0)),
            ("RightLeg", Some(18)),
            ("RightFoot", Some(19)),
            ("RightToeBase", Some(20)),
        ];
        names_and_parents
            .into_iter()
            .enumerate()
            .map(|(index, (name, parent))| {
                named_test_node(index, &format!("{prefix}{name}"), parent)
            })
            .collect()
    }

    #[test]
    fn normalizes_blender_joint_suffixes() {
        assert_eq!(normalize_name("Left arm_20"), "left arm 20");
        assert_eq!(detect_side(&["left", "arm"], "leftarm"), Some(Side::Left));
        assert_eq!(
            detect_side(&["source", "leftshoulder", "032"], "sourceleftshoulder032"),
            Some(Side::Left)
        );
        assert_eq!(detect_side(&["shoulder"], "shoulder"), None);
    }

    #[test]
    fn vrm1_metadata_is_authoritative_for_rig_detection() {
        let nodes = vec![
            named_test_node(0, "J_Bip_C_Hips", None),
            named_test_node(1, "J_Bip_L_UpperLeg", Some(0)),
            named_test_node(2, "J_Bip_L_LowerLeg", Some(1)),
        ];
        let document = serde_json::json!({
            "extensions": {
                "VRMC_vrm": {
                    "humanoid": {
                        "humanBones": {
                            "hips": { "node": 0 },
                            "leftUpperLeg": { "node": 1 },
                            "leftLowerLeg": { "node": 2 }
                        }
                    }
                }
            }
        });
        let (rig, mappings) = detect_humanoid_rig(&document, &nodes);
        assert_eq!(rig.family, "vrm_1");
        assert_eq!(rig.mapping_source, "metadata");
        assert_eq!(mappings["upper_leg_l"].node_index, 1);
    }

    #[test]
    fn vrm0_metadata_is_authoritative_for_rig_detection() {
        let nodes = vec![
            named_test_node(0, "J_Bip_C_Hips", None),
            named_test_node(1, "J_Bip_R_UpperLeg", Some(0)),
            named_test_node(2, "J_Bip_R_LowerLeg", Some(1)),
        ];
        let document = serde_json::json!({
            "extensions": {
                "VRM": {
                    "humanoid": {
                        "humanBones": [
                            { "bone": "hips", "node": 0 },
                            { "bone": "rightUpperLeg", "node": 1 },
                            { "bone": "rightLowerLeg", "node": 2 }
                        ]
                    }
                }
            }
        });
        let (rig, mappings) = detect_humanoid_rig(&document, &nodes);
        assert_eq!(rig.family, "vrm_0");
        assert_eq!(rig.mapping_source, "metadata");
        assert_eq!(mappings["lower_leg_r"].node_index, 2);
    }

    #[test]
    fn mixamo_namespace_and_hierarchy_select_known_preset() {
        let nodes = mixamo_test_nodes("mixamorig:");
        let (rig, mappings) = detect_humanoid_rig(&serde_json::json!({}), &nodes);
        assert_eq!(rig.family, "mixamo");
        assert_eq!(rig.mapping_source, "preset");
        assert_eq!(mappings["forearm_r"].node_index, 12);
        assert!(rig.core_bone_count >= 12);
    }

    #[test]
    fn ready_player_me_generator_labels_mixamo_compatible_hierarchy() {
        let nodes = mixamo_test_nodes("");
        let document = serde_json::json!({
            "asset": { "generator": "Ready Player Me Avatar API" }
        });
        let (rig, _) = detect_humanoid_rig(&document, &nodes);
        assert_eq!(rig.family, "ready_player_me");
        assert_eq!(rig.mapping_source, "preset");
    }

    #[test]
    fn profile_proposal_is_scene_model_profile_dsl() {
        let mappings = vec![HumanoidBoneProposal {
            canonical_bone: "head".into(),
            source_joint: "Head_47".into(),
            node_index: 1,
            confidence: 1.0,
            evidence: vec![],
            alternatives: vec![],
        }];
        let axes = vec![BoneAxisProposal {
            canonical_bone: "head".into(),
            source_joint: "Head_47".into(),
            forward: None,
            side: None,
            twist: None,
            bend: None,
            turn: Some(SemanticAxisProposal {
                raw: "rotationY:1".into(),
                confidence: 1.0,
                evidence: "test".into(),
            }),
            rest_forward: None,
            rest_side: None,
            confidence: 1.0,
            manual_review_required: false,
        }];
        let dsl = build_profile_dsl("girl_asset", &mappings, &axes);
        assert!(dsl.contains("<Map from=\"Head_47\" to=\"head\" />"));
        assert!(dsl.contains("turn=\"rotationY:1\""));
    }

    #[test]
    fn finite_probe_matches_renderer_override_order() {
        let mut root = test_node(0, None, [0.0, 0.0, 0.0]);
        root.children.push(1);
        let child = test_node(1, Some(0), [0.0, 1.0, 0.0]);
        let nodes = vec![root, child];
        let matrices = world_matrices(&nodes);
        let proposal = axis_for_probed_motion(
            0,
            1,
            &nodes,
            &matrices,
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            "test",
        )
        .expect("axis proposal");
        assert_eq!(proposal.raw, "rotationX:1");
    }

    #[test]
    fn toe_direction_disambiguates_character_forward() {
        let mut resolved = BTreeMap::new();
        for (name, index) in [
            ("hips", 0),
            ("head", 1),
            ("shoulder_l", 2),
            ("shoulder_r", 3),
            ("foot_l", 4),
            ("toe_l", 5),
            ("foot_r", 6),
            ("toe_r", 7),
        ] {
            resolved.insert(name.to_string(), index);
        }
        let matrices = [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.5, 0.0],
            [-1.0, 1.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 1.0],
            [-0.5, 0.0, 0.0],
            [-0.5, 0.0, 1.0],
        ]
        .map(|translation| {
            let mut matrix = identity();
            matrix[12] = translation[0];
            matrix[13] = translation[1];
            matrix[14] = translation[2];
            matrix
        });
        let mut diagnostics = Vec::new();
        let basis = infer_body_basis(&resolved, &matrices, &mut diagnostics);
        assert!(basis.forward[2] > 0.99);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn optional_bone_is_required_only_when_action_uses_it() {
        let maps = HUMANOID_BONES
            .iter()
            .filter(|bone| is_core_bone(bone.id))
            .map(|bone| format!("<Map from=\"{}\" to=\"{}\" />", bone.id, bone.id))
            .collect::<String>();
        let script = format!(
            r##"<Graph fps={{30}} duration="1s" size={{[64,64]}}>
  <ModelProfile id="target" kind="3d" model="model" preset="humanoid_v1">
    <Retarget preset="humanoid_v1">{maps}</Retarget>
  </ModelProfile>
  <Action id="gesture" skeleton="humanoid_v1" duration="1s">
    <Pose t="0s"><Bone id="index_1_l" bend="20" /></Pose>
  </Action>
  <World id="stage"><Background color="#000000" /></World>
  <Present from="stage" />
</Graph>"##
        );
        let graph = crate::world::parse_world_graph_script(&script).expect("compatibility graph");
        let report =
            inspect_humanoid_action_compatibility(&graph.actions[0], &graph.model_profiles[0]);
        assert!(report.missing_required_bones.is_empty());
        assert_eq!(report.missing_action_bones, ["index_1_l"]);
        assert!(!report.full_fidelity);
    }
}
