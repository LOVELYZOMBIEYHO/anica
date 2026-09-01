// =========================================
// =========================================
// crates/motionloom-action-tool/src/target.rs

//! Target-aware offline authoring. Nothing in this module runs in MotionLoom.
//! Reports distinguish a tool-side reconstruction from independent runtime evidence.

use crate::{source::*, *};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
#[path = "target_reference.rs"]
mod reference;

#[derive(Debug, Clone, PartialEq)]
pub struct TargetOptions {
    pub model: PathBuf,
    pub profile: PathBuf,
    pub profile_id: String,
    /// Preserve source metres, or scale motion by the rest hip-to-foot ratio.
    pub proportional: bool,
    pub actor_height: f32,
    pub max_position_mm: f32,
    pub max_rotation_deg: f32,
    pub editor_safe: bool,
}

impl TargetOptions {
    pub fn new(model: PathBuf, profile: PathBuf, profile_id: String) -> Self {
        Self {
            model,
            profile,
            profile_id,
            proportional: true,
            actor_height: 1.82,
            max_position_mm: 1.0,
            max_rotation_deg: 0.1,
            editor_safe: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FidelityReport {
    pub source_sha256: String,
    pub target_sha256: String,
    pub profile_sha256: String,
    pub action_sha256: String,
    pub profile_id: String,
    pub clip: String,
    pub duration_sec: f32,
    pub trajectory_scale: f32,
    pub actor_height: f32,
    pub mapped_bones: usize,
    pub output_poses: usize,
    pub evaluated_samples: usize,
    pub editor_millisecond_collisions: usize,
    pub time_grid: String,
    pub tool_reconstruction_pass: bool,
    pub tool_reconstruction: Metrics,
    pub max_position_mm: f32,
    pub max_rotation_deg: f32,
    pub source_reference: String,
    pub source_comparison: Option<reference::SourceComparison>,
    pub native_runtime: String,
    pub native_reconstruction: Metrics,
    pub native_evaluated_samples: usize,
    #[serde(default)]
    pub joint_trajectory_audit: Option<JointTrajectoryAudit>,
    pub wasm_runtime: String,
    pub strict_pass: bool,
    pub limitations: Vec<String>,
}

/// Read-only target-space trajectory evidence. This deliberately reports joint
/// centres rather than claiming skin-surface or collider contact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointTrajectoryAudit {
    pub sample_hz: u32,
    pub low_band_m: f32,
    pub slow_speed_mps: f32,
    pub effectors: Vec<EffectorTrajectory>,
    pub interpretation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectorTrajectory {
    pub bone: String,
    pub samples: usize,
    pub min_y_m: f32,
    pub max_y_m: f32,
    pub max_speed_mps: f32,
    pub low_and_slow_samples: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    pub max_position_mm: f32,
    pub max_rotation_deg: f32,
    pub worst_bone: String,
    pub worst_time_sec: f32,
}

fn err(message: impl Into<String>) -> ActionToolError {
    ActionToolError::Target {
        message: message.into(),
    }
}

pub fn fingerprint(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Audit the legacy native FBX decoder without loading a target model.
pub fn audit_fbx_source(path: &Path) -> Result<Vec<reference::SourceComparison>, ActionToolError> {
    let source = crate::fbx_source::load(path)?;
    let mapping = canonical_node_mappings(&source);
    let limits = TargetOptions::new(PathBuf::new(), PathBuf::new(), String::new());
    source
        .clips
        .iter()
        .map(|clip| {
            reference::compare(&source, clip, &mapping, &limits)?
                .ok_or_else(|| err("expected FBX source"))
        })
        .collect()
}

/// Combine separately-produced WASM evidence without trusting filenames.
pub fn certify_report(
    report_bytes: &[u8],
    wasm_bytes: &[u8],
) -> Result<FidelityReport, ActionToolError> {
    let mut report: FidelityReport =
        serde_json::from_slice(report_bytes).map_err(|e| err(format!("fidelity report: {e}")))?;
    let evidence: serde_json::Value =
        serde_json::from_slice(wasm_bytes).map_err(|e| err(format!("WASM evidence: {e}")))?;
    let value = |key: &str| {
        evidence
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
    };
    let passed = evidence.get("passed").and_then(|v| v.as_bool()) == Some(true)
        && value("action_sha256") == report.action_sha256
        && value("target_sha256") == report.target_sha256;
    report.wasm_runtime = if passed { "passed" } else { "failed" }.into();
    report.strict_pass = passed
        && report.source_reference == "passed"
        && report.native_runtime == "passed"
        && report.tool_reconstruction_pass;
    if !passed {
        report.limitations.push(
            "WASM evidence failed or its Action/target hashes do not match this report.".into(),
        );
    }
    Ok(report)
}

fn read(path: &Path) -> Result<Vec<u8>, ActionToolError> {
    fs::read(path).map_err(|e| err(format!("{}: {e}", path.display())))
}

// Extract existing profile declarations, not the scene's unfinished Action bindings.
// The MotionLoom parser remains authoritative for all profile/attribute semantics.
fn profile_document(text: &str) -> Result<String, ActionToolError> {
    let mut clean = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<!--") {
        clean.push_str(&rest[..start]);
        let end = rest[start + 4..]
            .find("-->")
            .ok_or_else(|| err("unterminated profile comment"))?;
        rest = &rest[start + 4 + end + 3..];
    }
    clean.push_str(rest);
    let mut profiles = String::new();
    let mut rest = clean.as_str();
    while let Some(start) = rest.find("<ModelProfile") {
        rest = &rest[start..];
        if !rest.as_bytes().get(13).is_some_and(u8::is_ascii_whitespace) {
            return Err(err("invalid ModelProfile opening tag"));
        }
        let end = rest
            .find("</ModelProfile>")
            .ok_or_else(|| err("ModelProfile block not closed"))?
            + 15;
        profiles.push_str(&rest[..end]);
        profiles.push('\n');
        rest = &rest[end..];
    }
    Ok(format!(
        "<Graph fps={{30}} duration=\"1s\" size={{[64,64]}}>\n{profiles}<Background color=\"#000000\" />\n<Present from=\"scene\" />\n</Graph>"
    ))
}

/// Rigid transform with positive uniform scale; unsupported affine data is rejected.
#[derive(Debug, Clone, Copy)]
struct Xform {
    p: [f32; 3],
    q: [f32; 4],
    s: f32,
}
impl Xform {
    fn identity() -> Self {
        Self {
            p: [0.; 3],
            q: [0., 0., 0., 1.],
            s: 1.,
        }
    }
    fn point(self, p: [f32; 3]) -> [f32; 3] {
        add3(self.p, rotate_vec3(self.q, p.map(|v| v * self.s)))
    }
    fn inverse_point(self, p: [f32; 3]) -> [f32; 3] {
        rotate_vec3(quat_conjugate(self.q), sub3(p, self.p)).map(|v| v / self.s)
    }
    fn then(self, child: Self) -> Self {
        Self {
            p: self.point(child.p),
            q: quat_normalize(quat_mul(self.q, child.q)),
            s: self.s * child.s,
        }
    }
}

#[derive(Debug, Clone)]
struct Axis {
    channel: &'static str,
    index: usize,
    multiplier: f32,
}
struct Rig {
    mesh: motionloom::GlbMeshData,
    nodes: Vec<AnimationNode>,
    order: Vec<usize>,
    mapping: BTreeMap<String, usize>,
    rest: Vec<Xform>,
    corrections: BTreeMap<String, [f32; 3]>,
    axes: BTreeMap<String, Vec<Axis>>,
    bounds_height: f32,
    model_hash: String,
    profile_hash: String,
}

fn uniform(scale: [f32; 3]) -> Result<f32, ActionToolError> {
    if scale.iter().any(|s| !s.is_finite() || *s <= 0.)
        || scale.iter().any(|s| (*s - scale[0]).abs() > 0.00001)
    {
        return Err(err(
            "non-uniform/reflected scale is not supported by the target solver",
        ));
    }
    Ok(scale[0])
}

fn node_xform(n: &AnimationNode) -> Result<Xform, ActionToolError> {
    Ok(Xform {
        p: n.rest_translation,
        q: quat_checked(n.rest_rotation)?,
        s: uniform(n.rest_scale)?,
    })
}

fn order(nodes: &[AnimationNode]) -> Vec<usize> {
    let mut indices = (0..nodes.len()).collect::<Vec<_>>();
    indices.sort_by_key(|i| {
        let mut depth = 0;
        let mut p = nodes[*i].parent;
        while let Some(i) = p {
            depth += 1;
            p = nodes[i].parent;
        }
        depth
    });
    indices
}

fn rest_world(nodes: &[AnimationNode], order: &[usize]) -> Result<Vec<Xform>, ActionToolError> {
    let mut result = vec![Xform::identity(); nodes.len()];
    for &i in order {
        result[i] = nodes[i]
            .parent
            .map(|p| result[p])
            .unwrap_or(Xform::identity())
            .then(node_xform(&nodes[i])?);
    }
    Ok(result)
}

fn binding(value: &str, channel: &'static str) -> Result<Axis, ActionToolError> {
    let (axis, multiplier) = value
        .split_once(':')
        .ok_or_else(|| err(format!("invalid axis {value}")))?;
    let index = match axis {
        "rotationX" => 0,
        "rotationY" => 1,
        "rotationZ" => 2,
        _ => return Err(err(format!("unsupported axis {value}"))),
    };
    let multiplier: f32 = multiplier
        .parse()
        .map_err(|_| err(format!("non-constant axis {value}")))?;
    if !multiplier.is_finite() || multiplier.abs() < 1e-6 {
        return Err(err(format!("invalid axis factor {value}")));
    }
    Ok(Axis {
        channel,
        index,
        multiplier,
    })
}

impl Rig {
    fn load(options: &TargetOptions) -> Result<Self, ActionToolError> {
        let model_bytes = read(&options.model)?;
        let profile_bytes = read(&options.profile)?;
        let text = std::str::from_utf8(&profile_bytes).map_err(|e| err(e.to_string()))?;
        let graph = motionloom::parse_graph_script(&profile_document(text)?)
            .map_err(|e| err(format!("target profile DSL: {e}")))?;
        let profile = graph
            .model_profiles
            .iter()
            .find(|p| p.id == options.profile_id)
            .ok_or_else(|| err("target profile id not found"))?;
        if profile.preset != "humanoid_v1" {
            return Err(err("target preset must be humanoid_v1"));
        }
        let mesh = motionloom::experimental::load_glb_mesh_data(&options.model)?;
        if mesh.nodes.iter().any(|n| n.matrix.is_some()) {
            return Err(err("target matrix nodes require explicit TRS baking"));
        }
        let nodes = crate::gltf_source::from_mesh(&options.model, &mesh).nodes;
        let source = AnimationSource {
            path: options.model.clone(),
            backend: "target".into(),
            world_basis: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            nodes: nodes.clone(),
            clips: vec![],
            diagnostics: vec![],
        };
        validate_source(&source)?;
        let mut mapping = BTreeMap::new();
        let maps = &profile
            .retarget
            .as_ref()
            .ok_or_else(|| err("target needs explicit Retarget mappings"))?
            .maps;
        for map in maps {
            let matches = nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.name == map.from)
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err(err(format!(
                    "target node {} is missing or ambiguous",
                    map.from
                )));
            }
            let index = matches[0].0;
            if mapping.values().any(|i| *i == index)
                || mapping.insert(map.to.clone(), index).is_some()
            {
                return Err(err("duplicate target bone mapping"));
            }
        }
        if !mapping.contains_key("hips") {
            return Err(err("target hips mapping is required"));
        }
        let mut corrections = BTreeMap::new();
        let mut axes = BTreeMap::new();
        if let Some(map) = &profile.bone_axis_map {
            for axis in &map.axes {
                let mut rest = [0.; 3];
                let mut bindings = vec![];
                // Prefer anatomical channels once per axis; residual axes stay raw.
                for (channel, b, r) in [
                    ("bend", &axis.bend, &axis.rest_bend),
                    ("forward", &axis.forward, &axis.rest_forward),
                    ("side", &axis.side, &axis.rest_side),
                    ("turn", &axis.turn, &axis.rest_turn),
                    ("twist", &axis.twist, &axis.rest_twist),
                ] {
                    if let Some(b) = b {
                        let a = binding(b, channel)?;
                        if let Some(r) = r {
                            let v: f32 = r.parse().map_err(|_| {
                                err("animated/non-numeric rest calibration is unsupported")
                            })?;
                            if !v.is_finite() {
                                return Err(err("non-finite rest calibration"));
                            }
                            rest[a.index] += v * a.multiplier;
                        }
                        bindings.push(a);
                    } else if r.is_some() {
                        return Err(err("rest calibration has no axis binding"));
                    }
                }
                if corrections.insert(axis.bone.clone(), rest).is_some() {
                    return Err(err("duplicate BoneAxisMap bone"));
                }
                axes.insert(axis.bone.clone(), bindings);
            }
        }
        let order = order(&nodes);
        let rest = rest_world(&nodes, &order)?;
        Ok(Self {
            mesh: mesh.clone(),
            nodes,
            order,
            mapping,
            rest,
            corrections,
            axes,
            bounds_height: (mesh.bounds_max[1] - mesh.bounds_min[1]).abs(),
            model_hash: fingerprint(&model_bytes),
            profile_hash: fingerprint(&profile_bytes),
        })
    }
}

fn euler_quat(r: [f32; 3]) -> [f32; 4] {
    let [x, y, z] = r.map(|v| v.to_radians() * 0.5);
    quat_normalize(quat_mul(
        quat_mul([0., 0., z.sin(), z.cos()], [0., y.sin(), 0., y.cos()]),
        [x.sin(), 0., 0., x.cos()],
    ))
}

fn basis_determinant(b: [[f32; 3]; 3]) -> f32 {
    b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0])
}

fn source_world(
    source: &AnimationSource,
    clip: &AnimationClip,
    time: f32,
    order: &[usize],
) -> Result<Vec<Xform>, ActionToolError> {
    let mut result = vec![Xform::identity(); source.nodes.len()];
    for &i in order {
        let n = sample_node(clip, i, time, &source.nodes[i]);
        let local = Xform {
            p: n.translation,
            q: quat_checked(n.rotation)?,
            s: uniform(n.scale)?,
        };
        result[i] = source.nodes[i]
            .parent
            .map(|p| result[p])
            .unwrap_or(Xform::identity())
            .then(local);
    }
    let det = basis_determinant(source.world_basis);
    let orthogonal = (0..3).all(|i| {
        (0..3).all(|j| {
            let dot: f32 = source.world_basis[i]
                .iter()
                .zip(source.world_basis[j])
                .map(|(a, b)| a * b)
                .sum();
            (dot - if i == j { 1. } else { 0. }).abs() < 1e-4
        })
    });
    if !orthogonal || (det.abs() - 1.).abs() > 1e-4 {
        return Err(err("source basis is not orthogonal"));
    }
    for node in &mut result {
        node.p = world_vector(source, node.p);
        // Quaternion vector parts are axial vectors under a reflected basis.
        let v = world_vector(source, [node.q[0], node.q[1], node.q[2]]).map(|v| v * det);
        node.q = quat_normalize([v[0], v[1], v[2], node.q[3]]);
    }
    Ok(result)
}

struct Solver<'a> {
    source: &'a AnimationSource,
    clip: &'a AnimationClip,
    rig: &'a Rig,
    mapping: BTreeMap<String, usize>,
    source_order: Vec<usize>,
    source_rest: Vec<Xform>,
    aligned_rest: Vec<[f32; 4]>,
    motion_scale: f32,
}

impl Solver<'_> {
    fn reference(&self, time: f32) -> Result<Vec<Xform>, ActionToolError> {
        let animated = source_world(self.source, self.clip, time, &self.source_order)?;
        let hips = self.rig.mapping["hips"];
        let source_hips = self.mapping["hips"];
        let mut desired = vec![Xform::identity(); self.rig.nodes.len()];
        let reverse = self
            .rig
            .mapping
            .iter()
            .map(|(b, i)| (*i, b))
            .collect::<BTreeMap<_, _>>();
        for &i in &self.rig.order {
            let parent = self.rig.nodes[i]
                .parent
                .map(|p| desired[p])
                .unwrap_or(Xform::identity());
            desired[i] = parent.then(node_xform(&self.rig.nodes[i])?);
            if let Some(bone) = reverse.get(&i) {
                if let Some(&src) = self.mapping.get(*bone) {
                    // Retarget global rest-relative motion, not source local Euler axes.
                    desired[i].q = quat_normalize(quat_mul(
                        quat_mul(animated[src].q, quat_conjugate(self.source_rest[src].q)),
                        self.aligned_rest[i],
                    ));
                }
            }
            if i == hips {
                desired[i].p = add3(
                    self.rig.rest[hips].p,
                    sub3(animated[source_hips].p, self.source_rest[source_hips].p)
                        .map(|v| v * self.motion_scale),
                );
            }
        }
        Ok(desired)
    }

    fn encode(
        &self,
        time: f32,
        previous: &mut BTreeMap<String, [f32; 3]>,
    ) -> Result<PoseSample, ActionToolError> {
        let desired = self.reference(time)?;
        let mut bones = BTreeMap::new();
        for (bone, &i) in &self.rig.mapping {
            if !self.mapping.contains_key(bone) {
                continue;
            }
            let parent = self.rig.nodes[i]
                .parent
                .map(|p| desired[p])
                .unwrap_or(Xform::identity());
            let base = parent.then(node_xform(&self.rig.nodes[i])?);
            let q = quat_mul(quat_conjugate(base.q), desired[i].q);
            let mut angles = quat_to_euler_xyz_deg(q);
            if let Some(last) = previous.get(bone) {
                angles = continuous_euler(angles, *last);
            }
            previous.insert(bone.clone(), angles);
            // Runtime adds calibration Euler angles before base * override.
            let delta = sub3(
                angles,
                self.rig.corrections.get(bone).copied().unwrap_or([0.; 3]),
            );
            let mut values = BTreeMap::new();
            let mut used = [false; 3];
            if let Some(axes) = self.rig.axes.get(bone) {
                for a in axes {
                    if !used[a.index] {
                        values.insert(a.channel, delta[a.index] / a.multiplier);
                        used[a.index] = true;
                    }
                }
            }
            for (i, key) in ["rotationX", "rotationY", "rotationZ"]
                .into_iter()
                .enumerate()
            {
                if !used[i] {
                    values.insert(key, delta[i]);
                }
            }
            if bone == "hips" {
                let translation = base.inverse_point(desired[i].p);
                for (key, value) in ["x", "y", "z"].into_iter().zip(translation) {
                    values.insert(key, value);
                }
            }
            bones.insert(bone.clone(), values);
        }
        Ok(PoseSample { time, bones })
    }
}

fn direction_child(bone: &str) -> Option<String> {
    let direct = match bone {
        "hips" => "spine",
        "spine" => "chest",
        "chest" => "upper_chest",
        "upper_chest" => "neck",
        "neck" => "head",
        "shoulder_l" => "upper_arm_l",
        "shoulder_r" => "upper_arm_r",
        "upper_arm_l" => "forearm_l",
        "upper_arm_r" => "forearm_r",
        "forearm_l" => "hand_l",
        "forearm_r" => "hand_r",
        "hand_l" => "middle_1_l",
        "hand_r" => "middle_1_r",
        "upper_leg_l" => "lower_leg_l",
        "upper_leg_r" => "lower_leg_r",
        "lower_leg_l" => "foot_l",
        "lower_leg_r" => "foot_r",
        "foot_l" => "toe_l",
        "foot_r" => "toe_r",
        _ => "",
    };
    if !direct.is_empty() {
        return Some(direct.into());
    }
    for finger in ["thumb", "index", "middle", "ring", "pinky"] {
        for side in ["l", "r"] {
            for joint in 1..3 {
                if bone == format!("{finger}_{joint}_{side}") {
                    return Some(format!("{finger}_{}_{side}", joint + 1));
                }
            }
        }
    }
    None
}

fn align_direction(from: [f32; 3], to: [f32; 3]) -> Result<[f32; 4], ActionToolError> {
    let a_len = length3(from);
    let b_len = length3(to);
    if a_len < 1e-6 || b_len < 1e-6 {
        return Err(err(
            "zero-length mapped bone cannot establish an anatomical axis",
        ));
    }
    let a = from.map(|v| v / a_len);
    let b = to.map(|v| v / b_len);
    let dot = a
        .iter()
        .zip(b)
        .map(|(a, b)| a * b)
        .sum::<f32>()
        .clamp(-1., 1.);
    if dot < -0.9999 {
        return Err(err(
            "opposite rest-bone directions need an explicit reference-pose calibration",
        ));
    }
    quat_checked([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
        1. + dot,
    ])
}

fn quat_checked(q: [f32; 4]) -> Result<[f32; 4], ActionToolError> {
    if q.iter().any(|v| !v.is_finite()) || q.iter().map(|v| v * v).sum::<f32>() < 1e-10 {
        return Err(err("invalid rest quaternion"));
    }
    Ok(quat_normalize(q))
}

// Align each limb's rest direction (T/A pose), preserving its bind twist.
// A single direction cannot determine roll; that ambiguity remains in the report.
fn align_rest(
    rig: &Rig,
    mapping: &BTreeMap<String, usize>,
    source_rest: &[Xform],
) -> Result<Vec<[f32; 4]>, ActionToolError> {
    let mut aligned = rig.rest.iter().map(|n| n.q).collect::<Vec<_>>();
    for (bone, &i) in &rig.mapping {
        let Some(child) = direction_child(bone) else {
            continue;
        };
        let (Some(&src), Some(&src_child), Some(&target_child)) = (
            mapping.get(bone),
            mapping.get(&child),
            rig.mapping.get(&child),
        ) else {
            continue;
        };
        let swing = align_direction(
            sub3(rig.rest[target_child].p, rig.rest[i].p),
            sub3(source_rest[src_child].p, source_rest[src].p),
        )?;
        aligned[i] = quat_mul(swing, rig.rest[i].q);
    }
    Ok(aligned)
}

fn rotation_distance(a: [f32; 4], b: [f32; 4]) -> f32 {
    // atan2 of the relative quaternion is stable near zero (acos(dot) is not).
    let q = quat_normalize(quat_mul(quat_conjugate(a), b));
    2. * length3([q[0], q[1], q[2]]).atan2(q[3].abs()).to_degrees()
}

/// Independent reconstruction of the serialized channels, including rest offsets.
/// This is deliberately labelled tool-side, not a MotionLoom runtime assertion.
fn reconstruct(rig: &Rig, poses: &[PoseSample], time: f32) -> Result<Vec<Xform>, ActionToolError> {
    let k = poses.partition_point(|p| p.time <= time).saturating_sub(1);
    let a = &poses[k];
    let b = &poses[(k + 1).min(poses.len() - 1)];
    let t = if b.time > a.time {
        ((time - a.time) / (b.time - a.time)).clamp(0., 1.)
    } else {
        0.
    };
    let reverse = rig
        .mapping
        .iter()
        .map(|(b, i)| (*i, b))
        .collect::<BTreeMap<_, _>>();
    let mut world = vec![Xform::identity(); rig.nodes.len()];
    for &i in &rig.order {
        let mut local = node_xform(&rig.nodes[i])?;
        if let Some(bone) = reverse.get(&i) {
            let mut r = rig.corrections.get(*bone).copied().unwrap_or([0.; 3]);
            let mut p = [0.; 3];
            let av = a.bones.get(*bone);
            let bv = b.bones.get(*bone);
            let get = |key: &str| {
                let x = av.and_then(|b| b.get(key)).copied().unwrap_or(0.);
                let y = bv.and_then(|b| b.get(key)).copied().unwrap_or(x);
                x + (y - x) * t
            };
            for (j, key) in ["rotationX", "rotationY", "rotationZ"].iter().enumerate() {
                r[j] += get(key);
            }
            for (j, key) in ["x", "y", "z"].iter().enumerate() {
                p[j] = get(key);
            }
            if let Some(axes) = rig.axes.get(*bone) {
                for axis in axes {
                    r[axis.index] += get(axis.channel) * axis.multiplier;
                }
            }
            local = local.then(Xform {
                p,
                q: euler_quat(r),
                s: 1.,
            });
        }
        world[i] = rig.nodes[i]
            .parent
            .map(|p| world[p])
            .unwrap_or(Xform::identity())
            .then(local);
    }
    Ok(world)
}

fn measure(
    solver: &Solver,
    poses: &[PoseSample],
    time: f32,
    mm_scale: f32,
) -> Result<Metrics, ActionToolError> {
    let expected = solver.reference(time)?;
    let actual = reconstruct(solver.rig, poses, time)?;
    let mut m = Metrics::default();
    for (bone, &i) in &solver.rig.mapping {
        if !solver.mapping.contains_key(bone) {
            continue;
        }
        let p = length3(sub3(expected[i].p, actual[i].p)) * mm_scale;
        let r = rotation_distance(expected[i].q, actual[i].q);
        if p > m.max_position_mm || r > m.max_rotation_deg {
            m.worst_bone = bone.clone();
            m.worst_time_sec = time;
        }
        m.max_position_mm = m.max_position_mm.max(p);
        m.max_rotation_deg = m.max_rotation_deg.max(r);
    }
    Ok(m)
}

fn merge(a: &mut Metrics, b: Metrics) {
    if b.max_position_mm > a.max_position_mm || b.max_rotation_deg > a.max_rotation_deg {
        a.worst_bone = b.worst_bone;
        a.worst_time_sec = b.worst_time_sec;
    }
    a.max_position_mm = a.max_position_mm.max(b.max_position_mm);
    a.max_rotation_deg = a.max_rotation_deg.max(b.max_rotation_deg);
}

// Use the public runtime evaluator, not the converter's inverse reconstruction.
fn runtime_script(target: &TargetOptions, dsl: &str) -> Result<String, ActionToolError> {
    let profile = profile_document(
        &String::from_utf8(read(&target.profile)?).map_err(|e| err(e.to_string()))?,
    )?;
    let declarations = profile
        .split_once('>')
        .ok_or_else(|| err("invalid profile wrapper"))?
        .1
        .split("<Background")
        .next()
        .unwrap_or_default();
    // The legacy World diagnostic wrapper requires model on profiles; Scene DSL
    // profiles instead bind the model on Model3D. This wrapper is never exported.
    let mut declarations = declarations.to_string();
    let mut cursor = 0;
    while let Some(start) = declarations[cursor..].find("<ModelProfile") {
        let start = cursor + start;
        let end = start
            + declarations[start..]
                .find('>')
                .ok_or_else(|| err("invalid profile tag"))?;
        if !declarations[start..end].contains(" model=") {
            declarations.insert_str(end, " model=\"target.glb\"");
            cursor = end + 20;
        } else {
            cursor = end + 1;
        }
    }
    let action_id = dsl
        .split("id=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .ok_or_else(|| err("missing action id"))?;
    let text = format!(
        "<Graph fps={{30}} duration=\"3600s\" size={{[64,64]}}>\n{declarations}\n{dsl}\n<World id=\"diagnostic\">\n<Actor id=\"actor\" model=\"target.glb\" profile=\"{}\" />\n</World>\n<ApplyAction action=\"{action_id}\" target=\"actor\" loop=\"false\" rootMotion=\"clip\" />\n<Present from=\"diagnostic\" />\n</Graph>",
        target.profile_id
    );
    Ok(text)
}

fn runtime_graph(
    target: &TargetOptions,
    dsl: &str,
) -> Result<motionloom::WorldGraph, ActionToolError> {
    motionloom::parse_world_graph_script(&runtime_script(target, dsl)?)
        .map_err(|e| err(format!("diagnostic world: {e}")))
}

/// Build reproducible native snapshots for the separate WASM parity runner.
/// This artifact contains no FBX data or JS rendering fallback.
pub fn validation_bundle(
    target: &TargetOptions,
    converted: &ConvertedAction,
) -> Result<String, ActionToolError> {
    let world_dsl = runtime_script(target, &converted.dsl)?;
    let graph = runtime_graph(target, &converted.dsl)?;
    let mesh = motionloom::experimental::load_glb_mesh_data(&target.model)?;
    let count = (converted.duration_sec * 120.).ceil() as usize;
    if count > 100_000 {
        return Err(err("validation sample budget exceeded"));
    }
    let mut snapshots = Vec::new();
    for f in 0..=count {
        let t = (f as f32 / 120.).min(converted.duration_sec);
        snapshots.push(
            motionloom::experimental::diagnose_world_actor_pose(
                &graph,
                &mesh,
                "actor",
                motionloom::WorldTime {
                    frame: (t * 1_000_000.).round() as u32,
                    fps: 1_000_000.,
                    duration_ms: graph.duration_ms,
                },
            )
            .map_err(|e| err(e.to_string()))?,
        );
    }
    serde_json::to_string(&serde_json::json!({
        "action_sha256":fingerprint(converted.dsl.as_bytes()),"target_sha256":fingerprint(&read(&target.model)?),
        "world_sha256":fingerprint(world_dsl.as_bytes()),"world_dsl":world_dsl,
        "stage":"model_global_before_scene_contacts","snapshots":snapshots,
    })).map_err(|e|err(e.to_string()))
}

fn matrix_error(expected: Xform, m: [f32; 16], mm_scale: f32) -> (f32, f32) {
    let p = length3(sub3(expected.p, [m[12], m[13], m[14]])) * mm_scale;
    // Relative rotation trace in f64 avoids the near-zero f32 acos noise floor.
    let mut trace = 0_f64;
    for j in 0..3 {
        let mut basis = [0.; 3];
        basis[j] = 1.;
        let e = rotate_vec3(expected.q, basis);
        let a = [m[j * 4], m[j * 4 + 1], m[j * 4 + 2]];
        let len = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let elen = e.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        trace += e
            .iter()
            .zip(a)
            .map(|(e, a)| *e as f64 * a as f64 / (len * elen))
            .sum::<f64>();
    }
    (
        p,
        ((trace - 1.) * 0.5).clamp(-1., 1.).acos().to_degrees() as f32,
    )
}

fn audit_joint_trajectories(
    solver: &Solver,
    duration_sec: f32,
    metres_per_model_unit: f32,
) -> Result<JointTrajectoryAudit, ActionToolError> {
    const SAMPLE_HZ: u32 = 120;
    const LOW_BAND_M: f32 = 0.02;
    const SLOW_SPEED_MPS: f32 = 0.2;
    let sample_count = (duration_sec * SAMPLE_HZ as f32).ceil() as usize + 1;
    if sample_count > 100_000 {
        return Err(err("joint trajectory audit sample budget exceeded"));
    }
    let mut effectors = Vec::new();
    for bone in [
        "foot_l", "toe_l", "foot_r", "toe_r", "hand_l", "hand_r", "head", "hips",
    ] {
        let Some(&index) = solver.rig.mapping.get(bone) else {
            continue;
        };
        if !solver.mapping.contains_key(bone) {
            continue;
        }
        let mut points = Vec::with_capacity(sample_count);
        for frame in 0..sample_count {
            let time = (frame as f32 / SAMPLE_HZ as f32).min(duration_sec);
            let point = solver.reference(time)?[index]
                .p
                .map(|value| value * metres_per_model_unit);
            points.push(point);
        }
        let min_y_m = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let max_y_m = points
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut speeds = vec![0.; points.len()];
        for i in 1..points.len() {
            let dt = if i + 1 == points.len() {
                (duration_sec - (i - 1) as f32 / SAMPLE_HZ as f32).max(1e-6)
            } else {
                1. / SAMPLE_HZ as f32
            };
            speeds[i] = length3(sub3(points[i], points[i - 1])) / dt;
        }
        if speeds.len() > 1 {
            speeds[0] = speeds[1];
        }
        let max_speed_mps = speeds.iter().copied().fold(0., f32::max);
        let low_and_slow_samples = points
            .iter()
            .zip(speeds)
            .filter(|(point, speed)| point[1] <= min_y_m + LOW_BAND_M && *speed <= SLOW_SPEED_MPS)
            .count();
        effectors.push(EffectorTrajectory {
            bone: bone.into(),
            samples: points.len(),
            min_y_m,
            max_y_m,
            max_speed_mps,
            low_and_slow_samples,
        });
    }
    Ok(JointTrajectoryAudit {
        sample_hz: SAMPLE_HZ,
        low_band_m: LOW_BAND_M,
        slow_speed_mps: SLOW_SPEED_MPS,
        effectors,
        interpretation: "Target model-global joint centres before scene contacts. Low/slow counts are review candidates relative to each joint's own minimum, not proof of floor contact, skin clearance, collision, or foot lock.".into(),
    })
}

fn measure_native(
    solver: &Solver,
    graph: &motionloom::WorldGraph,
    time: f32,
    mm_scale: f32,
) -> Result<Metrics, ActionToolError> {
    // Microsecond ticks preserve subframes without changing the runtime time contract.
    let tick = (time * 1_000_000.).round() as u32;
    let runtime_time = motionloom::WorldTime {
        frame: tick,
        fps: 1_000_000.,
        duration_ms: graph.duration_ms,
    };
    let snapshot = motionloom::experimental::diagnose_world_actor_pose(
        graph,
        &solver.rig.mesh,
        "actor",
        runtime_time,
    )
    .map_err(|e| err(e.to_string()))?;
    let expected = solver.reference(runtime_time.time_sec())?;
    let mut metrics = Metrics::default();
    for (bone, &i) in &solver.rig.mapping {
        if !solver.mapping.contains_key(bone) {
            continue;
        }
        let (p, r) = matrix_error(
            expected[i],
            snapshot.joints[i].model_global_matrix,
            mm_scale,
        );
        merge(
            &mut metrics,
            Metrics {
                max_position_mm: p,
                max_rotation_deg: r,
                worst_bone: bone.clone(),
                worst_time_sec: runtime_time.time_sec(),
            },
        );
    }
    Ok(metrics)
}

pub(crate) fn convert(
    source: &AnimationSource,
    clip: &AnimationClip,
    options: &ConvertOptions,
    target: &TargetOptions,
) -> Result<ConvertedAction, ActionToolError> {
    if [
        target.actor_height,
        target.max_position_mm,
        target.max_rotation_deg,
    ]
    .iter()
    .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err(err(
            "height and fidelity tolerances must be finite positive numbers",
        ));
    }
    if options.detect_contacts {
        return Err(err(
            "target faithful mode does not automatically add contacts; explicit contact adaptation is a separate operation",
        ));
    }
    if options.key_reduction_tolerance > 0. {
        return Err(err(
            "target mode currently retains fidelity samples; legacy channel reduction is not a spatial error bound",
        ));
    }
    if clip.tracks.iter().any(|t| {
        let constant = match &t.values {
            TrackValues::Vec3(v) => v.windows(2).all(|p| length3(sub3(p[0], p[1])) < 1e-6),
            TrackValues::Quat(v) => v.windows(2).all(|p| rotation_distance(p[0], p[1]) < 1e-5),
        };
        let scale_matches_rest = match &t.values {
            TrackValues::Vec3(v) => v
                .iter()
                .all(|v| length3(sub3(*v, source.nodes[t.node_index].rest_scale)) < 1e-5),
            _ => false,
        };
        (t.property == TrackProperty::Scale && !scale_matches_rest)
            || (t.interpolation == TrackInterpolation::Step && !constant)
    }) {
        return Err(err(
            "target mode requires constant scale and continuous baked curves; scale/step tracks cannot be silently approximated",
        ));
    }
    if source.diagnostics.iter().any(|d| d.contains("CUBICSPLINE")) {
        return Err(err(
            "unavailable source tangents prevent target fidelity conversion; bake the clip first",
        ));
    }
    let rig = Rig::load(target)?;
    if !rig.bounds_height.is_finite() || rig.bounds_height < 1e-5 {
        return Err(err("target mesh has no measurable height"));
    }
    let mapping = canonical_node_mappings(source);
    // Target limb lengths are retained; never silently drop authored limb translation.
    for track in &clip.tracks {
        if track.property == TrackProperty::Translation
            && mapping
                .iter()
                .any(|(bone, index)| bone != "hips" && *index == track.node_index)
            && let TrackValues::Vec3(values) = &track.values
            && values.iter().any(|value| {
                length3(sub3(
                    *value,
                    source.nodes[track.node_index].rest_translation,
                )) > 1e-5
            })
        {
            return Err(err(
                "non-hips bone translation requires a separate stretch/contact adaptation policy",
            ));
        }
    }
    for required in [
        "hips",
        "upper_leg_l",
        "lower_leg_l",
        "foot_l",
        "upper_leg_r",
        "lower_leg_r",
        "foot_r",
    ] {
        if !mapping.contains_key(required) || !rig.mapping.contains_key(required) {
            return Err(err(format!(
                "required source/target bone missing: {required}"
            )));
        }
    }
    for bone in mapping.keys() {
        if !rig.mapping.contains_key(bone) {
            return Err(err(format!("mapped source bone has no target: {bone}")));
        }
    }
    let source_order = order(&source.nodes);
    let rest_clip = AnimationClip {
        name: "rest".into(),
        duration_sec: 1.,
        tracks: vec![],
    };
    let source_rest = source_world(source, &rest_clip, 0., &source_order)?;
    let aligned_rest = align_rest(&rig, &mapping, &source_rest)?;
    let hip_height = |m: &BTreeMap<String, usize>, w: &[Xform]| {
        w[m["hips"]].p[1] - (w[m["foot_l"]].p[1] + w[m["foot_r"]].p[1]) * 0.5
    };
    let motion_scale = if target.proportional {
        let source_height = hip_height(&mapping, &source_rest);
        let target_height = hip_height(&rig.mapping, &rig.rest);
        if source_height <= 1e-4 || target_height <= 1e-4 {
            return Err(err(
                "invalid rest hip-to-foot height for proportional motion",
            ));
        }
        target_height / source_height
    } else {
        rig.bounds_height / target.actor_height
    };
    let solver = Solver {
        source,
        clip,
        rig: &rig,
        mapping,
        source_order,
        source_rest,
        aligned_rest,
        motion_scale,
    };
    let frame_count = sample_frame_count(clip.duration_sec, options.fps);
    if frame_count > 100_000 {
        return Err(err("target sample limit exceeded (100,000 base frames)"));
    }
    let output_duration = if target.editor_safe {
        (clip.duration_sec * 1000.).round() / 1000.
    } else {
        clip.duration_sec
    };
    let quantize = |t: f32| {
        if target.editor_safe {
            (t * 1000.).round() / 1000.
        } else {
            t
        }
    };
    let mut times = (0..=frame_count)
        .map(|f| (f as f32 / options.fps).min(clip.duration_sec))
        .map(quantize)
        .collect::<Vec<_>>();
    times.sort_by(f32::total_cmp);
    times.dedup();
    let mm_scale = 1000. * target.actor_height / rig.bounds_height;
    let mut poses = vec![];
    // Refine source-time intervals only where actual pose reconstruction needs it.
    for iteration in 0..=9 {
        let mut previous = BTreeMap::new();
        poses = times
            .iter()
            .map(|t| solver.encode(*t, &mut previous))
            .collect::<Result<Vec<_>, _>>()?;
        if iteration == 9 {
            break;
        }
        let mut insert = vec![];
        for pair in times.windows(2) {
            for alpha in [0.25, 0.5, 0.75] {
                let t = pair[0] + (pair[1] - pair[0]) * alpha;
                let m = measure(&solver, &poses, t, mm_scale)?;
                if m.max_position_mm > target.max_position_mm
                    || m.max_rotation_deg > target.max_rotation_deg
                {
                    let candidate = quantize(t);
                    if candidate > pair[0] + 1e-7 && candidate < pair[1] - 1e-7 {
                        insert.push(candidate);
                    }
                }
            }
        }
        if insert.is_empty() {
            break;
        }
        if times.len() + insert.len() > 100_000 {
            return Err(err("adaptive pose budget exceeded"));
        }
        times.extend(insert);
        times.sort_by(f32::total_cmp);
        times.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
    }
    let dsl = write_action(options, output_duration, &poses, &[]);
    validate_generated_action(&dsl, options.fps, output_duration)?;
    // Reparse decimal output so rounding/serialization are included in the metric.
    let wrapped = format!(
        "<Graph fps={{30}} duration=\"10s\" size={{[64,64]}}>\n{dsl}\n<Tex id=\"out\" fmt=\"rgba8unorm\" size={{[64,64]}} />\n<Present from=\"out\" />\n</Graph>"
    );
    let parsed = motionloom::parse_graph_script(&wrapped).map_err(|e| err(e.to_string()))?;
    poses = parsed.actions[0]
        .poses
        .iter()
        .map(|pose| {
            let mut bones = BTreeMap::new();
            for bone in &pose.bones {
                let mut channels = BTreeMap::new();
                for (key, value) in [
                    ("x", &bone.x),
                    ("y", &bone.y),
                    ("z", &bone.z),
                    ("rotationX", &bone.rotation_x),
                    ("rotationY", &bone.rotation_y),
                    ("rotationZ", &bone.rotation_z),
                    ("bend", &bone.bend),
                    ("forward", &bone.forward),
                    ("side", &bone.side),
                    ("turn", &bone.turn),
                    ("twist", &bone.twist),
                ] {
                    if let Some(value) = value {
                        channels.insert(
                            key,
                            value
                                .parse::<f32>()
                                .map_err(|_| err("serialized channel is not numeric"))?,
                        );
                    }
                }
                bones.insert(bone.id.clone(), channels);
            }
            Ok(PoseSample {
                time: pose.t,
                bones,
            })
        })
        .collect::<Result<Vec<_>, ActionToolError>>()?;
    let mut metrics = Metrics::default();
    let mut evaluated_samples = 0;
    for pair in times.windows(2) {
        for alpha in [0., 0.25, 0.5, 0.75, 1.] {
            let t = pair[0] + (pair[1] - pair[0]) * alpha;
            merge(&mut metrics, measure(&solver, &poses, t, mm_scale)?);
            evaluated_samples += 1;
        }
    }
    let editor_millisecond_collisions = poses.len()
        - poses
            .iter()
            .map(|p| (p.time * 1000.).round() as u64)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
    let tool_reconstruction_pass = metrics.max_position_mm <= target.max_position_mm
        && metrics.max_rotation_deg <= target.max_rotation_deg;
    let runtime = runtime_graph(target, &dsl)?;
    let mut native = Metrics::default();
    let mut native_evaluated_samples = 0;
    for pair in times.windows(2) {
        for alpha in [0., 0.25, 0.5, 0.75, 1.] {
            merge(
                &mut native,
                measure_native(
                    &solver,
                    &runtime,
                    pair[0] + (pair[1] - pair[0]) * alpha,
                    mm_scale,
                )?,
            );
            native_evaluated_samples += 1;
        }
    }
    let native_pass = native.max_position_mm <= target.max_position_mm
        && native.max_rotation_deg <= target.max_rotation_deg;
    let joint_trajectory_audit = Some(audit_joint_trajectories(
        &solver,
        output_duration,
        target.actor_height / rig.bounds_height,
    )?);
    let mut limitations=vec![
        "Target-bound Action: use the exact recorded ModelProfile and actor height; raw residual channels are not rig-independent.".into(),
        "No independent source evaluator reference supplied; native decoding fidelity is not certified.".into(),
        "Native model-global pose comparison excludes scene collision/contact and skin deformation; WASM parity requires a separate run.".into(),
        "Faithful mode does not correct source/target limb-length contact differences, add foot locks, or alter roll contact timing.".into(),
        "Euler interpolation is checked at sampled subframes, not proven for all continuous times.".into(),
        "Rest direction alignment preserves target bind twist; terminal/forked joints still need reference-pose calibration for exact roll.".into(),
    ];
    if editor_millisecond_collisions > 0 {
        limitations.push(format!("{editor_millisecond_collisions} subframe poses share editor milliseconds; this candidate must not be promoted as an editable default."));
    }
    let source_comparison = reference::compare(source, clip, &solver.mapping, target)?;
    if let Some(reference) = &source_comparison {
        limitations.retain(|s| !s.starts_with("No independent"));
        limitations.push(format!(
            "Independent source comparison ({}): {:.4} mm / {:.4} degrees; passed={}",
            reference.evaluator,
            reference.metrics.max_position_mm,
            reference.metrics.max_rotation_deg,
            reference.passed
        ));
    }
    let report = FidelityReport {
        source_sha256: fingerprint(&read(&source.path)?),
        target_sha256: rig.model_hash.clone(),
        profile_sha256: rig.profile_hash.clone(),
        action_sha256: fingerprint(dsl.as_bytes()),
        profile_id: target.profile_id.clone(),
        // Reports are distributable build evidence. Keep their public identity
        // tied to the authored Action instead of leaking source-file metadata.
        clip: options.action_id.clone(),
        duration_sec: output_duration,
        trajectory_scale: motion_scale,
        actor_height: target.actor_height,
        mapped_bones: solver.mapping.len(),
        output_poses: poses.len(),
        evaluated_samples,
        editor_millisecond_collisions,
        time_grid: if target.editor_safe {
            "milliseconds"
        } else {
            "subframes"
        }
        .into(),
        tool_reconstruction_pass,
        tool_reconstruction: metrics,
        max_position_mm: target.max_position_mm,
        max_rotation_deg: target.max_rotation_deg,
        source_reference: source_comparison
            .as_ref()
            .map(|r| if r.passed { "passed" } else { "failed" })
            .unwrap_or("unverified")
            .into(),
        source_comparison,
        native_runtime: if native_pass { "passed" } else { "failed" }.into(),
        native_reconstruction: native,
        native_evaluated_samples,
        joint_trajectory_audit,
        wasm_runtime: "unverified".into(),
        strict_pass: false,
        limitations,
    };
    let mut diagnostics = source.diagnostics.clone();
    diagnostics.extend(report.limitations.clone());
    diagnostics.push(format!("Target reconstruction: {:.4} mm / {:.4} degrees; {} poses. This is NOT independent/runtime fidelity acceptance.",report.tool_reconstruction.max_position_mm,report.tool_reconstruction.max_rotation_deg,poses.len()));
    Ok(ConvertedAction {
        dsl,
        clip_name: options.action_id.clone(),
        duration_sec: clip.duration_sec,
        pose_count: poses.len(),
        sampled_pose_count: frame_count + 1,
        mapped_bones: solver
            .mapping
            .iter()
            .map(|(b, i)| BoneMapping {
                source: source.nodes[*i].name.clone(),
                canonical: b.clone(),
            })
            .collect(),
        diagnostics,
        fidelity: Some(report),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_pelvis_translation_reconstructs_world_displacement() {
        let base = Xform {
            p: [0., 1., 0.],
            q: euler_quat([14.5, 0., 0.]),
            s: 0.92,
        };
        let desired = add3(base.p, [0., -0.225, -4.477]);
        let local = base.inverse_point(desired);
        assert!(length3(sub3(base.point(local), desired)) < 1e-5);
        assert!((base.point([0., -0.225, -4.477])[1] - desired[1]).abs() > 0.8);
    }
    #[test]
    fn runtime_rest_euler_correction_is_subtracted_once() {
        let desired = [24., -12., 15.];
        let rest = [-4.8, 0., -89.8];
        let encoded = sub3(desired, rest);
        assert!(rotation_distance(euler_quat(add3(encoded, rest)), euler_quat(desired)) < 0.001);
    }
    #[test]
    fn euler_round_trip_and_continuous_roll() {
        let mut prev = [0.; 3];
        for frame in 0..=720 {
            let r = [frame as f32, 27., -13.];
            let q = euler_quat(r);
            let out = continuous_euler(quat_to_euler_xyz_deg(q), prev);
            assert!(rotation_distance(q, euler_quat(out)) < 0.001);
            assert!(length3(sub3(out, prev)) < 32.);
            prev = out;
        }
        assert!(prev[0] > 700.);
    }

    #[test]
    fn a_pose_direction_aligns_to_t_pose_without_changing_length() {
        let a = [0.70710677, -0.70710677, 0.];
        let t = [1., 0., 0.];
        let q = align_direction(a, t).unwrap();
        assert!(length3(sub3(rotate_vec3(q, a), t)) < 1e-5);
        assert!(align_direction([1., 0., 0.], [-1., 0., 0.]).is_err());
    }
}
