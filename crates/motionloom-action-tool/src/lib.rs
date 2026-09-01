// =========================================
// =========================================
// crates/motionloom-action-tool/src/lib.rs

//! Offline conversion of authored animation clips into standalone MotionLoom Actions.
//!
//! Importers live only in this authoring crate. The MotionLoom renderer and its
//! WASM build do not depend on FBX, Blender, or this tool.

mod blender_source;
mod fbx_source;
mod gltf_source;
pub mod source;
pub mod target;
mod ufbx_source;

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use source::{
    AnimationClip, AnimationNode, AnimationSource, AnimationTrack, TrackInterpolation,
    TrackProperty, TrackValues, validate_source,
};
use thiserror::Error;

const SUPPORTED_SOURCE_PROFILE: &str = "fbx_humanoid";

#[derive(Debug, Error)]
pub enum ActionToolError {
    #[error("target conversion: {message}")]
    Target { message: String },
    #[error(transparent)]
    Glb(#[from] motionloom::GlbLoadError),
    #[error("failed to read FBX {path}: {source}")]
    FbxRead {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("unsupported animation input '{path}'; expected .fbx, .glb, or .gltf")]
    UnsupportedInput { path: PathBuf },
    #[error("animated asset contains no clips: {path}")]
    NoAnimations { path: PathBuf },
    #[error("animation clip not found: {clip}")]
    ClipNotFound { clip: String },
    #[error("unsupported source profile '{profile}'; expected fbx_humanoid")]
    UnsupportedSourceProfile { profile: String },
    #[error("sample fps must be finite and greater than zero")]
    InvalidFps,
    #[error("key reduction tolerance must be finite and non-negative")]
    InvalidKeyReductionTolerance,
    #[error("animation clip has no canonical humanoid bone mappings")]
    NoCanonicalBones,
    #[error("generated MotionLoom Action failed round-trip validation: {message}")]
    InvalidGeneratedDsl { message: String },
    #[error("Blender fallback requested but no Blender executable was found; set BLENDER")]
    BlenderNotFound,
    #[error("failed to launch Blender at {executable}: {source}")]
    BlenderLaunch {
        executable: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Blender FBX conversion failed (status {status:?}): {stderr}")]
    BlenderConversion { status: Option<i32>, stderr: String },
    #[error("failed to create temporary conversion directory: {0}")]
    TemporaryDirectory(#[source] io::Error),
    #[error("unknown FBX backend '{value}'; use auto, native, or blender")]
    InvalidBackend { value: String },
    #[error("invalid animation data: {message}")]
    InvalidAnimation { message: String },
    #[error(
        "FBX requires full evaluation: {reason}; explicitly use --fbx-backend blender if a working Blender installation is available"
    )]
    FbxNeedsEvaluation { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbxBackend {
    Auto,
    Native,
    Blender,
}

impl FromStr for FbxBackend {
    type Err = ActionToolError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "native" => Ok(Self::Native),
            "blender" => Ok(Self::Blender),
            _ => Err(ActionToolError::InvalidBackend {
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InspectReport {
    pub path: PathBuf,
    pub backend: String,
    pub clips: Vec<ClipSummary>,
    pub mapped_bones: Vec<BoneMapping>,
    pub unmapped_joints: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClipSummary {
    pub name: String,
    pub duration_sec: f32,
    pub channel_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoneMapping {
    pub source: String,
    pub canonical: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvertOptions {
    pub clip: Option<String>,
    pub source_profile: String,
    pub action_id: String,
    pub fps: f32,
    pub fbx_backend: FbxBackend,
    pub key_reduction_tolerance: f32,
    pub detect_contacts: bool,
    pub target: Option<target::TargetOptions>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            clip: None,
            source_profile: SUPPORTED_SOURCE_PROFILE.to_string(),
            action_id: "imported_action".to_string(),
            fps: 30.0,
            fbx_backend: FbxBackend::Auto,
            key_reduction_tolerance: 0.0,
            detect_contacts: false,
            target: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvertedAction {
    pub dsl: String,
    pub clip_name: String,
    pub duration_sec: f32,
    pub pose_count: usize,
    pub sampled_pose_count: usize,
    pub mapped_bones: Vec<BoneMapping>,
    pub diagnostics: Vec<String>,
    pub fidelity: Option<target::FidelityReport>,
}

/// Inspect FBX, GLB, or glTF animation metadata without writing output.
pub fn inspect_animation_file(
    path: impl AsRef<Path>,
    backend: FbxBackend,
) -> Result<InspectReport, ActionToolError> {
    let source = load_animation_source(path.as_ref(), backend)?;
    inspect_source(&source)
}

/// Convert FBX, GLB, or glTF animation into a standalone canonical Action.
pub fn convert_animation_file(
    path: impl AsRef<Path>,
    options: &ConvertOptions,
) -> Result<ConvertedAction, ActionToolError> {
    let source = if options.target.is_some()
        && options.fbx_backend == FbxBackend::Auto
        && path
            .as_ref()
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("fbx"))
    {
        ufbx_source::load(path.as_ref())?
    } else {
        load_animation_source(path.as_ref(), options.fbx_backend)?
    };
    convert_source(&source, options)
}

/// Compatibility entry point retained for existing glTF callers.
pub fn inspect_animated_gltf(path: impl AsRef<Path>) -> Result<InspectReport, ActionToolError> {
    inspect_animation_file(path, FbxBackend::Auto)
}

/// Compatibility entry point retained for existing glTF callers.
pub fn convert_animated_gltf(
    path: impl AsRef<Path>,
    options: &ConvertOptions,
) -> Result<ConvertedAction, ActionToolError> {
    convert_animation_file(path, options)
}

fn load_animation_source(
    path: &Path,
    backend: FbxBackend,
) -> Result<AnimationSource, ActionToolError> {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("glb" | "gltf") => gltf_source::load(path),
        Some("fbx") => match backend {
            // Auto selects the in-process reader only. Never launch an external
            // application after a parse failure without an explicit backend choice.
            FbxBackend::Auto | FbxBackend::Native => fbx_source::load(path),
            FbxBackend::Blender => blender_source::load(path),
        },
        _ => Err(ActionToolError::UnsupportedInput {
            path: path.to_path_buf(),
        }),
    }
}

fn inspect_source(source: &AnimationSource) -> Result<InspectReport, ActionToolError> {
    validate_source(source)?;
    if source.clips.is_empty() {
        return Err(ActionToolError::NoAnimations {
            path: source.path.clone(),
        });
    }
    let (mapped_bones, unmapped_joints) = inspect_bone_mappings(source);
    Ok(InspectReport {
        path: source.path.clone(),
        backend: source.backend.clone(),
        clips: source
            .clips
            .iter()
            .map(|clip| ClipSummary {
                name: clip.name.clone(),
                duration_sec: clip.duration_sec,
                channel_count: clip.tracks.len(),
            })
            .collect(),
        mapped_bones,
        unmapped_joints,
        diagnostics: source.diagnostics.clone(),
    })
}

/// Retain the in-memory GLB entry points used by existing authoring callers.
pub fn inspect_mesh(
    path: &Path,
    mesh: &motionloom::GlbMeshData,
) -> Result<InspectReport, ActionToolError> {
    inspect_source(&gltf_source::from_mesh(path, mesh))
}

pub fn convert_mesh(
    mesh: &motionloom::GlbMeshData,
    options: &ConvertOptions,
) -> Result<ConvertedAction, ActionToolError> {
    convert_source(&gltf_source::from_mesh(&mesh.path, mesh), options)
}

fn convert_source(
    source: &AnimationSource,
    options: &ConvertOptions,
) -> Result<ConvertedAction, ActionToolError> {
    validate_options(options)?;
    validate_source(source)?;
    let clip = select_clip(source, options.clip.as_deref())?;
    if let Some(target) = &options.target {
        return target::convert(source, clip, options, target);
    }
    let mappings = canonical_node_mappings(source);
    if mappings.is_empty() {
        return Err(ActionToolError::NoCanonicalBones);
    }
    let mapped_bones = mappings
        .iter()
        .map(|(canonical, index)| BoneMapping {
            source: source.nodes[*index].name.clone(),
            canonical: canonical.clone(),
        })
        .collect::<Vec<_>>();
    let frame_count = sample_frame_count(clip.duration_sec, options.fps);
    if frame_count > 1_000_000 {
        return Err(ActionToolError::InvalidAnimation {
            message:
                "requested sample count exceeds 1,000,000; reduce --fps or split the source clip"
                    .into(),
        });
    }
    let mut previous_angles = BTreeMap::new();
    let mut poses = (0..=frame_count)
        .map(|frame| {
            let time = (frame as f32 / options.fps).min(clip.duration_sec);
            sample_pose(source, clip, &mappings, time, &mut previous_angles)
        })
        .collect::<Vec<_>>();
    unwrap_pose_angles(&mut poses);
    let sampled_pose_count = poses.len();
    if options.key_reduction_tolerance > 0.0 {
        poses = reduce_poses(poses, options.key_reduction_tolerance);
    }
    let contacts = if options.detect_contacts {
        detect_foot_contacts(source, clip, &mappings, options.fps)
    } else {
        Vec::new()
    };
    let dsl = write_action(options, clip.duration_sec, &poses, &contacts);
    validate_generated_action(&dsl, options.fps, clip.duration_sec)?;
    let (_, unmapped_joints) = inspect_bone_mappings(source);
    let mut diagnostics = source.diagnostics.clone();
    // Successful decoding does not establish pose equivalence on another rig.
    diagnostics.push("Source-profile conversion only: no target rig was supplied. Validate bind-pose axes, limb lengths, pelvis translation and ground contact on the intended character before publishing.".into());
    if clip.tracks.iter().any(|track| track.property == TrackProperty::Scale && matches!(&track.values, TrackValues::Vec3(values) if values.windows(2).any(|p| sub3(p[0],p[1]).iter().any(|v| v.abs()>1e-5)))) {
        diagnostics.push("Animated bone scale is not exported by the canonical channel profile; review or bake scaling before publishing this Action.".into());
    }
    diagnostics.push(format!(
        "sampled {sampled_pose_count} poses at {} fps; retained {} poses",
        format_number(options.fps),
        poses.len()
    ));
    if !unmapped_joints.is_empty() {
        diagnostics.push(format!(
            "{} non-canonical source nodes: intermediate transforms are folded into mapped descendants; terminal nodes have no Action channel",
            unmapped_joints.len()
        ));
    }
    if options.detect_contacts {
        diagnostics.push(format!(
            "detected {} foot contact intervals",
            contacts.len()
        ));
    }
    Ok(ConvertedAction {
        dsl,
        clip_name: options.action_id.clone(),
        duration_sec: clip.duration_sec,
        pose_count: poses.len(),
        sampled_pose_count,
        mapped_bones,
        diagnostics,
        fidelity: None,
    })
}

fn validate_options(options: &ConvertOptions) -> Result<(), ActionToolError> {
    if options.source_profile != SUPPORTED_SOURCE_PROFILE {
        return Err(ActionToolError::UnsupportedSourceProfile {
            profile: options.source_profile.clone(),
        });
    }
    if !options.fps.is_finite() || options.fps <= 0.0 {
        return Err(ActionToolError::InvalidFps);
    }
    if !options.key_reduction_tolerance.is_finite() || options.key_reduction_tolerance < 0.0 {
        return Err(ActionToolError::InvalidKeyReductionTolerance);
    }
    Ok(())
}

fn select_clip<'a>(
    source: &'a AnimationSource,
    requested: Option<&str>,
) -> Result<&'a AnimationClip, ActionToolError> {
    if let Some(requested) = requested {
        return source
            .clips
            .iter()
            .find(|clip| clip.name == requested)
            .ok_or_else(|| ActionToolError::ClipNotFound {
                clip: requested.to_string(),
            });
    }
    source
        .clips
        .first()
        .ok_or_else(|| ActionToolError::NoAnimations {
            path: source.path.clone(),
        })
}

#[derive(Debug, Clone)]
struct PoseSample {
    time: f32,
    bones: BTreeMap<String, BTreeMap<&'static str, f32>>,
}

fn sample_pose(
    source: &AnimationSource,
    clip: &AnimationClip,
    mappings: &BTreeMap<String, usize>,
    time: f32,
    previous_angles: &mut BTreeMap<String, [f32; 3]>,
) -> PoseSample {
    let mut bones = BTreeMap::new();
    for (canonical, index) in mappings {
        let node = &source.nodes[*index];
        let mut sampled = sample_node(clip, *index, time, node);
        let mut rest_rotation = node.rest_rotation;
        let mut rest_translation = node.rest_translation;
        // Collapse extra source spine/root nodes instead of discarding their motion.
        let mut parent = node.parent;
        while let Some(parent_index) = parent {
            if mappings.values().any(|mapped| *mapped == parent_index) {
                break;
            }
            let rest = &source.nodes[parent_index];
            let animated = sample_node(clip, parent_index, time, rest);
            sampled.translation = add3(
                animated.translation,
                rotate_vec3(
                    animated.rotation,
                    mul_vec3(animated.scale, sampled.translation),
                ),
            );
            sampled.rotation = quat_mul(animated.rotation, sampled.rotation);
            rest_translation = add3(
                rest.rest_translation,
                rotate_vec3(
                    rest.rest_rotation,
                    mul_vec3(rest.rest_scale, rest_translation),
                ),
            );
            rest_rotation = quat_mul(rest.rest_rotation, rest_rotation);
            parent = rest.parent;
        }
        let translation = sub3(sampled.translation, rest_translation);
        let translation = if canonical == "hips" {
            world_vector(source, translation)
        } else {
            translation
        };
        let rotation = quat_mul(quat_conjugate(rest_rotation), sampled.rotation);
        let mut euler = quat_to_euler_xyz_deg(rotation);
        if let Some(previous) = previous_angles.get(canonical) {
            euler = continuous_euler(euler, *previous);
        }
        previous_angles.insert(canonical.clone(), euler);
        let channels = canonical_channels(canonical, translation, euler);
        if !channels.is_empty() {
            bones.insert(canonical.clone(), channels.into_iter().collect());
        }
    }
    PoseSample { time, bones }
}

#[derive(Debug, Clone, Copy)]
struct SampledNode {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

fn sample_node(
    clip: &AnimationClip,
    node_index: usize,
    time: f32,
    rest: &AnimationNode,
) -> SampledNode {
    let mut sampled = SampledNode {
        translation: rest.rest_translation,
        rotation: rest.rest_rotation,
        scale: rest.rest_scale,
    };
    for track in clip
        .tracks
        .iter()
        .filter(|track| track.node_index == node_index)
    {
        match (track.property, sample_track(track, time)) {
            (TrackProperty::Translation, Some(TrackValues::Vec3(value))) => {
                sampled.translation = value[0]
            }
            (TrackProperty::Rotation, Some(TrackValues::Quat(value))) => {
                sampled.rotation = value[0]
            }
            (TrackProperty::Scale, Some(TrackValues::Vec3(value))) => sampled.scale = value[0],
            _ => {}
        }
    }
    sampled
}

fn sample_track(track: &AnimationTrack, time: f32) -> Option<TrackValues> {
    if track.times.is_empty() {
        return None;
    }
    let last = track.times.len() - 1;
    let previous = track
        .times
        .partition_point(|key| *key <= time)
        .saturating_sub(1)
        .min(last);
    let next = (previous + 1).min(last);
    let span = (track.times[next] - track.times[previous]).max(f32::EPSILON);
    let alpha = if next == previous || track.interpolation == TrackInterpolation::Step {
        0.0
    } else {
        ((time - track.times[previous]) / span).clamp(0.0, 1.0)
    };
    match &track.values {
        TrackValues::Vec3(values) => {
            let from = *values.get(previous)?;
            let to = values.get(next).copied().unwrap_or(from);
            Some(TrackValues::Vec3(vec![lerp3(from, to, alpha)]))
        }
        TrackValues::Quat(values) => {
            let from = *values.get(previous)?;
            let to = values.get(next).copied().unwrap_or(from);
            Some(TrackValues::Quat(vec![slerp_quat(from, to, alpha)]))
        }
    }
}

fn write_action(
    options: &ConvertOptions,
    duration: f32,
    poses: &[PoseSample],
    contacts: &[ContactInterval],
) -> String {
    let mut dsl = format!(
        "<Action id=\"{}\" skeleton=\"humanoid_v1\" duration=\"{}s\">\n",
        escape_xml_attr(&options.action_id),
        format_number(duration)
    );
    for pose in poses {
        dsl.push_str(&format!("  <Pose t=\"{}s\">\n", format_number(pose.time)));
        for (bone, channels) in &pose.bones {
            dsl.push_str(&format!("    <Bone id=\"{}\"", escape_xml_attr(bone)));
            for (channel, value) in channels {
                dsl.push_str(&format!(" {channel}=\"{}\"", format_number(*value)));
            }
            dsl.push_str(" />\n");
        }
        dsl.push_str("  </Pose>\n");
    }
    for (index, contact) in contacts.iter().enumerate() {
        dsl.push_str(&format!(
            "  <Contact id=\"{}_{}\" effector=\"{}\" target=\"ground\" from=\"{}\" to=\"{}\" mode=\"lock\" />\n",
            contact.effector,
            index + 1,
            contact.effector,
            format_number(contact.from / duration.max(f32::EPSILON)),
            format_number(contact.to / duration.max(f32::EPSILON))
        ));
    }
    dsl.push_str("</Action>\n");
    dsl
}

fn reduce_poses(poses: Vec<PoseSample>, tolerance: f32) -> Vec<PoseSample> {
    if poses.len() <= 2 {
        return poses;
    }
    // Split at the worst sample until every retained segment satisfies the bound.
    let mut retained = BTreeSet::from([0, poses.len() - 1]);
    let mut segments = vec![(0, poses.len() - 1)];
    while let Some((first, last)) = segments.pop() {
        let mut worst = (tolerance, None);
        for index in first + 1..last {
            let alpha = (poses[index].time - poses[first].time)
                / (poses[last].time - poses[first].time).max(f32::EPSILON);
            let error = pose_error(&poses[first], &poses[index], &poses[last], alpha);
            if error > worst.0 {
                worst = (error, Some(index));
            }
        }
        if let Some(index) = worst.1 {
            retained.insert(index);
            segments.extend([(first, index), (index, last)]);
        }
    }
    poses
        .into_iter()
        .enumerate()
        .filter_map(|(index, pose)| retained.contains(&index).then_some(pose))
        .collect()
}

fn pose_error(from: &PoseSample, value: &PoseSample, to: &PoseSample, alpha: f32) -> f32 {
    value
        .bones
        .iter()
        .flat_map(|(bone, channels)| {
            channels.iter().map(move |(channel, actual)| {
                let a = from
                    .bones
                    .get(bone)
                    .and_then(|map| map.get(channel))
                    .copied()
                    .unwrap_or(0.0);
                let b = to
                    .bones
                    .get(bone)
                    .and_then(|map| map.get(channel))
                    .copied()
                    .unwrap_or(0.0);
                // The same numeric tolerance is degrees for angles and millimetres for translation.
                let units = if matches!(*channel, "x" | "y" | "z") {
                    1000.0
                } else {
                    1.0
                };
                (actual - (a + (b - a) * alpha)).abs() * units
            })
        })
        .fold(0.0, f32::max)
}

fn unwrap_pose_angles(poses: &mut [PoseSample]) {
    let mut previous = BTreeMap::<(String, &'static str), f32>::new();
    for pose in poses {
        for (bone, channels) in &mut pose.bones {
            for (channel, value) in channels {
                if matches!(*channel, "x" | "y" | "z") && bone == "hips" {
                    continue;
                }
                if let Some(last) = previous.get(&(bone.clone(), *channel)) {
                    while *value - last > 180.0 {
                        *value -= 360.0;
                    }
                    while *value - last < -180.0 {
                        *value += 360.0;
                    }
                }
                previous.insert((bone.clone(), *channel), *value);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ContactInterval {
    effector: &'static str,
    from: f32,
    to: f32,
}

fn detect_foot_contacts(
    source: &AnimationSource,
    clip: &AnimationClip,
    mappings: &BTreeMap<String, usize>,
    fps: f32,
) -> Vec<ContactInterval> {
    let frame_count = sample_frame_count(clip.duration_sec, fps);
    let mut result = Vec::new();
    for (canonical, effector) in [("foot_l", "foot_l"), ("foot_r", "foot_r")] {
        let Some(&node_index) = mappings.get(canonical) else {
            continue;
        };
        let samples = (0..=frame_count)
            .map(|frame| {
                let time = (frame as f32 / fps).min(clip.duration_sec);
                (time, global_position(source, clip, node_index, time))
            })
            .collect::<Vec<_>>();
        let ground = samples
            .iter()
            .map(|(_, p)| p[1])
            .fold(f32::INFINITY, f32::min);
        let height_limit = ground + 0.035;
        let mut start = None;
        for index in 0..samples.len() {
            let speed = if index == 0 {
                0.0
            } else {
                length3(sub3(samples[index].1, samples[index - 1].1)) * fps
            };
            let planted = samples[index].1[1] <= height_limit && speed <= 0.35;
            match (start, planted) {
                (None, true) => start = Some(samples[index].0),
                (Some(from), false) => {
                    if samples[index - 1].0 - from >= 2.0 / fps {
                        result.push(ContactInterval {
                            effector,
                            from,
                            to: samples[index - 1].0,
                        });
                    }
                    start = None;
                }
                (_, _) => {}
            }
        }
        if let Some(from) = start {
            if clip.duration_sec - from >= 2.0 / fps {
                result.push(ContactInterval {
                    effector,
                    from,
                    to: clip.duration_sec,
                });
            }
        }
    }
    result
}

fn global_position(
    source: &AnimationSource,
    clip: &AnimationClip,
    index: usize,
    time: f32,
) -> [f32; 3] {
    world_vector(
        source,
        global_transform(source, clip, index, time).translation,
    )
}

fn world_vector(source: &AnimationSource, vector: [f32; 3]) -> [f32; 3] {
    source
        .world_basis
        .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
}

fn global_transform(
    source: &AnimationSource,
    clip: &AnimationClip,
    index: usize,
    time: f32,
) -> SampledNode {
    let local = sample_node(clip, index, time, &source.nodes[index]);
    let Some(parent) = source.nodes[index].parent else {
        return local;
    };
    let parent = global_transform(source, clip, parent, time);
    SampledNode {
        translation: add3(
            parent.translation,
            rotate_vec3(parent.rotation, mul_vec3(parent.scale, local.translation)),
        ),
        rotation: quat_mul(parent.rotation, local.rotation),
        scale: mul_vec3(parent.scale, local.scale),
    }
}

fn inspect_bone_mappings(source: &AnimationSource) -> (Vec<BoneMapping>, Vec<String>) {
    let selected = canonical_node_mappings(source);
    let selected_indices = selected.values().copied().collect::<BTreeSet<_>>();
    let mut mapped = selected
        .into_iter()
        .map(|(canonical, index)| BoneMapping {
            source: source.nodes[index].name.clone(),
            canonical,
        })
        .collect::<Vec<_>>();
    let mut unmapped = source
        .nodes
        .iter()
        .enumerate()
        .filter(|(index, _)| !selected_indices.contains(index))
        .map(|(_, node)| node.name.clone())
        .collect::<Vec<_>>();
    mapped.sort_by(|a, b| a.canonical.cmp(&b.canonical));
    unmapped.sort();
    (mapped, unmapped)
}

fn canonical_node_mappings(source: &AnimationSource) -> BTreeMap<String, usize> {
    let mut result = BTreeMap::<String, (usize, usize)>::new();
    for (index, node) in source.nodes.iter().enumerate() {
        let Some((canonical, priority)) = canonical_fbx_humanoid_bone(&node.name) else {
            continue;
        };
        let replace = result
            .get(canonical)
            .map(|(_, old)| priority > *old)
            .unwrap_or(true);
        if replace {
            result.insert(canonical.to_string(), (index, priority));
        }
    }
    result
        .into_iter()
        .map(|(name, (index, _))| (name, index))
        .collect()
}

fn canonical_fbx_humanoid_bone(raw: &str) -> Option<(&'static str, usize)> {
    let local_name = raw
        .rsplit("::")
        .next()
        .unwrap_or(raw)
        .rsplit(':')
        .next()
        .unwrap_or(raw);
    let name = local_name
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-', '.', '_'], "");
    let value = match name.as_str() {
        "hips" | "pelvis" => ("hips", 0),
        "spine" | "spine01" => ("spine", 1),
        "spine1" | "spine02" | "chest" => ("chest", 1),
        "spine2" | "spine03" | "upperchest" => ("upper_chest", 2),
        "neck" | "neck01" => ("neck", 0),
        "head" => ("head", 0),
        "leftshoulder" | "claviclel" => ("shoulder_l", 0),
        "leftarm" | "leftupperarm" | "upperarml" => ("upper_arm_l", 0),
        "leftforearm" | "leftlowerarm" | "forearml" | "lowerarml" => ("forearm_l", 0),
        "lefthand" | "leftwrist" | "handl" => ("hand_l", 0),
        "rightshoulder" | "clavicler" => ("shoulder_r", 0),
        "rightarm" | "rightupperarm" | "upperarmr" => ("upper_arm_r", 0),
        "rightforearm" | "rightlowerarm" | "forearmr" | "lowerarmr" => ("forearm_r", 0),
        "righthand" | "rightwrist" | "handr" => ("hand_r", 0),
        "leftupleg" | "leftthigh" | "leftupperleg" | "upperlegl" | "thighl" => ("upper_leg_l", 0),
        "leftleg" | "leftcalf" | "leftlowerleg" | "lowerlegl" | "calfl" => ("lower_leg_l", 0),
        "leftfoot" | "leftankle" | "footl" => ("foot_l", 0),
        "lefttoebase" | "lefttoe" | "toel" | "balll" => ("toe_l", 0),
        "rightupleg" | "rightthigh" | "rightupperleg" | "upperlegr" | "thighr" => {
            ("upper_leg_r", 0)
        }
        "rightleg" | "rightcalf" | "rightlowerleg" | "lowerlegr" | "calfr" => ("lower_leg_r", 0),
        "rightfoot" | "rightankle" | "footr" => ("foot_r", 0),
        "righttoebase" | "righttoe" | "toer" | "ballr" => ("toe_r", 0),
        _ => return canonical_finger(&name),
    };
    Some(value)
}

fn canonical_finger(name: &str) -> Option<(&'static str, usize)> {
    const FINGERS: [(&str, &str, &str); 30] = [
        ("lefthandthumb1", "thumb_1_l", "leftthumb1"),
        ("lefthandthumb2", "thumb_2_l", "leftthumb2"),
        ("lefthandthumb3", "thumb_3_l", "leftthumb3"),
        ("lefthandindex1", "index_1_l", "leftindex1"),
        ("lefthandindex2", "index_2_l", "leftindex2"),
        ("lefthandindex3", "index_3_l", "leftindex3"),
        ("lefthandmiddle1", "middle_1_l", "leftmiddle1"),
        ("lefthandmiddle2", "middle_2_l", "leftmiddle2"),
        ("lefthandmiddle3", "middle_3_l", "leftmiddle3"),
        ("lefthandring1", "ring_1_l", "leftring1"),
        ("lefthandring2", "ring_2_l", "leftring2"),
        ("lefthandring3", "ring_3_l", "leftring3"),
        ("lefthandpinky1", "pinky_1_l", "leftpinky1"),
        ("lefthandpinky2", "pinky_2_l", "leftpinky2"),
        ("lefthandpinky3", "pinky_3_l", "leftpinky3"),
        ("righthandthumb1", "thumb_1_r", "rightthumb1"),
        ("righthandthumb2", "thumb_2_r", "rightthumb2"),
        ("righthandthumb3", "thumb_3_r", "rightthumb3"),
        ("righthandindex1", "index_1_r", "rightindex1"),
        ("righthandindex2", "index_2_r", "rightindex2"),
        ("righthandindex3", "index_3_r", "rightindex3"),
        ("righthandmiddle1", "middle_1_r", "rightmiddle1"),
        ("righthandmiddle2", "middle_2_r", "rightmiddle2"),
        ("righthandmiddle3", "middle_3_r", "rightmiddle3"),
        ("righthandring1", "ring_1_r", "rightring1"),
        ("righthandring2", "ring_2_r", "rightring2"),
        ("righthandring3", "ring_3_r", "rightring3"),
        ("righthandpinky1", "pinky_1_r", "rightpinky1"),
        ("righthandpinky2", "pinky_2_r", "rightpinky2"),
        ("righthandpinky3", "pinky_3_r", "rightpinky3"),
    ];
    FINGERS
        .iter()
        .find(|(long_name, _, short)| name == *long_name || name == *short)
        .map(|(_, canonical, _)| (*canonical, 0))
}

fn canonical_channels(
    bone: &str,
    translation: [f32; 3],
    rotation: [f32; 3],
) -> Vec<(&'static str, f32)> {
    let [rx, ry, rz] = rotation;
    let mut values = match bone {
        "hips" => vec![
            ("x", translation[0]),
            ("y", translation[1]),
            ("z", translation[2]),
            ("turn", ry),
            // Existing raw channels preserve flips and lateral lean even on a
            // target profile that only binds the semantic hips.turn channel.
            ("rotationX", rx),
            ("rotationZ", rz),
        ],
        "spine" | "chest" | "neck" | "head" => vec![("bend", -rx), ("turn", ry), ("twist", rz)],
        "upper_arm_l" => vec![("forward", rx), ("side", rz), ("twist", ry)],
        "upper_arm_r" => vec![("forward", rx), ("side", -rz), ("twist", ry)],
        "forearm_l" | "forearm_r" => vec![("bend", rx), ("twist", ry), ("rotationZ", rz)],
        "upper_leg_l" => vec![("forward", -rx), ("side", -rz), ("twist", ry)],
        "upper_leg_r" => vec![("forward", -rx), ("side", rz), ("twist", ry)],
        "lower_leg_l" | "lower_leg_r" => vec![("bend", -rx), ("twist", ry), ("rotationZ", rz)],
        "shoulder_l" | "shoulder_r" | "hand_l" | "hand_r" => {
            vec![("rotationX", rx), ("rotationY", ry), ("rotationZ", rz)]
        }
        "foot_l" | "foot_r" | "toe_l" | "toe_r" => vec![("bend", -rx), ("side", rz), ("twist", ry)],
        value if value.contains("thumb_") => vec![("bend", rx), ("side", rz), ("twist", ry)],
        value
            if value.contains("index_")
                || value.contains("middle_")
                || value.contains("ring_")
                || value.contains("pinky_") =>
        {
            vec![("bend", rx), ("side", rz), ("twist", ry)]
        }
        _ => Vec::new(),
    };
    values.retain(|(_, value)| value.is_finite());
    values
}

fn validate_generated_action(dsl: &str, fps: f32, duration: f32) -> Result<(), ActionToolError> {
    let wrapped = format!(
        "<Graph fps={{{}}} duration=\"{}s\" size={{[64,64]}}>\n{}  <Tex id=\"out\" fmt=\"rgba8unorm\" size={{[64,64]}} />\n  <Present from=\"out\" />\n</Graph>\n",
        format_number(fps),
        format_number(duration.max(0.001)),
        dsl
    );
    motionloom::parse_graph_script(&wrapped)
        .map(|_| ())
        .map_err(|error| ActionToolError::InvalidGeneratedDsl {
            message: error.to_string(),
        })
}

fn quat_to_euler_xyz_deg(value: [f32; 4]) -> [f32; 3] {
    let [x, y, z, w] = quat_normalize(value);
    [
        (2.0 * (w * x + y * z))
            .atan2(1.0 - 2.0 * (x * x + y * y))
            .to_degrees(),
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin().to_degrees(),
        (2.0 * (w * z + x * y))
            .atan2(1.0 - 2.0 * (y * y + z * z))
            .to_degrees(),
    ]
}

/// Choose equivalent Euler representatives continuously across the +/-90-degree
/// decomposition boundary, rather than injecting 180-degree jumps into a roll.
fn continuous_euler(value: [f32; 3], previous: [f32; 3]) -> [f32; 3] {
    let nearest = |candidate: [f32; 3]| {
        std::array::from_fn(|axis| {
            candidate[axis] + 360.0 * ((previous[axis] - candidate[axis]) / 360.0).round()
        })
    };
    let a: [f32; 3] = nearest(value);
    let b: [f32; 3] = nearest([value[0] + 180.0, 180.0 - value[1], value[2] + 180.0]);
    if length3(sub3(a, previous)) <= length3(sub3(b, previous)) {
        a
    } else {
        b
    }
}

fn sample_frame_count(duration: f32, fps: f32) -> usize {
    let count = duration.max(0.0) * fps;
    if (count - count.round()).abs() < 0.0001 {
        count.round() as usize
    } else {
        count.ceil() as usize
    }
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    quat_normalize(quat_mul_raw(a, b))
}
fn quat_mul_raw(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}
fn quat_conjugate(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}
fn quat_normalize(mut q: [f32; 4]) -> [f32; 4] {
    let length = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length <= f32::EPSILON {
        return [0.0, 0.0, 0.0, 1.0];
    }
    for v in &mut q {
        *v /= length;
    }
    q
}
fn slerp_quat(a: [f32; 4], mut b: [f32; 4], t: f32) -> [f32; 4] {
    let a = quat_normalize(a);
    b = quat_normalize(b);
    let mut dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    if dot < 0.0 {
        for v in &mut b {
            *v = -*v;
        }
        dot = -dot;
    }
    if dot > 0.9995 {
        return quat_normalize(std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t));
    }
    let angle = dot.clamp(-1.0, 1.0).acos();
    let wa = ((1.0 - t) * angle).sin() / angle.sin();
    let wb = (t * angle).sin() / angle.sin();
    quat_normalize(std::array::from_fn(|i| a[i] * wa + b[i] * wb))
}
fn rotate_vec3(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let r = quat_mul_raw(quat_mul_raw(q, [v[0], v[1], v[2], 0.0]), quat_conjugate(q));
    [r[0], r[1], r[2]]
}
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn mul_vec3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] * b[i])
}
fn length3(v: [f32; 3]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn format_number(value: f32) -> String {
    let integer = value.round();
    let value = if (value - integer).abs() < 0.0001 {
        integer
    } else {
        value
    };
    let mut text = format!("{value:.6}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    if text == "-0" { "0".into() } else { text }
}
fn escape_xml_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_decomposition_boundary_does_not_flip_axes() {
        let result = continuous_euler([180.0, 89.0, 180.0], [0.0, 89.0, 0.0]);
        assert_eq!(result, [0.0, 91.0, 0.0]);
    }

    #[test]
    fn step_sampling_changes_at_the_key_not_after_it() {
        let track = AnimationTrack {
            node_index: 0,
            property: TrackProperty::Translation,
            interpolation: TrackInterpolation::Step,
            times: vec![0.0, 1.0],
            values: TrackValues::Vec3(vec![[0.0; 3], [1.0; 3]]),
        };
        assert_eq!(
            sample_track(&track, 1.0),
            Some(TrackValues::Vec3(vec![[1.0; 3]]))
        );
    }

    #[test]
    fn reduction_bounds_every_original_sample() {
        let poses = (0..61)
            .map(|i| PoseSample {
                time: i as f32 / 30.0,
                bones: BTreeMap::from([(
                    "hips".into(),
                    BTreeMap::from([
                        ("x", 0.05 * (i as f32 / 12.0).sin()),
                        ("turn", 20.0 * (i as f32 / 10.0).sin()),
                    ]),
                )]),
            })
            .collect::<Vec<_>>();
        let reduced = reduce_poses(poses.clone(), 0.2);
        for segment in reduced.windows(2) {
            for sample in poses
                .iter()
                .filter(|pose| pose.time >= segment[0].time && pose.time <= segment[1].time)
            {
                let alpha = (sample.time - segment[0].time) / (segment[1].time - segment[0].time);
                assert!(pose_error(&segment[0], sample, &segment[1], alpha) <= 0.20001);
            }
        }
    }

    #[test]
    fn fbx_humanoid_knees_keep_positive_semantic_flexion() {
        assert_eq!(
            canonical_channels("lower_leg_r", [0.0; 3], [-40.0, 0.0, 0.0])[0],
            ("bend", 40.0)
        );
    }

    #[test]
    fn fbx_humanoid_spine_and_fingers_cover_full_humanoid_v1_profile() {
        assert_eq!(
            canonical_fbx_humanoid_bone("source:Spine"),
            Some(("spine", 1))
        );
        assert_eq!(
            canonical_fbx_humanoid_bone("source:Spine1"),
            Some(("chest", 1))
        );
        assert_eq!(
            canonical_fbx_humanoid_bone("source:Spine2"),
            Some(("upper_chest", 2))
        );
        assert_eq!(
            canonical_fbx_humanoid_bone("source:LeftHandIndex3"),
            Some(("index_3_l", 0))
        );
    }

    #[test]
    fn angle_unwrap_uses_short_path() {
        let mut poses = vec![
            PoseSample {
                time: 0.0,
                bones: BTreeMap::from([("head".into(), BTreeMap::from([("turn", 179.0)]))]),
            },
            PoseSample {
                time: 1.0,
                bones: BTreeMap::from([("head".into(), BTreeMap::from([("turn", -179.0)]))]),
            },
        ];
        unwrap_pose_angles(&mut poses);
        assert_eq!(poses[1].bones["head"]["turn"], 181.0);
    }

    #[test]
    fn key_reduction_retains_extrema() {
        let pose = |time, value| PoseSample {
            time,
            bones: BTreeMap::from([("head".into(), BTreeMap::from([("turn", value)]))]),
        };
        let reduced = reduce_poses(vec![pose(0.0, 0.0), pose(0.5, 10.0), pose(1.0, 0.0)], 0.1);
        assert_eq!(reduced.len(), 3);
    }
}
