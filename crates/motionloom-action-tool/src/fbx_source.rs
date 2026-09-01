// =========================================
// =========================================
// crates/motionloom-action-tool/src/fbx_source.rs

use std::collections::HashMap;
use std::path::Path;

use draco_io::fbx_reader::{FbxNode, FbxProperty};
use draco_io::{
    FbxAnimChannelPath, FbxAnimInterpolation, FbxReader, FbxSceneNode, FbxTransformStack,
};

use crate::ActionToolError;
use crate::source::{
    AnimationClip, AnimationNode, AnimationSource, AnimationTrack, TrackInterpolation,
    TrackProperty, TrackValues,
};

pub(crate) fn load(path: &Path) -> Result<AnimationSource, ActionToolError> {
    let mut reader = FbxReader::open(path).map_err(|source| ActionToolError::FbxRead {
        path: path.to_path_buf(),
        source,
    })?;
    // Inspect raw curves before the scene reader combines XYZ by key index.
    // Unaligned component keys require a full FBX evaluator, not guessed values.
    let raw = reader
        .read_nodes()
        .map_err(|source| ActionToolError::FbxRead {
            path: path.to_path_buf(),
            source,
        })?;
    validate_native_document(&raw)?;
    let scene = reader
        .read_scene()
        .map_err(|source| ActionToolError::FbxRead {
            path: path.to_path_buf(),
            source,
        })?;
    for root in &scene.root_nodes {
        validate_native_transform(root)?;
    }
    if scene.warnings.iter().any(|warning| {
        matches!(
            warning.code,
            draco_io::FbxWarningCode::UnsupportedTransformInherit
                | draco_io::FbxWarningCode::NameKeyedObjectModel
                | draco_io::FbxWarningCode::MissingNodeEndOffset
        )
    }) {
        return Err(ActionToolError::FbxNeedsEvaluation { reason: "source transform or object model cannot be evaluated faithfully by the native reader".into() });
    }
    let basis = SourceBasis::from_settings(scene.global_settings.as_ref());
    let unit_to_meters = scene
        .global_settings
        .as_ref()
        .and_then(|settings| settings.unit_scale_factor)
        .unwrap_or(1.0) as f32
        * 0.01;
    if !unit_to_meters.is_finite() || unit_to_meters <= 0.0 {
        return Err(ActionToolError::InvalidAnimation {
            message: "FBX has invalid UnitScaleFactor".into(),
        });
    }
    let mut nodes = Vec::new();
    let mut node_ids = HashMap::new();
    for root in &scene.root_nodes {
        flatten_node(root, None, unit_to_meters, &mut nodes, &mut node_ids);
    }
    let mut clips = Vec::new();
    for (clip_index, animation) in scene.animations.iter().enumerate() {
        let start = animation
            .channels
            .iter()
            .flat_map(|channel| channel.sampler.input.first().copied())
            .fold(f32::INFINITY, f32::min);
        let start = if start.is_finite() { start } else { 0.0 };
        let mut tracks = Vec::new();
        for channel in &animation.channels {
            let Some(&node_index) = node_ids.get(&channel.node_id) else {
                continue;
            };
            if channel.path == FbxAnimChannelPath::MorphWeight {
                continue;
            }
            let baked = bake_sampler(&channel.sampler)?;
            let source_values = baked.values;
            let interpolation = baked.interpolation;
            let times = baked
                .times
                .into_iter()
                .map(|time| (time - start).max(0.0))
                .collect();
            let (property, values) = match channel.path {
                FbxAnimChannelPath::Translation => (
                    TrackProperty::Translation,
                    TrackValues::Vec3(
                        source_values
                            .into_iter()
                            .map(|value| scale3(value, unit_to_meters))
                            .collect(),
                    ),
                ),
                FbxAnimChannelPath::Rotation => {
                    let stack = node_transform_stack(&scene.root_nodes, channel.node_id);
                    (
                        TrackProperty::Rotation,
                        TrackValues::Quat(
                            source_values
                                .into_iter()
                                .map(|rotation| {
                                    // draco-io exposes animated Euler keys in radians,
                                    // while the static FBX transform stack is degrees.
                                    fbx_stack_rotation(stack, rotation.map(f32::to_degrees))
                                })
                                .collect(),
                        ),
                    )
                }
                FbxAnimChannelPath::Scale => {
                    (TrackProperty::Scale, TrackValues::Vec3(source_values))
                }
                FbxAnimChannelPath::MorphWeight => continue,
            };
            tracks.push(AnimationTrack {
                node_index,
                property,
                interpolation,
                times,
                values,
            });
        }
        let duration_sec = tracks
            .iter()
            .flat_map(|track| track.times.last().copied())
            .fold(0.0, f32::max);
        clips.push(AnimationClip {
            name: animation
                .name
                .clone()
                .unwrap_or_else(|| format!("clip_{clip_index}")),
            duration_sec,
            tracks,
        });
    }
    let mut diagnostics = scene
        .warnings
        .iter()
        .map(|warning| {
            format!(
                "FBX {}{}: {}",
                warning.code.as_str(),
                warning
                    .subject
                    .as_deref()
                    .map(|subject| format!(" [{subject}]"))
                    .unwrap_or_default(),
                warning
            )
        })
        .collect::<Vec<_>>();
    if scene.global_settings.is_none() {
        diagnostics.push(
            "FBX omitted GlobalSettings; assumed +Y up, +X right, -Z forward, centimetres"
                .to_string(),
        );
    }
    if scene
        .warnings
        .iter()
        .any(|warning| warning.code.is_data_loss())
    {
        diagnostics.push(
            "FBX reader reported unsupported source data; use --fbx-backend blender for comparison"
                .to_string(),
        );
    }
    let source = AnimationSource {
        path: path.to_path_buf(),
        backend: "fbx-native".to_string(),
        world_basis: basis.rows,
        nodes,
        clips,
        diagnostics,
    };
    crate::source::validate_source(&source)?;
    Ok(source)
}

/// Native mode is deliberately conservative; fallback owns complex FBX evaluation.
fn validate_native_transform(node: &FbxSceneNode) -> Result<(), ActionToolError> {
    if let Some(stack) = &node.transform_stack {
        let nonzero =
            |value: Option<[f32; 3]>| value.is_some_and(|v| v.iter().any(|x| x.abs() > 1e-6));
        if nonzero(stack.rotation_pivot)
            || nonzero(stack.rotation_offset)
            || nonzero(stack.scaling_pivot)
            || nonzero(stack.scaling_offset)
            || stack
                .rotation_order
                .is_some_and(|order| !(0..=5).contains(&order))
        {
            return Err(ActionToolError::FbxNeedsEvaluation {
                reason: format!(
                    "{} uses pivot/offset or spherical rotation",
                    node.name.as_deref().unwrap_or("unnamed node")
                ),
            });
        }
        let scale = stack.scaling.unwrap_or([1.0; 3]);
        if (scale[0] - scale[1]).abs() > 1e-5
            || (scale[0] - scale[2]).abs() > 1e-5
            || scale[0] <= 0.0
        {
            return Err(ActionToolError::FbxNeedsEvaluation {
                reason: "non-uniform or reflected source scaling".into(),
            });
        }
    }
    for child in &node.children {
        validate_native_transform(child)?;
    }
    Ok(())
}

fn validate_native_document(roots: &[FbxNode]) -> Result<(), ActionToolError> {
    let Some(objects) = roots.iter().find(|node| node.name == "Objects") else {
        return Ok(());
    };
    let needs = |reason: &str| ActionToolError::FbxNeedsEvaluation {
        reason: reason.into(),
    };
    if objects
        .children
        .iter()
        .any(|node| node.name == "Constraint")
    {
        return Err(needs("source contains unbaked constraints"));
    }
    let stacks = objects
        .children
        .iter()
        .filter(|node| node.name == "AnimationStack")
        .count();
    let layers = objects
        .children
        .iter()
        .filter(|node| node.name == "AnimationLayer")
        .count();
    if layers > stacks {
        return Err(needs(
            "multiple animation layers require evaluated blending",
        ));
    }
    let id = |property: Option<&FbxProperty>| match property {
        Some(FbxProperty::I64(value)) => Some(*value),
        Some(FbxProperty::I32(value)) => Some(i64::from(*value)),
        _ => None,
    };
    let curves = objects
        .children
        .iter()
        .filter(|node| node.name == "AnimationCurve")
        .filter_map(|node| id(node.properties.first()).map(|key| (key, node)))
        .collect::<HashMap<_, _>>();
    for curve in curves.values() {
        if let Some(FbxProperty::I32Array(flags)) = curve
            .children
            .iter()
            .find(|child| child.name == "KeyAttrFlags")
            .and_then(|child| child.properties.first())
        {
            if flags.iter().any(|flag| flag & 0x03000000 != 0)
                || flags
                    .windows(2)
                    .any(|pair| pair[0] & 0x0e != pair[1] & 0x0e)
            {
                return Err(needs(
                    "weighted or mixed-interpolation FBX curves need evaluated baking",
                ));
            }
        }
    }
    let mut grouped = HashMap::<i64, Vec<&FbxNode>>::new();
    if let Some(connections) = roots.iter().find(|node| node.name == "Connections") {
        for connection in &connections.children {
            if let (Some(child), Some(parent)) = (
                id(connection.properties.get(1)),
                id(connection.properties.get(2)),
            ) {
                if let Some(curve) = curves.get(&child) {
                    grouped.entry(parent).or_default().push(curve);
                }
            }
        }
    }
    let times = |node: &FbxNode| {
        node.children
            .iter()
            .find(|child| child.name == "KeyTime")
            .and_then(|child| child.properties.first())
            .and_then(|value| match value {
                FbxProperty::I64Array(values) => Some(values.clone()),
                _ => None,
            })
    };
    for curves in grouped.values() {
        if curves.len() != 3
            || curves
                .windows(2)
                .any(|pair| times(pair[0]) != times(pair[1]))
        {
            return Err(needs(
                "XYZ animation keys are sparse or use different timestamps",
            ));
        }
    }
    Ok(())
}

fn flatten_node(
    source: &FbxSceneNode,
    parent: Option<usize>,
    unit_to_meters: f32,
    nodes: &mut Vec<AnimationNode>,
    node_ids: &mut HashMap<draco_io::FbxNodeId, usize>,
) {
    let stack = source.transform_stack.as_ref();
    let translation = stack
        .and_then(|stack| stack.translation)
        .unwrap_or([0.0; 3]);
    let rotation = stack.and_then(|stack| stack.rotation).unwrap_or([0.0; 3]);
    let scale = stack.and_then(|stack| stack.scaling).unwrap_or([1.0; 3]);
    let index = nodes.len();
    nodes.push(AnimationNode {
        name: source
            .name
            .clone()
            .unwrap_or_else(|| format!("fbx_node_{}", source.id.0)),
        parent,
        rest_translation: scale3(translation, unit_to_meters),
        rest_rotation: fbx_stack_rotation(stack, rotation),
        rest_scale: scale,
    });
    node_ids.insert(source.id, index);
    for child in &source.children {
        flatten_node(child, Some(index), unit_to_meters, nodes, node_ids);
    }
}

fn node_transform_stack(
    roots: &[FbxSceneNode],
    target: draco_io::FbxNodeId,
) -> Option<&FbxTransformStack> {
    fn find(node: &FbxSceneNode, target: draco_io::FbxNodeId) -> Option<&FbxTransformStack> {
        if node.id == target {
            return node.transform_stack.as_ref();
        }
        node.children.iter().find_map(|child| find(child, target))
    }
    roots.iter().find_map(|root| find(root, target))
}

fn chunk_vec3(values: &[f32]) -> Vec<[f32; 3]> {
    values
        .chunks_exact(3)
        .map(|value| [value[0], value[1], value[2]])
        .collect()
}

fn scale3(value: [f32; 3], scale: f32) -> [f32; 3] {
    value.map(|component| component * scale)
}

/// Bake source Euler curves before quaternion conversion so full turns and
/// cubic tangents survive. The final Action is sampled at the requested FPS.
struct BakedSampler {
    times: Vec<f32>,
    values: Vec<[f32; 3]>,
    interpolation: TrackInterpolation,
}

fn bake_sampler(sampler: &draco_io::FbxAnimSampler) -> Result<BakedSampler, ActionToolError> {
    let values = chunk_vec3(&sampler.output);
    let times = &sampler.input;
    if times.is_empty()
        || values.len() != times.len()
        || times.iter().any(|time| !time.is_finite())
        || times.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(ActionToolError::InvalidAnimation {
            message: "invalid FBX source sampler".into(),
        });
    }
    if sampler.interpolation == FbxAnimInterpolation::Step || times.len() == 1 {
        return Ok(BakedSampler {
            times: times.clone(),
            values,
            interpolation: TrackInterpolation::Step,
        });
    }
    let cubic = sampler.interpolation == FbxAnimInterpolation::Cubic;
    let incoming = sampler
        .in_tangents
        .as_deref()
        .map(chunk_vec3)
        .unwrap_or_default();
    let outgoing = sampler
        .out_tangents
        .as_deref()
        .map(chunk_vec3)
        .unwrap_or_default();
    if cubic && (incoming.len() != values.len() || outgoing.len() != values.len()) {
        return Err(ActionToolError::FbxNeedsEvaluation {
            reason: "missing FBX cubic tangents".into(),
        });
    }
    let mut baked_times = Vec::new();
    let mut baked_values = Vec::new();
    for index in 0..times.len() - 1 {
        let span = times[index + 1] - times[index];
        let steps = (span * 120.0).ceil().max(1.0) as usize;
        if baked_times.len().saturating_add(steps) > 1_000_000 {
            return Err(ActionToolError::InvalidAnimation {
                message: "FBX track exceeds bake sample limit".into(),
            });
        }
        for step in 0..steps {
            let t = step as f32 / steps as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            let value = std::array::from_fn(|axis| {
                if cubic {
                    (2.0 * t3 - 3.0 * t2 + 1.0) * values[index][axis]
                        + (t3 - 2.0 * t2 + t) * span * outgoing[index][axis]
                        + (-2.0 * t3 + 3.0 * t2) * values[index + 1][axis]
                        + (t3 - t2) * span * incoming[index + 1][axis]
                } else {
                    values[index][axis] + (values[index + 1][axis] - values[index][axis]) * t
                }
            });
            baked_times.push(times[index] + span * t);
            baked_values.push(value);
        }
    }
    baked_times.push(*times.last().expect("non-empty sampler"));
    baked_values.push(*values.last().expect("non-empty values"));
    Ok(BakedSampler {
        times: baked_times,
        values: baked_values,
        interpolation: TrackInterpolation::Linear,
    })
}

fn fbx_stack_rotation(stack: Option<&FbxTransformStack>, local: [f32; 3]) -> [f32; 4] {
    let order = stack.and_then(|stack| stack.rotation_order).unwrap_or(0);
    let pre = euler_quaternion(
        stack
            .and_then(|stack| stack.pre_rotation)
            .unwrap_or([0.0; 3]),
        0,
    );
    let animated = euler_quaternion(local, order);
    let post = euler_quaternion(
        stack
            .and_then(|stack| stack.post_rotation)
            .unwrap_or([0.0; 3]),
        0,
    );
    quat_normalize(quat_mul(pre, quat_mul(animated, quat_conjugate(post))))
}

fn euler_quaternion(rotation: [f32; 3], order: i32) -> [f32; 4] {
    let axes = match order {
        1 => [0, 2, 1],
        2 => [1, 2, 0],
        3 => [1, 0, 2],
        4 => [2, 0, 1],
        5 => [2, 1, 0],
        _ => [0, 1, 2],
    };
    let mut result = [0.0, 0.0, 0.0, 1.0];
    for axis in axes {
        let half = rotation[axis].to_radians() * 0.5;
        let mut part = [0.0, 0.0, 0.0, half.cos()];
        part[axis] = half.sin();
        result = quat_mul(part, result);
    }
    quat_normalize(result)
}

#[derive(Debug, Clone, Copy)]
struct SourceBasis {
    rows: [[f32; 3]; 3],
}

impl SourceBasis {
    fn from_settings(settings: Option<&draco_io::FbxGlobalSettings>) -> Self {
        let axis = |index: Option<i32>, sign: Option<i32>, fallback: [f32; 3]| {
            let Some(index) = index.and_then(|value| usize::try_from(value).ok()) else {
                return fallback;
            };
            if index > 2 {
                return fallback;
            }
            let mut value = [0.0; 3];
            value[index] = sign.unwrap_or(1) as f32;
            value
        };
        let right = axis(
            settings.and_then(|value| value.coord_axis),
            settings.and_then(|value| value.coord_axis_sign),
            [1.0, 0.0, 0.0],
        );
        let up = axis(
            settings.and_then(|value| value.up_axis),
            settings.and_then(|value| value.up_axis_sign),
            [0.0, 1.0, 0.0],
        );
        let forward = axis(
            settings.and_then(|value| value.front_axis),
            settings.and_then(|value| value.front_axis_sign),
            [0.0, 0.0, -1.0],
        );
        Self {
            rows: [right, up, forward.map(|component| -component)],
        }
    }
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn quat_conjugate(value: [f32; 4]) -> [f32; 4] {
    [-value[0], -value[1], -value[2], value[3]]
}

fn quat_normalize(mut value: [f32; 4]) -> [f32; 4] {
    let length = value.iter().map(|part| part * part).sum::<f32>().sqrt();
    if length <= f32::EPSILON {
        return [0.0, 0.0, 0.0, 1.0];
    }
    for part in &mut value {
        *part /= length;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fbx_humanoid_basis_is_identity() {
        let basis = SourceBasis::from_settings(None);
        assert_eq!(
            basis.rows,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn rotation_order_changes_quaternion_composition() {
        let xyz = euler_quaternion([30.0, 40.0, 50.0], 0);
        let zyx = euler_quaternion([30.0, 40.0, 50.0], 5);
        assert_ne!(xyz, zyx);
    }

    #[test]
    fn cubic_bake_evaluates_source_tangents() {
        let sampler = draco_io::FbxAnimSampler {
            input: vec![0.0, 1.0],
            output: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            interpolation: FbxAnimInterpolation::Cubic,
            in_tangents: Some(vec![0.0; 6]),
            out_tangents: Some(vec![0.0; 6]),
        };
        let baked = bake_sampler(&sampler).unwrap();
        let index = baked
            .times
            .iter()
            .position(|time| (*time - 0.25).abs() < 1e-5)
            .unwrap();
        assert!((baked.values[index][0] - 0.15625).abs() < 1e-5);
    }

    #[test]
    fn pre_and_post_rotations_cancel_at_rest() {
        let stack = FbxTransformStack {
            translation: None,
            rotation: None,
            scaling: None,
            rotation_order: Some(0),
            rotation_active: Some(true),
            pre_rotation: Some([30.0, 20.0, 10.0]),
            post_rotation: Some([30.0, 20.0, 10.0]),
            rotation_offset: None,
            rotation_pivot: None,
            scaling_offset: None,
            scaling_pivot: None,
            inherit_type: None,
        };
        let q = fbx_stack_rotation(Some(&stack), [0.0; 3]);
        assert!(q[..3].iter().all(|x| x.abs() < 1e-5));
    }
}
