// =========================================
// =========================================
// crates/motionloom-action-tool/src/gltf_source.rs

use std::path::Path;

use motionloom::GlbMeshData;
use motionloom::experimental::{
    GlbAnimationInterpolation, GlbAnimationProperty, GlbAnimationValues,
};

use crate::ActionToolError;
use crate::source::{
    AnimationClip, AnimationNode, AnimationSource, AnimationTrack, TrackInterpolation,
    TrackProperty, TrackValues,
};

pub(crate) fn load(path: &Path) -> Result<AnimationSource, ActionToolError> {
    let mesh = motionloom::experimental::load_glb_animation_data(path)?;
    Ok(from_mesh(path, &mesh))
}

pub(crate) fn from_mesh(path: &Path, mesh: &GlbMeshData) -> AnimationSource {
    let nodes = mesh
        .nodes
        .iter()
        .map(|node| AnimationNode {
            name: node
                .name
                .clone()
                .unwrap_or_else(|| format!("node_{}", node.index)),
            parent: node.parent,
            rest_translation: node.translation,
            rest_rotation: node.rotation,
            rest_scale: node.scale,
        })
        .collect();
    let clips = mesh
        .animations
        .iter()
        .enumerate()
        .map(|(index, clip)| AnimationClip {
            name: clip.name.clone().unwrap_or_else(|| format!("clip_{index}")),
            duration_sec: clip.duration,
            tracks: clip
                .channels
                .iter()
                .map(|channel| AnimationTrack {
                    node_index: channel.node_index,
                    property: match channel.property {
                        GlbAnimationProperty::Translation => TrackProperty::Translation,
                        GlbAnimationProperty::Rotation => TrackProperty::Rotation,
                        GlbAnimationProperty::Scale => TrackProperty::Scale,
                    },
                    interpolation: match channel.interpolation {
                        GlbAnimationInterpolation::Step => TrackInterpolation::Step,
                        GlbAnimationInterpolation::Linear => TrackInterpolation::Linear,
                        GlbAnimationInterpolation::CubicSpline => TrackInterpolation::Cubic,
                    },
                    times: channel.times.clone(),
                    values: match &channel.values {
                        GlbAnimationValues::Vec3(values) => TrackValues::Vec3(values.clone()),
                        GlbAnimationValues::Quat(values) => TrackValues::Quat(values.clone()),
                    },
                })
                .collect(),
        })
        .collect();
    let diagnostics = if mesh.animations.iter().any(|clip| {
        clip.channels
            .iter()
            .any(|channel| channel.interpolation == GlbAnimationInterpolation::CubicSpline)
    }) {
        vec!["glTF CUBICSPLINE tangents are not exposed by the existing loader; values are interpolated between stored keys. Bake glTF curves before import when sub-key fidelity is required.".into()]
    } else {
        Vec::new()
    };
    AnimationSource {
        path: path.to_path_buf(),
        backend: "gltf-native".to_string(),
        world_basis: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        nodes,
        clips,
        diagnostics,
    }
}
