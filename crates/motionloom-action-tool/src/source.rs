// =========================================
// =========================================
// crates/motionloom-action-tool/src/source.rs

use std::path::PathBuf;

/// Format-neutral animation data shared by the glTF and FBX importers.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationSource {
    pub path: PathBuf,
    pub backend: String,
    /// Basis applied to source-world positions, never to bone-local axes.
    /// Preserving local axes is essential for the source retarget profile.
    pub world_basis: [[f32; 3]; 3],
    pub nodes: Vec<AnimationNode>,
    pub clips: Vec<AnimationClip>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnimationNode {
    pub name: String,
    pub parent: Option<usize>,
    pub rest_translation: [f32; 3],
    pub rest_rotation: [f32; 4],
    pub rest_scale: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnimationClip {
    pub name: String,
    pub duration_sec: f32,
    pub tracks: Vec<AnimationTrack>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnimationTrack {
    pub node_index: usize,
    pub property: TrackProperty,
    pub interpolation: TrackInterpolation,
    pub times: Vec<f32>,
    pub values: TrackValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackProperty {
    Translation,
    Rotation,
    Scale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackInterpolation {
    Step,
    Linear,
    Cubic,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackValues {
    Vec3(Vec<[f32; 3]>),
    Quat(Vec<[f32; 4]>),
}

/// Reject malformed input before recursive FK or allocating dense samples.
pub(crate) fn validate_source(source: &AnimationSource) -> Result<(), crate::ActionToolError> {
    let invalid = |message: String| crate::ActionToolError::InvalidAnimation { message };
    for (index, node) in source.nodes.iter().enumerate() {
        if !node
            .rest_translation
            .iter()
            .chain(&node.rest_rotation)
            .chain(&node.rest_scale)
            .all(|v| v.is_finite())
        {
            return Err(invalid(format!(
                "node {} contains non-finite transforms",
                node.name
            )));
        }
        let mut parent = node.parent;
        let mut visited = std::collections::BTreeSet::from([index]);
        while let Some(index) = parent {
            if index >= source.nodes.len() || !visited.insert(index) {
                return Err(invalid(format!(
                    "node {} has an invalid or cyclic parent chain",
                    node.name
                )));
            }
            parent = source.nodes[index].parent;
        }
    }
    for clip in &source.clips {
        if !clip.duration_sec.is_finite() || clip.duration_sec <= 0.0 {
            return Err(invalid(format!(
                "clip {} must have positive finite duration",
                clip.name
            )));
        }
        for track in &clip.tracks {
            if track.node_index >= source.nodes.len()
                || track.times.is_empty()
                || track
                    .times
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0 || *v > clip.duration_sec + 0.001)
                || track.times.windows(2).any(|pair| pair[1] <= pair[0])
            {
                return Err(invalid(format!(
                    "clip {} contains invalid track keys or target",
                    clip.name
                )));
            }
            let (count, finite, valid_type) = match &track.values {
                TrackValues::Vec3(values) => (
                    values.len(),
                    values.iter().flatten().all(|v| v.is_finite()),
                    track.property != TrackProperty::Rotation,
                ),
                TrackValues::Quat(values) => (
                    values.len(),
                    values.iter().all(|q| {
                        q.iter().all(|v| v.is_finite())
                            && q.iter().map(|v| v * v).sum::<f32>() > f32::EPSILON
                    }),
                    track.property == TrackProperty::Rotation,
                ),
            };
            if count != track.times.len() || !finite || !valid_type {
                return Err(invalid(format!(
                    "clip {} contains malformed track values",
                    clip.name
                )));
            }
        }
    }
    Ok(())
}
