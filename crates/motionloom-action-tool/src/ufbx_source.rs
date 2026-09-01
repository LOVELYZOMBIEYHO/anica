// =========================================
// =========================================
// crates/motionloom-action-tool/src/ufbx_source.rs

//! Evaluated FBX import for offline authoring; this dependency never enters MotionLoom.
use crate::{ActionToolError, source::*};
use std::{collections::HashMap, path::Path};

pub(crate) fn load(path: &Path) -> Result<AnimationSource, ActionToolError> {
    let bytes = std::fs::read(path).map_err(|source| ActionToolError::FbxRead {
        path: path.into(),
        source,
    })?;
    let scene = ufbx::load_memory(
        &bytes,
        ufbx::LoadOpts {
            target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
            target_unit_meters: 1.,
            ignore_geometry: true,
            ignore_embedded: true,
            load_external_files: false,
            ..Default::default()
        },
    )
    .map_err(|e| ActionToolError::InvalidAnimation {
        message: format!("ufbx load: {}", e.description),
    })?;
    let mut nodes = Vec::with_capacity(scene.nodes.len());
    let mut ids = HashMap::new();
    for (i, node) in scene.nodes.iter().enumerate() {
        ids.insert(node.element.typed_id as usize, i);
    }
    for node in &scene.nodes {
        let t = &node.local_transform;
        nodes.push(AnimationNode {
            name: node.element.name.to_string(),
            parent: node
                .parent
                .as_ref()
                .and_then(|p| ids.get(&(p.element.typed_id as usize)).copied()),
            rest_translation: [
                t.translation.x as f32,
                t.translation.y as f32,
                t.translation.z as f32,
            ],
            rest_rotation: [
                t.rotation.x as f32,
                t.rotation.y as f32,
                t.rotation.z as f32,
                t.rotation.w as f32,
            ],
            rest_scale: [t.scale.x as f32, t.scale.y as f32, t.scale.z as f32],
        });
    }
    let mut clips = Vec::new();
    for stack in &scene.anim_stacks {
        let baked = ufbx::bake_anim(
            &scene,
            &stack.anim,
            ufbx::BakeOpts {
                trim_start_time: true,
                resample_rate: 120.,
                minimum_sample_rate: 120.,
                maximum_sample_rate: 120.,
                key_reduction_enabled: false,
                ..Default::default()
            },
        )
        .map_err(|e| ActionToolError::InvalidAnimation {
            message: format!("ufbx bake: {}", e.description),
        })?;
        let mut tracks = Vec::new();
        for node in &baked.nodes {
            let Some(&node_index) = ids.get(&(node.typed_id as usize)) else {
                continue;
            };
            let vec3 = |keys: &ufbx::List<ufbx::BakedVec3>| {
                (
                    keys.iter().map(|k| k.time as f32).collect::<Vec<_>>(),
                    TrackValues::Vec3(
                        keys.iter()
                            .map(|k| [k.value.x as f32, k.value.y as f32, k.value.z as f32])
                            .collect(),
                    ),
                )
            };
            let quat = |keys: &ufbx::List<ufbx::BakedQuat>| {
                (
                    keys.iter().map(|k| k.time as f32).collect::<Vec<_>>(),
                    TrackValues::Quat(
                        keys.iter()
                            .map(|k| {
                                [
                                    k.value.x as f32,
                                    k.value.y as f32,
                                    k.value.z as f32,
                                    k.value.w as f32,
                                ]
                            })
                            .collect(),
                    ),
                )
            };
            for (property, (times, values)) in [
                (TrackProperty::Translation, vec3(&node.translation_keys)),
                (TrackProperty::Rotation, quat(&node.rotation_keys)),
                (TrackProperty::Scale, vec3(&node.scale_keys)),
            ] {
                if !times.is_empty() {
                    tracks.push(AnimationTrack {
                        node_index,
                        property,
                        interpolation: TrackInterpolation::Linear,
                        times,
                        values,
                    });
                }
            }
        }
        clips.push(AnimationClip {
            name: stack.element.name.to_string(),
            duration_sec: baked.playback_duration as f32,
            tracks,
        });
    }
    let source=AnimationSource {path:path.into(),backend:"ufbx-evaluated".into(),world_basis:[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],nodes,clips,
        diagnostics:vec!["FBX curves, pivots, rotation order and layers evaluated by offline ufbx; no runtime dependency.".into()]};
    validate_source(&source)?;
    Ok(source)
}
