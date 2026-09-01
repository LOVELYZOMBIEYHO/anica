use std::collections::{HashMap, HashSet};

use crate::dsl::GraphScript;
use crate::scene::backend::sizing::format_scene_number;
use crate::scene::dsl::{
    ActionIkNode, ActionNode, ApplyActionNode, SkeletonBoneNode, SkeletonNode,
};
use crate::scene::model::{PartNode, SceneNode};
use crate::scene::render::{MotionLoomSceneRenderError, eval_scene_number};
use crate::scene::spatial::Affine2;

#[derive(Debug, Clone, Copy, Default)]
struct ActionBoneSample {
    x: Option<f32>,
    y: Option<f32>,
    rotation: Option<f32>,
    scale: Option<f32>,
    opacity: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
struct BoneWorldSample {
    transform: Affine2,
    rotation: f32,
    scale: f32,
    opacity: f32,
}

pub(crate) fn apply_action_graph_at_time(
    graph: &GraphScript,
    time_norm: f32,
    time_sec: f32,
) -> Result<Option<GraphScript>, MotionLoomSceneRenderError> {
    if graph.actions.is_empty() || graph.apply_actions.is_empty() {
        return Ok(None);
    }

    let action_map = graph
        .actions
        .iter()
        .map(|action| (action.id.as_str(), action))
        .collect::<HashMap<_, _>>();
    let skeleton_map = graph
        .skeletons
        .iter()
        .map(|skeleton| (skeleton.id.as_str(), skeleton))
        .collect::<HashMap<_, _>>();
    let mut next = graph.clone();

    for apply in &graph.apply_actions {
        let Some(action) = action_map.get(apply.action.as_str()) else {
            return Err(MotionLoomSceneRenderError::InvalidExpression {
                expr: apply.action.clone(),
                message: "ApplyAction references an unknown Action".to_string(),
            });
        };
        // External Actions are sampled by the retained 3D renderer. They are
        // still real Action nodes; AnimationAsset ids never enter this API.
        if action.source.is_some() {
            continue;
        }
        let samples = sample_action_bones(action, apply, time_norm, time_sec)?;
        if samples.is_empty() && action.iks.is_empty() {
            continue;
        }
        apply_action_to_nodes(
            &mut next.scene_nodes,
            &apply.target,
            action,
            &skeleton_map,
            &samples,
            time_norm,
            time_sec,
        )?;
        for scene in &mut next.scenes {
            apply_action_to_nodes(
                &mut scene.children,
                &apply.target,
                action,
                &skeleton_map,
                &samples,
                time_norm,
                time_sec,
            )?;
        }
    }

    Ok(Some(next))
}

fn sample_action_bones(
    action: &ActionNode,
    apply: &ApplyActionNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<HashMap<String, ActionBoneSample>, MotionLoomSceneRenderError> {
    let local_sec = time_sec - apply.at_ms as f32 / 1000.0;
    let duration_sec = (action.duration_ms as f32 / 1000.0).max(0.0001);
    if local_sec < -0.0001 || local_sec > duration_sec + 0.0001 || action.poses.is_empty() {
        return Ok(HashMap::new());
    }
    let local_sec = local_sec.clamp(0.0, duration_sec);

    let mut prev_ix = 0usize;
    let mut next_ix = action.poses.len().saturating_sub(1);
    for (ix, pose) in action.poses.iter().enumerate() {
        if pose.t <= local_sec {
            prev_ix = ix;
        }
        if pose.t >= local_sec {
            next_ix = ix;
            break;
        }
    }
    if action.poses[prev_ix].t > local_sec {
        prev_ix = next_ix;
    }

    let prev_pose = &action.poses[prev_ix];
    let next_pose = &action.poses[next_ix];
    let span = (next_pose.t - prev_pose.t).abs();
    let mix = if span <= 0.0001 {
        0.0
    } else {
        ((local_sec - prev_pose.t) / span).clamp(0.0, 1.0)
    };

    let mut bone_ids = Vec::<String>::new();
    for bone in &prev_pose.bones {
        if !bone_ids.iter().any(|id| id == &bone.id) {
            bone_ids.push(bone.id.clone());
        }
    }
    for bone in &next_pose.bones {
        if !bone_ids.iter().any(|id| id == &bone.id) {
            bone_ids.push(bone.id.clone());
        }
    }

    let mut samples = HashMap::<String, ActionBoneSample>::new();
    for bone_id in bone_ids {
        let prev = prev_pose.bones.iter().find(|bone| bone.id == bone_id);
        let next = next_pose.bones.iter().find(|bone| bone.id == bone_id);
        let bone_mix = action_key_mix(prev, next, mix);
        let sample = ActionBoneSample {
            x: interpolate_action_attr(
                prev.and_then(|bone| bone.x.as_ref()),
                next.and_then(|bone| bone.x.as_ref()),
                bone_mix,
                time_norm,
                time_sec,
            )?,
            y: interpolate_action_attr(
                prev.and_then(|bone| bone.y.as_ref()),
                next.and_then(|bone| bone.y.as_ref()),
                bone_mix,
                time_norm,
                time_sec,
            )?,
            rotation: interpolate_action_attr(
                prev.and_then(|bone| bone.rotation.as_ref()),
                next.and_then(|bone| bone.rotation.as_ref()),
                bone_mix,
                time_norm,
                time_sec,
            )?,
            scale: interpolate_action_attr(
                prev.and_then(|bone| bone.scale.as_ref()),
                next.and_then(|bone| bone.scale.as_ref()),
                bone_mix,
                time_norm,
                time_sec,
            )?,
            opacity: interpolate_action_attr(
                prev.and_then(|bone| bone.opacity.as_ref()),
                next.and_then(|bone| bone.opacity.as_ref()),
                bone_mix,
                time_norm,
                time_sec,
            )?,
        };
        samples.insert(bone_id, sample);
    }

    Ok(samples)
}

pub(crate) fn action_key_mix(
    prev: Option<&crate::scene::dsl::ActionBoneNode>,
    next: Option<&crate::scene::dsl::ActionBoneNode>,
    phase: f32,
) -> f32 {
    let phase = phase.clamp(0.0, 1.0);
    match prev
        .and_then(|bone| bone.interpolation.as_deref())
        .unwrap_or("linear")
    {
        "hold" => 0.0,
        "ease" => phase * phase * (3.0 - 2.0 * phase),
        "bezier" => {
            let m0 = prev
                .and_then(|bone| bone.out_tangent.as_deref())
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0);
            let m1 = next
                .and_then(|bone| bone.in_tangent.as_deref())
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0);
            let t2 = phase * phase;
            let t3 = t2 * phase;
            ((-2.0 * t3 + 3.0 * t2) + (t3 - 2.0 * t2 + phase) * m0 + (t3 - t2) * m1).clamp(0.0, 1.0)
        }
        _ => phase,
    }
}

fn interpolate_action_attr(
    prev: Option<&String>,
    next: Option<&String>,
    mix: f32,
    time_norm: f32,
    time_sec: f32,
) -> Result<Option<f32>, MotionLoomSceneRenderError> {
    let prev = prev
        .map(|value| eval_scene_number(value, time_norm, time_sec))
        .transpose()?;
    let next = next
        .map(|value| eval_scene_number(value, time_norm, time_sec))
        .transpose()?;
    Ok(match (prev, next) {
        (Some(a), Some(b)) => Some(a + (b - a) * mix),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    })
}

fn apply_action_to_nodes(
    nodes: &mut [SceneNode],
    target: &str,
    action: &ActionNode,
    skeleton_map: &HashMap<&str, &SkeletonNode>,
    samples: &HashMap<String, ActionBoneSample>,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    for node in nodes {
        match node {
            SceneNode::Timeline(timeline) => apply_action_to_nodes(
                &mut timeline.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Track(track) => apply_action_to_nodes(
                &mut track.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Sequence(sequence) => apply_action_to_nodes(
                &mut sequence.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Chain(chain) => apply_action_to_nodes(
                &mut chain.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Layer(layer) => apply_action_to_nodes(
                &mut layer.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Character(character) => {
                if character.id.as_deref() == Some(target) {
                    let skeleton_id = action.skeleton.as_deref().or(character.rig.as_deref());
                    if let Some(skeleton) = skeleton_id.and_then(|id| skeleton_map.get(id).copied())
                    {
                        let mut character_samples = samples.clone();
                        apply_ik_targets_to_samples(
                            skeleton,
                            &action.iks,
                            &mut character_samples,
                            time_norm,
                            time_sec,
                        )?;
                        apply_skeleton_rotation_limits(
                            skeleton,
                            &mut character_samples,
                            time_norm,
                            time_sec,
                        )?;
                        let bone_world = sample_skeleton_bones(
                            skeleton,
                            &character_samples,
                            time_norm,
                            time_sec,
                        )?;
                        apply_skeleton_action_to_character_children(
                            &mut character.children,
                            &bone_world,
                            &character_samples,
                            time_norm,
                            time_sec,
                        )?;
                    } else {
                        apply_action_to_character_children(
                            &mut character.children,
                            samples,
                            time_norm,
                            time_sec,
                        )?;
                    }
                } else {
                    apply_action_to_nodes(
                        &mut character.children,
                        target,
                        action,
                        skeleton_map,
                        samples,
                        time_norm,
                        time_sec,
                    )?;
                }
            }
            SceneNode::Group(group) => apply_action_to_nodes(
                &mut group.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Part(part) => apply_action_to_nodes(
                &mut part.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Repeat(repeat) => apply_action_to_nodes(
                &mut repeat.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Mask(mask) => apply_action_to_nodes(
                &mut mask.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Precompose(precompose) => apply_action_to_nodes(
                &mut precompose.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Camera(camera) => apply_action_to_nodes(
                &mut camera.children,
                target,
                action,
                skeleton_map,
                samples,
                time_norm,
                time_sec,
            )?,
            _ => {}
        }
    }
    Ok(())
}

fn apply_skeleton_rotation_limits(
    skeleton: &SkeletonNode,
    samples: &mut HashMap<String, ActionBoneSample>,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    // Angle limits are solved after FK/IK so every animation path obeys the same rig contract.
    for constraint in skeleton
        .constraints
        .iter()
        .filter(|constraint| constraint.kind == "anglelimit")
    {
        let (Some(bone_id), Some(min), Some(max)) = (
            constraint.bone.as_deref(),
            constraint.min.as_deref(),
            constraint.max.as_deref(),
        ) else {
            continue;
        };
        let bone = skeleton
            .bones
            .iter()
            .find(|bone| bone.id == bone_id)
            .ok_or_else(|| MotionLoomSceneRenderError::InvalidExpression {
                expr: bone_id.to_string(),
                message: format!("Skeleton {} constraint bone not found.", skeleton.id),
            })?;
        let base = eval_scene_number(&bone.rotation, time_norm, time_sec)?;
        let min = eval_scene_number(min, time_norm, time_sec)?;
        let max = eval_scene_number(max, time_norm, time_sec)?;
        let entry = samples.entry(bone_id.to_string()).or_default();
        let total = base + entry.rotation.unwrap_or(0.0);
        entry.rotation = Some(total.clamp(min.min(max), min.max(max)) - base);
    }
    Ok(())
}

fn sample_skeleton_bones(
    skeleton: &SkeletonNode,
    samples: &HashMap<String, ActionBoneSample>,
    time_norm: f32,
    time_sec: f32,
) -> Result<HashMap<String, BoneWorldSample>, MotionLoomSceneRenderError> {
    let bone_ids = skeleton
        .bones
        .iter()
        .map(|bone| bone.id.as_str())
        .collect::<HashSet<_>>();
    let mut out = HashMap::<String, BoneWorldSample>::new();
    let mut visiting = HashSet::<String>::new();
    for bone in &skeleton.bones {
        sample_skeleton_bone(
            skeleton,
            &bone_ids,
            &bone.id,
            samples,
            time_norm,
            time_sec,
            &mut visiting,
            &mut out,
        )?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn sample_skeleton_bone(
    skeleton: &SkeletonNode,
    bone_ids: &HashSet<&str>,
    bone_id: &str,
    samples: &HashMap<String, ActionBoneSample>,
    time_norm: f32,
    time_sec: f32,
    visiting: &mut HashSet<String>,
    out: &mut HashMap<String, BoneWorldSample>,
) -> Result<BoneWorldSample, MotionLoomSceneRenderError> {
    if let Some(existing) = out.get(bone_id).copied() {
        return Ok(existing);
    }
    if !visiting.insert(bone_id.to_string()) {
        return Err(MotionLoomSceneRenderError::InvalidExpression {
            expr: bone_id.to_string(),
            message: format!("Skeleton {} contains a cyclic bone hierarchy.", skeleton.id),
        });
    }

    let bone = skeleton
        .bones
        .iter()
        .find(|bone| bone.id == bone_id)
        .ok_or_else(|| MotionLoomSceneRenderError::InvalidExpression {
            expr: bone_id.to_string(),
            message: format!("Skeleton {} bone not found.", skeleton.id),
        })?;
    let parent = if let Some(parent_id) = bone.parent.as_deref() {
        if !bone_ids.contains(parent_id) {
            return Err(MotionLoomSceneRenderError::InvalidExpression {
                expr: parent_id.to_string(),
                message: format!(
                    "Skeleton {} bone {} references an unknown parent.",
                    skeleton.id, bone.id
                ),
            });
        }
        sample_skeleton_bone(
            skeleton, bone_ids, parent_id, samples, time_norm, time_sec, visiting, out,
        )?
    } else {
        BoneWorldSample {
            transform: Affine2::identity(),
            rotation: 0.0,
            scale: 1.0,
            opacity: 1.0,
        }
    };

    let sample = samples.get(&bone.id).copied().unwrap_or_default();
    let x = eval_scene_number(&bone.x, time_norm, time_sec)? + sample.x.unwrap_or(0.0);
    let y = eval_scene_number(&bone.y, time_norm, time_sec)? + sample.y.unwrap_or(0.0);
    let rotation =
        eval_scene_number(&bone.rotation, time_norm, time_sec)? + sample.rotation.unwrap_or(0.0);
    let scale = (eval_scene_number(&bone.scale, time_norm, time_sec)?
        * sample.scale.unwrap_or(1.0))
    .clamp(0.001, 64.0);
    let opacity = sample.opacity.unwrap_or(1.0).clamp(0.0, 1.0);
    let transform = parent
        .transform
        .mul(Affine2::translate(x, y))
        .mul(Affine2::rotate_deg(rotation))
        .mul(Affine2::scale(scale));
    let world = BoneWorldSample {
        transform,
        rotation: parent.rotation + rotation,
        scale: parent.scale * scale,
        opacity: parent.opacity * opacity,
    };

    visiting.remove(bone_id);
    out.insert(bone.id.clone(), world);
    Ok(world)
}

fn apply_ik_targets_to_samples(
    skeleton: &SkeletonNode,
    iks: &[ActionIkNode],
    samples: &mut HashMap<String, ActionBoneSample>,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    if iks.is_empty() {
        return Ok(());
    }

    for ik in iks {
        if ik.chain.len() > 3 {
            solve_chain_ccd_ik(skeleton, ik, samples, time_norm, time_sec)?;
        } else {
            // IK starts from the current FK pose and writes solved local rotations back into samples.
            let base_world = sample_skeleton_bones(skeleton, samples, time_norm, time_sec)?;
            solve_two_bone_ik(skeleton, &base_world, ik, samples, time_norm, time_sec)?;
        }
    }
    Ok(())
}

fn solve_chain_ccd_ik(
    skeleton: &SkeletonNode,
    ik: &ActionIkNode,
    samples: &mut HashMap<String, ActionBoneSample>,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    let chain = if ik.chain.is_empty() {
        vec![ik.root.clone(), ik.mid.clone(), ik.end.clone()]
    } else {
        ik.chain.clone()
    };
    validate_ik_chain(skeleton, &chain)?;
    let target_x = eval_scene_number(&ik.target_x, time_norm, time_sec)?;
    let target_y = eval_scene_number(&ik.target_y, time_norm, time_sec)?;
    let weight = eval_scene_number(&ik.weight, time_norm, time_sec)?.clamp(0.0, 1.0);
    if weight <= 0.0 {
        return Ok(());
    }
    let iterations = eval_scene_number(&ik.iterations, time_norm, time_sec)?
        .round()
        .clamp(1.0, 32.0) as usize;

    // CCD rotates each joint toward the target, then resamples the skeleton for the next joint.
    for _ in 0..iterations {
        for joint_id in chain.iter().take(chain.len().saturating_sub(1)).rev() {
            let bone_world = sample_skeleton_bones(skeleton, samples, time_norm, time_sec)?;
            let joint = bone_world.get(joint_id).ok_or_else(|| {
                MotionLoomSceneRenderError::InvalidExpression {
                    expr: joint_id.clone(),
                    message: format!("Skeleton {} IK joint not found.", skeleton.id),
                }
            })?;
            let end_id = chain.last().expect("validated non-empty chain");
            let end = bone_world.get(end_id).ok_or_else(|| {
                MotionLoomSceneRenderError::InvalidExpression {
                    expr: end_id.clone(),
                    message: format!("Skeleton {} IK end not found.", skeleton.id),
                }
            })?;
            let (joint_x, joint_y) = joint.transform.transform_point(0.0, 0.0);
            let (end_x, end_y) = end.transform.transform_point(0.0, 0.0);
            let current_angle = (end_y - joint_y).atan2(end_x - joint_x).to_degrees();
            let target_angle = (target_y - joint_y).atan2(target_x - joint_x).to_degrees();
            let delta = shortest_angle_delta(current_angle, target_angle);
            let entry = samples.entry(joint_id.clone()).or_default();
            entry.rotation = Some(entry.rotation.unwrap_or(0.0) + delta * weight);
        }
    }
    Ok(())
}

fn validate_ik_chain(
    skeleton: &SkeletonNode,
    chain: &[String],
) -> Result<(), MotionLoomSceneRenderError> {
    if chain.len() < 3 {
        return Err(MotionLoomSceneRenderError::InvalidExpression {
            expr: chain.join(","),
            message: "IK chain requires at least three bone ids.".to_string(),
        });
    }
    for id in chain {
        find_skeleton_bone(skeleton, id)?;
    }
    for pair in chain.windows(2) {
        let parent = &pair[0];
        let child = find_skeleton_bone(skeleton, &pair[1])?;
        if child.parent.as_deref() != Some(parent.as_str()) {
            return Err(MotionLoomSceneRenderError::InvalidExpression {
                expr: chain.join(","),
                message: "IK chain ids must be direct parent-child bones.".to_string(),
            });
        }
    }
    Ok(())
}

fn shortest_angle_delta(from_deg: f32, to_deg: f32) -> f32 {
    let mut delta = to_deg - from_deg;
    while delta > 180.0 {
        delta -= 360.0;
    }
    while delta < -180.0 {
        delta += 360.0;
    }
    delta
}

fn solve_two_bone_ik(
    skeleton: &SkeletonNode,
    base_world: &HashMap<String, BoneWorldSample>,
    ik: &ActionIkNode,
    samples: &mut HashMap<String, ActionBoneSample>,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    let root = find_skeleton_bone(skeleton, &ik.root)?;
    let mid = find_skeleton_bone(skeleton, &ik.mid)?;
    let end = find_skeleton_bone(skeleton, &ik.end)?;
    if mid.parent.as_deref() != Some(root.id.as_str())
        || end.parent.as_deref() != Some(mid.id.as_str())
    {
        return Err(MotionLoomSceneRenderError::InvalidExpression {
            expr: format!("{}>{}>{}", root.id, mid.id, end.id),
            message: "IK requires a direct root -> mid -> end bone chain.".to_string(),
        });
    }

    let parent = root
        .parent
        .as_deref()
        .and_then(|parent_id| base_world.get(parent_id).copied())
        .unwrap_or(BoneWorldSample {
            transform: Affine2::identity(),
            rotation: 0.0,
            scale: 1.0,
            opacity: 1.0,
        });
    let parent_inverse = parent.transform.inverse().ok_or_else(|| {
        MotionLoomSceneRenderError::InvalidExpression {
            expr: ik.root.clone(),
            message: "IK parent transform is not invertible.".to_string(),
        }
    })?;

    let target_world_x = eval_scene_number(&ik.target_x, time_norm, time_sec)?;
    let target_world_y = eval_scene_number(&ik.target_y, time_norm, time_sec)?;
    let (target_x, target_y) = parent_inverse.transform_point(target_world_x, target_world_y);

    let root_sample = samples.get(&root.id).copied().unwrap_or_default();
    let root_x = eval_scene_number(&root.x, time_norm, time_sec)? + root_sample.x.unwrap_or(0.0);
    let root_y = eval_scene_number(&root.y, time_norm, time_sec)? + root_sample.y.unwrap_or(0.0);
    let mid_x = eval_scene_number(&mid.x, time_norm, time_sec)?;
    let mid_y = eval_scene_number(&mid.y, time_norm, time_sec)?;
    let end_x = eval_scene_number(&end.x, time_norm, time_sec)?;
    let end_y = eval_scene_number(&end.y, time_norm, time_sec)?;

    let first_len = mid_x.hypot(mid_y).max(0.0001);
    let second_len = end_x.hypot(end_y).max(0.0001);
    let base_first_angle = mid_y.atan2(mid_x).to_degrees();
    let base_second_angle = end_y.atan2(end_x).to_degrees();
    let dx = target_x - root_x;
    let dy = target_y - root_y;
    let distance = dx.hypot(dy).clamp(0.0001, first_len + second_len - 0.0001);
    let target_angle = dy.atan2(dx).to_degrees();
    let bend_sign = if eval_scene_number(&ik.bend, time_norm, time_sec)? < 0.0 {
        -1.0
    } else {
        1.0
    };
    let weight = eval_scene_number(&ik.weight, time_norm, time_sec)?.clamp(0.0, 1.0);
    if weight <= 0.0 {
        return Ok(());
    }

    let root_offset = (((first_len * first_len) + (distance * distance)
        - (second_len * second_len))
        / (2.0 * first_len * distance))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    let elbow_internal = (((first_len * first_len) + (second_len * second_len)
        - (distance * distance))
        / (2.0 * first_len * second_len))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    let root_rest_rotation = eval_scene_number(&root.rotation, time_norm, time_sec)?;
    let mid_rest_rotation = eval_scene_number(&mid.rotation, time_norm, time_sec)?;
    let solved_root_delta =
        target_angle - bend_sign * root_offset - base_first_angle - root_rest_rotation;
    let solved_mid_delta =
        bend_sign * (180.0 - elbow_internal) - base_second_angle - mid_rest_rotation;

    blend_rotation_sample(samples, &root.id, solved_root_delta, weight);
    blend_rotation_sample(samples, &mid.id, solved_mid_delta, weight);
    Ok(())
}

fn find_skeleton_bone<'a>(
    skeleton: &'a SkeletonNode,
    bone_id: &str,
) -> Result<&'a SkeletonBoneNode, MotionLoomSceneRenderError> {
    skeleton
        .bones
        .iter()
        .find(|bone| bone.id == bone_id)
        .ok_or_else(|| MotionLoomSceneRenderError::InvalidExpression {
            expr: bone_id.to_string(),
            message: format!("Skeleton {} IK bone not found.", skeleton.id),
        })
}

fn blend_rotation_sample(
    samples: &mut HashMap<String, ActionBoneSample>,
    bone_id: &str,
    solved_rotation: f32,
    weight: f32,
) {
    let entry = samples.entry(bone_id.to_string()).or_default();
    let current = entry.rotation.unwrap_or(0.0);
    entry.rotation = Some(current + (solved_rotation - current) * weight);
}

fn apply_skeleton_action_to_character_children(
    nodes: &mut [SceneNode],
    bone_world: &HashMap<String, BoneWorldSample>,
    samples: &HashMap<String, ActionBoneSample>,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    for node in nodes {
        match node {
            SceneNode::Timeline(timeline) => apply_skeleton_action_to_character_children(
                &mut timeline.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Track(track) => apply_skeleton_action_to_character_children(
                &mut track.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Sequence(sequence) => apply_skeleton_action_to_character_children(
                &mut sequence.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Chain(chain) => apply_skeleton_action_to_character_children(
                &mut chain.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Layer(layer) => apply_skeleton_action_to_character_children(
                &mut layer.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Part(part) => {
                if let Some(attach_to) = part.attach_to.as_deref()
                    && let Some(bone) = bone_world.get(attach_to).copied()
                {
                    apply_bone_world_to_part(part, bone, time_norm, time_sec)?;
                } else if let Some(id) = part.id.as_deref()
                    && let Some(sample) = samples.get(id)
                {
                    apply_action_sample_to_part(part, *sample, time_norm, time_sec)?;
                }
                apply_skeleton_action_to_character_children(
                    &mut part.children,
                    bone_world,
                    samples,
                    time_norm,
                    time_sec,
                )?;
            }
            SceneNode::Group(group) => apply_skeleton_action_to_character_children(
                &mut group.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Puppet(puppet) => apply_skeleton_action_to_character_children(
                &mut puppet.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Repeat(repeat) => apply_skeleton_action_to_character_children(
                &mut repeat.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Mask(mask) => apply_skeleton_action_to_character_children(
                &mut mask.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Precompose(precompose) => apply_skeleton_action_to_character_children(
                &mut precompose.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Camera(camera) => apply_skeleton_action_to_character_children(
                &mut camera.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Character(character) => apply_skeleton_action_to_character_children(
                &mut character.children,
                bone_world,
                samples,
                time_norm,
                time_sec,
            )?,
            _ => {}
        }
    }
    Ok(())
}

fn apply_bone_world_to_part(
    part: &mut PartNode,
    bone: BoneWorldSample,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    let local_x = eval_scene_number(&part.x, time_norm, time_sec)?;
    let local_y = eval_scene_number(&part.y, time_norm, time_sec)?;
    let local_rotation = eval_scene_number(&part.rotation, time_norm, time_sec)?;
    let local_scale = eval_scene_number(&part.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
    let local_opacity = eval_scene_number(&part.opacity, time_norm, time_sec)?.clamp(0.0, 1.0);
    let (world_x, world_y) = bone.transform.transform_point(local_x, local_y);

    part.x = format_scene_number(world_x);
    part.y = format_scene_number(world_y);
    part.rotation = format_scene_number(bone.rotation + local_rotation);
    part.scale = format_scene_number((bone.scale * local_scale).clamp(0.001, 64.0));
    part.opacity = format_scene_number((bone.opacity * local_opacity).clamp(0.0, 1.0));
    Ok(())
}

fn apply_action_to_character_children(
    nodes: &mut [SceneNode],
    samples: &HashMap<String, ActionBoneSample>,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    for node in nodes {
        match node {
            SceneNode::Timeline(timeline) => apply_action_to_character_children(
                &mut timeline.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Track(track) => apply_action_to_character_children(
                &mut track.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Sequence(sequence) => apply_action_to_character_children(
                &mut sequence.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Chain(chain) => apply_action_to_character_children(
                &mut chain.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Layer(layer) => apply_action_to_character_children(
                &mut layer.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Part(part) => {
                if let Some(id) = part.id.as_deref()
                    && let Some(sample) = samples.get(id)
                {
                    apply_action_sample_to_part(part, *sample, time_norm, time_sec)?;
                }
                apply_action_to_character_children(
                    &mut part.children,
                    samples,
                    time_norm,
                    time_sec,
                )?;
            }
            SceneNode::Group(group) => apply_action_to_character_children(
                &mut group.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Puppet(puppet) => apply_action_to_character_children(
                &mut puppet.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Repeat(repeat) => apply_action_to_character_children(
                &mut repeat.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Mask(mask) => apply_action_to_character_children(
                &mut mask.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Precompose(precompose) => apply_action_to_character_children(
                &mut precompose.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Camera(camera) => apply_action_to_character_children(
                &mut camera.children,
                samples,
                time_norm,
                time_sec,
            )?,
            SceneNode::Character(character) => apply_action_to_character_children(
                &mut character.children,
                samples,
                time_norm,
                time_sec,
            )?,
            _ => {}
        }
    }
    Ok(())
}

fn apply_action_sample_to_part(
    part: &mut PartNode,
    sample: ActionBoneSample,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    if let Some(x) = sample.x {
        part.x = format_scene_number(eval_scene_number(&part.x, time_norm, time_sec)? + x);
    }
    if let Some(y) = sample.y {
        part.y = format_scene_number(eval_scene_number(&part.y, time_norm, time_sec)? + y);
    }
    if let Some(rotation) = sample.rotation {
        part.rotation =
            format_scene_number(eval_scene_number(&part.rotation, time_norm, time_sec)? + rotation);
    }
    if let Some(scale) = sample.scale {
        part.scale = format_scene_number(
            eval_scene_number(&part.scale, time_norm, time_sec)? * scale.max(0.001),
        );
    }
    if let Some(opacity) = sample.opacity {
        part.opacity = format_scene_number(
            eval_scene_number(&part.opacity, time_norm, time_sec)? * opacity.clamp(0.0, 1.0),
        );
    }
    Ok(())
}
