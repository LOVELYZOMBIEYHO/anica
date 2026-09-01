// =========================================
// =========================================
// crates/motionloom/src/world/render/pose_diagnostics.rs

//! Read-only evaluation of authored Action poses through the renderer's evaluator.
//! Matrices are column-major, model-global, before actor placement and scene contacts.

use super::*;
use crate::rig_diagnostics::{
    ActionDriver, ActionExecutionTrace, AppliedActionTrace, AxisEffectiveness, BoneDriver,
    BoneEvaluation, BonePoseStage, BoneStageTransform, ContactEvaluation,
    RIG_DIAGNOSTICS_SCHEMA_VERSION, RigActionProvenance, RigAssetProvenance, RigDiagnostic,
    RigDiagnosticSeverity, RigEvaluationCapabilities, RigEvaluationReport, RigEvaluationRequest,
    RigProfileProvenance, RigProvenance, RigReportDetail, RigSamplePoint, RigUnits,
    fingerprint_serializable, matrix_position, matrix_rotation_quaternion,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Error)]
pub enum PoseDiagnosticError {
    #[error(transparent)]
    Render(#[from] WorldRenderError),
    #[error("invalid diagnostic request: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointPoseDiagnostic {
    pub node_index: usize,
    pub node_name: String,
    pub parent_index: Option<usize>,
    pub canonical_bone: Option<String>,
    pub model_global_matrix: [f32; 16],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorPoseDiagnostic {
    pub actor_id: String,
    pub frame: u32,
    pub fps: f32,
    pub time_sec: f32,
    pub stage: String,
    pub joints: Vec<JointPoseDiagnostic>,
}

/// Evaluate existing Action/IK and embedded clips without GPU, I/O or mutations.
/// External AnimationAssets and cross-actor constraints require a full scene and
/// are rejected, never silently omitted. This does not certify scene collision.
pub fn diagnose_world_actor_pose(
    graph: &WorldGraph,
    mesh: &GlbMeshData,
    actor_id: &str,
    time: WorldTime,
) -> Result<ActorPoseDiagnostic, PoseDiagnosticError> {
    if !time.fps.is_finite() || time.fps <= 0. {
        return Err(PoseDiagnosticError::Invalid(
            "fps must be finite and positive".into(),
        ));
    }
    if !graph.constraints.is_empty() || !graph.animation_assets.is_empty() {
        return Err(PoseDiagnosticError::Invalid(
            "external assets/scene constraints are outside model-pose diagnostics".into(),
        ));
    }
    let world = graph
        .presented_world()
        .ok_or_else(|| WorldRenderError::MissingWorld(graph.present.from.clone()))?;
    let actor = world
        .actors
        .iter()
        .find(|a| a.id == actor_id)
        .ok_or_else(|| WorldRenderError::MissingActor(actor_id.into()))?;
    let overrides = actor_bone_overrides_for_mesh(graph, actor, Some(mesh), time)?;
    let matrices = actor_global_node_matrices(graph, actor, mesh, time, &overrides, None)?;
    let profile = actor_model_profile(graph, actor);
    let mut canonical = canonical_humanoid_editor_bones()
        .iter()
        .filter_map(|bone| {
            target_node_for_canonical_bone(mesh, profile, bone).map(|i| (i, (*bone).to_string()))
        })
        .collect::<HashMap<_, _>>();
    if let Some(retarget) = actor
        .retarget
        .as_ref()
        .and_then(|id| graph.retargets.iter().find(|r| &r.id == id))
    {
        canonical.clear();
        for map in &retarget.maps {
            if let Some(i) = mesh
                .nodes
                .iter()
                .position(|n| n.name.as_deref() == Some(map.from.as_str()))
            {
                canonical.insert(i, map.to.clone());
            }
        }
    }
    if matrices.iter().flatten().any(|v| !v.is_finite()) {
        return Err(PoseDiagnosticError::Invalid(
            "evaluation produced non-finite matrices".into(),
        ));
    }
    Ok(ActorPoseDiagnostic {
        actor_id: actor_id.into(),
        frame: time.frame,
        fps: time.fps,
        time_sec: time.time_sec(),
        stage: "model_global_before_scene_contacts".into(),
        joints: mesh
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| JointPoseDiagnostic {
                node_index: i,
                node_name: node.name.clone().unwrap_or_default(),
                parent_index: node.parent,
                canonical_bone: canonical.get(&i).cloned(),
                model_global_matrix: matrices[i],
            })
            .collect(),
    })
}

/// Evaluate one actor through the production World pose evaluator and return a
/// stable, versioned report. Scene-level contact stages are explicitly marked
/// unavailable instead of being inferred from model-space matrices.
pub fn evaluate_world_actor_rig(
    graph: &WorldGraph,
    mesh: &GlbMeshData,
    request: &RigEvaluationRequest,
) -> Result<RigEvaluationReport, PoseDiagnosticError> {
    let world = graph
        .presented_world()
        .ok_or_else(|| WorldRenderError::MissingWorld(graph.present.from.clone()))?;
    let actor = world
        .actors
        .iter()
        .find(|actor| actor.id == request.actor_id)
        .ok_or_else(|| WorldRenderError::MissingActor(request.actor_id.clone()))?;
    let time = diagnostic_sample_time(graph, actor, &request.sample)?;
    let pose = diagnose_world_actor_pose(graph, mesh, &request.actor_id, time)?;
    let rest_matrices = global_node_matrices(mesh, &HashMap::new());
    let profile = actor_model_profile(graph, actor);
    let action_execution = diagnostic_action_trace(graph, actor, time)?;
    let selected_action = action_execution.active_actions.iter().find_map(|trace| {
        graph
            .actions
            .iter()
            .find(|action| action.id == trace.action_id)
    });
    let baked_reference = selected_action.is_some_and(action_uses_baked_humanoid_reference);
    let active_additive = action_execution
        .active_actions
        .iter()
        .any(|trace| trace.mode.eq_ignore_ascii_case("additive"));
    let driver = if active_additive {
        BoneDriver::AdditiveAction
    } else if graph.animation_assets.iter().any(|asset| {
        action_execution
            .active_actions
            .iter()
            .any(|trace| trace.action_id == asset.id)
    }) {
        BoneDriver::ExternalClipRetarget
    } else if baked_reference {
        BoneDriver::BakedReferenceRetarget
    } else if selected_action.is_some() {
        BoneDriver::LegacyBoneAxis
    } else if actor.play.is_some() || !actor.plays.is_empty() {
        BoneDriver::EmbeddedClip
    } else {
        BoneDriver::BindPose
    };
    let canonical_by_node = pose
        .joints
        .iter()
        .filter_map(|joint| {
            joint
                .canonical_bone
                .as_ref()
                .map(|bone| (joint.node_index, bone.clone()))
        })
        .collect::<HashMap<_, _>>();
    let include_bone = |bone: &Option<String>| match request.detail {
        RigReportDetail::Summary => false,
        RigReportDetail::Body => bone.as_deref().is_some_and(|bone| {
            !bone.contains("thumb")
                && !bone.contains("index")
                && !bone.contains("middle")
                && !bone.contains("ring")
                && !bone.contains("pinky")
        }),
        RigReportDetail::Full => bone.is_some(),
    };
    let bones = pose
        .joints
        .iter()
        .filter(|joint| include_bone(&joint.canonical_bone))
        .filter_map(|joint| {
            let canonical = joint.canonical_bone.clone()?;
            let rest = rest_matrices.get(joint.node_index).copied()?;
            let mut stages = vec![BoneStageTransform {
                stage: BonePoseStage::ModelRest,
                space: "modelGlobal".into(),
                position: Some(matrix_position(rest)),
                rotation_quaternion: Some(matrix_rotation_quaternion(rest)),
                matrix: request.include_matrices.then_some(rest),
                screen: None,
            }];
            stages.push(BoneStageTransform {
                stage: if selected_action.is_some() {
                    BonePoseStage::Retargeted
                } else {
                    BonePoseStage::FinalScene
                },
                space: "modelGlobalBeforeSceneContacts".into(),
                position: Some(matrix_position(joint.model_global_matrix)),
                rotation_quaternion: Some(matrix_rotation_quaternion(joint.model_global_matrix)),
                matrix: request
                    .include_matrices
                    .then_some(joint.model_global_matrix),
                screen: None,
            });
            let axis = diagnostic_axis_effectiveness(profile, &canonical, driver);
            Some(BoneEvaluation {
                canonical_bone: canonical,
                target_node: Some(joint.node_name.clone()),
                node_index: Some(joint.node_index),
                parent_bone: joint
                    .parent_index
                    .and_then(|parent| canonical_by_node.get(&parent).cloned()),
                mapped: true,
                driver,
                stages,
                axis,
                diagnostics: Vec::new(),
            })
        })
        .collect::<Vec<_>>();
    let mapping_count = profile
        .and_then(|profile| profile.retarget.as_ref())
        .map_or(0, |retarget| retarget.maps.len());
    let axis_count = profile
        .and_then(|profile| profile.bone_axis_map.as_ref())
        .map_or(0, |axis_map| axis_map.axes.len());
    let action = selected_action;
    let mut diagnostics = Vec::new();
    if !graph.constraints.is_empty() {
        diagnostics.push(RigDiagnostic {
            severity: RigDiagnosticSeverity::Warning,
            code: "SCENE_CONSTRAINT_STAGE_UNAVAILABLE".into(),
            message: "World diagnostics cannot certify cross-actor Scene constraints.".into(),
            bone: None,
            evidence: Vec::new(),
        });
    }
    diagnostics.push(RigDiagnostic {
        severity: RigDiagnosticSeverity::Info,
        code: "POST_CONTACT_STAGE_UNAVAILABLE".into(),
        message: "This CPU report ends before Scene ground/contact correction.".into(),
        bone: None,
        evidence: vec!["Use SceneRenderer rig evaluation for final scene/contact stages.".into()],
    });
    Ok(RigEvaluationReport {
        schema_version: RIG_DIAGNOSTICS_SCHEMA_VERSION.into(),
        engine_version: env!("CARGO_PKG_VERSION").into(),
        units: RigUnits::default(),
        sample: request.sample.clone(),
        frame: pose.frame,
        fps: pose.fps,
        time_sec: pose.time_sec,
        actor_id: request.actor_id.clone(),
        body_height: Some((mesh.bounds_max[1] - mesh.bounds_min[1]).abs()),
        provenance: RigProvenance {
            document: None,
            actor_id: request.actor_id.clone(),
            model_asset: RigAssetProvenance {
                id: Some(actor.model.clone()),
                resolved_source: Some(mesh.path.display().to_string()),
                sha256: None,
                skin_joint_count: mesh.skin.as_ref().map(|skin| skin.joints.len()),
                animation_clip_count: Some(mesh.animations.len()),
            },
            profile: RigProfileProvenance {
                id: profile.map(|profile| profile.id.clone()),
                preset: profile.map(|profile| profile.preset.clone()),
                fingerprint: profile.and_then(fingerprint_serializable),
                mapping_count,
                axis_count,
            },
            action: RigActionProvenance {
                id: action.map(|action| action.id.clone()),
                fingerprint: action.and_then(fingerprint_serializable),
                duration_sec: action.map(|action| action.duration_ms as f32 / 1000.0),
                pose_count: action.map_or(0, |action| action.poses.len()),
                contact_count: 0,
            },
        },
        capabilities: RigEvaluationCapabilities {
            model_global_pose: true,
            action_execution: true,
            retarget_driver: true,
            axis_effectiveness: true,
            post_constraints: graph.constraints.is_empty(),
            post_contact: false,
            screen_projection: false,
        },
        action_execution,
        contact_evaluation: ContactEvaluation {
            available: false,
            ..ContactEvaluation::default()
        },
        bones,
        diagnostics,
    })
}

fn diagnostic_sample_time(
    graph: &WorldGraph,
    actor: &WorldActor,
    sample: &RigSamplePoint,
) -> Result<WorldTime, PoseDiagnosticError> {
    let fps = graph.fps.max(1.0);
    let frame = match sample {
        RigSamplePoint::Frame { frame } => *frame,
        RigSamplePoint::Time { time_sec } => {
            if !time_sec.is_finite() || *time_sec < 0.0 {
                return Err(PoseDiagnosticError::Invalid(
                    "timeSec must be finite and non-negative".into(),
                ));
            }
            (*time_sec * fps).round() as u32
        }
        RigSamplePoint::ActionPhase { action_id, phase } => {
            if !phase.is_finite() || !(0.0..=1.0).contains(phase) {
                return Err(PoseDiagnosticError::Invalid(
                    "Action phase must be within 0..=1".into(),
                ));
            }
            let apply = graph
                .apply_actions
                .iter()
                .find(|apply| apply.target == actor.id && apply.action == *action_id)
                .ok_or_else(|| {
                    PoseDiagnosticError::Invalid(format!(
                        "Action '{action_id}' is not applied to '{}'",
                        actor.id
                    ))
                })?;
            let action = graph
                .actions
                .iter()
                .find(|action| action.id == *action_id)
                .ok_or_else(|| {
                    PoseDiagnosticError::Invalid(format!("Action '{action_id}' was not found"))
                })?;
            let speed = apply.speed.parse::<f32>().unwrap_or(1.0).max(0.0001);
            let time_sec = apply.at_ms as f32 / 1000.0
                + phase.clamp(0.0, 1.0) * action.duration_ms as f32 / 1000.0 / speed;
            (time_sec * fps).round() as u32
        }
    };
    Ok(WorldTime {
        frame,
        fps,
        duration_ms: graph.duration_ms,
    })
}

pub(crate) fn diagnostic_action_trace(
    graph: &WorldGraph,
    actor: &WorldActor,
    time: WorldTime,
) -> Result<ActionExecutionTrace, PoseDiagnosticError> {
    let mut active_actions = Vec::new();
    let mut inactive_actions = Vec::new();
    for apply in graph
        .apply_actions
        .iter()
        .filter(|apply| apply.target == actor.id)
    {
        let action = graph
            .actions
            .iter()
            .find(|action| action.id == apply.action);
        let elapsed = time.time_sec() - apply.at_ms as f32 / 1000.0;
        let window = apply.duration_ms.map(|duration| duration as f32 / 1000.0);
        let active = elapsed >= 0.0 && window.is_none_or(|duration| elapsed <= duration);
        let speed = eval_number(&apply.speed, 1.0, time)?.max(0.0);
        let local_time = action.and_then(|action| {
            active
                .then(|| action_local_time_sec(action, apply.at_ms, apply.r#loop, speed, time))
                .flatten()
        });
        let normalized_phase = action
            .zip(local_time)
            .map(|(action, local)| local / (action.duration_ms as f32 / 1000.0).max(f32::EPSILON));
        let blend_weight = if let (true, Some(action)) = (active, action) {
            action_blend_envelope(action, apply, local_time.unwrap_or(0.0), speed, time)?
                * eval_number(&apply.weight, 1.0, time)?.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let driver = if apply.mode.eq_ignore_ascii_case("additive") {
            ActionDriver::AdditiveAction
        } else if graph
            .animation_assets
            .iter()
            .any(|asset| asset.id == apply.action)
        {
            ActionDriver::ExternalGlbAction
        } else if action.is_some_and(action_uses_baked_humanoid_reference) {
            ActionDriver::BakedHumanoidReference
        } else {
            ActionDriver::LegacySemanticAxes
        };
        let trace = AppliedActionTrace {
            action_id: apply.action.clone(),
            target: apply.target.clone(),
            authored_start_sec: apply.at_ms as f32 / 1000.0,
            authored_duration_sec: window,
            active,
            inactive_reason: (!active).then(|| {
                if elapsed < 0.0 {
                    "beforeAuthoredWindow"
                } else {
                    "outsideAuthoredWindow"
                }
                .into()
            }),
            looped: apply.r#loop,
            local_time_sec: local_time,
            normalized_phase,
            speed,
            blend_weight,
            mode: apply.mode.clone(),
            mask: apply.mask.clone(),
            root_motion: apply.root_motion.clone(),
            driver,
        };
        if active {
            active_actions.push(trace)
        } else {
            inactive_actions.push(trace)
        }
    }
    let selected_controller_action = active_actions.first().map(|trace| trace.action_id.clone());
    Ok(ActionExecutionTrace {
        selected_controller_action,
        active_actions,
        inactive_actions,
    })
}

pub(crate) fn diagnostic_axis_effectiveness(
    profile: Option<&WorldModelProfile>,
    bone: &str,
    driver: BoneDriver,
) -> AxisEffectiveness {
    let axis = profile
        .and_then(|profile| profile.bone_axis_map.as_ref())
        .and_then(|axis_map| axis_map.axes.iter().find(|axis| axis.bone == bone));
    let Some(axis) = axis else {
        return AxisEffectiveness::default();
    };
    let channels = [
        ("forward", axis.forward.as_ref()),
        ("side", axis.side.as_ref()),
        ("twist", axis.twist.as_ref()),
        ("bend", axis.bend.as_ref()),
        ("turn", axis.turn.as_ref()),
        ("restForward", axis.rest_forward.as_ref()),
        ("restSide", axis.rest_side.as_ref()),
        ("restTwist", axis.rest_twist.as_ref()),
        ("restBend", axis.rest_bend.as_ref()),
        ("restTurn", axis.rest_turn.as_ref()),
    ];
    let semantic_effective = driver == BoneDriver::LegacyBoneAxis;
    let rest_effective = matches!(
        driver,
        BoneDriver::LegacyBoneAxis | BoneDriver::BakedReferenceRetarget
    );
    let mut declared = BTreeMap::new();
    let mut effective = BTreeMap::new();
    for (name, value) in channels {
        if let Some(value) = value {
            declared.insert(name.into(), value.clone());
            effective.insert(
                name.into(),
                if name.starts_with("rest") {
                    rest_effective
                } else {
                    semantic_effective
                },
            );
        }
    }
    AxisEffectiveness {
        declared,
        effective,
        applied_at_stage: (semantic_effective || rest_effective)
            .then_some(BonePoseStage::ProfileCalibratedRest),
        bypassed_by: (!semantic_effective && !declared_is_rest_only(&axis))
            .then(|| format!("{driver:?}")),
    }
}

fn declared_is_rest_only(axis: &WorldBoneAxis) -> bool {
    axis.forward.is_none()
        && axis.side.is_none()
        && axis.twist.is_none()
        && axis.bend.is_none()
        && axis.turn.is_none()
}

/// Build a stable report from matrices already produced by the renderer. This
/// is the exact path used by Scene diagnostics after constraints are solved.
pub(crate) fn runtime_world_actor_rig_from_matrices(
    graph: &WorldGraph,
    mesh: &GlbMeshData,
    actor: &WorldActor,
    time: WorldTime,
    retargeted_matrices: Option<&[[f32; 16]]>,
    matrices: &[[f32; 16]],
    include_matrices: bool,
) -> Result<RigEvaluationReport, PoseDiagnosticError> {
    let profile = actor_model_profile(graph, actor);
    let rest_matrices = global_node_matrices(mesh, &HashMap::new());
    let action_execution = diagnostic_action_trace(graph, actor, time)?;
    let selected_action = action_execution.active_actions.iter().find_map(|trace| {
        graph
            .actions
            .iter()
            .find(|action| action.id == trace.action_id)
    });
    let baked_reference = selected_action.is_some_and(action_uses_baked_humanoid_reference);
    let driver = if action_execution
        .active_actions
        .iter()
        .any(|trace| trace.mode.eq_ignore_ascii_case("additive"))
    {
        BoneDriver::AdditiveAction
    } else if graph.animation_assets.iter().any(|asset| {
        action_execution
            .active_actions
            .iter()
            .any(|trace| trace.action_id == asset.id)
    }) {
        BoneDriver::ExternalClipRetarget
    } else if baked_reference {
        BoneDriver::BakedReferenceRetarget
    } else if selected_action.is_some() {
        BoneDriver::LegacyBoneAxis
    } else if actor.play.is_some() || !actor.plays.is_empty() {
        BoneDriver::EmbeddedClip
    } else {
        BoneDriver::BindPose
    };
    let canonical = canonical_humanoid_editor_bones()
        .iter()
        .filter_map(|bone| {
            target_node_for_canonical_bone(mesh, profile, bone)
                .map(|index| (index, (*bone).to_string()))
        })
        .collect::<HashMap<_, _>>();
    let mut bones = Vec::new();
    for (index, bone) in &canonical {
        let Some(node) = mesh.nodes.get(*index) else {
            continue;
        };
        let (Some(rest), Some(final_matrix)) = (rest_matrices.get(*index), matrices.get(*index))
        else {
            continue;
        };
        let mut stages = vec![BoneStageTransform {
            stage: BonePoseStage::ModelRest,
            space: "modelGlobal".into(),
            position: Some(matrix_position(*rest)),
            rotation_quaternion: Some(matrix_rotation_quaternion(*rest)),
            matrix: include_matrices.then_some(*rest),
            screen: None,
        }];
        if let Some(retargeted) = retargeted_matrices.and_then(|matrices| matrices.get(*index)) {
            stages.push(BoneStageTransform {
                stage: BonePoseStage::Retargeted,
                space: "modelGlobalBeforeConstraints".into(),
                position: Some(matrix_position(*retargeted)),
                rotation_quaternion: Some(matrix_rotation_quaternion(*retargeted)),
                matrix: include_matrices.then_some(*retargeted),
                screen: None,
            });
        }
        stages.push(BoneStageTransform {
            stage: BonePoseStage::PostConstraint,
            space: "modelGlobal".into(),
            position: Some(matrix_position(*final_matrix)),
            rotation_quaternion: Some(matrix_rotation_quaternion(*final_matrix)),
            matrix: include_matrices.then_some(*final_matrix),
            screen: None,
        });
        bones.push(BoneEvaluation {
            canonical_bone: bone.clone(),
            target_node: node.name.clone(),
            node_index: Some(*index),
            parent_bone: node
                .parent
                .and_then(|parent| canonical.get(&parent).cloned()),
            mapped: true,
            driver,
            stages,
            axis: diagnostic_axis_effectiveness(profile, bone, driver),
            diagnostics: Vec::new(),
        });
    }
    bones.sort_by(|left, right| left.canonical_bone.cmp(&right.canonical_bone));
    let mapping_count = profile
        .and_then(|profile| profile.retarget.as_ref())
        .map_or(0, |retarget| retarget.maps.len());
    let axis_count = profile
        .and_then(|profile| profile.bone_axis_map.as_ref())
        .map_or(0, |axis_map| axis_map.axes.len());
    Ok(RigEvaluationReport {
        schema_version: RIG_DIAGNOSTICS_SCHEMA_VERSION.into(),
        engine_version: env!("CARGO_PKG_VERSION").into(),
        units: RigUnits::default(),
        sample: RigSamplePoint::Frame { frame: time.frame },
        frame: time.frame,
        fps: time.fps,
        time_sec: time.time_sec(),
        actor_id: actor.id.clone(),
        body_height: Some((mesh.bounds_max[1] - mesh.bounds_min[1]).abs()),
        provenance: RigProvenance {
            document: None,
            actor_id: actor.id.clone(),
            model_asset: RigAssetProvenance {
                id: Some(actor.model.clone()),
                resolved_source: Some(mesh.path.display().to_string()),
                sha256: std::fs::read(&mesh.path)
                    .ok()
                    .map(|bytes| crate::rig_diagnostics::sha256_hex(&bytes)),
                skin_joint_count: mesh.skin.as_ref().map(|skin| skin.joints.len()),
                animation_clip_count: Some(mesh.animations.len()),
            },
            profile: RigProfileProvenance {
                id: profile.map(|profile| profile.id.clone()),
                preset: profile.map(|profile| profile.preset.clone()),
                fingerprint: profile.and_then(fingerprint_serializable),
                mapping_count,
                axis_count,
            },
            action: RigActionProvenance {
                id: selected_action.map(|action| action.id.clone()),
                fingerprint: selected_action.and_then(fingerprint_serializable),
                duration_sec: selected_action.map(|action| action.duration_ms as f32 / 1000.0),
                pose_count: selected_action.map_or(0, |action| action.poses.len()),
                contact_count: 0,
            },
        },
        capabilities: RigEvaluationCapabilities {
            model_global_pose: true,
            action_execution: true,
            retarget_driver: true,
            axis_effectiveness: true,
            post_constraints: true,
            post_contact: false,
            screen_projection: false,
        },
        action_execution,
        contact_evaluation: ContactEvaluation::default(),
        bones,
        diagnostics: Vec::new(),
    })
}
