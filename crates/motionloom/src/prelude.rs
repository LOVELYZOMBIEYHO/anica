//! Convenience prelude for common MotionLoom integrations.

pub use crate::{
    GraphParseError, GraphScript, MotionLoomAuthoringReport, MotionLoomError, RenderPassDag,
    RigComparisonOptions, RigComparisonReport, RigEvaluationReport, SceneRenderProfile,
    SceneRenderer, ShotValidationOptions, ShotValidationReport, analyze_motionloom_script,
    compare_humanoid_poses, compile_render_pass_dag, compile_runtime_program,
    inspect_gpu_compatibility, parse_graph_script, parse_process_graph_script,
    process_effect_for_id, process_effects, propose_rig_calibration, render_scene_graph_frame,
    render_scene_graph_frame_with_cpu_inputs,
};
