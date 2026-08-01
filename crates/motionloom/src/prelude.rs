//! Convenience prelude for common MotionLoom integrations.

pub use crate::{
    GraphParseError, GraphScript, MotionLoomError, RenderPassDag, SceneRenderProfile,
    SceneRenderer, compile_render_pass_dag, compile_runtime_program, inspect_gpu_compatibility,
    parse_graph_script, parse_process_graph_script, process_effect_for_id, process_effects,
    render_scene_graph_frame, render_scene_graph_frame_with_cpu_inputs,
};
