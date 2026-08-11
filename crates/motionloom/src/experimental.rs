//! Experimental and advanced MotionLoom APIs.
//!
//! These APIs are public for tooling and Anica integration, but they are not the
//! recommended starting point for general MotionLoom users.

pub use crate::{
    CharacterDesignGpuViewport, CharacterDesignViewportFrame, GlbLoadError, GlbMeshData,
    GlbMetadata, GlbNodeData, MotionLoomWorldError, WorldAction, WorldActionBone, WorldActionPose,
    WorldActor, WorldApplyAction, WorldAssetError, WorldBackground, WorldBackgroundFit,
    WorldBoneAxis, WorldBoneAxisMap, WorldCamera, WorldCameraControl, WorldCameraMode,
    WorldCameraProjection, WorldError, WorldFrameRenderer, WorldGpuDiagnostics, WorldGraph,
    WorldMaterial, WorldMaterialStyle, WorldModelProfile, WorldNode, WorldParseError,
    WorldPathStyle, WorldPlay, WorldPresent, WorldProfileRetarget, WorldRenderError,
    WorldRenderProgress, WorldRetarget, WorldRetargetMap, WorldSpritePlayback, WorldTime,
    diagnose_world_glb_gpu_plan, diagnose_world_graph_actor_gpu_frame, is_world_graph_script,
    load_glb_mesh_data, load_glb_metadata, parse_glb_mesh_data, parse_glb_metadata,
    parse_world_graph_script, render_world_frame, render_world_graph_to_png_sequence_with_progress,
    render_world_graph_to_png_sequence_with_progress_and_cancel,
    render_world_graph_to_video_with_progress,
};

pub use crate::scene::model::{Scene3DNode, SceneModel3DNode, SceneNode};
pub use crate::world::gltf_loader::{load_glb_mesh_data_from_bytes, load_glb_metadata_from_bytes};

pub mod clip {
    pub use crate::common::backend::OutputFormat;
    pub use crate::process::adapters::clip::export_adapter::{
        ExportSample, build_export_zoom_sequence,
    };
    pub use crate::process::adapters::clip::preview_adapter::{PreviewSample, sample_preview_zoom};
    pub use crate::process::graph::MotionGraph;
}

pub mod effects {
    pub use crate::process::effect::{
        LayerColorBlurEffects, PerClipColorBlurEffects, combine_clip_with_layer,
    };
}

pub mod keyframe {
    pub use crate::common::keyframe::{ScalarKeyframe, index_at, sample_linear, set_or_insert};
}

pub mod editor {
    pub use crate::scene::editor_keyframes::{
        AnimationKeyframeEditError, EditableAnimationKey, EditableAnimationTarget,
        EditableAnimationTimeline, animation_target_inline_curve_expression,
        editable_animation_target, extract_editable_animation_timeline,
        promote_inline_curve_to_animation_target, remove_editable_animation_target,
        replace_editable_animation_targets, upsert_editable_animation_target,
    };
}

pub mod transitions {
    pub use crate::process::adapters::clip::transitions::{
        build_dissolve_expr, build_fade_expr, build_shock_zoom_expr, build_slide_expr,
        build_zoom_expr, sample_dissolve_factor, sample_fade_factor, sample_shock_zoom_factor,
        sample_slide_offset, sample_zoom_factor,
    };
}

pub mod text {
    pub use crate::scene::text::render::{
        PreparedTextAnimatorTargets, PreparedTextLayout, prepare_text_layout,
        prepare_text_layout_for_value,
    };
}
