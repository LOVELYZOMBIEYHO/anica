pub mod dsl;
pub mod error;
pub mod gltf_loader;
pub mod model;
pub mod model_inspection;
pub mod render;

pub use dsl::{is_world_graph_script, parse_world_graph_script};
pub use gltf_loader::{
    GlbLoadError, GlbMeshData, GlbMetadata, GlbNodeData, load_glb_mesh_data, load_glb_metadata,
    parse_glb_mesh_data, parse_glb_metadata,
};
pub use model::{
    WorldAction, WorldActionBone, WorldActionIk, WorldActionPose, WorldActor, WorldApplyAction,
    WorldBackground, WorldBackgroundFit, WorldBoneAxis, WorldBoneAxisMap, WorldCamera,
    WorldCameraControl, WorldCameraMode, WorldCameraProjection, WorldGraph, WorldMaterial,
    WorldMaterialStyle, WorldModelProfile, WorldNode, WorldPathStyle, WorldPlay, WorldPresent,
    WorldProfileRetarget, WorldRetarget, WorldRetargetMap, WorldSpritePlayback, WorldTime,
};
pub(crate) use model::{WorldAnimationAsset, WorldConstraint};
pub use model_inspection::{
    BodyBasisProposal, BoneAxisProposal, EnvironmentAnchorProposal, EnvironmentCoordinateProfile,
    EnvironmentInspectionDiagnostic, EnvironmentSurfaceProposal, GlbEnvironmentInspectionReport,
    GlbSkeletonInspectionReport, HumanoidBoneProposal, JointAlternative, ModelInspectionDiagnostic,
    ModelInspectionError, RestPoseProposal, SemanticAxisProposal, inspect_glb_environment_bytes,
    inspect_glb_environment_json, inspect_glb_environment_path, inspect_glb_skeleton_bytes,
    inspect_glb_skeleton_json, inspect_glb_skeleton_path,
};
pub use render::{
    CharacterDesignGpuViewport, CharacterDesignViewportFrame, Scene3DFrameProfile,
    WorldFrameRenderer, WorldGpuDiagnostics, WorldRenderError, WorldRenderProgress,
    diagnose_world_glb_gpu_plan, diagnose_world_graph_actor_gpu_frame, render_world_frame,
    render_world_graph_to_png_sequence_with_progress,
    render_world_graph_to_png_sequence_with_progress_and_cancel,
    render_world_graph_to_video_with_progress,
    render_world_graph_to_video_with_progress_and_cancel,
};
