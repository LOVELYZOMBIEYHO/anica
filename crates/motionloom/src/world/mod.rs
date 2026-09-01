pub mod dsl;
pub mod error;
pub mod gltf_loader;
pub mod model;
pub mod model_inspection;
pub mod primitive;
pub mod render;
pub mod terrain;
pub mod vegetation;

pub use dsl::{is_world_graph_script, parse_world_graph_script};
pub use gltf_loader::{
    GlbLoadError, GlbMeshData, GlbMetadata, GlbNodeData, load_glb_mesh_data, load_glb_metadata,
    parse_glb_mesh_data, parse_glb_metadata,
};
pub use model::{
    WorldAction, WorldActionBone, WorldActionIk, WorldActionPose, WorldActor, WorldApplyAction,
    WorldAtmosphereFog, WorldBackground, WorldBackgroundFit, WorldBoneAxis, WorldBoneAxisMap,
    WorldCamera, WorldCameraControl, WorldCameraMode, WorldCameraProjection, WorldColorManagement,
    WorldDepthOfField, WorldEnvironmentLighting, WorldGraph, WorldLight, WorldLightKind,
    WorldLighting, WorldMaterial, WorldMaterialStyle, WorldModelProfile, WorldNode, WorldPathStyle,
    WorldPlay, WorldPresent, WorldProfileRetarget, WorldRetarget, WorldRetargetMap,
    WorldSpritePlayback, WorldTime,
};
pub(crate) use model::{WorldAnimationAsset, WorldConstraint};
pub use model_inspection::{
    BodyBasisProposal, BoneAxisProposal, DetectedHumanoidRig, EnvironmentAnchorProposal,
    EnvironmentCoordinateProfile, EnvironmentInspectionDiagnostic, EnvironmentSurfaceProposal,
    GlbEnvironmentInspectionReport, GlbHumanoidProfileInspectionReport,
    GlbSkeletonInspectionReport, HumanoidActionCompatibilityReport, HumanoidBoneProposal,
    JointAlternative, ModelInspectionDiagnostic, ModelInspectionError, RestPoseProposal,
    SemanticAxisProposal, inspect_glb_environment_bytes, inspect_glb_environment_json,
    inspect_glb_environment_path, inspect_glb_humanoid_profile_bytes,
    inspect_glb_humanoid_profile_json, inspect_glb_skeleton_bytes, inspect_glb_skeleton_json,
    inspect_glb_skeleton_path, inspect_humanoid_action_compatibility,
};
pub use render::pose_diagnostics::{
    ActorPoseDiagnostic, JointPoseDiagnostic, PoseDiagnosticError, diagnose_world_actor_pose,
    evaluate_world_actor_rig,
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
