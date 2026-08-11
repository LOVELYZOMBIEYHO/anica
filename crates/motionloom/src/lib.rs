// =========================================
// crates/motionloom/src/lib.rs

//! MotionLoom is a Rust parser and renderer for MotionLoom graph DSL.
//!
//! It supports scene graphs, process/effect graphs, live preview surfaces, PNG
//! sequence export, and video export through a caller-provided FFmpeg binary.
//!
//! Most integrations should import from [`api`] or [`prelude`] instead of
//! depending on MotionLoom's internal AST layout. The crate root keeps broader
//! re-exports for short-term compatibility with existing applications.
//!
//! # Quick start: render one frame
//!
//! ```no_run
//! use motionloom::api::{SceneRenderProfile, parse_graph_script, render_scene_graph_frame};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let script = r##"
//! <Graph fps={30} duration="1s" size={[640,360]}>
//!   <Background color="#101827" />
//!   <Scene id="example">
//!     <Timeline>
//!       <Track id="main" space="world" z="0">
//!         <Sequence from="0s" duration="1s" out="hold">
//!           <Layer>
//!             <Circle x="320" y="180" radius="96" color="#4CC9F0" />
//!             <Text x="320" y="306" value="MotionLoom" fontSize="34" color="#FFFFFF" />
//!           </Layer>
//!         </Sequence>
//!       </Track>
//!     </Timeline>
//!   </Scene>
//!   <Present from="example" />
//! </Graph>
//! "##;
//!
//! let graph = parse_graph_script(script)?;
//! let frame = render_scene_graph_frame(&graph, 0, SceneRenderProfile::Gpu).await?;
//! frame.save("frame.png")?;
//! # Ok(())
//! # }
//! ```
//!
//! # API stability
//!
//! - [`api`] is the recommended stable integration surface.
//! - [`prelude`] contains a small convenience import set.
//! - [`experimental`] exposes advanced editor, legacy GLB compatibility, and
//!   timeline helpers that are useful but may change before the main API.
//!
//! # Export paths
//!
//! Use PNG sequence export when FFmpeg is not available. Use video export when
//! the host application supplies an FFmpeg binary path.
//!
//! # Scene editor DSL
//!
//! MotionLoom scene graphs include editor-oriented controls in addition to
//! static vector/text/image nodes:
//!
//! - `curve(...)` remains the compact expression API for numeric animation.
//! - [`AnimationTargetNode`] and [`AnimationKeyNode`] represent UI-editable
//!   keyframes with `time` or `frame` timing. They support `x`, `y`,
//!   `rotation`, `scale`, extended transforms, `opacity`, and `d` path-shape
//!   keys.
//! - [`extract_editable_animation_timeline`],
//!   [`upsert_editable_animation_target`], and
//!   [`replace_editable_animation_targets`] provide a UI-facing read/write
//!   layer for `.motionloom` keyframe editing.
//! - [`PuppetNode`] and [`PinNode`] provide AE-style pin deformation with auto
//!   mesh by default; [`MeshTopologyNode`] plus [`VertexNode`] and
//!   [`TriangleNode`] provide optional manual topology. `PuppetWarp` can target
//!   one Group id or capture earlier artwork in its current Layer with
//!   `target="@layer" capture="before"`.
//! - [`SkeletonNode`], [`ActionNode`], [`CharacterNode`], [`PartNode`], and
//!   [`ApplyActionNode`] provide 2D character rigging. Actions can contain
//!   two-bone IK or CCD-chain IK targets.
//! - [`CharacterNode`] can draw vector children and/or a raster `src`, `image`,
//!   or `path` using the same loader as `<Image>`, including PNG/JPG paths and
//!   raster `data:image/*;base64,...` URIs.
//!
//! ```no_run
//! use std::path::Path;
//! use motionloom::api::{
//!     parse_graph_script, render_scene_graph_to_png_sequence_with_progress,
//! };
//!
//! # async fn run(script: &str) -> Result<(), Box<dyn std::error::Error>> {
//! let graph = parse_graph_script(script)?;
//! render_scene_graph_to_png_sequence_with_progress(
//!     &graph,
//!     Path::new("frames"),
//!     30,
//!     |progress| eprintln!("{}/{}", progress.rendered_frames, progress.total_frames),
//! ).await?;
//! # Ok(())
//! # }
//! ```

mod asset;
mod authoring;
mod common;
mod compat;
mod dsl;
mod error;
mod export;
pub mod preview;
pub mod preview_protocol;
mod process;
mod render_graph;
mod root;
mod scene;
pub mod simulation;
mod world;

pub mod api;
pub mod experimental;
pub mod prelude;

#[cfg(target_arch = "wasm32")]
pub mod wasm_api;

pub use asset::{AssetResolver, AssetSource, MemoryAssetResolver, PathAssetResolver};
pub use authoring::{
    AuthoringDiagnostic, AuthoringDiagnosticSeverity, AuthoringStatus, AuthoringSuggestion,
    AuthoringSummary, EffectiveGraphSummary, MotionLoomAuthoringReport, MotionLoomShowcaseSchema,
    ShowcaseAttributeSchema, ShowcaseTagSchema, analyze_motionloom_script,
    analyze_motionloom_script_for_target, motionloom_analyze_script_for_target_json,
    motionloom_analyze_script_json, motionloom_showcase_schema_json,
};
pub use compat::{
    GpuCompatibilityIssue, GpuCompatibilityReport, GpuCompatibilitySeverity,
    GpuCompatibilityTarget, ScenePreviewPath, inspect_gpu_compatibility,
};
pub use export::{EncodeError, VideoEncoder, VideoFrame, create_encoder};
pub use render_graph::{
    RenderEffectScope, RenderPassDag, RenderPassDagEdge, RenderPassDagKind, RenderPassDagNode,
    RenderPassSpace, compile_render_pass_dag,
};

pub use common::keyframe::ScalarKeyframe;
pub use dsl::{
    ActionBoneNode, ActionNode, ActionPoseNode, AnimationKeyNode, AnimationTargetNode,
    ApplyActionNode, BackgroundNode, GraphAssetKind, GraphAssetNode, GraphScript, ImageNode,
    ModelProfileBoneAxisMapNode, ModelProfileBoneAxisNode, ModelProfileNode,
    ModelProfileRetargetMapNode, ModelProfileRetargetNode, ProcessDefinitionNode, SkeletonBoneNode,
    SkeletonConstraintNode, SkeletonControlNode, SkeletonGuideNode, SkeletonLandmarkNode,
    SkeletonMeasureNode, SkeletonNode, SkeletonRatioNode, SkeletonRegionNode, SvgNode,
    is_graph_script, parse_graph_script,
};
pub use error::{GraphParseError, MotionLoomError, RootGraphError, RuntimeCompileError};
pub use preview::{
    WgpuPreviewAdaptiveController, WgpuPreviewEngine, WgpuPreviewEngineError, WgpuPreviewFrame,
    WgpuPreviewGraphCache, WgpuPreviewPreloadReport, WgpuPreviewQuality,
};
pub use preview_protocol::{
    PREVIEW_PROTOCOL_VERSION, PreviewCommand, PreviewEvent, PreviewInteractionMode,
    PreviewInteractionNode,
};
pub use process::adapters::clip::curve::sample_anim_f32;
pub use process::adapters::clip::model::{
    AnimF32, ClipZoomSpec, ColorRgba, LayerEffectClip, LocalMaskLayer, MAX_LOCAL_MASK_LAYERS,
    SlideDirection, TextStyle, VideoEffect, ZoomStyle,
};
pub use process::cpu_renderer::{
    ProcessCpuRenderError, ProcessCpuRenderer, render_process_frame_cpu,
};
pub use process::effect::{
    LayerColorBlurEffects, PerClipColorBlurEffects, combine_clip_with_layer,
};
pub use process::error::{
    MotionLoomProcessError, ProcessError, ProcessGraphError, ProcessParseError, ProcessRuntimeError,
};
pub use process::model::{
    AlphaMode, BlendMode, BufferElemType, BufferNode, BufferUsage, ColorSpace, EffectNode,
    GraphApplyScope, InputNode, InputType, LayerNode, LoadOp, OutputNode, OutputTarget, PassCache,
    PassKind, PassNode, PassParam, PassRole, PassTransitionClips, PassTransitionEasing,
    PassTransitionFallback, PassTransitionMode, PresentNode, PresentTarget, Quality, ResourceRef,
    SampleAddress, SampleConfig, SampleFilter, StoreOp, TexNode, TexUsage, TextureFormat,
};
pub use process::parser::{ProcessGraph, is_process_graph_script, parse_process_graph_script};
pub use process::pass::{default_kernel_for_effect, resolve_pass_kernel};
pub use process::process_catalog::{
    PROCESS_CATEGORIES, PROCESS_EFFECTS, ProcessCategory, ProcessEffectDefinition,
    is_known_process_kernel, kernel_source_by_name, process_effect_for_id, process_effects,
    process_effects_for_category,
};
pub use process::runtime::{
    BlurSharpenMode, RuntimeFrameOutput, RuntimeProcessEffectInstance, RuntimeProcessParamValue,
    RuntimeProgram, compile_runtime_program, eval_time_expr,
};
pub use root::{
    MotionLoomDocument, MotionLoomRenderProgress, RootGraphDomain, RootGraphShell,
    inspect_root_graph, parse_motionloom_document,
    render_motionloom_document_to_png_sequence_with_progress,
    render_motionloom_document_to_png_sequence_with_progress_and_cancel,
    render_motionloom_document_to_video_with_progress,
    render_motionloom_document_to_video_with_progress_and_cancel,
};
pub use scene::animation::{
    ANIMATION_PROPERTY_DESCRIPTORS, AnimationCapabilityReport, AnimationDiagnostic,
    AnimationDiagnosticSeverity, AnimationInterpolation, AnimationPropertyDescriptor,
    AnimationValueType, animation_properties_for_node_kind, animation_property_descriptor,
    animation_property_schema_json, inspect_animation_targets,
};
pub use scene::domain::{
    ProportionProfile, SkeletonDiagnostic, SkeletonDiagnosticSeverity, SkeletonOverlayPrimitive,
    SkeletonPosePreset, SkeletonValidationReport, auto_correct_skeleton, build_skeleton_overlay,
    builtin_proportion_profile, builtin_proportion_profiles, builtin_skeleton_pose_presets,
    validate_skeleton,
};
pub use scene::editor_keyframes::{
    AnimationKeyframeEditError, EditableAnimationKey, EditableAnimationTarget,
    EditableAnimationTimeline, animation_target_inline_curve_expression, editable_animation_target,
    extract_editable_animation_timeline, promote_inline_curve_to_animation_target,
    remove_editable_animation_target, replace_editable_animation_targets,
    upsert_editable_animation_target,
};
pub use scene::model::{
    BrushDef, CameraNode, CharacterNode, CircleNode, ComponentDerivedDef, ComponentNode,
    ComponentParamDef, ComponentParamValue, ComponentSlotDef, ComponentSlotValue, DefsNode,
    EdgeNode, FaceJawNode, FilterDef, FilterStepDef, FontDef, GradientDef, GradientStop, GroupNode,
    LineNode, LinearGradientDef, MaskNode, MeshTopologyNode, PaletteColorDef, PaletteNode,
    PartNode, PathNode, PinNode, PixelGridNode, PolylineNode, PrecomposeNode, PuppetNode,
    RadialGradientDef, RectNode, RegionNode, RepeatNode, SceneChainNode, SceneLayerNode, SceneNode,
    SceneRootNode, SceneSequenceNode, SceneTimelineNode, SceneTrackNode, ShadowNode, TriangleNode,
    UseNode, VertexNode,
};
#[cfg(all(unix, not(target_os = "macos"), not(target_arch = "wasm32")))]
pub use scene::render::DmabufPlane;
pub use scene::render::{
    GpuFrameTexture, MotionLoomSceneRenderError, SceneGpuTexture, ScenePlatformPreviewSurface,
    ScenePreviewBackend, ScenePreviewPixelFormat, ScenePreviewSurface, ScenePreviewSurfaceOptions,
    SceneRenderError, SceneRenderProfile, SceneRenderProgress, SceneRenderer,
    clear_scene_asset_roots, next_scene_output_path, next_scene_output_path_for_profile,
    render_scene_graph_frame, render_scene_graph_frame_with_cpu_inputs,
    render_scene_graph_frame_with_resolver, render_scene_graph_to_png_sequence_with_progress,
    render_scene_graph_to_png_sequence_with_progress_and_cancel, render_scene_graph_to_video,
    render_scene_graph_to_video_with_progress,
    render_scene_graph_to_video_with_progress_and_cancel, set_scene_asset_roots,
};
#[cfg(not(target_arch = "wasm32"))]
pub use scene::render::{SceneCpuFrameProfile, VectorFrameBenchmark};
#[cfg(target_os = "windows")]
pub use scene::render::{WindowsD3DSharedHandle, WindowsD3DSharedSurface};
pub use scene::text::{
    TextAlignMode, TextAnimatorNode, TextEffectNode, TextGlowEffectNode, TextLayoutNode, TextNode,
    TextOverflowMode, TextSelectorKind, TextStyleOverrideNode, TextTransformNode, TextWrapMode,
};
pub use world::error::{MotionLoomWorldError, WorldAssetError, WorldError, WorldParseError};
pub use world::{
    BodyBasisProposal, BoneAxisProposal, CharacterDesignGpuViewport, CharacterDesignViewportFrame,
    EnvironmentAnchorProposal, EnvironmentCoordinateProfile, EnvironmentInspectionDiagnostic,
    EnvironmentSurfaceProposal, GlbEnvironmentInspectionReport, GlbLoadError, GlbMeshData,
    GlbMetadata, GlbNodeData, GlbSkeletonInspectionReport, HumanoidBoneProposal, JointAlternative,
    ModelInspectionDiagnostic, ModelInspectionError, RestPoseProposal, Scene3DFrameProfile,
    SemanticAxisProposal, WorldAction, WorldActionBone, WorldActionIk, WorldActionPose, WorldActor,
    WorldApplyAction, WorldBackground, WorldBackgroundFit, WorldBoneAxis, WorldBoneAxisMap,
    WorldCamera, WorldCameraControl, WorldCameraMode, WorldCameraProjection, WorldFrameRenderer,
    WorldGpuDiagnostics, WorldGraph, WorldMaterial, WorldMaterialStyle, WorldModelProfile,
    WorldNode, WorldPathStyle, WorldPlay, WorldPresent, WorldProfileRetarget, WorldRenderError,
    WorldRenderProgress, WorldRetarget, WorldRetargetMap, WorldSpritePlayback, WorldTime,
    diagnose_world_glb_gpu_plan, diagnose_world_graph_actor_gpu_frame,
    inspect_glb_environment_bytes, inspect_glb_environment_json, inspect_glb_environment_path,
    inspect_glb_skeleton_bytes, inspect_glb_skeleton_json, inspect_glb_skeleton_path,
    is_world_graph_script, load_glb_mesh_data, load_glb_metadata, parse_glb_mesh_data,
    parse_glb_metadata, parse_world_graph_script, render_world_frame,
    render_world_graph_to_png_sequence_with_progress,
    render_world_graph_to_png_sequence_with_progress_and_cancel,
    render_world_graph_to_video_with_progress,
};

/// FrameContext carries the minimum timeline state needed for effect evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameContext {
    pub time_ms: u64,
    pub fps: f32,
}

impl FrameContext {
    /// Build a frame context from timeline time in milliseconds.
    pub const fn new(time_ms: u64, fps: f32) -> Self {
        Self { time_ms, fps }
    }

    /// Convert the timeline position to seconds for curve math.
    pub fn time_seconds(self) -> f32 {
        self.time_ms as f32 / 1000.0
    }
}

/// Lerp is the base interpolation primitive used by keyframe curves.
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Sample a simple linear segment between two keyed values in milliseconds.
pub fn sample_linear_segment(
    now_ms: u64,
    start_ms: u64,
    end_ms: u64,
    start_value: f32,
    end_value: f32,
) -> f32 {
    if end_ms <= start_ms {
        return end_value;
    }
    let span = (end_ms - start_ms) as f32;
    let elapsed = now_ms.saturating_sub(start_ms) as f32;
    lerp(start_value, end_value, elapsed / span)
}

#[cfg(test)]
mod tests {
    use super::{FrameContext, lerp, sample_linear_segment};

    #[test]
    fn context_seconds_conversion() {
        let ctx = FrameContext::new(1500, 30.0);
        assert!((ctx.time_seconds() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn lerp_clamps_input_t() {
        assert!((lerp(0.0, 10.0, -1.0) - 0.0).abs() < f32::EPSILON);
        assert!((lerp(0.0, 10.0, 2.0) - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn linear_segment_samples_midpoint() {
        let value = sample_linear_segment(500, 0, 1000, 0.0, 1.0);
        assert!((value - 0.5).abs() < 0.001);
    }
}
