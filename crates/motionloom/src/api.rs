//! Recommended MotionLoom public API surface.
//!
//! This module intentionally re-exports the main integration APIs without
//! exposing MotionLoom's internal module tree.
//!
//! Prefer this module for new host applications, examples, and third-party
//! integrations.
//!
//! # Parse and render
//!
//! ```no_run
//! use motionloom::api::{SceneRenderProfile, parse_graph_script, render_scene_graph_frame};
//!
//! # async fn run(script: &str) -> Result<(), Box<dyn std::error::Error>> {
//! let graph = parse_graph_script(script)?;
//! let frame = render_scene_graph_frame(&graph, 0, SceneRenderProfile::Gpu).await?;
//! frame.save("frame.png")?;
//! # Ok(())
//! # }
//! ```
//!
//! # Reusable renderer
//!
//! ```no_run
//! use motionloom::api::{SceneRenderProfile, SceneRenderer, parse_graph_script};
//!
//! # async fn run(script: &str) -> Result<(), Box<dyn std::error::Error>> {
//! let graph = parse_graph_script(script)?;
//! let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await?;
//! let frame_0 = renderer.render_frame(&graph, 0).await?;
//! let frame_1 = renderer.render_frame(&graph, 1).await?;
//! frame_0.save("frame_0000.png")?;
//! frame_1.save("frame_0001.png")?;
//! # Ok(())
//! # }
//! ```
//!
//! # Root document export
//!
//! `parse_motionloom_document` and the `render_motionloom_document_*`
//! functions auto-route unified Scene and Process documents. The former
//! `<World>` DSL tag has been removed; `world` remains a Scene coordinate space.
//!
//! ```no_run
//! use std::path::Path;
//! use motionloom::api::render_motionloom_document_to_png_sequence_with_progress;
//!
//! # async fn run(script: &str) -> Result<(), Box<dyn std::error::Error>> {
//! render_motionloom_document_to_png_sequence_with_progress(
//!     script,
//!     ".",
//!     Path::new("frames"),
//!     30,
//!     |progress| eprintln!("{}/{}", progress.rendered_frames(), progress.total_frames()),
//! ).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Process catalog
//!
//! ```no_run
//! use motionloom::api::{process_effect_for_id, process_effects};
//!
//! for effect in process_effects() {
//!     println!("{}: {}", effect.id, effect.display_name);
//! }
//!
//! let bloom = process_effect_for_id("glow_bloom");
//! assert!(bloom.is_some());
//! ```

pub use crate::{
    AssetResolver, AssetSource, GpuCompatibilityIssue, GpuCompatibilityReport,
    GpuCompatibilitySeverity, GpuCompatibilityTarget, GraphParseError, GraphScript,
    MemoryAssetResolver, MotionLoomDocument, MotionLoomError, MotionLoomRenderProgress,
    MotionLoomSceneRenderError, PathAssetResolver, ProcessCategory, ProcessEffectDefinition,
    ProcessGraph, RenderPassDag, RenderPassDagEdge, RenderPassDagKind, RenderPassDagNode,
    RenderPassSpace, RootGraphError, RuntimeCompileError, RuntimeFrameOutput,
    RuntimeProcessEffectInstance, RuntimeProcessParamValue, RuntimeProgram, SceneGpuTexture,
    ScenePlatformPreviewSurface, ScenePreviewBackend, ScenePreviewPath, ScenePreviewPixelFormat,
    ScenePreviewSurface, ScenePreviewSurfaceOptions, SceneRenderError, SceneRenderProfile,
    SceneRenderProgress, SceneRenderer, SkeletonDiagnostic, SkeletonDiagnosticSeverity,
    SkeletonOverlayPrimitive, SkeletonValidationReport, auto_correct_skeleton,
    build_skeleton_overlay, builtin_proportion_profile, builtin_proportion_profiles,
    builtin_skeleton_pose_presets, clear_scene_asset_roots, compile_render_pass_dag,
    compile_runtime_program, inspect_gpu_compatibility, inspect_root_graph, is_graph_script,
    is_known_process_kernel, is_process_graph_script, kernel_source_by_name,
    next_scene_output_path, next_scene_output_path_for_profile, parse_graph_script,
    parse_motionloom_document, parse_process_graph_script, process_effect_for_id, process_effects,
    process_effects_for_category, render_motionloom_document_to_png_sequence_with_progress,
    render_motionloom_document_to_png_sequence_with_progress_and_cancel,
    render_motionloom_document_to_video_with_progress,
    render_motionloom_document_to_video_with_progress_and_cancel, render_scene_graph_frame,
    render_scene_graph_frame_with_cpu_inputs, render_scene_graph_frame_with_resolver,
    render_scene_graph_to_png_sequence_with_progress,
    render_scene_graph_to_png_sequence_with_progress_and_cancel, render_scene_graph_to_video,
    render_scene_graph_to_video_with_progress,
    render_scene_graph_to_video_with_progress_and_cancel, set_scene_asset_roots,
};

#[cfg(not(target_arch = "wasm32"))]
pub use crate::VectorFrameBenchmark;

#[cfg(all(unix, not(target_os = "macos"), not(target_arch = "wasm32")))]
pub use crate::DmabufPlane;

#[cfg(target_os = "windows")]
pub use crate::{WindowsD3DSharedHandle, WindowsD3DSharedSurface};
