// =========================================
// =========================================
// crates/motionloom/src/weaver/api.rs

#[cfg(test)]
pub(crate) use super::jobs::converged;
pub use super::jobs::render;
pub use super::jobs::sequence::{MasterSequenceReport, render_master_sequence, render_sequence};
pub use super::preview::{PreviewDenoiser, PreviewSession};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderProgress {
    pub completed_tiles: u32,
    pub total_tiles: u32,
    pub tile_min_samples: u32,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderReport {
    pub renderer: String,
    pub backend: String,
    pub adapter: String,
    pub status: String,
    pub output: PathBuf,
    pub triangles: usize,
    pub elapsed_seconds: f64,
    pub converged_pixels: u64,
    pub sample_limit_pixels: u64,
    pub diagnostics: Vec<String>,
    #[serde(default)]
    pub timings: RenderTimings,
    #[serde(default)]
    pub frame_delta: FrameDeltaKind,
}

/// Work invalidated by this frame relative to the preceding sequence frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameDeltaKind {
    #[default]
    FirstFrame,
    CameraOrUniforms,
    SceneBufferUpdate,
    SceneRebuild,
    TwoDOnly,
}

/// Stable phase timings make sequence performance regressions attributable
/// without requiring an external profiler for every render.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderTimings {
    pub parse_seconds: f64,
    pub scene_evaluation_seconds: f64,
    pub geometry_pack_seconds: f64,
    pub gpu_setup_seconds: f64,
    pub path_trace_seconds: f64,
    pub composition_output_seconds: f64,
    pub denoise_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceProgress {
    pub frame: u32,
    pub completed_frames: u32,
    pub total_frames: u32,
    pub render: RenderProgress,
}
