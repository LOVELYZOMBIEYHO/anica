// =========================================
// =========================================
// crates/motionloom/src/weaver/api.rs

#[cfg(test)]
pub(crate) use super::jobs::converged;
pub use super::jobs::render;
pub use super::jobs::sequence::render_sequence;
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceProgress {
    pub frame: u32,
    pub completed_frames: u32,
    pub total_frames: u32,
    pub render: RenderProgress,
}
