// =========================================
// =========================================
// crates/motionloom/src/weaver/config/job.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Fully resolved settings: JSON has no implicit preset/override ambiguity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderJob {
    pub version: u32,
    pub scene: PathBuf,
    pub scene_id: String,
    pub frame: u32,
    pub resolution: [u32; 2],
    /// Pixel crop [x,y,width,height]; camera and RNG retain full-frame coordinates.
    pub region: Option<[u32; 4]>,
    pub memory_budget_mib: u32,
    pub render_style: String,
    pub sampling: Sampling,
    pub light_paths: LightPaths,
    pub lens: Lens,
    /// Offline-only look controls; never mutate the authored preview scene.
    #[serde(default)]
    pub lighting: LightingOverrides,
    pub seed: u32,
    pub sun_angular_diameter_degrees: f32,
    pub output: PathBuf,
    /// Optional native denoiser library supplied by the host; raw EXR is always kept.
    pub denoiser_library: Option<PathBuf>,
    /// World-space homogeneous medium; independent of legacy screen fog.
    pub volume: Option<Volume>,
    /// Explicitly acknowledge legacy screen effects that cannot be traced.
    pub allow_legacy_fog_omission: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightingOverrides {
    pub environment_intensity: Option<f32>,
    pub exposure: Option<f32>,
    /// Absolute intensities keyed by authored light ID; unknown IDs are errors.
    #[serde(default)]
    pub light_intensities: std::collections::BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    pub min_samples: u32,
    pub max_samples: u32,
    /// Relative standard error with a small dark-pixel floor; not a Cycles unit.
    pub noise_threshold: f32,
    pub batch_samples: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightPaths {
    pub total: u32,
    pub diffuse: u32,
    pub glossy: u32,
    pub transmission: u32,
    pub transparent: u32,
    pub roulette_start: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lens {
    pub enabled: bool,
    pub sensor_width_mm: f32,
    pub f_stop: f32,
    pub focus_distance: f32,
    pub aperture_blades: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Volume {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub extinction: f32,
    pub albedo: [f32; 3],
    pub anisotropy: f32,
    pub max_bounces: u32,
}
