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
    /// Select 3D AOV output or assemble the complete authored scene.
    #[serde(default)]
    pub output_mode: SceneOutputMode,
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
    /// Build box-filtered mip chains and select LOD from the ray footprint.
    /// Off by default so existing renders keep base-level sampling.
    #[serde(default)]
    pub texture_mips: bool,
    /// Explicit temporary fallback for transmissive materials until the BSDF lands.
    #[serde(default)]
    pub allow_transmission_stopgap: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneOutputMode {
    #[default]
    ThreeDOnly,
    CompositeScene,
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

/// Delivery settings for a resumable, compositor-complete master sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MasterSequenceSettings {
    pub version: u32,
    /// Keep the pre-display scene composite beside the display master sequence.
    pub write_scene_composite: bool,
    /// Encode a ProRes 4444 XQ mezzanine from the display-master sequence.
    pub encode_prores: bool,
    /// Encode a lightweight H.264 review movie from the same master sequence.
    pub encode_preview: bool,
    /// Render and mux authored audio when the graph contains audio clips.
    pub include_audio: bool,
    pub color_target: MasterColorTarget,
    /// Completed frames already have canonical master EXRs, so production
    /// sequences normally discard duplicate AOV/checkpoint job directories.
    #[serde(default)]
    pub checkpoint_retention: SequenceCheckpointRetention,
    /// Independent is deterministic per frame; temporal mode reprojects a
    /// conservative history through the motion AOV before master output.
    #[serde(default)]
    pub denoise_mode: SequenceDenoiseMode,
}

impl Default for MasterSequenceSettings {
    fn default() -> Self {
        Self {
            version: 1,
            write_scene_composite: true,
            encode_prores: true,
            encode_preview: true,
            include_audio: true,
            color_target: MasterColorTarget::SdrBt709,
            checkpoint_retention: SequenceCheckpointRetention::IncompleteOnly,
            denoise_mode: SequenceDenoiseMode::Independent,
        }
    }
}

/// Video delivery color is explicit. HDR targets will be added only with a
/// fully validated luminance transform rather than relabelling SDR pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MasterColorTarget {
    SdrBt709,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceCheckpointRetention {
    /// Keep only an interrupted frame's tile data; canonical EXRs resume every
    /// completed frame without retaining duplicate render-job output.
    #[default]
    IncompleteOnly,
    /// Keep every raw AOV, denoised output and tile checkpoint for diagnostics.
    All,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceDenoiseMode {
    #[default]
    Independent,
    Temporal,
}
