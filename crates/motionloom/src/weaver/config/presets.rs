// =========================================
// =========================================
// crates/motionloom/src/weaver/config/presets.rs

use super::*;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
pub enum QualityPreset {
    Production,
    Ultra,
    Reference,
}

impl RenderJob {
    /// Presets resolve once; callers may then explicitly adjust typed settings.
    pub fn new(scene: impl Into<PathBuf>, preset: QualityPreset) -> Self {
        let (min, max, noise, total, diffuse, glossy, roulette) = match preset {
            QualityPreset::Production => (128, 1024, 0.01, 12, 6, 8, 5),
            QualityPreset::Ultra => (256, 4096, 0.003, 16, 8, 12, 6),
            QualityPreset::Reference => (512, 16384, 0.001, 24, 12, 16, 8),
        };
        Self {
            version: 1,
            scene: scene.into(),
            scene_id: String::new(),
            frame: 0,
            resolution: [3840, 2160],
            region: None,
            memory_budget_mib: 4096,
            render_style: String::new(),
            sampling: Sampling {
                min_samples: min,
                max_samples: max,
                noise_threshold: noise,
                batch_samples: 4,
            },
            light_paths: LightPaths {
                total,
                diffuse,
                glossy,
                transmission: total,
                transparent: 64,
                roulette_start: roulette,
            },
            lens: Lens {
                enabled: true,
                sensor_width_mm: 36.0,
                f_stop: 4.0,
                focus_distance: 16.2,
                aperture_blades: 9,
            },
            seed: 89,
            lighting: LightingOverrides::default(),
            sun_angular_diameter_degrees: 0.53,
            output: PathBuf::from(".render-output/weaver"),
            denoiser_library: None,
            volume: None,
            allow_legacy_fog_omission: false,
        }
    }
}
