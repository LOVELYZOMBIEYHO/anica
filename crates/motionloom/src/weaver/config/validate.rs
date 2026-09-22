// =========================================
// =========================================
// crates/motionloom/src/weaver/config/validate.rs

use super::*;
use crate::weaver::WeaverError;

impl RenderJob {
    /// Reject invalid budgets before allocating GPU memory or loading assets.
    pub fn validate(&self) -> Result<(), WeaverError> {
        let fail = |s: &str| Err(WeaverError::Invalid(s.into()));
        if self.version != 1 {
            return fail("unsupported job version");
        }
        if self.resolution.iter().any(|&v| v == 0 || v > 16384) {
            return fail("resolution must be 1..16384");
        }
        if self.resolution[0] as u64 * self.resolution[1] as u64 * 256
            > self.memory_budget_mib as u64 * 1024 * 1024
        {
            return fail(
                "film/output estimate exceeds memory_budget_mib; lower resolution or explicitly raise the budget",
            );
        }
        let s = &self.sampling;
        for value in self
            .lighting
            .environment_intensity
            .iter()
            .chain(self.lighting.exposure.iter())
            .chain(self.lighting.light_intensities.values())
        {
            if !value.is_finite() || *value < 0.0 {
                return fail("lighting overrides must be finite and non-negative");
            }
        }
        if let Some([x, y, w, h]) = self.region {
            if self.output_mode == SceneOutputMode::CompositeScene {
                return fail("region rendering is not yet compatible with composite_scene");
            }
            if w == 0
                || h == 0
                || x.checked_add(w).is_none_or(|end| end > self.resolution[0])
                || y.checked_add(h).is_none_or(|end| end > self.resolution[1])
            {
                return fail("region must be inside full resolution");
            }
        }
        if s.min_samples < 2
            || s.max_samples < s.min_samples
            || s.max_samples > 1_000_000
            || s.batch_samples == 0
            || s.batch_samples > 64
        {
            return fail("invalid sample bounds or batch size");
        }
        if !s.noise_threshold.is_finite() || !(0.0..=1.0).contains(&s.noise_threshold) {
            return fail("noise threshold must be finite and 0..1");
        }
        let b = &self.light_paths;
        if !self.sun_angular_diameter_degrees.is_finite()
            || !(0.0..=10.0).contains(&self.sun_angular_diameter_degrees)
        {
            return fail("sun angular diameter must be finite and 0..10 degrees");
        }
        if b.total == 0
            || b.total > 64
            || b.roulette_start == 0
            || b.roulette_start > b.total
            || b.diffuse > b.total
            || b.glossy > b.total
            || b.transmission > b.total
            || b.transparent == 0
            || b.transparent > 256
        {
            return fail("invalid bounce budgets");
        }
        let l = &self.lens;
        if [l.sensor_width_mm, l.f_stop, l.focus_distance]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || (l.aperture_blades != 0 && !(3..=32).contains(&l.aperture_blades))
        {
            return fail("invalid physical lens");
        }
        if self.scene_id.is_empty() || self.output.as_os_str().is_empty() {
            return fail("scene_id and output are required");
        }
        Ok(())
    }
}
