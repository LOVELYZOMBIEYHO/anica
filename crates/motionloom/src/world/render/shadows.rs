// =========================================
// =========================================
// crates/motionloom/src/world/render/shadows.rs

//! Fit stable shadow volumes around frame-local actor bounds.

use super::*;

pub(super) fn fit_rigid_shadow_volume(
    mut lighting: GpuWorldLightingParams,
    bounds: &[GpuWorldActorBounds],
) -> GpuWorldLightingParams {
    if bounds.is_empty() || lighting.color1[3] <= 0.0 {
        return lighting;
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &((lo, hi), p) in bounds {
        let local = std::array::from_fn(|i| ((lo[i] + hi[i]) * 0.5 - p.model[i]) * p.model[3]);
        let center = quat_rotate_vec3(quat_normalize_xyzw(p.actor_rotation), local);
        let radius = (0..3)
            .map(|i| ((hi[i] - lo[i]) * 0.5).powi(2))
            .sum::<f32>()
            .sqrt()
            * p.model[3].abs();
        for i in 0..3 {
            min[i] = min[i].min(center[i] + p.actor[i] - radius);
            max[i] = max[i].max(center[i] + p.actor[i] + radius);
        }
    }
    let center: [f32; 3] = std::array::from_fn(|i| (min[i] + max[i]) * 0.5);
    let radius = (0..3)
        .map(|i| ((max[i] - min[i]) * 0.5).powi(2))
        .sum::<f32>()
        .sqrt()
        * 1.05;
    if !radius.is_finite() || radius <= 0.0 || radius >= 14.0 {
        return lighting;
    }
    let radius = radius.max(0.01);
    lighting.shadow0[3] = radius;
    lighting.shadow1[3] = radius;
    lighting.shadow2[3] = radius * 2.0;
    // The old normalized bias detached small-scale hair shadows from the forehead.
    lighting.shadow3 = [center[0], center[1], center[2], 0.00015];
    lighting
}
