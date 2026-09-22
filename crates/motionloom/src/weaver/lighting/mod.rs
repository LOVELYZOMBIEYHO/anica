// =========================================
// =========================================
// crates/motionloom/src/weaver/lighting/mod.rs

use crate::weaver::WeaverError;
use std::path::Path;

pub(crate) fn pack_atmosphere(
    params: &mut crate::weaver::backend::wgpu::CameraParams,
    medium: Option<&crate::scene::atmosphere::AtmosphereMediumPlan>,
    lights: &[crate::world::WorldLight],
    time_seconds: f32,
) {
    let Some(medium) = medium else { return };
    let (bounds_min, bounds_max) = match (medium.bounds_min, medium.bounds_max) {
        (Some(minimum), Some(maximum)) => (minimum, maximum),
        _ => ([-1.0e6; 3], [1.0e6; 3]),
    };
    params[16] = [bounds_min[0], bounds_min[1], bounds_min[2], medium.density];
    params[17] = [
        bounds_max[0],
        bounds_max[1],
        bounds_max[2],
        medium.anisotropy,
    ];
    params[18] = [
        medium.scattering_color[0],
        medium.scattering_color[1],
        medium.scattering_color[2],
        medium
            .volumetric_scattering
            .as_ref()
            .map_or(1, |scattering| scattering.max_bounces) as f32,
    ];
    params[9][0] = medium.base_height;
    params[9][1] = medium.height_falloff;
    params[9][2] = medium.edge_feather;
    params[15][2] = medium.affect_environment as u8 as f32;
    if let Some(scattering) = medium.volumetric_scattering.as_ref() {
        params[6][2] = scattering.max_distance;
        params[6][3] = lights
            .iter()
            .position(|light| light.id.as_deref() == Some(scattering.light_ref.as_str()))
            .map_or(0.0, |index| index as f32 + 1.0);
        params[15][3] = scattering.shaft_strength;
    }
    if let Some(caustics) = medium.water_caustics.as_ref() {
        params[20] = [
            caustics.intensity,
            caustics.scale,
            caustics.speed * time_seconds,
            caustics.attenuation,
        ];
        params[21] = [
            caustics.color[0],
            caustics.color[1],
            caustics.color[2],
            caustics.volume_term as u8 as f32 + 2.0 * caustics.surface_term as u8 as f32,
        ];
    }
}

#[cfg(test)]
mod atmosphere_tests {
    use super::*;
    use crate::scene::atmosphere::{
        AtmosphereMediumPlan, VolumetricQuality, VolumetricScatteringPlan, WaterCausticsPlan,
    };

    #[test]
    fn final_renderer_packs_the_renderer_independent_atmosphere_plan() {
        let plan = AtmosphereMediumPlan {
            density: 0.045,
            scattering_color: [0.2, 0.4, 0.6],
            anisotropy: 0.25,
            base_height: 1.5,
            height_falloff: 0.12,
            bounds_min: Some([-4.0, 0.0, -8.0]),
            bounds_max: Some([4.0, 6.0, -1.0]),
            edge_feather: 0.75,
            affect_environment: true,
            volumetric_scattering: Some(VolumetricScatteringPlan {
                light_ref: "sun".into(),
                shaft_strength: 1.2,
                max_distance: 36.0,
                shadowed: true,
                quality: VolumetricQuality::High,
                max_bounces: 4,
                debug_view: "none".into(),
            }),
            water_caustics: Some(WaterCausticsPlan {
                intensity: 0.3,
                scale: 0.5,
                speed: 0.2,
                attenuation: 0.6,
                color: [0.7, 0.8, 1.0],
                volume_term: true,
                surface_term: true,
            }),
        };
        let mut params = [[0.0; 4]; 26];
        pack_atmosphere(&mut params, Some(&plan), &[], 2.0);
        assert_eq!(params[16], [-4.0, 0.0, -8.0, 0.045]);
        assert_eq!(params[17], [4.0, 6.0, -1.0, 0.25]);
        assert_eq!(params[18], [0.2, 0.4, 0.6, 4.0]);
        assert_eq!(params[9][0..3], [1.5, 0.12, 0.75]);
        assert_eq!(params[15][2..4], [1.0, 1.2]);
        assert_eq!(params[20], [0.3, 0.5, 0.4, 0.6]);
        assert_eq!(params[21], [0.7, 0.8, 1.0, 3.0]);
    }
}

/// HDR radiance is preserved as float32; an equal-solid-angle mixture keeps
/// even dark texels sampleable and produces a matching directional PDF.
pub(crate) fn environment(
    path: &Path,
    data: &mut Vec<[f32; 4]>,
) -> Result<([f32; 4], [f32; 4]), WeaverError> {
    let hdr = matches!(
        path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "hdr" | "exr"
    );
    let image = image::open(path)?.to_rgb32f();
    let w = image.width();
    let h = image.height();
    let start = data.len();
    for pixel in image.pixels() {
        let rgb = pixel.0.map(|v| {
            if hdr {
                v
            } else if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        });
        if rgb.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(WeaverError::Invalid(
                "non-finite or negative environment radiance".into(),
            ));
        }
        data.push([rgb[0], rgb[1], rgb[2], 0.0]);
    }
    let cw = 128u32.min(w);
    let ch = 64u32.min(h);
    let mut weights = Vec::new();
    let mut total = 0.0f64;
    for y in 0..ch {
        for x in 0..cw {
            let mut lum = 0.0f64;
            let mut count = 0;
            for sy in y * h / ch..(y + 1) * h / ch {
                for sx in x * w / cw..(x + 1) * w / cw {
                    let c = data[start + (sy * w + sx) as usize];
                    lum += (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) as f64;
                    count += 1;
                }
            }
            let omega = 2.0 * std::f64::consts::PI / cw as f64
                * ((std::f64::consts::PI * y as f64 / ch as f64).cos()
                    - (std::f64::consts::PI * (y + 1) as f64 / ch as f64).cos());
            let weight = lum / count.max(1) as f64 * omega;
            weights.push((weight, omega));
            total += weight;
        }
    }
    let cdf_start = data.len();
    let mut cdf = 0.0;
    for (weight, omega) in weights {
        let prob = if total > 0.0 {
            0.9 * weight / total + 0.1 * omega / (4.0 * std::f64::consts::PI)
        } else {
            omega / (4.0 * std::f64::consts::PI)
        };
        cdf += prob;
        data.push([cdf as f32, (prob / omega) as f32, 0.0, 0.0]);
    }
    data.last_mut().unwrap()[0] = 1.0;
    Ok((
        // Buffer offsets travel through f32 uniforms as raw u32 bits so they
        // stay exact past 2^24 on large scenes.
        [f32::from_bits(start as u32), w as f32, h as f32, 0.0],
        [f32::from_bits(cdf_start as u32), cw as f32, ch as f32, 0.0],
    ))
}
