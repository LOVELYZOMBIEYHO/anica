// =========================================
// =========================================
// crates/motionloom/src/weaver/color/mod.rs

use crate::world::WorldLighting;
fn matrix(m: [[f32; 3]; 3], c: [f32; 3]) -> [f32; 3] {
    m.map(|row| row[0] * c[0] + row[1] * c[1] + row[2] * c[2])
}

fn white_balance_scale(kelvin: f32) -> [f32; 3] {
    // This is the canonical MotionLoom display transform used by Raster.
    let temperature = ((kelvin - 6500.0) / 6500.0).clamp(-0.75, 0.75);
    [1.0 + temperature * 0.16, 1.0, 1.0 - temperature * 0.16]
}

/// Match the authored display transform without quantizing the result.
fn display_encoded(rgb: [f32; 3], lighting: &WorldLighting) -> [f32; 3] {
    let color = &lighting.color_management;
    let mut rgb = rgb.map(|v| (v * color.exposure).max(0.0));
    let white_balance = white_balance_scale(color.white_balance_kelvin);
    rgb = std::array::from_fn(|index| rgb[index] * white_balance[index]);
    // Keep contrast and saturation in scene-linear space, before the final
    // tone mapper, exactly like Raster's color.wgsl + resolve_display path.
    rgb = rgb.map(|value| ((value - 0.18) * color.contrast + 0.18).max(0.0));
    let saturation = lighting
        .render_style
        .as_ref()
        .and_then(|s| s.post.saturation)
        .unwrap_or(1.0);
    let luminance = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
    rgb = rgb.map(|value| (luminance + (value - luminance) * saturation).max(0.0));
    if color.tone_mapping == "filmic_aces_v1" {
        rgb = matrix(
            [
                [0.59719, 0.35458, 0.04823],
                [0.076, 0.90834, 0.01566],
                [0.02840, 0.13383, 0.83777],
            ],
            rgb.map(|v| v / 0.6),
        );
        rgb = rgb.map(|v| {
            (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.432951) + 0.238081)
        });
        rgb = matrix(
            [
                [1.60475, -0.53108, -0.07367],
                [-0.10208, 1.10813, -0.00605],
                [-0.00327, -0.07276, 1.07602],
            ],
            rgb,
        );
    } else if color.tone_mapping == "aces" {
        rgb = rgb.map(|v| (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14));
    } else if color.tone_mapping == "reinhard" {
        rgb = rgb.map(|v| v / (1.0 + v));
    }
    rgb = if color.tone_mapping == "filmic_aces_v1" {
        rgb.map(|v| {
            if v <= 0.0031308 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
        })
    } else {
        rgb.map(|v| v.max(0.0).powf(1.0 / 2.2))
    };
    if let Some(style) = &lighting.render_style {
        let u = &style.universal;
        if u.enabled {
            rgb = rgb.map(|v| {
                ((v.max(0.0).powf(2.2) * u.exposure).powf(1.0 / 2.2) - 0.5) * u.contrast + 0.5
            });
            rgb = rgb.map(|v| v.clamp(0.0, 1.0));
            let lum = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
            rgb = std::array::from_fn(|i| {
                let tones = u.shadow_color[i] * (1.0 - lum) + u.highlight_color[i] * lum;
                let c = rgb[i] * (1.0 - u.tone_strength) + tones * u.tone_strength;
                c * (1.0 - u.tint_strength + u.tint_strength * u.tint[i])
            });
            let gray = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
            rgb = rgb.map(|v| gray + (v - gray) * u.saturation);
        }
    }
    rgb.map(|value| value.clamp(0.0, 1.0))
}

/// Return display-linear pixels for post-transform compositor layers.
pub(crate) fn display_linear(rgb: [f32; 3], lighting: &WorldLighting) -> [f32; 3] {
    display_encoded(rgb, lighting).map(crate::scene::compositor::srgb_to_linear)
}

/// Match the authored display transform once, after linear accumulation/denoising.
pub(crate) fn display(rgb: [f32; 3], lighting: &WorldLighting) -> [u8; 3] {
    display_encoded(rgb, lighting).map(|value| (value * 255.0 + 0.5) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s74_shadow_values_survive_the_raster_contrast_pipeline() {
        let mut lighting = WorldLighting::default();
        lighting.color_management.tone_mapping = "aces".into();
        lighting.color_management.exposure = 1.02;
        lighting.color_management.white_balance_kelvin = 6400.0;
        lighting.color_management.contrast = 1.06;

        let shadow = display([0.02, 0.02, 0.02], &lighting);
        assert!(shadow.iter().all(|channel| *channel > 0));
    }

    #[test]
    fn contrast_pivots_around_linear_middle_gray() {
        let mut lighting = WorldLighting::default();
        lighting.color_management.tone_mapping = "none".into();
        lighting.color_management.contrast = 1.5;

        let middle_gray = display([0.18; 3], &lighting);
        let expected = (0.18_f32.powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
        assert_eq!(middle_gray, [expected; 3]);
    }
}
