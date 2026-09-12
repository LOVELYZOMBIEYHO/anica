// =========================================
// =========================================
// crates/motionloom/src/weaver/color/mod.rs

use crate::world::WorldLighting;
fn matrix(m: [[f32; 3]; 3], c: [f32; 3]) -> [f32; 3] {
    m.map(|row| row[0] * c[0] + row[1] * c[1] + row[2] * c[2])
}

/// Match the authored display transform once, after linear accumulation/denoising.
pub(crate) fn display(rgb: [f32; 3], lighting: &WorldLighting) -> [u8; 3] {
    let color = &lighting.color_management;
    let mut rgb = rgb.map(|v| (v * color.exposure).max(0.0));
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
    let saturation = lighting
        .render_style
        .as_ref()
        .and_then(|s| s.post.saturation)
        .unwrap_or(1.0);
    let lum = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
    rgb =
        rgb.map(|v| ((lum + (v - lum) * saturation - 0.5) * color.contrast + 0.5).clamp(0.0, 1.0));
    rgb = rgb.map(|v| {
        if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    });
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
    rgb.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
}
