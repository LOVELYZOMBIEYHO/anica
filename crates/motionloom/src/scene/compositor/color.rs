// =========================================
// =========================================
// crates/motionloom/src/scene/compositor/color.rs

use serde::{Deserialize, Serialize};

/// The compositor working format is fixed so renderer hand-offs are explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompositionPixelFormat {
    LinearPremultipliedRgba16Float,
}

impl Default for CompositionPixelFormat {
    fn default() -> Self {
        Self::LinearPremultipliedRgba16Float
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceColorSpace {
    Srgb,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaMode {
    Straight,
    Premultiplied,
}

/// Converts an encoded sRGB channel into scene-linear light.
pub fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Encodes a display-linear channel for an sRGB output surface.
pub fn linear_to_srgb(value: f32) -> f32 {
    let value = value.max(0.0);
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Normalizes any declared source pixel into the compositor working contract.
pub fn to_linear_premultiplied(
    rgba: [f32; 4],
    color_space: SourceColorSpace,
    alpha_mode: AlphaMode,
) -> [f32; 4] {
    let alpha = rgba[3].clamp(0.0, 1.0);
    let mut rgb = [rgba[0], rgba[1], rgba[2]];
    if color_space == SourceColorSpace::Srgb {
        rgb = rgb.map(srgb_to_linear);
    }
    if alpha_mode == AlphaMode::Straight {
        rgb = rgb.map(|channel| channel * alpha);
    }
    [rgb[0], rgb[1], rgb[2], alpha]
}

/// Porter-Duff source-over for linear premultiplied pixels.
pub fn premultiplied_over(source: [f32; 4], destination: [f32; 4]) -> [f32; 4] {
    let remaining = 1.0 - source[3].clamp(0.0, 1.0);
    [
        source[0] + destination[0] * remaining,
        source[1] + destination[1] * remaining,
        source[2] + destination[2] * remaining,
        source[3] + destination[3] * remaining,
    ]
}
