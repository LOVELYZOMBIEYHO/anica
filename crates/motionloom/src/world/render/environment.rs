// =========================================
// =========================================
// crates/motionloom/src/world/render/environment.rs

//! Decode environment images into linear floating-point mip chains.

use super::*;

/// Decode HDR/EXR or ordinary images into the same linear environment format.
pub(super) fn load_environment_image_from_resolved(
    resolved: &ResolvedWorldAsset,
) -> Result<WorldEnvironmentImage, WorldRenderError> {
    let image = load_rgba_image_from_resolved(resolved, |path, source| {
        WorldRenderError::BackgroundImage { path, source }
    })?;
    let source_is_linear = matches!(
        image.color(),
        image::ColorType::Rgb32F | image::ColorType::Rgba32F
    );
    let rgba = image.to_rgba32f();
    let width = rgba.width().max(1);
    let height = rgba.height().max(1);
    let mut pixels = rgba
        .pixels()
        .map(|pixel| {
            let mut rgb = [
                pixel[0].max(0.0),
                pixel[1].max(0.0),
                pixel[2].max(0.0),
                pixel[3],
            ];
            if !source_is_linear {
                rgb[0] = rgb[0].powf(2.2);
                rgb[1] = rgb[1].powf(2.2);
                rgb[2] = rgb[2].powf(2.2);
            }
            rgb
        })
        .collect::<Vec<_>>();
    let mut mip_bytes = Vec::new();
    let mut mip_width = width;
    let mut mip_height = height;
    loop {
        let mut bytes = Vec::with_capacity(pixels.len() * 8);
        for pixel in &pixels {
            for component in pixel {
                bytes.extend_from_slice(&f16::from_f32(*component).to_bits().to_ne_bytes());
            }
        }
        mip_bytes.push(bytes);
        if mip_width == 1 && mip_height == 1 {
            break;
        }
        let next_width = (mip_width / 2).max(1);
        let next_height = (mip_height / 2).max(1);
        let mut next = vec![[0.0; 4]; (next_width * next_height) as usize];
        for y in 0..next_height {
            for x in 0..next_width {
                let mut sum = [0.0; 4];
                let mut samples = 0.0;
                for oy in 0..2 {
                    for ox in 0..2 {
                        let sx = (x * 2 + ox).min(mip_width - 1);
                        let sy = (y * 2 + oy).min(mip_height - 1);
                        let sample = pixels[(sy * mip_width + sx) as usize];
                        for channel in 0..4 {
                            sum[channel] += sample[channel];
                        }
                        samples += 1.0;
                    }
                }
                next[(y * next_width + x) as usize] = sum.map(|value| value / samples);
            }
        }
        pixels = next;
        mip_width = next_width;
        mip_height = next_height;
    }
    let mut hasher = DefaultHasher::new();
    width.hash(&mut hasher);
    height.hash(&mut hasher);
    for bytes in &mip_bytes {
        bytes.hash(&mut hasher);
    }
    Ok(WorldEnvironmentImage {
        width,
        height,
        mip_bytes,
        signature: hasher.finish(),
    })
}
