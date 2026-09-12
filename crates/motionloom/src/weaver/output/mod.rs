// =========================================
// =========================================
// crates/motionloom/src/weaver/output/mod.rs

use crate::weaver::WeaverError;
use std::path::Path;

pub(crate) fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Denoising changes beauty only; keep original auxiliary passes at the job root.
pub(crate) fn save_denoised(
    dir: &Path,
    size: [u32; 2],
    radiance: Vec<f32>,
    lighting: &crate::world::WorldLighting,
) -> Result<(), WeaverError> {
    let beauty = image::Rgb32FImage::from_raw(size[0], size[1], radiance)
        .ok_or_else(|| WeaverError::Invalid("denoised image dimensions mismatch".into()))?;
    std::fs::create_dir_all(dir)?;
    let mut display = image::RgbImage::new(size[0], size[1]);
    for (x, y, pixel) in beauty.enumerate_pixels() {
        display.put_pixel(x, y, image::Rgb(super::color::display(pixel.0, lighting)));
    }
    beauty.save(dir.join("beauty.exr"))?;
    display.save(dir.join("display.png"))?;
    Ok(())
}

/// Preserve scene-linear radiance; apply display conversion only to the PNG.
pub(crate) fn save(
    dir: &Path,
    size: [u32; 2],
    film: &[f32],
    lighting: &crate::world::WorldLighting,
) -> Result<(), WeaverError> {
    std::fs::create_dir_all(dir)?;
    let mut beauty = image::Rgb32FImage::new(size[0], size[1]);
    let mut albedo = beauty.clone();
    let mut normal = beauty.clone();
    let mut depth = beauty.clone();
    let mut variance = beauty.clone();
    let mut counts = beauty.clone();
    let mut display = image::RgbImage::new(size[0], size[1]);
    for (i, f) in film.chunks_exact(16).enumerate() {
        let x = i as u32 % size[0];
        let y = i as u32 / size[0];
        let n = f[3].max(1.0);
        let rgb = [f[0] / n, f[1] / n, f[2] / n];
        beauty.put_pixel(x, y, image::Rgb(rgb));
        albedo.put_pixel(x, y, image::Rgb([f[8] / n, f[9] / n, f[10] / n]));
        normal.put_pixel(x, y, image::Rgb([f[12] / n, f[13] / n, f[14] / n]));
        depth.put_pixel(x, y, image::Rgb([f[15] / n; 3]));
        variance.put_pixel(x, y, image::Rgb([f[5] / (n - 1.0).max(1.0); 3]));
        counts.put_pixel(x, y, image::Rgb([f[3]; 3]));
        display.put_pixel(x, y, image::Rgb(super::color::display(rgb, lighting)));
    }
    beauty.save(dir.join("beauty.exr"))?;
    display.save(dir.join("display.png"))?;
    albedo.save(dir.join("albedo.exr"))?;
    normal.save(dir.join("normal.exr"))?;
    depth.save(dir.join("depth.exr"))?;
    variance.save(dir.join("variance.exr"))?;
    counts.save(dir.join("sample-count.exr"))?;
    Ok(())
}
