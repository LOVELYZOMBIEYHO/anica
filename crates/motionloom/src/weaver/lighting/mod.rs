// =========================================
// =========================================
// crates/motionloom/src/weaver/lighting/mod.rs

use crate::weaver::WeaverError;
use std::path::Path;

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
        [start as f32, w as f32, h as f32, 0.0],
        [cdf_start as f32, cw as f32, ch as f32, 0.0],
    ))
}
