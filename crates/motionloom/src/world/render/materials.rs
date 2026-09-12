// =========================================
// =========================================
// crates/motionloom/src/world/render/materials.rs

//! Retained, semantic mip chains shared by native and browser uploads.

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub(super) enum TextureRole {
    Color,
    Normal,
    #[default]
    Data,
}

pub(super) fn srgb_decode(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_encode(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

pub(super) fn build_mips(
    width: u32,
    height: u32,
    rgba: &[u8],
    role: TextureRole,
) -> Vec<(u32, u32, Vec<u8>)> {
    let mut levels = vec![(width, height, rgba.to_vec())];
    while levels.last().is_some_and(|(w, h, _)| *w > 1 || *h > 1) {
        let (w, h, pixels) = levels.last().unwrap();
        let nw = (*w / 2).max(1);
        let nh = (*h / 2).max(1);
        let mut next = vec![0; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                // Proportional footprints retain the last row/column of odd images.
                let mut sum = [0.0_f32; 4];
                let mut count = 0.0;
                let mut coverage = 0.0;
                for sy in y * h / nh..(y + 1) * h / nh {
                    for sx in x * w / nw..(x + 1) * w / nw {
                        let p = &pixels[((sy * w + sx) * 4) as usize..][..4];
                        let a = p[3] as f32 / 255.0;
                        for c in 0..3 {
                            let v = p[c] as f32 / 255.0;
                            sum[c] += match role {
                                TextureRole::Color => srgb_decode(v) * a,
                                TextureRole::Normal => v * 2.0 - 1.0,
                                TextureRole::Data => v,
                            };
                        }
                        sum[3] += a;
                        coverage += a;
                        count += 1.0;
                    }
                }
                let length = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
                for c in 0..4 {
                    let v = if c == 3 {
                        sum[c] / count
                    } else {
                        match role {
                            TextureRole::Color => srgb_encode(sum[c] / coverage.max(0.000001)),
                            TextureRole::Normal if length > 0.000001 => sum[c] / length * 0.5 + 0.5,
                            TextureRole::Normal => {
                                if c == 2 {
                                    1.0
                                } else {
                                    0.5
                                }
                            }
                            TextureRole::Data => sum[c] / count,
                        }
                    };
                    next[((y * nw + x) * 4) as usize + c] =
                        (v * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        levels.push((nw, nh, next));
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transparent_color_does_not_bleed_into_visible_mips() {
        let pixels = [255, 0, 0, 255, 0, 0, 255, 0];
        assert_eq!(
            build_mips(2, 1, &pixels, TextureRole::Color)[1].2,
            [255, 0, 0, 128]
        );
    }

    #[test]
    fn odd_edge_pixels_contribute_to_mips() {
        let mut pixels = [0, 0, 0, 255].repeat(15);
        pixels[14 * 4] = 255;
        let mips = build_mips(5, 3, &pixels, TextureRole::Data);
        assert!(mips.last().unwrap().2[0] > 0);
    }

    #[test]
    fn color_mips_average_light_and_data_mips_average_numbers() {
        let pixels = [0, 0, 0, 255, 255, 255, 255, 255];
        assert_eq!(build_mips(2, 1, &pixels, TextureRole::Data)[1].2[0], 128);
        assert_eq!(build_mips(2, 1, &pixels, TextureRole::Color)[1].2[0], 188);
    }
    #[test]
    fn odd_dimensions_and_normal_length_survive_reduction() {
        let pixels = [128, 128, 255, 255].repeat(15);
        let mips = build_mips(5, 3, &pixels, TextureRole::Normal);
        assert_eq!(
            mips.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>(),
            vec![(5, 3), (2, 1), (1, 1)]
        );
        assert_eq!(&mips[2].2, &[128, 128, 255, 255]);
    }
}
