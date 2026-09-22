use image::RgbaImage;

use super::{AlphaMode, SourceColorSpace};
use super::{ColorStage, RenderDomain, SceneCompositionError, to_linear_premultiplied};

/// CPU-visible form of the compositor's canonical RGBA16F working image.
///
/// Values remain f32 here so no precision is lost before upload. The shared
/// GPU executor stores and composites them as linear premultiplied RGBA16F.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearPremultipliedImage {
    pub size: [u32; 2],
    pub pixels: Vec<[f32; 4]>,
}

impl LinearPremultipliedImage {
    pub fn new(size: [u32; 2], pixels: Vec<[f32; 4]>) -> Result<Self, SceneCompositionError> {
        let expected = size[0] as usize * size[1] as usize;
        if size.contains(&0) {
            return Err(SceneCompositionError::EmptyOutput);
        }
        if pixels.len() != expected {
            return Err(SceneCompositionError::InvalidPixelCount {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self { size, pixels })
    }

    /// Convert Raster's byte surface exactly once at the renderer boundary.
    pub fn from_srgb_straight(image: &RgbaImage) -> Self {
        let pixels = image
            .pixels()
            .map(|pixel| {
                to_linear_premultiplied(
                    pixel.0.map(|channel| channel as f32 / 255.0),
                    SourceColorSpace::Srgb,
                    AlphaMode::Straight,
                )
            })
            .collect();
        Self {
            size: [image.width(), image.height()],
            pixels,
        }
    }

    /// Build an opaque scene-linear base from Weaver beauty radiance.
    pub fn from_opaque_rgb(
        size: [u32; 2],
        rgb: &[[f32; 3]],
    ) -> Result<Self, SceneCompositionError> {
        Self::new(
            size,
            rgb.iter()
                .map(|color| [color[0], color[1], color[2], 1.0])
                .collect(),
        )
    }
}

/// One evaluated image-plane run bound to the renderer-independent plan.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedCompositionLayer {
    pub id: String,
    pub order: i32,
    pub domain: RenderDomain,
    pub color_stage: ColorStage,
    pub image: LinearPremultipliedImage,
}

/// Both persistent color stages produced by one compositor execution.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositedFrame {
    /// HDR result before tone mapping and display-referred Screen/Lens layers.
    pub scene_linear: LinearPremultipliedImage,
    /// Tone-mapped, still-float result including Screen/Lens layers.
    pub display_linear: LinearPremultipliedImage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_boundary_decodes_and_premultiplies() {
        let source = RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 128]));
        let converted = LinearPremultipliedImage::from_srgb_straight(&source);
        assert!((converted.pixels[0][0] - 128.0 / 255.0).abs() < 0.0001);
        assert_eq!(converted.pixels[0][1], 0.0);
        assert_eq!(converted.pixels[0][3], 128.0 / 255.0);
    }
}
