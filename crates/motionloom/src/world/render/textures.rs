// =========================================
// =========================================
// crates/motionloom/src/world/render/textures.rs

//! Convert Scene raster images into the orientation and storage used by world materials.

use super::{GpuWorldTexture, RgbaImage};
use std::sync::Arc;

pub(super) fn gpu_world_texture_from_image(image: &RgbaImage) -> GpuWorldTexture {
    let width = image.width().max(1);
    let height = image.height().max(1);
    let row_bytes = width as usize * 4;
    let mut rgba = Vec::with_capacity(row_bytes * height as usize);
    // Scene rasters use a top-left origin while glTF UVs use a bottom-left
    // texture origin. Flip only live Scene bindings; embedded GLB images keep
    // their authored orientation.
    for row in (0..height as usize).rev() {
        let start = row * row_bytes;
        rgba.extend_from_slice(&image.as_raw()[start..start + row_bytes]);
    }
    GpuWorldTexture::new(width, height, rgba)
}

pub(crate) fn gpu_world_texture_from_rgba_image(image: &RgbaImage) -> Arc<GpuWorldTexture> {
    Arc::new(gpu_world_texture_from_image(image))
}
