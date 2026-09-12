// =========================================
// =========================================
// crates/motionloom/src/world/render/resources.rs

//! GPU allocation sizing and alignment shared by retained render resources.

use super::{GPU_WORLD_DEFAULT_CHUNK_MIB, GPU_WORLD_VERTEX_STRIDE_BYTES};

pub(super) fn gpu_world_vertex_chunk_bytes(device: &wgpu::Device) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    let requested_mib = std::env::var("MOTIONLOOM_VERTEX_CHUNK_MIB")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(GPU_WORLD_DEFAULT_CHUNK_MIB);
    #[cfg(target_arch = "wasm32")]
    let requested_mib = GPU_WORLD_DEFAULT_CHUNK_MIB;

    let requested = requested_mib.saturating_mul(1024 * 1024);
    let device_limit = device.limits().max_buffer_size;
    requested
        .min(device_limit)
        .max((GPU_WORLD_VERTEX_STRIDE_BYTES * 3) as u64) as usize
}

pub(super) fn align_to_256(v: u32) -> u32 {
    const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    v.div_ceil(ALIGN) * ALIGN
}
