// =========================================
// =========================================
// crates/motionloom/src/world/render/params.rs

//! Byte-exact packing for Rust parameters mirrored by WGSL uniform layouts.

use super::{GpuGroundGridParams, GpuWorldLightingParams, GpuWorldParams};

pub(super) fn pack_gpu_world_params(params: GpuWorldParams) -> Vec<u8> {
    let mut out = Vec::with_capacity(544);
    for vector in [
        params.canvas,
        params.model,
        params.actor,
        params.actor_rotation,
        params.camera0,
        params.camera1,
        params.camera2,
        params.camera3,
        params.style,
        params.material0,
        params.material1,
        params.material2,
        params.material3,
        params.material4,
        params.material5,
        params.material6,
        params.material7,
        params.material8,
        params.cel_material0,
        params.cel_material1,
        params.vegetation,
        params.hidden0,
        params.hidden1,
        params.hidden2,
        params.hidden3,
        params.hidden4,
        params.hidden5,
        params.hidden6,
        params.hidden7,
        params.previous_model,
        params.previous_actor,
        params.previous_actor_rotation,
        params.previous_vegetation,
        params.motion0,
    ] {
        for value in vector {
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
    out
}

pub(super) fn pack_gpu_world_lighting(params: GpuWorldLightingParams) -> Vec<u8> {
    let mut out = Vec::with_capacity(1168);
    for vector in [
        params.environment0,
        params.environment1,
        params.environment2,
        params.color0,
        params.color1,
        params.fog0,
        params.fog1,
        params.fog2,
        params.fog3,
        params.fog4,
        params.caustics0,
        params.caustics1,
        params.optics0,
        params.dof_style,
        params.render_compat,
        params.camera0,
        params.camera1,
        params.camera2,
        params.camera3,
        params.previous_camera0,
        params.previous_camera1,
        params.previous_camera2,
        params.previous_camera3,
        params.preview0,
        params.preview1,
        params.preview2,
        params.shadow0,
        params.shadow1,
        params.shadow2,
        params.shadow3,
        params.surface0,
        params.surface1,
        params.surface2,
        params.surface3,
        params.cel0,
        params.cel1,
        params.cel2,
        params.universal_color,
        params.universal_tone,
        params.universal_shadow,
        params.universal_highlight,
    ] {
        for value in vector {
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
    for light in params.lights {
        for value in light {
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
    out
}

pub(super) fn pack_ground_grid_params(params: GpuGroundGridParams) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);
    for vector in [
        params.canvas,
        params.camera0,
        params.camera1,
        params.camera2,
        params.camera3,
        params.options,
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ] {
        for value in vector {
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
    out
}
