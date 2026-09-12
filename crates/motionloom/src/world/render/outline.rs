// =========================================
// =========================================
// crates/motionloom/src/world/render/outline.rs

//! Geometry-outline pipeline policy shared by every render style.

pub(super) fn cull_mode(fragment_entry: &str) -> Option<wgpu::Face> {
    if fragment_entry == "fs_outline" {
        Some(wgpu::Face::Front)
    } else {
        None
    }
}
