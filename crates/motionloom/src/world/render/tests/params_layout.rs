// =========================================
// =========================================
// crates/motionloom/src/world/render/tests/params_layout.rs

//! Locks CPU-packed byte lengths to the corresponding WGSL uniform layouts.

#[test]
fn world_and_grid_parameter_sizes_match_wgsl_layouts() {
    let world =
        super::super::params::pack_gpu_world_params(super::super::GpuWorldParams::default());
    let grid = super::super::params::pack_ground_grid_params(super::super::GpuGroundGridParams {
        canvas: [0.0; 4],
        camera0: [0.0; 4],
        camera1: [0.0; 4],
        camera2: [0.0; 4],
        camera3: [0.0; 4],
        options: [0.0; 4],
    });
    assert_eq!(world.len(), 33 * 16);
    assert_eq!(grid.len(), 8 * 16);
}
