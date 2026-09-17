// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/mod.rs

//! Embedded WGSL sources used to build the shared world-render pipelines.

/// Assemble one shader module so all shading modes retain the same bindings
/// and pipeline layout while each shader responsibility stays reviewable.
pub(super) static WGPU_WORLD_SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    [
        include_str!("bindings.wgsl"),
        include_str!("environment.wgsl"),
        include_str!("shadows.wgsl"),
        include_str!("geometry.wgsl"),
        include_str!("outline.wgsl"),
        include_str!("shading/physical.wgsl"),
        include_str!("shading/stylized.wgsl"),
        include_str!("shading/toon.wgsl"),
        include_str!("shading/cel.wgsl"),
        include_str!("shading/clay.wgsl"),
        include_str!("lighting.wgsl"),
        include_str!("color.wgsl"),
        include_str!("fog.wgsl"),
        include_str!("surface.wgsl"),
    ]
    .concat()
});
pub(super) const WGPU_WORLD_DOF_SHADER: &str = concat!(
    include_str!("dof.wgsl"),
    "\n",
    include_str!("presets/filmic_bokeh_v1.wgsl"),
    "\n",
    include_str!("presets/cinematic_bokeh_v1.wgsl"),
    "\n",
    include_str!("presets/ink_wash_soft_v1.wgsl"),
    "\n",
    include_str!("presets/pbr_npr_soft_v1.wgsl"),
    "\n",
    include_str!("universal_style.wgsl"),
);
pub(super) const WGPU_GROUND_GRID_SHADER: &str = include_str!("ground_grid.wgsl");
pub(super) const WGPU_FROXEL_INJECT_SHADER: &str = include_str!("volumetric_inject.wgsl");
pub(super) const WGPU_FROXEL_INTEGRATE_SHADER: &str = include_str!("volumetric_integrate.wgsl");
pub(super) const WGPU_FROXEL_COMPOSITE_SHADER: &str = include_str!("volumetric_composite.wgsl");
