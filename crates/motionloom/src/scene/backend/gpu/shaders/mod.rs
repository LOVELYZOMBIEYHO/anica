//! WGSL shader sources used by the Scene GPU backend.

pub(crate) const WGPU_SCENE_SHADER: &str = include_str!("scene.wgsl");
pub(crate) const WGPU_MATTE_TEXTURE_SHADER: &str = include_str!("matte_texture.wgsl");
pub(crate) const WGPU_PUPPET_DEFORM_SHADER: &str = include_str!("puppet_deform.wgsl");
#[allow(dead_code)]
pub(crate) const WGPU_SHAPE_SHADER: &str = include_str!("shape.wgsl");
pub(crate) const WGPU_BATCH_SHAPE_SHADER: &str = include_str!("batch_shape.wgsl");
pub(crate) const WGPU_POST_SHADER: &str = include_str!("post.wgsl");
pub(crate) const WGPU_BLOOM_SHADER: &str = include_str!("bloom.wgsl");
pub(crate) const WGPU_DOWNSAMPLE_SHADER: &str = include_str!("downsample.wgsl");
pub(crate) const WGPU_LIGHT_SWEEP_SHADER: &str = include_str!("light_sweep.wgsl");
pub(crate) const WGPU_SPECTRAL_ENERGY_SHADER: &str = include_str!("spectral_energy.wgsl");
