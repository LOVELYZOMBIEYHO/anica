// =========================================
// =========================================
// crates/motionloom/src/scene/compositor/gpu.rs

use std::sync::Arc;

use thiserror::Error;

use super::{AlphaMode, RequiredAovs, SourceColorSpace};

/// One GPU owner is injected into all renderers and the compositor.
#[derive(Clone)]
pub struct SceneGpuContext {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

impl SceneGpuContext {
    pub async fn request(label: &str) -> Result<Self, SceneGpuContextError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|error| SceneGpuContextError::Adapter(error.to_string()))?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(label),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| SceneGpuContextError::Device(error.to_string()))?;
        Ok(Self {
            device: Arc::new(device),
            queue,
            adapter_info: adapter.get_info(),
        })
    }

    pub fn from_device(
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    ) -> Self {
        Self {
            device,
            queue,
            adapter_info,
        }
    }

    pub fn device(&self) -> Arc<wgpu::Device> {
        self.device.clone()
    }

    pub fn queue(&self) -> wgpu::Queue {
        self.queue.clone()
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }
}

/// GPU textures are execution bindings and intentionally stay outside the IR.
pub struct GpuLayerTexture {
    pub texture: Arc<wgpu::Texture>,
    pub size: [u32; 2],
    pub format: wgpu::TextureFormat,
    pub color_space: SourceColorSpace,
    pub alpha_mode: AlphaMode,
}

impl GpuLayerTexture {
    pub fn view(&self) -> wgpu::TextureView {
        self.texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }
}

#[derive(Default)]
pub struct GpuLayerSet {
    pub beauty: Option<GpuLayerTexture>,
    pub coverage: Option<GpuLayerTexture>,
    pub depth: Option<GpuLayerTexture>,
    pub motion: Option<GpuLayerTexture>,
    pub normal: Option<GpuLayerTexture>,
    pub albedo: Option<GpuLayerTexture>,
}

impl GpuLayerSet {
    pub fn available_aovs(&self) -> RequiredAovs {
        RequiredAovs {
            coverage: self.coverage.is_some(),
            depth: self.depth.is_some(),
            motion: self.motion.is_some(),
            normal: self.normal.is_some(),
            albedo: self.albedo.is_some(),
        }
    }
}

#[derive(Debug, Error)]
pub enum SceneGpuContextError {
    #[error("no compatible GPU adapter: {0}")]
    Adapter(String),
    #[error("GPU device request failed: {0}")]
    Device(String),
    #[error("GPU texture transfer failed: {0}")]
    Transfer(String),
}
