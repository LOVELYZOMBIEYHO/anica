// =========================================
// =========================================
// crates/motionloom/src/weaver/backend/wgpu/mod.rs

use crate::weaver::{WeaverError, geometry::PackedScene};
use wgpu::util::DeviceExt;

pub(crate) fn bytes(values: &[[f32; 4]]) -> Vec<u8> {
    values
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

pub(crate) struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    scene: wgpu::Buffer,
    textures: wgpu::Buffer,
    pub name: String,
}

impl Gpu {
    pub async fn new(scene: &PackedScene) -> Result<Self, WeaverError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| WeaverError::Gpu(e.to_string()))?;
        let limits = adapter.limits();
        for n in [scene.data.len() * 16, scene.pixels.len() * 4] {
            if n as u64 > limits.max_buffer_size
                || n as u64 > limits.max_storage_buffer_binding_size as u64
            {
                return Err(WeaverError::Unsupported(format!(
                    "scene buffer {n} exceeds adapter storage limit"
                )));
            }
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("MotionLoom Weaver"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| WeaverError::Gpu(e.to_string()))?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Weaver path integrator"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/path_trace.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Weaver"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = device.pop_error_scope().await {
            return Err(WeaverError::Gpu(e.to_string()));
        }
        let geometry = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weaver geometry"),
            contents: &bytes(&scene.data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let texture_bytes: Vec<u8> = scene.pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
        let textures = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weaver textures"),
            contents: &texture_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            scene: geometry,
            textures,
            name: adapter.get_info().name,
        })
    }

    /// Tile-local film bounds memory independently of the full output resolution.
    pub fn tile(&self, count: usize, initial: &[u8]) -> Tile {
        let film = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Weaver film"),
                contents: initial,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver readback"),
            size: (count * 64) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver camera"),
            size: 320,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Weaver scene bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scene.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.textures.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: film.as_entire_binding(),
                },
            ],
        });
        Tile {
            film,
            staging,
            uniform,
            group,
            count,
        }
    }

    pub fn batch(&self, tile: &Tile, params: &[[f32; 4]; 20]) -> Result<Vec<u8>, WeaverError> {
        self.queue.write_buffer(&tile.uniform, 0, &bytes(params));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Weaver sample batch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &tile.group, &[]);
            pass.dispatch_workgroups(
                (params[0][0] as u32).div_ceil(8),
                (params[0][1] as u32).div_ceil(8),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&tile.film, 0, &tile.staging, 0, (tile.count * 64) as u64);
        self.queue.submit(Some(encoder.finish()));
        let slice = tile.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| WeaverError::Gpu(e.to_string()))?;
        rx.recv()
            .map_err(|e| WeaverError::Gpu(e.to_string()))?
            .map_err(|e| WeaverError::Gpu(e.to_string()))?;
        let data = slice.get_mapped_range().to_vec();
        tile.staging.unmap();
        Ok(data)
    }
}

pub(crate) struct Tile {
    film: wgpu::Buffer,
    staging: wgpu::Buffer,
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    count: usize,
}
