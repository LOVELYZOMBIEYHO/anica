// =========================================
// =========================================
// crates/motionloom/src/weaver/backend/wgpu/mod.rs

use crate::{
    scene::compositor::SceneGpuContext,
    weaver::{WeaverError, geometry::PackedScene},
};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub(crate) const FILM_FLOATS_PER_PIXEL: usize = 20;
pub(crate) const FILM_BYTES_PER_PIXEL: usize = FILM_FLOATS_PER_PIXEL * 4;
pub(crate) type CameraParams = [[f32; 4]; 26];
pub(crate) const CAMERA_UNIFORM_BYTES: u64 = std::mem::size_of::<CameraParams>() as u64;

pub(crate) fn bytes(values: &[[f32; 4]]) -> Vec<u8> {
    values
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn floats_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn floats(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four byte chunk")))
        .collect()
}

pub(crate) struct Gpu {
    context: SceneGpuContext,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    scene: wgpu::Buffer,
    textures: wgpu::Buffer,
    denoise_seed: wgpu::ComputePipeline,
    denoise_filter: wgpu::ComputePipeline,
    denoise_uniform_layout: wgpu::BindGroupLayout,
    denoise_seed_layout: wgpu::BindGroupLayout,
    denoise_filter_layout: wgpu::BindGroupLayout,
    denoise_unused_layout: wgpu::BindGroupLayout,
    pub name: String,
}

impl Gpu {
    pub async fn new(scene: &PackedScene) -> Result<Self, WeaverError> {
        let context = SceneGpuContext::request("MotionLoom Weaver")
            .await
            .map_err(|error| WeaverError::Gpu(error.to_string()))?;
        Self::new_with_context(scene, context).await
    }

    /// Shared construction keeps Weaver output on the compositor's device.
    pub async fn new_with_context(
        scene: &PackedScene,
        context: SceneGpuContext,
    ) -> Result<Self, WeaverError> {
        let device = context.device();
        let queue = context.queue();
        let limits = device.limits();
        for n in [scene.data.len() * 16, scene.pixels.len() * 4] {
            if n as u64 > limits.max_buffer_size
                || n as u64 > limits.max_storage_buffer_binding_size as u64
            {
                return Err(WeaverError::Unsupported(format!(
                    "scene buffer {n} exceeds adapter storage limit"
                )));
            }
        }
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
        let denoise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Weaver denoiser"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/denoise.wgsl").into()),
        });
        // Explicit layouts keep the two entry points from inheriting each
        // other's bind groups through auto-generated pipeline layouts.
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let denoise_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Weaver denoise uniform layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let denoise_seed_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Weaver denoise seed layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, false),
                ],
            });
        let denoise_filter_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Weaver denoise filter layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                ],
            });
        let denoise_seed_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Weaver denoise seed pipeline layout"),
                bind_group_layouts: &[&denoise_uniform_layout, &denoise_seed_layout],
                push_constant_ranges: &[],
            });
        let denoise_unused_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Weaver denoise unused layout"),
                entries: &[],
            });
        let denoise_filter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Weaver denoise filter pipeline layout"),
                // Group 1 belongs to the seed entry point and stays empty here.
                bind_group_layouts: &[
                    &denoise_uniform_layout,
                    &denoise_unused_layout,
                    &denoise_filter_layout,
                ],
                push_constant_ranges: &[],
            });
        let denoise_seed = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Weaver denoise seed"),
            layout: Some(&denoise_seed_pipeline_layout),
            module: &denoise_shader,
            entry_point: Some("denoise_seed"),
            compilation_options: Default::default(),
            cache: None,
        });
        let denoise_filter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Weaver denoise filter"),
            layout: Some(&denoise_filter_pipeline_layout),
            module: &denoise_shader,
            entry_point: Some("denoise_wavelet"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = device.pop_error_scope().await {
            return Err(WeaverError::Gpu(e.to_string()));
        }
        let geometry = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weaver geometry"),
            contents: &bytes(&scene.data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let texture_bytes: Vec<u8> = scene.pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
        let textures = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weaver textures"),
            contents: &texture_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        Ok(Self {
            context: context.clone(),
            device,
            queue,
            pipeline,
            scene: geometry,
            textures,
            denoise_seed,
            denoise_filter,
            denoise_uniform_layout,
            denoise_seed_layout,
            denoise_filter_layout,
            denoise_unused_layout,
            name: context.adapter_info().name.clone(),
        })
    }

    pub fn context(&self) -> SceneGpuContext {
        self.context.clone()
    }

    /// Refresh dynamic scene payloads without rebuilding the Metal device,
    /// shader pipelines or buffer bindings when the allocation still fits.
    pub fn upload_scene(&self, scene: &PackedScene, upload_textures: bool) -> bool {
        let geometry = bytes(&scene.data);
        let texture_bytes = scene.pixels.len() as u64 * std::mem::size_of::<u32>() as u64;
        if geometry.len() as u64 > self.scene.size() || texture_bytes > self.textures.size() {
            return false;
        }
        self.queue.write_buffer(&self.scene, 0, &geometry);
        if upload_textures {
            let textures = scene
                .pixels
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            self.queue.write_buffer(&self.textures, 0, &textures);
        }
        true
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
            size: (count * FILM_BYTES_PER_PIXEL) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver camera"),
            size: CAMERA_UNIFORM_BYTES,
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

    #[cfg(test)]
    pub fn batch(&self, tile: &Tile, params: &CameraParams) -> Result<Vec<u8>, WeaverError> {
        self.dispatch(tile, params)?;
        self.read(tile)
    }

    /// Submit one sample batch without waiting for the GPU.
    ///
    /// Preview sessions chain several dispatches and read once, which amortizes
    /// the blocking readback for every batch while leaving `batch` semantics
    /// unchanged for the final render path.
    pub fn dispatch(&self, tile: &Tile, params: &CameraParams) -> Result<(), WeaverError> {
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
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Submit one sampling round for every active tile in a single command
    /// buffer. This removes one queue submit and one CPU/GPU synchronization
    /// boundary per tile during sequence export.
    pub fn dispatch_all(&self, tiles: &[(&Tile, &CameraParams)]) -> Result<(), WeaverError> {
        if tiles.is_empty() {
            return Ok(());
        }
        for (tile, params) in tiles {
            self.queue.write_buffer(&tile.uniform, 0, &bytes(*params));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Weaver sequence sample round"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (tile, params) in tiles {
                pass.set_bind_group(0, &tile.group, &[]);
                pass.dispatch_workgroups(
                    (params[0][0] as u32).div_ceil(8),
                    (params[0][1] as u32).div_ceil(8),
                    1,
                );
            }
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Copy one tile film to CPU memory after previous dispatches completed.
    ///
    /// `PollType::Wait` was measured at ~2 s per tile on Metal even when the GPU
    /// work finished in under a millisecond, so the readback spins on the
    /// non-blocking poll instead. A generous timeout still surfaces a lost or
    /// hung device as an error.
    #[cfg(test)]
    pub fn read(&self, tile: &Tile) -> Result<Vec<u8>, WeaverError> {
        let debug = std::env::var_os("WEAVER_READ_DEBUG").is_some();
        let started = std::time::Instant::now();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &tile.film,
            0,
            &tile.staging,
            0,
            (tile.count * FILM_BYTES_PER_PIXEL) as u64,
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = tile.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let deadline = started + std::time::Duration::from_secs(60);
        let mut spins = 0u32;
        let result = loop {
            // Poll is comparatively expensive; drive it sparsely and busy-wait
            // between calls. Sleeping here is timer-throttled on macOS (App Nap),
            // which delayed a sub-millisecond copy by more than a second.
            if spins % 64 == 0 {
                self.device
                    .poll(wgpu::PollType::Poll)
                    .map_err(|e| WeaverError::Gpu(e.to_string()))?;
            }
            match rx.try_recv() {
                Ok(result) => break result,
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(WeaverError::Gpu("tile readback timed out".into()));
                    }
                    spins = spins.wrapping_add(1);
                    std::hint::spin_loop();
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(WeaverError::Gpu("tile readback channel closed".into()));
                }
            }
        };
        result.map_err(|e| WeaverError::Gpu(e.to_string()))?;
        let data = slice.get_mapped_range().to_vec();
        tile.staging.unmap();
        if debug {
            eprintln!("read: {:.2} ms", started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(data)
    }

    /// Copy and map several tile films with a single submit and a single wait.
    ///
    /// All copies share one command buffer, and the spin drives every outstanding
    /// map callback together, so readback cost does not scale with tile count.
    pub fn read_all(&self, tiles: &[&Tile]) -> Result<Vec<Vec<u8>>, WeaverError> {
        let debug = std::env::var_os("WEAVER_READ_DEBUG").is_some();
        let started = std::time::Instant::now();
        if tiles.is_empty() {
            return Ok(Vec::new());
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for tile in tiles {
            encoder.copy_buffer_to_buffer(
                &tile.film,
                0,
                &tile.staging,
                0,
                (tile.count * FILM_BYTES_PER_PIXEL) as u64,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        let mut receivers = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let (tx, rx) = std::sync::mpsc::channel();
            tile.staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            receivers.push(rx);
        }
        let mut pending = (0..tiles.len()).collect::<Vec<_>>();
        let mut results: Vec<Option<Result<(), String>>> = vec![None; tiles.len()];
        let deadline = started + std::time::Duration::from_secs(60);
        let mut spins = 0u32;
        let mut polls = 0u32;
        let mut poll_ms = 0.0f64;
        let mut spins_until_first = 0u32;
        while !pending.is_empty() {
            if spins % 64 == 0 {
                let poll_started = std::time::Instant::now();
                self.device
                    .poll(wgpu::PollType::Poll)
                    .map_err(|e| WeaverError::Gpu(e.to_string()))?;
                poll_ms += poll_started.elapsed().as_secs_f64() * 1000.0;
                polls += 1;
            }
            pending.retain(|index| match receivers[*index].try_recv() {
                Ok(result) => {
                    if spins_until_first == 0 {
                        spins_until_first = spins;
                    }
                    results[*index] = Some(result.map_err(|e| e.to_string()));
                    false
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => true,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    results[*index] = Some(Err("tile readback channel closed".into()));
                    false
                }
            });
            if pending.is_empty() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                return Err(WeaverError::Gpu("tile readback timed out".into()));
            }
            spins = spins.wrapping_add(1);
            std::hint::spin_loop();
        }
        let mut output = Vec::with_capacity(tiles.len());
        for (index, tile) in tiles.iter().enumerate() {
            match results[index].take() {
                Some(Ok(())) => {}
                Some(Err(message)) => return Err(WeaverError::Gpu(message)),
                None => return Err(WeaverError::Gpu("tile readback incomplete".into())),
            }
            output.push(tile.staging.slice(..).get_mapped_range().to_vec());
        }
        for tile in tiles {
            tile.staging.unmap();
        }
        if debug {
            eprintln!(
                "read_all: {} tiles {:.2} ms | polls {polls} poll {poll_ms:.1} ms spins-until-first {spins_until_first}",
                tiles.len(),
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(output)
    }

    /// Denoise one assembled film buffer on the GPU.
    ///
    /// The pass reads the film AOVs (color, albedo, normal, depth, variance)
    /// and returns display-referred radiance with three floats per pixel.
    pub fn denoise(
        &self,
        width: u32,
        height: u32,
        film: &[f32],
        passes: u32,
    ) -> Result<Vec<f32>, WeaverError> {
        let count = width as usize * height as usize;
        if count == 0 {
            return Ok(Vec::new());
        }
        if film.len() < count * FILM_FLOATS_PER_PIXEL {
            return Err(WeaverError::Invalid("denoise film is too small".into()));
        }
        let film_bytes = (count * FILM_BYTES_PER_PIXEL) as u64;
        let plane_bytes = (count * 4 * 4) as u64;
        let device = &self.device;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let film_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver denoise film"),
            size: film_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&film_buffer, 0, &floats_bytes(film));
        let color_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver denoise color a"),
            size: plane_bytes,
            usage,
            mapped_at_creation: false,
        });
        let color_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver denoise color b"),
            size: plane_bytes,
            usage,
            mapped_at_creation: false,
        });
        let plane_normal = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver denoise normal"),
            size: plane_bytes,
            usage,
            mapped_at_creation: false,
        });
        let plane_albedo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver denoise albedo"),
            size: plane_bytes,
            usage,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver denoise params"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Weaver denoise params"),
            layout: &self.denoise_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let unused_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Weaver denoise unused"),
            layout: &self.denoise_unused_layout,
            entries: &[],
        });
        let seed_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Weaver denoise seed"),
            layout: &self.denoise_seed_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: film_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: color_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: plane_normal.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: plane_albedo.as_entire_binding(),
                },
            ],
        });
        let filter_layout = self.denoise_filter_layout.clone();
        let make_filter_group =
            |label: &str, src: &wgpu::Buffer, dst: &wgpu::Buffer| -> wgpu::BindGroup {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &filter_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: plane_normal.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: plane_albedo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dst.as_entire_binding(),
                        },
                    ],
                })
            };
        let group_ab = make_filter_group("Weaver denoise a->b", &color_a, &color_b);
        let group_ba = make_filter_group("Weaver denoise b->a", &color_b, &color_a);

        let groups_x = width.div_ceil(8);
        let groups_y = height.div_ceil(8);
        let seed_params = [width as f32, height as f32, 0.0, 0.0];
        self.queue
            .write_buffer(&uniform, 0, &floats_bytes(&seed_params));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Weaver denoise seed pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.denoise_seed);
            pass.set_bind_group(0, &uniform_group, &[]);
            pass.set_bind_group(1, &seed_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        let mut source_is_a = true;
        let filter_params = [width as f32, height as f32, 0.0, 0.0, 32.0, 0.08, 0.2, 4.0];
        for index in 0..passes.max(1) {
            let step = (1u32 << index.min(7)) as f32;
            let mut params = filter_params;
            params[2] = step;
            self.queue.write_buffer(&uniform, 0, &floats_bytes(&params));
            let group = if source_is_a { &group_ab } else { &group_ba };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Weaver denoise filter pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.denoise_filter);
            pass.set_bind_group(0, &uniform_group, &[]);
            pass.set_bind_group(1, &unused_group, &[]);
            pass.set_bind_group(2, group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
            source_is_a = !source_is_a;
        }
        let final_color = if source_is_a { &color_a } else { &color_b };
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weaver denoise readback"),
            size: plane_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(final_color, 0, &staging, 0, plane_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut spins = 0u32;
        let result = loop {
            if spins % 64 == 0 {
                device
                    .poll(wgpu::PollType::Poll)
                    .map_err(|e| WeaverError::Gpu(e.to_string()))?;
            }
            match rx.try_recv() {
                Ok(result) => break result,
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(WeaverError::Gpu("denoise readback timed out".into()));
                    }
                    spins = spins.wrapping_add(1);
                    std::hint::spin_loop();
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(WeaverError::Gpu("denoise readback channel closed".into()));
                }
            }
        };
        result.map_err(|e| WeaverError::Gpu(e.to_string()))?;
        let pixels = {
            let data = slice.get_mapped_range();
            floats(data.as_ref())
        };
        staging.unmap();
        let mut output = Vec::with_capacity(count * 3);
        for pixel in pixels.chunks_exact(4) {
            output.extend_from_slice(&pixel[..3]);
        }
        Ok(output)
    }
}

pub(crate) struct Tile {
    film: wgpu::Buffer,
    staging: wgpu::Buffer,
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    count: usize,
}
