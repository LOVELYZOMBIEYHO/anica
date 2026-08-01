// =========================================
// crates/motionloom/src/process/wasm_webgpu.rs
// =========================================

use std::borrow::Cow;
use std::sync::Arc;

use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;

use crate::common::gpu_async::{DevicePoller, request_adapter_async, request_device_async};
use crate::dsl::{PassNode, parse_graph_script};
use crate::process::runtime::{compile_runtime_program, eval_time_expr};
use crate::scene::drawable::parse_color;
use crate::scene::render::{SceneRenderProfile, render_scene_graph_frame};

#[derive(Debug, thiserror::Error)]
pub enum ProcessWebGpuRenderError {
    #[error("invalid RGBA buffer: expected {expected} bytes for {width}x{height}, got {actual}")]
    InvalidRgbaBuffer {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
    #[error("WebGPU adapter request failed: {0}")]
    Adapter(String),
    #[error("WebGPU device request failed: {0}")]
    Device(String),
    #[error("canvas surface creation failed: {0}")]
    Surface(String),
    #[error("canvas surface has no supported texture formats")]
    SurfaceFormat,
    #[error("canvas surface frame acquisition failed: {0}")]
    SurfaceFrame(String),
    #[error("unsupported WebGPU process effect: {0}")]
    UnsupportedEffect(String),
    #[error("process pass mask source not found: {0}")]
    MissingMaskSource(String),
    #[error("process pass mask render failed for {id}: {message}")]
    MaskRender { id: String, message: String },
    #[error(transparent)]
    Parse(#[from] crate::error::GraphParseError),
    #[error(transparent)]
    Compile(#[from] crate::error::RuntimeCompileError),
}

pub async fn render_process_frame_to_canvas_gpu(
    script: &str,
    frame: u32,
    width: u32,
    height: u32,
    rgba: &[u8],
    canvas: HtmlCanvasElement,
) -> Result<(), ProcessWebGpuRenderError> {
    let expected = width as usize * height as usize * 4;
    if width == 0 || height == 0 || rgba.len() != expected {
        return Err(ProcessWebGpuRenderError::InvalidRgbaBuffer {
            width,
            height,
            expected,
            actual: rgba.len(),
        });
    }

    let graph = parse_graph_script(script)?;
    compile_runtime_program(graph.clone())?;
    let time_sec = frame as f32 / graph.fps.max(1.0);
    let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / graph.fps.max(1.0));
    let time_norm = (time_sec / duration_sec).clamp(0.0, 1.0);

    let renderer = ProcessWebGpuRenderer::new(width, height).await?;
    renderer
        .render_to_canvas(&graph, frame, time_norm, time_sec, rgba, canvas)
        .await
}

struct ProcessWebGpuRenderer {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    _poller: DevicePoller,
    sampler: wgpu::Sampler,
    pass_bind_group_layout: wgpu::BindGroupLayout,
    pass_pipeline: wgpu::RenderPipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_pipeline: wgpu::RenderPipeline,
    present_bind_group_layout: wgpu::BindGroupLayout,
    present_pipeline_layout: wgpu::PipelineLayout,
    width: u32,
    height: u32,
}

impl ProcessWebGpuRenderer {
    async fn new(width: u32, height: u32) -> Result<Self, ProcessWebGpuRenderError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = request_adapter_async(
            &instance,
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        )
        .await
        .map_err(|err| ProcessWebGpuRenderError::Adapter(err.to_string()))?;
        let (device, queue) = request_device_async(
            &adapter,
            &wgpu::DeviceDescriptor {
                label: Some("anica-motionloom-process-webgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            },
        )
        .await
        .map_err(|err| ProcessWebGpuRenderError::Device(err.to_string()))?;
        let device = Arc::new(device);
        let poller = DevicePoller::start(device.clone());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("anica-motionloom-process-webgpu-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let pass_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-process-webgpu-pass-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let pass_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anica-motionloom-process-webgpu-pass-pipeline-layout"),
            bind_group_layouts: &[&pass_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-process-webgpu-pass-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PROCESS_PASS_SHADER)),
        });
        let pass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("anica-motionloom-process-webgpu-pass-pipeline"),
            layout: Some(&pass_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &pass_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &pass_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-process-webgpu-composite-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-process-webgpu-composite-pipeline-layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-process-webgpu-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPOSITE_SHADER)),
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("anica-motionloom-process-webgpu-composite-pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let present_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-process-webgpu-present-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let present_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-process-webgpu-present-pipeline-layout"),
                bind_group_layouts: &[&present_bind_group_layout],
                push_constant_ranges: &[],
            });

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            _poller: poller,
            sampler,
            pass_bind_group_layout,
            pass_pipeline,
            composite_bind_group_layout,
            composite_pipeline,
            present_bind_group_layout,
            present_pipeline_layout,
            width,
            height,
        })
    }

    async fn render_to_canvas(
        &self,
        graph: &crate::dsl::GraphScript,
        frame: u32,
        time_norm: f32,
        time_sec: f32,
        rgba: &[u8],
        canvas: HtmlCanvasElement,
    ) -> Result<(), ProcessWebGpuRenderError> {
        let passes = &graph.passes;
        let tex_a = self.create_render_texture("anica-motionloom-process-webgpu-tex-a");
        let tex_b = self.create_render_texture("anica-motionloom-process-webgpu-tex-b");
        let tex_backup = self.create_render_texture("anica-motionloom-process-webgpu-tex-backup");
        let white_mask = self.create_render_texture("anica-motionloom-process-webgpu-white-mask");
        self.write_texture_rgba(&tex_a, rgba);
        self.write_texture_rgba(
            &white_mask,
            &solid_rgba(self.width, self.height, [255, 255, 255, 255]),
        );

        let mut current_is_a = true;
        let mut uniform_buffers = Vec::with_capacity(passes.len().saturating_mul(4));
        let mut mask_textures = Vec::<wgpu::Texture>::new();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-motionloom-process-webgpu-encoder"),
            });

        for pass in passes {
            let effect_ids = process_effect_ids(pass)?;
            for effect_id in effect_ids {
                let uniform_buffer = self.make_uniform_buffer(pass, effect_id, time_norm, time_sec);
                let mask_texture = if pass.mask.is_some() {
                    Some(self.create_pass_mask_texture(graph, frame, pass).await?)
                } else {
                    None
                };
                let src = if current_is_a { &tex_a } else { &tex_b };
                let dst = if current_is_a { &tex_b } else { &tex_a };
                let mask = mask_texture.as_ref().unwrap_or(&white_mask);
                if matches!(effect_id, 4 | 14 | 16 | 18) {
                    // Preserve the current image before a bloom/glow stage mutates it.
                    self.copy_texture_to_texture(&mut encoder, src, &tex_backup);
                }
                if matches!(effect_id, 5 | 15 | 17 | 19) {
                    // Composite pass: blend the blurred image (src) with the
                    // preserved original (tex_backup).
                    self.encode_composite_pass(
                        &mut encoder,
                        src,
                        &tex_backup,
                        dst,
                        &uniform_buffer,
                    );
                } else {
                    self.encode_process_pass(&mut encoder, src, dst, mask, &uniform_buffer);
                }
                uniform_buffers.push(uniform_buffer);
                if let Some(mask_texture) = mask_texture {
                    mask_textures.push(mask_texture);
                }
                current_is_a = !current_is_a;
            }
        }

        let final_texture = if current_is_a { &tex_a } else { &tex_b };
        self.present_texture_to_canvas(&mut encoder, final_texture, &canvas)?;
        self.queue.submit([encoder.finish()]);
        drop(uniform_buffers);
        drop(mask_textures);
        Ok(())
    }

    async fn create_pass_mask_texture(
        &self,
        graph: &crate::dsl::GraphScript,
        frame: u32,
        pass: &PassNode,
    ) -> Result<wgpu::Texture, ProcessWebGpuRenderError> {
        let mask_id = pass.mask.as_deref().unwrap_or_default();
        let Some(tex) = graph.textures.iter().find(|tex| tex.id == mask_id) else {
            return Err(ProcessWebGpuRenderError::MissingMaskSource(
                mask_id.to_string(),
            ));
        };
        let Some(from) = tex.from.as_deref() else {
            return Err(ProcessWebGpuRenderError::MissingMaskSource(
                mask_id.to_string(),
            ));
        };
        let Some(scene_id) = from.strip_prefix("scene:") else {
            return Err(ProcessWebGpuRenderError::MissingMaskSource(
                mask_id.to_string(),
            ));
        };

        let mut mask_graph = graph.clone();
        mask_graph.textures.clear();
        mask_graph.passes.clear();
        mask_graph.outputs.clear();
        mask_graph.present.from = format!("scene:{scene_id}");
        let image = render_scene_graph_frame(&mask_graph, frame, SceneRenderProfile::Cpu)
            .await
            .map_err(|err| ProcessWebGpuRenderError::MaskRender {
                id: mask_id.to_string(),
                message: err.to_string(),
            })?;
        let resized = if image.width() == self.width && image.height() == self.height {
            image
        } else {
            image::imageops::resize(
                &image,
                self.width,
                self.height,
                image::imageops::FilterType::Triangle,
            )
        };
        let texture = self.create_render_texture("anica-motionloom-process-webgpu-pass-mask");
        self.write_texture_rgba(&texture, resized.as_raw());
        Ok(texture)
    }

    fn create_render_texture(&self, label: &'static str) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: process_render_texture_usages(),
            view_formats: &[],
        })
    }

    fn write_texture_rgba(&self, texture: &wgpu::Texture, rgba: &[u8]) {
        let row_bytes = self.width * 4;
        const ROW_ALIGNMENT: u32 = 256;
        let padded_row_bytes = row_bytes.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
        let upload: Cow<'_, [u8]> = if padded_row_bytes == row_bytes {
            Cow::Borrowed(rgba)
        } else {
            let mut padded = vec![0u8; padded_row_bytes as usize * self.height as usize];
            for row in 0..self.height as usize {
                let src_start = row * row_bytes as usize;
                let dst_start = row * padded_row_bytes as usize;
                padded[dst_start..dst_start + row_bytes as usize]
                    .copy_from_slice(&rgba[src_start..src_start + row_bytes as usize]);
            }
            Cow::Owned(padded)
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn make_uniform_buffer(
        &self,
        pass: &PassNode,
        effect_id: u32,
        time_norm: f32,
        time_sec: f32,
    ) -> wgpu::Buffer {
        let resolved = crate::process::effect_kind::resolve_process_effect(&pass.effect);
        let (p0, p1, p2, p3, p4) = match resolved {
            Some(crate::process::effect_kind::ProcessEffect::Brightness) => (
                wasm_brightness_amount(pass, time_norm, time_sec).clamp(-1.0, 1.0),
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            Some(crate::process::effect_kind::ProcessEffect::Opacity) => (
                process_param_f32(pass, &["opacity", "alpha", "a"], time_norm, time_sec, 1.0)
                    .clamp(0.0, 1.0),
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            Some(
                crate::process::effect_kind::ProcessEffect::GlowBloom
                | crate::process::effect_kind::ProcessEffect::GlowStack,
            ) => (
                0.0,
                0.0,
                0.0,
                0.0,
                process_bloom_sigma(pass, effect_id, time_norm, time_sec),
            ),
            Some(crate::process::effect_kind::ProcessEffect::ToneMap) => (
                process_param_f32(pass, &["exposure"], time_norm, time_sec, 0.0),
                process_param_f32(pass, &["contrast"], time_norm, time_sec, 1.0),
                process_param_f32(pass, &["shoulder"], time_norm, time_sec, 1.0),
                process_param_f32(pass, &["gamma"], time_norm, time_sec, 2.2),
                process_param_f32(pass, &["saturation"], time_norm, time_sec, 1.0),
            ),
            Some(crate::process::effect_kind::ProcessEffect::LightSweep) => (
                process_param_f32(pass, &["position"], time_norm, time_sec, 0.5),
                process_param_f32(pass, &["angle"], time_norm, time_sec, -18.0),
                process_param_f32(pass, &["width"], time_norm, time_sec, 0.16),
                process_param_f32(pass, &["softness"], time_norm, time_sec, 0.08),
                process_param_f32(pass, &["intensity"], time_norm, time_sec, 1.0),
            ),
            Some(crate::process::effect_kind::ProcessEffect::TextureOverlay) => (
                texture_kind_id(process_param_string(pass, &["kind", "texture"], "paper")),
                process_param_f32(pass, &["scale"], time_norm, time_sec, 42.0),
                process_param_f32(pass, &["strength", "amount"], time_norm, time_sec, 0.25),
                process_param_f32(pass, &["contrast"], time_norm, time_sec, 0.5),
                process_param_f32(pass, &["seed"], time_norm, time_sec, 0.0),
            ),
            Some(crate::process::effect_kind::ProcessEffect::MagnifyLens) => (
                process_param_f32(
                    pass,
                    &["x", "center_x", "centerX"],
                    time_norm,
                    time_sec,
                    0.0,
                ),
                process_param_f32(
                    pass,
                    &["y", "center_y", "centerY"],
                    time_norm,
                    time_sec,
                    0.0,
                ),
                process_param_f32(pass, &["radius"], time_norm, time_sec, 180.0),
                process_param_f32(pass, &["zoom"], time_norm, time_sec, 1.85),
                process_param_f32(pass, &["distortion"], time_norm, time_sec, 0.18),
            ),
            Some(crate::process::effect_kind::ProcessEffect::SpectralEnergy) => (
                process_param_f32(pass, &["active", "mix"], time_norm, time_sec, 1.0)
                    .clamp(0.0, 1.0),
                process_param_f32(pass, &["intensity"], time_norm, time_sec, 1.0).max(0.0),
                process_param_f32(pass, &["density", "grain"], time_norm, time_sec, 1.0).max(0.05),
                process_param_f32(pass, &["drift", "motion"], time_norm, time_sec, 0.0),
                process_param_f32(pass, &["seed"], time_norm, time_sec, 67.0),
            ),
            _ => (
                process_param_f32(pass, &["hue", "h"], time_norm, time_sec, 0.0),
                process_param_f32(pass, &["saturation", "sat", "s"], time_norm, time_sec, 0.0),
                process_param_f32(pass, &["lightness", "lum", "l"], time_norm, time_sec, 0.0),
                process_param_f32(pass, &["alpha", "a"], time_norm, time_sec, 0.0),
                process_param_f32(pass, &["sigma"], time_norm, time_sec, 1.0),
            ),
        };
        let color = process_param_color(pass, &["tint", "color"], [255, 255, 255, 255]);
        let is_magnify_lens = matches!(
            resolved,
            Some(crate::process::effect_kind::ProcessEffect::MagnifyLens)
        );
        let bloom_threshold = process_bloom_threshold(pass, effect_id, time_norm, time_sec);
        let bloom_intensity = process_bloom_intensity(pass, effect_id, time_norm, time_sec);
        let bloom_sigma = process_bloom_sigma(pass, effect_id, time_norm, time_sec);
        let values = [
            self.width as f32,
            self.height as f32,
            effect_id as f32,
            0.0,
            p0,
            p1,
            p2,
            p3,
            p4,
            if is_magnify_lens {
                process_param_f32(pass, &["feather"], time_norm, time_sec, 10.0)
            } else {
                bloom_threshold
            },
            if is_magnify_lens {
                process_param_f32(pass, &["glass"], time_norm, time_sec, 0.32)
            } else {
                bloom_intensity
            },
            bloom_sigma,
            color[0],
            color[1],
            color[2],
            color[3],
            process_param_f32(pass, &["brush_angle", "angle"], time_norm, time_sec, -8.0),
            process_param_f32(
                pass,
                &["bump_strength", "bump", "impasto_strength"],
                time_norm,
                time_sec,
                0.35,
            ),
            process_param_f32(pass, &["relief"], time_norm, time_sec, 0.45),
            pass_mask_mode_id(pass),
        ];
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        self.device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("anica-motionloom-process-webgpu-pass-uniform"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            })
            .tap_mapped_write(&bytes)
    }

    fn encode_process_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        mask: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
    ) {
        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = dst.create_view(&wgpu::TextureViewDescriptor::default());
        let mask_view = mask.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-process-webgpu-pass-bg"),
            layout: &self.pass_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("anica-motionloom-process-webgpu-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pass_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn encode_composite_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        blurred: &wgpu::Texture,
        original: &wgpu::Texture,
        dst: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
    ) {
        let blurred_view = blurred.create_view(&wgpu::TextureViewDescriptor::default());
        let original_view = original.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = dst.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-process-webgpu-composite-bg"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blurred_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&original_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("anica-motionloom-process-webgpu-composite-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn copy_texture_to_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
    ) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn present_texture_to_canvas(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        canvas: &HtmlCanvasElement,
    ) -> Result<(), ProcessWebGpuRenderError> {
        canvas.set_width(self.width);
        canvas.set_height(self.height);

        let surface = self
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|err| ProcessWebGpuRenderError::Surface(err.to_string()))?;
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|format| {
                matches!(
                    format,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .or_else(|| caps.formats.first().copied())
            .ok_or(ProcessWebGpuRenderError::SurfaceFormat)?;
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto)
        };
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            caps.present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::AutoVsync)
        };
        surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: self.width,
                height: self.height,
                present_mode,
                desired_maximum_frame_latency: 2,
                alpha_mode,
                view_formats: vec![],
            },
        );

        let frame = surface
            .get_current_texture()
            .map_err(|err| ProcessWebGpuRenderError::SurfaceFrame(err.to_string()))?;
        let target_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let source_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("anica-motionloom-process-webgpu-present-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PROCESS_PRESENT_SHADER)),
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("anica-motionloom-process-webgpu-present-pipeline"),
                layout: Some(&self.present_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-process-webgpu-present-bg"),
            layout: &self.present_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&source_view),
            }],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("anica-motionloom-process-webgpu-present-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        frame.present();
        Ok(())
    }
}

trait MappedBufferWrite {
    fn tap_mapped_write(self, bytes: &[u8]) -> Self;
}

impl MappedBufferWrite for wgpu::Buffer {
    fn tap_mapped_write(self, bytes: &[u8]) -> Self {
        self.slice(..).get_mapped_range_mut().copy_from_slice(bytes);
        self.unmap();
        self
    }
}

fn process_render_texture_usages() -> wgpu::TextureUsages {
    wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::COPY_DST
}

fn solid_rgba(width: u32, height: u32, color: [u8; 4]) -> Vec<u8> {
    let mut rgba = vec![0; width as usize * height as usize * 4];
    for px in rgba.chunks_exact_mut(4) {
        px.copy_from_slice(&color);
    }
    rgba
}

fn pass_mask_mode_id(pass: &PassNode) -> f32 {
    if pass.mask.is_none() {
        return 0.0;
    }
    let mode = pass.mask_mode.trim().to_ascii_lowercase().replace('_', "-");
    let invert = pass.mask_invert.trim().eq_ignore_ascii_case("true")
        || mode == "inverse"
        || mode == "invert"
        || mode == "inverted"
        || mode == "inverse-luma"
        || mode == "inverted-luma"
        || mode == "inverse-alpha"
        || mode == "inverted-alpha";
    let base = if mode.contains("luma") { 2.0 } else { 1.0 };
    if invert { -base } else { base }
}

fn process_effect_ids(pass: &PassNode) -> Result<Vec<u32>, ProcessWebGpuRenderError> {
    use crate::process::effect_kind::resolve_process_effect;
    match resolve_process_effect(&pass.effect) {
        Some(crate::process::effect_kind::ProcessEffect::HslaOverlay) => Ok(vec![1]),
        Some(crate::process::effect_kind::ProcessEffect::GaussianBlur) => Ok(vec![2, 3]),
        Some(crate::process::effect_kind::ProcessEffect::GaussianBlurHorizontal) => Ok(vec![2]),
        Some(crate::process::effect_kind::ProcessEffect::GaussianBlurVertical) => Ok(vec![3]),
        Some(crate::process::effect_kind::ProcessEffect::GlowBloom) => Ok(vec![4, 2, 3, 5]),
        Some(crate::process::effect_kind::ProcessEffect::GlowStack) => {
            Ok(vec![14, 2, 3, 15, 16, 2, 3, 17, 18, 2, 3, 19])
        }
        Some(crate::process::effect_kind::ProcessEffect::ToneMap) => Ok(vec![6]),
        Some(crate::process::effect_kind::ProcessEffect::LightSweep) => Ok(vec![7]),
        Some(crate::process::effect_kind::ProcessEffect::TextureOverlay) => Ok(vec![8]),
        Some(crate::process::effect_kind::ProcessEffect::MagnifyLens) => Ok(vec![9]),
        Some(crate::process::effect_kind::ProcessEffect::Brightness) => Ok(vec![11]),
        Some(crate::process::effect_kind::ProcessEffect::Opacity) => Ok(vec![12]),
        Some(crate::process::effect_kind::ProcessEffect::SpectralEnergy) => Ok(vec![20]),
        None => Err(ProcessWebGpuRenderError::UnsupportedEffect(
            pass.effect.clone(),
        )),
    }
}

fn pass_has_param(pass: &PassNode, key: &str) -> bool {
    pass.params
        .iter()
        .any(|param| param.key.eq_ignore_ascii_case(key))
}

fn wasm_brightness_amount(pass: &PassNode, time_norm: f32, time_sec: f32) -> f32 {
    if pass_has_param(pass, "amount") {
        process_param_f32(pass, &["amount"], time_norm, time_sec, 0.0)
    } else {
        process_param_f32(pass, &["brightness", "value"], time_norm, time_sec, 0.0)
    }
}

fn process_bloom_threshold(pass: &PassNode, effect_id: u32, time_norm: f32, time_sec: f32) -> f32 {
    let threshold = process_param_f32(
        pass,
        &["threshold", "glowThreshold", "glow_threshold"],
        time_norm,
        time_sec,
        0.72,
    )
    .clamp(0.0, 1.0);
    match effect_id {
        16 | 17 => threshold * 0.85,
        18 | 19 => threshold * 0.65,
        _ => threshold,
    }
}

fn process_bloom_intensity(pass: &PassNode, effect_id: u32, time_norm: f32, time_sec: f32) -> f32 {
    let intensity = process_param_f32(
        pass,
        &[
            "intensity",
            "strength",
            "amount",
            "glowIntensity",
            "glow_intensity",
        ],
        time_norm,
        time_sec,
        1.0,
    )
    .clamp(0.0, 8.0);
    match effect_id {
        14 | 15 => intensity * 0.45,
        16 | 17 => intensity * 0.35,
        18 | 19 => intensity * 0.20,
        _ => intensity,
    }
}

fn process_bloom_sigma(pass: &PassNode, effect_id: u32, time_norm: f32, time_sec: f32) -> f32 {
    match effect_id {
        14 | 15 => process_param_f32(
            pass,
            &["radiusSmall", "radius_small", "small"],
            time_norm,
            time_sec,
            6.0,
        )
        .clamp(0.0, 64.0),
        16 | 17 => process_param_f32(
            pass,
            &["radiusMedium", "radius_medium", "medium"],
            time_norm,
            time_sec,
            18.0,
        )
        .clamp(0.0, 96.0),
        18 | 19 => process_param_f32(
            pass,
            &["radiusLarge", "radius_large", "large"],
            time_norm,
            time_sec,
            48.0,
        )
        .clamp(0.0, 160.0),
        _ => process_param_f32(pass, &["sigma", "radius"], time_norm, time_sec, 18.0)
            .clamp(0.0, 64.0),
    }
}

fn process_param_f32(
    pass: &PassNode,
    keys: &[&str],
    time_norm: f32,
    time_sec: f32,
    fallback: f32,
) -> f32 {
    keys.iter()
        .find_map(|key| {
            pass.params
                .iter()
                .find(|param| param.key.eq_ignore_ascii_case(key))
                .and_then(|param| eval_time_expr(&param.value, time_norm, time_sec).ok())
        })
        .unwrap_or(fallback)
}

fn process_param_color(pass: &PassNode, keys: &[&str], fallback: [u8; 4]) -> [f32; 4] {
    let color = keys
        .iter()
        .find_map(|key| {
            pass.params
                .iter()
                .find(|param| param.key.eq_ignore_ascii_case(key))
                .and_then(|param| {
                    parse_color(param.value.trim().trim_matches('"').trim_matches('\'')).ok()
                })
        })
        .unwrap_or(fallback);
    [
        color[0] as f32 / 255.0,
        color[1] as f32 / 255.0,
        color[2] as f32 / 255.0,
        color[3] as f32 / 255.0,
    ]
}

fn process_param_string<'a>(pass: &'a PassNode, keys: &[&str], fallback: &'a str) -> &'a str {
    keys.iter()
        .find_map(|key| {
            pass.params
                .iter()
                .find(|param| param.key.eq_ignore_ascii_case(key))
                .map(|param| param.value.trim().trim_matches('"').trim_matches('\''))
        })
        .unwrap_or(fallback)
}

fn texture_kind_id(kind: &str) -> f32 {
    match kind
        .trim()
        .to_ascii_lowercase()
        .replace(['-', '_'], "")
        .as_str()
    {
        "noise" => 0.0,
        "paper" | "papergrain" | "papertexture" => 1.0,
        "film" | "filmgrain" | "grain" => 2.0,
        "scanline" | "scanlines" => 3.0,
        "canvas" | "fabric" | "cloth" => 4.0,
        "impasto" | "thickpaint" | "oilpaint" | "oilpainting" => 5.0,
        "brushedpaint" | "brushpaint" | "paintbrush" | "brushed" => 6.0,
        _ => 1.0,
    }
}

impl From<ProcessWebGpuRenderError> for JsValue {
    fn from(err: ProcessWebGpuRenderError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_render_textures_support_copy_for_bloom_backup() {
        let usage = process_render_texture_usages();
        assert!(usage.contains(wgpu::TextureUsages::COPY_SRC));
        assert!(usage.contains(wgpu::TextureUsages::COPY_DST));
        assert!(usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert!(usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    }
}

const PROCESS_PASS_SHADER: &str = concat!(
    include_str!("kernels/color_tone/color_core.wgsl"),
    "\n",
    include_str!("kernels/blur_sharpen_detail/blur_sharpen_detail_gaussian.wgsl"),
    "\n",
    include_str!("kernels/light_atmosphere/light_atmosphere_bloom_prefilter.wgsl"),
    r#"

struct ProcessParams {
    width: f32,
    height: f32,
    effect_id: f32,
    _pad0: f32,
    hue: f32,
    saturation: f32,
    lightness: f32,
    alpha: f32,
    sigma: f32,
    bloom_threshold: f32,
    bloom_intensity: f32,
    bloom_sigma: f32,
    color: vec4<f32>,
    extra: vec4<f32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> params: ProcessParams;
@group(0) @binding(3) var mask_tex: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let pos = positions[vertex_index];
    var out: VertexOut;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

fn ml_process_aces_fitted(rgb: vec3<f32>, shoulder: f32) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59 + clamp(shoulder, 0.0, 2.0) * 0.24;
    let e = 0.14;
    return clamp((rgb * (a * rgb + vec3<f32>(b))) / (rgb * (c * rgb + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn ml_process_hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn ml_process_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (vec2<f32>(3.0) - 2.0 * f);
    return mix(
        mix(ml_process_hash21(i), ml_process_hash21(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(ml_process_hash21(i + vec2<f32>(0.0, 1.0)), ml_process_hash21(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

fn ml_process_fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var amp = 0.5;
    var sum = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        sum = sum + ml_process_noise(p) * amp;
        p = p * 2.03 + vec2<f32>(17.1, 9.2);
        amp = amp * 0.5;
    }
    return sum;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let sigma = clamp(params.sigma, 0.0, 64.0);
    let blur_step = max(sigma, 1.0);
    let texel = vec2<f32>(
        blur_step / max(params.width, 1.0),
        blur_step / max(params.height, 1.0)
    );
    let base = textureSampleLevel(src_tex, src_samp, uv, 0.0);
    var rgb = base.rgb;
    var out_alpha = base.a;
    let base_rgb = base.rgb;
    let base_alpha = base.a;
    if params.effect_id < 1.5 {
        // HSLA overlay
        rgb = ml_hsla_overlay(rgb, params.hue, params.saturation, params.lightness, params.alpha);
    } else if params.effect_id < 2.5 {
        // Gaussian blur horizontal
        rgb = ml_blur_sharpen_detail_gaussian_5tap_h(src_tex, src_samp, uv, texel);
    } else if params.effect_id < 3.5 {
        // Gaussian blur vertical
        rgb = ml_blur_sharpen_detail_gaussian_5tap_v(src_tex, src_samp, uv, texel);
    } else if params.effect_id < 4.5 || (params.effect_id > 13.5 && params.effect_id < 14.5) || (params.effect_id > 15.5 && params.effect_id < 16.5) || (params.effect_id > 17.5 && params.effect_id < 18.5) {
        // Bloom prefilter: extract bright pixels
        rgb = ml_light_atmosphere_bloom_prefilter(rgb, params.bloom_threshold, 0.5);
    } else if params.effect_id > 5.5 && params.effect_id < 6.5 {
        // Tone map: exposure=hue, contrast=saturation, shoulder=lightness, gamma=alpha, saturation=sigma.
        let exposure_scale = exp2(params.hue);
        let shoulder = clamp(params.lightness, 0.0, 2.0);
        let gamma = max(params.alpha, 0.0001);
        rgb = rgb * exposure_scale;
        rgb = ml_process_aces_fitted(max(rgb, vec3<f32>(0.0)), shoulder);
        rgb = (rgb - vec3<f32>(0.5)) * params.saturation + vec3<f32>(0.5);
        let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        rgb = vec3<f32>(luma) + (rgb - vec3<f32>(luma)) * params.sigma;
        rgb = pow(max(rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
    } else if params.effect_id > 6.5 && params.effect_id < 7.5 {
        // Light sweep: position=hue, angle=saturation, width=lightness, softness=alpha, intensity=sigma.
        let aspect = params.width / max(params.height, 1.0);
        let centered = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
        let angle = radians(params.saturation);
        let normal = vec2<f32>(cos(angle), sin(angle));
        let position = (params.hue - 0.5) * (aspect + 1.0);
        let half_width = max(params.lightness * 0.5, 0.0001);
        let softness = max(params.alpha, 0.0001);
        let distance = dot(centered, normal) - position;
        let band = 1.0 - smoothstep(half_width, half_width + softness, abs(distance));
        rgb = rgb + params.color.rgb * band * max(params.sigma, 0.0) * params.color.a;
    } else if params.effect_id > 8.5 && params.effect_id < 9.5 {
        // Magnify lens: hue=x, saturation=y, lightness=radius, alpha=zoom, sigma=distortion.
        let center_px = vec2<f32>(params.hue, params.saturation);
        let radius = max(params.lightness, 0.001);
        let zoom = max(params.alpha, 0.001);
        let distortion = params.sigma;
        let feather = max(params.bloom_threshold, 0.0);
        let glass = clamp(params.bloom_intensity, 0.0, 1.0);
        let pixel = uv * vec2<f32>(params.width, params.height);
        let delta = pixel - center_px;
        let dist = length(delta);
        let influence = 1.0 - smoothstep(radius, radius + feather, dist);
        if influence > 0.0 {
            let normalized = clamp(dist / radius, 0.0, 1.0);
            let warp = max(0.001, zoom * (1.0 + distortion * (1.0 - normalized * normalized)));
            let sample_uv = clamp((center_px + delta / warp) / vec2<f32>(params.width, params.height), vec2<f32>(0.0), vec2<f32>(1.0));
            var lens = textureSampleLevel(src_tex, src_samp, sample_uv, 0.0).rgb;
            let lens_pos = delta / radius;
            let highlight = pow(max(0.0, 1.0 - length(lens_pos - vec2<f32>(-0.38, -0.42))), 5.0);
            let rim_highlight = (1.0 - clamp(abs(normalized - 0.92) / 0.055, 0.0, 1.0)) * glass;
            let inner_shadow = (1.0 - clamp(abs(normalized - 0.78) / 0.18, 0.0, 1.0)) * glass;
            let rim = 1.0 - smoothstep(0.82, 0.98, normalized);
            let edge_shadow = smoothstep(0.78, 1.0, normalized) * 0.18 * glass;
            lens = lens + vec3<f32>(highlight * glass * 0.32);
            lens = lens * (1.0 - edge_shadow - inner_shadow * 0.08)
                + vec3<f32>(0.92, 0.96, 1.0) * (1.0 - rim) * glass * 0.18
                + vec3<f32>(rim_highlight * 0.22);
            rgb = mix(rgb, lens, influence);
        }
    } else if params.effect_id > 10.5 && params.effect_id < 11.5 {
        // Brightness: hue is additive brightness amount. -1=black, 0=normal, +1=white.
        rgb = clamp(rgb + vec3<f32>(params.hue), vec3<f32>(0.0), vec3<f32>(1.0));
    } else if params.effect_id > 11.5 && params.effect_id < 12.5 {
        // Opacity: hue is alpha multiplier.
        out_alpha = base.a * clamp(params.hue, 0.0, 1.0);
    } else if params.effect_id > 19.5 && params.effect_id < 20.5 {
        // Spectral energy field. This remains an ordinary Process Effect:
        // hue=active mix, saturation=intensity, lightness=density,
        // alpha=drift and sigma=seed.
        let effect_mix = clamp(params.hue, 0.0, 1.0);
        let intensity = max(params.saturation, 0.0);
        let density = max(params.lightness, 0.05);
        let drift = params.alpha;
        let seed = params.sigma;
        let aspect = params.width / max(params.height, 1.0);
        let p = vec2<f32>(uv.x * aspect, uv.y);

        let upper_curve = 0.30 + 0.16 * sin(uv.x * 5.7 + drift * 0.7);
        let middle_curve = 0.52 + 0.075 * sin(uv.x * 8.4 - 0.9 + drift);
        let lower_curve = 0.73 + 0.12 * sin(uv.x * 4.8 + 1.7 - drift * 0.5);
        let cyan_upper = exp(-pow(abs(uv.y - upper_curve) / 0.19, 2.0));
        let cyan_lower = exp(-pow(abs(uv.y - lower_curve) / 0.25, 2.0));
        let magenta_mid = exp(-pow(abs(uv.y - middle_curve) / 0.105, 2.0));
        let dark_cut = smoothstep(0.22, 0.78, 1.0 - uv.x)
            + 0.55 * smoothstep(0.72, 0.28, abs(uv.x - 0.72));

        let flow_noise = ml_process_fbm(
            p * vec2<f32>(3.1, 5.7) * density
                + vec2<f32>(seed * 0.017 + drift, seed * 0.011)
        );
        let fine_noise = ml_process_hash21(
            floor(uv * vec2<f32>(params.width, params.height) * density)
                + vec2<f32>(seed, seed * 1.91)
        );
        let sparkle = pow(fine_noise, 3.6);
        let granular = 0.18 + flow_noise * 0.82 + sparkle * 1.45;

        let cyan = vec3<f32>(0.00, 0.78, 1.00);
        let ice = vec3<f32>(0.30, 0.96, 1.00);
        let magenta = vec3<f32>(1.00, 0.00, 0.52);
        let blue = vec3<f32>(0.00, 0.22, 0.62);
        var field = vec3<f32>(0.002, 0.009, 0.018);
        field = field
            + (cyan_upper * cyan + cyan_lower * ice) * granular
            + magenta_mid * magenta * (0.45 + granular * 0.95)
            + (cyan_upper + cyan_lower) * blue * 0.32;
        field = field * mix(0.78, 1.08, clamp(dark_cut, 0.0, 1.0)) * intensity;
        field = min(field, vec3<f32>(1.35));

        // Keep authored foreground graphics (white title and colored mark)
        // crisp while replacing only the dark scene plate.
        let foreground = smoothstep(0.10, 0.34, max(base_rgb.r, max(base_rgb.g, base_rgb.b)));
        let composed = mix(field, base_rgb, foreground);
        rgb = mix(base_rgb, composed, effect_mix);
        // The Scene input may contain transparent pixels even though this
        // procedural Process effect creates a full-frame energy plate. Give
        // the generated pixels coverage while the effect is active, otherwise
        // valid RGB is discarded by the canvas compositor as transparent.
        out_alpha = mix(base_alpha, 1.0, effect_mix);
    } else if params.effect_id > 7.5 && params.effect_id < 8.5 {
        // Texture overlay: hue=kind, saturation=scale, lightness=strength, alpha=contrast, sigma=seed.
        let kind = i32(round(params.hue));
        let scale = max(params.saturation, 0.001);
        let strength = clamp(params.lightness, 0.0, 1.0);
        let contrast = clamp(params.alpha, 0.0, 2.0);
        let seed = params.sigma;
        let pixel = vec2<f32>(f32(i32(uv.x * params.width)), f32(i32(uv.y * params.height)));
        var tex_value = ml_process_fbm(uv * scale + vec2<f32>(seed, seed * 1.73));
        if kind == 1 {
            let fibers = 0.5 + 0.5 * sin((uv.y * scale * 8.0 + tex_value * 4.0 + seed) * 6.28318);
            tex_value = mix(tex_value, fibers, 0.35);
        } else if kind == 2 {
            tex_value = ml_process_hash21(pixel + vec2<f32>(seed * 19.17, seed * 7.31));
        } else if kind == 3 {
            tex_value = 0.5 + 0.5 * sin((uv.y * params.height * 0.85 + seed) * 6.28318);
        } else if kind == 4 {
            let weave_x = 0.5 + 0.5 * sin((uv.x * scale * 10.0 + seed) * 6.28318);
            let weave_y = 0.5 + 0.5 * sin((uv.y * scale * 12.0 + seed * 1.37) * 6.28318);
            let ridges = sqrt(max(weave_x * weave_y, 0.0));
            tex_value = mix(tex_value, ridges, 0.55);
        } else if kind == 5 || kind == 6 {
            let brush_angle = radians(params.extra.x);
            let bump_strength = clamp(params.extra.y, 0.0, 2.0);
            let relief = clamp(params.extra.z, 0.0, 2.0);
            let brush_x = uv.x * cos(brush_angle) - uv.y * sin(brush_angle);
            let brush_y = uv.x * sin(brush_angle) + uv.y * cos(brush_angle);
            let low = ml_process_fbm(uv * scale * 0.18 + vec2<f32>(seed, seed * 0.61));
            let ridge = 0.5 + 0.5 * sin((brush_x * scale * 18.0 + low * 6.0 + seed) * 6.28318);
            let cross = 0.5 + 0.5 * sin((brush_y * scale * 3.0 + tex_value * 2.0 + seed * 0.7) * 6.28318);
            if kind == 5 {
                tex_value = ridge * 0.62 + cross * 0.18 + low * 0.20;
            } else {
                tex_value = ridge * 0.50 + tex_value * 0.25 + low * 0.25;
            }
            tex_value = (tex_value - 0.5) * (1.0 + relief * 0.45 + bump_strength * 0.20) + 0.5;
        }
        let centered_tex = (tex_value - 0.5) * (1.0 + contrast) + 0.5;
        let material_bump = select(0.0, clamp(params.extra.y, 0.0, 2.0), kind >= 4);
        let bump_shade = 1.0 + (centered_tex - 0.5) * strength * material_bump * 0.55;
        let texture_rgb = mix(vec3<f32>(1.0), params.color.rgb * (0.55 + centered_tex * 0.9) * bump_shade, strength * params.color.a);
        rgb = mix(rgb, clamp(rgb * texture_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), strength);
    }
    var mask_factor = 1.0;
    if abs(params.extra.w) > 0.5 {
        let mask_sample = textureSampleLevel(mask_tex, src_samp, uv, 0.0);
        if abs(params.extra.w) > 1.5 {
            mask_factor = dot(mask_sample.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        } else {
            mask_factor = mask_sample.a;
        }
        if params.extra.w < 0.0 {
            mask_factor = 1.0 - mask_factor;
        }
    }
    mask_factor = clamp(mask_factor, 0.0, 1.0);
    let effected = vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(out_alpha, 0.0, 1.0));
    let original = vec4<f32>(base_rgb, base_alpha);
    return mix(original, effected, mask_factor);
}
"#
);

const COMPOSITE_SHADER: &str = concat!(
    include_str!("kernels/color_tone/color_core.wgsl"),
    r#"

struct ProcessParams {
    width: f32,
    height: f32,
    effect_id: f32,
    _pad0: f32,
    hue: f32,
    saturation: f32,
    lightness: f32,
    alpha: f32,
    sigma: f32,
    bloom_threshold: f32,
    bloom_intensity: f32,
    bloom_sigma: f32,
    color: vec4<f32>,
};

@group(0) @binding(0) var blurred_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var original_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: ProcessParams;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let pos = positions[vertex_index];
    var out: VertexOut;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let blurred = textureSampleLevel(blurred_tex, src_samp, uv, 0.0);
    let original = textureSampleLevel(original_tex, src_samp, uv, 0.0);
    let lum = ml_luma(original.rgb);
    let mask = smoothstep(params.bloom_threshold - 0.1, params.bloom_threshold + 0.1, lum);
    let glow = blurred.rgb * params.color.rgb * mask * max(params.bloom_intensity, 0.0) * params.color.a;
    let rgb = original.rgb + glow;
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), original.a);
}
"#
);

const PROCESS_PRESENT_SHADER: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dims = textureDimensions(src_tex);
    let max_px = dims - vec2<u32>(1u, 1u);
    let px = min(vec2<u32>(u32(in.position.x), u32(in.position.y)), max_px);
    let color = textureLoad(src_tex, vec2<i32>(px), 0);
    return vec4<f32>(color.rgb * color.a, 1.0);
}
"#;
