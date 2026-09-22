// =========================================
// =========================================
// crates/motionloom/examples/weaver_preview.rs

//! Progressive Weaver preview host.
//!
//! This is a standalone look-development window, independent of the Anica app
//! and of the immediate raster preview. It reads a `.motionloom` document,
//! accumulates Weaver samples over time and displays the tone-mapped result.
//! The final `render` job path is unchanged.
//!
//! Usage:
//!   cargo run -p motionloom --release --features weaver --example weaver_preview -- \
//!     <scene.motionloom> [sceneId] [renderStyle] [frame] [width] [height] [samples]
//!
//! Controls: Esc quit, Space pause, R reset, D toggle denoise,
//! Left/Right change frame, Up/Down change samples per update.

#[cfg(all(feature = "weaver", not(target_arch = "wasm32")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    host::run()
}

#[cfg(not(all(feature = "weaver", not(target_arch = "wasm32"))))]
fn main() {
    eprintln!(
        "weaver_preview requires: cargo run -p motionloom --release --features weaver --example weaver_preview -- ..."
    );
}

#[cfg(all(feature = "weaver", not(target_arch = "wasm32")))]
mod host {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    use motionloom::api::weaver::{PreviewSession, QualityPreset, RenderJob};
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::{Window, WindowId};

    const BLIT_SHADER: &str = r#"
struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VOut;
    let corner = corners[index];
    out.position = vec4<f32>(corner, 0.0, 1.0);
    out.uv = vec2<f32>((corner.x + 1.0) * 0.5, 1.0 - (corner.y + 1.0) * 0.5);
    return out;
}
@group(0) @binding(0) var preview_texture: texture_2d<f32>;
@group(0) @binding(1) var preview_sampler: sampler;
@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return textureSample(preview_texture, preview_sampler, in.uv);
}
"#;

    struct Args {
        scene: PathBuf,
        scene_id: String,
        style: String,
        frame: u32,
        width: u32,
        height: u32,
        samples: u32,
        total_frames: u32,
        mips: bool,
    }

    impl Args {
        fn parse() -> Result<Self, String> {
            let usage = "usage: weaver_preview <scene.motionloom> [--scene-id <id|auto>] [--style <id|auto>]\n\
                 \x20      [--frame N] [--size WxH] [--samples N] [--mips]";
            let raw: Vec<String> = std::env::args().skip(1).collect();
            let scene = PathBuf::from(raw.first().ok_or(usage)?);
            let script = std::fs::read_to_string(&scene).map_err(|e| e.to_string())?;
            let graph = motionloom::parse_graph_script(&script).map_err(|e| e.to_string())?;
            // `auto` lets the engine pick a scene and a Weaver-compatible style
            // instead of forcing every host to know authored ids.
            let mut scene_id = "auto".to_string();
            let mut style = "auto".to_string();
            let mut frame = 0u32;
            let mut width = 640u32;
            let mut height = 360u32;
            let mut samples = 2u32;
            let mut mips = false;
            let mut it = raw.iter().skip(1);
            while let Some(arg) = it.next() {
                let mut value = || it.next().map(String::as_str).unwrap_or("");
                match arg.as_str() {
                    "-h" | "--help" => return Err(usage.into()),
                    "--scene-id" => scene_id = value().to_string(),
                    "--style" => style = value().to_string(),
                    "--frame" => frame = value().parse().expect("--frame must be an integer"),
                    "--samples" => samples = value().parse().expect("--samples must be an integer"),
                    "--mips" => mips = true,
                    "--size" => {
                        let size = value();
                        let (w, h) = size.split_once(['x', 'X']).expect("--size must be WxH");
                        width = w.parse().expect("width must be an integer");
                        height = h.parse().expect("height must be an integer");
                    }
                    other => return Err(format!("unknown flag `{other}`\n{usage}")),
                }
            }
            let total_frames = ((graph.duration_ms as f64 / 1000.0) * graph.fps as f64)
                .round()
                .max(1.0) as u32;
            Ok(Self {
                scene,
                scene_id,
                style,
                frame,
                width,
                height,
                samples,
                total_frames,
                mips,
            })
        }

        fn job(&self) -> RenderJob {
            let mut job = RenderJob::new(&self.scene, QualityPreset::Ultra);
            job.scene_id = self.scene_id.clone();
            job.render_style = self.style.clone();
            job.frame = self.frame;
            job.resolution = [self.width.max(1), self.height.max(1)];
            job.memory_budget_mib = 8192;
            job.texture_mips = self.mips;
            job
        }
    }

    /// `WEAVER_PREVIEW_BOUNCES` restores parity with a final job's bounce budget.
    fn preview_bounce_override() -> Option<u32> {
        std::env::var("WEAVER_PREVIEW_BOUNCES")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .map(|value| value.max(1))
    }

    struct GpuState {
        surface: wgpu::Surface<'static>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
        pipeline: wgpu::RenderPipeline,
        bind_group: wgpu::BindGroup,
        texture: wgpu::Texture,
        size: (u32, u32),
    }

    impl GpuState {
        fn update_texture(&self, rgba: &[u8]) {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.size.0 * 4),
                    rows_per_image: Some(self.size.1),
                },
                wgpu::Extent3d {
                    width: self.size.0,
                    height: self.size.1,
                    depth_or_array_layers: 1,
                },
            );
        }

        fn present(&self) {
            let frame = match self.surface.get_current_texture() {
                Ok(frame) => frame,
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    self.surface.configure(&self.device, &self.config);
                    return;
                }
                Err(_) => return,
            };
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("weaver-preview-blit"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("weaver-preview-blit-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.queue.submit(Some(encoder.finish()));
            frame.present();
        }

        fn resize(&mut self, width: u32, height: u32) {
            self.config.width = width.max(1);
            self.config.height = height.max(1);
            self.surface.configure(&self.device, &self.config);
        }
    }

    struct App {
        args: Args,
        session: Option<PreviewSession>,
        window: Option<Arc<Window>>,
        gpu: Option<GpuState>,
        paused: bool,
        denoise: bool,
        samples_per_update: u32,
        status: String,
        last_update_ms: f64,
    }

    impl App {
        fn new(args: Args) -> Self {
            Self {
                samples_per_update: args.samples,
                args,
                session: None,
                window: None,
                gpu: None,
                paused: false,
                denoise: true,
                status: String::new(),
                last_update_ms: 0.0,
            }
        }

        fn create_gpu(window: Arc<Window>, session: &PreviewSession) -> Result<GpuState, String> {
            let size = window.inner_size();
            let instance = wgpu::Instance::default();
            let surface = instance
                .create_surface(window.clone())
                .map_err(|e| e.to_string())?;
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                }))
                .map_err(|e| e.to_string())?;
            let limits = adapter.limits();
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("weaver-preview-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                }))
                .map_err(|e| e.to_string())?;
            let caps = surface.get_capabilities(&adapter);
            let format = caps
                .formats
                .iter()
                .copied()
                .find(|format| !format.is_srgb())
                .unwrap_or(caps.formats[0]);
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
            };
            surface.configure(&device, &config);

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("weaver-preview-blit-shader"),
                source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("weaver-preview-bind-group-layout"),
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
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("weaver-preview-pipeline-layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("weaver-preview-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            let [width, height] = session.resolution();
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("weaver-preview-texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("weaver-preview-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("weaver-preview-bind-group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            Ok(GpuState {
                surface,
                device,
                queue,
                config,
                pipeline,
                bind_group,
                texture,
                size: (width, height),
            })
        }

        fn update_title(&self) {
            let Some(window) = &self.window else {
                return;
            };
            let (frame, total) = (self.args.frame, self.args.total_frames);
            let samples = self
                .session
                .as_ref()
                .map(|session| session.samples())
                .unwrap_or(0);
            let [width, height] = self
                .session
                .as_ref()
                .map(|session| session.resolution())
                .unwrap_or([0, 0]);
            window.set_title(&format!(
                "MotionLoom Weaver preview | frame {frame}/{total} | {width}x{height} | spp {samples} | {:.2} s/update | denoise {} | {}",
                self.last_update_ms / 1000.0,
                if self.denoise { "on" } else { "off" },
                self.status,
            ));
        }

        fn step(&mut self) {
            if self.paused {
                return;
            }
            let (Some(session), Some(gpu)) = (self.session.as_mut(), self.gpu.as_ref()) else {
                return;
            };
            let started = Instant::now();
            match session.advance(self.samples_per_update) {
                Ok(_) => {}
                Err(error) => {
                    self.status = format!("advance failed: {error}");
                    self.paused = true;
                    return;
                }
            }
            let rgba = if self.denoise {
                match session.denoise_rgba(3) {
                    Ok(rgba) => rgba,
                    Err(error) => {
                        self.status = format!("denoise failed: {error}");
                        session.display_rgba()
                    }
                }
            } else {
                session.display_rgba()
            };
            gpu.update_texture(&rgba);
            self.last_update_ms = started.elapsed().as_secs_f64() * 1000.0;
            let samples = session.samples();
            let [width, height] = session.resolution();
            eprintln!(
                "update: spp {samples} | {:.2} s | {width}x{height}",
                self.last_update_ms / 1000.0
            );
            self.update_title();
        }

        fn step_frame(&mut self, delta: i64) {
            let Some(session) = self.session.as_mut() else {
                return;
            };
            let next = (self.args.frame as i64 + delta).max(0) as u32;
            if next == self.args.frame {
                return;
            }
            self.args.frame = next;
            let job = self.args.job();
            if let Err(error) = pollster::block_on(session.reload(&job)) {
                self.status = format!("reload failed: {error}");
                return;
            }
            self.status = String::new();
            self.update_title();
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attributes = Window::default_attributes()
                .with_title("MotionLoom Weaver preview")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
            let window = Arc::new(
                event_loop
                    .create_window(attributes)
                    .expect("failed to create preview window"),
            );
            let job = self.args.job();
            let mut session = match pollster::block_on(PreviewSession::new(&job)) {
                Ok(session) => session,
                Err(error) => {
                    eprintln!("weaver preview could not start: {error}");
                    event_loop.exit();
                    return;
                }
            };
            if let Some(bounces) = preview_bounce_override() {
                session.set_bounce_budget(bounces, bounces, bounces);
            }
            let gpu = match Self::create_gpu(window.clone(), &session) {
                Ok(gpu) => gpu,
                Err(error) => {
                    eprintln!("weaver preview GPU init failed: {error}");
                    event_loop.exit();
                    return;
                }
            };
            eprintln!(
                "weaver preview: {} | scene {} | {} triangles | style {}",
                session.gpu_name(),
                self.args.scene_id,
                session.triangles(),
                self.args.style.as_str(),
            );
            for line in session.diagnostics() {
                eprintln!("diagnostic: {line}");
            }
            self.window = Some(window);
            self.session = Some(session);
            self.gpu = Some(gpu);
            self.update_title();
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if let Some(gpu) = self.gpu.as_mut() {
                        gpu.resize(size.width, size.height);
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state != ElementState::Pressed {
                        return;
                    }
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        PhysicalKey::Code(KeyCode::Space) => {
                            self.paused = !self.paused;
                            self.status = if self.paused {
                                "paused".to_string()
                            } else {
                                String::new()
                            };
                            self.update_title();
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            if let Some(session) = self.session.as_mut() {
                                session.reset_accumulation();
                            }
                            self.update_title();
                        }
                        PhysicalKey::Code(KeyCode::KeyD) => {
                            self.denoise = !self.denoise;
                            self.update_title();
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => self.step_frame(1),
                        PhysicalKey::Code(KeyCode::ArrowLeft) => self.step_frame(-1),
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.samples_per_update = (self.samples_per_update * 2).min(256);
                            self.update_title();
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.samples_per_update = (self.samples_per_update / 2).max(1);
                            self.update_title();
                        }
                        _ => {}
                    }
                }
                WindowEvent::RedrawRequested => {
                    self.step();
                    if let Some(gpu) = self.gpu.as_ref() {
                        gpu.present();
                    }
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
                _ => {}
            }
        }

        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::parse()?;
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = App::new(args);
        event_loop.run_app(&mut app)?;
        Ok(())
    }
}
