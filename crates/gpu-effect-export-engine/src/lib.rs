use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct SingleClipOpacityVideoToolboxRequest {
    pub source_start: Duration,
    pub duration: Duration,
    pub fps: u32,
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub opacity: f32,
    pub opacity_filter_suffix: Option<String>,
    pub audio_bitrate_kbps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    EmptyPath,
    DurationTooShort,
    OpacityIsIdentity,
}

impl BuildError {
    pub const fn as_str(self) -> &'static str {
        match self {
            BuildError::EmptyPath => "source/output path is empty",
            BuildError::DurationTooShort => "export duration is too short",
            BuildError::OpacityIsIdentity => "opacity is identity (1.0)",
        }
    }
}

/// Build a narrow, phase-1 GPU export command:
/// - single clip
/// - opacity-only effect
/// - H.264 VideoToolbox encoder
///
/// This crate intentionally does not validate timeline semantics. Callers should
/// gate usage with their own feature checks and fallback behavior.
pub fn build_single_clip_opacity_videotoolbox_args(
    source_path: &str,
    request: SingleClipOpacityVideoToolboxRequest,
    out_path: &str,
) -> Result<Vec<String>, BuildError> {
    if source_path.trim().is_empty() || out_path.trim().is_empty() {
        return Err(BuildError::EmptyPath);
    }
    if request.duration <= Duration::from_millis(1) {
        return Err(BuildError::DurationTooShort);
    }

    let opacity = request.opacity.clamp(0.0, 1.0);
    if (opacity - 1.0).abs() <= 0.001 {
        return Err(BuildError::OpacityIsIdentity);
    }

    let fps = request.fps.clamp(1, 144);
    let width = request.canvas_width.max(1);
    let height = request.canvas_height.max(1);
    let audio_kbps = request.audio_bitrate_kbps.clamp(64, 512);

    let opacity_filter = request.opacity_filter_suffix.unwrap_or_else(|| {
        format!(
            ",colorchannelmixer=rr={o:.6}:gg={o:.6}:bb={o:.6}",
            o = opacity
        )
    });
    let vf = format!(
        "fps={fps},setsar=1,scale=w={w}:h={h}:force_original_aspect_ratio=decrease:eval=frame,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,format=bgra{opacity_filter},format=yuv420p",
        fps = fps,
        w = width,
        h = height,
        opacity_filter = opacity_filter
    );

    let mut args = vec![
        "-y".to_string(),
        "-hide_banner".to_string(),
        "-ss".to_string(),
        format!("{:.6}", request.source_start.as_secs_f64()),
        "-t".to_string(),
        format!("{:.6}", request.duration.as_secs_f64()),
        "-i".to_string(),
        source_path.to_string(),
        "-map".to_string(),
        "0:v:0".to_string(),
        "-map".to_string(),
        "0:a?".to_string(),
        "-vf".to_string(),
        vf,
        "-c:v".to_string(),
        "h264_videotoolbox".to_string(),
        "-allow_sw".to_string(),
        // VideoToolbox hardware encoder requires ~23.976+ fps.
        // Allow software fallback at low fps so the encoder does not hang.
        if fps < 24 { "1" } else { "0" }.to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-b:v".to_string(),
        "12M".to_string(),
        "-maxrate".to_string(),
        "16M".to_string(),
        "-bufsize".to_string(),
        "24M".to_string(),
        "-c:a".to_string(),
        "aac".to_string(),
        "-b:a".to_string(),
        format!("{audio_kbps}k"),
        out_path.to_string(),
    ];

    // Keep parity with core exporter's arg conventions.
    if args.is_empty() {
        args.push(out_path.to_string());
    }

    Ok(args)
}

const WGPU_OPACITY_SHADER: &str = r#"
struct OpacityParams {
    opacity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0)
var src_tex: texture_2d<f32>;

@group(0) @binding(1)
var dst_tex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: OpacityParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(dst_tex);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }
    let color = textureLoad(src_tex, vec2<i32>(gid.xy), 0);
    let o = clamp(params.opacity, 0.0, 1.0);
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(color.rgb * o, color.a));
}
"#;

#[derive(Debug)]
pub enum GpuProcessError {
    AdapterUnavailable,
    DeviceRequest(String),
    InvalidFrameSize { got: usize, expected: usize },
    MissingResource(&'static str),
    DevicePoll(String),
    Map(String),
}

impl std::fmt::Display for GpuProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuProcessError::AdapterUnavailable => {
                write!(f, "no suitable WGPU adapter found")
            }
            GpuProcessError::DeviceRequest(msg) => write!(f, "wgpu device request failed: {msg}"),
            GpuProcessError::InvalidFrameSize { got, expected } => {
                write!(f, "invalid RGBA frame size: got={got}, expected={expected}")
            }
            GpuProcessError::MissingResource(name) => write!(f, "missing GPU resource: {name}"),
            GpuProcessError::DevicePoll(msg) => write!(f, "wgpu device.poll failed: {msg}"),
            GpuProcessError::Map(msg) => write!(f, "wgpu buffer map failed: {msg}"),
        }
    }
}

impl std::error::Error for GpuProcessError {}

fn align_to_256(v: u32) -> u32 {
    const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    ((v + ALIGN - 1) / ALIGN) * ALIGN
}

/// Minimal phase-1 GPU frame processor for opacity-only export:
/// - input/output format: RGBA8
/// - effect: multiply RGB by opacity, keep alpha
pub struct WgpuOpacityProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    src_texture: Option<wgpu::Texture>,
    dst_texture: Option<wgpu::Texture>,
    readback_buffer: Option<wgpu::Buffer>,
    padded_bytes_per_row: u32,
}

impl WgpuOpacityProcessor {
    pub fn new(width: u32, height: u32) -> Result<Self, GpuProcessError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|_| GpuProcessError::AdapterUnavailable)?;

        let required_limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("anica-export-gpu-opacity-device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|err| GpuProcessError::DeviceRequest(err.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-export-gpu-opacity-shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGPU_OPACITY_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("anica-export-gpu-opacity-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anica-export-gpu-opacity-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("anica-export-gpu-opacity-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anica-export-gpu-opacity-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut this = Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
            uniform_buffer,
            width: 0,
            height: 0,
            src_texture: None,
            dst_texture: None,
            readback_buffer: None,
            padded_bytes_per_row: 0,
        };
        this.ensure_resources(width.max(1), height.max(1));
        Ok(this)
    }

    fn make_rgba_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anica-export-gpu-opacity-tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn ensure_resources(&mut self, width: u32, height: u32) {
        if self.width == width
            && self.height == height
            && self.src_texture.is_some()
            && self.dst_texture.is_some()
            && self.readback_buffer.is_some()
        {
            return;
        }

        self.width = width;
        self.height = height;
        self.src_texture = Some(Self::make_rgba_texture(&self.device, width, height));
        self.dst_texture = Some(Self::make_rgba_texture(&self.device, width, height));

        let row_bytes = width.saturating_mul(4);
        let padded = align_to_256(row_bytes);
        self.padded_bytes_per_row = padded;
        let total_size = padded as u64 * height as u64;
        self.readback_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anica-export-gpu-opacity-readback"),
            size: total_size.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }

    pub fn process_rgba_frame(
        &mut self,
        rgba: &[u8],
        opacity: f32,
    ) -> Result<Vec<u8>, GpuProcessError> {
        let expected = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        if rgba.len() != expected {
            return Err(GpuProcessError::InvalidFrameSize {
                got: rgba.len(),
                expected,
            });
        }

        let src = self
            .src_texture
            .as_ref()
            .ok_or(GpuProcessError::MissingResource("src_texture"))?;
        let dst = self
            .dst_texture
            .as_ref()
            .ok_or(GpuProcessError::MissingResource("dst_texture"))?;
        let readback = self
            .readback_buffer
            .as_ref()
            .ok_or(GpuProcessError::MissingResource("readback_buffer"))?;

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width.saturating_mul(4)),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let mut uniform = [0u8; 16];
        uniform[0..4].copy_from_slice(&opacity.clamp(0.0, 1.0).to_ne_bytes());
        self.queue.write_buffer(&self.uniform_buffer, 0, &uniform);

        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = dst.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-export-gpu-opacity-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-export-gpu-opacity-encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("anica-export-gpu-opacity-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = self.width.div_ceil(16);
            let wg_y = self.height.div_ceil(16);
            pass.dispatch_workgroups(wg_x.max(1), wg_y.max(1), 1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: dst,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let _submission = self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device
            .poll(wgpu::PollType::wait())
            .map_err(|err| GpuProcessError::DevicePoll(err.to_string()))?;
        rx.recv()
            .map_err(|err| GpuProcessError::Map(err.to_string()))?
            .map_err(|err| GpuProcessError::Map(err.to_string()))?;

        let mapped = slice.get_mapped_range();
        let mut out = vec![0u8; expected];
        let row_bytes = self.width as usize * 4;
        let padded_row = self.padded_bytes_per_row as usize;
        for row in 0..self.height as usize {
            let src_off = row * padded_row;
            let dst_off = row * row_bytes;
            out[dst_off..(dst_off + row_bytes)]
                .copy_from_slice(&mapped[src_off..(src_off + row_bytes)]);
        }
        drop(mapped);
        readback.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BuildError, SingleClipOpacityVideoToolboxRequest, align_to_256,
        build_single_clip_opacity_videotoolbox_args,
    };
    use std::time::Duration;

    #[test]
    fn builds_videotoolbox_opacity_command() {
        let req = SingleClipOpacityVideoToolboxRequest {
            source_start: Duration::from_secs(1),
            duration: Duration::from_secs(2),
            fps: 30,
            canvas_width: 1920,
            canvas_height: 1080,
            opacity: 0.7,
            opacity_filter_suffix: None,
            audio_bitrate_kbps: 192,
        };
        let args = build_single_clip_opacity_videotoolbox_args("in.mp4", req, "out.mp4").unwrap();
        assert!(args.iter().any(|v| v == "h264_videotoolbox"));
        assert!(args.iter().any(|v| v.contains("colorchannelmixer")));
        assert_eq!(args.last().map(String::as_str), Some("out.mp4"));
    }

    #[test]
    fn rejects_identity_opacity() {
        let req = SingleClipOpacityVideoToolboxRequest {
            source_start: Duration::ZERO,
            duration: Duration::from_secs(1),
            fps: 30,
            canvas_width: 1280,
            canvas_height: 720,
            opacity: 1.0,
            opacity_filter_suffix: None,
            audio_bitrate_kbps: 192,
        };
        let err =
            build_single_clip_opacity_videotoolbox_args("in.mp4", req, "out.mp4").unwrap_err();
        assert_eq!(err, BuildError::OpacityIsIdentity);
    }

    #[test]
    fn align_to_256_rounds_up() {
        assert_eq!(align_to_256(1), 256);
        assert_eq!(align_to_256(255), 256);
        assert_eq!(align_to_256(256), 256);
        assert_eq!(align_to_256(257), 512);
    }
}
