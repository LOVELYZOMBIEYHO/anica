// =========================================
// =========================================
// crates/motionloom/src/scene/compositor/executor.rs

use super::{SceneCompositionError, SceneCompositionPlan};

/// Validated composition request independent of its execution backend.
pub struct SceneCompositor {
    plan: SceneCompositionPlan,
}

impl SceneCompositor {
    pub fn new(plan: SceneCompositionPlan) -> Result<Self, SceneCompositionError> {
        plan.validate()?;
        Ok(Self { plan })
    }

    pub fn plan(&self) -> &SceneCompositionPlan {
        &self.plan
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::{sync::Arc, time::Duration};

    use half::f16;
    use wgpu::util::DeviceExt;

    use super::super::{
        AlphaMode, GpuLayerTexture, LinearPremultipliedImage, SceneGpuContext,
        SceneGpuContextError, SourceColorSpace,
    };

    const COMPOSITE_SHADER: &str = r#"
struct LayerParams {
    straight_alpha: u32,
    _padding_0: u32,
    _padding_1: u32,
    _padding_2: u32,
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: LayerParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[index], 0.0, 1.0);
    return output;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(source);
    let pixel = min(vec2<u32>(input.position.xy), dimensions - vec2<u32>(1u));
    var color = textureLoad(source, vec2<i32>(pixel), 0);
    if params.straight_alpha != 0u {
        color = vec4<f32>(color.rgb * color.a, color.a);
    }
    return color;
}
"#;

    pub struct GpuCompositionInput<'a> {
        pub texture: &'a GpuLayerTexture,
        pub alpha: AlphaMode,
    }

    /// Executes ordered layers entirely on the shared device.
    pub struct WgpuSceneCompositorExecutor {
        context: SceneGpuContext,
        layout: wgpu::BindGroupLayout,
        pipeline: wgpu::RenderPipeline,
    }

    impl WgpuSceneCompositorExecutor {
        pub fn new(context: SceneGpuContext) -> Self {
            let device = context.device();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("MotionLoom SceneCompositor layer layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MotionLoom SceneCompositor pipeline layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MotionLoom SceneCompositor shader"),
                source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
            });
            let over = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("MotionLoom SceneCompositor pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(over),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });
            Self {
                context,
                layout,
                pipeline,
            }
        }

        pub fn compose(
            &self,
            size: [u32; 2],
            layers: &[GpuCompositionInput<'_>],
        ) -> GpuLayerTexture {
            let device = self.context.device();
            let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("MotionLoom SceneCompositor RGBA16F output"),
                size: wgpu::Extent3d {
                    width: size[0].max(1),
                    height: size[1].max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }));
            let target = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MotionLoom SceneCompositor encoder"),
            });
            if layers.is_empty() {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("MotionLoom SceneCompositor clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
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
            }
            for (index, layer) in layers.iter().enumerate() {
                let source = layer.texture.view();
                let flag = u32::from(layer.alpha == AlphaMode::Straight).to_le_bytes();
                let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("MotionLoom SceneCompositor layer params"),
                    contents: &[flag.as_slice(), &[0u8; 12]].concat(),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MotionLoom SceneCompositor layer"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&source),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("MotionLoom SceneCompositor over"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if index == 0 {
                                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.context.queue().submit([encoder.finish()]);
            GpuLayerTexture {
                texture,
                size: [size[0].max(1), size[1].max(1)],
                format: wgpu::TextureFormat::Rgba16Float,
                color_space: SourceColorSpace::Linear,
                alpha_mode: AlphaMode::Premultiplied,
            }
        }

        /// Upload one canonical layer without an intermediate RGBA8 plate.
        pub fn upload(
            &self,
            image: &LinearPremultipliedImage,
        ) -> Result<GpuLayerTexture, SceneGpuContextError> {
            let expected = image.size[0] as usize * image.size[1] as usize;
            if image.pixels.len() != expected {
                return Err(SceneGpuContextError::Transfer(format!(
                    "image has {} pixels, expected {expected}",
                    image.pixels.len()
                )));
            }
            let texture = Arc::new(self.context.device().create_texture(
                &wgpu::TextureDescriptor {
                    label: Some("MotionLoom SceneCompositor uploaded RGBA16F layer"),
                    size: wgpu::Extent3d {
                        width: image.size[0],
                        height: image.size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            ));
            let bytes = image
                .pixels
                .iter()
                .flatten()
                .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            self.context.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(image.size[0] * 8),
                    rows_per_image: Some(image.size[1]),
                },
                wgpu::Extent3d {
                    width: image.size[0],
                    height: image.size[1],
                    depth_or_array_layers: 1,
                },
            );
            Ok(GpuLayerTexture {
                texture,
                size: image.size,
                format: wgpu::TextureFormat::Rgba16Float,
                color_space: SourceColorSpace::Linear,
                alpha_mode: AlphaMode::Premultiplied,
            })
        }

        /// Read the final RGBA16F texture for EXR/PNG encoding only.
        pub fn readback(
            &self,
            texture: &GpuLayerTexture,
        ) -> Result<LinearPremultipliedImage, SceneGpuContextError> {
            if texture.format != wgpu::TextureFormat::Rgba16Float {
                return Err(SceneGpuContextError::Transfer(format!(
                    "expected RGBA16F, received {:?}",
                    texture.format
                )));
            }
            let [width, height] = texture.size;
            let row_bytes = width * 8;
            let padded_row_bytes = row_bytes.div_ceil(256) * 256;
            let buffer = self
                .context
                .device()
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("MotionLoom SceneCompositor RGBA16F readback"),
                    size: padded_row_bytes as u64 * height as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
            let mut encoder =
                self.context
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MotionLoom SceneCompositor readback encoder"),
                    });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row_bytes),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            self.context.queue().submit([encoder.finish()]);

            let slice = buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            let mut spins = 0u32;
            let result = loop {
                if spins % 64 == 0 {
                    self.context
                        .device()
                        .poll(wgpu::PollType::Poll)
                        .map_err(|error| SceneGpuContextError::Transfer(error.to_string()))?;
                }
                match receiver.try_recv() {
                    Ok(result) => break result,
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        if std::time::Instant::now() >= deadline {
                            return Err(SceneGpuContextError::Transfer(
                                "RGBA16F readback timed out".into(),
                            ));
                        }
                        spins = spins.wrapping_add(1);
                        std::hint::spin_loop();
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        return Err(SceneGpuContextError::Transfer(
                            "RGBA16F readback channel closed".into(),
                        ));
                    }
                }
            };
            result.map_err(|error| SceneGpuContextError::Transfer(error.to_string()))?;
            let mapped = slice.get_mapped_range();
            let mut pixels = Vec::with_capacity(width as usize * height as usize);
            for row in mapped
                .chunks_exact(padded_row_bytes as usize)
                .take(height as usize)
            {
                for pixel in row[..row_bytes as usize].chunks_exact(8) {
                    pixels.push(std::array::from_fn(|channel| {
                        let offset = channel * 2;
                        f16::from_bits(u16::from_le_bytes([pixel[offset], pixel[offset + 1]]))
                            .to_f32()
                    }));
                }
            }
            drop(mapped);
            buffer.unmap();
            LinearPremultipliedImage::new(texture.size, pixels)
                .map_err(|error| SceneGpuContextError::Transfer(error.to_string()))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;
