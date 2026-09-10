// =========================================
// =========================================
// crates/motionloom/src/process/procedural_surface.rs

use crate::{dsl::PassNode, process::runtime::eval_time_expr, scene::drawable::parse_color};

#[derive(Debug, thiserror::Error)]
#[error("procedural_surface parameter {parameter}: {message}")]
pub struct SurfaceParameterError {
    parameter: String,
    message: String,
}

/// Shared, explicitly packed uniforms keep native and browser evaluation identical.
pub(crate) fn surface_values(
    pass: &PassNode,
    size: [u32; 2],
    time_norm: f32,
    time_sec: f32,
) -> Result<[f32; 32], SurfaceParameterError> {
    let raw = |key: &str| {
        pass.params
            .iter()
            .find(|p| p.key == key)
            .map(|p| p.value.as_str())
    };
    let number = |key: &str, default: f32, min: f32, max: f32| {
        let value = match raw(key) {
            Some(expr) => eval_time_expr(expr.trim().trim_matches('"'), time_norm, time_sec)
                .map_err(|message| SurfaceParameterError {
                    parameter: key.into(),
                    message,
                })?,
            None => default,
        };
        if !value.is_finite() {
            return Err(SurfaceParameterError {
                parameter: key.into(),
                message: "must be finite".into(),
            });
        }
        Ok(value.clamp(min, max))
    };
    let color = |key: &str, default: &str| {
        parse_color(
            raw(key)
                .unwrap_or(default)
                .trim()
                .trim_matches('"')
                .trim_matches('\''),
        )
        .map_err(|e| SurfaceParameterError {
            parameter: key.into(),
            message: e.to_string(),
        })
    };
    let base = color("baseColor", "#171C32")?;
    let gold = color("veinColor", "#D9AE50")?;
    Ok([
        size[0] as f32,
        size[1] as f32,
        number("amount", 1.0, 0.0, 1.0)?,
        number("seed", 83.0, 0.0, 65535.0)?,
        number("scale", 3.8, 0.1, 30.0)?,
        number("evolution", 0.0, -10000.0, 10000.0)?,
        number("warpStrength", 1.4, 0.0, 4.0)?,
        number("detailStrength", 0.65, 0.0, 2.0)?,
        number("veinWidth", 0.035, 0.001, 0.2)?,
        number("poolAmount", 0.22, 0.0, 1.0)?,
        number("relief", 0.6, 0.0, 2.0)?,
        number("roughness", 0.28, 0.04, 1.0)?,
        number("lightAzimuth", 125.0, -3600.0, 3600.0)?.to_radians(),
        number("lightElevation", 40.0, 1.0, 89.0)?.to_radians(),
        number("viewX", 0.0, -10000.0, 10000.0)?,
        number("viewY", 0.0, -10000.0, 10000.0)?,
        number("viewZoom", 1.0, 0.1, 20.0)?,
        number("viewRotation", 0.0, -3600.0, 3600.0)?.to_radians(),
        number("flowStrength", 0.0, 0.0, 1.0)?,
        0.0,
        base[0] as f32 / 255.0,
        base[1] as f32 / 255.0,
        base[2] as f32 / 255.0,
        0.0,
        gold[0] as f32 / 255.0,
        gold[1] as f32 / 255.0,
        gold[2] as f32 / 255.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
}

pub(crate) fn surface_bytes(values: &[f32; 32]) -> [u8; 128] {
    let mut bytes = [0; 128];
    for (slot, value) in bytes.chunks_exact_mut(4).zip(values) {
        slot.copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}

/// The same pipeline implementation is used by the scene and WASM process hosts.
pub(crate) struct SurfacePipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl SurfacePipeline {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("procedural-surface-layout"),
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
                        min_binding_size: wgpu::BufferSize::new(128),
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("procedural-surface"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("kernels/stylize_look/procedural_surface.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("procedural-surface-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("procedural-surface-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Self { layout, pipeline }
    }

    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        uniform: &wgpu::Buffer,
    ) {
        let view = input.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("procedural-surface-bindings"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("procedural-surface"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(params: &str) -> PassNode {
        crate::dsl::parse_graph_script(&format!(r#"
<Graph fps={{30}} duration="20s" size={{[64,64]}}>
<Tex id="src" fmt="rgba8" from="input:clip0" />
<Tex id="out" fmt="rgba8" />
<Pass id="surface" kind="compute" effect="procedural_surface" in={{["src"]}} out={{["out"]}} params={{{{{params}}}}} />
<Present from="out" />
</Graph>
"#)).unwrap().passes.remove(0)
    }

    #[test]
    fn defaults_are_finite_and_uniform_aligned() {
        let v = surface_values(&pass(""), [1920, 1080], 0.0, 0.0).unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
        assert_eq!(surface_bytes(&v).len(), 128);
        assert_eq!(v[2], 1.0);
        assert_eq!(v[16], 1.0);
        assert_eq!(v[18], 0.0);
    }

    #[test]
    fn flow_is_opt_in_clamped_and_seekable() {
        let p = pass(r#"flowStrength: "2", evolution: "$time.sec*0.2""#);
        let a = surface_values(&p, [640, 360], 0.5, 10.0).unwrap();
        assert_eq!(a[18], 1.0);
        let _ = surface_values(&p, [640, 360], 0.8, 16.0).unwrap();
        assert_eq!(a, surface_values(&p, [640, 360], 0.5, 10.0).unwrap());
        assert!(surface_values(&pass(r#"flowStrength: "invalid""#), [64, 64], 0.0, 0.0).is_err());
    }

    #[test]
    fn sampling_is_seek_and_fps_independent() {
        let p = pass(r#"evolution: "$time.sec*0.12""#);
        let a = surface_values(&p, [64, 64], 0.5, 300.0 / 30.0).unwrap();
        let _later = surface_values(&p, [64, 64], 0.9, 18.0).unwrap();
        let b = surface_values(&p, [64, 64], 0.5, 600.0 / 60.0).unwrap();
        assert_eq!(a, b);
        assert!((a[5] - 1.2).abs() < 0.0001);
    }

    #[test]
    fn parameters_clamp_and_invalid_values_fail() {
        let p = pass(r#"amount: "-2", roughness: "0", viewZoom: "0""#);
        let v = surface_values(&p, [64, 64], 0.0, 0.0).unwrap();
        assert_eq!(v[2], 0.0);
        assert_eq!(v[11], 0.04);
        assert_eq!(v[16], 0.1);
        assert!(surface_values(&pass(r#"scale: "missing_variable""#), [64, 64], 0.0, 0.0).is_err());
        assert!(surface_values(&pass(r#"baseColor: "not-a-color""#), [64, 64], 0.0, 0.0).is_err());
    }

    #[test]
    fn registration_and_cpu_rejection_are_explicit() {
        assert_eq!(
            crate::process::pass::default_kernel_for_effect("procedural_surface"),
            Some("procedural_surface.wgsl")
        );
        assert!(
            crate::process::effect_kind::is_wasm_webgpu_compatible_effect("procedural_surface")
        );
        assert!(
            crate::process::process_catalog::kernel_source_by_name("procedural_surface.wgsl")
                .is_some()
        );
        let mut graph = crate::dsl::parse_graph_script(
            "<Graph fps={30} size={[64,64]}>\n<Background color=\"#000000\" />\n<Tex id=\"out\" fmt=\"rgba8\" />\n<Present from=\"out\" />\n</Graph>",
        )
        .unwrap();
        graph.passes.push(pass(""));
        assert!(matches!(
            crate::process::cpu_renderer::ProcessCpuRenderer::new(graph),
            Err(crate::process::cpu_renderer::ProcessCpuRenderError::ProceduralSurfaceRequiresGpu)
        ));
    }
}
