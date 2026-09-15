// =========================================
// =========================================
// crates/motionloom/src/world/render/tests/mod.rs

//! Native renderer regression tests grouped outside the production module.

mod params_layout;
mod render_regression;
mod shader_validation;
fn test_gpu_vertex(x: f32) -> super::GpuWorldVertex {
    super::GpuWorldVertex {
        outline_normal: [0.0, 0.0, 1.0],
        position: [x, 0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        tangent: [1.0, 0.0, 0.0],
        bitangent: [0.0, 1.0, 0.0],
        joints: [0.0; 4],
        weights: [1.0, 0.0, 0.0, 0.0],
        uv: [0.0; 2],
        color: [1.0; 4],
    }
}

#[test]
fn indexed_geometry_reuses_matching_vertices() {
    let triangle = [
        test_gpu_vertex(0.0),
        test_gpu_vertex(1.0),
        test_gpu_vertex(2.0),
    ];
    let expanded = triangle.into_iter().chain(triangle).collect::<Vec<_>>();
    let (vertices, indices) = super::index_gpu_world_vertices(&expanded);
    let chunks = super::split_gpu_world_indexed_chunks(&vertices, &indices, 1024 * 1024);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].vertices.len(), 3);
    assert_eq!(chunks[0].indices, vec![0, 1, 2, 0, 1, 2]);
}

#[test]
fn indexed_geometry_splits_only_between_triangles() {
    let expanded = (0..6)
        .map(|index| test_gpu_vertex(index as f32))
        .collect::<Vec<_>>();
    let (vertices, indices) = super::index_gpu_world_vertices(&expanded);
    let chunks = super::split_gpu_world_indexed_chunks(
        &vertices,
        &indices,
        super::GPU_WORLD_VERTEX_STRIDE_BYTES * 3,
    );
    assert_eq!(chunks.len(), 2);
    assert!(chunks.iter().all(|chunk| chunk.vertices.len() == 3));
    assert!(chunks.iter().all(|chunk| chunk.indices == [0, 1, 2]));
}

#[test]
fn rigid_visibility_is_conservative_at_camera_edges() {
    let mut p = super::GpuWorldParams {
        canvas: [100.0, 100.0, 50.0, 50.0],
        camera0: [0.0, 0.0, 0.0, 50.0],
        camera1: [1.0, 0.0, 0.0, 0.1],
        camera2: [0.0, 1.0, 0.0, 100.0],
        camera3: [0.0, 0.0, 1.0, 0.0],
        ..Default::default()
    };
    p.model[3] = 1.0;
    p.actor_rotation[3] = 1.0;
    let bounds = Some(([-0.5; 3], [0.5; 3]));
    p.actor[2] = 5.0;
    assert!(super::rigid_draw_visible(bounds, p));
    p.actor[0] = 20.0;
    assert!(!super::rigid_draw_visible(bounds, p));
    p.actor[2] = 0.0;
    assert!(super::rigid_draw_visible(bounds, p));
    p.actor[2] = 5.0;
    p.vegetation[0] = 1.0;
    assert!(super::rigid_draw_visible(bounds, p));
    assert!(super::rigid_draw_visible(None, p));
    assert_eq!(
        super::pack_gpu_world_params(p).len(),
        std::mem::size_of::<super::GpuWorldParams>()
    );
}

use std::{
    collections::HashMap,
    fs,
    io::{Cursor, Read, Write},
    net::TcpListener,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use crate::asset::{AssetResolver, AssetSource, MemoryAssetResolver, PathAssetResolver};
use crate::world::{parse_world_graph_script, render_world_frame};

// Check projection containment separately from uniform packing assertions.
fn assert_fitted_shadow_volume(params: super::GpuWorldLightingParams) {
    // Small translated actors must stay inside the fitted volume at every rotation.
    let mut shadow = params;
    shadow.color1[3] = 0.8;
    shadow.shadow0 = [1.0, 0.0, 0.0, 14.0];
    shadow.shadow1 = [0.0, 1.0, 0.0, 14.0];
    shadow.shadow2 = [0.0, 0.0, 1.0, 28.0];
    let mut actor = super::GpuWorldParams::default();
    actor.model[3] = 1.0;
    actor.actor = [3.0, -2.0, 1.0, 0.0];
    actor.actor_rotation[3] = 1.0;
    let bounds = ([-0.5; 3], [0.5; 3]);
    let fitted = super::fit_rigid_shadow_volume(shadow, &[(bounds, actor)]);
    assert!(fitted.shadow0[3] < 2.0);
    assert_eq!(&fitted.shadow3[..3], &[3.0, -2.0, 1.0]);
    actor.actor_rotation = [0.0, 0.70710677, 0.0, 0.70710677];
    let rotated = super::fit_rigid_shadow_volume(shadow, &[(bounds, actor)]);
    assert_eq!(rotated.shadow0, fitted.shadow0);
    assert_eq!(rotated.shadow3, fitted.shadow3);
    for corner in 0..8 {
        let local = std::array::from_fn(|axis| if corner & (1 << axis) == 0 { -0.5 } else { 0.5 });
        let world = super::quat_rotate_vec3(actor.actor_rotation, local);
        for (axis, value) in world.iter().enumerate() {
            assert!((*value + actor.actor[axis] - fitted.shadow3[axis]).abs() < fitted.shadow0[3]);
        }
    }
    assert_eq!(
        super::fit_rigid_shadow_volume(shadow, &[]).shadow0,
        shadow.shadow0
    );
    let large = (([-20.0; 3], [20.0; 3]), actor);
    assert_eq!(
        super::fit_rigid_shadow_volume(shadow, &[large]).shadow0,
        shadow.shadow0
    );
}

#[test]
fn fog_and_optics_pack_into_distinct_gpu_uniform_slots() {
    let lighting = crate::world::WorldLighting {
        atmosphere_fog: Some(crate::world::WorldAtmosphereFog {
            mode: "height".to_string(),
            color: [0.5, 0.6, 0.7],
            density: 0.02,
            start: 3.0,
            end: 40.0,
            base_height: 0.4,
            height_falloff: 0.2,
            scattering: 0.1,
            affect_sky: true,
            bounds_min: Some([-4.0, 0.0, -8.0]),
            bounds_max: Some([4.0, 6.0, -1.0]),
            edge_feather: 0.75,
        }),
        ..Default::default()
    };
    let camera = super::PerspectiveCameraView {
        eye: [0.0, 1.0, 5.0],
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, -1.0],
        focal_px: 800.0,
        near: 0.02,
        far: 40.0,
        aspect: 16.0 / 9.0,
        optics: [5.0, 50.0, 2.8, 8.0],
    };

    let params = super::GpuWorldLightingParams::from_world(&lighting, camera, false, 1);
    assert_fitted_shadow_volume(params);
    assert_eq!(params.fog0, [3.0, 0.02, 3.0, 40.0]);
    assert_eq!(params.fog2, [0.2, 0.1, 1.0, 1.0]);
    assert_eq!(params.fog3, [-4.0, 0.0, -8.0, 1.0]);
    assert_eq!(params.fog4, [4.0, 6.0, -1.0, 0.75]);
    assert_eq!(params.optics0, [5.0, 50.0, 2.8, 8.0]);
    assert_eq!(super::pack_gpu_world_lighting(params).len(), 1120);
}

#[test]
fn camera_hidden_hips_hides_the_whole_actor_color_pass() {
    assert!(super::camera_hidden_bones_hide_whole_actor(&[
        "hips".to_string(),
        "head".to_string(),
    ]));
    assert!(!super::camera_hidden_bones_hide_whole_actor(&[
        "head".to_string(),
    ]));
}

#[test]
fn transmissive_material_defaults_to_sorted_non_depth_writing_phase() {
    let material = super::GlbMaterialData {
        transmission_factor: 0.94,
        ..Default::default()
    };
    let phase = super::gpu_world_material_phase(Some(&material));
    assert_eq!(phase, super::GpuWorldDrawPhase::Transmissive);
    assert!(!super::gpu_world_material_depth_write(
        Some(&material),
        phase
    ));
}

#[test]
fn explicit_transparent_depth_write_override_is_preserved() {
    let material = super::GlbMaterialData {
        alpha_mode: super::GlbAlphaMode::Blend,
        depth_write: super::GlbDepthWriteMode::Enabled,
        ..Default::default()
    };
    let phase = super::gpu_world_material_phase(Some(&material));
    assert_eq!(phase, super::GpuWorldDrawPhase::AlphaBlend);
    assert!(super::gpu_world_material_depth_write(
        Some(&material),
        phase
    ));
}

#[test]
fn explicit_opaque_depth_write_disable_is_preserved() {
    let material = super::GlbMaterialData {
        depth_write: super::GlbDepthWriteMode::Disabled,
        ..Default::default()
    };
    let phase = super::gpu_world_material_phase(Some(&material));
    assert_eq!(phase, super::GpuWorldDrawPhase::Opaque);
    assert!(!super::gpu_world_material_depth_write(
        Some(&material),
        phase
    ));
}

fn png_fixture(color: [u8; 4]) -> Vec<u8> {
    let image = image::RgbaImage::from_pixel(2, 2, image::Rgba(color));
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .expect("encode in-memory PNG fixture");
    bytes.into_inner()
}

struct CountingImageResolver {
    bytes: Vec<u8>,
    calls: AtomicUsize,
}

impl AssetResolver for CountingImageResolver {
    fn resolve(&self, _src: &str) -> Result<AssetSource, String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(AssetSource::Bytes(self.bytes.clone()))
    }
}

#[test]
fn environment_source_cache_is_checked_before_resolving_bytes() {
    let resolver = Arc::new(CountingImageResolver {
        bytes: png_fixture([40, 60, 80, 255]),
        calls: AtomicUsize::new(0),
    });
    let mut renderer = super::WorldFrameRenderer::with_resolver(resolver.clone());
    let lighting = crate::world::WorldLighting {
        environment: Some(crate::world::WorldEnvironmentLighting {
            src: "sky.png".to_string(),
            mapping: "equirectangular".to_string(),
            intensity: 1.0,
            rotation_y_degrees: 0.0,
            visible: true,
            background_intensity: 1.0,
            background_blur: 0.0,
            diffuse_intensity: 1.0,
            specular_intensity: 1.0,
        }),
        ..Default::default()
    };
    let camera = super::PerspectiveCameraView {
        eye: [0.0, 1.0, 3.0],
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, -1.0],
        focal_px: 100.0,
        near: 0.01,
        far: 100.0,
        aspect: 1.0,
        optics: [0.0; 4],
    };

    renderer
        .prepare_gpu_lighting(&lighting, Path::new("."), camera)
        .expect("decode environment once");
    renderer
        .prepare_gpu_lighting(&lighting, Path::new("."), camera)
        .expect("reuse decoded environment");

    assert_eq!(resolver.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn primitive_image_asset_decodes_once_and_content_revision_invalidates_it() {
    let resolver = MemoryAssetResolver::new();
    resolver.insert("stone.png".into(), png_fixture([80, 90, 100, 255]));
    let mut cache = HashMap::new();
    let mut stats = super::PrimitiveResourceLoadStats::default();
    let first = super::load_cached_primitive_texture(
        Path::new("."),
        crate::world::WorldPathStyle::Relative,
        "stone.png",
        &resolver,
        &mut cache,
        &mut stats,
    )
    .expect("first texture decode");
    let second = super::load_cached_primitive_texture(
        Path::new("."),
        crate::world::WorldPathStyle::Relative,
        "stone.png",
        &resolver,
        &mut cache,
        &mut stats,
    )
    .expect("shared texture decode");
    assert_eq!(stats.texture_decode_count, 1);
    assert_eq!(stats.texture_cache_hits, 1);
    assert!(Arc::ptr_eq(&first.rgba, &second.rgba));

    resolver.insert("stone.png".into(), png_fixture([120, 130, 140, 255]));
    let revised = super::load_cached_primitive_texture(
        Path::new("."),
        crate::world::WorldPathStyle::Relative,
        "stone.png",
        &resolver,
        &mut cache,
        &mut stats,
    )
    .expect("revised texture decode");
    assert_eq!(stats.texture_decode_count, 2);
    assert!(!Arc::ptr_eq(&first.rgba, &revised.rgba));
}

#[test]
fn native_world_asset_resolver_fetches_http_url_as_bytes() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local asset server");
    let address = listener.local_addr().expect("local asset server address");
    let body = b"small universal GLB fixture".to_vec();
    let expected = body.clone();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept asset request");
        let mut request = [0_u8; 1024];
        let _ = stream.read(&mut request).expect("read asset request");
        write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: model/gltf-binary\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .expect("write asset response headers");
        stream.write_all(&body).expect("write asset response body");
    });

    let url = format!("http://{address}/iphone.glb");
    let resolved = super::resolve_world_asset_source(
        Path::new("."),
        &url,
        crate::world::WorldPathStyle::Relative,
        &PathAssetResolver,
    )
    .expect("native URL asset should resolve");
    server.join().expect("asset server thread");

    match resolved {
        super::ResolvedWorldAsset::Bytes { key, bytes } => {
            assert_eq!(key, std::path::PathBuf::from(url));
            assert_eq!(bytes, expected);
        }
        _ => panic!("remote URL must resolve to in-memory bytes"),
    }
}

#[test]
fn remote_glb_url_is_its_stable_mesh_cache_key() {
    let url = "https://raw.githubusercontent.com/example/assets/iphone.glb";
    assert!(super::is_remote_world_asset_source(url));
    assert_eq!(
        super::glb_mesh_source_cache_key(
            Path::new("ignored"),
            url,
            crate::world::WorldPathStyle::Relative,
        ),
        std::path::PathBuf::from(url)
    );
}

#[test]
fn world_asset_resolver_decodes_inline_base64_bytes() {
    let src = "data:image/png;base64,AQIDBA==";
    let resolved = super::resolve_world_asset_source(
        Path::new("ignored"),
        src,
        crate::world::WorldPathStyle::Relative,
        &PathAssetResolver,
    )
    .expect("inline world asset should resolve");

    match resolved {
        super::ResolvedWorldAsset::Bytes { key, bytes } => {
            assert_eq!(key, std::path::PathBuf::from(src));
            assert_eq!(bytes, vec![1, 2, 3, 4]);
        }
        _ => panic!("inline data URI must resolve to in-memory bytes"),
    }
}

#[test]
fn world_pbr_shader_parses_and_validates() {
    let module = wgpu::naga::front::wgsl::parse_str(super::WGPU_WORLD_SHADER.as_str())
        .expect("world PBR WGSL must parse");
    let mut validator = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("world PBR WGSL must validate");
}

#[test]
fn world_dof_shader_is_webgpu_derivative_safe() {
    let module = wgpu::naga::front::wgsl::parse_str(super::WGPU_WORLD_DOF_SHADER)
        .expect("world DoF WGSL must parse");
    let mut validator = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("world DoF WGSL must validate");
    assert!(super::WGPU_WORLD_DOF_SHADER.contains("textureSampleLevel"));
    assert!(!super::WGPU_WORLD_DOF_SHADER.contains("textureSample(scene_color"));
}

#[test]
fn temporal_jitter_is_bounded_and_camera_cuts_reset_history() {
    for frame in 0..32 {
        let jitter = super::preview_frame_jitter(frame);
        assert!((-0.5..=0.5).contains(&jitter[0]));
        assert!((-0.5..=0.5).contains(&jitter[1]));
    }
    let camera = super::PreviewCameraHistory {
        camera0: [0.0, 1.0, -4.0, 900.0],
        camera1: [1.0, 0.0, 0.0, 0.1],
        camera2: [0.0, 1.0, 0.0, 100.0],
        camera3: [0.0, 0.0, 1.0, 0.0],
        jitter: [0.0; 2],
    };
    assert!(!super::preview_camera_cut(camera, camera));
    let mut cut = camera;
    cut.camera3 = [0.0, 0.0, -1.0, 0.0];
    assert!(super::preview_camera_cut(camera, cut));
}

#[test]
fn temporal_signature_changes_with_render_style_controls() {
    let camera = super::PerspectiveCameraView {
        eye: [0.0, 1.0, -4.0],
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, 1.0],
        focal_px: 900.0,
        near: 0.1,
        far: 100.0,
        aspect: 16.0 / 9.0,
        optics: [0.0; 4],
    };
    let base = super::GpuWorldLightingParams::from_world(
        &super::WorldLighting::default(),
        camera,
        false,
        1,
    );
    let signature = super::preview_temporal_style_signature(&base);
    let mut changed = base;
    changed.universal_tone[1] = 1.25;
    assert_ne!(signature, super::preview_temporal_style_signature(&changed));
}

#[test]
fn object_motion_keeps_distinct_current_and_previous_bone_palettes() {
    let current = [super::mat4_identity()];
    let mut previous_matrix = super::mat4_identity();
    previous_matrix[12] = -2.0;
    let bytes = super::pack_gpu_world_bone_pair(&current, &[previous_matrix]);
    assert_eq!(bytes.len(), 2 * 64);
    assert_ne!(&bytes[..64], &bytes[64..]);
    assert!(super::WGPU_WORLD_SHADER.contains("fn previous_bone_transform"));
    assert!(super::WGPU_WORLD_SHADER.contains("fn fs_main_gbuffer"));
    assert!(super::WGPU_WORLD_SHADER.contains("@location(2) material"));
}

#[test]
fn temporal_velocity_separates_projection_jitter_from_physical_motion() {
    assert!(super::WGPU_WORLD_SHADER.contains("@location(11) current_clip"));
    assert!(super::WGPU_WORLD_SHADER.contains("input.current_clip.xy / input.current_clip.w"));
    assert!(!super::WGPU_WORLD_SHADER.contains("let current_uv = input.pos.xy / params.canvas.xy"));
    assert!(super::WGPU_WORLD_SHADER.contains(
        "velocity = current_uv - previous_uv\n                - (lighting.preview1.xy - lighting.preview1.zw) / params.canvas.xy"
    ));
    assert!(
        super::WGPU_WORLD_DOF_SHADER
            .contains("let physical_velocity = preview_physical_velocity(sample_uv)")
    );
    assert!(super::WGPU_WORLD_DOF_SHADER.contains("return preview_velocity(uv)"));
}

#[test]
fn authored_anti_aliasing_overrides_host_policy_and_portable_falls_back() {
    let source = r#"<Graph fps={30} duration="1s" size={[320,180]}>
  <RenderStyle id="styled_off"><SurfaceStyle shading="physical" /></RenderStyle>
  <RenderStyle id="styled_taa">
    <AntiAliasingStyle method="taa" quality="high" fallback="smaa" sharpness="0.2" />
  </RenderStyle>
  <Scene id="off_scene" renderStyle="styled_off"></Scene>
  <Scene id="taa_scene" renderStyle="styled_taa"></Scene>
  <Present from="off_scene" />
</Graph>"#
        .replace("><", ">\n<");
    let graph = crate::dsl::parse_graph_script(&source).unwrap();
    let balanced = crate::preview::ImmediatePreviewSettings::default();
    let host = super::resolve_effective_anti_aliasing(None, balanced);
    assert_eq!(host.requested, "host");
    assert!(host.temporal);

    let off_style = crate::render_style::resolve_scene_render_style(&graph, "off_scene").unwrap();
    let off = super::resolve_effective_anti_aliasing(Some(&off_style), balanced);
    assert_eq!(off.spatial_selector, 0.0);
    assert!(!off.temporal);

    let taa_style = crate::render_style::resolve_scene_render_style(&graph, "taa_scene").unwrap();
    let taa = super::resolve_effective_anti_aliasing(Some(&taa_style), balanced);
    assert!(taa.temporal);
    assert_eq!(taa.jitter_phases, 4);
    assert_eq!(taa.sharpness, 0.2);

    let portable = crate::preview::ImmediatePreviewSettings {
        profile: crate::preview::ImmediatePreviewProfile::Portable,
        ..Default::default()
    };
    let fallback = super::resolve_effective_anti_aliasing(Some(&taa_style), portable);
    assert!(!fallback.temporal);
    assert_eq!(fallback.spatial_selector, 2.0);
}

#[test]
fn unavailable_multisample_methods_report_a_spatial_fallback() {
    let style = crate::render_style::ResolvedSceneRenderStyle {
        anti_aliasing: Some(crate::render_style::ResolvedAntiAliasingStyle {
            method: "msaa".into(),
            quality: "ultra".into(),
            fallback: "auto".into(),
            sharpness: 0.1,
        }),
        depth_of_field: None,
        universal: Default::default(),
        cel: Default::default(),
        scene_id: "test".into(),
        style_id: Some("test".into()),
        shading: "physical".into(),
        shading_steps: 3,
        diffuse_wrap: 0.0,
        rim_light: 0.0,
        rim_power: 3.0,
        specular: 1.0,
        roughness_bias: 0.0,
        surface_saturation: 1.0,
        ambient_intensity: 1.0,
        ambient_color: [1.0; 3],
        hard_shadows: false,
        lighting_preset: None,
        post: Default::default(),
        overrides: Vec::new(),
    };
    let effective = super::resolve_effective_anti_aliasing(
        Some(&style),
        crate::preview::ImmediatePreviewSettings {
            profile: crate::preview::ImmediatePreviewProfile::Cinematic,
            ..Default::default()
        },
    );
    assert_eq!(effective.requested, "msaa");
    assert_eq!(effective.effective, "smaa");
    assert!(effective.fallback_used);
}

#[test]
fn vegetation_wind_is_shader_driven_and_auto_lod_is_distance_relative() {
    assert!(super::WGPU_WORLD_SHADER.contains("fn vegetation_deform"));
    assert_eq!(
        super::vegetation_auto_lod(2.0, [0.0, 0.0, 0.0], [0.0, 0.0, 8.0]),
        crate::dsl::VegetationLod::Full
    );
    assert_eq!(
        super::vegetation_auto_lod(2.0, [0.0, 0.0, 0.0], [0.0, 0.0, 16.0]),
        crate::dsl::VegetationLod::Half
    );
    assert_eq!(
        super::vegetation_auto_lod(2.0, [0.0, 0.0, 0.0], [0.0, 0.0, 30.0]),
        crate::dsl::VegetationLod::Quarter
    );
}

#[test]
fn world_pbr_shadow_sampling_uses_explicit_level_for_webgpu() {
    assert!(
        super::WGPU_WORLD_SHADER.contains("textureSampleCompareLevel("),
        "shadow comparison sampling must not require uniform derivative control flow"
    );
    assert!(!super::WGPU_WORLD_SHADER.contains("textureSampleCompare("));
}

#[test]
fn world_pbr_shader_preserves_clip_w_for_perspective_correct_uvs() {
    assert!(
        super::WGPU_WORLD_SHADER.contains("out.pos = vec4<f32>(clip_x, clip_y, clip_z, view_z);")
    );
    assert!(!super::WGPU_WORLD_SHADER.contains("out.pos = vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);"));
}

#[test]
fn scene_material_texture_flips_top_left_raster_to_gltf_uv_origin() {
    let mut image = image::RgbaImage::new(1, 2);
    image.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
    image.put_pixel(0, 1, image::Rgba([0, 0, 255, 255]));
    let texture = super::gpu_world_texture_from_image(&image);
    assert_eq!(texture.rgba.as_slice(), &[0, 0, 255, 255, 255, 0, 0, 255]);
}

#[test]
fn world_camera_yaw_rotates_actor_world_position() {
    let front = super::camera_actor_view(1.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert!((front.x - 1.0).abs() < 0.001);
    assert!(front.depth.abs() < 0.001);
    assert!((front.yaw - 30.0).abs() < 0.001);

    let side = super::camera_actor_view(0.0, 0.0, 1.0, 135.0, 0.0, 0.0, 0.0, 90.0, 0.0);
    assert!((side.x - 1.0).abs() < 0.001);
    assert!(side.depth.abs() < 0.001);
    assert!((side.yaw - 45.0).abs() < 0.001);
}

#[test]
fn glb_clip_channel_samples_between_keyframes() {
    let channel = super::GlbAnimationChannelData {
        node_index: 0,
        property: super::GlbAnimationProperty::Translation,
        interpolation: super::GlbAnimationInterpolation::Linear,
        times: vec![0.0, 1.0],
        values: super::GlbAnimationValues::Vec3(vec![[0.0, 2.0, 4.0], [8.0, 6.0, 0.0]]),
    };

    let Some(super::GlbAnimationValues::Vec3(values)) =
        super::sample_animation_channel(&channel, 0.25)
    else {
        panic!("expected sampled translation");
    };
    assert_eq!(values, vec![[2.0, 3.0, 3.0]]);
}

#[test]
fn multiple_glb_clip_layers_crossfade_in_source_order() {
    let channel = |value| super::GlbAnimationChannelData {
        node_index: 0,
        property: super::GlbAnimationProperty::Translation,
        interpolation: super::GlbAnimationInterpolation::Linear,
        times: vec![0.0, 1.0],
        values: super::GlbAnimationValues::Vec3(vec![value, value]),
    };
    let mesh = super::GlbMeshData {
        path: std::path::PathBuf::from("clips.glb"),
        positions: Vec::new(),
        normals: Vec::new(),
        texcoords: Vec::new(),
        colors: Vec::new(),
        joints: Vec::new(),
        weights: Vec::new(),
        indices: Vec::new(),
        triangles: Vec::new(),
        materials: Vec::new(),
        textures: Vec::new(),
        mesh_names: Vec::new(),
        nodes: vec![crate::world::GlbNodeData {
            index: 0,
            name: Some("hips".to_string()),
            parent: None,
            children: Vec::new(),
            mesh: None,
            skin: None,
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            matrix: None,
        }],
        skin: None,
        animations: vec![
            crate::world::gltf_loader::GlbAnimationData {
                name: Some("A".to_string()),
                duration: 1.0,
                channels: vec![channel([10.0, 0.0, 0.0])],
            },
            crate::world::gltf_loader::GlbAnimationData {
                name: Some("B".to_string()),
                duration: 1.0,
                channels: vec![channel([20.0, 0.0, 0.0])],
            },
        ],
        bounds_min: [0.0, 0.0, 0.0],
        bounds_max: [0.0, 0.0, 0.0],
    };
    let play = |name: &str| crate::world::WorldPlay {
        clip: Some(name.to_string()),
        r#loop: false,
        speed: "1".to_string(),
        weight: "0.5".to_string(),
        blend_in: "0".to_string(),
        blend_out: "0".to_string(),
        mask: Vec::new(),
    };
    let actor = crate::world::WorldActor {
        cel_materials: Vec::new(),
        id: "girl".to_string(),
        model: "clips.glb".to_string(),
        primitive: None,
        terrain: None,
        vegetation: None,
        native_skin: None,
        path_style: crate::world::WorldPathStyle::Relative,
        hide_meshes: Vec::new(),
        hide_materials: Vec::new(),
        camera_hidden_bones: Vec::new(),
        profile: None,
        rig: None,
        retarget: None,
        x: "0".to_string(),
        y: "0".to_string(),
        z: "0".to_string(),
        yaw: "0".to_string(),
        pitch: "0".to_string(),
        roll: "0".to_string(),
        rotation_quaternion: None,
        scale: "1".to_string(),
        scale_mode: "none".to_string(),
        opacity: "1".to_string(),
        material: None,
        play: Some(play("A")),
        plays: vec![play("B")],
    };
    let graph = crate::world::WorldGraph {
        id: None,
        version: None,
        fps: 30.0,
        duration_ms: 1_000,
        duration_explicit: true,
        size: (1, 1),
        render_size: None,
        model_profiles: Vec::new(),
        worlds: Vec::new(),
        retargets: Vec::new(),
        actions: Vec::new(),
        apply_actions: Vec::new(),
        animation_assets: Vec::new(),
        constraints: Vec::new(),
        attachments: Vec::new(),
        lighting: crate::world::WorldLighting::default(),
        present: crate::world::WorldPresent {
            from: String::new(),
        },
    };
    let sampled = super::sample_actor_clip(
        &graph,
        &actor,
        &mesh,
        crate::world::WorldTime {
            frame: 30,
            fps: 30.0,
            duration_ms: 1_000,
        },
    )
    .expect("sample clip layers");
    assert_eq!(sampled[&0].translation, Some([12.5, 0.0, 0.0]));
}

#[test]
fn humanoid_body_masks_match_canonical_sides() {
    assert!(super::bone_matches_body_mask(
        "upper_arm_r",
        &["right_arm".to_string()]
    ));
    assert!(!super::bone_matches_body_mask(
        "upper_arm_l",
        &["right_arm".to_string()]
    ));
    assert!(super::bone_matches_body_mask(
        "lower_leg_l",
        &["lower_body".to_string()]
    ));
}

#[test]
fn humanoid_action_quaternion_adapter_preserves_renderer_euler_order() {
    for rotation_deg in [
        [18.0, -27.0, 43.0],
        [-72.0, 12.0, -9.0],
        [4.979, 6.178, 4.611],
    ] {
        let transform = super::BoneOverride {
            translation: [0.0; 3],
            rotation_deg,
            scale: 1.0,
        };
        let quaternion = super::quat_from_bone_override(transform);
        let recovered = super::quat_to_zyx_euler_degrees(quaternion);
        for axis in 0..3 {
            assert!(
                (recovered[axis] - rotation_deg[axis]).abs() < 0.001,
                "axis {axis}: expected {}, got {}",
                rotation_deg[axis],
                recovered[axis]
            );
        }
    }
}

#[test]
fn humanoid_retarget_preserves_model_space_rotation_delta() {
    let quaternion = |rotation_deg| {
        super::quat_from_bone_override(super::BoneOverride {
            translation: [0.0; 3],
            rotation_deg,
            scale: 1.0,
        })
    };
    let source_rest = quaternion([23.0, 11.0, -7.0]);
    let authored_delta = quaternion([-9.0, 38.0, 14.0]);
    let source_animated = super::quat_mul_xyzw(authored_delta, source_rest);
    let target_rest = quaternion([-17.0, 6.0, 31.0]);

    let target_animated =
        super::model_space_retarget_global(source_rest, source_animated, target_rest);
    let recovered_delta = super::quat_normalize_xyzw(super::quat_mul_xyzw(
        target_animated,
        super::quat_conjugate_xyzw(target_rest),
    ));
    let alignment = recovered_delta
        .iter()
        .zip(authored_delta)
        .map(|(actual, expected)| actual * expected)
        .sum::<f32>()
        .abs();
    assert!(
        alignment > 0.99999,
        "retarget changed model-space rotation delta: {alignment}"
    );
}

#[test]
fn canonical_action_honors_root_motion_none_for_hips_only() {
    let authored = [0.019, 0.105, 1.001];
    assert_eq!(
        super::canonical_action_translation("hips", Some("none"), authored),
        [0.0; 3]
    );
    assert_eq!(
        super::canonical_action_translation("hips", None, authored),
        [0.0; 3]
    );
    assert_eq!(
        super::canonical_action_translation("hips", Some("clip"), authored),
        authored
    );
    assert_eq!(
        super::canonical_action_translation("hand_l", Some("none"), authored),
        authored
    );
}

#[test]
fn baked_reference_retarget_does_not_capture_small_semantic_actions() {
    let bone = |raw_rotation: bool| crate::world::WorldActionBone {
        id: "hips".to_string(),
        x: None,
        y: None,
        z: None,
        rotation: None,
        rotation_x: raw_rotation.then(|| "10".to_string()),
        rotation_y: None,
        rotation_z: None,
        forward: None,
        side: None,
        twist: None,
        bend: Some("5".to_string()),
        turn: None,
        scale: None,
        opacity: None,
        interpolation: None,
        in_tangent: None,
        out_tangent: None,
    };
    let action = |pose_count: usize, raw_rotation: bool| crate::world::WorldAction {
        id: "test".to_string(),
        skeleton: "humanoid_v1".to_string(),
        intent: None,
        duration_ms: 1_000,
        poses: (0..pose_count)
            .map(|index| crate::world::WorldActionPose {
                t: index as f32 / 30.0,
                label: None,
                bones: vec![bone(raw_rotation)],
            })
            .collect(),
        iks: Vec::new(),
    };

    assert!(!super::action_uses_baked_humanoid_reference(&action(
        9, false
    )));
    assert!(!super::action_uses_baked_humanoid_reference(&action(
        9, true
    )));
    assert!(!super::action_uses_baked_humanoid_reference(&action(
        120, false
    )));
    assert!(super::action_uses_baked_humanoid_reference(&action(
        120, true
    )));
}

#[test]
fn quaternius_humanoid_joint_names_map_to_canonical_bones() {
    let cases = [
        ("pelvis", "hips"),
        ("spine_01", "spine"),
        ("spine_02", "chest"),
        ("spine_03", "upper_chest"),
        ("neck_01", "neck"),
        ("clavicle_l", "shoulder_l"),
        ("upperarm_l", "upper_arm_l"),
        ("lowerarm_r", "forearm_r"),
        ("thigh_l", "upper_leg_l"),
        ("calf_r", "lower_leg_r"),
        ("ball_l", "toe_l"),
    ];
    for (source, expected) in cases {
        assert_eq!(
            super::canonical_humanoid_bone(source, "quaternius_humanoid").as_deref(),
            Some(expected),
            "failed to canonicalize Quaternius joint '{source}'"
        );
    }
}

#[test]
fn standard_humanoid_finger_names_map_without_changing_aliases() {
    let cases = [
        ("mixamorig:LeftHandThumb1", "thumb_1_l"),
        ("LeftIndex2", "index_2_l"),
        ("Middle3_R", "middle_3_r"),
        ("RightHandPinky3", "pinky_3_r"),
    ];
    for (source, expected) in cases {
        assert_eq!(
            super::canonical_humanoid_bone(source, "auto").as_deref(),
            Some(expected),
            "failed to canonicalize finger joint '{source}'"
        );
    }
    assert_eq!(
        super::canonical_humanoid_bone("weapon_socket", "auto"),
        None
    );
}

#[test]
fn external_humanoid_clip_maps_rotation_to_canonical_target_bone() {
    let node = |name: &str| crate::world::GlbNodeData {
        index: 0,
        name: Some(name.to_string()),
        parent: None,
        children: Vec::new(),
        mesh: None,
        skin: None,
        translation: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
        matrix: None,
    };
    let empty_mesh = |path: &str, node| super::GlbMeshData {
        path: std::path::PathBuf::from(path),
        positions: Vec::new(),
        normals: Vec::new(),
        texcoords: Vec::new(),
        colors: Vec::new(),
        joints: Vec::new(),
        weights: Vec::new(),
        indices: Vec::new(),
        triangles: Vec::new(),
        materials: Vec::new(),
        textures: Vec::new(),
        mesh_names: Vec::new(),
        nodes: vec![node],
        skin: None,
        animations: Vec::new(),
        bounds_min: [0.0, 0.0, 0.0],
        bounds_max: [1.0, 1.0, 1.0],
    };
    let mut source = empty_mesh("walk.glb", node("source:RightArm"));
    source
        .animations
        .push(crate::world::gltf_loader::GlbAnimationData {
            name: Some("Walk".to_string()),
            duration: 1.0,
            channels: vec![super::GlbAnimationChannelData {
                node_index: 0,
                property: super::GlbAnimationProperty::Rotation,
                interpolation: super::GlbAnimationInterpolation::Linear,
                times: vec![0.0, 1.0],
                values: super::GlbAnimationValues::Quat(vec![
                    [0.0, 0.0, 0.0, 1.0],
                    [
                        0.0,
                        0.0,
                        std::f32::consts::FRAC_1_SQRT_2,
                        std::f32::consts::FRAC_1_SQRT_2,
                    ],
                ]),
            }],
        });
    let target = empty_mesh("target.glb", node("upper_arm_r"));
    let actor = crate::world::WorldActor {
        id: "character_a".to_string(),
        cel_materials: Vec::new(),
        model: "target.glb".to_string(),
        primitive: None,
        terrain: None,
        vegetation: None,
        native_skin: None,
        path_style: crate::world::WorldPathStyle::Relative,
        hide_meshes: Vec::new(),
        hide_materials: Vec::new(),
        camera_hidden_bones: Vec::new(),
        profile: Some("motionloom_humanoid_v1".to_string()),
        rig: None,
        retarget: None,
        x: "0".to_string(),
        y: "0".to_string(),
        z: "0".to_string(),
        yaw: "0".to_string(),
        pitch: "0".to_string(),
        roll: "0".to_string(),
        rotation_quaternion: None,
        scale: "1".to_string(),
        scale_mode: "none".to_string(),
        opacity: "1".to_string(),
        material: None,
        play: None,
        plays: Vec::new(),
    };
    let graph = crate::world::WorldGraph {
        id: None,
        version: None,
        fps: 30.0,
        duration_ms: 1_000,
        duration_explicit: true,
        size: (1, 1),
        render_size: None,
        model_profiles: vec![crate::world::WorldModelProfile {
            id: "motionloom_humanoid_v1".to_string(),
            model: "target.glb".to_string(),
            preset: "humanoid_v1".to_string(),
            retarget: Some(crate::world::WorldProfileRetarget {
                preset: "humanoid_v1".to_string(),
                maps: vec![crate::world::WorldRetargetMap {
                    from: "upper_arm_r".to_string(),
                    to: "upper_arm_r".to_string(),
                }],
            }),
            bone_axis_map: Some(crate::world::WorldBoneAxisMap {
                axes: vec![crate::world::WorldBoneAxis {
                    bone: "upper_arm_r".to_string(),
                    forward: Some("rotationZ:-1".to_string()),
                    side: Some("rotationX:-1".to_string()),
                    twist: Some("rotationY:1".to_string()),
                    bend: None,
                    turn: None,
                    rest_forward: None,
                    rest_side: Some("-90".to_string()),
                    rest_twist: None,
                    rest_bend: None,
                    rest_turn: None,
                }],
            }),
        }],
        worlds: Vec::new(),
        retargets: Vec::new(),
        actions: Vec::new(),
        apply_actions: vec![crate::world::WorldApplyAction {
            target: "character_a".to_string(),
            action: "walk".to_string(),
            at_ms: 0,
            r#loop: false,
            weight: "1".to_string(),
            speed: "1".to_string(),
            blend_in: "0".to_string(),
            blend_out: "0".to_string(),
            mode: "override".to_string(),
            mask: Vec::new(),
            duration_ms: None,
            root_motion: None,
            destination: None,
            face: None,
            sync_group: None,
            sync_marker: None,
        }],
        animation_assets: vec![crate::world::WorldAnimationAsset {
            id: "walk".to_string(),
            src: "walk.glb".to_string(),
            profile: "fbx_humanoid".to_string(),
            clip: Some("Walk".to_string()),
        }],
        constraints: Vec::new(),
        attachments: Vec::new(),
        lighting: crate::world::WorldLighting::default(),
        present: crate::world::WorldPresent {
            from: String::new(),
        },
    };
    let source_key = std::path::PathBuf::from("walk.glb");
    let mesh_cache = std::collections::HashMap::from([(source_key.clone(), source)]);
    let sampled = super::sample_external_actor_actions(
        &graph,
        &actor,
        &target,
        &std::collections::HashMap::from([("walk".to_string(), source_key)]),
        &mesh_cache,
        crate::world::WorldTime {
            frame: 15,
            fps: 30.0,
            duration_ms: 1_000,
        },
    )
    .expect("sample external humanoid clip");
    let rotation = sampled[&0].rotation.expect("mapped target rotation");
    assert!(rotation[2].abs() > 0.3, "rotation={rotation:?}");
    let overrides = super::actor_bone_overrides_for_mesh(
        &graph,
        &actor,
        Some(&target),
        crate::world::WorldTime {
            frame: 15,
            fps: 30.0,
            duration_ms: 1_000,
        },
    )
    .expect("external clip rest calibration");
    assert!(
        !overrides.contains_key("upper_arm_r"),
        "externally driven arm must not receive its semantic rest offset twice"
    );
}

#[test]
fn action_blend_envelope_fades_in_and_out() {
    let action = crate::world::WorldAction {
        id: "wave".to_string(),
        skeleton: "humanoid_v1".to_string(),
        intent: None,
        duration_ms: 2_000,
        poses: Vec::new(),
        iks: Vec::new(),
    };
    let apply = crate::world::WorldApplyAction {
        target: "girl".to_string(),
        action: "wave".to_string(),
        at_ms: 0,
        r#loop: false,
        weight: "1".to_string(),
        speed: "1".to_string(),
        blend_in: "0.5".to_string(),
        blend_out: "0.5".to_string(),
        mode: "override".to_string(),
        mask: Vec::new(),
        duration_ms: None,
        root_motion: None,
        destination: None,
        face: None,
        sync_group: None,
        sync_marker: None,
    };
    let entering = crate::world::WorldTime {
        frame: 6,
        fps: 30.0,
        duration_ms: 2_000,
    };
    let leaving = crate::world::WorldTime {
        frame: 57,
        fps: 30.0,
        duration_ms: 2_000,
    };

    let fade_in = super::action_blend_envelope(&action, &apply, 0.2, 1.0, entering)
        .expect("fade-in envelope");
    let fade_out = super::action_blend_envelope(&action, &apply, 1.9, 1.0, leaving)
        .expect("fade-out envelope");
    assert!((fade_in - 0.4).abs() < 0.001);
    assert!((fade_out - 0.2).abs() < 0.001);
}

#[test]
fn world_action_interpolation_preserves_linear_default_and_authored_curves() {
    let key = |interpolation: Option<&str>, in_tangent: Option<&str>, out_tangent: Option<&str>| {
        crate::world::WorldActionBone {
            id: "hips".to_string(),
            x: None,
            y: None,
            z: None,
            rotation: None,
            rotation_x: None,
            rotation_y: None,
            rotation_z: None,
            forward: None,
            side: None,
            twist: None,
            bend: None,
            turn: None,
            scale: None,
            opacity: None,
            interpolation: interpolation.map(ToString::to_string),
            in_tangent: in_tangent.map(ToString::to_string),
            out_tangent: out_tangent.map(ToString::to_string),
        }
    };
    let linear = key(None, None, None);
    let hold = key(Some("hold"), None, None);
    let ease = key(Some("ease"), None, None);
    let bezier = key(Some("bezier"), None, Some("1"));
    let incoming = key(None, Some("-1"), None);

    assert!((super::world_action_key_mix(Some(&linear), None, 0.25) - 0.25).abs() < 0.0001);
    assert_eq!(super::world_action_key_mix(Some(&hold), None, 0.75), 0.0);
    assert!((super::world_action_key_mix(Some(&ease), None, 0.25) - 0.15625).abs() < 0.0001);
    let curved = super::world_action_key_mix(Some(&bezier), Some(&incoming), 0.5);
    assert!((curved - 0.75).abs() < 0.0001, "curved={curved}");
}

#[test]
fn binary_action_pose_lookup_preserves_legacy_boundaries() {
    let pose = |t| crate::world::WorldActionPose {
        t,
        label: None,
        bones: Vec::new(),
    };
    let poses = vec![pose(0.0), pose(0.25), pose(0.75), pose(1.0)];

    let pair = super::action_pose_pair(&poses, -0.1);
    assert_eq!((pair.0.t, pair.1.t), (0.0, 0.0));
    let pair = super::action_pose_pair(&poses, 0.25);
    assert_eq!((pair.0.t, pair.1.t), (0.0, 0.25));
    let pair = super::action_pose_pair(&poses, 0.5);
    assert_eq!((pair.0.t, pair.1.t), (0.25, 0.75));
    let pair = super::action_pose_pair(&poses, 1.0);
    assert_eq!((pair.0.t, pair.1.t), (1.0, 1.0));
}

#[test]
fn positive_bend_uses_the_model_profile_axis_without_changing_semantics() {
    let bone = crate::world::WorldActionBone {
        id: "lower_leg_l".to_string(),
        x: None,
        y: None,
        z: None,
        rotation: None,
        rotation_x: None,
        rotation_y: None,
        rotation_z: None,
        forward: None,
        side: None,
        twist: None,
        bend: Some("30".to_string()),
        turn: None,
        scale: None,
        opacity: None,
        interpolation: None,
        in_tangent: None,
        out_tangent: None,
    };
    let axis_map = |binding: &str| crate::world::WorldBoneAxisMap {
        axes: vec![crate::world::WorldBoneAxis {
            bone: "lower_leg_l".to_string(),
            forward: None,
            side: None,
            twist: None,
            bend: Some(binding.to_string()),
            turn: None,
            rest_forward: None,
            rest_side: None,
            rest_twist: None,
            rest_bend: None,
            rest_turn: None,
        }],
    };
    let time = crate::world::WorldTime {
        frame: 0,
        fps: 30.0,
        duration_ms: 1_000,
    };

    let positive = super::interpolate_bone(
        Some(&bone),
        Some(&bone),
        0.0,
        time,
        Some(&axis_map("rotationX:1")),
    )
    .expect("positive bend mapping");
    let mirrored = super::interpolate_bone(
        Some(&bone),
        Some(&bone),
        0.0,
        time,
        Some(&axis_map("rotationX:-1")),
    )
    .expect("negative bend mapping");

    assert!((positive.rotation_deg[0] - 30.0).abs() < 0.0001);
    assert!((mirrored.rotation_deg[0] + 30.0).abs() < 0.0001);
}

#[test]
fn looping_action_blends_only_at_authored_window_boundaries() {
    let action = crate::world::WorldAction {
        id: "walk".to_string(),
        skeleton: "humanoid_v1".to_string(),
        intent: None,
        duration_ms: 1_000,
        poses: Vec::new(),
        iks: Vec::new(),
    };
    let apply = crate::world::WorldApplyAction {
        target: "actor".to_string(),
        action: "walk".to_string(),
        at_ms: 0,
        r#loop: true,
        weight: "1".to_string(),
        speed: "1".to_string(),
        blend_in: "0.1".to_string(),
        blend_out: "0.2".to_string(),
        mode: "override".to_string(),
        mask: Vec::new(),
        duration_ms: Some(3_500),
        root_motion: None,
        destination: None,
        face: None,
        sync_group: None,
        sync_marker: None,
    };
    let time = |frame| crate::world::WorldTime {
        frame,
        fps: 100.0,
        duration_ms: 4_000,
    };

    let internal_seam = super::action_blend_envelope(&action, &apply, 0.99, 1.0, time(99))
        .expect("internal loop seam envelope");
    let final_window_fade = super::action_blend_envelope(&action, &apply, 0.4, 1.0, time(340))
        .expect("authored window fade-out");
    assert!((internal_seam - 1.0).abs() < 0.001);
    assert!((final_window_fade - 0.5).abs() < 0.001);
}

#[test]
fn authored_action_phase_matches_loop_and_authored_window_timing() {
    let action = crate::world::WorldAction {
        id: "walk".to_string(),
        skeleton: "humanoid_v1".to_string(),
        intent: None,
        duration_ms: 1_000,
        poses: Vec::new(),
        iks: Vec::new(),
    };
    let mut apply = crate::world::WorldApplyAction {
        target: "actor".to_string(),
        action: "walk".to_string(),
        at_ms: 500,
        r#loop: true,
        weight: "1".to_string(),
        speed: "1".to_string(),
        blend_in: "0".to_string(),
        blend_out: "0".to_string(),
        mode: "override".to_string(),
        mask: Vec::new(),
        duration_ms: Some(3_000),
        root_motion: None,
        destination: None,
        face: None,
        sync_group: None,
        sync_marker: None,
    };
    let time = |frame| crate::world::WorldTime {
        frame,
        fps: 30.0,
        duration_ms: 4_000,
    };

    let half = super::authored_action_phase(&action, &apply, time(30))
        .expect("loop phase")
        .expect("active loop");
    let seam = super::authored_action_phase(&action, &apply, time(45))
        .expect("loop seam phase")
        .expect("active loop");
    assert!((half - 0.5).abs() < 0.001);
    assert!(seam.abs() < 0.001);

    apply.r#loop = false;
    apply.at_ms = 0;
    apply.duration_ms = Some(2_000);
    let stretched = super::authored_action_phase(&action, &apply, time(30))
        .expect("stretched phase")
        .expect("active authored window");
    assert!((stretched - 0.5).abs() < 0.001);
}

#[test]
fn canonical_action_delta_preserves_target_rest_axis_calibration() {
    let rest = super::BoneOverride {
        translation: [0.0, 0.0, 0.0],
        rotation_deg: [0.0, 0.0, 90.0],
        scale: 1.0,
    };
    let action_delta = super::BoneOverride {
        translation: [0.0, 0.02, 0.0],
        rotation_deg: [24.0, 0.0, 0.0],
        scale: 1.0,
    };

    let full = rest.composed_with(action_delta);
    assert_eq!(full.translation, [0.0, 0.02, 0.0]);
    assert_eq!(full.rotation_deg, [24.0, 0.0, 90.0]);

    let half = rest.blended_to(full, 0.5);
    assert_eq!(half.translation, [0.0, 0.01, 0.0]);
    assert_eq!(half.rotation_deg, [12.0, 0.0, 90.0]);
}

#[test]
fn two_bone_ik_reaches_target_with_local_axis_calibration() {
    let node = |index, name: &str, parent, children, translation| crate::world::GlbNodeData {
        index,
        name: Some(name.to_string()),
        parent,
        children,
        mesh: None,
        skin: None,
        translation,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
        matrix: None,
    };
    let mesh = super::GlbMeshData {
        path: std::path::PathBuf::from("analytic-ik.glb"),
        positions: Vec::new(),
        normals: Vec::new(),
        texcoords: Vec::new(),
        colors: Vec::new(),
        joints: Vec::new(),
        weights: Vec::new(),
        indices: Vec::new(),
        triangles: Vec::new(),
        materials: Vec::new(),
        textures: Vec::new(),
        mesh_names: Vec::new(),
        nodes: vec![
            node(0, "upper", None, vec![1], [0.0, 0.0, 0.0]),
            node(1, "lower", Some(0), vec![2], [1.0, 0.0, 0.0]),
            node(2, "hand", Some(1), Vec::new(), [1.0, 0.0, 0.0]),
        ],
        skin: None,
        animations: Vec::new(),
        bounds_min: [0.0, 0.0, 0.0],
        bounds_max: [2.0, 0.0, 0.0],
    };
    let action = crate::world::WorldAction {
        id: "reach".to_string(),
        skeleton: "humanoid_v1".to_string(),
        intent: None,
        duration_ms: 1_000,
        poses: Vec::new(),
        iks: vec![crate::world::WorldActionIk {
            root: "upper".to_string(),
            mid: "lower".to_string(),
            end: "hand".to_string(),
            target_x: "1".to_string(),
            target_y: "1".to_string(),
            target_z: "0".to_string(),
            pole_x: None,
            pole_y: None,
            pole_z: None,
            plane: "xy".to_string(),
            bend: "1".to_string(),
            weight: "1".to_string(),
        }],
    };
    let mut overrides = std::collections::HashMap::new();
    super::apply_two_bone_ik_overrides(
        &mesh,
        &action,
        &std::collections::HashMap::new(),
        &[],
        1.0,
        crate::world::WorldTime {
            frame: 0,
            fps: 30.0,
            duration_ms: 1_000,
        },
        &mut overrides,
    )
    .expect("solve analytic IK");
    let matrices = super::global_node_matrices(&mesh, &overrides);
    let hand = super::matrix_translation(matrices[2]);
    assert!((hand[0] - 1.0).abs() < 0.02, "hand x={}", hand[0]);
    assert!((hand[1] - 1.0).abs() < 0.02, "hand y={}", hand[1]);
}

#[test]
fn scene_constraint_two_bone_solver_reaches_sampled_target() {
    let node = |index, name: &str, parent, children, translation| crate::world::GlbNodeData {
        index,
        name: Some(name.to_string()),
        parent,
        children,
        mesh: None,
        skin: None,
        translation,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
        matrix: None,
    };
    let mesh = super::GlbMeshData {
        path: std::path::PathBuf::from("scene-constraint.glb"),
        positions: Vec::new(),
        normals: Vec::new(),
        texcoords: Vec::new(),
        colors: Vec::new(),
        joints: Vec::new(),
        weights: Vec::new(),
        indices: Vec::new(),
        triangles: Vec::new(),
        materials: Vec::new(),
        textures: Vec::new(),
        mesh_names: Vec::new(),
        nodes: vec![
            node(0, "upper_arm_r", None, vec![1], [0.0, 0.0, 0.0]),
            node(1, "forearm_r", Some(0), vec![2], [1.0, 0.0, 0.0]),
            node(2, "hand_r", Some(1), Vec::new(), [1.0, 0.0, 0.0]),
        ],
        skin: None,
        animations: Vec::new(),
        bounds_min: [0.0, 0.0, 0.0],
        bounds_max: [2.0, 1.0, 0.0],
    };
    let mut overrides = std::collections::HashMap::new();
    super::solve_two_bone_constraint(
        &mesh,
        "upper_arm_r",
        "forearm_r",
        "hand_r",
        0,
        1,
        2,
        [1.0, 1.0, 0.0],
        1.0,
        &std::collections::HashMap::new(),
        &mut overrides,
    );
    let matrices = super::global_node_matrices(&mesh, &overrides);
    let hand = super::matrix_translation(matrices[2]);
    assert!((hand[0] - 1.0).abs() < 0.02, "hand x={}", hand[0]);
    assert!((hand[1] - 1.0).abs() < 0.02, "hand y={}", hand[1]);
}

#[test]
fn renders_world_placeholder_frame() {
    let script = r##"<Graph fps={30} duration="2s" size={[320,180]}>
  <World id="stage">
    <Background src="../scene/environments/forest_path_static.png" fit="cover" color="#87c9ff" />
    <Camera target="hero" yaw={curve("0:0:linear,2:360:linear")} distance="3" fov="35" />
    <Actor id="hero" model="../sample_assets/glb/mammuthus_primigenius_blumbach.glb" x="0" y="0" yaw="0" scale="0.001" />
  </World>
  <Present from="stage" />
</Graph>"##;
    let graph = parse_world_graph_script(script).expect("world graph");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/motionloom/world");
    let model = root.join("../sample_assets/glb/mammuthus_primigenius_blumbach.glb");
    if !model.exists() {
        return;
    }
    let frame = pollster::block_on(render_world_frame(&graph, 0, &root)).expect("world frame");
    assert_eq!(frame.width(), 320);
    assert_eq!(frame.height(), 180);
}

#[test]
fn renders_world_directional_character_by_yaw_and_pitch() {
    let root = std::env::temp_dir().join(format!(
        "motionloom_directional_character_test_{}",
        std::process::id()
    ));
    let character_dir = root.join("characters");
    fs::create_dir_all(&character_dir).expect("test character dir");
    let sheet_path = character_dir.join("hero_sheet.png");
    let mut sheet = image::RgbaImage::from_pixel(30, 10, image::Rgba([0, 0, 0, 0]));
    for y in 0..10 {
        for x in 0..10 {
            sheet.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
            sheet.put_pixel(x + 10, y, image::Rgba([0, 255, 0, 255]));
            sheet.put_pixel(x + 20, y, image::Rgba([0, 0, 255, 255]));
        }
    }
    sheet.save(&sheet_path).expect("test sheet png");

    let script = r##"<Graph fps={30} duration="1s" size={[40,20]}>
  <World id="sprite_stage">
    <Background color="#000000" />
    <Camera yaw="0" pitch="0" zoom="1" />
    <DirectionalCharacter id="hero" sheet="characters/hero_sheet.png" x="10" y="10" scale="1" yaw="90">
      <DirectionMap>
        <Direction angle="0" rect={[0,0,10,10]} anchor={[0,0]} />
        <Direction angle="90" rect={[10,0,10,10]} anchor={[0,0]} />
        <Direction name="top" cameraPitch="90" rect={[20,0,10,10]} anchor={[0,0]} />
      </DirectionMap>
    </DirectionalCharacter>
  </World>
  <Present from="sprite_stage" />
</Graph>"##;
    let graph = parse_world_graph_script(script).expect("directional graph");
    let frame =
        pollster::block_on(render_world_frame(&graph, 0, &root)).expect("directional frame");
    assert_eq!(frame.get_pixel(10, 10).0, [0, 255, 0, 255]);

    let top_script = script.replace("pitch=\"0\"", "pitch=\"90\"");
    let graph = parse_world_graph_script(&top_script).expect("top directional graph");
    let frame =
        pollster::block_on(render_world_frame(&graph, 0, &root)).expect("top directional frame");
    assert_eq!(frame.get_pixel(10, 10).0, [0, 0, 255, 255]);

    let scaled_script = script.replace("size={[40,20]}", "size={[40,20]} renderSize={[20,10]}");
    let graph = parse_world_graph_script(&scaled_script).expect("scaled directional graph");
    let frame =
        pollster::block_on(render_world_frame(&graph, 0, &root)).expect("scaled directional frame");
    assert_eq!(frame.width(), 20);
    assert_eq!(frame.height(), 10);
    assert_eq!(frame.get_pixel(5, 5).0, [0, 255, 0, 255]);
}

#[test]
fn renders_directional_character_play_sprite_frames() {
    let root = std::env::temp_dir().join(format!(
        "motionloom_play_sprite_test_{}",
        std::process::id()
    ));
    let character_dir = root.join("characters");
    fs::create_dir_all(&character_dir).expect("test character dir");
    let sheet_path = character_dir.join("runner.png");
    let mut sheet = image::RgbaImage::from_pixel(30, 10, image::Rgba([0, 0, 0, 0]));
    for y in 0..10 {
        for x in 0..10 {
            sheet.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
            sheet.put_pixel(x + 10, y, image::Rgba([0, 255, 0, 255]));
            sheet.put_pixel(x + 20, y, image::Rgba([0, 0, 255, 255]));
        }
    }
    sheet.save(&sheet_path).expect("test play sprite png");

    let script = r##"<Graph fps={1} duration="3s" size={[20,20]}>
  <World id="sprite_stage">
    <Background color="#000000" />
    <Camera yaw="0" pitch="0" zoom="1" />
    <DirectionalCharacter id="hero" sheet="characters/runner.png" x="0" y="0" scale="1" yaw="0">
      <PlaySprite fps="1" loop="true" frameSize={[10,10]} columns="3" frames="3" />
      <DirectionMap>
        <Direction angle="0" rect={[0,0,10,10]} anchor={[0,0]} />
      </DirectionMap>
    </DirectionalCharacter>
  </World>
  <Present from="sprite_stage" />
</Graph>"##;
    let graph = parse_world_graph_script(script).expect("play sprite graph");
    let frame0 =
        pollster::block_on(render_world_frame(&graph, 0, &root)).expect("play sprite frame 0");
    let frame1 =
        pollster::block_on(render_world_frame(&graph, 1, &root)).expect("play sprite frame 1");
    let frame2 =
        pollster::block_on(render_world_frame(&graph, 2, &root)).expect("play sprite frame 2");

    assert_eq!(frame0.get_pixel(0, 0).0, [255, 0, 0, 255]);
    assert_eq!(frame1.get_pixel(0, 0).0, [0, 255, 0, 255]);
    assert_eq!(frame2.get_pixel(0, 0).0, [0, 0, 255, 255]);
}

#[test]
fn renders_split_directional_character_png_with_alpha() {
    let root = std::env::temp_dir().join(format!(
        "motionloom_directional_character_split_test_{}",
        std::process::id()
    ));
    let character_dir = root.join("characters");
    fs::create_dir_all(&character_dir).expect("test character dir");
    let image_path = character_dir.join("hero_front.png");
    let frame = image::RgbaImage::from_pixel(10, 10, image::Rgba([0, 255, 0, 128]));
    frame.save(&image_path).expect("test direction png");

    let script = r##"<Graph fps={30} duration="1s" size={[40,20]} renderSize={[20,10]}>
  <World id="sprite_stage">
    <Background color="#000000" />
    <Camera yaw="0" pitch="0" zoom="1" />
    <DirectionalCharacter id="hero" pathstyle="relative" x="10" y="10" scale="1" yaw="0">
      <DirectionMap>
        <Direction angle="0" image="characters/hero_front.png" anchor={[0,0]} />
      </DirectionMap>
    </DirectionalCharacter>
  </World>
  <Present from="sprite_stage" />
</Graph>"##;
    let graph = parse_world_graph_script(script).expect("split directional graph");
    let rendered =
        pollster::block_on(render_world_frame(&graph, 0, &root)).expect("split directional frame");
    let pixel = rendered.get_pixel(5, 5).0;
    assert_eq!(pixel[0], 0);
    assert!(
        (120..=136).contains(&pixel[1]),
        "expected alpha-blended green, got {pixel:?}"
    );
    assert_eq!(pixel[2], 0);
    assert_eq!(pixel[3], 255);
}

#[test]
fn terrain_rgba_blend_weights_cover_all_four_layers() {
    let equal = super::terrain_blend_weights(&[64, 64, 64, 64], 4);
    assert!(equal.iter().all(|weight| (*weight - 0.25).abs() < 1.0e-6));
    let leaf_litter = super::terrain_blend_weights(&[0, 0, 0, 255], 4);
    assert_eq!(leaf_litter, vec![0.0, 0.0, 0.0, 1.0]);
    let empty = super::terrain_blend_weights(&[0, 0, 0, 0], 4);
    assert_eq!(empty, vec![1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn attachment_rotation_weight_uses_shortest_quaternion_path() {
    let identity = [0.0, 0.0, 0.0, 1.0];
    let same_rotation_with_opposite_sign = [0.0, 0.0, 0.0, -1.0];
    let halfway = super::quat_slerp_shortest(identity, same_rotation_with_opposite_sign, 0.5);
    assert!((halfway[3].abs() - 1.0).abs() < 1.0e-6);
    assert!(halfway[..3].iter().all(|value| value.abs() < 1.0e-6));
}

#[test]
fn attachment_parent_delta_keeps_compound_children_rigid() {
    let quarter_turn = super::actor_yxz_quaternion(0.0, 90.0, 0.0);
    let rotated = super::mat4_transform_point(super::mat4_from_quat(quarter_turn), [1.0, 0.0, 0.0]);
    assert!(rotated[0].abs() < 1.0e-5);
    assert!(rotated[1].abs() < 1.0e-5);
    assert!((rotated[2] + 1.0).abs() < 1.0e-5);
}
