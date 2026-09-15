// =========================================
// =========================================
// crates/motionloom/src/world/render/mod.rs

mod environment;
mod lighting;
mod materials;
mod outline;
mod params;
mod pipeline;
mod resources;
mod shaders;
mod shadows;
mod textures;
use environment::load_environment_image_from_resolved;
use materials::{TextureRole, build_mips};
use params::{pack_gpu_world_lighting, pack_gpu_world_params, pack_ground_grid_params};
use pipeline::create_world_surface_pipeline;
use resources::{align_to_256, gpu_world_vertex_chunk_bytes};
use shaders::{WGPU_GROUND_GRID_SHADER, WGPU_WORLD_DOF_SHADER, WGPU_WORLD_SHADER};
use shadows::fit_rigid_shadow_volume;
#[cfg(test)]
use textures::gpu_world_texture_from_image;
pub(crate) use textures::gpu_world_texture_from_rgba_image;

use std::collections::{HashMap, HashSet};
pub mod pose_diagnostics;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[cfg(target_arch = "wasm32")]
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;

use base64::Engine as _;
use half::f16;
use image::{Rgba, RgbaImage, imageops};
use thiserror::Error;

use crate::asset::{AssetResolver, AssetSource, PathAssetResolver};
use crate::common::gpu_async::{
    BufferMapAsyncFuture, DevicePoller, request_adapter_async, request_device_async,
};
use crate::process::runtime::eval_time_expr;
use crate::scene::render::SceneRenderProfile;
use crate::world::gltf_loader::{
    GlbAlphaMode, GlbAnimationChannelData, GlbAnimationData, GlbAnimationInterpolation,
    GlbAnimationProperty, GlbAnimationValues, GlbDepthWriteMode, GlbLoadError, GlbMaterialData,
    GlbMeshData, GlbTextureData, GlbTriangle, load_glb_animation_data,
    load_glb_animation_data_from_bytes, load_glb_mesh_data, load_glb_mesh_data_from_bytes,
};
use crate::world::model::{
    WorldAction, WorldActionBone, WorldActionPose, WorldActor, WorldApplyAction,
    WorldBackgroundFit, WorldBoneAxis, WorldBoneAxisMap, WorldDirectionFrame,
    WorldDirectionalCharacter, WorldGraph, WorldLightKind, WorldLighting, WorldMaterialStyle,
    WorldModelProfile, WorldNode, WorldPathStyle, WorldPlay, WorldRetargetMap, WorldSpritePlayback,
    WorldTime,
};

/// Keep native profiling instrumentation out of the browser runtime because
/// `std::time::Instant` is unsupported on wasm32-unknown-unknown.
#[cfg(not(target_arch = "wasm32"))]
type ProfileClock = Instant;

#[cfg(target_arch = "wasm32")]
struct ProfileClock;

#[cfg(target_arch = "wasm32")]
impl ProfileClock {
    fn now() -> Self {
        Self
    }

    fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

#[derive(Debug, Error)]
pub enum WorldRenderError {
    #[error("world graph has no presented world '{0}'")]
    MissingWorld(String),
    #[error("world graph has no actor '{0}'")]
    MissingActor(String),
    #[error("failed to load background image {path}: {source}")]
    BackgroundImage {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("failed to load directional character sheet {path}: {source}")]
    DirectionalCharacterImage {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("failed to load primitive material texture {source_ref}: {source}")]
    PrimitiveMaterialTexture {
        source_ref: String,
        #[source]
        source: image::ImageError,
    },
    #[error("directional character sheet does not exist: {0}")]
    MissingDirectionalCharacterImage(PathBuf),
    #[error("failed to load GLB model: {0}")]
    Glb(#[from] GlbLoadError),
    #[error("failed to fetch remote asset {url}: {message}")]
    RemoteAsset { url: String, message: String },
    #[error("invalid inline asset data URI: {message}")]
    InvalidDataUri { message: String },
    #[error("invalid world expression '{expr}': {message}")]
    Expression { expr: String, message: String },
    #[error("failed to create output directory ({path}): {source}")]
    CreateOutputDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to start ffmpeg: {source}")]
    StartFfmpeg {
        #[source]
        source: std::io::Error,
    },
    #[error("ffmpeg stdin was not available")]
    MissingFfmpegStdin,
    #[error("failed to write raw frame to ffmpeg: {source}. ffmpeg stderr: {stderr}")]
    WriteFrame {
        #[source]
        source: std::io::Error,
        stderr: String,
    },
    #[error("failed to wait for ffmpeg: {source}")]
    WaitFfmpeg {
        #[source]
        source: std::io::Error,
    },
    #[error("ffmpeg failed: {stderr}")]
    FfmpegFailed { stderr: String },
    #[error("failed to save PNG frame ({path}): {source}")]
    SavePngFrame {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("world GPU render failed: {message}")]
    GpuRender { message: String },
    #[error("video export is not available on this platform: {message}")]
    VideoExportNotAvailable { message: String },
    #[error("world render cancelled")]
    Cancelled,
}

/// CPU-side timings for the most recently submitted true-3D Scene island.
/// GPU execution time remains available from the parent Scene compositor's
/// timestamp queries; these fields expose work that previously appeared as an
/// unexplained preview stall before queue submission.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
#[non_exhaustive]
pub struct Scene3DFrameProfile {
    pub prepare_ms: f64,
    pub canvas_ms: f64,
    pub background_ms: f64,
    pub actor_build_ms: f64,
    pub asset_resolve_ms: f64,
    pub animation_sample_ms: f64,
    pub constraints_ms: f64,
    pub draw_assembly_ms: f64,
    pub texture_decode_ms: f64,
    pub texture_decode_count: usize,
    pub texture_cache_hits: usize,
    pub texture_decoded_bytes: usize,
    pub renderer_init_ms: f64,
    pub submit_ms: f64,
    pub readback_ms: f64,
    pub draw_calls: usize,
    pub mesh_cache_entries: usize,
    pub static_draw_plans: usize,
    pub gpu_resource_entries: usize,
    pub gpu_texture_resources: usize,
    pub gpu_geometry_resources: usize,
    pub target_pool_size: usize,
    pub visible_triangles: u64,
    pub light_count: usize,
    pub shadow_map_size: u32,
    /// True once a sequential prior frame is available for temporal resolve.
    pub temporal_history_valid: bool,
    pub temporal_antialiasing: bool,
    pub screen_space_reflections: bool,
    pub motion_blur: bool,
    pub anti_aliasing_requested: &'static str,
    pub anti_aliasing_effective: &'static str,
    pub anti_aliasing_fallback_used: bool,
    /// Approximate bytes held by frame-sized color/depth targets and the
    /// active shadow target. Retained mesh/texture caches are reported
    /// separately because their allocations are asset-dependent.
    pub render_target_bytes: u64,
}

impl From<crate::export::EncodeError> for WorldRenderError {
    fn from(err: crate::export::EncodeError) -> Self {
        use crate::export::EncodeError;
        match err {
            EncodeError::Audio(error) => Self::FfmpegFailed {
                stderr: error.to_string(),
            },
            EncodeError::AudioPlan(error) => Self::FfmpegFailed {
                stderr: error.to_string(),
            },
            EncodeError::CreateOutputDir { path, source } => Self::CreateOutputDir { path, source },
            EncodeError::StartEncoder(message) => Self::StartFfmpeg {
                source: std::io::Error::other(message),
            },
            EncodeError::MissingEncoderInput => Self::MissingFfmpegStdin,
            EncodeError::WriteFrame(source) => Self::WriteFrame {
                source,
                stderr: String::new(),
            },
            EncodeError::EncoderFailed(stderr) => Self::FfmpegFailed { stderr },
            EncodeError::NotImplemented(message) => Self::VideoExportNotAvailable { message },
            EncodeError::NotStarted => Self::GpuRender {
                message: "encoder was not started".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldRenderProgress {
    pub rendered_frames: u32,
    pub total_frames: u32,
}

/// One frame-local Scene texture replacing a named GLB material base color.
pub(crate) struct WorldMaterialTextureOverride {
    pub actor_id: String,
    pub material: String,
    pub texture: Arc<GpuWorldTexture>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldGpuDiagnostics {
    pub mesh_loaded: bool,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub material_count: usize,
    pub texture_count: usize,
    pub decoded_texture_count: usize,
    pub skin_joint_count: usize,
    pub gpu_draw_count: usize,
    pub gpu_vertex_count: usize,
    pub bone_override_count: usize,
    pub projected_bounds: Option<String>,
    pub projected_inside_count: usize,
    pub projected_nonfinite_count: usize,
    pub ndc_z_range: Option<String>,
    pub depth_pass_estimate_count: usize,
    pub depth_reject_estimate_count: usize,
    pub alpha_sample_count: usize,
    pub alpha_visible_sample_count: usize,
    pub alpha_zero_sample_count: usize,
    pub alpha_range: Option<String>,
    pub uv_outside_sample_count: usize,
    pub raw_draw_bounds: Option<String>,
    pub shader_local_bounds: Option<String>,
    pub shader_projected_bounds: Option<String>,
    pub shader_projected_inside_count: usize,
    pub shader_projected_nonfinite_count: usize,
    pub shader_joint_oob_count: usize,
    pub skipped_reasons: Vec<String>,
}

pub async fn render_world_frame(
    graph: &WorldGraph,
    frame: u32,
    asset_root: impl AsRef<Path>,
) -> Result<RgbaImage, WorldRenderError> {
    WorldFrameRenderer::new().render_frame(graph, frame, asset_root)
}

pub fn diagnose_world_glb_gpu_plan(mesh: &GlbMeshData) -> WorldGpuDiagnostics {
    let texture_count = mesh.textures.len();
    let decoded_texture_count = mesh
        .textures
        .iter()
        .filter(|texture| texture.is_some())
        .count();
    let skin_joint_count = mesh.skin.as_ref().map_or(0, |skin| skin.joints.len());
    let mut skipped_reasons = Vec::<String>::new();

    if mesh.positions.is_empty() {
        skipped_reasons.push("mesh has 0 positions; actor cannot create GPU vertices".to_string());
    }
    if mesh.triangles.is_empty() {
        skipped_reasons.push("mesh has 0 triangles; actor has nothing to draw".to_string());
    }

    let mut transparent_materials = 0usize;
    let mut missing_texture_materials = 0usize;
    for material in &mesh.materials {
        if material.base_color_factor[3] <= 0.001 {
            transparent_materials += 1;
        }
        if let Some(texture_index) = material.base_color_texture
            && mesh
                .textures
                .get(texture_index)
                .and_then(Option::as_ref)
                .is_none()
        {
            missing_texture_materials += 1;
        }
    }
    if transparent_materials > 0 {
        skipped_reasons.push(format!(
            "{transparent_materials} material(s) have near-zero base alpha; if draw count is non-zero, invisibility may be alpha/material related"
        ));
    }
    if missing_texture_materials > 0 {
        skipped_reasons.push(format!(
            "{missing_texture_materials} material(s) reference missing/undecoded textures; GPU will use flat fallback color"
        ));
    }

    let mut chunks = HashMap::<GpuWorldDrawKey, usize>::new();
    let mut invalid_index_triangles = 0usize;
    let mut missing_uv_textured_triangles = 0usize;
    let mut alpha_sample_count = 0usize;
    let mut alpha_visible_sample_count = 0usize;
    let mut alpha_zero_sample_count = 0usize;
    let mut uv_outside_sample_count = 0usize;
    let mut min_alpha = f32::INFINITY;
    let mut max_alpha = f32::NEG_INFINITY;
    for triangle in &mesh.triangles {
        let has_invalid_index = triangle
            .indices
            .iter()
            .any(|index| *index as usize >= mesh.positions.len());
        if has_invalid_index {
            invalid_index_triangles += 1;
            continue;
        }

        let material = triangle
            .material
            .and_then(|index| mesh.materials.get(index));
        let texture = material
            .and_then(|material| material.base_color_texture)
            .and_then(|texture_index| {
                mesh.textures
                    .get(texture_index)
                    .and_then(Option::as_ref)
                    .map(|_| texture_index)
            });
        if texture.is_some()
            && triangle.indices.iter().any(|index| {
                mesh.texcoords
                    .get(*index as usize)
                    .and_then(|uv| *uv)
                    .is_none()
            })
        {
            missing_uv_textured_triangles += 1;
        }

        let triangle_uvs = [
            mesh.texcoords
                .get(triangle.indices[0] as usize)
                .copied()
                .flatten(),
            mesh.texcoords
                .get(triangle.indices[1] as usize)
                .copied()
                .flatten(),
            mesh.texcoords
                .get(triangle.indices[2] as usize)
                .copied()
                .flatten(),
        ];
        if let Some((texture, material_factor, uvs)) =
            textured_triangle_source(mesh, triangle.material, triangle_uvs)
        {
            let centroid = [
                (uvs[0][0] + uvs[1][0] + uvs[2][0]) / 3.0,
                (uvs[0][1] + uvs[1][1] + uvs[2][1]) / 3.0,
            ];
            for uv in [uvs[0], uvs[1], uvs[2], centroid] {
                if uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0 {
                    uv_outside_sample_count += 1;
                }
                let alpha = sampled_texture_alpha(texture, uv, material_factor);
                alpha_sample_count += 1;
                min_alpha = min_alpha.min(alpha);
                max_alpha = max_alpha.max(alpha);
                if alpha <= 0.001 {
                    alpha_zero_sample_count += 1;
                } else {
                    alpha_visible_sample_count += 1;
                }
            }
        }

        let key = GpuWorldDrawKey {
            material: triangle.material,
            texture,
            mesh: triangle.mesh,
            mesh_node: triangle.mesh_node,
        };
        *chunks.entry(key).or_insert(0) += 3;
    }

    let gpu_draw_count = chunks.values().filter(|vertices| **vertices > 0).count();
    let gpu_vertex_count = chunks.values().sum();
    if invalid_index_triangles > 0 {
        skipped_reasons.push(format!(
            "{invalid_index_triangles} triangle(s) skipped because they reference missing vertex positions"
        ));
    }
    if missing_uv_textured_triangles > 0 {
        skipped_reasons.push(format!(
            "{missing_uv_textured_triangles} textured triangle(s) have missing UVs; shader may sample texture corner"
        ));
    }
    if alpha_sample_count > 0 && alpha_visible_sample_count == 0 {
        skipped_reasons.push(
            "CPU texture alpha probe found 0 visible alpha samples; fragment shader will discard all sampled pixels"
                .to_string(),
        );
    } else if alpha_sample_count > 0 && alpha_visible_sample_count < alpha_sample_count / 20 {
        skipped_reasons.push(format!(
            "CPU texture alpha probe found very few visible samples ({alpha_visible_sample_count}/{alpha_sample_count}); alpha/UV path is suspicious"
        ));
    }
    if uv_outside_sample_count > 0 {
        skipped_reasons.push(format!(
            "{uv_outside_sample_count}/{alpha_sample_count} texture alpha probe sample(s) use UV outside 0..1; shader clamps these to texture edges"
        ));
    }
    if gpu_draw_count == 0 && !mesh.triangles.is_empty() {
        skipped_reasons.push(format!(
            "0 GPU draw chunks generated from {} triangle(s); inspect primitive indices/material grouping",
            mesh.triangles.len()
        ));
    } else if gpu_draw_count > 0 {
        skipped_reasons.push(
            "GPU draw chunks generated; if preview is still blank, inspect alpha/depth/texture/shader path"
                .to_string(),
        );
    }

    WorldGpuDiagnostics {
        mesh_loaded: true,
        vertex_count: mesh.positions.len(),
        triangle_count: mesh.triangles.len(),
        material_count: mesh.materials.len(),
        texture_count,
        decoded_texture_count,
        skin_joint_count,
        gpu_draw_count,
        gpu_vertex_count,
        bone_override_count: 0,
        projected_bounds: None,
        projected_inside_count: 0,
        projected_nonfinite_count: 0,
        ndc_z_range: None,
        depth_pass_estimate_count: 0,
        depth_reject_estimate_count: 0,
        alpha_sample_count,
        alpha_visible_sample_count,
        alpha_zero_sample_count,
        alpha_range: if min_alpha.is_finite() && max_alpha.is_finite() {
            Some(format!("{min_alpha:.3}..{max_alpha:.3}"))
        } else {
            None
        },
        uv_outside_sample_count,
        raw_draw_bounds: None,
        shader_local_bounds: None,
        shader_projected_bounds: None,
        shader_projected_inside_count: 0,
        shader_projected_nonfinite_count: 0,
        shader_joint_oob_count: 0,
        skipped_reasons,
    }
}

pub fn diagnose_world_graph_actor_gpu_frame(
    graph: &WorldGraph,
    actor_id: &str,
    frame: u32,
    asset_root: impl AsRef<Path>,
) -> Result<WorldGpuDiagnostics, WorldRenderError> {
    let asset_root = asset_root.as_ref();
    let world = graph
        .presented_world()
        .ok_or_else(|| WorldRenderError::MissingWorld(graph.present.from.clone()))?;
    let actor = world
        .actors
        .iter()
        .find(|actor| actor.id == actor_id)
        .or_else(|| world.actors.first())
        .ok_or_else(|| WorldRenderError::GpuRender {
            message: "GPU diagnostics found no Actor in presented world".to_string(),
        })?;
    let (model_key, mesh) = load_glb_mesh_resolved(
        asset_root,
        &actor.model,
        actor.path_style,
        &PathAssetResolver,
    )?;
    let mut diagnostics = diagnose_world_glb_gpu_plan(&mesh);
    let time = WorldTime {
        frame,
        fps: graph.fps,
        duration_ms: graph.duration_ms,
    };

    let overrides = actor_bone_overrides_for_mesh(graph, actor, Some(&mesh), time)?;
    diagnostics.bone_override_count = overrides.len();

    let positions = skinned_actor_positions(graph, actor, &mesh, time)?
        .unwrap_or_else(|| mesh.positions.clone());
    let (width, height) = graph.output_size();
    let width_f = width.max(1) as f32;
    let height_f = height.max(1) as f32;
    let camera_yaw = eval_number(&world.camera.yaw, 0.0, time)?;
    let camera_pitch = eval_number(&world.camera.pitch, 0.0, time)?;
    let camera_x = eval_number(&world.camera.x, 0.0, time)?;
    let camera_y = eval_number(&world.camera.y, 0.0, time)?;
    let camera_z = eval_number(&world.camera.z, 0.0, time)?;
    let camera_zoom = eval_number(&world.camera.zoom, 1.0, time)?.max(0.05);
    let fov = eval_number(&world.camera.fov, 35.0, time)?.clamp(10.0, 100.0);
    let distance = eval_number(&world.camera.distance, 3.2, time)?.max(0.2);
    let actor_x = eval_number(&actor.x, 0.0, time)?;
    let actor_y = eval_number(&actor.y, 0.0, time)?;
    let actor_z = eval_number(&actor.z, 0.0, time)?;
    let actor_yaw = eval_number(&actor.yaw, 0.0, time)?;
    let actor_scale = eval_number(&actor.scale, 1.0, time)?.max(0.01) * camera_zoom;
    let view = camera_actor_view(
        actor_x,
        actor_y,
        actor_z,
        actor_yaw,
        camera_x,
        camera_y,
        camera_z,
        camera_yaw,
        camera_pitch,
    );

    let model_height = (mesh.bounds_max[1] - mesh.bounds_min[1]).abs().max(0.001);
    let model_width = (mesh.bounds_max[0] - mesh.bounds_min[0]).abs().max(0.001);
    let model_depth = (mesh.bounds_max[2] - mesh.bounds_min[2]).abs().max(0.001);
    let px_per_world = (height_f / distance) * (35.0 / fov).clamp(0.35, 2.5);
    let normalize_height = actor.scale_mode.eq_ignore_ascii_case("normalize_height");
    let (model_center_x, model_origin_y, model_center_z, model_px) = if normalize_height {
        (
            (mesh.bounds_min[0] + mesh.bounds_max[0]) * 0.5,
            mesh.bounds_min[1],
            (mesh.bounds_min[2] + mesh.bounds_max[2]) * 0.5,
            (height_f * 0.58 * actor_scale * (3.2 / distance).clamp(0.25, 4.0)) / model_height,
        )
    } else {
        (0.0, 0.0, 0.0, px_per_world * actor_scale)
    };
    let cx = width_f * 0.5 + view.x * px_per_world;
    let ground_y = height_f * 0.82 - view.y * px_per_world;
    let yaw = view.yaw.to_radians();
    let cos_y = yaw.cos();
    let sin_y = yaw.sin();
    let pitch = camera_pitch.to_radians();
    let cos_p = pitch.cos();
    let sin_p = pitch.sin();
    let depth_scale = 0.45 / model_width.max(model_depth);

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut inside = 0usize;
    let mut nonfinite = 0usize;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    let mut depth_pass = 0usize;
    let mut depth_reject = 0usize;
    for position in positions {
        let x = position[0] - model_center_x;
        let y = position[1] - model_origin_y;
        let z = position[2] - model_center_z;
        let rx = x * cos_y + z * sin_y;
        let rz = -x * sin_y + z * cos_y;
        let ry = y * cos_p - rz * sin_p;
        let rz = y * sin_p + rz * cos_p + view.depth * WORLD_DEPTH_SORT_SCALE;
        let screen_x = cx + rx * model_px;
        let screen_y = ground_y - ry * model_px;
        let ndc_z = (0.5 + rz * depth_scale).clamp(0.0, 1.0);
        if !screen_x.is_finite() || !screen_y.is_finite() || !ndc_z.is_finite() {
            nonfinite += 1;
            continue;
        }
        min_x = min_x.min(screen_x);
        min_y = min_y.min(screen_y);
        max_x = max_x.max(screen_x);
        max_y = max_y.max(screen_y);
        min_z = min_z.min(ndc_z);
        max_z = max_z.max(ndc_z);
        if ndc_z > 0.0 {
            depth_pass += 1;
        } else {
            depth_reject += 1;
        }
        if screen_x >= 0.0 && screen_x <= width_f && screen_y >= 0.0 && screen_y <= height_f {
            inside += 1;
        }
    }

    diagnostics.projected_inside_count = inside;
    diagnostics.projected_nonfinite_count = nonfinite;
    diagnostics.depth_pass_estimate_count = depth_pass;
    diagnostics.depth_reject_estimate_count = depth_reject;
    if min_z.is_finite() && max_z.is_finite() {
        diagnostics.ndc_z_range = Some(format!("{min_z:.3}..{max_z:.3}"));
        if depth_pass == 0 {
            diagnostics.skipped_reasons.push(
                "estimated GPU depth test rejects all vertices with current Greater/clear(0.0) setup"
                    .to_string(),
            );
        }
    }
    if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite() {
        diagnostics.projected_bounds = Some(format!(
            "x {:.1}..{:.1}, y {:.1}..{:.1} on {}x{}",
            min_x, max_x, min_y, max_y, width, height
        ));
        if inside == 0 {
            diagnostics.skipped_reasons.push(
                "projected screen bbox has 0 vertices inside viewport; inspect actor/camera/skin transform"
                    .to_string(),
            );
        } else {
            diagnostics.skipped_reasons.push(format!(
                "{inside} projected vertex/vertices are inside viewport before GPU shader"
            ));
        }
    } else {
        diagnostics.skipped_reasons.push(
            "projected screen bbox is non-finite/empty; inspect bone matrices and node transforms"
                .to_string(),
        );
    }

    diagnose_gpu_shader_projection(
        graph,
        actor,
        &mesh,
        &model_key,
        width,
        height,
        world,
        time,
        &mut diagnostics,
    )?;

    Ok(diagnostics)
}

#[allow(clippy::too_many_arguments)]
fn diagnose_gpu_shader_projection(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
    model_path: &Path,
    width: u32,
    height: u32,
    world: &WorldNode,
    time: WorldTime,
    diagnostics: &mut WorldGpuDiagnostics,
) -> Result<(), WorldRenderError> {
    let width_f = width.max(1) as f32;
    let height_f = height.max(1) as f32;
    let camera_zoom = eval_number(&world.camera.zoom, 1.0, time)?.max(0.05);
    let camera_view = perspective_camera_view(world, width, height, time)?;
    let actor_x = eval_number(&actor.x, 0.0, time)?;
    let actor_y = eval_number(&actor.y, 0.0, time)?;
    let actor_z = eval_number(&actor.z, 0.0, time)?;
    let actor_yaw = eval_number(&actor.yaw, 0.0, time)?;
    let actor_pitch = eval_number(&actor.pitch, 0.0, time)?;
    let actor_roll = eval_number(&actor.roll, 0.0, time)?;
    let actor_scale = eval_number(&actor.scale, 1.0, time)?.max(0.01) * camera_zoom;
    let actor_opacity = eval_number(&actor.opacity, 1.0, time)?.clamp(0.0, 1.0);
    let static_draws = build_actor_mesh_gpu_static_draws(actor, mesh, model_path);
    let mut skinning_strategy_cache = HashMap::new();
    let draw_calls = build_actor_mesh_gpu_draws(
        graph,
        actor,
        mesh,
        effective_mesh_bounds(mesh),
        &static_draws,
        width,
        height,
        actor_x,
        actor_y,
        actor_z,
        actor_yaw,
        actor_pitch,
        actor_roll,
        camera_view,
        actor_scale,
        actor_opacity,
        time,
        model_path,
        &mut skinning_strategy_cache,
        &[],
        &HashMap::new(),
        &HashMap::new(),
        &HashMap::new(),
    )?;

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut inside = 0usize;
    let mut nonfinite = 0usize;
    let mut joint_oob = 0usize;
    let mut raw_min = [f32::INFINITY; 3];
    let mut raw_max = [f32::NEG_INFINITY; 3];
    let mut local_min = [f32::INFINITY; 3];
    let mut local_max = [f32::NEG_INFINITY; 3];
    for draw in &draw_calls {
        for vertex in draw.vertices.iter() {
            accumulate_bounds3(&mut raw_min, &mut raw_max, vertex.position);
            let skinned = simulate_gpu_vertex_skinning(vertex, &draw.bone_matrices, &mut joint_oob);
            accumulate_bounds3(&mut local_min, &mut local_max, skinned);
            let local = [
                (skinned[0] - draw.params.model[0]) * draw.params.model[3],
                (skinned[1] - draw.params.model[1]) * draw.params.model[3],
                (skinned[2] - draw.params.model[2]) * draw.params.model[3],
            ];
            let actor_cos = draw.params.actor[3].cos();
            let actor_sin = draw.params.actor[3].sin();
            let world = [
                draw.params.actor[0] + local[0] * actor_cos + local[2] * actor_sin,
                draw.params.actor[1] + local[1],
                draw.params.actor[2] - local[0] * actor_sin + local[2] * actor_cos,
            ];
            let rel = [
                world[0] - draw.params.camera0[0],
                world[1] - draw.params.camera0[1],
                world[2] - draw.params.camera0[2],
            ];
            let view_x = dot3(
                rel,
                [
                    draw.params.camera1[0],
                    draw.params.camera1[1],
                    draw.params.camera1[2],
                ],
            );
            let view_y = dot3(
                rel,
                [
                    draw.params.camera2[0],
                    draw.params.camera2[1],
                    draw.params.camera2[2],
                ],
            );
            let view_z = dot3(
                rel,
                [
                    draw.params.camera3[0],
                    draw.params.camera3[1],
                    draw.params.camera3[2],
                ],
            )
            .max(draw.params.camera1[3]);
            let screen_x = draw.params.canvas[2] + view_x * draw.params.camera0[3] / view_z;
            let screen_y = draw.params.canvas[3] - view_y * draw.params.camera0[3] / view_z;
            if !screen_x.is_finite() || !screen_y.is_finite() {
                nonfinite += 1;
                continue;
            }
            min_x = min_x.min(screen_x);
            min_y = min_y.min(screen_y);
            max_x = max_x.max(screen_x);
            max_y = max_y.max(screen_y);
            if screen_x >= 0.0 && screen_x <= width_f && screen_y >= 0.0 && screen_y <= height_f {
                inside += 1;
            }
        }
    }
    if raw_min[0].is_finite() && raw_max[0].is_finite() {
        diagnostics.raw_draw_bounds = Some(format_bounds3(raw_min, raw_max));
    }
    if local_min[0].is_finite() && local_max[0].is_finite() {
        diagnostics.shader_local_bounds = Some(format_bounds3(local_min, local_max));
    }
    diagnostics.shader_projected_inside_count = inside;
    diagnostics.shader_projected_nonfinite_count = nonfinite;
    diagnostics.shader_joint_oob_count = joint_oob;
    if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite() {
        diagnostics.shader_projected_bounds = Some(format!(
            "x {:.1}..{:.1}, y {:.1}..{:.1} on {}x{}",
            min_x, max_x, min_y, max_y, width, height
        ));
        if inside == 0 {
            diagnostics.skipped_reasons.push(
                "simulated GPU vertex shader projects 0 vertices inside viewport; inspect joint matrices / mesh inverse / actor transform"
                    .to_string(),
            );
        }
    } else {
        diagnostics.skipped_reasons.push(
            "simulated GPU vertex shader projection is empty/non-finite; inspect joint matrices and vertex attributes"
                .to_string(),
        );
    }
    if joint_oob > 0 {
        diagnostics.skipped_reasons.push(format!(
            "simulated GPU shader saw {joint_oob} joint reference(s) outside current bone matrix buffer"
        ));
    }
    Ok(())
}

fn simulate_gpu_vertex_skinning(
    vertex: &GpuWorldVertex,
    bone_matrices: &[[f32; 16]],
    joint_oob: &mut usize,
) -> [f32; 3] {
    let weight_sum = vertex.weights[0] + vertex.weights[1] + vertex.weights[2] + vertex.weights[3];
    if weight_sum <= 0.000001 {
        return vertex.position;
    }
    let mut out = [0.0f32; 3];
    for slot in 0..4 {
        let weight = vertex.weights[slot] / weight_sum;
        if weight <= 0.0 {
            continue;
        }
        let joint_index = (vertex.joints[slot] + 0.5).max(0.0) as usize;
        let Some(matrix) = bone_matrices.get(joint_index) else {
            *joint_oob += 1;
            continue;
        };
        let transformed = mat4_transform_point(*matrix, vertex.position);
        out[0] += transformed[0] * weight;
        out[1] += transformed[1] * weight;
        out[2] += transformed[2] * weight;
    }
    out
}

fn accumulate_bounds3(min: &mut [f32; 3], max: &mut [f32; 3], point: [f32; 3]) {
    for axis in 0..3 {
        min[axis] = min[axis].min(point[axis]);
        max[axis] = max[axis].max(point[axis]);
    }
}

fn format_bounds3(min: [f32; 3], max: [f32; 3]) -> String {
    format!(
        "x {:.3}..{:.3}, y {:.3}..{:.3}, z {:.3}..{:.3}",
        min[0], max[0], min[1], max[1], min[2], max[2]
    )
}

// Preparation and actor assembly keep their existing tuple layout at internal call sites.
type PreparedGpuFrame = (
    Option<RgbaImage>,
    u32,
    u32,
    Vec<GpuWorldDraw>,
    Option<GpuGroundGridParams>,
    GpuWorldLighting,
);
type ActorGpuDrawBuild = (
    Vec<GpuWorldDraw>,
    ActorBuildStages,
    Vec<Scene3DEditorJointProjection>,
    Vec<crate::rig_diagnostics::RigEvaluationReport>,
);

pub struct WorldFrameRenderer {
    asset_resolver: Arc<dyn AssetResolver>,
    image_cache: HashMap<PathBuf, RgbaImage>,
    environment_cache: HashMap<PathBuf, Arc<WorldEnvironmentImage>>,
    mesh_cache: HashMap<PathBuf, GlbMeshData>,
    primitive_texture_cache: HashMap<PrimitiveTextureSourceKey, Arc<GlbTextureData>>,
    effective_bounds_cache: HashMap<PathBuf, ([f32; 3], [f32; 3])>,
    gpu_static_draw_cache: HashMap<GpuWorldStaticPlanKey, Vec<GpuWorldStaticDraw>>,
    skinning_strategy_cache: HashMap<SkinningStrategyKey, SkinningMatrixStrategy>,
    humanoid_rig_metrics_cache: HashMap<HumanoidRigMetricsKey, Scene3DHumanoidRigMetrics>,
    gpu_renderer: Option<GpuWorldRenderer>,
    /// Shared Scene GPU context used by the internal 3D backend.
    gpu_device_queue: Option<(Arc<wgpu::Device>, wgpu::Queue)>,
    last_frame_profile: Scene3DFrameProfile,
    last_prepare_stages: Scene3DPrepareStages,
    last_prepared_draw_stats: PreparedDrawStats,
    last_editor_rig_snapshot: Option<Scene3DEditorRigSnapshot>,
    /// Browser picking needs this continuously; native validation enables it
    /// only for sampled frames so live preview keeps its fast path.
    collect_editor_rig_snapshot: bool,
    /// Full stage/provenance reports are opt-in even on WASM; normal editor
    /// joint picking must not pay for diagnostic matrix reconstruction.
    collect_rig_diagnostics: bool,
    immediate_preview_settings: crate::preview::ImmediatePreviewSettings,
}

#[derive(Clone, Copy, Debug, Default)]
struct Scene3DPrepareStages {
    canvas_ms: f64,
    background_ms: f64,
    actor_build_ms: f64,
    actor: ActorBuildStages,
}

#[derive(Clone, Copy, Debug, Default)]
struct ActorBuildStages {
    asset_resolve_ms: f64,
    animation_sample_ms: f64,
    constraints_ms: f64,
    draw_assembly_ms: f64,
    texture_decode_ms: f64,
    texture_decode_count: usize,
    texture_cache_hits: usize,
    texture_decoded_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct PreparedDrawStats {
    visible_triangles: u64,
    light_count: usize,
    anti_aliasing_requested: &'static str,
    anti_aliasing_effective: &'static str,
    anti_aliasing_fallback_used: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PrimitiveTextureSourceKey {
    identity: PathBuf,
    revision: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct PrimitiveResourceLoadStats {
    texture_decode_ms: f64,
    texture_decode_count: usize,
    texture_cache_hits: usize,
    texture_decoded_bytes: usize,
}

/// Internal Scene 3D backend name; the legacy World type remains for API compatibility.
pub(crate) type Scene3DRenderer = WorldFrameRenderer;

mod geometry;

/// Screen-space canonical joints from the exact pose and camera used by the
/// most recent 3D render. Editor hosts use these for real model picking; they
/// are not a separately-authored or approximate stick figure.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub(crate) struct Scene3DEditorRigSnapshot {
    pub width: u32,
    pub height: u32,
    pub joints: Vec<Scene3DEditorJointProjection>,
    #[serde(skip)]
    pub rig_reports: Vec<crate::rig_diagnostics::RigEvaluationReport>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Scene3DEditorJointProjection {
    pub actor: String,
    pub bone: String,
    pub x: f32,
    pub y: f32,
    pub depth: f32,
}

/// Canonical humanoid joints sampled from the exact animation/retarget path
/// used by the GPU renderer. Scene collision consumes this lightweight frame
/// instead of reading back skinned vertices from the GPU.
#[derive(Clone, Debug, Default)]
pub(crate) struct Scene3DHumanoidFrame {
    pub joints: HashMap<String, [f32; 3]>,
    pub actor_scale: f32,
    /// Bind-pose calibration keeps floor queries independent from the current
    /// animation pose and from each GLB author's choice of model origin.
    pub rig_metrics: Scene3DHumanoidRigMetrics,
    /// Normalized phases for active imported Actions, sampled from the same
    /// GLB clip timing path as the renderer.
    pub action_phases: HashMap<String, f32>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct HumanoidRigMetricsKey {
    model: PathBuf,
    profile: String,
}

/// Normalized bind-pose measurements. Values use MotionLoom's one-unit model
/// height convention and are scaled per actor when sampled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scene3DHumanoidRigMetrics {
    pub sole_offset: f32,
    pub knee_offset: f32,
    pub hips_offset: f32,
    pub head_offset: f32,
    pub body_height: f32,
}

impl Default for Scene3DHumanoidRigMetrics {
    fn default() -> Self {
        Self {
            sole_offset: 0.0,
            knee_offset: 0.28,
            hips_offset: 0.52,
            head_offset: 0.94,
            body_height: 1.0,
        }
    }
}

impl Scene3DHumanoidRigMetrics {
    fn scaled(self, scale: f32) -> Self {
        let scale = scale.max(0.01);
        Self {
            sole_offset: self.sole_offset * scale,
            knee_offset: self.knee_offset * scale,
            hips_offset: self.hips_offset * scale,
            head_offset: self.head_offset * scale,
            body_height: self.body_height * scale,
        }
    }
}

impl Default for WorldFrameRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldFrameRenderer {
    pub fn new() -> Self {
        Self::with_resolver(Arc::new(PathAssetResolver))
    }

    pub fn with_resolver(asset_resolver: Arc<dyn AssetResolver>) -> Self {
        Self {
            asset_resolver,
            image_cache: HashMap::new(),
            environment_cache: HashMap::new(),
            mesh_cache: HashMap::new(),
            primitive_texture_cache: HashMap::new(),
            effective_bounds_cache: HashMap::new(),
            gpu_static_draw_cache: HashMap::new(),
            skinning_strategy_cache: HashMap::new(),
            humanoid_rig_metrics_cache: HashMap::new(),
            gpu_renderer: None,
            gpu_device_queue: None,
            last_frame_profile: Scene3DFrameProfile::default(),
            last_prepare_stages: Scene3DPrepareStages::default(),
            last_prepared_draw_stats: PreparedDrawStats::default(),
            last_editor_rig_snapshot: None,
            collect_editor_rig_snapshot: cfg!(target_arch = "wasm32"),
            collect_rig_diagnostics: false,
            immediate_preview_settings: crate::preview::ImmediatePreviewSettings::default(),
        }
    }

    pub(crate) fn set_immediate_preview_settings(
        &mut self,
        settings: crate::preview::ImmediatePreviewSettings,
    ) {
        let settings = settings.normalized();
        if settings != self.immediate_preview_settings {
            if let Some(renderer) = self.gpu_renderer.as_mut() {
                renderer.reset_temporal_history();
            }
            self.immediate_preview_settings = settings;
        }
    }

    pub fn last_frame_profile(&self) -> Scene3DFrameProfile {
        self.last_frame_profile
    }

    pub(crate) fn last_editor_rig_snapshot(&self) -> Option<&Scene3DEditorRigSnapshot> {
        self.last_editor_rig_snapshot.as_ref()
    }

    pub(crate) fn set_collect_editor_rig_snapshot(&mut self, enabled: bool) {
        self.collect_editor_rig_snapshot = enabled || cfg!(target_arch = "wasm32");
        if !self.collect_editor_rig_snapshot {
            self.last_editor_rig_snapshot = None;
        }
    }

    pub(crate) fn set_collect_rig_diagnostics(&mut self, enabled: bool) {
        self.collect_rig_diagnostics = enabled;
        if !enabled && !self.collect_editor_rig_snapshot {
            self.last_editor_rig_snapshot = None;
        }
    }

    pub(crate) fn collects_rig_diagnostics(&self) -> bool {
        self.collect_rig_diagnostics
    }

    /// Sample canonical humanoid joints without submitting a render pass.
    /// Asset and animation meshes share the renderer's retained GLB cache.
    pub(crate) fn sample_humanoid_frame(
        &mut self,
        graph: &WorldGraph,
        actor_id: &str,
        frame: u32,
        asset_root: impl AsRef<Path>,
    ) -> Result<Scene3DHumanoidFrame, WorldRenderError> {
        let asset_root = asset_root.as_ref();
        let world = graph
            .presented_world()
            .ok_or_else(|| WorldRenderError::MissingWorld(graph.present.from.clone()))?;
        let actor = world
            .actors
            .iter()
            .find(|actor| actor.id == actor_id)
            .ok_or_else(|| WorldRenderError::MissingActor(actor_id.to_string()))?;
        let time = WorldTime {
            frame,
            fps: graph.fps,
            duration_ms: graph.duration_ms,
        };
        let mut animation_keys = HashMap::<String, PathBuf>::new();
        for asset in &graph.animation_assets {
            let (key, _) = load_cached_glb_animation_resolved(
                asset_root,
                &asset.src,
                WorldPathStyle::Relative,
                self.asset_resolver.as_ref(),
                &mut self.mesh_cache,
            )?;
            animation_keys.insert(asset.id.clone(), key);
        }
        let (model_key, _) = load_cached_actor_mesh(
            asset_root,
            actor,
            self.asset_resolver.as_ref(),
            &mut self.mesh_cache,
            &mut self.primitive_texture_cache,
            &mut PrimitiveResourceLoadStats::default(),
        )?;
        let mesh = self
            .mesh_cache
            .get(&model_key)
            .expect("model mesh inserted before humanoid collision sampling");
        let profile = actor_model_profile(graph, actor);
        let metrics_key = HumanoidRigMetricsKey {
            model: model_key.clone(),
            profile: profile.map(|value| value.id.clone()).unwrap_or_default(),
        };
        let rig_metrics = if let Some(metrics) = self.humanoid_rig_metrics_cache.get(&metrics_key) {
            *metrics
        } else {
            let metrics = humanoid_bind_pose_metrics(mesh, profile);
            self.humanoid_rig_metrics_cache.insert(metrics_key, metrics);
            metrics
        };
        let sampled = sample_external_actor_actions(
            graph,
            actor,
            mesh,
            &animation_keys,
            &self.mesh_cache,
            time,
        )?;
        let mut action_phases = HashMap::<String, f32>::new();
        // Canonical Pose Actions need the same phase metadata as imported
        // clips so Scene Contact declarations can drive grounding and IK.
        for apply in graph
            .apply_actions
            .iter()
            .filter(|apply| apply.target == actor.id)
        {
            let Some(action) = graph
                .actions
                .iter()
                .find(|action| action.id == apply.action)
            else {
                continue;
            };
            if let Some(phase) = authored_action_phase(action, apply, time)? {
                action_phases.insert(apply.action.clone(), phase);
            }
        }
        for apply in graph
            .apply_actions
            .iter()
            .filter(|apply| apply.target == actor.id)
        {
            let Some(asset) = graph
                .animation_assets
                .iter()
                .find(|asset| asset.id == apply.action)
            else {
                continue;
            };
            let Some(source_mesh) = animation_keys
                .get(&asset.id)
                .and_then(|key| self.mesh_cache.get(key))
            else {
                continue;
            };
            let animation = if let Some(name) = asset.clip.as_deref() {
                source_mesh
                    .animations
                    .iter()
                    .find(|animation| animation.name.as_deref() == Some(name))
            } else {
                source_mesh.animations.first()
            };
            let Some(animation) = animation else {
                continue;
            };
            let speed = eval_number(&apply.speed, 1.0, time)?.max(0.0);
            if let Some((clip_time, clip_duration)) =
                external_action_clip_time(apply, animation, speed, time)
                && clip_duration > f32::EPSILON
            {
                action_phases.insert(
                    apply.action.clone(),
                    (clip_time / clip_duration).clamp(0.0, 1.0),
                );
            }
        }
        let overrides = actor_bone_overrides_for_mesh(graph, actor, Some(mesh), time)?;
        let matrices = global_node_matrices_with_sampled(mesh, &overrides, &sampled);
        let pose = actor_frame_pose(actor, time)?;
        let joints = canonical_humanoid_editor_bones()
            .iter()
            .copied()
            .filter_map(|bone| {
                let node_index = target_node_for_canonical_bone(mesh, profile, bone)?;
                let matrix = matrices.get(node_index).copied()?;
                Some((
                    bone.to_string(),
                    actor_model_point_to_world(matrix_translation(matrix), mesh, pose),
                ))
            })
            .collect();
        Ok(Scene3DHumanoidFrame {
            joints,
            actor_scale: pose.scale,
            rig_metrics: rig_metrics.scaled(pose.scale),
            action_phases,
        })
    }

    /// Attach the 3D backend to the Scene compositor's device for zero-copy handoff.
    pub(crate) fn set_gpu_context(&mut self, device: Arc<wgpu::Device>, queue: wgpu::Queue) {
        let same_device = self
            .gpu_device_queue
            .as_ref()
            .is_some_and(|(current, _)| Arc::ptr_eq(current, &device));
        if !same_device {
            self.gpu_renderer = None;
            self.gpu_device_queue = Some((device, queue));
        }
    }

    pub fn render_frame(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
    ) -> Result<RgbaImage, WorldRenderError> {
        let asset_root = asset_root.as_ref();
        let (width, height) = graph.output_size();
        let mut canvas = RgbaImage::from_pixel(width.max(1), height.max(1), Rgba([0, 0, 0, 255]));
        let world = graph
            .presented_world()
            .ok_or_else(|| WorldRenderError::MissingWorld(graph.present.from.clone()))?;
        let time = WorldTime {
            frame,
            fps: graph.fps,
            duration_ms: graph.duration_ms,
        };

        let resolver = self.asset_resolver.as_ref();
        draw_world_background(
            &mut canvas,
            world,
            asset_root,
            resolver,
            time,
            &mut self.image_cache,
        )?;
        draw_directional_characters(
            &mut canvas,
            world,
            graph.size,
            asset_root,
            resolver,
            time,
            &mut self.image_cache,
        )?;
        draw_actor_debug_projections(
            &mut canvas,
            graph,
            world,
            asset_root,
            resolver,
            time,
            &mut self.mesh_cache,
            &mut self.primitive_texture_cache,
        )?;
        Ok(canvas)
    }

    pub async fn render_frame_gpu(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
    ) -> Result<RgbaImage, WorldRenderError> {
        self.render_frame_gpu_internal(graph, frame, asset_root, false, false, &[])
            .await
    }

    /// Render the internal Scene 3D island without crossing back through CPU memory.
    pub(crate) async fn render_frame_to_gpu_texture_with_material_overrides(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
        material_overrides: &[WorldMaterialTextureOverride],
    ) -> Result<crate::scene::preview_surface::GpuFrameTexture, WorldRenderError> {
        self.render_frame_to_gpu_texture_internal(
            graph,
            frame,
            asset_root,
            false,
            false,
            material_overrides,
        )
        .await
    }

    pub async fn render_frame_gpu_with_ground_grid(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
    ) -> Result<RgbaImage, WorldRenderError> {
        self.render_frame_gpu_internal(graph, frame, asset_root, true, false, &[])
            .await
    }

    pub async fn render_frame_gpu_with_ground_grid_mode(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
        debug_grid: bool,
    ) -> Result<RgbaImage, WorldRenderError> {
        self.render_frame_gpu_internal(graph, frame, asset_root, true, debug_grid, &[])
            .await
    }

    async fn render_frame_gpu_internal(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
        ground_grid: bool,
        ground_grid_debug: bool,
        material_overrides: &[WorldMaterialTextureOverride],
    ) -> Result<RgbaImage, WorldRenderError> {
        let prepare_started = ProfileClock::now();
        let (canvas, width, height, draw_calls, grid_params, lighting) = self.prepare_gpu_frame(
            graph,
            frame,
            asset_root.as_ref(),
            ground_grid,
            ground_grid_debug,
            material_overrides,
            true,
        )?;
        let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1000.0;
        if draw_calls.is_empty() && !ground_grid {
            return Ok(canvas.expect("readback frame requested a CPU background"));
        }
        let init_started = ProfileClock::now();
        self.ensure_gpu_renderer(width, height, lighting.params.preview0[0] > 0.5)
            .await?;
        let renderer_init_ms = init_started.elapsed().as_secs_f64() * 1000.0;
        let render_started = ProfileClock::now();
        let rendered = self
            .gpu_renderer
            .as_mut()
            .expect("GPU renderer initialized above")
            .render_readback(
                canvas
                    .as_ref()
                    .expect("readback frame requested a CPU background"),
                &draw_calls,
                grid_params,
                &lighting,
            )
            .await?;
        let readback_ms = render_started.elapsed().as_secs_f64() * 1000.0;
        self.update_last_frame_profile(
            prepare_ms,
            renderer_init_ms,
            0.0,
            readback_ms,
            draw_calls.len(),
        );
        Ok(rendered)
    }

    async fn render_frame_to_gpu_texture_internal(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
        ground_grid: bool,
        ground_grid_debug: bool,
        material_overrides: &[WorldMaterialTextureOverride],
    ) -> Result<crate::scene::preview_surface::GpuFrameTexture, WorldRenderError> {
        let prepare_started = ProfileClock::now();
        let (canvas, width, height, draw_calls, grid_params, lighting) = self.prepare_gpu_frame(
            graph,
            frame,
            asset_root.as_ref(),
            ground_grid,
            ground_grid_debug,
            material_overrides,
            false,
        )?;
        let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1000.0;
        let init_started = ProfileClock::now();
        self.ensure_gpu_renderer(width, height, lighting.params.preview0[0] > 0.5)
            .await?;
        let renderer_init_ms = init_started.elapsed().as_secs_f64() * 1000.0;
        let submit_started = ProfileClock::now();
        let texture = self
            .gpu_renderer
            .as_mut()
            .expect("GPU renderer initialized above")
            .render_to_texture(
                canvas.as_ref(),
                &draw_calls,
                grid_params,
                &lighting,
                Some(wgpu::Color::TRANSPARENT),
            )?;
        let submit_ms = submit_started.elapsed().as_secs_f64() * 1000.0;
        self.update_last_frame_profile(
            prepare_ms,
            renderer_init_ms,
            submit_ms,
            0.0,
            draw_calls.len(),
        );
        Ok(texture)
    }

    fn update_last_frame_profile(
        &mut self,
        prepare_ms: f64,
        renderer_init_ms: f64,
        submit_ms: f64,
        readback_ms: f64,
        draw_calls: usize,
    ) {
        let (gpu_resource_entries, gpu_texture_resources, gpu_geometry_resources, target_pool_size) =
            self.gpu_renderer
                .as_ref()
                .map(|renderer| {
                    (
                        renderer.actor_resource_cache.len(),
                        renderer.texture_resource_cache.len(),
                        renderer.vertex_resource_cache.len(),
                        renderer.targets.len(),
                    )
                })
                .unwrap_or_default();
        let visible_triangles = self.last_prepared_draw_stats.visible_triangles;
        let light_count = self.last_prepared_draw_stats.light_count;
        let shadow_map_size = self.immediate_preview_settings.budget().shadow_map_size;
        let preview_budget = self.immediate_preview_settings.budget();
        let temporal_history_valid = self
            .gpu_renderer
            .as_ref()
            .is_some_and(|renderer| renderer.history_valid);
        let (target_width, target_height) = self
            .gpu_renderer
            .as_ref()
            .map(|renderer| (renderer.width as u64, renderer.height as u64))
            .unwrap_or_default();
        let frame_pixels = target_width.saturating_mul(target_height);
        let temporal_bytes_per_pixel = if self
            .gpu_renderer
            .as_ref()
            .is_some_and(|renderer| renderer.temporal_history_enabled)
        {
            16
        } else {
            0
        };
        let render_target_bytes = frame_pixels
            // HDR scene, current/history depth, colour/geometry history,
            // transmission, normal/velocity and material MRT attachments,
            // plus the pooled display targets.
            .saturating_mul(32 + temporal_bytes_per_pixel + target_pool_size as u64 * 4)
            .saturating_add(
                (shadow_map_size as u64)
                    .saturating_mul(shadow_map_size as u64)
                    .saturating_mul(4),
            );
        self.last_frame_profile = Scene3DFrameProfile {
            prepare_ms,
            canvas_ms: self.last_prepare_stages.canvas_ms,
            background_ms: self.last_prepare_stages.background_ms,
            actor_build_ms: self.last_prepare_stages.actor_build_ms,
            asset_resolve_ms: self.last_prepare_stages.actor.asset_resolve_ms,
            animation_sample_ms: self.last_prepare_stages.actor.animation_sample_ms,
            constraints_ms: self.last_prepare_stages.actor.constraints_ms,
            draw_assembly_ms: self.last_prepare_stages.actor.draw_assembly_ms,
            texture_decode_ms: self.last_prepare_stages.actor.texture_decode_ms,
            texture_decode_count: self.last_prepare_stages.actor.texture_decode_count,
            texture_cache_hits: self.last_prepare_stages.actor.texture_cache_hits,
            texture_decoded_bytes: self.last_prepare_stages.actor.texture_decoded_bytes,
            renderer_init_ms,
            submit_ms,
            readback_ms,
            draw_calls,
            mesh_cache_entries: self.mesh_cache.len(),
            static_draw_plans: self.gpu_static_draw_cache.len(),
            gpu_resource_entries,
            gpu_texture_resources,
            gpu_geometry_resources,
            target_pool_size,
            visible_triangles,
            light_count,
            shadow_map_size,
            temporal_history_valid,
            temporal_antialiasing: self
                .gpu_renderer
                .as_ref()
                .is_some_and(|renderer| renderer.temporal_history_enabled),
            screen_space_reflections: preview_budget.screen_space_reflections,
            motion_blur: preview_budget.motion_blur
                && self
                    .gpu_renderer
                    .as_ref()
                    .is_some_and(|renderer| renderer.temporal_history_enabled),
            anti_aliasing_requested: self.last_prepared_draw_stats.anti_aliasing_requested,
            anti_aliasing_effective: self.last_prepared_draw_stats.anti_aliasing_effective,
            anti_aliasing_fallback_used: self.last_prepared_draw_stats.anti_aliasing_fallback_used,
            render_target_bytes,
        };
    }

    /// Build CPU-authored background and cached GLB draws before GPU submission.
    #[expect(
        clippy::too_many_arguments,
        reason = "Preparation receives independent debug, background and material options."
    )]
    fn prepare_gpu_frame(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: &Path,
        ground_grid: bool,
        ground_grid_debug: bool,
        material_overrides: &[WorldMaterialTextureOverride],
        include_cpu_background: bool,
    ) -> Result<PreparedGpuFrame, WorldRenderError> {
        let (width, height) = graph.output_size();
        let width = width.max(1);
        let height = height.max(1);
        let canvas_started = ProfileClock::now();
        let mut canvas = include_cpu_background
            .then(|| RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 255])));
        let canvas_ms = canvas_started.elapsed().as_secs_f64() * 1000.0;
        let world = graph
            .presented_world()
            .ok_or_else(|| WorldRenderError::MissingWorld(graph.present.from.clone()))?;
        let time = WorldTime {
            frame,
            fps: graph.fps,
            duration_ms: graph.duration_ms,
        };

        let resolver = self.asset_resolver.as_ref();
        let background_started = ProfileClock::now();
        if let Some(canvas) = canvas.as_mut() {
            draw_world_background(
                canvas,
                world,
                asset_root,
                resolver,
                time,
                &mut self.image_cache,
            )?;
            draw_directional_characters(
                canvas,
                world,
                graph.size,
                asset_root,
                resolver,
                time,
                &mut self.image_cache,
            )?;
        }
        let background_ms = background_started.elapsed().as_secs_f64() * 1000.0;
        let actor_started = ProfileClock::now();
        let (draw_calls, actor, editor_joints, rig_reports) = build_actor_gpu_draws(
            canvas.as_mut(),
            width,
            height,
            self.collect_editor_rig_snapshot,
            self.collect_rig_diagnostics,
            graph,
            world,
            asset_root,
            resolver,
            time,
            &mut self.mesh_cache,
            &mut self.primitive_texture_cache,
            &mut self.effective_bounds_cache,
            &mut self.gpu_static_draw_cache,
            &mut self.skinning_strategy_cache,
            material_overrides,
        )?;
        let actor_build_ms = actor_started.elapsed().as_secs_f64() * 1000.0;
        self.last_prepare_stages = Scene3DPrepareStages {
            canvas_ms,
            background_ms,
            actor_build_ms,
            actor,
        };
        self.last_editor_rig_snapshot = Some(Scene3DEditorRigSnapshot {
            width,
            height,
            joints: editor_joints,
            rig_reports,
        });
        let camera_view = perspective_camera_view(world, width, height, time)?;
        let grid_params = if ground_grid {
            Some(if ground_grid_debug {
                GpuGroundGridParams::debug_from_camera(width, height, camera_view)
            } else {
                GpuGroundGridParams::from_camera(width, height, camera_view)
            })
        } else {
            None
        };
        let mut lighting = self.prepare_gpu_lighting(&graph.lighting, asset_root, camera_view)?;
        let budget = self.immediate_preview_settings.budget();
        let anti_aliasing = resolve_effective_anti_aliasing(
            graph.lighting.render_style.as_ref(),
            self.immediate_preview_settings,
        );
        lighting.params.surface3[2] = budget.shadow_map_size as f32;
        lighting.params.surface3[3] = anti_aliasing.spatial_selector;
        lighting.params.render_compat[2] = anti_aliasing.quality_selector;
        lighting.params.render_compat[3] = anti_aliasing.sharpness;
        lighting.params.environment2[1] =
            lighting.params.environment2[1].min(budget.max_lights as f32);
        lighting.params.render_compat[1] = budget.screen_space_ao as u8 as f32;
        lighting.params.preview0 = [
            anti_aliasing.temporal as u8 as f32,
            budget.screen_space_reflections as u8 as f32,
            budget.motion_blur as u8 as f32,
            0.0,
        ];
        lighting.frame_index = frame;
        lighting.temporal_jitter = anti_aliasing.jitter_phases > 1;
        if lighting.params.dof_style[0] > 0.5 {
            lighting.params.dof_style[1] =
                lighting.params.dof_style[1].min(budget.dof_sample_limit as f32);
        }
        self.last_prepared_draw_stats = PreparedDrawStats {
            visible_triangles: draw_calls
                .iter()
                .map(|draw| draw.indices.len() as u64 / 3)
                .sum(),
            light_count: lighting.params.environment2[1].max(0.0) as usize,
            anti_aliasing_requested: anti_aliasing.requested,
            anti_aliasing_effective: anti_aliasing.effective,
            anti_aliasing_fallback_used: anti_aliasing.fallback_used,
        };
        Ok((canvas, width, height, draw_calls, grid_params, lighting))
    }

    /// Resolve and decode one environment map, then retain its linear mip chain
    /// for subsequent frames. Missing maps fail explicitly instead of silently
    /// falling back to the legacy studio lights.
    fn prepare_gpu_lighting(
        &mut self,
        lighting: &WorldLighting,
        asset_root: &Path,
        camera: PerspectiveCameraView,
    ) -> Result<GpuWorldLighting, WorldRenderError> {
        let Some(environment) = lighting.environment.as_ref() else {
            let mut fallback = GpuWorldLighting::fallback(camera);
            fallback.params = GpuWorldLightingParams::from_world(lighting, camera, false, 1);
            return Ok(fallback);
        };
        // URL and memory assets use their authored source as the stable cache
        // key. Check it before resolution: resolving a native URL downloads
        // its complete payload, so checking only afterwards silently fetched
        // the same HDRI on every preview frame even though decoding was cached.
        let source_key = PathBuf::from(&environment.src);
        if let Some(image) = self.environment_cache.get(&source_key).cloned() {
            return Ok(GpuWorldLighting {
                params: GpuWorldLightingParams::from_world(
                    lighting,
                    camera,
                    true,
                    image.mip_bytes.len(),
                ),
                environment: image,
                frame_index: 0,
                temporal_jitter: false,
            });
        }
        let resolved = resolve_world_asset_source(
            asset_root,
            &environment.src,
            WorldPathStyle::Relative,
            self.asset_resolver.as_ref(),
        )?;
        let key = resolved.key().to_path_buf();
        if !self.environment_cache.contains_key(&key) {
            let decoded = Arc::new(load_environment_image_from_resolved(&resolved)?);
            self.environment_cache.insert(key.clone(), decoded);
        }
        let image = self
            .environment_cache
            .get(&key)
            .expect("environment image inserted before frame lighting")
            .clone();
        Ok(GpuWorldLighting {
            params: GpuWorldLightingParams::from_world(
                lighting,
                camera,
                true,
                image.mip_bytes.len(),
            ),
            environment: image,
            frame_index: 0,
            temporal_jitter: false,
        })
    }

    /// Reuse the Scene compositor device whenever the 3D backend is embedded.
    async fn ensure_gpu_renderer(
        &mut self,
        width: u32,
        height: u32,
        temporal_history_enabled: bool,
    ) -> Result<(), WorldRenderError> {
        let preview_budget = self.immediate_preview_settings.budget();
        let texture_anisotropy = preview_budget.texture_anisotropy;
        let needs_renderer = self.gpu_renderer.as_ref().is_none_or(|renderer| {
            renderer.width != width
                || renderer.height != height
                || renderer.texture_anisotropy != texture_anisotropy
                || renderer.temporal_history_enabled != temporal_history_enabled
        });
        if needs_renderer {
            self.gpu_renderer = Some(
                if let Some((device, queue)) = self.gpu_device_queue.clone() {
                    GpuWorldRenderer::new_with_device(
                        device,
                        queue,
                        width,
                        height,
                        texture_anisotropy,
                        temporal_history_enabled,
                    )
                    .await?
                } else {
                    GpuWorldRenderer::new(
                        width,
                        height,
                        texture_anisotropy,
                        temporal_history_enabled,
                    )
                    .await?
                },
            );
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct CharacterDesignGpuViewport {
    renderer: WorldFrameRenderer,
    diagnostics_cache: HashMap<PathBuf, WorldGpuDiagnostics>,
}

pub struct CharacterDesignViewportFrame {
    pub image: RgbaImage,
    pub diagnostics: Option<WorldGpuDiagnostics>,
}

impl CharacterDesignGpuViewport {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn render_frame(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
        actor_id: &str,
    ) -> Result<CharacterDesignViewportFrame, WorldRenderError> {
        let asset_root = asset_root.as_ref();
        let world = graph
            .presented_world()
            .ok_or_else(|| WorldRenderError::MissingWorld(graph.present.from.clone()))?;
        let actor = world
            .actors
            .iter()
            .find(|actor| actor.id == actor_id)
            .or_else(|| world.actors.first());
        let diagnostics = if let Some(actor) = actor {
            let (model_key, mesh) = load_glb_mesh_resolved(
                asset_root,
                &actor.model,
                actor.path_style,
                self.renderer.asset_resolver.as_ref(),
            )?;
            if !self.renderer.mesh_cache.contains_key(&model_key) {
                self.renderer.mesh_cache.insert(model_key.clone(), mesh);
            }
            if !self.diagnostics_cache.contains_key(&model_key) {
                let mesh = self
                    .renderer
                    .mesh_cache
                    .get(&model_key)
                    .expect("character viewport mesh cache entry inserted before diagnostics");
                let mut diagnostics = diagnose_world_glb_gpu_plan(mesh);
                diagnostics.skipped_reasons.push(
                    "Character Design viewport uses cached GLB/GPU resources; per-frame heavy diagnostics are intentionally skipped"
                        .to_string(),
                );
                self.diagnostics_cache
                    .insert(model_key.clone(), diagnostics);
            }
            let mut diagnostics = self.diagnostics_cache.get(&model_key).cloned();
            if let Some(diagnostics) = diagnostics.as_mut() {
                let time = WorldTime {
                    frame,
                    fps: graph.fps,
                    duration_ms: graph.duration_ms,
                };
                let cached_mesh = self.renderer.mesh_cache.get(&model_key);
                diagnostics.bone_override_count =
                    actor_bone_overrides_for_mesh(graph, actor, cached_mesh, time)
                        .map_or(0, |overrides| overrides.len());
            }
            diagnostics
        } else {
            None
        };

        let image = self
            .renderer
            .render_frame_gpu(graph, frame, asset_root)
            .await?;
        Ok(CharacterDesignViewportFrame { image, diagnostics })
    }

    pub async fn render_frame_with_ground_grid(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
        actor_id: &str,
    ) -> Result<CharacterDesignViewportFrame, WorldRenderError> {
        self.render_frame_with_ground_grid_mode(graph, frame, asset_root, actor_id, false)
            .await
    }

    pub async fn render_frame_with_ground_grid_mode(
        &mut self,
        graph: &WorldGraph,
        frame: u32,
        asset_root: impl AsRef<Path>,
        actor_id: &str,
        debug_grid: bool,
    ) -> Result<CharacterDesignViewportFrame, WorldRenderError> {
        let asset_root = asset_root.as_ref();
        let world = graph
            .presented_world()
            .ok_or_else(|| WorldRenderError::MissingWorld(graph.present.from.clone()))?;
        let actor = world
            .actors
            .iter()
            .find(|actor| actor.id == actor_id)
            .or_else(|| world.actors.first());
        let diagnostics = if let Some(actor) = actor {
            let (model_key, mesh) = load_glb_mesh_resolved(
                asset_root,
                &actor.model,
                actor.path_style,
                self.renderer.asset_resolver.as_ref(),
            )?;
            if !self.renderer.mesh_cache.contains_key(&model_key) {
                self.renderer.mesh_cache.insert(model_key.clone(), mesh);
            }
            if !self.diagnostics_cache.contains_key(&model_key) {
                let mesh = self
                    .renderer
                    .mesh_cache
                    .get(&model_key)
                    .expect("character viewport mesh cache entry inserted before diagnostics");
                let mut diagnostics = diagnose_world_glb_gpu_plan(mesh);
                diagnostics.skipped_reasons.push(
                    "Character Design viewport uses cached GLB/GPU resources; per-frame heavy diagnostics are intentionally skipped"
                        .to_string(),
                );
                self.diagnostics_cache
                    .insert(model_key.clone(), diagnostics);
            }
            let mut diagnostics = self.diagnostics_cache.get(&model_key).cloned();
            if let Some(diagnostics) = diagnostics.as_mut() {
                let time = WorldTime {
                    frame,
                    fps: graph.fps,
                    duration_ms: graph.duration_ms,
                };
                let cached_mesh = self.renderer.mesh_cache.get(&model_key);
                diagnostics.bone_override_count =
                    actor_bone_overrides_for_mesh(graph, actor, cached_mesh, time)
                        .map_or(0, |overrides| overrides.len());
            }
            diagnostics
        } else {
            None
        };

        let image = self
            .renderer
            .render_frame_gpu_with_ground_grid_mode(graph, frame, asset_root, debug_grid)
            .await?;
        Ok(CharacterDesignViewportFrame { image, diagnostics })
    }
}

fn inverse_display(value: f32, mode: f32) -> f32 {
    let y = value.clamp(0.0, 1.0).powf(2.2);
    if mode > 1.5 {
        let a = 2.51 - y * 2.43;
        let b = 0.03 - y * 0.59;
        (-b + (b * b + 4.0 * a * y * 0.14).sqrt()) / (2.0 * a)
    } else if mode > 0.5 {
        y / (1.0 - y).max(0.0001)
    } else {
        y
    }
}

struct GpuWorldRenderer {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    _poller: DevicePoller,
    bind_group_layout: wgpu::BindGroupLayout,
    lighting_bind_group_layout: wgpu::BindGroupLayout,
    background_empty_bind_group: wgpu::BindGroup,
    shadow_lighting_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    outline_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    transparent_depth_write_pipeline: wgpu::RenderPipeline,
    transmissive_pipeline: wgpu::RenderPipeline,
    transmissive_depth_write_pipeline: wgpu::RenderPipeline,
    transmission_scene_texture: wgpu::Texture,
    transmission_scene_bind_group: wgpu::BindGroup,
    background_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,
    dof_pipeline: wgpu::RenderPipeline,
    motion_blur_pipeline: wgpu::RenderPipeline,
    dof_bind_group_layout: wgpu::BindGroupLayout,
    grid_bind_group: wgpu::BindGroup,
    grid_params_buffer: wgpu::Buffer,
    grid_vertex_buffer: wgpu::Buffer,
    actor_resource_cache: HashMap<GpuWorldResourceKey, GpuWorldActorResource>,
    vertex_resource_cache: HashMap<u64, Arc<GpuWorldVertexResource>>,
    texture_resource_cache: HashMap<GpuWorldTextureKey, Arc<GpuWorldTextureResource>>,
    instance_resource_cache: HashMap<GpuWorldInstanceKey, GpuWorldInstanceResource>,
    actor_sampler: wgpu::Sampler,
    environment_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    dof_sampler: wgpu::Sampler,
    lighting_params_buffer: wgpu::Buffer,
    environment_resource: Option<GpuWorldEnvironmentResource>,
    /// Reusable render targets. Triple buffering avoids allocating a new 1080p
    /// texture merely because the Scene compositor still owns the previous
    /// frame. Extra targets are only added when an external caller deliberately
    /// retains every pooled texture.
    targets: Vec<Arc<wgpu::Texture>>,
    hdr_target: Arc<wgpu::Texture>,
    history_texture: wgpu::Texture,
    history_depth_texture: wgpu::Texture,
    history_gbuffer_texture: wgpu::Texture,
    temporal_history_enabled: bool,
    preview_gbuffer_texture: wgpu::Texture,
    preview_material_texture: wgpu::Texture,
    history_valid: bool,
    last_history_frame: Option<u32>,
    last_camera: Option<PreviewCameraHistory>,
    last_temporal_style_signature: Option<u64>,
    object_motion_history: HashMap<GpuWorldInstanceKey, PreviousObjectMotion>,
    target_cursor: usize,
    depth_texture: wgpu::Texture,
    shadow_texture: wgpu::Texture,
    readback_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    texture_anisotropy: u16,
}

#[derive(Clone, Copy)]
struct PreviewCameraHistory {
    camera0: [f32; 4],
    camera1: [f32; 4],
    camera2: [f32; 4],
    camera3: [f32; 4],
    jitter: [f32; 2],
}

#[derive(Clone)]
struct PreviousObjectMotion {
    params: GpuWorldParams,
    bone_matrices: Vec<[f32; 16]>,
    frame: u32,
}

fn preview_halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0;
    let mut fraction = 1.0;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

fn preview_frame_jitter(frame: u32) -> [f32; 2] {
    let sample = frame % 8 + 1;
    // A sub-eighth-pixel radius retains phase coverage without making an editor
    // preview appear to move when history is rejected on a thin silhouette.
    const PREVIEW_JITTER_SCALE: f32 = 0.125;
    [
        (preview_halton(sample, 2) - 0.5) * PREVIEW_JITTER_SCALE,
        (preview_halton(sample, 3) - 0.5) * PREVIEW_JITTER_SCALE,
    ]
}

#[derive(Clone, Copy)]
struct EffectiveAntiAliasing {
    requested: &'static str,
    effective: &'static str,
    fallback_used: bool,
    spatial_selector: f32,
    temporal: bool,
    jitter_phases: u32,
    quality_selector: f32,
    sharpness: f32,
}

fn canonical_anti_aliasing_method(value: &str) -> &'static str {
    match value {
        "auto" => "auto",
        "off" => "off",
        "fxaa" => "fxaa",
        "smaa" => "smaa",
        "msaa" => "msaa",
        "taa" => "taa",
        "ssaa" => "ssaa",
        _ => "off",
    }
}

fn resolve_effective_anti_aliasing(
    style: Option<&crate::render_style::ResolvedSceneRenderStyle>,
    settings: crate::preview::ImmediatePreviewSettings,
) -> EffectiveAntiAliasing {
    let budget = settings.budget();
    let Some(style) = style.filter(|style| style.style_id.is_some()) else {
        return EffectiveAntiAliasing {
            requested: "host",
            effective: if budget.temporal_antialiasing {
                "taa"
            } else if budget.antialiasing != crate::preview::ImmediatePreviewAntialiasing::Off {
                "fxaa"
            } else {
                "off"
            },
            fallback_used: false,
            spatial_selector: (budget.antialiasing
                != crate::preview::ImmediatePreviewAntialiasing::Off)
                as u8 as f32,
            temporal: budget.temporal_antialiasing,
            jitter_phases: if budget.temporal_jitter { 8 } else { 1 },
            quality_selector: 1.0,
            sharpness: 0.0,
        };
    };
    let Some(authored) = style.anti_aliasing.as_ref() else {
        return EffectiveAntiAliasing {
            requested: "off",
            effective: "off",
            fallback_used: false,
            spatial_selector: 0.0,
            temporal: false,
            jitter_phases: 1,
            quality_selector: 0.0,
            sharpness: 0.0,
        };
    };
    let quality = match authored.quality.as_str() {
        "low" => 0,
        "high" => 2,
        "ultra" => 3,
        _ => 1,
    };
    let automatic = match settings.profile {
        crate::preview::ImmediatePreviewProfile::Portable => "fxaa",
        crate::preview::ImmediatePreviewProfile::Balanced
        | crate::preview::ImmediatePreviewProfile::Cinematic => "taa",
    };
    let requested = if authored.method == "auto" {
        automatic
    } else {
        canonical_anti_aliasing_method(&authored.method)
    };
    let supported = match settings.profile {
        crate::preview::ImmediatePreviewProfile::Portable => {
            matches!(requested, "off" | "fxaa" | "smaa")
        }
        crate::preview::ImmediatePreviewProfile::Balanced
        | crate::preview::ImmediatePreviewProfile::Cinematic => {
            matches!(requested, "off" | "fxaa" | "smaa" | "taa")
        }
    };
    let method = if supported {
        requested
    } else if authored.fallback == "auto" {
        if settings.profile == crate::preview::ImmediatePreviewProfile::Portable {
            "fxaa"
        } else {
            "smaa"
        }
    } else {
        canonical_anti_aliasing_method(&authored.fallback)
    };
    let (spatial_selector, temporal) = match method {
        "off" => (0.0, false),
        "smaa" => (2.0, false),
        "taa" => (1.0, true),
        _ => (1.0, false),
    };
    EffectiveAntiAliasing {
        requested: canonical_anti_aliasing_method(&authored.method),
        effective: method,
        fallback_used: method != requested,
        spatial_selector,
        temporal,
        jitter_phases: if temporal {
            match quality {
                0 => 1,
                1 => 2,
                2 => 4,
                _ => 8,
            }
        } else {
            1
        },
        quality_selector: quality as f32,
        sharpness: authored.sharpness,
    }
}

fn preview_camera_cut(previous: PreviewCameraHistory, current: PreviewCameraHistory) -> bool {
    let eye_delta = (0..3)
        .map(|axis| (current.camera0[axis] - previous.camera0[axis]).powi(2))
        .sum::<f32>()
        .sqrt();
    let forward_dot = (0..3)
        .map(|axis| current.camera3[axis] * previous.camera3[axis])
        .sum::<f32>();
    let focal_ratio = current.camera0[3].max(0.001) / previous.camera0[3].max(0.001);
    eye_delta > current.camera2[3].max(1.0) * 0.2
        || forward_dot < 0.65
        || !(0.5..=2.0).contains(&focal_ratio)
}

fn preview_temporal_style_signature(params: &GpuWorldLightingParams) -> u64 {
    let mut hasher = DefaultHasher::new();
    for values in [
        params.color0,
        params.color1,
        params.optics0,
        params.dof_style,
        params.surface0,
        params.surface1,
        params.surface2,
        params.surface3,
        params.cel0,
        params.cel1,
        params.cel2,
        params.universal_color,
        params.universal_tone,
        params.universal_shadow,
        params.universal_highlight,
    ] {
        values.map(f32::to_bits).hash(&mut hasher);
    }
    hasher.finish()
}

/// Transparent surfaces use the same shader and bindings as opaque PBR draws,
struct GpuWorldEnvironmentResource {
    signature: u64,
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

impl GpuWorldRenderer {
    fn reset_temporal_history(&mut self) {
        self.history_valid = false;
        self.last_history_frame = None;
        self.last_camera = None;
        self.last_temporal_style_signature = None;
        self.object_motion_history.clear();
    }

    async fn new(
        width: u32,
        height: u32,
        texture_anisotropy: u16,
        temporal_history_enabled: bool,
    ) -> Result<Self, WorldRenderError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = request_adapter_async(
            &instance,
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        )
        .await
        .map_err(|_| WorldRenderError::GpuRender {
            message: "no high-performance GPU adapter was available".to_string(),
        })?;
        let adapter_limits = adapter.limits();
        let max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;
        if width > max_texture_dimension_2d || height > max_texture_dimension_2d {
            return Err(WorldRenderError::GpuRender {
                message: format!(
                    "requested world render size {}x{} exceeds GPU max 2D texture dimension {}",
                    width, height, max_texture_dimension_2d
                ),
            });
        }

        let (device, queue) = request_device_async(
            &adapter,
            &wgpu::DeviceDescriptor {
                label: Some("anica-motionloom-world-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            },
        )
        .await
        .map_err(|err| WorldRenderError::GpuRender {
            message: format!("device request failed: {err}"),
        })?;
        Self::new_with_device(
            Arc::new(device),
            queue,
            width,
            height,
            texture_anisotropy,
            temporal_history_enabled,
        )
        .await
    }

    /// Build the legacy 3D backend on the Scene compositor's GPU context.
    async fn new_with_device(
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
        texture_anisotropy: u16,
        temporal_history_enabled: bool,
    ) -> Result<Self, WorldRenderError> {
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        if width > max_texture_dimension_2d || height > max_texture_dimension_2d {
            return Err(WorldRenderError::GpuRender {
                message: format!(
                    "requested Scene 3D render size {}x{} exceeds GPU max 2D texture dimension {}",
                    width, height, max_texture_dimension_2d
                ),
            });
        }
        let poller = DevicePoller::start(device.clone());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-world-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                WGPU_WORLD_SHADER.as_str(),
            )),
        });
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-ground-grid-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGPU_GROUND_GRID_SHADER)),
        });
        let dof_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-world-dof-shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGPU_WORLD_DOF_SHADER)),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("anica-motionloom-world-gpu-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
        });
        let lighting_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-world-lighting-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });
        let dof_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-world-dof-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
            });
        let transmission_scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-world-transmission-scene-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
        // The background shader only consumes @group(1) lighting resources.
        // Keep group 0 explicitly empty instead of reusing the model material
        // layout: doing the latter makes every background draw require a model
        // bind group and causes WGPU to reject scenes that begin with an HDRI.
        let background_empty_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-world-background-empty-layout"),
                entries: &[],
            });
        let background_empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-world-background-empty-bind-group"),
            layout: &background_empty_bind_group_layout,
            entries: &[],
        });
        // The shadow pass only reads matrices from the lighting uniform. Its
        // bind group must not contain the shadow texture that the pass writes.
        let shadow_lighting_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-world-shadow-lighting-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let lighting_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anica-motionloom-world-lighting-params"),
            size: 1120,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_lighting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-world-shadow-lighting-bind-group"),
            layout: &shadow_lighting_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lighting_params_buffer.as_entire_binding(),
            }],
        });
        let grid_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-ground-grid-bind-group-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anica-motionloom-world-gpu-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout, &lighting_bind_group_layout],
            push_constant_ranges: &[],
        });
        let transmissive_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-world-transmissive-pipeline-layout"),
                bind_group_layouts: &[
                    &bind_group_layout,
                    &lighting_bind_group_layout,
                    &transmission_scene_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let background_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-world-background-pipeline-layout"),
                bind_group_layouts: &[
                    &background_empty_bind_group_layout,
                    &lighting_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-world-shadow-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout, &shadow_lighting_bind_group_layout],
                push_constant_ranges: &[],
            });
        let grid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anica-motionloom-ground-grid-pipeline-layout"),
            bind_group_layouts: &[&grid_bind_group_layout],
            push_constant_ranges: &[],
        });
        let dof_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anica-motionloom-world-dof-pipeline-layout"),
            bind_group_layouts: &[&dof_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("anica-motionloom-world-gpu-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 116,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 40,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 56,
                            shader_location: 4,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 64,
                            shader_location: 5,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 80,
                            shader_location: 6,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 92,
                            shader_location: 7,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 104,
                            shader_location: 8,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main_gbuffer"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let transparent_pipeline = create_world_surface_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            "anica-motionloom-world-transparent-pipeline",
            "fs_main",
            false,
        );
        let transparent_depth_write_pipeline = create_world_surface_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            "anica-motionloom-world-transparent-depth-write-pipeline",
            "fs_main",
            true,
        );
        let transmissive_pipeline = create_world_surface_pipeline(
            &device,
            &shader,
            &transmissive_pipeline_layout,
            "anica-motionloom-world-transmissive-pipeline",
            "fs_transmissive",
            false,
        );
        let transmissive_depth_write_pipeline = create_world_surface_pipeline(
            &device,
            &shader,
            &transmissive_pipeline_layout,
            "anica-motionloom-world-transmissive-depth-write-pipeline",
            "fs_transmissive",
            true,
        );
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("anica-motionloom-world-environment-background-pipeline"),
            layout: Some(&background_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_background"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_background"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("anica-motionloom-world-shadow-map-pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 116,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 40,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 56,
                            shader_location: 4,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 64,
                            shader_location: 5,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 80,
                            shader_location: 6,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 92,
                            shader_location: 7,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 104,
                            shader_location: 8,
                        },
                    ],
                }],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("anica-motionloom-ground-grid-pipeline"),
            layout: Some(&grid_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let dof_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("anica-motionloom-world-dof-pipeline"),
            layout: Some(&dof_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &dof_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &dof_shader,
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
        let motion_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("anica-motionloom-world-motion-blur-pipeline"),
            layout: Some(&dof_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &dof_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &dof_shader,
                entry_point: Some("fs_motion_blur"),
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
        let targets = (0..3)
            .map(|_| Arc::new(Self::make_target_texture(&device, width, height)))
            .collect();
        let hdr_target = Arc::new(Self::make_hdr_texture(&device, width, height));
        let history_width = if temporal_history_enabled { width } else { 1 };
        let history_height = if temporal_history_enabled { height } else { 1 };
        let history_texture = Self::make_target_texture(&device, history_width, history_height);
        let history_depth_texture =
            Self::make_depth_texture(&device, history_width, history_height);
        let history_gbuffer_texture =
            Self::make_hdr_texture(&device, history_width, history_height);
        let preview_gbuffer_texture = Self::make_hdr_texture(&device, width, height);
        let preview_material_texture = Self::make_target_texture(&device, width, height);
        let transmission_scene_texture = Self::make_hdr_texture(&device, width, height);
        let transmission_scene_view =
            transmission_scene_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let transmission_scene_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("anica-motionloom-world-transmission-scene-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let transmission_scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-world-transmission-scene-bind-group"),
            layout: &transmission_scene_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&transmission_scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&transmission_scene_sampler),
                },
            ],
        });
        let depth_texture = Self::make_depth_texture(&device, width, height);
        let actor_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("anica-motionloom-world-gpu-sampler"),
            // glTF and typed PBR materials default to repeat wrapping; primitive
            // UV scaling therefore tiles instead of stretching edge pixels.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: texture_anisotropy.clamp(1, 16),
            ..Default::default()
        });
        let environment_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("anica-motionloom-world-environment-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("anica-motionloom-world-shadow-comparison-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let dof_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("anica-motionloom-world-dof-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let padded_bytes_per_row = align_to_256(width.saturating_mul(4));
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anica-motionloom-world-gpu-readback"),
            size: (padded_bytes_per_row as u64 * height as u64).max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anica-motionloom-ground-grid-params"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-ground-grid-bind-group"),
            layout: &grid_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_params_buffer.as_entire_binding(),
            }],
        });
        let grid_vertex_buffer = {
            let half = 200.0f32;
            let vertices = [
                [-half, 0.0, -half],
                [half, 0.0, -half],
                [half, 0.0, half],
                [-half, 0.0, -half],
                [half, 0.0, half],
                [-half, 0.0, half],
            ];
            let bytes = pack_f32x3_vertices(&vertices);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("anica-motionloom-ground-grid-vertices"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX,
                mapped_at_creation: true,
            });
            buffer
                .slice(..bytes.len() as u64)
                .get_mapped_range_mut()
                .copy_from_slice(&bytes);
            buffer.unmap();
            buffer
        };
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anica-motionloom-world-shadow-map"),
            size: wgpu::Extent3d {
                width: 1536,
                height: 1536,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let outline_pipeline = create_world_surface_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            "cel-outline",
            "fs_outline",
            false,
        );
        Ok(Self {
            device,
            queue,
            _poller: poller,
            bind_group_layout,
            lighting_bind_group_layout,
            background_empty_bind_group,
            shadow_lighting_bind_group,
            outline_pipeline,
            pipeline,
            transparent_pipeline,
            transparent_depth_write_pipeline,
            transmissive_pipeline,
            transmissive_depth_write_pipeline,
            transmission_scene_texture,
            transmission_scene_bind_group,
            background_pipeline,
            shadow_pipeline,
            grid_pipeline,
            dof_pipeline,
            motion_blur_pipeline,
            dof_bind_group_layout,
            grid_bind_group,
            grid_params_buffer,
            grid_vertex_buffer,
            actor_resource_cache: HashMap::new(),
            vertex_resource_cache: HashMap::new(),
            texture_resource_cache: HashMap::new(),
            instance_resource_cache: HashMap::new(),
            actor_sampler,
            environment_sampler,
            shadow_sampler,
            dof_sampler,
            lighting_params_buffer,
            environment_resource: None,
            targets,
            hdr_target,
            history_texture,
            history_depth_texture,
            history_gbuffer_texture,
            temporal_history_enabled,
            preview_gbuffer_texture,
            preview_material_texture,
            history_valid: false,
            last_history_frame: None,
            last_camera: None,
            last_temporal_style_signature: None,
            object_motion_history: HashMap::new(),
            target_cursor: 0,
            depth_texture,
            shadow_texture,
            readback_buffer,
            width,
            height,
            padded_bytes_per_row,
            texture_anisotropy: texture_anisotropy.clamp(1, 16),
        })
    }

    async fn render_readback(
        &mut self,
        background: &RgbaImage,
        draw_calls: &[GpuWorldDraw],
        ground_grid: Option<GpuGroundGridParams>,
        lighting: &GpuWorldLighting,
    ) -> Result<RgbaImage, WorldRenderError> {
        let frame =
            self.render_to_texture(Some(background), draw_calls, ground_grid, lighting, None)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-motionloom-scene-3d-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: frame.texture.as_ref(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
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
        self.queue.submit([encoder.finish()]);
        self.readback_rgba_async().await
    }

    /// Submit the 3D draw and return its sampleable texture without readback.
    fn render_to_texture(
        &mut self,
        background: Option<&RgbaImage>,
        draw_calls: &[GpuWorldDraw],
        ground_grid: Option<GpuGroundGridParams>,
        lighting: &GpuWorldLighting,
        clear_color: Option<wgpu::Color>,
    ) -> Result<crate::scene::preview_surface::GpuFrameTexture, WorldRenderError> {
        if let Some(background) = background
            && (background.width() != self.width || background.height() != self.height)
        {
            return Err(WorldRenderError::GpuRender {
                message: format!(
                    "world GPU background size {}x{} does not match renderer {}x{}",
                    background.width(),
                    background.height(),
                    self.width,
                    self.height
                ),
            });
        }
        // Resize only on quality changes; environment bindings reference this texture.
        let shadow_size = (lighting.params.surface3[2] as u32)
            .clamp(128, self.device.limits().max_texture_dimension_2d);
        if self.shadow_texture.width() != shadow_size {
            self.shadow_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("motionloom-style-shadow-map"),
                size: wgpu::Extent3d {
                    width: shadow_size,
                    height: shadow_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.environment_resource = None;
        }
        let target = Arc::clone(&self.hdr_target);
        if clear_color.is_none() {
            let background = background.ok_or_else(|| WorldRenderError::GpuRender {
                message: "world GPU render requires a background when no clear color is supplied"
                    .to_string(),
            })?;
            self.write_texture_hdr_background(
                &target,
                background.as_raw(),
                lighting.params.color0[3],
            );
        }
        // Fit small rigid scenes without increasing shadow texture resolution.
        let mut shadow_bounds = Vec::with_capacity(draw_calls.len());
        for draw in draw_calls {
            if draw.params.vegetation[0] != 0.0
                || draw
                    .bone_matrices
                    .iter()
                    .any(|matrix| *matrix != mat4_identity())
            {
                shadow_bounds.clear();
                break;
            }
            let geometry =
                self.shared_actor_geometry(draw.vertex_signature, &draw.vertices, &draw.indices);
            let Some(bounds) = geometry.rigid_bounds else {
                shadow_bounds.clear();
                break;
            };
            shadow_bounds.push((bounds, draw.params));
        }
        let mut fitted_lighting = fit_rigid_shadow_volume(lighting.params, &shadow_bounds);
        let taa_enabled = fitted_lighting.preview0[0] > 0.5;
        let current_jitter = if taa_enabled && lighting.temporal_jitter {
            let phases = match fitted_lighting.render_compat[2] as u32 {
                1 => 2,
                2 => 4,
                _ => 8,
            };
            preview_frame_jitter(lighting.frame_index % phases)
        } else {
            [0.0; 2]
        };
        let current_camera = PreviewCameraHistory {
            camera0: fitted_lighting.camera0,
            camera1: fitted_lighting.camera1,
            camera2: fitted_lighting.camera2,
            camera3: fitted_lighting.camera3,
            jitter: current_jitter,
        };
        let current_style_signature = preview_temporal_style_signature(&fitted_lighting);
        let sequential = self.last_history_frame.is_some_and(|previous| {
            lighting.frame_index == previous || lighting.frame_index == previous.saturating_add(1)
        });
        let usable_previous = self.last_camera.filter(|previous| {
            sequential
                && self.last_temporal_style_signature == Some(current_style_signature)
                && !preview_camera_cut(*previous, current_camera)
        });
        let previous = usable_previous.unwrap_or(current_camera);
        let history_valid = self.history_valid && usable_previous.is_some() && taa_enabled;
        fitted_lighting.previous_camera0 = previous.camera0;
        fitted_lighting.previous_camera1 = previous.camera1;
        fitted_lighting.previous_camera2 = previous.camera2;
        fitted_lighting.previous_camera3 = previous.camera3;
        fitted_lighting.preview0[3] = history_valid as u8 as f32;
        fitted_lighting.preview1 = [
            current_jitter[0],
            current_jitter[1],
            previous.jitter[0],
            previous.jitter[1],
        ];
        self.queue.write_buffer(
            &self.lighting_params_buffer,
            0,
            &pack_gpu_world_lighting(fitted_lighting),
        );
        self.ensure_environment_resource(lighting);
        let lighting_bind_group = self
            .environment_resource
            .as_ref()
            .expect("environment resource prepared before 3D draw")
            .bind_group
            .clone();
        let mut gpu_draws = Vec::<GpuWorldDrawResources>::with_capacity(draw_calls.len());
        let mut active_texture_keys = HashSet::<GpuWorldTextureKey>::new();
        let mut buffer_writes = 0usize;
        // Keep authoring order (including coplanar ties); only adjacent compatible
        // opaque draws coalesce. Transparent and skinned draws remain independent.
        let mut batches: Vec<(Vec<&GpuWorldDraw>, bool)> = Vec::new();
        let max_instances = (self.device.limits().max_storage_buffer_binding_size as usize
            / std::mem::size_of::<GpuWorldParams>())
        .max(1);
        for draw in draw_calls {
            let geometry =
                self.shared_actor_geometry(draw.vertex_signature, &draw.vertices, &draw.indices);
            // Procedural meshes use an identity joint, not necessarily zero weights.
            // Only identity palettes can safely use the cached undeformed bounds.
            let rigid = draw
                .bone_matrices
                .iter()
                .all(|matrix| *matrix == mat4_identity());
            let visible = rigid_draw_visible(
                if rigid { geometry.rigid_bounds } else { None },
                draw.params,
            );
            let compatible = batches.last().is_some_and(|(batch, previous_visible)| {
                let previous = batch[0];
                batch.len() < max_instances
                    && *previous_visible == visible
                    && draw.phase == GpuWorldDrawPhase::Opaque
                    && draw.depth_write
                    && previous.phase == draw.phase
                    && previous.depth_write
                    && previous.resource_key == draw.resource_key
                    && previous.vertex_signature == draw.vertex_signature
                    && previous.sort_priority == draw.sort_priority
                    && previous.texture.signature == draw.texture.signature
                    && previous.normal_texture.signature == draw.normal_texture.signature
                    && previous.metallic_roughness_texture.signature
                        == draw.metallic_roughness_texture.signature
                    && previous.emissive_texture.signature == draw.emissive_texture.signature
                    && previous.occlusion_texture.signature == draw.occlusion_texture.signature
                    && previous.cel_texture.signature == draw.cel_texture.signature
                    && previous.bone_matrices == draw.bone_matrices
                    && rigid
                    && geometry.rigid_bounds.is_some()
            });
            if compatible {
                batches.last_mut().unwrap().0.push(draw);
            } else {
                batches.push((vec![draw], visible));
            }
        }
        for (batch, camera_visible) in batches {
            let draw = batch[0];
            if draw.indices.is_empty() {
                continue;
            }
            let geometry = self.shared_actor_geometry(
                draw.vertex_signature,
                draw.vertices.as_ref(),
                draw.indices.as_ref(),
            );
            let texture_keys = [
                GpuWorldTextureKey {
                    role: TextureRole::Color,
                    ..GpuWorldTextureKey::from(draw.texture.as_ref())
                },
                GpuWorldTextureKey {
                    role: TextureRole::Normal,
                    ..GpuWorldTextureKey::from(draw.normal_texture.as_ref())
                },
                GpuWorldTextureKey::from(draw.metallic_roughness_texture.as_ref()),
                GpuWorldTextureKey {
                    role: TextureRole::Color,
                    ..GpuWorldTextureKey::from(draw.emissive_texture.as_ref())
                },
                GpuWorldTextureKey::from(draw.cel_texture.as_ref()),
                GpuWorldTextureKey::from(draw.occlusion_texture.as_ref()),
            ];
            active_texture_keys.extend(texture_keys);
            let texture = self.shared_actor_texture(
                texture_keys[0],
                draw.texture.as_ref(),
                "anica-motionloom-world-gpu-texture",
            );
            let normal_texture = self.shared_actor_texture(
                texture_keys[1],
                draw.normal_texture.as_ref(),
                "anica-motionloom-world-gpu-normal-texture",
            );
            let metallic_roughness_texture = self.shared_actor_texture(
                texture_keys[2],
                draw.metallic_roughness_texture.as_ref(),
                "anica-motionloom-world-gpu-metallic-roughness-texture",
            );
            let emissive_texture = self.shared_actor_texture(
                texture_keys[3],
                draw.emissive_texture.as_ref(),
                "anica-motionloom-world-gpu-emissive-texture",
            );
            let cel_texture = self.shared_actor_texture(
                texture_keys[4],
                draw.cel_texture.as_ref(),
                "cel-control-map",
            );
            let occlusion_texture = self.shared_actor_texture(
                texture_keys[5],
                draw.occlusion_texture.as_ref(),
                "material-occlusion",
            );
            let texture_binding_changed = self
                .actor_resource_cache
                .get(&draw.resource_key)
                .is_some_and(|resource| resource.texture_keys != texture_keys);
            if texture_binding_changed {
                let resource = self
                    .actor_resource_cache
                    .get_mut(&draw.resource_key)
                    .expect("GPU actor resource exists before texture rebinding");
                resource.texture_keys = texture_keys;
                resource.texture = Arc::clone(&texture);
                resource.normal_texture = Arc::clone(&normal_texture);
                resource.metallic_roughness_texture = Arc::clone(&metallic_roughness_texture);
                resource.emissive_texture = Arc::clone(&emissive_texture);
                resource.occlusion_texture = Arc::clone(&occlusion_texture);
                resource.cel_texture = Arc::clone(&cel_texture);
                self.instance_resource_cache
                    .retain(|key, _| key.resource_key != draw.resource_key);
            }
            if !self.actor_resource_cache.contains_key(&draw.resource_key) {
                self.actor_resource_cache.insert(
                    draw.resource_key.clone(),
                    GpuWorldActorResource {
                        geometry,
                        texture_keys,
                        texture,
                        normal_texture,
                        metallic_roughness_texture,
                        emissive_texture,
                        occlusion_texture,
                        cel_texture,
                    },
                );
            }

            let (
                geometry,
                texture_view,
                normal_texture_view,
                metallic_roughness_texture_view,
                emissive_texture_view,
                occlusion_texture_view,
                cel_texture_view,
            ) = {
                let resource = self
                    .actor_resource_cache
                    .get(&draw.resource_key)
                    .expect("GPU actor resource inserted before draw");
                (
                    Arc::clone(&resource.geometry),
                    resource.texture.view.clone(),
                    resource.normal_texture.view.clone(),
                    resource.metallic_roughness_texture.view.clone(),
                    resource.emissive_texture.view.clone(),
                    resource.occlusion_texture.view.clone(),
                    resource.cel_texture.view.clone(),
                )
            };
            let bone_offset = draw.bone_matrices.len().max(1) as f32;
            let motion_params: Vec<GpuWorldParams> = batch
                .iter()
                .map(|draw| {
                    let previous =
                        self.object_motion_history
                            .get(&draw.instance_key)
                            .filter(|previous| {
                                history_valid
                                    && (lighting.frame_index == previous.frame
                                        || lighting.frame_index == previous.frame.saturating_add(1))
                                    && previous.bone_matrices.len() == draw.bone_matrices.len()
                            });
                    let previous_params = previous.map_or(draw.params, |state| state.params);
                    let mut params = draw.params;
                    params.previous_model = previous_params.model;
                    params.previous_actor = previous_params.actor;
                    params.previous_actor_rotation = previous_params.actor_rotation;
                    params.previous_vegetation = previous_params.vegetation;
                    params.motion0 = [bone_offset, previous.is_some() as u8 as f32, 0.0, 0.0];
                    params
                })
                .collect();
            let params_bytes: Vec<u8> = motion_params
                .into_iter()
                .flat_map(pack_gpu_world_params)
                .collect();
            let previous_bones = self
                .object_motion_history
                .get(&draw.instance_key)
                .filter(|previous| {
                    history_valid
                        && (lighting.frame_index == previous.frame
                            || lighting.frame_index == previous.frame.saturating_add(1))
                        && previous.bone_matrices.len() == draw.bone_matrices.len()
                })
                .map_or(draw.bone_matrices.as_slice(), |previous| {
                    previous.bone_matrices.as_slice()
                });
            let bone_bytes = pack_gpu_world_bone_pair(&draw.bone_matrices, previous_bones);
            let bone_buffer_size = bone_bytes.len().max(64) as u64;
            let needs_instance = self
                .instance_resource_cache
                .get(&draw.instance_key)
                .is_none_or(|resource| {
                    resource.bone_buffer_size < bone_buffer_size
                        || resource.params_buffer_size < params_bytes.len() as u64
                });
            if needs_instance {
                let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("anica-motionloom-world-gpu-params"),
                    size: params_bytes.len().max(4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bone_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("anica-motionloom-world-gpu-bones"),
                    size: bone_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("anica-motionloom-world-gpu-bind-group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: bone_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.actor_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&normal_texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(
                                &metallic_roughness_texture_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&emissive_texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&cel_texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(&occlusion_texture_view),
                        },
                    ],
                });
                self.instance_resource_cache.insert(
                    draw.instance_key.clone(),
                    GpuWorldInstanceResource {
                        params_buffer,
                        bone_buffer,
                        bone_buffer_size,
                        params_buffer_size: params_bytes.len() as u64,
                        last_params: Vec::new(),
                        last_bones: Vec::new(),
                        bind_group,
                    },
                );
            }
            let instance_resource = self
                .instance_resource_cache
                .get_mut(&draw.instance_key)
                .expect("GPU world instance resource inserted before draw");
            // Retained buffers need no upload when their exact contents are unchanged.
            if instance_resource.last_params != params_bytes {
                buffer_writes += 1;
                self.queue
                    .write_buffer(&instance_resource.params_buffer, 0, &params_bytes);
                instance_resource.last_params = params_bytes;
            }
            if instance_resource.last_bones != bone_bytes {
                buffer_writes += 1;
                self.queue
                    .write_buffer(&instance_resource.bone_buffer, 0, &bone_bytes);
                instance_resource.last_bones = bone_bytes;
            }
            let bind_group = instance_resource.bind_group.clone();
            for chunk in &geometry.chunks {
                gpu_draws.push(GpuWorldDrawResources {
                    vertex_buffer: chunk.vertex_buffer.clone(),
                    index_buffer: chunk.index_buffer.clone(),
                    bind_group: bind_group.clone(),
                    index_count: chunk.index_count,
                    instance_count: batch.len() as u32,
                    camera_visible,
                    phase: draw.phase,
                    depth_write: draw.depth_write,
                    sort_priority: draw.sort_priority,
                    camera_depth: draw.camera_depth,
                });
            }
        }
        // Optional native diagnostics do not change the public profile/DSL contract.
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var_os("MOTIONLOOM_TRACE_BATCHES").is_some() {
            eprintln!(
                "motionloom batches: items={} batches={} camera_batches={} buffer_writes={}",
                draw_calls.len(),
                gpu_draws.len(),
                gpu_draws.iter().filter(|draw| draw.camera_visible).count(),
                buffer_writes
            );
        }
        #[cfg(target_arch = "wasm32")]
        let _ = buffer_writes;
        // Opaque draws keep authoring order. Transparent draws follow them and
        // blend from far to near, with priority as an explicit expert override.
        gpu_draws.sort_by(|left, right| match (left.phase, right.phase) {
            (GpuWorldDrawPhase::Opaque, GpuWorldDrawPhase::Opaque) => std::cmp::Ordering::Equal,
            (GpuWorldDrawPhase::Opaque, _) => std::cmp::Ordering::Less,
            (_, GpuWorldDrawPhase::Opaque) => std::cmp::Ordering::Greater,
            _ => left
                .sort_priority
                .cmp(&right.sort_priority)
                .then_with(|| right.camera_depth.total_cmp(&left.camera_depth)),
        });
        // Drop superseded hot-reload/live-binding uploads after each frame;
        // actor resources retain any texture still referenced by in-flight work.
        for resource in self.actor_resource_cache.values() {
            active_texture_keys.extend(resource.texture_keys);
        }
        self.texture_resource_cache
            .retain(|key, _| active_texture_keys.contains(key));

        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let preview_gbuffer_view = self
            .preview_gbuffer_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let preview_material_view = self
            .preview_material_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-motionloom-world-gpu-encoder"),
            });
        if lighting.params.color1[3] > 0.0 {
            let shadow_view = self
                .shadow_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("anica-motionloom-world-shadow-map-pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            shadow_pass.set_pipeline(&self.shadow_pipeline);
            shadow_pass.set_bind_group(1, &self.shadow_lighting_bind_group, &[]);
            for draw in gpu_draws
                .iter()
                .filter(|draw| draw.phase == GpuWorldDrawPhase::Opaque)
            {
                shadow_pass.set_bind_group(0, &draw.bind_group, &[]);
                shadow_pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                shadow_pass
                    .set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                shadow_pass.draw_indexed(0..draw.index_count, 0, 0..draw.instance_count);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("anica-motionloom-world-background-grid-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: clear_color.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if lighting.params.environment2[0] > 0.5 && lighting.params.environment0[3] > 0.5 {
                pass.set_pipeline(&self.background_pipeline);
                pass.set_bind_group(0, &self.background_empty_bind_group, &[]);
                pass.set_bind_group(1, &lighting_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            if let Some(grid) = ground_grid {
                self.queue.write_buffer(
                    &self.grid_params_buffer,
                    0,
                    &pack_ground_grid_params(grid),
                );
                pass.set_pipeline(&self.grid_pipeline);
                pass.set_bind_group(0, &self.grid_bind_group, &[]);
                pass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
                pass.draw(0..6, 0..1);
            }
        }
        // Shade opaque geometry and produce its temporal/material data in the
        // same submission. This replaces two additional full-scene passes.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("anica-motionloom-world-opaque-mrt-pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &preview_gbuffer_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.5,
                                g: 0.5,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &preview_material_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 0.0,
                                b: 1.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(1, &lighting_bind_group, &[]);
            for draw in gpu_draws.iter().filter(|draw| {
                draw.camera_visible && draw.phase == GpuWorldDrawPhase::Opaque && draw.depth_write
            }) {
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..draw.index_count, 0, 0..draw.instance_count);
            }
        }
        // Completed opaque depth hides occluded backface outlines.
        if lighting.params.cel0[2] > 0.0 {
            let view = target.create_view(&Default::default());
            let depth = self.depth_texture.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cel-outline"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.outline_pipeline);
            pass.set_bind_group(1, &lighting_bind_group, &[]);
            for draw in gpu_draws
                .iter()
                .filter(|d| d.camera_visible && d.phase == GpuWorldDrawPhase::Opaque)
            {
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..draw.index_count, 0, 0..draw.instance_count);
            }
        }
        // Transmissive shaders sample a stable opaque snapshot. Reading the
        // active render attachment directly would violate WebGPU alias rules.
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: target.as_ref(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.transmission_scene_texture,
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
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("anica-motionloom-world-transparent-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(1, &lighting_bind_group, &[]);
            for draw in gpu_draws.iter().filter(|draw| {
                draw.camera_visible
                    && (draw.phase != GpuWorldDrawPhase::Opaque || !draw.depth_write)
            }) {
                match (draw.phase, draw.depth_write) {
                    (GpuWorldDrawPhase::AlphaBlend, false) => {
                        pass.set_pipeline(&self.transparent_pipeline);
                    }
                    (GpuWorldDrawPhase::AlphaBlend, true) => {
                        pass.set_pipeline(&self.transparent_depth_write_pipeline);
                    }
                    (GpuWorldDrawPhase::Transmissive, false) => {
                        pass.set_pipeline(&self.transmissive_pipeline);
                        pass.set_bind_group(2, &self.transmission_scene_bind_group, &[]);
                    }
                    (GpuWorldDrawPhase::Transmissive, true) => {
                        pass.set_pipeline(&self.transmissive_depth_write_pipeline);
                        pass.set_bind_group(2, &self.transmission_scene_bind_group, &[]);
                    }
                    (GpuWorldDrawPhase::Opaque, false) => {
                        pass.set_pipeline(&self.transparent_pipeline);
                    }
                    (GpuWorldDrawPhase::Opaque, true) => unreachable!(),
                }
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..draw.index_count, 0, 0..draw.instance_count);
            }
        }
        // Resolve HDR after transparency and depth-aware optics, even with DoF disabled.
        let output_target = {
            let dof_target = self.acquire_target();
            let dof_view = dof_target.create_view(&wgpu::TextureViewDescriptor::default());
            let history_view = self
                .history_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let history_depth_view = self
                .history_depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let history_gbuffer_view = self
                .history_gbuffer_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let dof_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("anica-motionloom-world-dof-bind-group"),
                layout: &self.dof_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.lighting_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.dof_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&history_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&preview_gbuffer_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&preview_material_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&history_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&history_gbuffer_view),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("anica-motionloom-world-dof-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &dof_view,
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
                pass.set_pipeline(&self.dof_pipeline);
                pass.set_bind_group(0, &dof_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            dof_target
        };
        if taa_enabled {
            // Preserve the surface identity alongside colour so the next frame
            // can reject history across disocclusions and moving silhouettes.
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: output_target.as_ref(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.history_texture,
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
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.depth_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.history_depth_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.preview_gbuffer_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.history_gbuffer_texture,
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
        let final_target = if (fitted_lighting.preview0[2] > 0.5 && history_valid)
            || fitted_lighting.surface3[3] > 0.5
            || fitted_lighting.render_compat[3] > 0.0001
        {
            let target = self.acquire_target();
            let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let temporal_view = output_target.create_view(&wgpu::TextureViewDescriptor::default());
            let history_view = self
                .history_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let history_depth_view = self
                .history_depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let history_gbuffer_view = self
                .history_gbuffer_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let motion_blur_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("anica-motionloom-world-motion-blur-bind-group"),
                    layout: &self.dof_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.lighting_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&temporal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.dof_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&history_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&preview_gbuffer_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&preview_material_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&history_depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(&history_gbuffer_view),
                        },
                    ],
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("anica-motionloom-world-motion-blur-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
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
                pass.set_pipeline(&self.motion_blur_pipeline);
                pass.set_bind_group(0, &motion_blur_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            target
        } else {
            output_target
        };
        self.queue.submit([encoder.finish()]);
        self.last_camera = Some(current_camera);
        self.last_temporal_style_signature = Some(current_style_signature);
        self.last_history_frame = Some(lighting.frame_index);
        self.history_valid = taa_enabled;
        self.object_motion_history.clear();
        self.object_motion_history
            .extend(draw_calls.iter().map(|draw| {
                (
                    draw.instance_key.clone(),
                    PreviousObjectMotion {
                        params: draw.params,
                        bone_matrices: draw.bone_matrices.clone(),
                        frame: lighting.frame_index,
                    },
                )
            }));
        Ok(crate::scene::preview_surface::GpuFrameTexture {
            texture: final_target,
            width: self.width,
            height: self.height,
            format: wgpu::TextureFormat::Rgba8Unorm,
        })
    }

    fn acquire_target(&mut self) -> Arc<wgpu::Texture> {
        let pool_len = self.targets.len();
        for offset in 0..pool_len {
            let index = (self.target_cursor + offset) % pool_len;
            if Arc::strong_count(&self.targets[index]) == 1 {
                self.target_cursor = (index + 1) % pool_len;
                return Arc::clone(&self.targets[index]);
            }
        }
        let target = Arc::new(Self::make_target_texture(
            self.device.as_ref(),
            self.width,
            self.height,
        ));
        self.targets.push(Arc::clone(&target));
        self.target_cursor = 0;
        target
    }

    fn make_target_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anica-motionloom-world-gpu-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn make_hdr_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anica-motionloom-world-gpu-hdr-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn make_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anica-motionloom-world-gpu-depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    // Background pixels are display-referred; invert only the final curve so
    // the shared resolve preserves them without grading the host's 2D canvas.
    fn write_texture_hdr_background(&self, texture: &wgpu::Texture, rgba: &[u8], mode: f32) {
        let bytes: Vec<u8> = rgba
            .chunks_exact(4)
            .flat_map(|p| {
                let alpha = p[3] as f32 / 255.0;
                [
                    inverse_display(p[0] as f32 / 255.0, mode) * alpha,
                    inverse_display(p[1] as f32 / 255.0, mode) * alpha,
                    inverse_display(p[2] as f32 / 255.0, mode) * alpha,
                    alpha,
                ]
                .into_iter()
                .flat_map(|v| half::f16::from_f32(v).to_bits().to_ne_bytes())
            })
            .collect();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 8),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload the linear environment mip chain only when the decoded source
    /// changes. Animated intensity and rotation update the uniform separately.
    fn ensure_environment_resource(&mut self, lighting: &GpuWorldLighting) {
        if self
            .environment_resource
            .as_ref()
            .is_some_and(|resource| resource.signature == lighting.environment.signature)
        {
            return;
        }
        let image = lighting.environment.as_ref();
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anica-motionloom-world-environment-texture"),
            size: wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: image.mip_bytes.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut mip_width = image.width;
        let mut mip_height = image.height;
        for (level, bytes) in image.mip_bytes.iter().enumerate() {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(mip_width * 8),
                    rows_per_image: Some(mip_height),
                },
                wgpu::Extent3d {
                    width: mip_width,
                    height: mip_height,
                    depth_or_array_layers: 1,
                },
            );
            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_view = self
            .shadow_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-world-lighting-bind-group"),
            layout: &self.lighting_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.lighting_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.environment_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
            ],
        });
        self.environment_resource = Some(GpuWorldEnvironmentResource {
            signature: image.signature,
            _texture: texture,
            bind_group,
        });
    }

    fn shared_actor_geometry(
        &mut self,
        signature: u64,
        vertices: &[GpuWorldVertex],
        indices: &[u32],
    ) -> Arc<GpuWorldVertexResource> {
        if let Some(resource) = self.vertex_resource_cache.get(&signature) {
            return Arc::clone(resource);
        }
        // Indexed chunks keep imported GLB vertices shared and ensure no single
        // allocation approaches the backend's advertised buffer limit.
        let chunk_bytes = gpu_world_vertex_chunk_bytes(&self.device);
        let chunks = split_gpu_world_indexed_chunks(vertices, indices, chunk_bytes)
            .into_iter()
            .map(|chunk| {
                let vertex_bytes = pack_gpu_world_vertices(&chunk.vertices);
                let index_bytes = pack_gpu_world_indices(&chunk.indices);
                let vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("anica-motionloom-world-gpu-vertices"),
                    size: vertex_bytes.len().max(4) as u64,
                    usage: wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: true,
                });
                vertex_buffer
                    .slice(..vertex_bytes.len() as u64)
                    .get_mapped_range_mut()
                    .copy_from_slice(&vertex_bytes);
                vertex_buffer.unmap();
                let index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("anica-motionloom-world-gpu-indices"),
                    size: index_bytes.len().max(4) as u64,
                    usage: wgpu::BufferUsages::INDEX,
                    mapped_at_creation: true,
                });
                index_buffer
                    .slice(..index_bytes.len() as u64)
                    .get_mapped_range_mut()
                    .copy_from_slice(&index_bytes);
                index_buffer.unmap();
                GpuWorldGeometryChunk {
                    vertex_buffer,
                    index_buffer,
                    index_count: chunk.indices.len() as u32,
                }
            })
            .collect();
        let resource = Arc::new(GpuWorldVertexResource {
            chunks,
            rigid_bounds: rigid_vertex_bounds(vertices),
        });
        self.vertex_resource_cache
            .insert(signature, Arc::clone(&resource));
        resource
    }

    fn shared_actor_texture(
        &mut self,
        key: GpuWorldTextureKey,
        texture: &GpuWorldTexture,
        label: &'static str,
    ) -> Arc<GpuWorldTextureResource> {
        if let Some(resource) = self.texture_resource_cache.get(&key) {
            return Arc::clone(resource);
        }
        let gpu_texture = self.make_actor_texture(texture, label, key.role);
        let view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let resource = Arc::new(GpuWorldTextureResource {
            _texture: gpu_texture,
            view,
        });
        self.texture_resource_cache
            .insert(key, Arc::clone(&resource));
        resource
    }

    fn make_actor_texture(
        &self,
        texture: &GpuWorldTexture,
        label: &'static str,
        role: TextureRole,
    ) -> wgpu::Texture {
        let mut width = texture.width.max(1);
        let mut height = texture.height.max(1);
        let expected_len = width as usize * height as usize * 4;
        let fallback;
        let rgba = if texture.rgba.len() == expected_len {
            texture.rgba.as_slice()
        } else {
            width = 1;
            height = 1;
            fallback = vec![255, 255, 255, 255];
            fallback.as_slice()
        };
        let levels = build_mips(width, height, rgba, role);
        let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if role == TextureRole::Color {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, (width, height, rgba)) in levels.iter().enumerate() {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &gpu_texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width.saturating_mul(4)),
                    rows_per_image: Some(*height),
                },
                wgpu::Extent3d {
                    width: *width,
                    height: *height,
                    depth_or_array_layers: 1,
                },
            );
        }
        gpu_texture
    }

    async fn readback_rgba_async(&self) -> Result<RgbaImage, WorldRenderError> {
        let slice = self.readback_buffer.slice(..);
        BufferMapAsyncFuture::new(&self._poller, &self.readback_buffer)
            .await
            .map_err(|err| WorldRenderError::GpuRender {
                message: format!("readback map failed: {err}"),
            })?;

        let mapped = slice.get_mapped_range();
        let row_bytes = self.width as usize * 4;
        let padded_row = self.padded_bytes_per_row as usize;
        let mut out = vec![0u8; row_bytes * self.height as usize];
        for row in 0..self.height as usize {
            let src_off = row * padded_row;
            let dst_off = row * row_bytes;
            out[dst_off..dst_off + row_bytes]
                .copy_from_slice(&mapped[src_off..src_off + row_bytes]);
        }
        drop(mapped);
        self.readback_buffer.unmap();
        RgbaImage::from_raw(self.width, self.height, out).ok_or_else(|| {
            WorldRenderError::GpuRender {
                message: "failed to build RGBA image from world GPU readback".to_string(),
            }
        })
    }
}

struct GpuWorldDrawResources {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    index_count: u32,
    instance_count: u32,
    camera_visible: bool,
    phase: GpuWorldDrawPhase,
    depth_write: bool,
    sort_priority: i32,
    camera_depth: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuWorldDrawPhase {
    Opaque,
    AlphaBlend,
    Transmissive,
}

fn gpu_world_material_phase(material: Option<&GlbMaterialData>) -> GpuWorldDrawPhase {
    material.map_or(GpuWorldDrawPhase::Opaque, |material| {
        if material.transmission_factor > 0.001 {
            GpuWorldDrawPhase::Transmissive
        } else if material.alpha_mode == GlbAlphaMode::Blend
            || material.base_color_factor[3] < 0.999
        {
            GpuWorldDrawPhase::AlphaBlend
        } else {
            GpuWorldDrawPhase::Opaque
        }
    })
}

fn gpu_world_material_depth_write(
    material: Option<&GlbMaterialData>,
    phase: GpuWorldDrawPhase,
) -> bool {
    material.is_none_or(|material| match material.depth_write {
        GlbDepthWriteMode::Enabled => true,
        GlbDepthWriteMode::Disabled => false,
        GlbDepthWriteMode::Auto => phase == GpuWorldDrawPhase::Opaque,
    })
}

struct GpuWorldActorResource {
    geometry: Arc<GpuWorldVertexResource>,
    texture_keys: [GpuWorldTextureKey; 6],
    texture: Arc<GpuWorldTextureResource>,
    normal_texture: Arc<GpuWorldTextureResource>,
    metallic_roughness_texture: Arc<GpuWorldTextureResource>,
    emissive_texture: Arc<GpuWorldTextureResource>,
    occlusion_texture: Arc<GpuWorldTextureResource>,
    cel_texture: Arc<GpuWorldTextureResource>,
}

// Cache undeformed bounds once; the current bone palette gates their use.
fn rigid_vertex_bounds(vertices: &[GpuWorldVertex]) -> Option<([f32; 3], [f32; 3])> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            let value = vertex.position[axis];
            if !value.is_finite() {
                return None;
            }
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    (!vertices.is_empty()).then_some((min, max))
}

// Reject only rigid bounds outside a side plane in front of the near plane.
// Deformation, near-plane intersections and shadow casters are left untouched.
fn rigid_draw_visible(bounds: Option<([f32; 3], [f32; 3])>, p: GpuWorldParams) -> bool {
    let Some((min, max)) = bounds else {
        return true;
    };
    if p.vegetation[0] != 0.0 {
        return true;
    }
    let mut outside = [true; 4];
    for corner in 0..8 {
        let local = std::array::from_fn(|axis| {
            let v = if corner & (1 << axis) == 0 {
                min[axis]
            } else {
                max[axis]
            };
            (v - p.model[axis]) * p.model[3]
        });
        let rotated = quat_rotate_vec3(quat_normalize_xyzw(p.actor_rotation), local);
        let rel: [f32; 3] =
            std::array::from_fn(|axis| rotated[axis] + p.actor[axis] - p.camera0[axis]);
        let dot = |basis: [f32; 4]| (0..3).map(|axis| rel[axis] * basis[axis]).sum::<f32>();
        let x = dot(p.camera1);
        let y = dot(p.camera2);
        let z = dot(p.camera3);
        if !x.is_finite() || !y.is_finite() || !z.is_finite() || z <= p.camera1[3] {
            return true;
        }
        let half_x = z * p.canvas[0] * 0.5 / p.camera0[3];
        let half_y = z * p.canvas[1] * 0.5 / p.camera0[3];
        for (index, test) in [
            x < -half_x - 0.01,
            x > half_x + 0.01,
            y < -half_y - 0.01,
            y > half_y + 0.01,
        ]
        .into_iter()
        .enumerate()
        {
            outside[index] &= test;
        }
    }
    !outside.into_iter().any(|value| value)
}

struct GpuWorldVertexResource {
    chunks: Vec<GpuWorldGeometryChunk>,
    rigid_bounds: Option<([f32; 3], [f32; 3])>,
}

struct GpuWorldGeometryChunk {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

struct GpuWorldTextureResource {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GpuWorldTextureKey {
    role: TextureRole,
    width: u32,
    height: u32,
    signature: u64,
}

impl From<&GpuWorldTexture> for GpuWorldTextureKey {
    fn from(texture: &GpuWorldTexture) -> Self {
        Self {
            width: texture.width.max(1),
            height: texture.height.max(1),
            signature: texture.signature,
            role: TextureRole::Data,
        }
    }
}

struct GpuWorldInstanceResource {
    params_buffer: wgpu::Buffer,
    bone_buffer: wgpu::Buffer,
    bone_buffer_size: u64,
    params_buffer_size: u64,
    last_params: Vec<u8>,
    last_bones: Vec<u8>,
    bind_group: wgpu::BindGroup,
}

const GPU_WORLD_VERTEX_STRIDE_BYTES: usize = 116;
const GPU_WORLD_DEFAULT_CHUNK_MIB: u64 = 128;

struct CpuGpuWorldGeometryChunk {
    vertices: Vec<GpuWorldVertex>,
    indices: Vec<u32>,
}

// Keep allocations modest even when a native adapter advertises multi-GiB
// buffers. The environment override is an expert tuning knob, not a DSL rule.
fn gpu_world_vertex_key(vertex: GpuWorldVertex) -> [u32; 29] {
    let mut words = [0u32; 29];
    for (cursor, value) in vertex
        .position
        .into_iter()
        .chain(vertex.normal)
        .chain(vertex.joints)
        .chain(vertex.weights)
        .chain(vertex.uv)
        .chain(vertex.color)
        .chain(vertex.tangent)
        .chain(vertex.bitangent)
        .chain(vertex.outline_normal)
        .enumerate()
    {
        words[cursor] = value.to_bits();
    }
    words
}

// Collapse the renderer's material-expanded triangles before they enter the
// retained cache, so repeated GLB vertices do not remain duplicated on CPU.
fn index_gpu_world_vertices(
    expanded_vertices: &[GpuWorldVertex],
) -> (Vec<GpuWorldVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::with_capacity(expanded_vertices.len());
    let mut index_by_vertex = HashMap::<[u32; 29], u32>::new();
    for vertex in expanded_vertices.iter().copied() {
        let key = gpu_world_vertex_key(vertex);
        let index = *index_by_vertex.entry(key).or_insert_with(|| {
            let index = vertices.len() as u32;
            vertices.push(vertex);
            index
        });
        indices.push(index);
    }
    (vertices, indices)
}

// Remap indexed triangles into local chunks before either GPU allocation
// exceeds the configured upload target.
fn split_gpu_world_indexed_chunks(
    source_vertices: &[GpuWorldVertex],
    source_indices: &[u32],
    chunk_bytes: usize,
) -> Vec<CpuGpuWorldGeometryChunk> {
    let mut chunks = Vec::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut local_by_source = HashMap::<u32, u32>::new();

    for triangle in source_indices.chunks(3) {
        let new_vertices = triangle
            .iter()
            .enumerate()
            .filter(|(index, source_index)| {
                !local_by_source.contains_key(*source_index)
                    && !triangle[..*index].contains(source_index)
            })
            .count();
        let vertex_bytes = vertices
            .len()
            .saturating_add(new_vertices)
            .saturating_mul(GPU_WORLD_VERTEX_STRIDE_BYTES);
        let index_bytes = indices
            .len()
            .saturating_add(triangle.len())
            .saturating_mul(std::mem::size_of::<u32>());
        if !indices.is_empty() && (vertex_bytes > chunk_bytes || index_bytes > chunk_bytes) {
            chunks.push(CpuGpuWorldGeometryChunk {
                vertices: std::mem::take(&mut vertices),
                indices: std::mem::take(&mut indices),
            });
            local_by_source.clear();
        }

        for source_index in triangle.iter().copied() {
            let Some(vertex) = source_vertices.get(source_index as usize).copied() else {
                continue;
            };
            let local_index = *local_by_source.entry(source_index).or_insert_with(|| {
                let local_index = vertices.len() as u32;
                vertices.push(vertex);
                local_index
            });
            indices.push(local_index);
        }
    }
    if !indices.is_empty() {
        chunks.push(CpuGpuWorldGeometryChunk { vertices, indices });
    }
    chunks
}

fn pack_gpu_world_vertices(vertices: &[GpuWorldVertex]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vertices.len().saturating_mul(GPU_WORLD_VERTEX_STRIDE_BYTES));
    for vertex in vertices {
        for value in vertex.position {
            out.extend_from_slice(&value.to_ne_bytes());
        }
        for value in vertex.normal {
            out.extend_from_slice(&value.to_ne_bytes());
        }
        for value in vertex.joints {
            out.extend_from_slice(&value.to_ne_bytes());
        }
        for value in vertex.weights {
            out.extend_from_slice(&value.to_ne_bytes());
        }
        for value in vertex.uv {
            out.extend_from_slice(&value.to_ne_bytes());
        }
        for value in vertex.color {
            out.extend_from_slice(&value.to_ne_bytes());
        }
        for value in vertex.tangent {
            out.extend_from_slice(&value.to_ne_bytes());
        }
        for value in vertex.bitangent {
            out.extend_from_slice(&value.to_ne_bytes());
        }
        for value in vertex.outline_normal {
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
    out
}

fn pack_gpu_world_indices(indices: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(indices.len().saturating_mul(std::mem::size_of::<u32>()));
    for index in indices {
        out.extend_from_slice(&index.to_ne_bytes());
    }
    out
}

fn gpu_world_geometry_signature(vertices: &[GpuWorldVertex], indices: &[u32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for vertex in vertices.iter().copied() {
        gpu_world_vertex_key(vertex).hash(&mut hasher);
    }
    indices.hash(&mut hasher);
    hasher.finish()
}

fn pack_f32x3_vertices(vertices: &[[f32; 3]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vertices.len().saturating_mul(12));
    for vertex in vertices {
        for value in *vertex {
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
    out
}

fn pack_gpu_world_bone_pair(current: &[[f32; 16]], previous: &[[f32; 16]]) -> Vec<u8> {
    let matrix_count = current.len().max(1) + previous.len().max(1);
    let mut out = Vec::with_capacity(matrix_count.saturating_mul(64));
    for palette in [current, previous] {
        if palette.is_empty() {
            for value in mat4_identity() {
                out.extend_from_slice(&value.to_ne_bytes());
            }
        } else {
            for matrix in palette {
                for value in matrix {
                    out.extend_from_slice(&value.to_ne_bytes());
                }
            }
        }
    }
    out
}

#[cfg_attr(target_arch = "wasm32", allow(unused_mut, unused_variables))]
pub async fn render_world_graph_to_video_with_progress<F>(
    ffmpeg_bin: &str,
    graph: &WorldGraph,
    asset_root: impl AsRef<Path>,
    output_path: &Path,
    profile: SceneRenderProfile,
    progress_every_frames: u32,
    progress_callback: F,
) -> Result<(), WorldRenderError>
where
    F: FnMut(WorldRenderProgress),
{
    render_world_graph_to_video_with_progress_and_cancel(
        ffmpeg_bin,
        graph,
        asset_root,
        output_path,
        profile,
        progress_every_frames,
        None,
        progress_callback,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_arch = "wasm32", allow(unused_mut, unused_variables))]
pub async fn render_world_graph_to_video_with_progress_and_cancel<F>(
    ffmpeg_bin: &str,
    graph: &WorldGraph,
    asset_root: impl AsRef<Path>,
    output_path: &Path,
    profile: SceneRenderProfile,
    progress_every_frames: u32,
    cancel: Option<Arc<AtomicBool>>,
    mut progress_callback: F,
) -> Result<(), WorldRenderError>
where
    F: FnMut(WorldRenderProgress),
{
    if profile.is_png_sequence() {
        return render_world_graph_to_png_sequence_internal(
            graph,
            asset_root,
            output_path,
            profile,
            progress_every_frames,
            cancel,
            progress_callback,
        )
        .await;
    }

    #[cfg(target_arch = "wasm32")]
    {
        Err(WorldRenderError::VideoExportNotAvailable {
            message: "FFmpeg video export is not available in WASM".to_string(),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::export::{FfmpegVideoEncoder, VideoEncoder};

        let asset_root = asset_root.as_ref().to_path_buf();
        let (w, h) = graph.output_size();
        let fps = graph.fps.max(1.0);
        let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
        let total_frames = ((duration_sec * fps).round() as u32).max(1);
        let encoder_args = world_encoder_args(profile);
        let mut renderer = WorldFrameRenderer::default();
        progress_callback(WorldRenderProgress {
            rendered_frames: 0,
            total_frames,
        });
        if cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
        {
            return Err(WorldRenderError::Cancelled);
        }

        let mut encoder =
            FfmpegVideoEncoder::new(ffmpeg_bin, output_path).with_encoder_args(encoder_args);
        encoder.begin(w, h, fps)?;

        for frame in 0..total_frames {
            if cancel
                .as_ref()
                .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
            {
                encoder.abort();
                return Err(WorldRenderError::Cancelled);
            }
            let image = if profile.uses_gpu_compositor() {
                renderer.render_frame_gpu(graph, frame, &asset_root).await?
            } else {
                renderer.render_frame(graph, frame, &asset_root)?
            };
            encoder.push_frame(frame, image.as_raw())?;
            let rendered_frames = frame + 1;
            if rendered_frames == total_frames
                || (progress_every_frames > 0 && rendered_frames % progress_every_frames == 0)
            {
                progress_callback(WorldRenderProgress {
                    rendered_frames,
                    total_frames,
                });
            }
        }
        encoder.finish()?;
        Ok(())
    }
}

pub async fn render_world_graph_to_png_sequence_with_progress<F>(
    graph: &WorldGraph,
    asset_root: impl AsRef<Path>,
    output_dir: &Path,
    progress_every_frames: u32,
    progress_callback: F,
) -> Result<(), WorldRenderError>
where
    F: FnMut(WorldRenderProgress),
{
    render_world_graph_to_png_sequence_with_progress_and_cancel(
        graph,
        asset_root,
        output_dir,
        progress_every_frames,
        None,
        progress_callback,
    )
    .await
}

pub async fn render_world_graph_to_png_sequence_with_progress_and_cancel<F>(
    graph: &WorldGraph,
    asset_root: impl AsRef<Path>,
    output_dir: &Path,
    progress_every_frames: u32,
    cancel: Option<Arc<AtomicBool>>,
    progress_callback: F,
) -> Result<(), WorldRenderError>
where
    F: FnMut(WorldRenderProgress),
{
    render_world_graph_to_png_sequence_internal(
        graph,
        asset_root,
        output_dir,
        SceneRenderProfile::GpuPngSequence,
        progress_every_frames,
        cancel,
        progress_callback,
    )
    .await
}

async fn render_world_graph_to_png_sequence_internal<F>(
    graph: &WorldGraph,
    asset_root: impl AsRef<Path>,
    output_dir: &Path,
    profile: SceneRenderProfile,
    progress_every_frames: u32,
    cancel: Option<Arc<AtomicBool>>,
    mut progress_callback: F,
) -> Result<(), WorldRenderError>
where
    F: FnMut(WorldRenderProgress),
{
    fs::create_dir_all(output_dir).map_err(|source| WorldRenderError::CreateOutputDir {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let asset_root = asset_root.as_ref().to_path_buf();
    let fps = graph.fps.max(1.0);
    let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
    let total_frames = ((duration_sec * fps).round() as u32).max(1);
    let mut renderer = WorldFrameRenderer::default();
    progress_callback(WorldRenderProgress {
        rendered_frames: 0,
        total_frames,
    });

    for frame in 0..total_frames {
        if cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
        {
            return Err(WorldRenderError::Cancelled);
        }
        let image = if profile.uses_gpu_compositor() {
            renderer.render_frame_gpu(graph, frame, &asset_root).await?
        } else {
            renderer.render_frame(graph, frame, &asset_root)?
        };
        let path = output_dir.join(format!("frame_{frame:06}.png"));
        image
            .save(&path)
            .map_err(|source| WorldRenderError::SavePngFrame { path, source })?;

        let rendered_frames = frame + 1;
        if rendered_frames == total_frames
            || (progress_every_frames > 0 && rendered_frames % progress_every_frames == 0)
        {
            progress_callback(WorldRenderProgress {
                rendered_frames,
                total_frames,
            });
        }
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn world_encoder_args(profile: SceneRenderProfile) -> Vec<String> {
    match profile {
        SceneRenderProfile::Cpu | SceneRenderProfile::GpuProRes => world_prores_encoder_args(),
        SceneRenderProfile::Gpu => world_gpu_h264_encoder_args(),
        SceneRenderProfile::GpuProRes4444 => world_prores_4444_encoder_args(),
        SceneRenderProfile::GpuPngSequence => Vec::new(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn world_prores_encoder_args() -> Vec<String> {
    vec![
        "-vf".to_string(),
        "format=yuv422p10le".to_string(),
        "-c:v".to_string(),
        "prores_ks".to_string(),
        "-profile:v".to_string(),
        "3".to_string(),
        "-vendor".to_string(),
        "apl0".to_string(),
        "-pix_fmt".to_string(),
        "yuv422p10le".to_string(),
        "-color_primaries".to_string(),
        "bt709".to_string(),
        "-color_trc".to_string(),
        "bt709".to_string(),
        "-colorspace".to_string(),
        "bt709".to_string(),
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn world_prores_4444_encoder_args() -> Vec<String> {
    vec![
        "-vf".to_string(),
        "format=yuva444p10le".to_string(),
        "-c:v".to_string(),
        "prores_ks".to_string(),
        "-profile:v".to_string(),
        "4".to_string(),
        "-vendor".to_string(),
        "apl0".to_string(),
        "-alpha_bits".to_string(),
        "16".to_string(),
        "-vtag".to_string(),
        "ap4h".to_string(),
        "-pix_fmt".to_string(),
        "yuva444p10le".to_string(),
        "-color_primaries".to_string(),
        "bt709".to_string(),
        "-color_trc".to_string(),
        "bt709".to_string(),
        "-colorspace".to_string(),
        "bt709".to_string(),
    ]
}

#[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
fn world_gpu_h264_encoder_args() -> Vec<String> {
    vec![
        "-c:v".to_string(),
        "h264_videotoolbox".to_string(),
        "-allow_sw".to_string(),
        "1".to_string(),
        "-profile:v".to_string(),
        "high".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-b:v".to_string(),
        "30M".to_string(),
        "-maxrate".to_string(),
        "45M".to_string(),
        "-bufsize".to_string(),
        "90M".to_string(),
        "-color_primaries".to_string(),
        "bt709".to_string(),
        "-color_trc".to_string(),
        "bt709".to_string(),
        "-colorspace".to_string(),
        "bt709".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
    ]
}

#[cfg(all(not(target_arch = "wasm32"), target_os = "windows"))]
fn world_gpu_h264_encoder_args() -> Vec<String> {
    vec![
        "-c:v".to_string(),
        "h264_mf".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-b:v".to_string(),
        "30M".to_string(),
        "-maxrate".to_string(),
        "45M".to_string(),
        "-bufsize".to_string(),
        "90M".to_string(),
        "-color_primaries".to_string(),
        "bt709".to_string(),
        "-color_trc".to_string(),
        "bt709".to_string(),
        "-colorspace".to_string(),
        "bt709".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
    ]
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_os = "macos"),
    not(target_os = "windows")
))]
fn world_gpu_h264_encoder_args() -> Vec<String> {
    vec![
        "-c:v".to_string(),
        "libopenh264".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-b:v".to_string(),
        "30M".to_string(),
        "-maxrate".to_string(),
        "45M".to_string(),
        "-bufsize".to_string(),
        "90M".to_string(),
        "-color_primaries".to_string(),
        "bt709".to_string(),
        "-color_trc".to_string(),
        "bt709".to_string(),
        "-colorspace".to_string(),
        "bt709".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
    ]
}

fn draw_world_background(
    canvas: &mut RgbaImage,
    world: &WorldNode,
    asset_root: &Path,
    resolver: &dyn AssetResolver,
    time: WorldTime,
    image_cache: &mut HashMap<PathBuf, RgbaImage>,
) -> Result<(), WorldRenderError> {
    let background = world.background.as_ref();
    let color = background
        .map(|bg| parse_rgba(&bg.color))
        .unwrap_or(Rgba([0, 0, 0, 255]));
    fill(canvas, color);

    let Some(background) = background else {
        return Ok(());
    };
    let Some(src) = background.src.as_deref() else {
        return Ok(());
    };
    let resolved = resolve_world_asset_source(asset_root, src, WorldPathStyle::Relative, resolver)?;
    let key = resolved.key().to_path_buf();
    if matches!(resolved, ResolvedWorldAsset::Missing { .. }) {
        return Ok(());
    }
    let opacity = eval_number(&background.opacity, 1.0, time)?.clamp(0.0, 1.0);
    if !image_cache.contains_key(&key) {
        let image = load_rgba_image_from_resolved(&resolved, |path, source| {
            WorldRenderError::BackgroundImage { path, source }
        })?
        .to_rgba8();
        image_cache.insert(key.clone(), image);
    }
    if let Some(image) = image_cache.get(&key) {
        composite_background(canvas, image, &background.fit, opacity);
    }
    Ok(())
}

fn draw_directional_characters(
    canvas: &mut RgbaImage,
    world: &WorldNode,
    logical_size: (u32, u32),
    asset_root: &Path,
    resolver: &dyn AssetResolver,
    time: WorldTime,
    image_cache: &mut HashMap<PathBuf, RgbaImage>,
) -> Result<(), WorldRenderError> {
    if world.directional_characters.is_empty() {
        return Ok(());
    }
    let camera_yaw = eval_number(&world.camera.yaw, 0.0, time)?;
    let camera_pitch = eval_number(&world.camera.pitch, 0.0, time)?;
    let camera_x = eval_number(&world.camera.x, 0.0, time)?;
    let camera_y = eval_number(&world.camera.y, 0.0, time)?;
    let camera_zoom = eval_number(&world.camera.zoom, 1.0, time)?.max(0.01);
    let logical_w = logical_size.0.max(1) as f32;
    let logical_h = logical_size.1.max(1) as f32;
    let output_scale_x = canvas.width().max(1) as f32 / logical_w;
    let output_scale_y = canvas.height().max(1) as f32 / logical_h;

    for character in &world.directional_characters {
        let Some(direction) = select_direction_frame(character, camera_yaw, camera_pitch, time)?
        else {
            continue;
        };
        let Some(image_src) = direction.image.as_deref().or(character.sheet.as_deref()) else {
            continue;
        };
        let resolved =
            resolve_world_asset_source(asset_root, image_src, character.path_style, resolver)?;
        let key = resolved.key().to_path_buf();
        if matches!(resolved, ResolvedWorldAsset::Missing { .. }) {
            return Err(WorldRenderError::MissingDirectionalCharacterImage(key));
        }
        if !image_cache.contains_key(&key) {
            let image = load_rgba_image_from_resolved(&resolved, |path, source| {
                WorldRenderError::DirectionalCharacterImage { path, source }
            })?
            .to_rgba8();
            image_cache.insert(key.clone(), image);
        }
        let Some(source_image) = image_cache.get(&key) else {
            continue;
        };
        let (rect_x, rect_y, rect_w, rect_h) = if let Some(play_sprite) =
            character.play_sprite.as_ref()
        {
            let Some(rect) = play_sprite_rect(play_sprite, direction, source_image, time)? else {
                continue;
            };
            rect
        } else if let Some(rect) = direction.rect {
            let Some(clamped) = clamp_direction_rect(rect, source_image) else {
                continue;
            };
            clamped
        } else {
            (
                0,
                0,
                source_image.width().max(1),
                source_image.height().max(1),
            )
        };
        if rect_w == 0 || rect_h == 0 {
            continue;
        };
        let frame = imageops::crop_imm(source_image, rect_x, rect_y, rect_w, rect_h).to_image();
        let scale = eval_number(&character.scale, 1.0, time)?.max(0.01) * camera_zoom;
        let scale_x = (scale * output_scale_x).max(0.01);
        let scale_y = (scale * output_scale_y).max(0.01);
        let scaled_w = ((frame.width() as f32 * scale_x).round() as u32).max(1);
        let scaled_h = ((frame.height() as f32 * scale_y).round() as u32).max(1);
        let scaled = imageops::resize(&frame, scaled_w, scaled_h, imageops::FilterType::Lanczos3);
        let x = (eval_number(&character.x, 0.0, time)? - camera_x) * output_scale_x;
        let y = (eval_number(&character.y, 0.0, time)? - camera_y) * output_scale_y;
        let opacity = eval_number(&character.opacity, 1.0, time)?.clamp(0.0, 1.0);
        let anchor = direction
            .anchor
            .unwrap_or((rect_w as f32 * 0.5, rect_h as f32));
        let draw_x = (x - anchor.0 * scale_x).round() as i32;
        let draw_y = (y - anchor.1 * scale_y).round() as i32;
        blend_image_i32(canvas, &scaled, draw_x, draw_y, opacity);
    }

    Ok(())
}

fn play_sprite_rect(
    play_sprite: &WorldSpritePlayback,
    direction: &WorldDirectionFrame,
    source_image: &RgbaImage,
    time: WorldTime,
) -> Result<Option<(u32, u32, u32, u32)>, WorldRenderError> {
    let fps = eval_number(&play_sprite.fps, 12.0, time)?.max(0.01);
    let elapsed = (time.time_sec() * fps).floor().max(0.0) as u32;
    let local_frame = if play_sprite.r#loop {
        elapsed % play_sprite.frames.max(1)
    } else {
        elapsed.min(play_sprite.frames.saturating_sub(1))
    };
    let frame_index = play_sprite.start.saturating_add(local_frame);
    let column = frame_index % play_sprite.columns.max(1);
    let row = frame_index / play_sprite.columns.max(1);
    let (base_x, base_y) = direction
        .rect
        .map(|rect| (rect.0, rect.1))
        .unwrap_or((play_sprite.margin_x, play_sprite.margin_y));
    let x = base_x.saturating_add(
        column.saturating_mul(
            play_sprite
                .frame_width
                .saturating_add(play_sprite.spacing_x),
        ),
    );
    let y = base_y.saturating_add(
        row.saturating_mul(
            play_sprite
                .frame_height
                .saturating_add(play_sprite.spacing_y),
        ),
    );
    Ok(clamp_direction_rect(
        (x, y, play_sprite.frame_width, play_sprite.frame_height),
        source_image,
    ))
}

fn select_direction_frame(
    character: &WorldDirectionalCharacter,
    camera_yaw: f32,
    camera_pitch: f32,
    time: WorldTime,
) -> Result<Option<&WorldDirectionFrame>, WorldRenderError> {
    if camera_pitch.abs() >= 60.0 {
        if let Some(direction) = character
            .directions
            .iter()
            .filter(|direction| direction.camera_pitch.is_some())
            .min_by(|a, b| {
                let a_dist = (a.camera_pitch.unwrap_or(0.0) - camera_pitch).abs();
                let b_dist = (b.camera_pitch.unwrap_or(0.0) - camera_pitch).abs();
                a_dist.total_cmp(&b_dist)
            })
        {
            return Ok(Some(direction));
        }
        if camera_pitch > 0.0 {
            if let Some(direction) = character.directions.iter().find(|direction| {
                direction
                    .name
                    .as_deref()
                    .is_some_and(|name| name.eq_ignore_ascii_case("top"))
            }) {
                return Ok(Some(direction));
            }
        }
    }

    let yaw = eval_number(&character.yaw, 0.0, time)?;
    let view_yaw = normalize_degrees(yaw - camera_yaw);
    Ok(character
        .directions
        .iter()
        .filter(|direction| direction.angle.is_some())
        .min_by(|a, b| {
            let a_dist = angular_distance(view_yaw, a.angle.unwrap_or(0.0));
            let b_dist = angular_distance(view_yaw, b.angle.unwrap_or(0.0));
            a_dist.total_cmp(&b_dist)
        })
        .or_else(|| character.directions.first()))
}

fn clamp_direction_rect(
    rect: (u32, u32, u32, u32),
    image: &RgbaImage,
) -> Option<(u32, u32, u32, u32)> {
    let (x, y, w, h) = rect;
    let x = x.min(image.width());
    let y = y.min(image.height());
    let w = w.min(image.width().saturating_sub(x));
    let h = h.min(image.height().saturating_sub(y));
    if w == 0 || h == 0 {
        None
    } else {
        Some((x, y, w, h))
    }
}

fn normalize_degrees(value: f32) -> f32 {
    value.rem_euclid(360.0)
}

fn angular_distance(a: f32, b: f32) -> f32 {
    let diff = (normalize_degrees(a) - normalize_degrees(b)).abs();
    diff.min(360.0 - diff)
}

#[derive(Debug, Clone, Copy)]
struct CameraActorView {
    x: f32,
    y: f32,
    depth: f32,
    yaw: f32,
    pitch: f32,
}

const WORLD_DEPTH_SORT_SCALE: f32 = 0.25;

#[allow(clippy::too_many_arguments)]
fn camera_actor_view(
    actor_x: f32,
    actor_y: f32,
    actor_z: f32,
    actor_yaw: f32,
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    camera_yaw: f32,
    camera_pitch: f32,
) -> CameraActorView {
    let dx = actor_x - camera_x;
    let dy = actor_y - camera_y;
    let dz = actor_z - camera_z;
    let yaw = camera_yaw.to_radians();
    let cos_y = yaw.cos();
    let sin_y = yaw.sin();
    let view_x = dx * cos_y + dz * sin_y;
    let yaw_depth = -dx * sin_y + dz * cos_y;
    let pitch = camera_pitch.to_radians();
    let cos_p = pitch.cos();
    let sin_p = pitch.sin();
    let view_y = dy * cos_p - yaw_depth * sin_p;
    let depth = dy * sin_p + yaw_depth * cos_p;
    CameraActorView {
        x: view_x,
        y: view_y,
        depth,
        yaw: actor_yaw - camera_yaw,
        pitch: camera_pitch,
    }
}

fn perspective_camera_view(
    world: &WorldNode,
    width: u32,
    height: u32,
    time: WorldTime,
) -> Result<PerspectiveCameraView, WorldRenderError> {
    let width_f = width.max(1) as f32;
    let height_f = height.max(1) as f32;
    let target_x = eval_number(&world.camera.target_x, 0.0, time)?;
    let target_y = eval_number(&world.camera.target_y, 1.0, time)?;
    let target_z = eval_number(&world.camera.target_z, 0.0, time)?;
    let yaw = eval_number(&world.camera.yaw, 0.0, time)?.to_radians();
    let pitch = eval_number(&world.camera.pitch, 0.0, time)?
        .clamp(-89.0, 89.0)
        .to_radians();
    let distance = eval_number(&world.camera.distance, 3.2, time)?.max(0.05);
    let fov = eval_number(&world.camera.fov, 35.0, time)?
        .clamp(10.0, 100.0)
        .to_radians();
    let yaw_sin = yaw.sin();
    let yaw_cos = yaw.cos();
    let pitch_sin = pitch.sin();
    let pitch_cos = pitch.cos();
    let target = [target_x, target_y, target_z];
    let eye = [
        target_x + yaw_sin * pitch_cos * distance,
        target_y + pitch_sin * distance,
        target_z + yaw_cos * pitch_cos * distance,
    ];
    let mut forward = normalize3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    if !forward[0].is_finite() || !forward[1].is_finite() || !forward[2].is_finite() {
        forward = [0.0, 0.0, -1.0];
    }
    let mut world_up = [
        eval_number(&world.camera.up_x, 0.0, time)?,
        eval_number(&world.camera.up_y, 1.0, time)?,
        eval_number(&world.camera.up_z, 0.0, time)?,
    ];
    world_up = normalize3(world_up);
    if !world_up[0].is_finite() || !world_up[1].is_finite() || !world_up[2].is_finite() {
        world_up = [0.0, 1.0, 0.0];
    }
    let mut right = normalize3(cross3(forward, world_up));
    if !right[0].is_finite() || !right[1].is_finite() || !right[2].is_finite() {
        right = [1.0, 0.0, 0.0];
    }
    let mut up = normalize3(cross3(right, forward));
    let roll = eval_number(&world.camera.roll, 0.0, time)?.to_radians();
    if roll.abs() > f32::EPSILON {
        let cos_r = roll.cos();
        let sin_r = roll.sin();
        let rolled_right = [
            right[0] * cos_r + up[0] * sin_r,
            right[1] * cos_r + up[1] * sin_r,
            right[2] * cos_r + up[2] * sin_r,
        ];
        up = [
            up[0] * cos_r - right[0] * sin_r,
            up[1] * cos_r - right[1] * sin_r,
            up[2] * cos_r - right[2] * sin_r,
        ];
        right = rolled_right;
    }
    let focal_px = (height_f * 0.5) / (fov * 0.5).tan().max(0.001);
    let far = distance.max(1.0) + width_f.max(height_f) / height_f * 24.0;
    let optics = world
        .camera
        .depth_of_field
        .as_ref()
        .map(|value| {
            Ok::<[f32; 4], WorldRenderError>([
                eval_number(&value.focus_distance, distance, time)?.max(0.05),
                eval_number(&value.focal_length_mm, 50.0, time)?.clamp(1.0, 300.0),
                eval_number(&value.f_stop, 2.8, time)?.clamp(0.7, 64.0),
                if value.max_blur_percent_height {
                    eval_number(&value.max_blur_px, 10.0, time)?.clamp(0.0, 10.0) * height_f / 100.0
                } else {
                    eval_number(&value.max_blur_px, 10.0, time)?.clamp(0.0, 32.0)
                },
            ])
        })
        .transpose()?
        .unwrap_or([0.0; 4]);
    Ok(PerspectiveCameraView {
        eye,
        right,
        up,
        forward,
        focal_px,
        near: 0.02,
        far,
        aspect: width_f / height_f,
        optics,
    })
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 0.000001 {
        return [f32::NAN, f32::NAN, f32::NAN];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[expect(
    clippy::too_many_arguments,
    reason = "Debug projections share the renderer caches and resolved scene context."
)]
fn draw_actor_debug_projections(
    canvas: &mut RgbaImage,
    graph: &WorldGraph,
    world: &WorldNode,
    asset_root: &Path,
    resolver: &dyn AssetResolver,
    time: WorldTime,
    mesh_cache: &mut HashMap<PathBuf, GlbMeshData>,
    primitive_texture_cache: &mut HashMap<PrimitiveTextureSourceKey, Arc<GlbTextureData>>,
) -> Result<(), WorldRenderError> {
    let camera_yaw = eval_number(&world.camera.yaw, 0.0, time)?;
    let camera_pitch = eval_number(&world.camera.pitch, 0.0, time)?;
    let camera_x = eval_number(&world.camera.x, 0.0, time)?;
    let camera_y = eval_number(&world.camera.y, 0.0, time)?;
    let camera_z = eval_number(&world.camera.z, 0.0, time)?;
    let camera_zoom = eval_number(&world.camera.zoom, 1.0, time)?.max(0.05);
    let fov = eval_number(&world.camera.fov, 35.0, time)?.clamp(10.0, 100.0);
    let distance = eval_number(&world.camera.distance, 3.2, time)?.max(0.2);

    for actor in &world.actors {
        let (_, mesh) = load_cached_actor_mesh(
            asset_root,
            actor,
            resolver,
            mesh_cache,
            primitive_texture_cache,
            &mut PrimitiveResourceLoadStats::default(),
        )?;
        let x = eval_number(&actor.x, 0.0, time)?;
        let y = eval_number(&actor.y, 0.0, time)?;
        let z = eval_number(&actor.z, 0.0, time)?;
        let yaw = eval_number(&actor.yaw, 0.0, time)?;
        let scale = eval_number(&actor.scale, 1.0, time)?.max(0.01);
        let opacity = eval_number(&actor.opacity, 1.0, time)?.clamp(0.0, 1.0);
        let view = camera_actor_view(
            x,
            y,
            z,
            yaw,
            camera_x,
            camera_y,
            camera_z,
            camera_yaw,
            camera_pitch,
        );
        if mesh.positions.is_empty() || mesh.indices.len() < 3 {
            draw_actor_placeholder(
                canvas,
                actor,
                false,
                view.x,
                view.y,
                view.yaw,
                view.pitch,
                fov,
                distance,
                scale * camera_zoom,
                opacity,
            );
        } else {
            let skinned_positions = skinned_actor_positions(graph, actor, mesh, time)?;
            let positions = skinned_positions.as_deref().unwrap_or(&mesh.positions);
            draw_actor_mesh_projection(
                canvas,
                graph,
                actor,
                mesh,
                positions,
                view.x,
                view.y,
                view.depth,
                view.yaw,
                view.pitch,
                fov,
                distance,
                scale * camera_zoom,
                opacity,
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_actor_gpu_draws(
    mut canvas: Option<&mut RgbaImage>,
    width: u32,
    height: u32,
    collect_editor_rig_snapshot: bool,
    collect_rig_diagnostics: bool,
    graph: &WorldGraph,
    world: &WorldNode,
    asset_root: &Path,
    resolver: &dyn AssetResolver,
    time: WorldTime,
    mesh_cache: &mut HashMap<PathBuf, GlbMeshData>,
    primitive_texture_cache: &mut HashMap<PrimitiveTextureSourceKey, Arc<GlbTextureData>>,
    effective_bounds_cache: &mut HashMap<PathBuf, ([f32; 3], [f32; 3])>,
    gpu_static_draw_cache: &mut HashMap<GpuWorldStaticPlanKey, Vec<GpuWorldStaticDraw>>,
    skinning_strategy_cache: &mut HashMap<SkinningStrategyKey, SkinningMatrixStrategy>,
    material_overrides: &[WorldMaterialTextureOverride],
) -> Result<ActorGpuDrawBuild, WorldRenderError> {
    let camera_zoom = eval_number(&world.camera.zoom, 1.0, time)?.max(0.05);
    let camera_view = perspective_camera_view(world, width, height, time)?;
    let mut draws = Vec::new();
    let mut editor_joints = Vec::new();
    let mut rig_reports = Vec::new();

    // Resolve animation-only GLBs into the same retained mesh cache. Only
    // skeleton channels are sampled; their geometry is never submitted.
    let asset_started = ProfileClock::now();
    let mut resource_stats = PrimitiveResourceLoadStats::default();
    let mut animation_keys = HashMap::<String, PathBuf>::new();
    for asset in &graph.animation_assets {
        let (key, _) = load_cached_glb_animation_resolved(
            asset_root,
            &asset.src,
            WorldPathStyle::Relative,
            resolver,
            mesh_cache,
        )?;
        animation_keys.insert(asset.id.clone(), key);
    }

    // Prepare every actor before drawing so cross-actor constraints can inspect
    // both sampled skeletons at the same frame.
    let mut model_keys = HashMap::<String, PathBuf>::new();
    for actor in &world.actors {
        let mut lod_actor = actor.clone();
        if let Some(vegetation) = lod_actor.vegetation.as_mut()
            && vegetation.lod == crate::dsl::VegetationLod::Auto
        {
            let pose = actor_frame_pose(actor, time)?;
            vegetation.lod = vegetation_auto_lod(
                vegetation.height * pose.scale.abs(),
                pose.position,
                camera_view.eye,
            );
        }
        let (model_key, _) = load_cached_actor_mesh(
            asset_root,
            &lod_actor,
            resolver,
            mesh_cache,
            primitive_texture_cache,
            &mut resource_stats,
        )?;
        model_keys.insert(actor.id.clone(), model_key);
    }
    let asset_resolve_ms = asset_started.elapsed().as_secs_f64() * 1000.0;
    let animation_started = ProfileClock::now();
    let mut sampled_by_actor = HashMap::<String, HashMap<usize, SampledNodeTrs>>::new();
    let mut poses = HashMap::<String, ActorFramePose>::new();
    for actor in &world.actors {
        let model_key = model_keys
            .get(&actor.id)
            .expect("actor model key prepared before animation sampling");
        let mesh = mesh_cache
            .get(model_key)
            .expect("target GLB inserted before external animation sampling");
        let has_external_action = graph
            .apply_actions
            .iter()
            .any(|apply| apply.target == actor.id && animation_keys.contains_key(&apply.action));
        let sampled = if has_external_action {
            sample_external_actor_actions(graph, actor, mesh, &animation_keys, mesh_cache, time)?
        } else {
            HashMap::new()
        };
        sampled_by_actor.insert(actor.id.clone(), sampled);
        poses.insert(actor.id.clone(), actor_frame_pose(actor, time)?);
    }
    let animation_sample_ms = animation_started.elapsed().as_secs_f64() * 1000.0;
    let constraints_started = ProfileClock::now();
    let mut constraint_overrides = scene_constraint_overrides(
        graph,
        world,
        &model_keys,
        mesh_cache,
        &sampled_by_actor,
        &poses,
        time,
    )?;
    let mut render_actors = world.actors.clone();
    apply_world_attachments(
        graph,
        &mut render_actors,
        &model_keys,
        mesh_cache,
        &sampled_by_actor,
        &mut constraint_overrides,
        &mut poses,
        time,
    )?;
    let constraints_ms = constraints_started.elapsed().as_secs_f64() * 1000.0;

    let draw_started = ProfileClock::now();
    for actor in &render_actors {
        let model_key = model_keys
            .get(&actor.id)
            .expect("actor model key prepared before rendering");
        let mesh = mesh_cache
            .get(model_key)
            .expect("target GLB inserted before rendering");
        let effective_bounds = *effective_bounds_cache
            .entry(model_key.clone())
            .or_insert_with(|| effective_mesh_bounds(mesh));
        let external_sampled = sampled_by_actor
            .get(&actor.id)
            .expect("actor sample prepared before rendering");
        let pose = poses
            .get(&actor.id)
            .copied()
            .expect("actor pose prepared before rendering");
        let actor_constraint_overrides = constraint_overrides
            .get(&actor.id)
            .cloned()
            .unwrap_or_default();
        let x = pose.position[0];
        let y = pose.position[1];
        let z = pose.position[2];
        let yaw = pose.rotation_deg[1];
        let pitch = pose.rotation_deg[0];
        let roll = pose.rotation_deg[2];
        let scale = pose.scale;
        let opacity = pose.opacity;
        if mesh.positions.is_empty() || mesh.indices.len() < 3 {
            if let Some(canvas) = canvas.as_deref_mut() {
                draw_actor_placeholder(
                    canvas,
                    actor,
                    false,
                    x,
                    y,
                    yaw,
                    0.0,
                    35.0,
                    3.2,
                    scale * camera_zoom,
                    opacity,
                );
            }
            continue;
        }
        let static_plan_key = GpuWorldStaticPlanKey {
            model_path: model_key.clone(),
            outline: actor
                .material
                .as_ref()
                .is_some_and(|material| material.outline),
            hide_meshes: actor.hide_meshes.clone(),
            hide_materials: actor.hide_materials.clone(),
        };
        if !gpu_static_draw_cache.contains_key(&static_plan_key) {
            let static_draws = build_actor_mesh_gpu_static_draws(actor, mesh, model_key);
            gpu_static_draw_cache.insert(static_plan_key.clone(), static_draws);
        }
        let static_draws = gpu_static_draw_cache
            .get(&static_plan_key)
            .expect("GPU static draw cache entry inserted before render");
        let mut cel_textures = HashMap::new();
        for binding in &actor.cel_materials {
            if let Some(source) = &binding.cel.control_map {
                let data = load_cached_primitive_texture(
                    asset_root,
                    WorldPathStyle::Relative,
                    source,
                    resolver,
                    primitive_texture_cache,
                    &mut resource_stats,
                )?;
                cel_textures.insert(
                    binding.material.clone(),
                    Arc::new(GpuWorldTexture::new(data.width, data.height, data.rgba)),
                );
            }
        }
        let actor_draws = build_actor_mesh_gpu_draws(
            graph,
            actor,
            mesh,
            effective_bounds,
            static_draws,
            width,
            height,
            x,
            y,
            z,
            yaw,
            pitch,
            roll,
            camera_view,
            scale * camera_zoom,
            opacity,
            time,
            model_key,
            skinning_strategy_cache,
            material_overrides,
            &cel_textures,
            external_sampled,
            &actor_constraint_overrides,
        )?;
        draws.extend(actor_draws);
        if collect_editor_rig_snapshot || collect_rig_diagnostics {
            let (actor_joints, rig_report) = project_actor_editor_joints(
                graph,
                actor,
                mesh,
                effective_bounds,
                width,
                height,
                pose,
                time,
                camera_view,
                external_sampled,
                &actor_constraint_overrides,
                collect_rig_diagnostics,
            )?;
            if collect_editor_rig_snapshot {
                editor_joints.extend(actor_joints);
            }
            if let Some(rig_report) = rig_report {
                rig_reports.push(rig_report);
            }
        }
    }
    let draw_assembly_ms = draw_started.elapsed().as_secs_f64() * 1000.0;
    Ok((
        draws,
        ActorBuildStages {
            asset_resolve_ms,
            animation_sample_ms,
            constraints_ms,
            draw_assembly_ms,
            texture_decode_ms: resource_stats.texture_decode_ms,
            texture_decode_count: resource_stats.texture_decode_count,
            texture_cache_hits: resource_stats.texture_cache_hits,
            texture_decoded_bytes: resource_stats.texture_decoded_bytes,
        },
        editor_joints,
        rig_reports,
    ))
}

fn vegetation_auto_lod(
    world_height: f32,
    position: [f32; 3],
    camera_eye: [f32; 3],
) -> crate::dsl::VegetationLod {
    let distance = position
        .iter()
        .zip(camera_eye)
        .map(|(value, eye)| (value - eye) * (value - eye))
        .sum::<f32>()
        .sqrt();
    let relative_distance = distance / world_height.max(0.05);
    if relative_distance < 5.0 {
        crate::dsl::VegetationLod::Full
    } else if relative_distance < 12.0 {
        crate::dsl::VegetationLod::Half
    } else {
        crate::dsl::VegetationLod::Quarter
    }
}

fn canonical_humanoid_editor_bones() -> &'static [&'static str] {
    &[
        "hips",
        "spine",
        "chest",
        "upper_chest",
        "neck",
        "head",
        "shoulder_l",
        "upper_arm_l",
        "forearm_l",
        "hand_l",
        "shoulder_r",
        "upper_arm_r",
        "forearm_r",
        "hand_r",
        "upper_leg_l",
        "lower_leg_l",
        "foot_l",
        "toe_l",
        "upper_leg_r",
        "lower_leg_r",
        "foot_r",
        "toe_r",
        "thumb_1_l",
        "thumb_2_l",
        "thumb_3_l",
        "index_1_l",
        "index_2_l",
        "index_3_l",
        "middle_1_l",
        "middle_2_l",
        "middle_3_l",
        "ring_1_l",
        "ring_2_l",
        "ring_3_l",
        "pinky_1_l",
        "pinky_2_l",
        "pinky_3_l",
        "thumb_1_r",
        "thumb_2_r",
        "thumb_3_r",
        "index_1_r",
        "index_2_r",
        "index_3_r",
        "middle_1_r",
        "middle_2_r",
        "middle_3_r",
        "ring_1_r",
        "ring_2_r",
        "ring_3_r",
        "pinky_1_r",
        "pinky_2_r",
        "pinky_3_r",
    ]
}

#[allow(clippy::too_many_arguments)]
fn project_actor_editor_joints(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
    effective_bounds: ([f32; 3], [f32; 3]),
    width: u32,
    height: u32,
    pose: ActorFramePose,
    time: WorldTime,
    camera: PerspectiveCameraView,
    external_sampled: &HashMap<usize, SampledNodeTrs>,
    constraint_overrides: &HashMap<String, BoneOverride>,
    collect_rig_diagnostics: bool,
) -> Result<
    (
        Vec<Scene3DEditorJointProjection>,
        Option<crate::rig_diagnostics::RigEvaluationReport>,
    ),
    WorldRenderError,
> {
    if let Some(native) = actor.native_skin.as_ref() {
        let rotation = actor
            .rotation_quaternion
            .map(mat4_from_quat)
            .unwrap_or_else(|| {
                mat4_from_quat(actor_yxz_quaternion(
                    pose.rotation_deg[0],
                    pose.rotation_deg[1],
                    pose.rotation_deg[2],
                ))
            });
        let mut joints = Vec::with_capacity(native.joint_positions.len());
        for (bone, point) in native.joint_names.iter().zip(&native.joint_positions) {
            let local = point.map(|value| value * pose.scale);
            let rotated = mat4_transform_point(rotation, local);
            let world = [
                pose.position[0] + rotated[0],
                pose.position[1] + rotated[1],
                pose.position[2] + rotated[2],
            ];
            let relative = [
                world[0] - camera.eye[0],
                world[1] - camera.eye[1],
                world[2] - camera.eye[2],
            ];
            let depth = dot3(relative, camera.forward);
            if depth <= camera.near || !depth.is_finite() {
                continue;
            }
            let x = width as f32 * 0.5 + dot3(relative, camera.right) * camera.focal_px / depth;
            let y = height as f32 * 0.5 - dot3(relative, camera.up) * camera.focal_px / depth;
            if x.is_finite() && y.is_finite() {
                joints.push(Scene3DEditorJointProjection {
                    actor: actor.id.clone(),
                    bone: bone.clone(),
                    x,
                    y,
                    depth,
                });
            }
        }
        return Ok((joints, None));
    }
    let frame = prepare_actor_joint_frame(
        graph,
        actor,
        mesh,
        time,
        external_sampled,
        constraint_overrides,
    )?;
    let matrices = frame.global_matrices.unwrap_or_else(|| {
        global_node_matrices_with_sampled(mesh, &HashMap::new(), external_sampled)
    });
    let retargeted_matrices = if collect_rig_diagnostics {
        let retargeted_overrides = actor_bone_overrides_for_mesh(graph, actor, Some(mesh), time)?;
        Some(actor_global_node_matrices(
            graph,
            actor,
            mesh,
            time,
            &retargeted_overrides,
            Some(external_sampled),
        )?)
    } else {
        None
    };
    let profile = actor_model_profile(graph, actor);
    let (effective_min, effective_max) = effective_bounds;
    let normalize_height = actor.scale_mode.eq_ignore_ascii_case("normalize_height");
    let (center_x, origin_y, center_z, world_scale) = if normalize_height {
        let model_height = (effective_max[1] - effective_min[1]).abs().max(0.001);
        (
            (effective_min[0] + effective_max[0]) * 0.5,
            effective_min[1],
            (effective_min[2] + effective_max[2]) * 0.5,
            pose.scale / model_height,
        )
    } else {
        (0.0, 0.0, 0.0, pose.scale)
    };
    let mut report = if let Some(retargeted_matrices) = retargeted_matrices.as_deref() {
        Some(
            pose_diagnostics::runtime_world_actor_rig_from_matrices(
                graph,
                mesh,
                actor,
                time,
                Some(retargeted_matrices),
                &matrices,
                true,
            )
            .map_err(|error| WorldRenderError::Expression {
                expr: format!("rig diagnostics for actor '{}'", actor.id),
                message: error.to_string(),
            })?,
        )
    } else {
        None
    };
    let mut joints = Vec::new();
    for bone in canonical_humanoid_editor_bones() {
        let Some(index) = target_node_for_canonical_bone(mesh, profile, bone) else {
            continue;
        };
        let Some(matrix) = matrices.get(index).copied() else {
            continue;
        };
        let point = matrix_translation(matrix);
        let local = [
            (point[0] - center_x) * world_scale,
            (point[1] - origin_y) * world_scale,
            (point[2] - center_z) * world_scale,
        ];
        let rotated = rotate_actor_vector(local, pose.rotation_deg);
        let world = [
            pose.position[0] + rotated[0],
            pose.position[1] + rotated[1],
            pose.position[2] + rotated[2],
        ];
        let relative = [
            world[0] - camera.eye[0],
            world[1] - camera.eye[1],
            world[2] - camera.eye[2],
        ];
        let depth = dot3(relative, camera.forward);
        if depth <= camera.near || !depth.is_finite() {
            continue;
        }
        let x = width as f32 * 0.5 + dot3(relative, camera.right) * camera.focal_px / depth;
        let y = height as f32 * 0.5 - dot3(relative, camera.up) * camera.focal_px / depth;
        if x.is_finite() && y.is_finite() {
            if let Some(bone_report) = report.as_mut().and_then(|report| {
                report
                    .bones
                    .iter_mut()
                    .find(|entry| entry.canonical_bone == *bone)
            }) {
                bone_report
                    .stages
                    .push(crate::rig_diagnostics::BoneStageTransform {
                        stage: crate::rig_diagnostics::BonePoseStage::ScreenProjected,
                        space: "authoredPixels".into(),
                        position: None,
                        rotation_quaternion: None,
                        matrix: None,
                        screen: Some(crate::rig_diagnostics::ScreenProjection {
                            x,
                            y,
                            depth,
                            width,
                            height,
                        }),
                    });
            }
            joints.push(Scene3DEditorJointProjection {
                actor: actor.id.clone(),
                bone: (*bone).to_string(),
                x,
                y,
                depth,
            });
        }
    }
    if let Some(report) = report.as_mut() {
        report.capabilities.screen_projection = true;
    }
    Ok((joints, report))
}

#[derive(Debug, Clone, Copy)]
struct ActorFramePose {
    position: [f32; 3],
    rotation_deg: [f32; 3],
    scale: f32,
    opacity: f32,
}

fn actor_frame_pose(
    actor: &WorldActor,
    time: WorldTime,
) -> Result<ActorFramePose, WorldRenderError> {
    Ok(ActorFramePose {
        position: [
            eval_number(&actor.x, 0.0, time)?,
            eval_number(&actor.y, 0.0, time)?,
            eval_number(&actor.z, 0.0, time)?,
        ],
        rotation_deg: [
            eval_number(&actor.pitch, 0.0, time)?,
            eval_number(&actor.yaw, 0.0, time)?,
            eval_number(&actor.roll, 0.0, time)?,
        ],
        scale: eval_number(&actor.scale, 1.0, time)?.max(0.01),
        opacity: eval_number(&actor.opacity, 1.0, time)?.clamp(0.0, 1.0),
    })
}

/// Resolve every socket binding after animation and IK, before draw transforms are built.
#[allow(clippy::too_many_arguments)]
fn apply_world_attachments(
    graph: &WorldGraph,
    actors: &mut [WorldActor],
    model_keys: &HashMap<String, PathBuf>,
    mesh_cache: &HashMap<PathBuf, GlbMeshData>,
    sampled_by_actor: &HashMap<String, HashMap<usize, SampledNodeTrs>>,
    constraint_overrides: &mut HashMap<String, HashMap<String, BoneOverride>>,
    poses: &mut HashMap<String, ActorFramePose>,
    time: WorldTime,
) -> Result<(), WorldRenderError> {
    let owners = graph
        .attachments
        .iter()
        .map(|attachment| attachment.object.as_str())
        .collect::<HashSet<_>>();
    let mut resolved = HashSet::<String>::new();
    let mut solved = HashMap::<String, AttachmentObjectTransform>::new();
    let mut remaining = graph.attachments.len();
    while remaining > 0 {
        let before = remaining;
        for attachment in &graph.attachments {
            if resolved.contains(&attachment.id)
                || (owners.contains(attachment_primary(attachment).target_model.as_str())
                    && !graph.attachments.iter().any(|candidate| {
                        candidate.object == attachment_primary(attachment).target_model
                            && resolved.contains(&candidate.id)
                    }))
            {
                continue;
            }
            let transform = resolve_world_attachment(
                graph,
                actors,
                attachment,
                model_keys,
                mesh_cache,
                sampled_by_actor,
                constraint_overrides,
                poses,
                time,
            )?;
            solved.insert(attachment.id.clone(), transform);
            resolved.insert(attachment.id.clone());
            remaining -= 1;
        }
        if remaining == before {
            return Err(WorldRenderError::GpuRender {
                message: "Attachment dependency cycle reached the renderer.".to_string(),
            });
        }
    }
    for attachment in &graph.attachments {
        apply_secondary_attachment_ik(
            graph,
            actors,
            attachment,
            solved
                .get(&attachment.id)
                .expect("primary attachment transform was solved"),
            model_keys,
            mesh_cache,
            sampled_by_actor,
            constraint_overrides,
            poses,
            time,
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AttachmentObjectTransform {
    position: [f32; 3],
    rotation: [f32; 4],
    scale: f32,
}

fn attachment_primary(
    attachment: &crate::world::WorldAttachment,
) -> &crate::world::WorldAttachmentTarget {
    attachment
        .attaches
        .iter()
        .find(|attach| attach.drive == "object")
        .expect("validated Attachment has one object driver")
}

#[allow(clippy::too_many_arguments)]
fn resolve_world_attachment(
    graph: &WorldGraph,
    actors: &mut [WorldActor],
    attachment: &crate::world::WorldAttachment,
    model_keys: &HashMap<String, PathBuf>,
    mesh_cache: &HashMap<PathBuf, GlbMeshData>,
    sampled_by_actor: &HashMap<String, HashMap<usize, SampledNodeTrs>>,
    constraint_overrides: &HashMap<String, HashMap<String, BoneOverride>>,
    poses: &mut HashMap<String, ActorFramePose>,
    time: WorldTime,
) -> Result<AttachmentObjectTransform, WorldRenderError> {
    let primary = attachment_primary(attachment);
    let socket = attachment
        .sockets
        .iter()
        .find(|socket| socket.socket_id == primary.socket_id)
        .expect("validated primary Attachment Socket exists");
    let target_actor = actors
        .iter()
        .find(|actor| actor.id == primary.target_model)
        .ok_or_else(|| WorldRenderError::GpuRender {
            message: format!(
                "Attachment '{}' target Model not found: {}",
                attachment.id, primary.target_model
            ),
        })?;
    let target_mesh = model_keys
        .get(&primary.target_model)
        .and_then(|key| mesh_cache.get(key))
        .ok_or_else(|| WorldRenderError::GpuRender {
            message: format!("Attachment '{}' target mesh is unavailable.", attachment.id),
        })?;
    let target_profile = actor_model_profile(graph, target_actor);
    let target_node =
        target_node_for_canonical_bone(target_mesh, target_profile, &primary.target_bone)
            .ok_or_else(|| WorldRenderError::GpuRender {
                message: format!(
                    "Attachment '{}' target bone not found: {}.{}",
                    attachment.id, primary.target_model, primary.target_bone
                ),
            })?;
    let mut overrides =
        actor_bone_overrides_for_mesh(graph, target_actor, Some(target_mesh), time)?;
    if let Some(values) = constraint_overrides.get(&primary.target_model) {
        overrides.extend(values.iter().map(|(bone, value)| (bone.clone(), *value)));
    }
    let sampled = sampled_by_actor
        .get(&primary.target_model)
        .cloned()
        .unwrap_or_default();
    let matrices = actor_global_node_matrices(
        graph,
        target_actor,
        target_mesh,
        time,
        &overrides,
        Some(&sampled),
    )?;
    let bone_matrix =
        matrices
            .get(target_node)
            .copied()
            .ok_or_else(|| WorldRenderError::GpuRender {
                message: format!(
                    "Attachment '{}' target bone matrix is unavailable.",
                    attachment.id
                ),
            })?;
    let target_pose =
        poses
            .get(&primary.target_model)
            .copied()
            .ok_or_else(|| WorldRenderError::GpuRender {
                message: format!("Attachment '{}' target pose is unavailable.", attachment.id),
            })?;
    let target_root_rotation = target_actor.rotation_quaternion.unwrap_or_else(|| {
        actor_yxz_quaternion(
            target_pose.rotation_deg[0],
            target_pose.rotation_deg[1],
            target_pose.rotation_deg[2],
        )
    });
    let target_bone_rotation = quat_normalize_xyzw(quat_mul_xyzw(
        target_root_rotation,
        quat_from_mat4_rotation(bone_matrix),
    ));
    let target_bone_position = actor_model_point_to_world_for_attachment(
        matrix_translation(bone_matrix),
        target_actor,
        target_mesh,
        target_pose,
        target_root_rotation,
    );

    let socket_position = eval_attachment_vec3(&socket.socket_position, time)?;
    let socket_rotation = eval_attachment_vec3(&socket.socket_rotation, time)?;
    let socket_quaternion =
        actor_yxz_quaternion(socket_rotation[0], socket_rotation[1], socket_rotation[2]);
    let desired_rotation = quat_normalize_xyzw(quat_mul_xyzw(
        target_bone_rotation,
        quat_conjugate_xyzw(socket_quaternion),
    ));
    let compound_prefix = format!("{}::", attachment.object);
    let object_actor_index = actors
        .iter()
        .position(|actor| actor.id == attachment.object);
    let compound_actor_indices = actors
        .iter()
        .enumerate()
        .filter_map(|(index, actor)| actor.id.starts_with(&compound_prefix).then_some(index))
        .collect::<Vec<_>>();
    if object_actor_index.is_none() && compound_actor_indices.is_empty() {
        return Err(WorldRenderError::GpuRender {
            message: format!(
                "Attachment '{}' object Model not found: {}",
                attachment.id, attachment.object
            ),
        });
    }
    let object_pose = object_actor_index
        .and_then(|index| poses.get(&actors[index].id).copied())
        .unwrap_or(ActorFramePose {
            position: attachment.object_position,
            rotation_deg: attachment.object_rotation,
            scale: attachment.object_scale,
            opacity: 1.0,
        });
    let socket_local = if let Some(index) = object_actor_index {
        let object_actor = &actors[index];
        let object_mesh = model_keys
            .get(&object_actor.id)
            .and_then(|key| mesh_cache.get(key))
            .ok_or_else(|| WorldRenderError::GpuRender {
                message: format!("Attachment '{}' object mesh is unavailable.", attachment.id),
            })?;
        actor_model_local_point_for_attachment(
            socket_position,
            object_actor,
            object_mesh,
            object_pose.scale,
        )
    } else {
        socket_position.map(|value| value * object_pose.scale)
    };
    let socket_offset = mat4_transform_point(mat4_from_quat(desired_rotation), socket_local);
    let desired_position: [f32; 3] =
        std::array::from_fn(|axis| target_bone_position[axis] - socket_offset[axis]);
    let position_weight = eval_number(&primary.position_weight, 1.0, time)?.clamp(0.0, 1.0);
    let rotation_weight = eval_number(&primary.rotation_weight, 1.0, time)?.clamp(0.0, 1.0);
    let authored_rotation = object_actor_index
        .and_then(|index| actors[index].rotation_quaternion)
        .unwrap_or_else(|| {
            actor_yxz_quaternion(
                object_pose.rotation_deg[0],
                object_pose.rotation_deg[1],
                object_pose.rotation_deg[2],
            )
        });
    let final_rotation = quat_slerp_shortest(authored_rotation, desired_rotation, rotation_weight);
    let final_position = std::array::from_fn(|axis| {
        object_pose.position[axis] * (1.0 - position_weight)
            + desired_position[axis] * position_weight
    });
    if let Some(index) = object_actor_index {
        actors[index].rotation_quaternion = Some(final_rotation);
        if let Some(pose) = poses.get_mut(&attachment.object) {
            pose.position = final_position;
        }
    } else {
        // CompoundAsset instances are flattened by the Scene bridge. Apply the
        // solved parent delta to every child so the assembled prop stays rigid.
        let parent_delta_rotation = quat_normalize_xyzw(quat_mul_xyzw(
            final_rotation,
            quat_conjugate_xyzw(authored_rotation),
        ));
        for index in compound_actor_indices {
            let child_id = actors[index].id.clone();
            let Some(child_pose) = poses.get(&child_id).copied() else {
                continue;
            };
            let relative_position =
                std::array::from_fn(|axis| child_pose.position[axis] - object_pose.position[axis]);
            let rotated_relative =
                mat4_transform_point(mat4_from_quat(parent_delta_rotation), relative_position);
            let child_rotation = actors[index].rotation_quaternion.unwrap_or_else(|| {
                actor_yxz_quaternion(
                    child_pose.rotation_deg[0],
                    child_pose.rotation_deg[1],
                    child_pose.rotation_deg[2],
                )
            });
            actors[index].rotation_quaternion = Some(quat_normalize_xyzw(quat_mul_xyzw(
                parent_delta_rotation,
                child_rotation,
            )));
            if let Some(pose) = poses.get_mut(&child_id) {
                pose.position =
                    std::array::from_fn(|axis| final_position[axis] + rotated_relative[axis]);
            }
        }
    }
    Ok(AttachmentObjectTransform {
        position: final_position,
        rotation: final_rotation,
        scale: object_pose.scale,
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_secondary_attachment_ik(
    graph: &WorldGraph,
    actors: &[WorldActor],
    attachment: &crate::world::WorldAttachment,
    object_transform: &AttachmentObjectTransform,
    model_keys: &HashMap<String, PathBuf>,
    mesh_cache: &HashMap<PathBuf, GlbMeshData>,
    sampled_by_actor: &HashMap<String, HashMap<usize, SampledNodeTrs>>,
    constraint_overrides: &mut HashMap<String, HashMap<String, BoneOverride>>,
    poses: &HashMap<String, ActorFramePose>,
    time: WorldTime,
) -> Result<(), WorldRenderError> {
    for attach in attachment
        .attaches
        .iter()
        .filter(|attach| attach.drive == "target" && attach.mode == "ik")
    {
        let socket = attachment
            .sockets
            .iter()
            .find(|socket| socket.socket_id == attach.socket_id)
            .expect("validated secondary Attachment Socket exists");
        let socket_position = eval_attachment_vec3(&socket.socket_position, time)?;
        let socket_rotation = eval_attachment_vec3(&socket.socket_rotation, time)?;
        let socket_local = socket_position.map(|value| value * object_transform.scale);
        let socket_world_offset =
            mat4_transform_point(mat4_from_quat(object_transform.rotation), socket_local);
        let socket_world: [f32; 3] =
            std::array::from_fn(|axis| object_transform.position[axis] + socket_world_offset[axis]);
        let desired_hand_world_rotation = quat_normalize_xyzw(quat_mul_xyzw(
            object_transform.rotation,
            actor_yxz_quaternion(socket_rotation[0], socket_rotation[1], socket_rotation[2]),
        ));
        let actor = actors
            .iter()
            .find(|actor| actor.id == attach.target_model)
            .ok_or_else(|| WorldRenderError::GpuRender {
                message: format!(
                    "Attachment '{}' secondary target is unavailable.",
                    attachment.id
                ),
            })?;
        let mesh = model_keys
            .get(&actor.id)
            .and_then(|key| mesh_cache.get(key))
            .ok_or_else(|| WorldRenderError::GpuRender {
                message: format!(
                    "Attachment '{}' secondary target mesh is unavailable.",
                    attachment.id
                ),
            })?;
        let pose = poses[&actor.id];
        let sampled = sampled_by_actor.get(&actor.id).cloned().unwrap_or_default();
        let profile = actor_model_profile(graph, actor);
        let Some((root_bone, mid_bone, end_bone)) = humanoid_two_bone_chain(&attach.target_bone)
        else {
            continue;
        };
        let indices = [root_bone, mid_bone, end_bone]
            .map(|bone| target_node_for_canonical_bone(mesh, profile, bone));
        let [Some(root_index), Some(mid_index), Some(end_index)] = indices else {
            continue;
        };
        let names = [
            (root_index, root_bone),
            (mid_index, mid_bone),
            (end_index, end_bone),
        ]
        .map(|(index, fallback)| mesh.nodes[index].name.as_deref().unwrap_or(fallback));
        let root_rotation = actor.rotation_quaternion.unwrap_or_else(|| {
            actor_yxz_quaternion(
                pose.rotation_deg[0],
                pose.rotation_deg[1],
                pose.rotation_deg[2],
            )
        });
        let world_delta = std::array::from_fn(|axis| socket_world[axis] - pose.position[axis]);
        let local_scaled = mat4_transform_point(
            mat4_from_quat(quat_conjugate_xyzw(root_rotation)),
            world_delta,
        );
        let height = (mesh.bounds_max[1] - mesh.bounds_min[1]).abs().max(0.001);
        let mut target_local = if actor.scale_mode.eq_ignore_ascii_case("normalize_height") {
            let center_x = (mesh.bounds_min[0] + mesh.bounds_max[0]) * 0.5;
            let center_z = (mesh.bounds_min[2] + mesh.bounds_max[2]) * 0.5;
            [
                local_scaled[0] * height / pose.scale + center_x,
                local_scaled[1] * height / pose.scale + mesh.bounds_min[1],
                local_scaled[2] * height / pose.scale + center_z,
            ]
        } else {
            local_scaled.map(|value| value / pose.scale.max(0.0001))
        };
        let overrides =
            constraint_overrides
                .entry(actor.id.clone())
                .or_insert(actor_bone_overrides_for_mesh(
                    graph,
                    actor,
                    Some(mesh),
                    time,
                )?);
        let before = global_node_matrices_with_sampled(mesh, overrides, &sampled);
        let root = matrix_translation(before[root_index]);
        let mid = matrix_translation(before[mid_index]);
        let end = matrix_translation(before[end_index]);
        let reach = (attachment_length3(attachment_sub3(root, mid))
            + attachment_length3(attachment_sub3(mid, end)))
        .max(0.0001);
        let max_stretch = eval_number(&attach.max_stretch, 1.05, time)?.max(1.0);
        let delta = attachment_sub3(target_local, root);
        let distance = attachment_length3(delta);
        if distance > reach * max_stretch {
            if std::env::var_os("MOTIONLOOM_ATTACHMENT_DIAGNOSTICS").is_some() {
                eprintln!(
                    "motionloom attachment '{}': {}.{} cannot reach socket '{}' ({distance:.4} > {:.4}); clamping IK target without moving the prop",
                    attachment.id,
                    attach.target_model,
                    attach.target_bone,
                    attach.socket_id,
                    reach * max_stretch
                );
            }
            let direction = normalize3(delta);
            target_local =
                std::array::from_fn(|axis| root[axis] + direction[axis] * reach * max_stretch);
        }
        let position_weight = eval_number(&attach.position_weight, 1.0, time)?.clamp(0.0, 1.0);
        let iteration_weight = 1.0 - (1.0 - position_weight).powf(1.0 / 3.0);
        for _ in 0..3 {
            solve_two_bone_constraint(
                mesh,
                names[0],
                names[1],
                names[2],
                root_index,
                mid_index,
                end_index,
                target_local,
                iteration_weight,
                &sampled,
                overrides,
            );
        }
        let rotation_weight = eval_number(&attach.rotation_weight, 1.0, time)?.clamp(0.0, 1.0);
        if rotation_weight > f32::EPSILON {
            let matrices = global_node_matrices_with_sampled(mesh, overrides, &sampled);
            let current_global = quat_from_mat4_rotation(matrices[end_index]);
            let desired_model = quat_normalize_xyzw(quat_mul_xyzw(
                quat_conjugate_xyzw(root_rotation),
                desired_hand_world_rotation,
            ));
            let parent_global = mesh.nodes[end_index]
                .parent
                .and_then(|index| matrices.get(index).copied())
                .map(quat_from_mat4_rotation)
                .unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let current_local = quat_mul_xyzw(quat_conjugate_xyzw(parent_global), current_global);
            let desired_local = quat_mul_xyzw(quat_conjugate_xyzw(parent_global), desired_model);
            let delta_local = quat_mul_xyzw(quat_conjugate_xyzw(current_local), desired_local);
            let rotation_delta = quat_to_zyx_euler_degrees(delta_local);
            let end_override = overrides
                .entry(names[2].to_string())
                .or_insert_with(BoneOverride::identity);
            for (axis, delta) in rotation_delta.iter().enumerate() {
                end_override.rotation_deg[axis] += delta * rotation_weight;
            }
        }
    }
    Ok(())
}

fn eval_attachment_vec3(
    values: &[String; 3],
    time: WorldTime,
) -> Result<[f32; 3], WorldRenderError> {
    Ok([
        eval_number(&values[0], 0.0, time)?,
        eval_number(&values[1], 0.0, time)?,
        eval_number(&values[2], 0.0, time)?,
    ])
}

fn attachment_sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|axis| a[axis] - b[axis])
}

fn attachment_length3(value: [f32; 3]) -> f32 {
    value
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt()
}

fn actor_model_point_to_world_for_attachment(
    point: [f32; 3],
    actor: &WorldActor,
    mesh: &GlbMeshData,
    pose: ActorFramePose,
    rotation: [f32; 4],
) -> [f32; 3] {
    let local = actor_model_local_point_for_attachment(point, actor, mesh, pose.scale);
    let rotated = mat4_transform_point(mat4_from_quat(rotation), local);
    std::array::from_fn(|axis| pose.position[axis] + rotated[axis])
}

fn actor_model_local_point_for_attachment(
    point: [f32; 3],
    actor: &WorldActor,
    mesh: &GlbMeshData,
    scale: f32,
) -> [f32; 3] {
    if actor.scale_mode.eq_ignore_ascii_case("normalize_height") {
        let height = (mesh.bounds_max[1] - mesh.bounds_min[1]).abs().max(0.001);
        let center_x = (mesh.bounds_min[0] + mesh.bounds_max[0]) * 0.5;
        let center_z = (mesh.bounds_min[2] + mesh.bounds_max[2]) * 0.5;
        [
            (point[0] - center_x) * scale / height,
            (point[1] - mesh.bounds_min[1]) * scale / height,
            (point[2] - center_z) * scale / height,
        ]
    } else {
        point.map(|value| value * scale)
    }
}

fn quat_slerp_shortest(a: [f32; 4], mut b: [f32; 4], weight: f32) -> [f32; 4] {
    let a = quat_normalize_xyzw(a);
    b = quat_normalize_xyzw(b);
    let mut dot = a
        .iter()
        .zip(b)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    if dot < 0.0 {
        b = b.map(|value| -value);
        dot = -dot;
    }
    if dot > 0.9995 {
        return quat_normalize_xyzw(std::array::from_fn(|axis| {
            a[axis] + (b[axis] - a[axis]) * weight
        }));
    }
    let angle = dot.clamp(-1.0, 1.0).acos();
    let sine = angle.sin().max(1.0e-6);
    let left = ((1.0 - weight) * angle).sin() / sine;
    let right = (weight * angle).sin() / sine;
    quat_normalize_xyzw(std::array::from_fn(|axis| a[axis] * left + b[axis] * right))
}

#[allow(clippy::too_many_arguments)]
fn scene_constraint_overrides(
    graph: &WorldGraph,
    world: &WorldNode,
    model_keys: &HashMap<String, PathBuf>,
    mesh_cache: &HashMap<PathBuf, GlbMeshData>,
    sampled_by_actor: &HashMap<String, HashMap<usize, SampledNodeTrs>>,
    poses: &HashMap<String, ActorFramePose>,
    time: WorldTime,
) -> Result<HashMap<String, HashMap<String, BoneOverride>>, WorldRenderError> {
    let mut out = HashMap::<String, HashMap<String, BoneOverride>>::new();
    let now_ms = time.time_sec() * 1000.0;
    for constraint in &graph.constraints {
        if now_ms < constraint.at_ms as f32
            || now_ms > constraint.at_ms.saturating_add(constraint.duration_ms) as f32
        {
            continue;
        }
        let Some((source_actor_id, source_bone)) = constraint.source.rsplit_once('.') else {
            continue;
        };
        let Some(source_actor) = world
            .actors
            .iter()
            .find(|actor| actor.id == source_actor_id)
        else {
            continue;
        };
        let Some(source_mesh) = model_keys
            .get(source_actor_id)
            .and_then(|key| mesh_cache.get(key))
        else {
            continue;
        };
        let Some(source_pose) = poses.get(source_actor_id).copied() else {
            continue;
        };
        let source_sampled = sampled_by_actor
            .get(source_actor_id)
            .cloned()
            .unwrap_or_default();
        let target_world = if let Some(point) = constraint.target_point {
            point
        } else {
            let Some((target_actor_id, target_bone)) = constraint.target.rsplit_once('.') else {
                continue;
            };
            let Some(target_actor) = world
                .actors
                .iter()
                .find(|actor| actor.id == target_actor_id)
            else {
                continue;
            };
            let Some(target_mesh) = model_keys
                .get(target_actor_id)
                .and_then(|key| mesh_cache.get(key))
            else {
                continue;
            };
            let Some(target_pose) = poses.get(target_actor_id).copied() else {
                continue;
            };
            let target_sampled = sampled_by_actor
                .get(target_actor_id)
                .cloned()
                .unwrap_or_default();
            let target_profile = actor_model_profile(graph, target_actor);
            let Some(target_node_index) =
                target_node_for_canonical_bone(target_mesh, target_profile, target_bone)
            else {
                continue;
            };
            let target_overrides =
                actor_bone_overrides_for_mesh(graph, target_actor, Some(target_mesh), time)?;
            let target_matrices =
                global_node_matrices_with_sampled(target_mesh, &target_overrides, &target_sampled);
            let Some(target_matrix) = target_matrices.get(target_node_index).copied() else {
                continue;
            };
            actor_model_point_to_world(matrix_translation(target_matrix), target_mesh, target_pose)
        };
        let target_source_local =
            actor_world_point_to_model(target_world, source_mesh, source_pose);
        let Some((root_bone, mid_bone, end_bone)) = humanoid_two_bone_chain(source_bone) else {
            continue;
        };
        let source_profile = actor_model_profile(graph, source_actor);
        let Some(root_index) =
            target_node_for_canonical_bone(source_mesh, source_profile, root_bone)
        else {
            continue;
        };
        let Some(mid_index) = target_node_for_canonical_bone(source_mesh, source_profile, mid_bone)
        else {
            continue;
        };
        let Some(end_index) = target_node_for_canonical_bone(source_mesh, source_profile, end_bone)
        else {
            continue;
        };
        let root_name = source_mesh.nodes[root_index]
            .name
            .as_deref()
            .unwrap_or(root_bone);
        let mid_name = source_mesh.nodes[mid_index]
            .name
            .as_deref()
            .unwrap_or(mid_bone);
        let end_name = source_mesh.nodes[end_index]
            .name
            .as_deref()
            .unwrap_or(end_bone);
        let weight = eval_number(&constraint.weight, 1.0, time)?.clamp(0.0, 1.0);
        if weight <= f32::EPSILON {
            continue;
        }
        let source_overrides = if let Some(overrides) = out.get_mut(source_actor_id) {
            overrides
        } else {
            let base = actor_bone_overrides_for_mesh(graph, source_actor, Some(source_mesh), time)?;
            out.entry(source_actor_id.to_string()).or_insert(base)
        };
        solve_two_bone_constraint(
            source_mesh,
            root_name,
            mid_name,
            end_name,
            root_index,
            mid_index,
            end_index,
            target_source_local,
            weight,
            &source_sampled,
            source_overrides,
        );
    }
    Ok(out)
}

fn humanoid_two_bone_chain(end: &str) -> Option<(&'static str, &'static str, &'static str)> {
    match end {
        "head" => Some(("chest", "neck", "head")),
        "forearm_l" => Some(("shoulder_l", "upper_arm_l", "forearm_l")),
        "forearm_r" => Some(("shoulder_r", "upper_arm_r", "forearm_r")),
        "hand_l" => Some(("upper_arm_l", "forearm_l", "hand_l")),
        "hand_r" => Some(("upper_arm_r", "forearm_r", "hand_r")),
        "lower_leg_l" => Some(("hips", "upper_leg_l", "lower_leg_l")),
        "lower_leg_r" => Some(("hips", "upper_leg_r", "lower_leg_r")),
        "foot_l" => Some(("upper_leg_l", "lower_leg_l", "foot_l")),
        "foot_r" => Some(("upper_leg_r", "lower_leg_r", "foot_r")),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_two_bone_constraint(
    mesh: &GlbMeshData,
    root_name: &str,
    mid_name: &str,
    _end_name: &str,
    root_index: usize,
    mid_index: usize,
    end_index: usize,
    target: [f32; 3],
    weight: f32,
    sampled: &HashMap<usize, SampledNodeTrs>,
    overrides: &mut HashMap<String, BoneOverride>,
) {
    let matrices = global_node_matrices_with_sampled(mesh, overrides, sampled);
    let root = matrix_translation(matrices[root_index]);
    let mid = matrix_translation(matrices[mid_index]);
    let end = matrix_translation(matrices[end_index]);
    let planes = [
        (0usize, 1usize, 2usize),
        (0usize, 2usize, 1usize),
        (1usize, 2usize, 0usize),
    ];
    let (u, v, rotation_axis) = planes
        .into_iter()
        .max_by(|(a_u, a_v, _), (b_u, b_v, _)| {
            let a = (target[*a_u] - root[*a_u]).powi(2) + (target[*a_v] - root[*a_v]).powi(2);
            let b = (target[*b_u] - root[*b_u]).powi(2) + (target[*b_v] - root[*b_v]).powi(2);
            a.total_cmp(&b)
        })
        .unwrap_or((0, 1, 2));
    let length_a = distance_2d(root, mid, u, v).max(0.0001);
    let length_b = distance_2d(mid, end, u, v).max(0.0001);
    let delta = [target[u] - root[u], target[v] - root[v]];
    let distance = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt().clamp(
        (length_a - length_b).abs() + 0.0001,
        length_a + length_b - 0.0001,
    );
    let target_angle = delta[1].atan2(delta[0]);
    let root_offset = ((length_a * length_a + distance * distance - length_b * length_b)
        / (2.0 * length_a * distance))
        .clamp(-1.0, 1.0)
        .acos();
    let current_bend =
        (mid[v] - root[v]) * (end[u] - mid[u]) - (mid[u] - root[u]) * (end[v] - mid[v]);
    let bend_sign = if current_bend.abs() <= 0.0001 {
        if root_name.ends_with("_l") { 1.0 } else { -1.0 }
    } else {
        current_bend.signum()
    };
    let desired_root_angle = target_angle + bend_sign * root_offset;
    let desired_mid = [
        root[u] + desired_root_angle.cos() * length_a,
        root[v] + desired_root_angle.sin() * length_a,
    ];
    let desired_second_angle = (target[v] - desired_mid[1]).atan2(target[u] - desired_mid[0]);
    let current_root_angle = segment_angle(root, mid, u, v);
    let root_response = local_axis_plane_response_with_sampled(
        mesh,
        overrides,
        sampled,
        root_name,
        root_index,
        mid_index,
        rotation_axis,
        u,
        v,
    );
    let root_delta = normalize_angle(desired_root_angle - current_root_angle).to_degrees()
        / stable_axis_response(root_response);
    let mut root_solved = overrides.clone();
    root_solved
        .entry(root_name.to_string())
        .or_insert_with(BoneOverride::identity)
        .rotation_deg[rotation_axis] += root_delta;
    let solved = global_node_matrices_with_sampled(mesh, &root_solved, sampled);
    let solved_mid = matrix_translation(solved[mid_index]);
    let solved_end = matrix_translation(solved[end_index]);
    let current_second_angle = segment_angle(solved_mid, solved_end, u, v);
    let mid_response = local_axis_plane_response_with_sampled(
        mesh,
        &root_solved,
        sampled,
        mid_name,
        mid_index,
        end_index,
        rotation_axis,
        u,
        v,
    );
    let mid_delta = normalize_angle(desired_second_angle - current_second_angle).to_degrees()
        / stable_axis_response(mid_response);
    overrides
        .entry(root_name.to_string())
        .or_insert_with(BoneOverride::identity)
        .rotation_deg[rotation_axis] += root_delta * weight;
    overrides
        .entry(mid_name.to_string())
        .or_insert_with(BoneOverride::identity)
        .rotation_deg[rotation_axis] += mid_delta * weight;
}

#[allow(clippy::too_many_arguments)]
fn local_axis_plane_response_with_sampled(
    mesh: &GlbMeshData,
    overrides: &HashMap<String, BoneOverride>,
    sampled: &HashMap<usize, SampledNodeTrs>,
    node_name: &str,
    start_index: usize,
    end_index: usize,
    rotation_axis: usize,
    u: usize,
    v: usize,
) -> f32 {
    const PROBE_DEGREES: f32 = 1.0;
    let matrices = global_node_matrices_with_sampled(mesh, overrides, sampled);
    let base_angle = segment_angle(
        matrix_translation(matrices[start_index]),
        matrix_translation(matrices[end_index]),
        u,
        v,
    );
    let mut probed = overrides.clone();
    probed
        .entry(node_name.to_string())
        .or_insert_with(BoneOverride::identity)
        .rotation_deg[rotation_axis] += PROBE_DEGREES;
    let matrices = global_node_matrices_with_sampled(mesh, &probed, sampled);
    let probed_angle = segment_angle(
        matrix_translation(matrices[start_index]),
        matrix_translation(matrices[end_index]),
        u,
        v,
    );
    normalize_angle(probed_angle - base_angle).to_degrees() / PROBE_DEGREES
}

fn actor_model_point_to_world(
    point: [f32; 3],
    mesh: &GlbMeshData,
    pose: ActorFramePose,
) -> [f32; 3] {
    let height = (mesh.bounds_max[1] - mesh.bounds_min[1]).abs().max(0.001);
    let center_x = (mesh.bounds_min[0] + mesh.bounds_max[0]) * 0.5;
    let center_z = (mesh.bounds_min[2] + mesh.bounds_max[2]) * 0.5;
    let local = [
        (point[0] - center_x) * pose.scale / height,
        (point[1] - mesh.bounds_min[1]) * pose.scale / height,
        (point[2] - center_z) * pose.scale / height,
    ];
    let rotated = rotate_actor_vector(local, pose.rotation_deg);
    std::array::from_fn(|axis| pose.position[axis] + rotated[axis])
}

fn actor_world_point_to_model(
    point: [f32; 3],
    mesh: &GlbMeshData,
    pose: ActorFramePose,
) -> [f32; 3] {
    let world = std::array::from_fn(|axis| point[axis] - pose.position[axis]);
    let local = inverse_rotate_actor_vector(world, pose.rotation_deg);
    let height = (mesh.bounds_max[1] - mesh.bounds_min[1]).abs().max(0.001);
    let center_x = (mesh.bounds_min[0] + mesh.bounds_max[0]) * 0.5;
    let center_z = (mesh.bounds_min[2] + mesh.bounds_max[2]) * 0.5;
    [
        local[0] * height / pose.scale + center_x,
        local[1] * height / pose.scale + mesh.bounds_min[1],
        local[2] * height / pose.scale + center_z,
    ]
}

fn rotate_actor_vector(mut value: [f32; 3], rotation_deg: [f32; 3]) -> [f32; 3] {
    let [pitch, yaw, roll] = rotation_deg.map(f32::to_radians);
    value = rotate_y(value, yaw);
    value = rotate_x(value, pitch);
    rotate_z(value, roll)
}

fn inverse_rotate_actor_vector(mut value: [f32; 3], rotation_deg: [f32; 3]) -> [f32; 3] {
    let [pitch, yaw, roll] = rotation_deg.map(f32::to_radians);
    value = rotate_z(value, -roll);
    value = rotate_x(value, -pitch);
    rotate_y(value, -yaw)
}

fn rotate_x(value: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin, cos) = angle.sin_cos();
    [
        value[0],
        value[1] * cos - value[2] * sin,
        value[1] * sin + value[2] * cos,
    ]
}

fn rotate_y(value: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin, cos) = angle.sin_cos();
    [
        value[0] * cos + value[2] * sin,
        value[1],
        -value[0] * sin + value[2] * cos,
    ]
}

fn rotate_z(value: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin, cos) = angle.sin_cos();
    [
        value[0] * cos - value[1] * sin,
        value[0] * sin + value[1] * cos,
        value[2],
    ]
}

fn skinned_actor_positions(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
    time: WorldTime,
) -> Result<Option<Vec<[f32; 3]>>, WorldRenderError> {
    if let Some(native) = actor.native_skin.as_ref() {
        if mesh.joints.len() != mesh.positions.len() || mesh.weights.len() != mesh.positions.len() {
            return Ok(None);
        }
        return Ok(Some(apply_joint_matrices_to_positions(
            mesh,
            &native.joint_matrices,
        )));
    }
    let Some(skin) = mesh.skin.as_ref() else {
        return Ok(None);
    };
    if mesh.nodes.is_empty()
        || mesh.joints.len() != mesh.positions.len()
        || mesh.weights.len() != mesh.positions.len()
    {
        return Ok(None);
    }

    let overrides = actor_bone_overrides_for_mesh(graph, actor, Some(mesh), time)?;
    let has_clip = actor_has_clip_layers(actor) && !mesh.animations.is_empty();
    if overrides.is_empty() && !has_clip {
        return Ok(None);
    }
    if !has_clip && !overrides_match_nodes(mesh, &overrides) {
        return Ok(None);
    }

    let global_matrices = actor_global_node_matrices(graph, actor, mesh, time, &overrides, None)?;
    let joint_matrices = skin
        .joints
        .iter()
        .map(|joint| {
            let global = global_matrices
                .get(joint.node_index)
                .copied()
                .unwrap_or_else(mat4_identity);
            mat4_mul(global, joint.inverse_bind_matrix)
        })
        .collect::<Vec<_>>();

    Ok(Some(apply_joint_matrices_to_positions(
        mesh,
        &joint_matrices,
    )))
}

fn apply_joint_matrices_to_positions(
    mesh: &GlbMeshData,
    joint_matrices: &[[f32; 16]],
) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(mesh.positions.len());
    for (idx, position) in mesh.positions.iter().copied().enumerate() {
        let Some(joints) = mesh.joints.get(idx).copied().flatten() else {
            out.push(position);
            continue;
        };
        let Some(weights) = mesh.weights.get(idx).copied().flatten() else {
            out.push(position);
            continue;
        };
        let weight_sum = weights.iter().copied().sum::<f32>();
        if weight_sum <= f32::EPSILON {
            out.push(position);
            continue;
        }

        let mut skinned = [0.0f32; 3];
        for slot in 0..4 {
            let joint_index = joints[slot] as usize;
            let Some(matrix) = joint_matrices.get(joint_index).copied() else {
                continue;
            };
            let weight = weights[slot] / weight_sum;
            if weight <= 0.0 {
                continue;
            }
            let transformed = mat4_transform_point(matrix, position);
            for axis in 0..3 {
                skinned[axis] += transformed[axis] * weight;
            }
        }
        out.push(skinned);
    }
    out
}

struct ActorJointFrame {
    global_matrices: Option<Vec<[f32; 16]>>,
    native_matrices: Option<Vec<[f32; 16]>>,
}

fn prepare_actor_joint_frame(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
    time: WorldTime,
    external_sampled: &HashMap<usize, SampledNodeTrs>,
    constraint_overrides: &HashMap<String, BoneOverride>,
) -> Result<ActorJointFrame, WorldRenderError> {
    if let Some(native) = actor.native_skin.as_ref() {
        return Ok(ActorJointFrame {
            global_matrices: None,
            native_matrices: Some(native.joint_matrices.clone()),
        });
    }
    let Some(skin) = mesh.skin.as_ref() else {
        return Ok(ActorJointFrame {
            global_matrices: None,
            native_matrices: None,
        });
    };
    if mesh.nodes.is_empty()
        || mesh.joints.len() != mesh.positions.len()
        || mesh.weights.len() != mesh.positions.len()
    {
        return Ok(ActorJointFrame {
            global_matrices: None,
            native_matrices: None,
        });
    }

    let mut overrides = actor_bone_overrides_for_mesh(graph, actor, Some(mesh), time)?;
    for (bone, value) in constraint_overrides {
        overrides.insert(bone.clone(), *value);
    }
    let has_clip = (actor_has_clip_layers(actor) && !mesh.animations.is_empty())
        || !external_sampled.is_empty();
    if !has_clip && (overrides.is_empty() || !overrides_match_nodes(mesh, &overrides)) {
        let _ = skin;
        return Ok(ActorJointFrame {
            global_matrices: None,
            native_matrices: None,
        });
    }
    let global_matrices =
        actor_global_node_matrices(graph, actor, mesh, time, &overrides, Some(external_sampled))?;
    Ok(ActorJointFrame {
        global_matrices: Some(global_matrices),
        native_matrices: None,
    })
}

fn actor_joint_matrices(
    mesh: &GlbMeshData,
    model_path: &Path,
    mesh_node: Option<usize>,
    frame: &ActorJointFrame,
    skinning_strategy_cache: &mut HashMap<SkinningStrategyKey, SkinningMatrixStrategy>,
) -> Vec<[f32; 16]> {
    if let Some(matrices) = frame.native_matrices.as_ref() {
        return if matrices.is_empty() {
            vec![mat4_identity()]
        } else {
            matrices.clone()
        };
    }
    let Some(skin) = mesh.skin.as_ref() else {
        return vec![mat4_identity()];
    };
    let Some(global_matrices) = frame.global_matrices.as_deref() else {
        return vec![mat4_identity(); skin.joints.len().max(1)];
    };
    let cache_key = SkinningStrategyKey {
        model_path: model_path.to_path_buf(),
        mesh_node,
    };
    let strategy = if let Some(strategy) = skinning_strategy_cache.get(&cache_key).copied() {
        strategy
    } else {
        let candidates = skinning_matrix_candidates(mesh, skin, mesh_node, global_matrices);
        let strategy =
            choose_skinning_matrix_candidate(mesh, mesh_node, candidates).unwrap_or_default();
        skinning_strategy_cache.insert(cache_key, strategy);
        strategy
    };
    let mut joint_matrices =
        matrices_for_skinning_strategy(mesh, skin, mesh_node, global_matrices, strategy);
    if joint_matrices.is_empty() {
        joint_matrices.push(mat4_identity());
    }
    joint_matrices
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SkinningStrategyKey {
    model_path: PathBuf,
    mesh_node: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkinningMatrixStrategy {
    BindSpace,
    MeshLocal,
    SkeletonLocal,
    MeshParentLocal,
}

impl Default for SkinningMatrixStrategy {
    fn default() -> Self {
        Self::MeshLocal
    }
}

struct SkinningMatrixCandidate {
    strategy: SkinningMatrixStrategy,
    matrices: Vec<[f32; 16]>,
}

fn skinning_matrix_candidates(
    mesh: &GlbMeshData,
    skin: &crate::world::gltf_loader::GlbSkinData,
    mesh_node: Option<usize>,
    global_matrices: &[[f32; 16]],
) -> Vec<SkinningMatrixCandidate> {
    let mut candidates = vec![
        SkinningMatrixCandidate {
            strategy: SkinningMatrixStrategy::BindSpace,
            matrices: matrices_for_skinning_strategy(
                mesh,
                skin,
                mesh_node,
                global_matrices,
                SkinningMatrixStrategy::BindSpace,
            ),
        },
        SkinningMatrixCandidate {
            strategy: SkinningMatrixStrategy::MeshLocal,
            matrices: matrices_for_skinning_strategy(
                mesh,
                skin,
                mesh_node,
                global_matrices,
                SkinningMatrixStrategy::MeshLocal,
            ),
        },
    ];

    if let Some(skeleton) = skin.skeleton {
        if let Some(skeleton_inverse) = global_matrices
            .get(skeleton)
            .copied()
            .and_then(mat4_inverse_affine)
        {
            candidates.push(SkinningMatrixCandidate {
                strategy: SkinningMatrixStrategy::SkeletonLocal,
                matrices: matrices_for_skinning_strategy(
                    mesh,
                    skin,
                    mesh_node,
                    global_matrices,
                    SkinningMatrixStrategy::SkeletonLocal,
                ),
            });
            let _ = skeleton_inverse;
        }
    }

    if let Some(parent_inverse) = mesh_node
        .and_then(|node_index| mesh.nodes.get(node_index).and_then(|node| node.parent))
        .and_then(|parent| global_matrices.get(parent).copied())
        .and_then(mat4_inverse_affine)
    {
        candidates.push(SkinningMatrixCandidate {
            strategy: SkinningMatrixStrategy::MeshParentLocal,
            matrices: matrices_for_skinning_strategy(
                mesh,
                skin,
                mesh_node,
                global_matrices,
                SkinningMatrixStrategy::MeshParentLocal,
            ),
        });
        let _ = parent_inverse;
    }

    candidates
}

fn matrices_for_skinning_strategy(
    mesh: &GlbMeshData,
    skin: &crate::world::gltf_loader::GlbSkinData,
    mesh_node: Option<usize>,
    global_matrices: &[[f32; 16]],
    strategy: SkinningMatrixStrategy,
) -> Vec<[f32; 16]> {
    let space_inverse = match strategy {
        SkinningMatrixStrategy::BindSpace => mat4_identity(),
        SkinningMatrixStrategy::MeshLocal => mesh_node
            .and_then(|node_index| global_matrices.get(node_index).copied())
            .and_then(mat4_inverse_affine)
            .unwrap_or_else(mat4_identity),
        SkinningMatrixStrategy::SkeletonLocal => skin
            .skeleton
            .and_then(|node_index| global_matrices.get(node_index).copied())
            .and_then(mat4_inverse_affine)
            .unwrap_or_else(mat4_identity),
        SkinningMatrixStrategy::MeshParentLocal => mesh_node
            .and_then(|node_index| mesh.nodes.get(node_index).and_then(|node| node.parent))
            .and_then(|parent| global_matrices.get(parent).copied())
            .and_then(mat4_inverse_affine)
            .unwrap_or_else(mat4_identity),
    };

    skin.joints
        .iter()
        .map(|joint| {
            let global = global_matrices
                .get(joint.node_index)
                .copied()
                .unwrap_or_else(mat4_identity);
            let bind = mat4_mul(global, joint.inverse_bind_matrix);
            match strategy {
                SkinningMatrixStrategy::BindSpace => bind,
                _ => mat4_mul(space_inverse, bind),
            }
        })
        .collect()
}

fn choose_skinning_matrix_candidate(
    mesh: &GlbMeshData,
    mesh_node: Option<usize>,
    candidates: Vec<SkinningMatrixCandidate>,
) -> Option<SkinningMatrixStrategy> {
    if candidates.is_empty() {
        return None;
    }
    let sample_indices = skinning_strategy_sample_indices(mesh, mesh_node, 4096);
    if sample_indices.is_empty() {
        return candidates
            .into_iter()
            .next()
            .map(|candidate| candidate.strategy);
    }
    let (raw_min, raw_max) = bounds_for_indices(mesh, &sample_indices, None);
    let raw_extent = bounds_extent(raw_min, raw_max);
    let raw_center = bounds_center(raw_min, raw_max);

    candidates
        .into_iter()
        .map(|candidate| {
            let (min, max) = bounds_for_indices(mesh, &sample_indices, Some(&candidate.matrices));
            let extent = bounds_extent(min, max);
            let center = bounds_center(min, max);
            let score = skinning_strategy_score(raw_extent, raw_center, extent, center);
            (score, candidate)
        })
        .min_by(|(a, a_candidate), (b, b_candidate)| {
            a.total_cmp(b).then_with(|| {
                // Prefer the standard glTF mesh-local path when candidates are effectively tied.
                skinning_strategy_rank(a_candidate.strategy)
                    .cmp(&skinning_strategy_rank(b_candidate.strategy))
            })
        })
        .map(|(_, candidate)| candidate.strategy)
}

fn skinning_strategy_rank(strategy: SkinningMatrixStrategy) -> u8 {
    match strategy {
        SkinningMatrixStrategy::MeshLocal => 0,
        SkinningMatrixStrategy::MeshParentLocal => 1,
        SkinningMatrixStrategy::SkeletonLocal => 2,
        SkinningMatrixStrategy::BindSpace => 3,
    }
}

fn skinning_strategy_sample_indices(
    mesh: &GlbMeshData,
    mesh_node: Option<usize>,
    max_samples: usize,
) -> Vec<usize> {
    let mut seen = vec![false; mesh.positions.len()];
    let mut indices = Vec::with_capacity(max_samples.min(mesh.positions.len()));
    for triangle in &mesh.triangles {
        if mesh_node.is_some() && triangle.mesh_node != mesh_node {
            continue;
        }
        for index in triangle.indices {
            let index = index as usize;
            if index >= seen.len() || seen[index] {
                continue;
            }
            seen[index] = true;
            indices.push(index);
            if indices.len() >= max_samples {
                return indices;
            }
        }
    }
    if indices.is_empty() && mesh_node.is_some() {
        return skinning_strategy_sample_indices(mesh, None, max_samples);
    }
    indices
}

fn bounds_for_indices(
    mesh: &GlbMeshData,
    indices: &[usize],
    matrices: Option<&[[f32; 16]]>,
) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for index in indices {
        let Some(position) = mesh.positions.get(*index).copied() else {
            continue;
        };
        let point = matrices
            .and_then(|matrices| {
                let joints = mesh.joints.get(*index).copied().flatten()?;
                let weights = mesh.weights.get(*index).copied().flatten()?;
                Some(skinning_transform_position(
                    position, joints, weights, matrices,
                ))
            })
            .unwrap_or(position);
        if point.iter().all(|value| value.is_finite()) {
            accumulate_bounds3(&mut min, &mut max, point);
        }
    }
    (min, max)
}

fn skinning_transform_position(
    position: [f32; 3],
    joints: [u16; 4],
    weights: [f32; 4],
    matrices: &[[f32; 16]],
) -> [f32; 3] {
    let weight_sum = weights.iter().sum::<f32>();
    if weight_sum <= f32::EPSILON {
        return position;
    }
    let mut out = [0.0f32; 3];
    for slot in 0..4 {
        let weight = weights[slot] / weight_sum;
        if weight <= 0.0 {
            continue;
        }
        let Some(matrix) = matrices.get(joints[slot] as usize).copied() else {
            continue;
        };
        let transformed = mat4_transform_point(matrix, position);
        for axis in 0..3 {
            out[axis] += transformed[axis] * weight;
        }
    }
    out
}

fn bounds_extent(min: [f32; 3], max: [f32; 3]) -> [f32; 3] {
    [
        (max[0] - min[0]).abs().max(0.0001),
        (max[1] - min[1]).abs().max(0.0001),
        (max[2] - min[2]).abs().max(0.0001),
    ]
}

fn bounds_center(min: [f32; 3], max: [f32; 3]) -> [f32; 3] {
    [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ]
}

fn skinning_strategy_score(
    raw_extent: [f32; 3],
    raw_center: [f32; 3],
    extent: [f32; 3],
    center: [f32; 3],
) -> f32 {
    let mut score = 0.0;
    for axis in 0..3 {
        let scale_ratio = (extent[axis] / raw_extent[axis]).max(0.0001);
        score += scale_ratio.ln().abs() * 4.0;
        score += ((center[axis] - raw_center[axis]).abs() / raw_extent[axis].max(0.0001)).min(10.0);
    }
    score
}

fn overrides_match_nodes(mesh: &GlbMeshData, overrides: &HashMap<String, BoneOverride>) -> bool {
    mesh.nodes
        .iter()
        .filter_map(|node| node.name.as_deref())
        .any(|name| overrides.contains_key(name))
}

#[derive(Debug, Clone, Copy)]
struct BoneOverride {
    translation: [f32; 3],
    rotation_deg: [f32; 3],
    scale: f32,
}

impl BoneOverride {
    fn identity() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: 1.0,
        }
    }

    fn add_weighted(&mut self, other: Self, weight: f32) {
        for axis in 0..3 {
            self.translation[axis] += other.translation[axis] * weight;
            self.rotation_deg[axis] += other.rotation_deg[axis] * weight;
        }
        self.scale *= 1.0 + (other.scale - 1.0) * weight;
    }

    fn composed_with(mut self, delta: Self) -> Self {
        for axis in 0..3 {
            self.translation[axis] += delta.translation[axis];
            self.rotation_deg[axis] += delta.rotation_deg[axis];
        }
        self.scale *= delta.scale;
        self
    }

    fn blended_to(self, target: Self, weight: f32) -> Self {
        let weight = weight.clamp(0.0, 1.0);
        Self {
            translation: std::array::from_fn(|axis| {
                self.translation[axis] * (1.0 - weight) + target.translation[axis] * weight
            }),
            rotation_deg: std::array::from_fn(|axis| {
                self.rotation_deg[axis] * (1.0 - weight) + target.rotation_deg[axis] * weight
            }),
            scale: self.scale * (1.0 - weight) + target.scale * weight,
        }
    }

    fn is_identity(self) -> bool {
        self.translation.iter().all(|value| value.abs() <= 0.0001)
            && self.rotation_deg.iter().all(|value| value.abs() <= 0.0001)
            && (self.scale - 1.0).abs() <= 0.0001
    }
}

fn actor_bone_overrides_for_mesh(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: Option<&GlbMeshData>,
    time: WorldTime,
) -> Result<HashMap<String, BoneOverride>, WorldRenderError> {
    let profile = actor_model_profile(graph, actor);
    let axis_map = profile.and_then(|profile| profile.bone_axis_map.as_ref());
    let retarget = actor
        .retarget
        .as_deref()
        .and_then(|id| graph.retargets.iter().find(|retarget| retarget.id == id));
    let profile_retarget = profile.and_then(|profile| profile.retarget.as_ref());
    let retarget_preset = retarget
        .map(|retarget| retarget.preset.as_str())
        .or_else(|| profile_retarget.map(|retarget| retarget.preset.as_str()))
        .or_else(|| profile.map(|profile| profile.preset.as_str()));
    let bone_to_node = if let Some(retarget) = retarget {
        retarget_maps_to_node_lookup(&retarget.maps)
    } else if let Some(retarget) = profile_retarget {
        retarget_maps_to_node_lookup(&retarget.maps)
    } else {
        HashMap::new()
    };

    let mut out = HashMap::<String, BoneOverride>::new();
    if let Some(axis_map) = axis_map {
        for axis in &axis_map.axes {
            // A raw external clip already carries a complete pose relative to
            // its source bind pose. Reapplying the target's semantic rest
            // offset double-corrects calibrated limbs (for example, Reze's
            // relaxed-arm -90 degree offset turned an imported walk forwards).
            if external_action_drives_bone(graph, actor, &axis.bone, time)? {
                continue;
            }
            let transform = rest_pose_axis_transform(axis, axis_map, time)?;
            if transform.is_identity() {
                continue;
            }
            let node_name = bone_to_node
                .get(axis.bone.as_str())
                .copied()
                .unwrap_or(axis.bone.as_str());
            out.entry(node_name.to_string())
                .or_insert_with(BoneOverride::identity)
                .add_weighted(transform, 1.0);
        }
    }
    // Rest-axis calibration belongs to the model adapter, not to an Action.
    // Keep an immutable baseline so an authored canonical pose is layered on
    // top of the target GLB's relaxed rest pose. Replacing the whole override
    // here would erase unmentioned calibration channels (for example, a walk
    // `forward` swing used to discard the shoulder's `restSide` correction
    // and snap a relaxed arm back towards the imported T-pose).
    let rest_overrides = out.clone();

    for apply in graph
        .apply_actions
        .iter()
        .filter(|apply| apply.target == actor.id)
    {
        let elapsed_ms = time.time_sec() * 1000.0 - apply.at_ms as f32;
        if elapsed_ms < 0.0
            || (!apply.r#loop
                && apply
                    .duration_ms
                    .is_some_and(|duration| elapsed_ms > duration as f32))
        {
            continue;
        }
        let Some(action) = graph
            .actions
            .iter()
            .find(|action| action.id == apply.action)
        else {
            continue;
        };
        if let Some(retarget_preset) = retarget_preset {
            if action.skeleton != retarget_preset {
                continue;
            }
        }
        let mut speed = eval_number(&apply.speed, 1.0, time)?.max(0.0);
        if !apply.r#loop
            && let Some(duration_ms) = apply.duration_ms.filter(|duration| *duration > 0)
        {
            speed *= action.duration_ms as f32 / duration_ms as f32;
        }
        let Some(action_time) =
            action_local_time_sec(action, apply.at_ms, apply.r#loop, speed, time)
        else {
            continue;
        };
        let weight = (eval_number(&apply.weight, 1.0, time)?
            * action_blend_envelope(action, apply, action_time, speed, time)?)
        .clamp(0.0, 1.0);
        if weight <= 0.0 {
            continue;
        }

        let legacy_pose = action_pose_transform(action, action_time, time, axis_map)?;
        let mut retargeted_pose = if !apply.mode.eq_ignore_ascii_case("additive")
            && action.skeleton == "humanoid_v1"
            && action_uses_baked_humanoid_reference(action)
            && mesh.is_some()
            && !bone_to_node.is_empty()
        {
            retarget_humanoid_v1_action_pose(
                mesh.expect("humanoid action retarget requires a target mesh"),
                &bone_to_node,
                &rest_overrides,
                action,
                action_time,
                time,
            )?
        } else {
            None
        };
        if let Some(pose) = retargeted_pose.as_mut() {
            // The embedded reference hierarchy covers the portable body
            // chain. Keep target-aware legacy channels for extra mapped
            // joints (notably Mixamo/VRM fingers) so this adapter does not
            // discard animation that lies outside the core humanoid rig.
            for (bone, transform) in &legacy_pose {
                pose.entry(bone.clone()).or_insert(*transform);
            }
        }
        let pose = retargeted_pose.as_ref().unwrap_or(&legacy_pose);

        for (bone_id, mut transform) in pose.iter().map(|(bone, value)| (bone.clone(), *value)) {
            if transform.is_identity() || !bone_matches_body_mask(&bone_id, &apply.mask) {
                continue;
            }
            // `rootMotion="none"` has always been the editor template's
            // default, but canonical Pose Actions previously ignored it and
            // copied Character1's local hips translation into the target GLB.
            // That turns a one-unit forward stride into a one-metre lift or
            // depth jump on rigs whose hips use a different local basis.
            transform.translation = canonical_action_translation(
                &bone_id,
                apply.root_motion.as_deref(),
                transform.translation,
            );
            let node_name = bone_to_node
                .get(bone_id.as_str())
                .copied()
                .unwrap_or(bone_id.as_str());
            if apply.mode.eq_ignore_ascii_case("additive") {
                out.entry(node_name.to_string())
                    .or_insert_with(BoneOverride::identity)
                    .add_weighted(transform, weight);
            } else if retargeted_pose.is_some() {
                // The quaternion adapter returns the complete target-local
                // override, including its calibrated rest pose.
                out.insert(
                    node_name.to_string(),
                    BoneOverride::identity().blended_to(transform, weight),
                );
            } else {
                let rest = rest_overrides
                    .get(node_name)
                    .copied()
                    .unwrap_or_else(BoneOverride::identity);
                let target = rest.composed_with(transform);
                out.insert(node_name.to_string(), rest.blended_to(target, weight));
            }
        }
        if let Some(mesh) = mesh {
            apply_two_bone_ik_overrides(
                mesh,
                action,
                &bone_to_node,
                &apply.mask,
                weight,
                time,
                &mut out,
            )?;
        }
    }
    Ok(out)
}

pub(crate) fn action_uses_baked_humanoid_reference(action: &WorldAction) -> bool {
    // The bundled mocap-derived library is densely sampled and retains raw
    // local Euler channels alongside semantic channels. Small authored
    // Actions (for example Look Around and Kick) are already portable
    // `humanoid_v1` deltas and must keep the legacy semantic-axis path.
    action.poses.len() >= 32
        && action.poses.iter().any(|pose| {
            pose.bones.iter().any(|bone| {
                bone.rotation.is_some()
                    || bone.rotation_x.is_some()
                    || bone.rotation_y.is_some()
                    || bone.rotation_z.is_some()
            })
        })
}

fn canonical_action_translation(
    bone: &str,
    root_motion: Option<&str>,
    translation: [f32; 3],
) -> [f32; 3] {
    if bone == "hips" && root_motion != Some("clip") {
        [0.0; 3]
    } else {
        translation
    }
}

/// Retarget baked `humanoid_v1` Action channels through the model-space pose
/// delta of the Character1 reference rig.  This is deliberately an internal
/// adapter: existing Action DSL and the public `humanoid_v1` contract stay
/// unchanged, while target rigs no longer receive source-local Euler angles.
fn retarget_humanoid_v1_action_pose(
    target_mesh: &GlbMeshData,
    bone_to_node: &HashMap<&str, &str>,
    target_rest_overrides: &HashMap<String, BoneOverride>,
    action: &WorldAction,
    action_time: f32,
    time: WorldTime,
) -> Result<Option<HashMap<String, BoneOverride>>, WorldRenderError> {
    let source_state = humanoid_v1_reference_pose_state(action, action_time, time)?;
    let source_pose = source_state.pose;
    let reference = humanoid_v1_reference_bones();
    let mapped_count = reference
        .iter()
        .filter(|(bone, _, _)| bone_to_node.contains_key(*bone))
        .count();
    // A partial hand or prop mapping is not enough to reconstruct a stable
    // humanoid hierarchy. Preserve the legacy path in that case.
    if mapped_count < 12 || !bone_to_node.contains_key("hips") {
        return Ok(None);
    }

    let source_rest_global = source_state.rest_global;
    let source_animated_global = source_state.animated_global;

    // Use the same calibrated rest pose that the renderer will compose with
    // the returned overrides. This avoids applying arms-down rest offsets a
    // second time after quaternion retargeting.
    let raw_target_rest_matrices = global_node_matrices(target_mesh, &HashMap::new());
    let raw_target_rest_global = raw_target_rest_matrices
        .iter()
        .copied()
        .map(quat_from_mat4_rotation)
        .collect::<Vec<_>>();
    let target_rest_matrices = global_node_matrices(target_mesh, target_rest_overrides);
    let target_rest_global = target_rest_matrices
        .iter()
        .copied()
        .map(quat_from_mat4_rotation)
        .collect::<Vec<_>>();
    let mut mapped = reference
        .iter()
        .filter_map(|(bone, _, _)| {
            let node_name = bone_to_node.get(*bone).copied()?;
            let target_index = node_index_by_name(target_mesh, node_name)?;
            Some((*bone, target_index))
        })
        .collect::<Vec<_>>();
    mapped.sort_by_key(|(_, index)| node_hierarchy_depth(target_mesh, *index));

    let mut desired_target_global = HashMap::<usize, [f32; 4]>::new();
    let mut out = HashMap::<String, BoneOverride>::new();
    for (bone, target_index) in mapped {
        let Some(target_node) = target_mesh.nodes.get(target_index) else {
            continue;
        };
        let Some(source_rest) = source_rest_global.get(bone).copied() else {
            continue;
        };
        let Some(source_animated) = source_animated_global.get(bone).copied() else {
            continue;
        };
        let target_rest = target_rest_global
            .get(target_index)
            .copied()
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let desired_global = model_space_retarget_global(source_rest, source_animated, target_rest);
        let parent_global = target_node
            .parent
            .and_then(|parent| desired_target_global.get(&parent).copied())
            .or_else(|| {
                target_node
                    .parent
                    .and_then(|parent| target_rest_global.get(parent).copied())
            })
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let desired_local = quat_normalize_xyzw(quat_mul_xyzw(
            quat_conjugate_xyzw(parent_global),
            desired_global,
        ));
        let raw_rest = raw_target_rest_global
            .get(target_index)
            .copied()
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let raw_rest_parent = target_node
            .parent
            .and_then(|parent| raw_target_rest_global.get(parent).copied())
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let raw_rest_local = quat_normalize_xyzw(quat_mul_xyzw(
            quat_conjugate_xyzw(raw_rest_parent),
            raw_rest,
        ));
        let complete_override = quat_normalize_xyzw(quat_mul_xyzw(
            quat_conjugate_xyzw(raw_rest_local),
            desired_local,
        ));
        desired_target_global.insert(target_index, desired_global);
        let source_transform = source_pose
            .get(bone)
            .copied()
            .unwrap_or_else(BoneOverride::identity);
        out.insert(
            bone.to_string(),
            BoneOverride {
                translation: source_transform.translation,
                rotation_deg: quat_to_zyx_euler_degrees(complete_override),
                scale: source_transform.scale,
            },
        );
    }
    Ok(Some(out))
}

struct HumanoidReferencePoseState {
    pose: HashMap<String, BoneOverride>,
    rest_global: HashMap<&'static str, [f32; 4]>,
    animated_global: HashMap<&'static str, [f32; 4]>,
}

fn humanoid_v1_reference_pose_state(
    action: &WorldAction,
    action_time: f32,
    time: WorldTime,
) -> Result<HumanoidReferencePoseState, WorldRenderError> {
    let reference_axes = humanoid_v1_reference_axis_map();
    let pose = action_pose_transform(action, action_time, time, Some(&reference_axes))?;
    let mut rest_global = HashMap::<&'static str, [f32; 4]>::new();
    let mut animated_global = HashMap::<&'static str, [f32; 4]>::new();
    for (bone, parent, local_rest) in humanoid_v1_reference_bones() {
        let parent_rest = parent
            .and_then(|id| rest_global.get(id).copied())
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let parent_animated = parent
            .and_then(|id| animated_global.get(id).copied())
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let calibrated_local = quat_normalize_xyzw(quat_mul_xyzw(
            *local_rest,
            quat_from_bone_override(humanoid_v1_reference_rest_override(bone)),
        ));
        let animated_local = quat_normalize_xyzw(quat_mul_xyzw(
            calibrated_local,
            quat_from_bone_override(
                pose.get(*bone)
                    .copied()
                    .unwrap_or_else(BoneOverride::identity),
            ),
        ));
        rest_global.insert(
            bone,
            quat_normalize_xyzw(quat_mul_xyzw(parent_rest, calibrated_local)),
        );
        animated_global.insert(
            bone,
            quat_normalize_xyzw(quat_mul_xyzw(parent_animated, animated_local)),
        );
    }
    Ok(HumanoidReferencePoseState {
        pose,
        rest_global,
        animated_global,
    })
}

pub(crate) fn humanoid_v1_reference_model_rotation_deltas(
    action: &WorldAction,
    action_time: f32,
    time: WorldTime,
) -> Result<HashMap<String, [f32; 4]>, WorldRenderError> {
    let state = humanoid_v1_reference_pose_state(action, action_time, time)?;
    Ok(state
        .rest_global
        .iter()
        .filter_map(|(bone, rest)| {
            let animated = state.animated_global.get(bone)?;
            Some((
                (*bone).to_string(),
                quat_normalize_xyzw(quat_mul_xyzw(*animated, quat_conjugate_xyzw(*rest))),
            ))
        })
        .collect())
}

fn model_space_retarget_global(
    source_rest: [f32; 4],
    source_animated: [f32; 4],
    target_rest: [f32; 4],
) -> [f32; 4] {
    // With global transforms expressed as `parent * local`, the world-space
    // animation delta is `animated * inverse(rest)`. Apply that delta on the
    // left of the target rest. Reversing either multiplication conjugates the
    // motion through the source/target bind axes and can turn a forward run
    // into a visually backward leg cycle.
    let model_delta = quat_mul_xyzw(source_animated, quat_conjugate_xyzw(source_rest));
    quat_normalize_xyzw(quat_mul_xyzw(model_delta, target_rest))
}

fn quat_from_bone_override(transform: BoneOverride) -> [f32; 4] {
    quat_from_mat4_rotation(mat4_from_override(BoneOverride {
        translation: [0.0; 3],
        scale: 1.0,
        ..transform
    }))
}

/// Inverse of the renderer's `Rz * Ry * Rx` override order.
fn quat_to_zyx_euler_degrees(rotation: [f32; 4]) -> [f32; 3] {
    let [x, y, z, w] = quat_normalize_xyzw(rotation);
    let sin_x_cos_y = 2.0 * (w * x + y * z);
    let cos_x_cos_y = 1.0 - 2.0 * (x * x + y * y);
    let rotation_x = sin_x_cos_y.atan2(cos_x_cos_y);
    let sin_y = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0);
    let rotation_y = sin_y.asin();
    let sin_z_cos_y = 2.0 * (w * z + x * y);
    let cos_z_cos_y = 1.0 - 2.0 * (y * y + z * z);
    let rotation_z = sin_z_cos_y.atan2(cos_z_cos_y);
    [rotation_x, rotation_y, rotation_z].map(f32::to_degrees)
}

fn humanoid_v1_reference_rest_override(bone: &str) -> BoneOverride {
    let mut value = BoneOverride::identity();
    match bone {
        "upper_arm_l" => {
            value.rotation_deg[0] = -4.8;
            value.rotation_deg[2] = -89.8;
        }
        "upper_arm_r" => {
            value.rotation_deg[0] = -4.8;
            value.rotation_deg[2] = 89.8;
        }
        _ => {}
    }
    value
}

fn humanoid_v1_reference_axis_map() -> WorldBoneAxisMap {
    let axis = |bone: &str,
                forward: Option<&str>,
                side: Option<&str>,
                twist: Option<&str>,
                bend: Option<&str>,
                turn: Option<&str>| WorldBoneAxis {
        bone: bone.to_string(),
        forward: forward.map(str::to_string),
        side: side.map(str::to_string),
        twist: twist.map(str::to_string),
        bend: bend.map(str::to_string),
        turn: turn.map(str::to_string),
        rest_forward: None,
        rest_side: None,
        rest_twist: None,
        rest_bend: None,
        rest_turn: None,
    };
    WorldBoneAxisMap {
        axes: vec![
            axis("hips", None, None, None, None, Some("rotationY:1")),
            axis(
                "spine",
                None,
                None,
                None,
                Some("rotationX:-1"),
                Some("rotationY:1"),
            ),
            axis(
                "chest",
                None,
                None,
                None,
                Some("rotationX:-1"),
                Some("rotationY:1"),
            ),
            axis(
                "upper_chest",
                None,
                None,
                None,
                Some("rotationX:-1"),
                Some("rotationY:1"),
            ),
            axis(
                "neck",
                None,
                None,
                None,
                Some("rotationX:-1"),
                Some("rotationY:1"),
            ),
            axis(
                "head",
                None,
                None,
                None,
                Some("rotationX:-1"),
                Some("rotationY:1"),
            ),
            axis(
                "upper_arm_l",
                Some("rotationX:1"),
                Some("rotationZ:1"),
                Some("rotationY:1"),
                None,
                None,
            ),
            axis(
                "forearm_l",
                None,
                None,
                Some("rotationY:1"),
                Some("rotationX:1"),
                None,
            ),
            axis("hand_l", None, None, Some("rotationY:1"), None, None),
            axis(
                "upper_arm_r",
                Some("rotationX:1"),
                Some("rotationZ:-1"),
                Some("rotationY:1"),
                None,
                None,
            ),
            axis(
                "forearm_r",
                None,
                None,
                Some("rotationY:1"),
                Some("rotationX:1"),
                None,
            ),
            axis("hand_r", None, None, Some("rotationY:1"), None, None),
            axis(
                "upper_leg_l",
                Some("rotationX:-1"),
                Some("rotationZ:-1"),
                Some("rotationY:1"),
                None,
                None,
            ),
            axis(
                "lower_leg_l",
                None,
                None,
                Some("rotationY:1"),
                Some("rotationX:1"),
                None,
            ),
            axis("foot_l", None, None, None, Some("rotationX:-1"), None),
            axis(
                "upper_leg_r",
                Some("rotationX:-1"),
                Some("rotationZ:1"),
                Some("rotationY:1"),
                None,
                None,
            ),
            axis(
                "lower_leg_r",
                None,
                None,
                Some("rotationY:1"),
                Some("rotationX:1"),
                None,
            ),
            axis("foot_r", None, None, None, Some("rotationX:-1"), None),
        ],
    }
}

/// Character1 is the authored reference pose for the bundled Action Library.
/// Values are canonical-parent-relative bind rotations, extracted once from
/// the public reference GLB so runtime retargeting does not fetch an asset.
fn humanoid_v1_reference_bones() -> &'static [(&'static str, Option<&'static str>, [f32; 4])] {
    &[
        ("hips", None, [0.1258408, 0.0, 0.0, 0.9920505]),
        ("spine", Some("hips"), [-0.06470263, 0.0, 0.0, 0.99790466]),
        ("chest", Some("spine"), [-0.07727985, 0.0, 0.0, 0.9970095]),
        ("upper_chest", Some("chest"), [-0.00026859, 0.0, 0.0, 1.0]),
        (
            "neck",
            Some("upper_chest"),
            [0.11098593, 0.0, 0.0, 0.99382216],
        ),
        ("head", Some("neck"), [-0.07867424, 0.0, 0.0, 0.99690056]),
        (
            "shoulder_l",
            Some("upper_chest"),
            [-0.6040207, -0.3451031, -0.35671774, 0.6235509],
        ),
        (
            "upper_arm_l",
            Some("shoulder_l"),
            [0.18026963, 0.6838502, -0.17983645, 0.6837479],
        ),
        (
            "forearm_l",
            Some("upper_arm_l"),
            [0.0171823, -0.00002035, 0.00000037, 0.9998527],
        ),
        (
            "hand_l",
            Some("forearm_l"),
            [-0.00861969, -0.00000042, 0.0, 0.9999633],
        ),
        (
            "shoulder_r",
            Some("upper_chest"),
            [-0.6040207, 0.3451031, 0.35671774, 0.6235509],
        ),
        (
            "upper_arm_r",
            Some("shoulder_r"),
            [0.18026963, -0.6838502, 0.17983645, 0.6837479],
        ),
        (
            "forearm_r",
            Some("upper_arm_r"),
            [0.01718231, 0.00002035, -0.00000037, 0.9998527],
        ),
        (
            "hand_r",
            Some("forearm_r"),
            [-0.00861969, 0.00000054, 0.0, 0.9999633],
        ),
        (
            "upper_leg_l",
            Some("hips"),
            [0.9924845, 0.0, 0.0, 0.12237063],
        ),
        (
            "lower_leg_l",
            Some("upper_leg_l"),
            [0.03658591, -0.00013117, -0.0000048, 0.9993305],
        ),
        (
            "foot_l",
            Some("lower_leg_l"),
            [-0.5290718, -0.00032809, 0.00034345, 0.8485771],
        ),
        (
            "toe_l",
            Some("foot_l"),
            [0.00013713, -0.9643069, 0.26478714, 0.00049943],
        ),
        (
            "upper_leg_r",
            Some("hips"),
            [0.9924845, 0.0, 0.0, 0.12237063],
        ),
        (
            "lower_leg_r",
            Some("upper_leg_r"),
            [0.03658591, -0.00013117, -0.0000048, 0.9993305],
        ),
        (
            "foot_r",
            Some("lower_leg_r"),
            [-0.5290718, -0.00032809, 0.00034345, 0.8485771],
        ),
        (
            "toe_r",
            Some("foot_r"),
            [0.00013713, -0.9643069, 0.26478714, 0.00049943],
        ),
    ]
}

fn external_action_drives_bone(
    graph: &WorldGraph,
    actor: &WorldActor,
    canonical_bone: &str,
    time: WorldTime,
) -> Result<bool, WorldRenderError> {
    for apply in graph
        .apply_actions
        .iter()
        .filter(|apply| apply.target == actor.id)
    {
        if !graph
            .animation_assets
            .iter()
            .any(|asset| asset.id == apply.action)
        {
            continue;
        }
        let elapsed_ms = time.time_sec() * 1000.0 - apply.at_ms as f32;
        if elapsed_ms < 0.0
            || (!apply.r#loop
                && apply
                    .duration_ms
                    .is_some_and(|duration| elapsed_ms > duration as f32))
        {
            continue;
        }
        if eval_number(&apply.weight, 1.0, time)? <= f32::EPSILON {
            continue;
        }
        if bone_matches_body_mask(canonical_bone, &apply.mask) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn apply_two_bone_ik_overrides(
    mesh: &GlbMeshData,
    action: &WorldAction,
    bone_to_node: &HashMap<&str, &str>,
    mask: &[String],
    action_weight: f32,
    time: WorldTime,
    overrides: &mut HashMap<String, BoneOverride>,
) -> Result<(), WorldRenderError> {
    for ik in &action.iks {
        if !bone_matches_body_mask(&ik.root, mask) && !bone_matches_body_mask(&ik.mid, mask) {
            continue;
        }
        let root_name = bone_to_node
            .get(ik.root.as_str())
            .copied()
            .unwrap_or(ik.root.as_str());
        let mid_name = bone_to_node
            .get(ik.mid.as_str())
            .copied()
            .unwrap_or(ik.mid.as_str());
        let end_name = bone_to_node
            .get(ik.end.as_str())
            .copied()
            .unwrap_or(ik.end.as_str());
        let Some(root_index) = node_index_by_name(mesh, root_name) else {
            continue;
        };
        let Some(mid_index) = node_index_by_name(mesh, mid_name) else {
            continue;
        };
        let Some(end_index) = node_index_by_name(mesh, end_name) else {
            continue;
        };

        let matrices = global_node_matrices(mesh, overrides);
        let root = matrix_translation(matrices[root_index]);
        let mid = matrix_translation(matrices[mid_index]);
        let end = matrix_translation(matrices[end_index]);
        let target = [
            eval_number(&ik.target_x, end[0], time)?,
            eval_number(&ik.target_y, end[1], time)?,
            eval_number(&ik.target_z, end[2], time)?,
        ];
        let ik_weight = (eval_number(&ik.weight, 1.0, time)? * action_weight).clamp(0.0, 1.0);
        if ik_weight <= f32::EPSILON {
            continue;
        }
        let (u, v, rotation_axis) = match ik.plane.to_ascii_lowercase().as_str() {
            "xz" => (0usize, 2usize, 1usize),
            "yz" => (1usize, 2usize, 0usize),
            _ => (0usize, 1usize, 2usize),
        };
        let length_a = distance_2d(root, mid, u, v).max(0.0001);
        let length_b = distance_2d(mid, end, u, v).max(0.0001);
        let target_delta = [target[u] - root[u], target[v] - root[v]];
        let target_distance = (target_delta[0] * target_delta[0]
            + target_delta[1] * target_delta[1])
            .sqrt()
            .clamp(
                (length_a - length_b).abs() + 0.0001,
                length_a + length_b - 0.0001,
            );
        let target_angle = target_delta[1].atan2(target_delta[0]);
        let mut bend_sign = eval_number(&ik.bend, 1.0, time)?.signum();
        if bend_sign == 0.0 {
            bend_sign = 1.0;
        }
        if let (Some(px), Some(py)) = (&ik.pole_x, &ik.pole_y) {
            let pole = [
                eval_number(px, mid[0], time)?,
                eval_number(py, mid[1], time)?,
                ik.pole_z
                    .as_ref()
                    .map(|value| eval_number(value, mid[2], time))
                    .transpose()?
                    .unwrap_or(mid[2]),
            ];
            let cross =
                target_delta[0] * (pole[v] - root[v]) - target_delta[1] * (pole[u] - root[u]);
            if cross.abs() > 0.0001 {
                bend_sign = cross.signum();
            }
        }
        let root_offset = ((length_a * length_a + target_distance * target_distance
            - length_b * length_b)
            / (2.0 * length_a * target_distance))
            .clamp(-1.0, 1.0)
            .acos();
        let desired_root_angle = target_angle + bend_sign * root_offset;
        let desired_mid = [
            root[u] + desired_root_angle.cos() * length_a,
            root[v] + desired_root_angle.sin() * length_a,
        ];
        let desired_second_angle = (target[v] - desired_mid[1]).atan2(target[u] - desired_mid[0]);
        let current_root_angle = segment_angle(root, mid, u, v);
        let root_response = local_axis_plane_response(
            mesh,
            overrides,
            root_name,
            root_index,
            mid_index,
            rotation_axis,
            u,
            v,
        );
        let root_delta = normalize_angle(desired_root_angle - current_root_angle).to_degrees()
            / stable_axis_response(root_response);

        // The elbow is solved after applying the full root correction. Measuring
        // both local-axis responses from the actual node matrices handles mirrored
        // limbs and rigs whose local Z axis points opposite to the solve plane.
        let mut root_solved_overrides = overrides.clone();
        root_solved_overrides
            .entry(root_name.to_string())
            .or_insert_with(BoneOverride::identity)
            .rotation_deg[rotation_axis] += root_delta;
        let root_solved_matrices = global_node_matrices(mesh, &root_solved_overrides);
        let solved_mid = matrix_translation(root_solved_matrices[mid_index]);
        let solved_end = matrix_translation(root_solved_matrices[end_index]);
        let current_second_angle = segment_angle(solved_mid, solved_end, u, v);
        let mid_response = local_axis_plane_response(
            mesh,
            &root_solved_overrides,
            mid_name,
            mid_index,
            end_index,
            rotation_axis,
            u,
            v,
        );
        let mid_delta = normalize_angle(desired_second_angle - current_second_angle).to_degrees()
            / stable_axis_response(mid_response);

        overrides
            .entry(root_name.to_string())
            .or_insert_with(BoneOverride::identity)
            .rotation_deg[rotation_axis] += root_delta * ik_weight;
        overrides
            .entry(mid_name.to_string())
            .or_insert_with(BoneOverride::identity)
            .rotation_deg[rotation_axis] += mid_delta * ik_weight;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn local_axis_plane_response(
    mesh: &GlbMeshData,
    overrides: &HashMap<String, BoneOverride>,
    node_name: &str,
    start_index: usize,
    end_index: usize,
    rotation_axis: usize,
    u: usize,
    v: usize,
) -> f32 {
    const PROBE_DEGREES: f32 = 1.0;
    let matrices = global_node_matrices(mesh, overrides);
    let start = matrix_translation(matrices[start_index]);
    let end = matrix_translation(matrices[end_index]);
    let base_angle = segment_angle(start, end, u, v);

    let mut probed = overrides.clone();
    probed
        .entry(node_name.to_string())
        .or_insert_with(BoneOverride::identity)
        .rotation_deg[rotation_axis] += PROBE_DEGREES;
    let probed_matrices = global_node_matrices(mesh, &probed);
    let probed_start = matrix_translation(probed_matrices[start_index]);
    let probed_end = matrix_translation(probed_matrices[end_index]);
    normalize_angle(segment_angle(probed_start, probed_end, u, v) - base_angle).to_degrees()
        / PROBE_DEGREES
}

fn stable_axis_response(response: f32) -> f32 {
    if response.abs() < 0.05 { 1.0 } else { response }
}

fn segment_angle(start: [f32; 3], end: [f32; 3], u: usize, v: usize) -> f32 {
    (end[v] - start[v]).atan2(end[u] - start[u])
}

fn node_index_by_name(mesh: &GlbMeshData, name: &str) -> Option<usize> {
    mesh.nodes
        .iter()
        .find(|node| node.name.as_deref() == Some(name))
        .map(|node| node.index)
}

fn matrix_translation(matrix: [f32; 16]) -> [f32; 3] {
    [matrix[12], matrix[13], matrix[14]]
}

fn distance_2d(a: [f32; 3], b: [f32; 3], u: usize, v: usize) -> f32 {
    let du = b[u] - a[u];
    let dv = b[v] - a[v];
    (du * du + dv * dv).sqrt()
}

fn normalize_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

fn actor_model_profile<'a>(
    graph: &'a WorldGraph,
    actor: &WorldActor,
) -> Option<&'a WorldModelProfile> {
    actor
        .profile
        .as_deref()
        .and_then(|id| graph.model_profiles.iter().find(|profile| profile.id == id))
}

fn retarget_maps_to_node_lookup(maps: &[WorldRetargetMap]) -> HashMap<&str, &str> {
    maps.iter()
        .map(|map| (map.to.as_str(), map.from.as_str()))
        .collect::<HashMap<_, _>>()
}

fn rest_pose_axis_transform(
    axis: &WorldBoneAxis,
    axis_map: &WorldBoneAxisMap,
    time: WorldTime,
) -> Result<BoneOverride, WorldRenderError> {
    let bone = WorldActionBone {
        id: axis.bone.clone(),
        x: None,
        y: None,
        z: None,
        rotation: None,
        rotation_x: None,
        rotation_y: None,
        rotation_z: None,
        forward: axis.rest_forward.clone(),
        side: axis.rest_side.clone(),
        twist: axis.rest_twist.clone(),
        bend: axis.rest_bend.clone(),
        turn: axis.rest_turn.clone(),
        scale: None,
        opacity: None,
        interpolation: None,
        in_tangent: None,
        out_tangent: None,
    };
    interpolate_bone(Some(&bone), Some(&bone), 0.0, time, Some(axis_map))
}

fn action_local_time_sec(
    action: &WorldAction,
    at_ms: u64,
    should_loop: bool,
    speed: f32,
    time: WorldTime,
) -> Option<f32> {
    let duration_sec = action.duration_ms as f32 / 1000.0;
    if duration_sec <= f32::EPSILON {
        return Some(0.0);
    }
    let local = (time.time_sec() - at_ms as f32 / 1000.0) * speed;
    if local < 0.0 {
        return None;
    }
    if should_loop {
        Some(local % duration_sec)
    } else {
        Some(local.min(duration_sec))
    }
}

/// Return canonical Action phase using the exact timing rules used for bone
/// sampling. Contact correction consumes this value independently of GLB data.
fn authored_action_phase(
    action: &WorldAction,
    apply: &WorldApplyAction,
    time: WorldTime,
) -> Result<Option<f32>, WorldRenderError> {
    let timeline_elapsed_ms = time.time_sec() * 1000.0 - apply.at_ms as f32;
    if timeline_elapsed_ms < 0.0
        || (!apply.r#loop
            && apply
                .duration_ms
                .is_some_and(|duration| timeline_elapsed_ms > duration as f32))
    {
        return Ok(None);
    }
    let duration_sec = action.duration_ms as f32 / 1000.0;
    if duration_sec <= f32::EPSILON {
        return Ok(Some(0.0));
    }
    let mut speed = eval_number(&apply.speed, 1.0, time)?.max(0.0);
    if !apply.r#loop
        && let Some(duration_ms) = apply.duration_ms.filter(|duration| *duration > 0)
    {
        speed *= action.duration_ms as f32 / duration_ms as f32;
    }
    Ok(
        action_local_time_sec(action, apply.at_ms, apply.r#loop, speed, time)
            .map(|local| (local / duration_sec).clamp(0.0, 1.0)),
    )
}

fn action_blend_envelope(
    action: &WorldAction,
    apply: &WorldApplyAction,
    _action_time: f32,
    _speed: f32,
    time: WorldTime,
) -> Result<f32, WorldRenderError> {
    let blend_in = eval_number(&apply.blend_in, 0.0, time)?.max(0.0);
    let blend_out = eval_number(&apply.blend_out, 0.0, time)?.max(0.0);
    let elapsed = (time.time_sec() - apply.at_ms as f32 / 1000.0).max(0.0);
    let fade_in = if blend_in <= f32::EPSILON {
        1.0
    } else {
        (elapsed / blend_in).clamp(0.0, 1.0)
    };
    // A looping clip crosses its internal seam many times, but ApplyAction
    // blending belongs to the authored timeline window rather than each loop.
    let timeline_duration = apply
        .duration_ms
        .map(|duration| duration as f32 / 1000.0)
        .or_else(|| (!apply.r#loop).then_some(action.duration_ms as f32 / 1000.0));
    let fade_out = if blend_out <= f32::EPSILON {
        1.0
    } else if let Some(duration) = timeline_duration.filter(|duration| *duration > f32::EPSILON) {
        ((duration - elapsed) / blend_out).clamp(0.0, 1.0)
    } else {
        1.0
    };
    Ok(fade_in.min(fade_out))
}

fn bone_matches_body_mask(bone: &str, masks: &[String]) -> bool {
    if masks.is_empty() {
        return true;
    }
    let normalized = bone.to_ascii_lowercase().replace([' ', '-', '.'], "_");
    masks.iter().any(|mask| {
        let mask = mask.to_ascii_lowercase().replace([' ', '-', '.'], "_");
        if mask == normalized || mask == "all" || mask == "full_body" {
            return true;
        }
        match mask.as_str() {
            "upper_body" => [
                "hips", "spine", "chest", "neck", "head", "shoulder", "arm", "forearm", "hand",
                "wrist", "finger", "thumb", "index", "middle", "ring", "pinky",
            ]
            .iter()
            .any(|part| normalized.contains(part)),
            "lower_body" => [
                "hips", "pelvis", "leg", "thigh", "knee", "calf", "ankle", "foot", "toe",
            ]
            .iter()
            .any(|part| normalized.contains(part)),
            "left_arm" => {
                (normalized.contains("left") || normalized.ends_with("_l"))
                    && [
                        "shoulder", "arm", "elbow", "forearm", "wrist", "hand", "finger", "thumb",
                        "index", "middle", "ring", "pinky",
                    ]
                    .iter()
                    .any(|part| normalized.contains(part))
            }
            "right_arm" => {
                (normalized.contains("right") || normalized.ends_with("_r"))
                    && [
                        "shoulder", "arm", "elbow", "forearm", "wrist", "hand", "finger", "thumb",
                        "index", "middle", "ring", "pinky",
                    ]
                    .iter()
                    .any(|part| normalized.contains(part))
            }
            "left_leg" => {
                (normalized.contains("left") || normalized.ends_with("_l"))
                    && ["leg", "thigh", "knee", "calf", "ankle", "foot", "toe"]
                        .iter()
                        .any(|part| normalized.contains(part))
            }
            "right_leg" => {
                (normalized.contains("right") || normalized.ends_with("_r"))
                    && ["leg", "thigh", "knee", "calf", "ankle", "foot", "toe"]
                        .iter()
                        .any(|part| normalized.contains(part))
            }
            _ => normalized.starts_with(&mask),
        }
    })
}

fn action_pose_transform(
    action: &WorldAction,
    action_time_sec: f32,
    time: WorldTime,
    axis_map: Option<&WorldBoneAxisMap>,
) -> Result<HashMap<String, BoneOverride>, WorldRenderError> {
    if action.poses.is_empty() {
        return Ok(HashMap::new());
    }
    let (before, after) = action_pose_pair(&action.poses, action_time_sec);
    let linear_alpha = if (after.t - before.t).abs() <= f32::EPSILON {
        0.0
    } else {
        ((action_time_sec - before.t) / (after.t - before.t)).clamp(0.0, 1.0)
    };

    let before_bones = before
        .bones
        .iter()
        .map(|bone| (bone.id.as_str(), bone))
        .collect::<HashMap<_, _>>();
    let after_bones = after
        .bones
        .iter()
        .map(|bone| (bone.id.as_str(), bone))
        .collect::<HashMap<_, _>>();
    let mut ids = before_bones.keys().copied().collect::<Vec<_>>();
    for id in after_bones.keys().copied() {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }

    let mut out = HashMap::new();
    for id in ids {
        let before_bone = before_bones.get(id).copied();
        let after_bone = after_bones.get(id).copied();
        let alpha = world_action_key_mix(before_bone, after_bone, linear_alpha);
        let transform = interpolate_bone(before_bone, after_bone, alpha, time, axis_map)?;
        out.insert(id.to_string(), transform);
    }
    Ok(out)
}

/// Find the authored window without allocating, sorting, or scanning every prior pose.
fn action_pose_pair(
    poses: &[WorldActionPose],
    action_time_sec: f32,
) -> (&WorldActionPose, &WorldActionPose) {
    let first = &poses[0];
    let last = poses.last().expect("poses is not empty");
    if action_time_sec <= first.t {
        return (first, first);
    }
    if action_time_sec >= last.t {
        return (last, last);
    }

    // Strict `<` preserves the legacy exact-key behavior: key t is the end
    // of the preceding interpolation window rather than the next window.
    let after_index = poses.partition_point(|pose| pose.t < action_time_sec);
    (&poses[after_index - 1], &poses[after_index])
}

/// Apply per-key interpolation without changing the legacy linear default.
fn world_action_key_mix(
    before: Option<&WorldActionBone>,
    after: Option<&WorldActionBone>,
    phase: f32,
) -> f32 {
    let phase = phase.clamp(0.0, 1.0);
    match before
        .and_then(|bone| bone.interpolation.as_deref())
        .unwrap_or("linear")
    {
        "hold" => 0.0,
        "ease" => phase * phase * (3.0 - 2.0 * phase),
        "bezier" => {
            let outgoing = before
                .and_then(|bone| bone.out_tangent.as_deref())
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(0.0);
            let incoming = after
                .and_then(|bone| bone.in_tangent.as_deref())
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(0.0);
            let squared = phase * phase;
            let cubed = squared * phase;
            ((-2.0 * cubed + 3.0 * squared)
                + (cubed - 2.0 * squared + phase) * outgoing
                + (cubed - squared) * incoming)
                .clamp(0.0, 1.0)
        }
        _ => phase,
    }
}

fn interpolate_bone(
    before: Option<&WorldActionBone>,
    after: Option<&WorldActionBone>,
    alpha: f32,
    time: WorldTime,
    axis_map: Option<&WorldBoneAxisMap>,
) -> Result<BoneOverride, WorldRenderError> {
    let lerp_field =
        |a: Option<&String>, b: Option<&String>, default: f32| -> Result<f32, WorldRenderError> {
            let av = match a {
                Some(expr) => eval_number(expr, default, time)?,
                None => default,
            };
            let bv = match b {
                Some(expr) => eval_number(expr, av, time)?,
                None => av,
            };
            Ok(av + (bv - av) * alpha)
        };

    let before_rotation_z =
        before.and_then(|bone| bone.rotation_z.as_ref().or(bone.rotation.as_ref()));
    let after_rotation_z =
        after.and_then(|bone| bone.rotation_z.as_ref().or(bone.rotation.as_ref()));
    let mut rotation_deg = [
        lerp_field(
            before.and_then(|bone| bone.rotation_x.as_ref()),
            after.and_then(|bone| bone.rotation_x.as_ref()),
            0.0,
        )?,
        lerp_field(
            before.and_then(|bone| bone.rotation_y.as_ref()),
            after.and_then(|bone| bone.rotation_y.as_ref()),
            0.0,
        )?,
        lerp_field(before_rotation_z, after_rotation_z, 0.0)?,
    ];
    if let Some(bone_id) = before
        .map(|bone| bone.id.as_str())
        .or_else(|| after.map(|bone| bone.id.as_str()))
        && let Some(axis) = bone_axis(axis_map, bone_id)
    {
        apply_semantic_rotation(
            &mut rotation_deg,
            axis.forward.as_deref(),
            lerp_field(
                before.and_then(|bone| bone.forward.as_ref()),
                after.and_then(|bone| bone.forward.as_ref()),
                0.0,
            )?,
        );
        apply_semantic_rotation(
            &mut rotation_deg,
            axis.side.as_deref(),
            lerp_field(
                before.and_then(|bone| bone.side.as_ref()),
                after.and_then(|bone| bone.side.as_ref()),
                0.0,
            )?,
        );
        apply_semantic_rotation(
            &mut rotation_deg,
            axis.twist.as_deref(),
            lerp_field(
                before.and_then(|bone| bone.twist.as_ref()),
                after.and_then(|bone| bone.twist.as_ref()),
                0.0,
            )?,
        );
        apply_semantic_rotation(
            &mut rotation_deg,
            axis.bend.as_deref(),
            lerp_field(
                before.and_then(|bone| bone.bend.as_ref()),
                after.and_then(|bone| bone.bend.as_ref()),
                0.0,
            )?,
        );
        apply_semantic_rotation(
            &mut rotation_deg,
            axis.turn.as_deref(),
            lerp_field(
                before.and_then(|bone| bone.turn.as_ref()),
                after.and_then(|bone| bone.turn.as_ref()),
                0.0,
            )?,
        );
    }
    Ok(BoneOverride {
        translation: [
            lerp_field(
                before.and_then(|bone| bone.x.as_ref()),
                after.and_then(|bone| bone.x.as_ref()),
                0.0,
            )?,
            lerp_field(
                before.and_then(|bone| bone.y.as_ref()),
                after.and_then(|bone| bone.y.as_ref()),
                0.0,
            )?,
            lerp_field(
                before.and_then(|bone| bone.z.as_ref()),
                after.and_then(|bone| bone.z.as_ref()),
                0.0,
            )?,
        ],
        rotation_deg,
        scale: lerp_field(
            before.and_then(|bone| bone.scale.as_ref()),
            after.and_then(|bone| bone.scale.as_ref()),
            1.0,
        )?,
    })
}

fn bone_axis<'a>(
    axis_map: Option<&'a WorldBoneAxisMap>,
    bone_id: &str,
) -> Option<&'a WorldBoneAxis> {
    axis_map.and_then(|axis_map| axis_map.axes.iter().find(|axis| axis.bone == bone_id))
}

fn apply_semantic_rotation(rotation_deg: &mut [f32; 3], binding: Option<&str>, value: f32) {
    if value.abs() <= f32::EPSILON {
        return;
    }
    let Some((axis, scale)) = binding.and_then(parse_axis_binding) else {
        return;
    };
    rotation_deg[axis] += value * scale;
}

fn parse_axis_binding(raw: &str) -> Option<(usize, f32)> {
    let text = raw.trim();
    if text.is_empty() {
        return None;
    }
    let (axis_raw, scale_raw) = text.split_once(':').unwrap_or((text, "1"));
    let mut axis_text = axis_raw.trim();
    let mut sign = 1.0f32;
    if let Some(stripped) = axis_text.strip_prefix('-') {
        sign = -1.0;
        axis_text = stripped.trim();
    } else if let Some(stripped) = axis_text.strip_prefix('+') {
        axis_text = stripped.trim();
    }
    let axis_key = axis_text.to_ascii_lowercase().replace(['_', '-'], "");
    let axis = match axis_key.as_str() {
        "x" | "rx" | "rotationx" => 0,
        "y" | "ry" | "rotationy" => 1,
        "z" | "rz" | "rotationz" | "rotation" => 2,
        _ => return None,
    };
    let scale = scale_raw.trim().parse::<f32>().unwrap_or(1.0);
    Some((axis, sign * scale))
}

#[derive(Debug, Clone, Copy, Default)]
struct SampledNodeTrs {
    translation: Option<[f32; 3]>,
    rotation: Option<[f32; 4]>,
    scale: Option<[f32; 3]>,
}

fn actor_global_node_matrices(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
    time: WorldTime,
    overrides: &HashMap<String, BoneOverride>,
    external_sampled: Option<&HashMap<usize, SampledNodeTrs>>,
) -> Result<Vec<[f32; 16]>, WorldRenderError> {
    let mut sampled = sample_actor_clip(graph, actor, mesh, time)?;
    if let Some(external) = external_sampled {
        for (node, value) in external {
            sampled.insert(*node, *value);
        }
    }
    Ok(global_node_matrices_with_sampled(mesh, overrides, &sampled))
}

fn sample_actor_clip(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
    time: WorldTime,
) -> Result<HashMap<usize, SampledNodeTrs>, WorldRenderError> {
    let profile = actor_model_profile(graph, actor);
    let mut sampled = HashMap::<usize, SampledNodeTrs>::new();
    for play in actor_play_layers(actor) {
        let animation = match play.clip.as_deref() {
            Some(name) => mesh
                .animations
                .iter()
                .find(|animation| animation.name.as_deref() == Some(name)),
            None => mesh.animations.first(),
        };
        let Some(animation) = animation else {
            continue;
        };
        let speed = eval_number(&play.speed, 1.0, time)?.max(0.0);
        let raw_time = time.time_sec() * speed;
        let duration = animation.duration.max(0.0);
        let clip_time = if play.r#loop && duration > f32::EPSILON {
            raw_time % duration
        } else {
            raw_time.min(duration)
        };
        let blend_in = eval_number(&play.blend_in, 0.0, time)?.max(0.0);
        let blend_out = eval_number(&play.blend_out, 0.0, time)?.max(0.0);
        let mut weight = eval_number(&play.weight, 1.0, time)?.clamp(0.0, 1.0);
        if blend_in > f32::EPSILON {
            weight *= (raw_time / blend_in).clamp(0.0, 1.0);
        }
        if !play.r#loop && blend_out > f32::EPSILON && duration > f32::EPSILON {
            weight *= ((duration - clip_time) / blend_out).clamp(0.0, 1.0);
        }
        if weight <= f32::EPSILON {
            continue;
        }

        for channel in &animation.channels {
            let Some(node) = mesh.nodes.get(channel.node_index) else {
                continue;
            };
            if !clip_node_matches_mask(node.name.as_deref(), profile, &play.mask) {
                continue;
            }
            let entry = sampled.entry(channel.node_index).or_default();
            match sample_animation_channel(channel, clip_time) {
                Some(GlbAnimationValues::Vec3(values)) => {
                    let Some(value) = values.first().copied() else {
                        continue;
                    };
                    match channel.property {
                        GlbAnimationProperty::Translation => {
                            entry.translation = Some(lerp_vec3(
                                entry.translation.unwrap_or(node.translation),
                                value,
                                weight,
                            ))
                        }
                        GlbAnimationProperty::Scale => {
                            entry.scale =
                                Some(lerp_vec3(entry.scale.unwrap_or(node.scale), value, weight))
                        }
                        GlbAnimationProperty::Rotation => {}
                    }
                }
                Some(GlbAnimationValues::Quat(values)) => {
                    if let Some(value) = values.first().copied() {
                        entry.rotation = Some(nlerp_quat(
                            entry.rotation.unwrap_or(node.rotation),
                            value,
                            weight,
                        ));
                    }
                }
                None => {}
            }
        }
    }
    Ok(sampled)
}

fn sample_external_actor_actions(
    graph: &WorldGraph,
    actor: &WorldActor,
    target_mesh: &GlbMeshData,
    animation_keys: &HashMap<String, PathBuf>,
    mesh_cache: &HashMap<PathBuf, GlbMeshData>,
    time: WorldTime,
) -> Result<HashMap<usize, SampledNodeTrs>, WorldRenderError> {
    let target_profile = actor_model_profile(graph, actor);
    let target_rest_matrices = bind_pose_node_matrices(target_mesh);
    let target_rest_global = target_rest_matrices
        .iter()
        .copied()
        .map(quat_from_mat4_rotation)
        .collect::<Vec<_>>();
    let mut sampled = HashMap::<usize, SampledNodeTrs>::new();
    for apply in graph
        .apply_actions
        .iter()
        .filter(|apply| apply.target == actor.id)
    {
        let Some(asset) = graph
            .animation_assets
            .iter()
            .find(|asset| asset.id == apply.action)
        else {
            continue;
        };
        let Some(source_mesh) = animation_keys
            .get(&asset.id)
            .and_then(|key| mesh_cache.get(key))
        else {
            continue;
        };
        let source_rest_matrices = global_node_matrices(source_mesh, &HashMap::new());
        let source_rest_global = source_rest_matrices
            .iter()
            .copied()
            .map(quat_from_mat4_rotation)
            .collect::<Vec<_>>();
        let animation = if let Some(name) = asset.clip.as_deref() {
            source_mesh
                .animations
                .iter()
                .find(|animation| animation.name.as_deref() == Some(name))
                .ok_or_else(|| WorldRenderError::GpuRender {
                    message: format!(
                        "AnimationAsset '{}' clip not found in '{}': {name}",
                        asset.id, asset.src
                    ),
                })?
        } else {
            source_mesh
                .animations
                .first()
                .ok_or_else(|| WorldRenderError::GpuRender {
                    message: format!(
                        "AnimationAsset '{}' contains no animation clips: {}",
                        asset.id, asset.src
                    ),
                })?
        };
        let speed = eval_number(&apply.speed, 1.0, time)?.max(0.0);
        let Some((clip_time, clip_duration)) =
            external_action_clip_time(apply, animation, speed, time)
        else {
            continue;
        };
        let timeline_elapsed = time.time_sec() - apply.at_ms as f32 / 1000.0;
        let authored_duration = apply.duration_ms.map(|value| value as f32 / 1000.0);
        let mut weight = eval_number(&apply.weight, 1.0, time)?.clamp(0.0, 1.0);
        let blend_in = eval_number(&apply.blend_in, 0.0, time)?.max(0.0);
        let blend_out = eval_number(&apply.blend_out, 0.0, time)?.max(0.0);
        if blend_in > f32::EPSILON {
            weight *= (timeline_elapsed / blend_in).clamp(0.0, 1.0);
        }
        if !apply.r#loop && blend_out > f32::EPSILON {
            let end = authored_duration.unwrap_or(clip_duration);
            weight *= ((end - timeline_elapsed) / blend_out).clamp(0.0, 1.0);
        }
        if weight <= f32::EPSILON {
            continue;
        }

        // Reconstruct the complete source hierarchy for this frame, transfer
        // each canonical bone's model-space delta, then derive the target's
        // local rotation from its already-retargeted parent. This is the key
        // difference between humanoid retargeting and copying local channels.
        let source_local_rotations = animation
            .channels
            .iter()
            .filter(|channel| channel.property == GlbAnimationProperty::Rotation)
            .filter_map(|channel| {
                let GlbAnimationValues::Quat(values) =
                    sample_animation_channel(channel, clip_time)?
                else {
                    return None;
                };
                values
                    .first()
                    .copied()
                    .map(|value| (channel.node_index, quat_normalize_xyzw(value)))
            })
            .collect::<HashMap<_, _>>();
        let source_sampled_trs = source_local_rotations
            .iter()
            .map(|(index, rotation)| {
                (
                    *index,
                    SampledNodeTrs {
                        rotation: Some(*rotation),
                        ..Default::default()
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let source_animated_matrices =
            global_node_matrices_with_sampled(source_mesh, &HashMap::new(), &source_sampled_trs);
        let source_animated_global = source_animated_matrices
            .iter()
            .copied()
            .map(quat_from_mat4_rotation)
            .collect::<Vec<_>>();
        let mut mapped_rotations = animation
            .channels
            .iter()
            .filter(|channel| channel.property == GlbAnimationProperty::Rotation)
            .filter_map(|channel| {
                let source_node = source_mesh.nodes.get(channel.node_index)?;
                let canonical =
                    canonical_humanoid_bone(source_node.name.as_deref()?, &asset.profile)?;
                if !bone_matches_body_mask(&canonical, &apply.mask) {
                    return None;
                }
                let target_index =
                    target_node_for_canonical_bone(target_mesh, target_profile, &canonical)?;
                Some((channel.node_index, target_index, canonical))
            })
            .collect::<Vec<_>>();
        mapped_rotations
            .sort_by_key(|(_, target_index, _)| node_hierarchy_depth(target_mesh, *target_index));
        let mut target_animated_global = HashMap::<usize, [f32; 4]>::new();
        for (source_index, target_index, canonical) in mapped_rotations {
            let Some(target_node) = target_mesh.nodes.get(target_index) else {
                continue;
            };
            let source_rest = source_rest_global
                .get(source_index)
                .copied()
                .unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let source_animated = source_animated_global
                .get(source_index)
                .copied()
                .unwrap_or(source_rest);
            let target_rest = target_rest_global
                .get(target_index)
                .copied()
                .unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let desired_global = canonical_child_bone(&canonical)
                .and_then(|child| {
                    let source_child = target_node_for_canonical_bone(source_mesh, None, child)?;
                    let target_child =
                        target_node_for_canonical_bone(target_mesh, target_profile, child)?;
                    let source_rest_direction =
                        direction_between_nodes(&source_rest_matrices, source_index, source_child)?;
                    let source_animated_direction = direction_between_nodes(
                        &source_animated_matrices,
                        source_index,
                        source_child,
                    )?;
                    let target_rest_direction =
                        direction_between_nodes(&target_rest_matrices, target_index, target_child)?;
                    let source_swing =
                        quat_from_to(source_rest_direction, source_animated_direction)?;
                    let desired_target_direction =
                        quat_rotate_vec3(source_swing, target_rest_direction);
                    let target_swing =
                        quat_from_to(target_rest_direction, desired_target_direction)?;
                    Some(quat_normalize_xyzw(quat_mul_xyzw(
                        target_swing,
                        target_rest,
                    )))
                })
                .unwrap_or_else(|| {
                    let rest_space_delta =
                        quat_mul_xyzw(quat_conjugate_xyzw(source_rest), source_animated);
                    quat_normalize_xyzw(quat_mul_xyzw(target_rest, rest_space_delta))
                });
            let parent_global = target_node
                .parent
                .and_then(|parent| target_animated_global.get(&parent).copied())
                .or_else(|| {
                    target_node
                        .parent
                        .and_then(|parent| target_rest_global.get(parent).copied())
                })
                .unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let desired_local = quat_normalize_xyzw(quat_mul_xyzw(
                quat_conjugate_xyzw(parent_global),
                desired_global,
            ));
            target_animated_global.insert(target_index, desired_global);
            let entry = sampled.entry(target_index).or_default();
            entry.rotation = Some(nlerp_quat(
                entry.rotation.unwrap_or(target_node.rotation),
                desired_local,
                weight,
            ));
        }

        for channel in &animation.channels {
            let Some(source_node) = source_mesh.nodes.get(channel.node_index) else {
                continue;
            };
            let Some(source_name) = source_node.name.as_deref() else {
                continue;
            };
            let Some(canonical) = canonical_humanoid_bone(source_name, &asset.profile) else {
                continue;
            };
            if !bone_matches_body_mask(&canonical, &apply.mask) {
                continue;
            }
            let Some(target_index) =
                target_node_for_canonical_bone(target_mesh, target_profile, &canonical)
            else {
                continue;
            };
            let Some(target_node) = target_mesh.nodes.get(target_index) else {
                continue;
            };
            let entry = sampled.entry(target_index).or_default();
            match sample_animation_channel(channel, clip_time) {
                Some(GlbAnimationValues::Quat(_)) => {}
                Some(GlbAnimationValues::Vec3(values)) => {
                    let Some(value) = values.first().copied() else {
                        continue;
                    };
                    match channel.property {
                        GlbAnimationProperty::Translation
                            if canonical == "hips"
                                && apply.root_motion.as_deref() == Some("clip") =>
                        {
                            let delta: [f32; 3] = std::array::from_fn(|axis| {
                                value[axis] - source_node.translation[axis]
                            });
                            let translated: [f32; 3] = std::array::from_fn(|axis| {
                                target_node.translation[axis] + delta[axis]
                            });
                            entry.translation = Some(lerp_vec3(
                                entry.translation.unwrap_or(target_node.translation),
                                translated,
                                weight,
                            ));
                        }
                        GlbAnimationProperty::Scale => {
                            entry.scale = Some(lerp_vec3(
                                entry.scale.unwrap_or(target_node.scale),
                                value,
                                weight,
                            ));
                        }
                        _ => {}
                    }
                }
                None => {}
            }
        }
    }
    Ok(sampled)
}

fn external_action_clip_time(
    apply: &WorldApplyAction,
    animation: &GlbAnimationData,
    speed: f32,
    time: WorldTime,
) -> Option<(f32, f32)> {
    let timeline_elapsed = time.time_sec() - apply.at_ms as f32 / 1000.0;
    if timeline_elapsed < 0.0 {
        return None;
    }
    let authored_duration = apply.duration_ms.map(|value| value as f32 / 1000.0);
    if !apply.r#loop && authored_duration.is_some_and(|duration| timeline_elapsed > duration) {
        return None;
    }
    let elapsed = timeline_elapsed * speed;
    let clip_duration = animation.duration.max(0.0);
    let clip_time = if !apply.r#loop
        && let Some(duration) = authored_duration.filter(|duration| *duration > f32::EPSILON)
    {
        (timeline_elapsed / duration).clamp(0.0, 1.0) * clip_duration * speed
    } else if apply.r#loop && clip_duration > f32::EPSILON {
        elapsed % clip_duration
    } else {
        elapsed.min(clip_duration)
    };
    Some((clip_time, clip_duration))
}

fn target_node_for_canonical_bone(
    mesh: &GlbMeshData,
    profile: Option<&WorldModelProfile>,
    canonical: &str,
) -> Option<usize> {
    let mapped = profile
        .and_then(|profile| profile.retarget.as_ref())
        .and_then(|retarget| {
            retarget
                .maps
                .iter()
                .find(|map| map.to == canonical)
                .map(|map| map.from.as_str())
        });
    if let Some(mapped) = mapped
        && let Some(index) = node_index_by_name(mesh, mapped)
    {
        return Some(index);
    }
    mesh.nodes.iter().find_map(|node| {
        let name = node.name.as_deref()?;
        (canonical_humanoid_bone(name, "auto").as_deref() == Some(canonical)).then_some(node.index)
    })
}

fn camera_hidden_joint_indices(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
) -> Vec<usize> {
    let Some(skin) = mesh.skin.as_ref() else {
        return Vec::new();
    };
    let profile = actor
        .profile
        .as_deref()
        .and_then(|id| graph.model_profiles.iter().find(|profile| profile.id == id));
    let mut hidden_nodes = HashSet::<usize>::new();
    for canonical in &actor.camera_hidden_bones {
        let Some(root) = target_node_for_canonical_bone(mesh, profile, canonical) else {
            continue;
        };
        let mut pending = vec![root];
        while let Some(node_index) = pending.pop() {
            if !hidden_nodes.insert(node_index) {
                continue;
            }
            if let Some(node) = mesh.nodes.get(node_index) {
                pending.extend(node.children.iter().copied());
            }
        }
    }
    skin.joints
        .iter()
        .enumerate()
        .filter_map(|(joint_index, joint)| {
            hidden_nodes
                .contains(&joint.node_index)
                .then_some(joint_index)
        })
        .take(32)
        .collect()
}

fn camera_hidden_bones_hide_whole_actor(hidden_bones: &[String]) -> bool {
    hidden_bones.iter().any(|bone| bone == "hips")
}

fn camera_hidden_joint_slots(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
) -> [[f32; 4]; 8] {
    let mut slots = [[0.0; 4]; 8];
    for (slot, joint) in camera_hidden_joint_indices(graph, actor, mesh)
        .into_iter()
        .enumerate()
    {
        slots[slot / 4][slot % 4] = joint as f32 + 1.0;
    }
    slots
}

fn canonical_humanoid_bone(raw: &str, _profile: &str) -> Option<String> {
    let local_name = raw.rsplit(':').next().unwrap_or(raw);
    let normalized = local_name
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-', '.', '_'], "");
    let canonical = match normalized.as_str() {
        "hips" | "pelvis" => "hips",
        "spine" | "spine01" => "spine",
        "spine1" | "spine02" | "chest" => "chest",
        "spine2" | "spine03" | "upperchest" => "upper_chest",
        "neck" | "neck01" => "neck",
        "head" => "head",
        "leftshoulder" => "shoulder_l",
        "leftarm" | "leftupperarm" => "upper_arm_l",
        "leftforearm" | "leftlowerarm" => "forearm_l",
        "lefthand" | "leftwrist" => "hand_l",
        "rightshoulder" => "shoulder_r",
        "rightarm" | "rightupperarm" => "upper_arm_r",
        "rightforearm" | "rightlowerarm" => "forearm_r",
        "righthand" | "rightwrist" => "hand_r",
        "leftupleg" | "leftthigh" | "leftupperleg" => "upper_leg_l",
        "leftleg" | "leftcalf" | "leftlowerleg" => "lower_leg_l",
        "leftfoot" | "leftankle" => "foot_l",
        "lefttoebase" | "lefttoe" => "toe_l",
        "rightupleg" | "rightthigh" | "rightupperleg" => "upper_leg_r",
        "rightleg" | "rightcalf" | "rightlowerleg" => "lower_leg_r",
        "rightfoot" | "rightankle" => "foot_r",
        "righttoebase" | "righttoe" => "toe_r",
        "shoulderl" | "claviclel" => "shoulder_l",
        "upperarml" => "upper_arm_l",
        "forearml" | "lowerarml" => "forearm_l",
        "handl" => "hand_l",
        "shoulderr" | "clavicler" => "shoulder_r",
        "upperarmr" => "upper_arm_r",
        "forearmr" | "lowerarmr" => "forearm_r",
        "handr" => "hand_r",
        "upperlegl" | "thighl" => "upper_leg_l",
        "lowerlegl" | "calfl" => "lower_leg_l",
        "footl" => "foot_l",
        "toel" | "balll" | "ballleafl" => "toe_l",
        "upperlegr" | "thighr" => "upper_leg_r",
        "lowerlegr" | "calfr" => "lower_leg_r",
        "footr" => "foot_r",
        "toer" | "ballr" | "ballleafr" => "toe_r",
        _ => return canonical_humanoid_finger(&normalized),
    };
    Some(canonical.to_string())
}

fn canonical_humanoid_finger(normalized: &str) -> Option<String> {
    // This mapper runs once per animation channel and can be reached several
    // times per actor/frame (collision, camera anchors and final drawing).
    // Parse the three supported naming shapes without constructing every
    // possible candidate. Unknown helper bones therefore stay allocation-free.
    let (side, finger_joint) = if let Some(value) = normalized.strip_prefix("lefthand") {
        ("l", value)
    } else if let Some(value) = normalized.strip_prefix("righthand") {
        ("r", value)
    } else if let Some(value) = normalized.strip_prefix("left") {
        ("l", value)
    } else if let Some(value) = normalized.strip_prefix("right") {
        ("r", value)
    } else if let Some(value) = normalized.strip_suffix('l') {
        ("l", value)
    } else if let Some(value) = normalized.strip_suffix('r') {
        ("r", value)
    } else {
        return None;
    };

    let (finger, joint) = ["thumb", "index", "middle", "ring", "pinky"]
        .into_iter()
        .find_map(|finger| {
            let joint = finger_joint.strip_prefix(finger)?;
            matches!(joint, "1" | "2" | "3").then_some((finger, joint))
        })?;
    Some(format!("{finger}_{joint}_{side}"))
}

fn quat_conjugate_xyzw(value: [f32; 4]) -> [f32; 4] {
    [-value[0], -value[1], -value[2], value[3]]
}

fn quat_mul_xyzw(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn quat_normalize_xyzw(value: [f32; 4]) -> [f32; 4] {
    let length = value.iter().map(|part| part * part).sum::<f32>().sqrt();
    if length <= f32::EPSILON {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        std::array::from_fn(|axis| value[axis] / length)
    }
}

fn canonical_child_bone(canonical: &str) -> Option<&'static str> {
    match canonical {
        "hips" => Some("spine"),
        "spine" => Some("chest"),
        "chest" => Some("upper_chest"),
        "upper_chest" => Some("neck"),
        "neck" => Some("head"),
        "shoulder_l" => Some("upper_arm_l"),
        "upper_arm_l" => Some("forearm_l"),
        "forearm_l" => Some("hand_l"),
        "shoulder_r" => Some("upper_arm_r"),
        "upper_arm_r" => Some("forearm_r"),
        "forearm_r" => Some("hand_r"),
        "upper_leg_l" => Some("lower_leg_l"),
        "lower_leg_l" => Some("foot_l"),
        "foot_l" => Some("toe_l"),
        "upper_leg_r" => Some("lower_leg_r"),
        "lower_leg_r" => Some("foot_r"),
        "foot_r" => Some("toe_r"),
        _ => None,
    }
}

fn direction_between_nodes(matrices: &[[f32; 16]], from: usize, to: usize) -> Option<[f32; 3]> {
    let from = matrix_translation(*matrices.get(from)?);
    let to = matrix_translation(*matrices.get(to)?);
    normalize_vec3([to[0] - from[0], to[1] - from[1], to[2] - from[2]])
}

fn quat_rotate_vec3(rotation: [f32; 4], value: [f32; 3]) -> [f32; 3] {
    let vector = [value[0], value[1], value[2], 0.0];
    let rotated = quat_mul_xyzw(
        quat_mul_xyzw(rotation, vector),
        quat_conjugate_xyzw(rotation),
    );
    [rotated[0], rotated[1], rotated[2]]
}

fn quat_from_to(from: [f32; 3], to: [f32; 3]) -> Option<[f32; 4]> {
    let from = normalize_vec3(from)?;
    let to = normalize_vec3(to)?;
    let dot = (from[0] * to[0] + from[1] * to[1] + from[2] * to[2]).clamp(-1.0, 1.0);
    if dot > 1.0 - 1.0e-6 {
        return Some([0.0, 0.0, 0.0, 1.0]);
    }
    if dot < -1.0 + 1.0e-6 {
        let seed = if from[0].abs() < from[1].abs() {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let axis = normalize_vec3([
            from[1] * seed[2] - from[2] * seed[1],
            from[2] * seed[0] - from[0] * seed[2],
            from[0] * seed[1] - from[1] * seed[0],
        ])?;
        return Some([axis[0], axis[1], axis[2], 0.0]);
    }
    let cross = [
        from[1] * to[2] - from[2] * to[1],
        from[2] * to[0] - from[0] * to[2],
        from[0] * to[1] - from[1] * to[0],
    ];
    Some(quat_normalize_xyzw([
        cross[0],
        cross[1],
        cross[2],
        1.0 + dot,
    ]))
}

fn node_hierarchy_depth(mesh: &GlbMeshData, mut index: usize) -> usize {
    let mut depth = 0usize;
    while let Some(parent) = mesh.nodes.get(index).and_then(|node| node.parent) {
        depth += 1;
        index = parent;
        if depth > mesh.nodes.len() {
            break;
        }
    }
    depth
}

fn actor_play_layers(actor: &WorldActor) -> impl Iterator<Item = &WorldPlay> {
    actor.play.iter().chain(actor.plays.iter())
}

fn actor_has_clip_layers(actor: &WorldActor) -> bool {
    actor_play_layers(actor).next().is_some()
}

fn clip_node_matches_mask(
    raw_name: Option<&str>,
    profile: Option<&WorldModelProfile>,
    masks: &[String],
) -> bool {
    if masks.is_empty() {
        return true;
    }
    let Some(raw_name) = raw_name else {
        return false;
    };
    let canonical = profile
        .and_then(|profile| profile.retarget.as_ref())
        .and_then(|retarget| {
            retarget
                .maps
                .iter()
                .find(|map| map.from == raw_name)
                .map(|map| map.to.as_str())
        })
        .unwrap_or(raw_name);
    bone_matches_body_mask(canonical, masks)
}

fn sample_animation_channel(
    channel: &GlbAnimationChannelData,
    time: f32,
) -> Option<GlbAnimationValues> {
    if channel.times.is_empty() {
        return None;
    }
    let last_index = channel.times.len().saturating_sub(1);
    let next = channel
        .times
        .partition_point(|key| *key < time)
        .min(last_index);
    let previous = next.saturating_sub(1);
    let span = (channel.times[next] - channel.times[previous]).max(f32::EPSILON);
    let alpha = if next == previous || channel.interpolation == GlbAnimationInterpolation::Step {
        0.0
    } else {
        ((time - channel.times[previous]) / span).clamp(0.0, 1.0)
    };
    match &channel.values {
        GlbAnimationValues::Vec3(values) => {
            let from = values.get(previous).copied()?;
            let to = values.get(next).copied().unwrap_or(from);
            Some(GlbAnimationValues::Vec3(vec![lerp_vec3(from, to, alpha)]))
        }
        GlbAnimationValues::Quat(values) => {
            let from = values.get(previous).copied()?;
            let to = values.get(next).copied().unwrap_or(from);
            Some(GlbAnimationValues::Quat(vec![nlerp_quat(from, to, alpha)]))
        }
    }
}

fn lerp_vec3(from: [f32; 3], to: [f32; 3], alpha: f32) -> [f32; 3] {
    std::array::from_fn(|axis| from[axis] + (to[axis] - from[axis]) * alpha)
}

fn nlerp_quat(mut from: [f32; 4], mut to: [f32; 4], alpha: f32) -> [f32; 4] {
    let dot = (0..4).map(|axis| from[axis] * to[axis]).sum::<f32>();
    if dot < 0.0 {
        for value in &mut to {
            *value = -*value;
        }
    }
    let mut out = std::array::from_fn(|axis| from[axis] + (to[axis] - from[axis]) * alpha);
    let length = out.iter().map(|value| value * value).sum::<f32>().sqrt();
    if length > f32::EPSILON {
        for value in &mut out {
            *value /= length;
        }
    } else {
        from = [0.0, 0.0, 0.0, 1.0];
        out = from;
    }
    out
}

fn global_node_matrices(
    mesh: &GlbMeshData,
    overrides: &HashMap<String, BoneOverride>,
) -> Vec<[f32; 16]> {
    global_node_matrices_with_sampled(mesh, overrides, &HashMap::new())
}

/// Joint bind transforms are the authoritative rest coordinate frames for a
/// skinned mesh. Node TRS alone can describe the same joint positions while
/// carrying a different bone roll, which breaks cross-rig animation transfer.
fn bind_pose_node_matrices(mesh: &GlbMeshData) -> Vec<[f32; 16]> {
    let mut matrices = global_node_matrices(mesh, &HashMap::new());
    if let Some(skin) = &mesh.skin {
        for joint in &skin.joints {
            if let Some(bind_global) = mat4_inverse_affine(joint.inverse_bind_matrix)
                && let Some(slot) = matrices.get_mut(joint.node_index)
            {
                *slot = bind_global;
            }
        }
    }
    matrices
}

fn humanoid_bind_pose_metrics(
    mesh: &GlbMeshData,
    profile: Option<&WorldModelProfile>,
) -> Scene3DHumanoidRigMetrics {
    let matrices = bind_pose_node_matrices(mesh);
    let height = (mesh.bounds_max[1] - mesh.bounds_min[1]).abs().max(0.001);
    let normalized_y = |bone: &str| {
        let index = target_node_for_canonical_bone(mesh, profile, bone)?;
        let point = matrices.get(index).copied().map(matrix_translation)?;
        Some((point[1] - mesh.bounds_min[1]) / height)
    };
    let sole_offset = ["foot_l", "toe_l", "foot_r", "toe_r"]
        .into_iter()
        .filter_map(normalized_y)
        .min_by(f32::total_cmp)
        .unwrap_or(0.0)
        .clamp(-0.25, 0.35);
    let knee_offset = ["lower_leg_l", "lower_leg_r"]
        .into_iter()
        .filter_map(normalized_y)
        .reduce(|left, right| (left + right) * 0.5)
        .unwrap_or(0.28);
    let hips_offset = normalized_y("hips").unwrap_or(0.52);
    let head_offset = normalized_y("head").unwrap_or(0.94);
    Scene3DHumanoidRigMetrics {
        sole_offset,
        knee_offset,
        hips_offset,
        head_offset,
        body_height: (head_offset - sole_offset).abs().max(0.5),
    }
}

fn global_node_matrices_with_sampled(
    mesh: &GlbMeshData,
    overrides: &HashMap<String, BoneOverride>,
    sampled: &HashMap<usize, SampledNodeTrs>,
) -> Vec<[f32; 16]> {
    let local = mesh
        .nodes
        .iter()
        .map(|node| {
            let sample = sampled.get(&node.index).copied().unwrap_or_default();
            let base = if sample.translation.is_some()
                || sample.rotation.is_some()
                || sample.scale.is_some()
            {
                mat4_from_trs(
                    sample.translation.unwrap_or(node.translation),
                    sample.rotation.unwrap_or(node.rotation),
                    sample.scale.unwrap_or(node.scale),
                )
            } else {
                node.matrix
                    .unwrap_or_else(|| mat4_from_trs(node.translation, node.rotation, node.scale))
            };
            let Some(name) = node.name.as_deref() else {
                return base;
            };
            let Some(override_transform) = overrides.get(name).copied() else {
                return base;
            };
            mat4_mul(base, mat4_from_override(override_transform))
        })
        .collect::<Vec<_>>();
    let mut global = vec![None; mesh.nodes.len()];
    for index in 0..mesh.nodes.len() {
        compute_global_node_matrix(index, &mesh.nodes, &local, &mut global);
    }
    global
        .into_iter()
        .map(|matrix| matrix.unwrap_or_else(mat4_identity))
        .collect()
}

fn compute_global_node_matrix(
    index: usize,
    nodes: &[crate::world::gltf_loader::GlbNodeData],
    local: &[[f32; 16]],
    global: &mut [Option<[f32; 16]>],
) -> [f32; 16] {
    if let Some(matrix) = global.get(index).copied().flatten() {
        return matrix;
    }
    let local_matrix = local.get(index).copied().unwrap_or_else(mat4_identity);
    let matrix = nodes
        .get(index)
        .and_then(|node| node.parent)
        .map(|parent| {
            mat4_mul(
                compute_global_node_matrix(parent, nodes, local, global),
                local_matrix,
            )
        })
        .unwrap_or(local_matrix);
    if let Some(slot) = global.get_mut(index) {
        *slot = Some(matrix);
    }
    matrix
}

fn mat4_from_override(transform: BoneOverride) -> [f32; 16] {
    let translation = mat4_translation(transform.translation);
    let rotation = mat4_mul(
        mat4_mul(
            mat4_rotation_z(transform.rotation_deg[2].to_radians()),
            mat4_rotation_y(transform.rotation_deg[1].to_radians()),
        ),
        mat4_rotation_x(transform.rotation_deg[0].to_radians()),
    );
    let scale = mat4_scale([transform.scale, transform.scale, transform.scale]);
    mat4_mul(mat4_mul(translation, rotation), scale)
}

fn mat4_from_trs(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> [f32; 16] {
    mat4_mul(
        mat4_mul(mat4_translation(translation), mat4_from_quat(rotation)),
        mat4_scale(scale),
    )
}

fn mat4_identity() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn mat4_translation(translation: [f32; 3]) -> [f32; 16] {
    [
        1.0,
        0.0,
        0.0,
        0.0, //
        0.0,
        1.0,
        0.0,
        0.0, //
        0.0,
        0.0,
        1.0,
        0.0, //
        translation[0],
        translation[1],
        translation[2],
        1.0,
    ]
}

fn mat4_scale(scale: [f32; 3]) -> [f32; 16] {
    [
        scale[0], 0.0, 0.0, 0.0, //
        0.0, scale[1], 0.0, 0.0, //
        0.0, 0.0, scale[2], 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn mat4_rotation_x(angle: f32) -> [f32; 16] {
    let (sin, cos) = angle.sin_cos();
    [
        1.0, 0.0, 0.0, 0.0, //
        0.0, cos, sin, 0.0, //
        0.0, -sin, cos, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn mat4_rotation_y(angle: f32) -> [f32; 16] {
    let (sin, cos) = angle.sin_cos();
    [
        cos, 0.0, -sin, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        sin, 0.0, cos, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn mat4_rotation_z(angle: f32) -> [f32; 16] {
    let (sin, cos) = angle.sin_cos();
    [
        cos, sin, 0.0, 0.0, //
        -sin, cos, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn quat_from_mat4_rotation(matrix: [f32; 16]) -> [f32; 4] {
    let x = normalize_vec3([matrix[0], matrix[1], matrix[2]]).unwrap_or([1.0, 0.0, 0.0]);
    let y = normalize_vec3([matrix[4], matrix[5], matrix[6]]).unwrap_or([0.0, 1.0, 0.0]);
    let z = normalize_vec3([matrix[8], matrix[9], matrix[10]]).unwrap_or([0.0, 0.0, 1.0]);
    let m00 = x[0];
    let m01 = y[0];
    let m02 = z[0];
    let m10 = x[1];
    let m11 = y[1];
    let m12 = z[1];
    let m20 = x[2];
    let m21 = y[2];
    let m22 = z[2];
    let trace = m00 + m11 + m22;
    let quat = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
    };
    quat_normalize_xyzw(quat)
}

fn mat4_from_quat(quat: [f32; 4]) -> [f32; 16] {
    let [x, y, z, w] = quat;
    let len = (x * x + y * y + z * z + w * w).sqrt();
    if len <= f32::EPSILON {
        return mat4_identity();
    }
    let x = x / len;
    let y = y / len;
    let z = z / len;
    let w = w / len;
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
    [
        1.0 - (yy + zz),
        xy + wz,
        xz - wy,
        0.0,
        xy - wz,
        1.0 - (xx + zz),
        yz + wx,
        0.0,
        xz + wy,
        yz - wx,
        1.0 - (xx + yy),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum();
        }
    }
    out
}

fn mat4_inverse_affine(matrix: [f32; 16]) -> Option<[f32; 16]> {
    let a00 = matrix[0];
    let a01 = matrix[4];
    let a02 = matrix[8];
    let a10 = matrix[1];
    let a11 = matrix[5];
    let a12 = matrix[9];
    let a20 = matrix[2];
    let a21 = matrix[6];
    let a22 = matrix[10];
    let det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20);
    if det.abs() <= 1.0e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    let r00 = (a11 * a22 - a12 * a21) * inv_det;
    let r01 = (a02 * a21 - a01 * a22) * inv_det;
    let r02 = (a01 * a12 - a02 * a11) * inv_det;
    let r10 = (a12 * a20 - a10 * a22) * inv_det;
    let r11 = (a00 * a22 - a02 * a20) * inv_det;
    let r12 = (a02 * a10 - a00 * a12) * inv_det;
    let r20 = (a10 * a21 - a11 * a20) * inv_det;
    let r21 = (a01 * a20 - a00 * a21) * inv_det;
    let r22 = (a00 * a11 - a01 * a10) * inv_det;
    let tx = matrix[12];
    let ty = matrix[13];
    let tz = matrix[14];
    Some([
        r00,
        r10,
        r20,
        0.0,
        r01,
        r11,
        r21,
        0.0,
        r02,
        r12,
        r22,
        0.0,
        -(r00 * tx + r01 * ty + r02 * tz),
        -(r10 * tx + r11 * ty + r12 * tz),
        -(r20 * tx + r21 * ty + r22 * tz),
        1.0,
    ])
}

fn mat4_transform_point(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
    ]
}

#[allow(clippy::too_many_arguments)]
fn draw_actor_mesh_projection(
    canvas: &mut RgbaImage,
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
    positions: &[[f32; 3]],
    world_x: f32,
    world_y: f32,
    world_depth: f32,
    view_yaw_deg: f32,
    camera_pitch_deg: f32,
    fov: f32,
    distance: f32,
    scale: f32,
    opacity: f32,
) {
    let width = canvas.width() as f32;
    let height = canvas.height() as f32;
    let model_height = (mesh.bounds_max[1] - mesh.bounds_min[1]).abs().max(0.001);
    let px_per_world = (height / distance) * (35.0 / fov).clamp(0.35, 2.5);
    let normalize_height = actor.scale_mode.eq_ignore_ascii_case("normalize_height");
    let (model_center_x, model_origin_y, model_center_z, model_px) = if normalize_height {
        (
            (mesh.bounds_min[0] + mesh.bounds_max[0]) * 0.5,
            mesh.bounds_min[1],
            (mesh.bounds_min[2] + mesh.bounds_max[2]) * 0.5,
            (height * 0.58 * scale * (3.2 / distance).clamp(0.25, 4.0)) / model_height,
        )
    } else {
        (0.0, 0.0, 0.0, px_per_world * scale)
    };
    let cx = width * 0.5 + world_x * px_per_world;
    let ground_y = height * 0.82 - world_y * px_per_world;
    let yaw = view_yaw_deg.to_radians();
    let cos_y = yaw.cos();
    let sin_y = yaw.sin();
    let pitch = camera_pitch_deg.to_radians();
    let cos_p = pitch.cos();
    let sin_p = pitch.sin();

    let mut projected = Vec::<([f32; 2], f32)>::with_capacity(positions.len());
    for position in positions {
        let x = position[0] - model_center_x;
        let y = position[1] - model_origin_y;
        let z = position[2] - model_center_z;
        let rx = x * cos_y + z * sin_y;
        let rz = -x * sin_y + z * cos_y;
        let ry = y * cos_p - rz * sin_p;
        let rz = y * sin_p + rz * cos_p + world_depth * WORLD_DEPTH_SORT_SCALE;
        projected.push(([cx + rx * model_px, ground_y - ry * model_px], rz));
    }

    let base = if actor
        .material
        .as_ref()
        .is_some_and(|material| material.outline)
    {
        [93, 126, 178]
    } else {
        [111, 145, 190]
    };
    let mut triangles = Vec::<ProjectedTriangle>::new();
    let triangle_source = mesh
        .triangles
        .iter()
        .map(|triangle| (triangle.indices, triangle.material));
    let hide_actor_from_camera = camera_hidden_bones_hide_whole_actor(&actor.camera_hidden_bones);
    let hidden_joints = camera_hidden_joint_indices(graph, actor, mesh)
        .into_iter()
        .collect::<HashSet<_>>();
    for (indices, material) in triangle_source {
        let hidden = hide_actor_from_camera
            || indices.iter().any(|index| {
                let vertex = *index as usize;
                let Some(joints) = mesh.joints.get(vertex).copied().flatten() else {
                    return false;
                };
                let weights = mesh
                    .weights
                    .get(vertex)
                    .copied()
                    .flatten()
                    .unwrap_or([0.0; 4]);
                joints.into_iter().zip(weights).any(|(joint, weight)| {
                    hidden_joints.contains(&(joint as usize)) && weight > 0.01
                })
            });
        if hidden {
            continue;
        }
        let Some(a) = projected.get(indices[0] as usize) else {
            continue;
        };
        let Some(b) = projected.get(indices[1] as usize) else {
            continue;
        };
        let Some(c) = projected.get(indices[2] as usize) else {
            continue;
        };
        let ax = b.0[0] - a.0[0];
        let ay = b.0[1] - a.0[1];
        let bx = c.0[0] - a.0[0];
        let by = c.0[1] - a.0[1];
        let screen_cross = ax * by - ay * bx;
        if screen_cross.abs() < 0.01 {
            continue;
        }
        let shade = if screen_cross < 0.0 { 0.78 } else { 1.0 };
        let uvs = [
            mesh.texcoords.get(indices[0] as usize).copied().flatten(),
            mesh.texcoords.get(indices[1] as usize).copied().flatten(),
            mesh.texcoords.get(indices[2] as usize).copied().flatten(),
        ];
        triangles.push(ProjectedTriangle {
            depth: (a.1 + b.1 + c.1) / 3.0,
            points: [a.0, b.0, c.0],
            uvs,
            material,
            shade,
        });
    }
    triangles.sort_by(|a, b| a.depth.total_cmp(&b.depth));
    for triangle in triangles {
        if let Some((texture, material_factor, uvs)) =
            textured_triangle_source(mesh, triangle.material, triangle.uvs)
        {
            fill_textured_triangle(
                canvas,
                triangle.points,
                uvs,
                texture.width,
                texture.height,
                &texture.rgba,
                material_factor,
                triangle.shade,
                opacity,
            );
        } else {
            fill_triangle(
                canvas,
                triangle.points,
                material_color(mesh, triangle.material, base, triangle.shade, opacity),
            );
        }
    }
}

struct ProjectedTriangle {
    depth: f32,
    points: [[f32; 2]; 3],
    uvs: [Option<[f32; 2]>; 3],
    material: Option<usize>,
    shade: f32,
}

#[derive(Debug, Clone, Copy)]
struct GpuWorldVertex {
    outline_normal: [f32; 3],
    position: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
    joints: [f32; 4],
    weights: [f32; 4],
    uv: [f32; 2],
    color: [f32; 4],
}

#[derive(Debug)]
struct GpuWorldDraw {
    resource_key: GpuWorldResourceKey,
    instance_key: GpuWorldInstanceKey,
    vertices: Arc<Vec<GpuWorldVertex>>,
    indices: Arc<Vec<u32>>,
    vertex_signature: u64,
    texture: Arc<GpuWorldTexture>,
    normal_texture: Arc<GpuWorldTexture>,
    metallic_roughness_texture: Arc<GpuWorldTexture>,
    emissive_texture: Arc<GpuWorldTexture>,
    occlusion_texture: Arc<GpuWorldTexture>,
    cel_texture: Arc<GpuWorldTexture>,
    bone_matrices: Vec<[f32; 16]>,
    params: GpuWorldParams,
    phase: GpuWorldDrawPhase,
    depth_write: bool,
    sort_priority: i32,
    camera_depth: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuWorldTexture {
    width: u32,
    height: u32,
    rgba: Arc<Vec<u8>>,
    signature: u64,
}

/// Linear floating-point equirectangular environment with a CPU-built mip chain.
#[derive(Debug)]
struct WorldEnvironmentImage {
    width: u32,
    height: u32,
    mip_bytes: Vec<Vec<u8>>,
    signature: u64,
}

/// Frame-local lighting ready for one GPU submission. Environment pixels stay
/// shared across frames while animated light values are repacked every frame.
#[derive(Debug, Clone)]
struct GpuWorldLighting {
    params: GpuWorldLightingParams,
    environment: Arc<WorldEnvironmentImage>,
    frame_index: u32,
    temporal_jitter: bool,
}

#[derive(Debug, Clone, Copy)]
struct GpuWorldLightingParams {
    universal_color: [f32; 4],
    universal_tone: [f32; 4],
    universal_shadow: [f32; 4],
    universal_highlight: [f32; 4],
    cel0: [f32; 4],
    cel1: [f32; 4],
    cel2: [f32; 4],
    surface0: [f32; 4],
    surface1: [f32; 4],
    surface2: [f32; 4],
    surface3: [f32; 4],
    environment0: [f32; 4],
    environment1: [f32; 4],
    environment2: [f32; 4],
    color0: [f32; 4],
    color1: [f32; 4],
    fog0: [f32; 4],
    fog1: [f32; 4],
    fog2: [f32; 4],
    fog3: [f32; 4],
    fog4: [f32; 4],
    optics0: [f32; 4],
    dof_style: [f32; 4],
    render_compat: [f32; 4],
    camera0: [f32; 4],
    camera1: [f32; 4],
    camera2: [f32; 4],
    camera3: [f32; 4],
    previous_camera0: [f32; 4],
    previous_camera1: [f32; 4],
    previous_camera2: [f32; 4],
    previous_camera3: [f32; 4],
    preview0: [f32; 4],
    preview1: [f32; 4],
    shadow0: [f32; 4],
    shadow1: [f32; 4],
    shadow2: [f32; 4],
    shadow3: [f32; 4],
    lights: [[f32; 16]; 8],
}

type GpuWorldActorBounds = (([f32; 3], [f32; 3]), GpuWorldParams);

// Bounding spheres remain stable as rigid actors rotate and include off-camera casters.
// Deformed scenes retain the existing projection until deformed bounds are available.
impl GpuWorldTexture {
    fn new(width: u32, height: u32, rgba: impl Into<Arc<Vec<u8>>>) -> Self {
        let rgba = rgba.into();
        let mut hasher = DefaultHasher::new();
        width.hash(&mut hasher);
        height.hash(&mut hasher);
        rgba.hash(&mut hasher);
        Self {
            width,
            height,
            rgba,
            signature: hasher.finish(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GpuWorldDrawKey {
    material: Option<usize>,
    texture: Option<usize>,
    mesh: Option<usize>,
    mesh_node: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GpuWorldResourceKey {
    model_path: PathBuf,
    draw_key: GpuWorldDrawKey,
    binding_actor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GpuWorldInstanceKey {
    actor_id: String,
    resource_key: GpuWorldResourceKey,
}

#[derive(Debug)]
struct GpuWorldDrawChunk {
    key: GpuWorldDrawKey,
    texture: GpuWorldTexture,
    vertices: Vec<GpuWorldVertex>,
    mesh_node: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GpuWorldStaticPlanKey {
    model_path: PathBuf,
    outline: bool,
    hide_meshes: Vec<String>,
    hide_materials: Vec<String>,
}

#[derive(Debug, Clone)]
struct GpuWorldStaticDraw {
    resource_key: GpuWorldResourceKey,
    vertices: Arc<Vec<GpuWorldVertex>>,
    indices: Arc<Vec<u32>>,
    vertex_signature: u64,
    texture: Arc<GpuWorldTexture>,
    normal_texture: Arc<GpuWorldTexture>,
    metallic_roughness_texture: Arc<GpuWorldTexture>,
    emissive_texture: Arc<GpuWorldTexture>,
    occlusion_texture: Arc<GpuWorldTexture>,
    mesh_node: Option<usize>,
    bounds: ([f32; 3], [f32; 3]),
}

#[derive(Debug, Clone, Copy, Default)]
struct GpuWorldParams {
    canvas: [f32; 4],
    model: [f32; 4],
    actor: [f32; 4],
    actor_rotation: [f32; 4],
    camera0: [f32; 4],
    camera1: [f32; 4],
    camera2: [f32; 4],
    camera3: [f32; 4],
    style: [f32; 4],
    material0: [f32; 4],
    material1: [f32; 4],
    material2: [f32; 4],
    material3: [f32; 4],
    material4: [f32; 4],
    material5: [f32; 4],
    material6: [f32; 4],
    material7: [f32; 4],
    cel_material0: [f32; 4],
    cel_material1: [f32; 4],
    vegetation: [f32; 4],
    hidden0: [f32; 4],
    hidden1: [f32; 4],
    hidden2: [f32; 4],
    hidden3: [f32; 4],
    hidden4: [f32; 4],
    hidden5: [f32; 4],
    hidden6: [f32; 4],
    hidden7: [f32; 4],
    previous_model: [f32; 4],
    previous_actor: [f32; 4],
    previous_actor_rotation: [f32; 4],
    previous_vegetation: [f32; 4],
    /// x is the previous bone-palette offset; y marks valid object history.
    motion0: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
struct GpuGroundGridParams {
    canvas: [f32; 4],
    camera0: [f32; 4],
    camera1: [f32; 4],
    camera2: [f32; 4],
    camera3: [f32; 4],
    options: [f32; 4],
}

impl GpuGroundGridParams {
    fn from_camera(width: u32, height: u32, camera_view: PerspectiveCameraView) -> Self {
        let width_f = width.max(1) as f32;
        let height_f = height.max(1) as f32;
        Self {
            canvas: [width_f, height_f, width_f * 0.5, height_f * 0.5],
            camera0: [
                camera_view.eye[0],
                camera_view.eye[1],
                camera_view.eye[2],
                camera_view.focal_px,
            ],
            camera1: [
                camera_view.right[0],
                camera_view.right[1],
                camera_view.right[2],
                camera_view.near,
            ],
            camera2: [
                camera_view.up[0],
                camera_view.up[1],
                camera_view.up[2],
                camera_view.far,
            ],
            camera3: [
                camera_view.forward[0],
                camera_view.forward[1],
                camera_view.forward[2],
                0.0,
            ],
            // x: opacity, y/z: distance fade start/end, w: base grid size.
            options: [0.95, 30.0, 70.0, 1.0],
        }
    }

    fn debug_from_camera(width: u32, height: u32, camera_view: PerspectiveCameraView) -> Self {
        let mut params = Self::from_camera(width, height, camera_view);
        // options.y < 0 acts as a debug sentinel in WGSL:
        // high contrast, thicker lines, and no distance fade.
        params.options = [1.0, -1.0, -1.0, 1.0];
        params
    }
}

#[derive(Debug, Clone, Copy)]
struct PerspectiveCameraView {
    eye: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
    focal_px: f32,
    near: f32,
    far: f32,
    aspect: f32,
    /// Focus distance, focal length (mm), f-stop, and maximum blur (px).
    optics: [f32; 4],
}

fn actor_yxz_quaternion(pitch_deg: f32, yaw_deg: f32, roll_deg: f32) -> [f32; 4] {
    fn axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
        let (sine, cosine) = (angle * 0.5).sin_cos();
        [axis[0] * sine, axis[1] * sine, axis[2] * sine, cosine]
    }
    fn multiply(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ]
    }
    let qx = axis_angle([1.0, 0.0, 0.0], pitch_deg.to_radians());
    let qy = axis_angle([0.0, 1.0, 0.0], yaw_deg.to_radians());
    let qz = axis_angle([0.0, 0.0, 1.0], roll_deg.to_radians());
    let value = multiply(qz, multiply(qx, qy));
    let magnitude = value
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if magnitude > 1.0e-8 {
        value.map(|component| component / magnitude)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

#[allow(clippy::too_many_arguments)]
fn build_actor_mesh_gpu_draws(
    graph: &WorldGraph,
    actor: &WorldActor,
    mesh: &GlbMeshData,
    effective_bounds: ([f32; 3], [f32; 3]),
    static_draws: &[GpuWorldStaticDraw],
    width: u32,
    height: u32,
    actor_x: f32,
    actor_y: f32,
    actor_z: f32,
    actor_yaw_deg: f32,
    actor_pitch_deg: f32,
    actor_roll_deg: f32,
    camera_view: PerspectiveCameraView,
    scale: f32,
    opacity: f32,
    time: WorldTime,
    model_path: &Path,
    skinning_strategy_cache: &mut HashMap<SkinningStrategyKey, SkinningMatrixStrategy>,
    material_overrides: &[WorldMaterialTextureOverride],
    cel_textures: &HashMap<String, Arc<GpuWorldTexture>>,
    external_sampled: &HashMap<usize, SampledNodeTrs>,
    constraint_overrides: &HashMap<String, BoneOverride>,
) -> Result<Vec<GpuWorldDraw>, WorldRenderError> {
    let mut cel_slots = std::collections::HashSet::new();
    for binding in actor
        .cel_materials
        .iter()
        .filter(|b| b.cel != crate::render_style::CelMaterialSettings::default())
    {
        if !cel_slots.insert(&binding.material) {
            return Err(WorldRenderError::GpuRender {
                message: format!(
                    "Duplicate MaterialBinding on actor '{}' for '{}'",
                    actor.id, binding.material
                ),
            });
        }
        if binding.material != "*"
            && !mesh
                .materials
                .iter()
                .any(|m| m.name.as_deref() == Some(binding.material.as_str()))
        {
            return Err(WorldRenderError::GpuRender {
                message: format!(
                    "MaterialBinding on actor '{}' references missing material '{}'",
                    actor.id, binding.material
                ),
            });
        }
    }
    for binding in material_overrides
        .iter()
        .filter(|binding| binding.actor_id == actor.id)
    {
        let exists = mesh.materials.iter().any(|material| {
            material
                .name
                .as_deref()
                .is_some_and(|name| name.eq_ignore_ascii_case(&binding.material))
        });
        if !exists {
            return Err(WorldRenderError::GpuRender {
                message: format!(
                    "MaterialBinding on actor '{}' references missing GLB material '{}'",
                    actor.id, binding.material
                ),
            });
        }
    }
    let width_f = width.max(1) as f32;
    let height_f = height.max(1) as f32;
    let (effective_min, effective_max) = effective_bounds;
    let normalize_height = actor.scale_mode.eq_ignore_ascii_case("normalize_height");
    let (model_center_x, model_origin_y, model_center_z, world_scale) = if normalize_height {
        let model_height = (effective_max[1] - effective_min[1]).abs().max(0.001);
        (
            (effective_min[0] + effective_max[0]) * 0.5,
            effective_min[1],
            (effective_min[2] + effective_max[2]) * 0.5,
            scale / model_height,
        )
    } else {
        (0.0, 0.0, 0.0, scale)
    };
    let exposure = actor
        .material
        .as_ref()
        .map(|material| eval_number(&material.exposure, 1.0, time))
        .transpose()?
        .unwrap_or(1.0)
        .clamp(0.05, 8.0);
    let actor_quaternion = actor
        .rotation_quaternion
        .unwrap_or_else(|| actor_yxz_quaternion(actor_pitch_deg, actor_yaw_deg, actor_roll_deg));
    let hidden_joints = camera_hidden_joint_slots(graph, actor, mesh);
    let params = GpuWorldParams {
        canvas: [width_f, height_f, width_f * 0.5, height_f * 0.5],
        model: [model_center_x, model_origin_y, model_center_z, world_scale],
        actor: [actor_x, actor_y, actor_z, actor_yaw_deg.to_radians()],
        actor_rotation: actor_quaternion,
        camera0: [
            camera_view.eye[0],
            camera_view.eye[1],
            camera_view.eye[2],
            camera_view.focal_px,
        ],
        camera1: [
            camera_view.right[0],
            camera_view.right[1],
            camera_view.right[2],
            camera_view.near,
        ],
        camera2: [
            camera_view.up[0],
            camera_view.up[1],
            camera_view.up[2],
            camera_view.far,
        ],
        camera3: [
            camera_view.forward[0],
            camera_view.forward[1],
            camera_view.forward[2],
            0.0,
        ],
        style: [
            opacity.clamp(0.0, 1.0),
            actor_material_light_mix(actor),
            exposure,
            if camera_hidden_bones_hide_whole_actor(&actor.camera_hidden_bones) {
                1.0
            } else {
                0.0
            },
        ],
        material0: [1.0, 1.0, 1.0, 1.0],
        material1: [0.0, 0.0, 0.0, 0.0],
        material2: [1.0, 1.0, 1.0, 0.0],
        material3: [0.0, 0.0, 1.0, 0.0],
        material4: [1.0; 4],
        material5: [1.0, 1.0, 0.0, 0.0],
        // material6: transmission, IOR, optical thickness, attenuation distance.
        material6: [0.0, 1.5, 0.0, 1_000_000.0],
        // material7: attenuation RGB; w carries a positive alpha-mask cutoff.
        material7: [1.0, 1.0, 1.0, 0.0],
        cel_material0: [-1.0, 0.0, 0.0, 0.0],
        cel_material1: [0.0; 4],
        // Vegetation wind is gated per actor; all existing asset paths retain zero deformation.
        vegetation: actor.vegetation.as_ref().map_or([0.0; 4], |vegetation| {
            [
                if vegetation.wind { 1.0 } else { 0.0 },
                vegetation.height,
                (vegetation.seed % 65_521) as f32 * 0.017,
                time.time_sec(),
            ]
        }),
        hidden0: hidden_joints[0],
        hidden1: hidden_joints[1],
        hidden2: hidden_joints[2],
        hidden3: hidden_joints[3],
        hidden4: hidden_joints[4],
        hidden5: hidden_joints[5],
        hidden6: hidden_joints[6],
        hidden7: hidden_joints[7],
        previous_model: [0.0; 4],
        previous_actor: [0.0; 4],
        previous_actor_rotation: [0.0; 4],
        previous_vegetation: [0.0; 4],
        motion0: [0.0; 4],
    };
    let mut draws = Vec::<GpuWorldDraw>::with_capacity(static_draws.len());
    let joint_frame = prepare_actor_joint_frame(
        graph,
        actor,
        mesh,
        time,
        external_sampled,
        constraint_overrides,
    )?;
    // A GLB commonly splits one skinned mesh into several material draws. Bone
    // matrices depend on the actor and mesh node, not on the material, so do
    // not resample/retarget the entire skeleton once per material primitive.
    let mut bone_matrices_by_node =
        HashMap::<Option<usize>, Vec<[f32; 16]>>::with_capacity(static_draws.len().min(8));
    for static_draw in static_draws {
        if static_draw.vertices.is_empty() {
            continue;
        }
        if actor.terrain.is_some()
            && !terrain_chunk_visible(
                static_draw.bounds,
                [model_center_x, model_origin_y, model_center_z],
                world_scale,
                actor_quaternion,
                [actor_x, actor_y, actor_z],
                camera_view,
                width_f,
                height_f,
            )
        {
            continue;
        }
        let bone_matrices =
            if let Some(matrices) = bone_matrices_by_node.get(&static_draw.mesh_node) {
                matrices.clone()
            } else {
                let matrices = actor_joint_matrices(
                    mesh,
                    model_path,
                    static_draw.mesh_node,
                    &joint_frame,
                    skinning_strategy_cache,
                );
                bone_matrices_by_node.insert(static_draw.mesh_node, matrices.clone());
                matrices
            };
        let material = static_draw
            .resource_key
            .draw_key
            .material
            .and_then(|index| mesh.materials.get(index));
        let mut draw_params = params;
        // Exact material slots take precedence over an explicit wildcard.
        let cel_binding = actor
            .cel_materials
            .iter()
            .find(|b| material.and_then(|m| m.name.as_deref()) == Some(b.material.as_str()))
            .or_else(|| actor.cel_materials.iter().find(|b| b.material == "*"));
        if let Some(binding) = cel_binding {
            let c = &binding.cel;
            draw_params.cel_material0 = [
                c.outline_width.unwrap_or(-1.0),
                match c.role.as_deref() {
                    Some("skin") => 1.0,
                    Some("hair") => 2.0,
                    Some("face") => 3.0,
                    _ => 0.0,
                },
                c.hair_highlight.unwrap_or(0.25),
                if c.control_map.is_some() { 1.0 } else { 0.0 },
            ];
            if let Some(color) = &c.shadow_color {
                let rgb = crate::render_style::cel_color(color);
                draw_params.cel_material1 = [rgb[0], rgb[1], rgb[2], 1.0];
            }
        }
        if let Some(primitive) = actor.primitive.as_ref() {
            let offset = crate::world::primitive::primitive_material_seed_offset(primitive);
            let material = primitive.material_definition.as_ref();
            let angle = material.map_or(0.0, |value| value.texture_rotation.to_radians());
            let (sin, cos) = angle.sin_cos();
            draw_params.material3 = [offset[0], offset[1], cos, sin];
            draw_params.material4 = material.map_or(primitive.color, |value| {
                std::array::from_fn(|index| value.base_color[index] * primitive.color[index])
            });
            draw_params.material5 = material.map_or([1.0, 1.0, 0.0, 0.0], |value| {
                [
                    value.texture_scale[0],
                    value.texture_scale[1],
                    value.texture_offset[0],
                    value.texture_offset[1],
                ]
            });
        }
        if let Some(material) = material {
            draw_params.material0 = [
                material.metallic_factor.clamp(0.0, 1.0),
                material.roughness_factor.clamp(0.04, 1.0),
                material.normal_scale.clamp(0.0, 4.0),
                material.specular_factor.clamp(0.0, 2.0),
            ];
            draw_params.material1 = [
                material.emissive_factor[0].max(0.0) * material.emissive_strength.max(0.0),
                material.emissive_factor[1].max(0.0) * material.emissive_strength.max(0.0),
                material.emissive_factor[2].max(0.0) * material.emissive_strength.max(0.0),
                if material.unlit { 1.0 } else { 0.0 },
            ];
            draw_params.material2 = [
                material.specular_color_factor[0].clamp(0.0, 2.0),
                material.specular_color_factor[1].clamp(0.0, 2.0),
                material.specular_color_factor[2].clamp(0.0, 2.0),
                0.0,
            ];
            draw_params.material6 = [
                material.transmission_factor.clamp(0.0, 1.0),
                material.ior.clamp(1.0, 3.0),
                material.thickness_factor.max(0.0),
                material.attenuation_distance.max(0.0001),
            ];
            draw_params.material7 = [
                material.attenuation_color[0].clamp(0.0001, 1.0),
                material.attenuation_color[1].clamp(0.0001, 1.0),
                material.attenuation_color[2].clamp(0.0001, 1.0),
                if material.alpha_mode == GlbAlphaMode::Mask {
                    material.alpha_cutoff.clamp(0.0001, 1.0)
                } else {
                    0.0
                },
            ];
            // Generated foliage atlases already contain photographic/baked colour.
            // Preserve it for the thin alpha-mask cards while woody submeshes stay PBR.
            if actor.vegetation.is_some() && material.alpha_mode == GlbAlphaMode::Mask {
                draw_params.material1[3] = 1.0;
                draw_params.material2[3] = 1.0;
            }
        }
        let texture_override = material.and_then(|material| {
            let material_name = material.name.as_deref()?;
            material_overrides.iter().find(|binding| {
                binding.actor_id == actor.id && binding.material.eq_ignore_ascii_case(material_name)
            })
        });
        let texture = texture_override.map_or_else(
            || Arc::clone(&static_draw.texture),
            |binding| Arc::clone(&binding.texture),
        );
        if texture_override.is_some() {
            // A bound Scene is a display surface: preserve the authored UI rather than
            // allowing the room lighting to turn it grey or black.
            draw_params.material1[3] = 1.0;
            draw_params.material2[3] = 1.0;
        }
        let mut resource_key = static_draw.resource_key.clone();
        if texture_override.is_some() || cel_binding.is_some_and(|b| b.cel.control_map.is_some()) {
            resource_key.binding_actor = Some(actor.id.clone());
        }
        let phase = gpu_world_material_phase(material);
        let depth_write = gpu_world_material_depth_write(material, phase);
        let relative_to_camera = [
            actor_x - camera_view.eye[0],
            actor_y - camera_view.eye[1],
            actor_z - camera_view.eye[2],
        ];
        draws.push(GpuWorldDraw {
            instance_key: GpuWorldInstanceKey {
                actor_id: actor.id.clone(),
                resource_key: resource_key.clone(),
            },
            resource_key,
            vertices: Arc::clone(&static_draw.vertices),
            indices: Arc::clone(&static_draw.indices),
            vertex_signature: static_draw.vertex_signature,
            texture,
            normal_texture: Arc::clone(&static_draw.normal_texture),
            metallic_roughness_texture: Arc::clone(&static_draw.metallic_roughness_texture),
            emissive_texture: Arc::clone(&static_draw.emissive_texture),
            occlusion_texture: Arc::clone(&static_draw.occlusion_texture),
            cel_texture: cel_binding
                .and_then(|b| cel_textures.get(&b.material))
                .cloned()
                .unwrap_or_else(|| Arc::new(GpuWorldTexture::new(1, 1, vec![255u8; 4]))),
            bone_matrices: bone_matrices.clone(),
            params: draw_params,
            phase,
            depth_write,
            sort_priority: material.map_or(0, |material| material.sort_priority),
            camera_depth: dot3(relative_to_camera, camera_view.forward),
        });
    }
    Ok(draws)
}

fn actor_material_light_mix(actor: &WorldActor) -> f32 {
    match actor.material.as_ref().map(|material| &material.style) {
        Some(WorldMaterialStyle::Pbr) | None => 1.0,
        Some(WorldMaterialStyle::Toon | WorldMaterialStyle::Unlit) => 0.0,
    }
}

fn build_actor_mesh_gpu_static_draws(
    actor: &WorldActor,
    mesh: &GlbMeshData,
    model_path: &Path,
) -> Vec<GpuWorldStaticDraw> {
    let primitive_material_uniform = actor.primitive.is_some();
    let fallback = if actor
        .material
        .as_ref()
        .is_some_and(|material| material.outline)
    {
        [93, 126, 178]
    } else {
        [111, 145, 190]
    };
    let mut chunks = Vec::<GpuWorldDrawChunk>::new();
    let static_node_matrices = if mesh.skin.is_none() {
        Some(global_node_matrices_with_sampled(
            mesh,
            &HashMap::new(),
            &HashMap::new(),
        ))
    } else {
        None
    };
    for triangle in &mesh.triangles {
        if actor_hides_triangle(actor, mesh, triangle) {
            continue;
        }
        let indices = triangle.indices;
        let mesh_node = triangle.mesh_node;
        let uvs = [
            mesh.texcoords.get(indices[0] as usize).copied().flatten(),
            mesh.texcoords.get(indices[1] as usize).copied().flatten(),
            mesh.texcoords.get(indices[2] as usize).copied().flatten(),
        ];
        let (key, color_factor) = gpu_triangle_material_factor(
            mesh,
            triangle.material,
            triangle.mesh,
            mesh_node,
            1.0,
            1.0,
            primitive_material_uniform,
        );
        let chunk_index = if let Some(index) = chunks.iter().position(|chunk| chunk.key == key) {
            index
        } else {
            let texture = gpu_texture_for_material(
                mesh,
                triangle.material,
                key,
                fallback,
                primitive_material_uniform,
            );
            let index = chunks.len();
            chunks.push(GpuWorldDrawChunk {
                key,
                texture,
                vertices: Vec::new(),
                mesh_node,
            });
            index
        };
        let chunk = chunks
            .get_mut(chunk_index)
            .expect("GPU world draw chunk inserted before vertex push");
        let indices = triangle.indices;
        let node_matrix = triangle
            .mesh_node
            .and_then(|index| static_node_matrices.as_ref()?.get(index))
            .copied();
        let transformed_triangle = indices.map(|index| {
            let position = mesh
                .positions
                .get(index as usize)
                .copied()
                .unwrap_or([0.0; 3]);
            node_matrix
                .map(|matrix| transform_point_mat4(matrix, position))
                .unwrap_or(position)
        });
        let fallback_normal = triangle_normal_from_points(transformed_triangle)
            .or_else(|| triangle_normal(mesh, indices))
            .unwrap_or([0.0, 0.0, 1.0]);
        let (mut tangent, mut bitangent) =
            triangle_tangent_frame(mesh, indices, uvs, fallback_normal);
        if let Some(matrix) = node_matrix {
            tangent = normalize3(transform_vector_mat4(matrix, tangent));
            bitangent = normalize3(transform_vector_mat4(matrix, bitangent));
        }
        for i in 0..3 {
            let vertex_index = indices[i] as usize;
            let Some(mut position) = mesh.positions.get(vertex_index).copied() else {
                continue;
            };
            let mut normal = mesh
                .normals
                .get(vertex_index)
                .copied()
                .flatten()
                .unwrap_or(fallback_normal);
            if let Some(matrix) = node_matrix {
                position = transform_point_mat4(matrix, position);
                normal = normalize3(transform_vector_mat4(matrix, normal));
            }
            let joints = mesh
                .joints
                .get(vertex_index)
                .copied()
                .flatten()
                .map(|joints| {
                    [
                        joints[0] as f32,
                        joints[1] as f32,
                        joints[2] as f32,
                        joints[3] as f32,
                    ]
                })
                .unwrap_or([0.0, 0.0, 0.0, 0.0]);
            let weights = mesh
                .weights
                .get(vertex_index)
                .copied()
                .flatten()
                .unwrap_or([1.0, 0.0, 0.0, 0.0]);
            let vertex_color = mesh
                .colors
                .get(vertex_index)
                .copied()
                .flatten()
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
            // glTF COLOR_0 is linear; the shader decodes the combined legacy
            // color factor, so encode imported vertex RGB before that decode.
            let vertex_color = if primitive_material_uniform {
                vertex_color
            } else {
                [
                    vertex_color[0].max(0.0).powf(1.0 / 2.2),
                    vertex_color[1].max(0.0).powf(1.0 / 2.2),
                    vertex_color[2].max(0.0).powf(1.0 / 2.2),
                    vertex_color[3],
                ]
            };
            let color = [
                color_factor[0] * vertex_color[0],
                color_factor[1] * vertex_color[1],
                color_factor[2] * vertex_color[2],
                color_factor[3] * vertex_color[3],
            ];
            chunk.vertices.push(GpuWorldVertex {
                outline_normal: normal,
                position,
                normal,
                tangent,
                bitangent,
                joints,
                weights,
                uv: uvs[i].unwrap_or([0.0, 0.0]),
                color,
            });
        }
    }
    let mut draws = Vec::<GpuWorldStaticDraw>::with_capacity(chunks.len());
    for mut chunk in chunks {
        if chunk.vertices.is_empty() {
            continue;
        }
        // Only weld outline normals within the same mesh/material and skin weights.
        // Lighting normals are untouched; this cache is built once per static mesh.
        let key = |v: &GpuWorldVertex| {
            (
                v.position.map(f32::to_bits),
                v.joints.map(f32::to_bits),
                v.weights.map(f32::to_bits),
            )
        };
        let mut normals = HashMap::<_, [f32; 3]>::new();
        for v in &chunk.vertices {
            let n = normals.entry(key(v)).or_default();
            for (axis, value) in n.iter_mut().enumerate() {
                *value += v.normal[axis];
            }
        }
        for v in &mut chunk.vertices {
            if let Some(n) = normals.get(&key(v)) {
                if dot3(*n, *n) > 0.000001 {
                    v.outline_normal = normalize3(*n);
                }
            }
        }
        let resource_key = GpuWorldResourceKey {
            model_path: model_path.to_path_buf(),
            draw_key: chunk.key,
            binding_actor: None,
        };
        let (indexed_vertices, indexed_indices) = index_gpu_world_vertices(&chunk.vertices);
        let vertices = Arc::new(indexed_vertices);
        let indices = Arc::new(indexed_indices);
        let bounds = gpu_world_vertices_bounds(vertices.as_ref());
        let vertex_signature = gpu_world_geometry_signature(vertices.as_ref(), indices.as_ref());
        draws.push(GpuWorldStaticDraw {
            resource_key,
            vertices,
            indices,
            vertex_signature,
            texture: Arc::new(chunk.texture),
            normal_texture: Arc::new(gpu_texture_for_index(
                mesh,
                chunk
                    .key
                    .material
                    .and_then(|index| mesh.materials.get(index))
                    .and_then(|material| material.normal_texture),
                [128, 128, 255, 255],
            )),
            metallic_roughness_texture: Arc::new(gpu_texture_for_index(
                mesh,
                chunk
                    .key
                    .material
                    .and_then(|index| mesh.materials.get(index))
                    .and_then(|material| material.metallic_roughness_texture),
                [255, 255, 255, 255],
            )),
            emissive_texture: Arc::new(gpu_texture_for_index(
                mesh,
                chunk
                    .key
                    .material
                    .and_then(|index| mesh.materials.get(index))
                    .and_then(|material| material.emissive_texture),
                [255, 255, 255, 255],
            )),
            occlusion_texture: Arc::new(gpu_occlusion_texture(
                mesh,
                chunk
                    .key
                    .material
                    .and_then(|index| mesh.materials.get(index))
                    .and_then(|material| material.occlusion_texture),
                chunk
                    .key
                    .material
                    .and_then(|i| mesh.materials.get(i))
                    .map_or(1.0, |m| m.occlusion_strength),
            )),
            mesh_node: chunk.mesh_node,
            bounds,
        });
    }
    draws
}

// Resolve AO strength once while retaining a neutral white fallback.
fn gpu_occlusion_texture(
    mesh: &GlbMeshData,
    index: Option<usize>,
    strength: f32,
) -> GpuWorldTexture {
    let mut texture = gpu_texture_for_index(mesh, index, [255; 4]);
    let mut pixels = texture.rgba.as_ref().clone();
    for pixel in pixels.chunks_exact_mut(4) {
        pixel[0] = ((1.0 - strength.clamp(0.0, 1.0) * (1.0 - pixel[0] as f32 / 255.0)) * 255.0)
            .round() as u8;
    }
    texture.signature ^= u64::from(strength.to_bits()).rotate_left(17);
    texture.rgba = Arc::new(pixels);
    texture
}

fn gpu_world_vertices_bounds(vertices: &[GpuWorldVertex]) -> ([f32; 3], [f32; 3]) {
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(vertex.position[axis]);
            maximum[axis] = maximum[axis].max(vertex.position[axis]);
        }
    }
    if vertices.is_empty() {
        ([0.0; 3], [0.0; 3])
    } else {
        (minimum, maximum)
    }
}

#[allow(clippy::too_many_arguments)]
fn terrain_chunk_visible(
    bounds: ([f32; 3], [f32; 3]),
    model_origin: [f32; 3],
    scale: f32,
    rotation: [f32; 4],
    actor_position: [f32; 3],
    camera: PerspectiveCameraView,
    viewport_width: f32,
    viewport_height: f32,
) -> bool {
    let (minimum, maximum) = bounds;
    let local_center = std::array::from_fn(|axis| {
        ((minimum[axis] + maximum[axis]) * 0.5 - model_origin[axis]) * scale
    });
    let rotated_center = quat_rotate_vec3(rotation, local_center);
    let center: [f32; 3] = std::array::from_fn(|axis| actor_position[axis] + rotated_center[axis]);
    let half_extent = std::array::from_fn::<_, 3, _>(|axis| {
        (maximum[axis] - minimum[axis]).abs() * 0.5 * scale.abs()
    });
    let radius = dot3(half_extent, half_extent).sqrt();
    let relative = std::array::from_fn::<_, 3, _>(|axis| center[axis] - camera.eye[axis]);
    let depth = dot3(relative, camera.forward);
    if depth + radius < camera.near || depth - radius > camera.far {
        return false;
    }
    let visible_depth = depth.max(camera.near);
    let half_width = visible_depth * viewport_width * 0.5 / camera.focal_px.max(1.0);
    let half_height = visible_depth * viewport_height * 0.5 / camera.focal_px.max(1.0);
    dot3(relative, camera.right).abs() <= half_width + radius
        && dot3(relative, camera.up).abs() <= half_height + radius
}

fn effective_mesh_bounds(mesh: &GlbMeshData) -> ([f32; 3], [f32; 3]) {
    if mesh.skin.is_some() || mesh.triangles.is_empty() {
        return (mesh.bounds_min, mesh.bounds_max);
    }
    let matrices = global_node_matrices_with_sampled(mesh, &HashMap::new(), &HashMap::new());
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut found = false;
    for triangle in &mesh.triangles {
        let matrix = triangle
            .mesh_node
            .and_then(|index| matrices.get(index))
            .copied();
        for index in triangle.indices {
            let Some(mut point) = mesh.positions.get(index as usize).copied() else {
                continue;
            };
            if let Some(matrix) = matrix {
                point = transform_point_mat4(matrix, point);
            }
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
            found = true;
        }
    }
    if found {
        (min, max)
    } else {
        (mesh.bounds_min, mesh.bounds_max)
    }
}

fn transform_point_mat4(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
    ]
}

fn transform_vector_mat4(matrix: [f32; 16], vector: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2],
        matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2],
        matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2],
    ]
}

fn triangle_normal_from_points(points: [[f32; 3]; 3]) -> Option<[f32; 3]> {
    let ab = [
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
        points[1][2] - points[0][2],
    ];
    let ac = [
        points[2][0] - points[0][0],
        points[2][1] - points[0][1],
        points[2][2] - points[0][2],
    ];
    let normal = cross3(ab, ac);
    let normalized = normalize3(normal);
    normalized
        .iter()
        .all(|value| value.is_finite())
        .then_some(normalized)
}

fn triangle_normal(mesh: &GlbMeshData, indices: [u32; 3]) -> Option<[f32; 3]> {
    let a = mesh.positions.get(indices[0] as usize).copied()?;
    let b = mesh.positions.get(indices[1] as usize).copied()?;
    let c = mesh.positions.get(indices[2] as usize).copied()?;
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    normalize_vec3([
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ])
}

fn triangle_tangent_frame(
    mesh: &GlbMeshData,
    indices: [u32; 3],
    uvs: [Option<[f32; 2]>; 3],
    normal: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let positions = indices.map(|index| mesh.positions.get(index as usize).copied());
    if let ([Some(p0), Some(p1), Some(p2)], [Some(uv0), Some(uv1), Some(uv2)]) = (positions, uvs) {
        let edge1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let edge2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
        let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];
        let determinant = duv1[0] * duv2[1] - duv1[1] * duv2[0];
        if determinant.abs() > 0.000001 {
            let inverse = determinant.recip();
            let tangent = normalize_vec3([
                (edge1[0] * duv2[1] - edge2[0] * duv1[1]) * inverse,
                (edge1[1] * duv2[1] - edge2[1] * duv1[1]) * inverse,
                (edge1[2] * duv2[1] - edge2[2] * duv1[1]) * inverse,
            ]);
            let bitangent = normalize_vec3([
                (edge2[0] * duv1[0] - edge1[0] * duv2[0]) * inverse,
                (edge2[1] * duv1[0] - edge1[1] * duv2[0]) * inverse,
                (edge2[2] * duv1[0] - edge1[2] * duv2[0]) * inverse,
            ]);
            if let (Some(tangent), Some(bitangent)) = (tangent, bitangent) {
                return (tangent, bitangent);
            }
        }
    }
    let reference = if normal[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize_vec3(cross3(reference, normal)).unwrap_or([1.0, 0.0, 0.0]);
    let bitangent = normalize_vec3(cross3(normal, tangent)).unwrap_or([0.0, 1.0, 0.0]);
    (tangent, bitangent)
}

fn normalize_vec3(value: [f32; 3]) -> Option<[f32; 3]> {
    let len = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if len <= f32::EPSILON || !len.is_finite() {
        None
    } else {
        Some([value[0] / len, value[1] / len, value[2] / len])
    }
}

fn actor_hides_triangle(actor: &WorldActor, mesh: &GlbMeshData, triangle: &GlbTriangle) -> bool {
    if actor.hide_materials.iter().any(|name| {
        triangle
            .material
            .and_then(|index| mesh.materials.get(index))
            .and_then(|material| material.name.as_deref())
            .is_some_and(|material_name| material_name.eq_ignore_ascii_case(name))
    }) {
        return true;
    }
    if actor.hide_meshes.iter().any(|name| {
        let mesh_name = triangle
            .mesh
            .and_then(|index| mesh.mesh_names.get(index))
            .and_then(|name| name.as_deref());
        let node_name = triangle
            .mesh_node
            .and_then(|index| mesh.nodes.get(index))
            .and_then(|node| node.name.as_deref());
        mesh_name.is_some_and(|mesh_name| mesh_name.eq_ignore_ascii_case(name))
            || node_name.is_some_and(|node_name| node_name.eq_ignore_ascii_case(name))
    }) {
        return true;
    }
    false
}

fn gpu_triangle_material_factor(
    mesh: &GlbMeshData,
    material_index: Option<usize>,
    mesh_index: Option<usize>,
    mesh_node: Option<usize>,
    shade: f32,
    opacity: f32,
    primitive_material_uniform: bool,
) -> (GpuWorldDrawKey, [f32; 4]) {
    if let Some(material) = material_index.and_then(|index| mesh.materials.get(index)) {
        if let Some(texture_index) = material.base_color_texture {
            if mesh
                .textures
                .get(texture_index)
                .and_then(Option::as_ref)
                .is_some()
            {
                return (
                    GpuWorldDrawKey {
                        material: material_index,
                        texture: Some(texture_index),
                        mesh: mesh_index,
                        mesh_node,
                    },
                    if primitive_material_uniform {
                        [shade, shade, shade, opacity.clamp(0.0, 1.0)]
                    } else {
                        [
                            material.base_color_factor[0].clamp(0.0, 1.0) * shade,
                            material.base_color_factor[1].clamp(0.0, 1.0) * shade,
                            material.base_color_factor[2].clamp(0.0, 1.0) * shade,
                            material.base_color_factor[3].clamp(0.0, 1.0) * opacity.clamp(0.0, 1.0),
                        ]
                    },
                );
            }
        }
    }

    (
        GpuWorldDrawKey {
            material: material_index,
            texture: None,
            mesh: mesh_index,
            mesh_node,
        },
        [shade, shade, shade, opacity.clamp(0.0, 1.0)],
    )
}

fn gpu_texture_for_material(
    mesh: &GlbMeshData,
    material_index: Option<usize>,
    key: GpuWorldDrawKey,
    fallback: [u8; 3],
    primitive_material_uniform: bool,
) -> GpuWorldTexture {
    if let Some(texture_index) = key.texture {
        if let Some(texture) = mesh.textures.get(texture_index).and_then(Option::as_ref) {
            return GpuWorldTexture::new(texture.width, texture.height, texture.rgba.clone());
        }
    }

    let color = if primitive_material_uniform {
        Rgba([255; 4])
    } else {
        material_color(mesh, material_index, fallback, 1.0, 1.0)
    };
    GpuWorldTexture::new(1, 1, vec![color[0], color[1], color[2], color[3]])
}

fn gpu_texture_for_index(
    mesh: &GlbMeshData,
    texture_index: Option<usize>,
    fallback: [u8; 4],
) -> GpuWorldTexture {
    if let Some(texture) = texture_index
        .and_then(|index| mesh.textures.get(index))
        .and_then(Option::as_ref)
    {
        return GpuWorldTexture::new(texture.width, texture.height, texture.rgba.clone());
    }
    GpuWorldTexture::new(1, 1, fallback.to_vec())
}

type TexturedTriangleSource<'a> = (&'a GlbTextureData, [f32; 4], [[f32; 2]; 3]);

fn textured_triangle_source(
    mesh: &GlbMeshData,
    material_index: Option<usize>,
    uvs: [Option<[f32; 2]>; 3],
) -> Option<TexturedTriangleSource<'_>> {
    let material = material_index.and_then(|index| mesh.materials.get(index))?;
    let texture_index = material.base_color_texture?;
    let texture = mesh.textures.get(texture_index)?.as_ref()?;
    let uvs = [uvs[0]?, uvs[1]?, uvs[2]?];
    Some((texture, material.base_color_factor, uvs))
}

fn sampled_texture_alpha(texture: &GlbTextureData, uv: [f32; 2], material_factor: [f32; 4]) -> f32 {
    if texture.width == 0 || texture.height == 0 || texture.rgba.len() < 4 {
        return 0.0;
    }
    let u = uv[0].clamp(0.0, 1.0);
    let v = uv[1].clamp(0.0, 1.0);
    let tx = (u * texture.width.saturating_sub(1) as f32)
        .round()
        .clamp(0.0, texture.width.saturating_sub(1) as f32) as u32;
    let ty = (v * texture.height.saturating_sub(1) as f32)
        .round()
        .clamp(0.0, texture.height.saturating_sub(1) as f32) as u32;
    let offset = ((ty * texture.width + tx) as usize).saturating_mul(4);
    texture
        .rgba
        .get(offset + 3)
        .map_or(0.0, |alpha| *alpha as f32 / 255.0)
        * material_factor[3].clamp(0.0, 1.0)
}

fn material_color(
    mesh: &GlbMeshData,
    material_index: Option<usize>,
    fallback: [u8; 3],
    shade: f32,
    opacity: f32,
) -> Rgba<u8> {
    let mut rgb = [
        fallback[0] as f32 / 255.0,
        fallback[1] as f32 / 255.0,
        fallback[2] as f32 / 255.0,
    ];
    let mut alpha = 220.0 / 255.0;
    if let Some(material) = material_index.and_then(|index| mesh.materials.get(index)) {
        let factor = material.base_color_factor;
        let has_visible_tint = (factor[0] - 1.0).abs() > 0.001
            || (factor[1] - 1.0).abs() > 0.001
            || (factor[2] - 1.0).abs() > 0.001;
        if has_visible_tint {
            rgb = [
                factor[0].clamp(0.0, 1.0),
                factor[1].clamp(0.0, 1.0),
                factor[2].clamp(0.0, 1.0),
            ];
        }
        alpha *= factor[3].clamp(0.0, 1.0);
    }
    Rgba([
        (rgb[0] * shade * 255.0).round().clamp(0.0, 255.0) as u8,
        (rgb[1] * shade * 255.0).round().clamp(0.0, 255.0) as u8,
        (rgb[2] * shade * 255.0).round().clamp(0.0, 255.0) as u8,
        (alpha * opacity.clamp(0.0, 1.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8,
    ])
}

#[allow(clippy::too_many_arguments)]
fn draw_actor_placeholder(
    canvas: &mut RgbaImage,
    actor: &WorldActor,
    has_skin: bool,
    world_x: f32,
    world_y: f32,
    view_yaw_deg: f32,
    camera_pitch_deg: f32,
    fov: f32,
    distance: f32,
    scale: f32,
    opacity: f32,
) {
    let width = canvas.width() as f32;
    let height = canvas.height() as f32;
    let px_per_world = (height / distance) * (35.0 / fov).clamp(0.35, 2.5);
    let cx = width * 0.5 + world_x * px_per_world;
    let ground_y = height * 0.78 - world_y * px_per_world;
    let body_h = (height * 0.34 * scale).clamp(32.0, height * 0.9);
    let head_r = body_h * 0.115;
    let yaw = view_yaw_deg.to_radians();
    let facing_width = yaw.cos().abs().mul_add(0.72, 0.28);
    let body_w = body_h * 0.20 * facing_width;
    let outline = Rgba([35, 45, 58, (255.0 * opacity) as u8]);
    let skin = if has_skin {
        Rgba([248, 229, 218, (235.0 * opacity) as u8])
    } else {
        Rgba([210, 218, 225, (235.0 * opacity) as u8])
    };
    let cloth = Rgba([94, 132, 190, (220.0 * opacity) as u8]);
    let shadow = Rgba([30, 45, 45, (70.0 * opacity) as u8]);

    fill_ellipse(
        canvas,
        cx,
        ground_y + body_h * 0.035,
        body_w * 1.5,
        body_h * 0.035,
        shadow,
    );
    fill_ellipse(
        canvas,
        cx,
        ground_y - body_h * 0.88,
        head_r * facing_width.max(0.55),
        head_r,
        outline,
    );
    fill_ellipse(
        canvas,
        cx,
        ground_y - body_h * 0.88,
        (head_r - 3.0).max(1.0) * facing_width.max(0.55),
        (head_r - 3.0).max(1.0),
        skin,
    );
    fill_ellipse(
        canvas,
        cx,
        ground_y - body_h * 0.48,
        body_w,
        body_h * 0.33,
        outline,
    );
    fill_ellipse(
        canvas,
        cx,
        ground_y - body_h * 0.48,
        (body_w - 3.0).max(1.0),
        (body_h * 0.33 - 3.0).max(1.0),
        cloth,
    );

    let nose_x = cx + yaw.sin() * head_r * 0.55 * facing_width.max(0.4);
    let nose_y = ground_y - body_h * 0.88 - camera_pitch_deg.to_radians().sin() * head_r * 0.3;
    draw_line(
        canvas,
        cx,
        ground_y - body_h * 0.88,
        nose_x,
        nose_y,
        outline,
        3.0,
    );

    let label_hint = if actor_play_layers(actor).any(|play| play.clip.is_some()) {
        Rgba([255, 225, 120, (190.0 * opacity) as u8])
    } else {
        Rgba([180, 210, 255, (170.0 * opacity) as u8])
    };
    fill_ellipse(
        canvas,
        cx + body_w * 0.7,
        ground_y - body_h * 0.78,
        5.0,
        5.0,
        label_hint,
    );
}

fn composite_background(
    canvas: &mut RgbaImage,
    image: &RgbaImage,
    fit: &WorldBackgroundFit,
    opacity: f32,
) {
    let cw = canvas.width();
    let ch = canvas.height();
    let iw = image.width().max(1);
    let ih = image.height().max(1);
    let (scaled_w, scaled_h) = match fit {
        WorldBackgroundFit::Stretch => (cw, ch),
        WorldBackgroundFit::Contain => {
            let scale = (cw as f32 / iw as f32).min(ch as f32 / ih as f32);
            (
                (iw as f32 * scale).round() as u32,
                (ih as f32 * scale).round() as u32,
            )
        }
        WorldBackgroundFit::Cover => {
            let scale = (cw as f32 / iw as f32).max(ch as f32 / ih as f32);
            (
                (iw as f32 * scale).round() as u32,
                (ih as f32 * scale).round() as u32,
            )
        }
    };
    let scaled = imageops::resize(
        image,
        scaled_w.max(1),
        scaled_h.max(1),
        imageops::FilterType::Triangle,
    );
    let offset_x = ((cw as i64 - scaled.width() as i64) / 2)
        .min(0)
        .unsigned_abs() as u32;
    let offset_y = ((ch as i64 - scaled.height() as i64) / 2)
        .min(0)
        .unsigned_abs() as u32;
    let crop_w = cw.min(scaled.width().saturating_sub(offset_x));
    let crop_h = ch.min(scaled.height().saturating_sub(offset_y));
    let cropped =
        imageops::crop_imm(&scaled, offset_x, offset_y, crop_w.max(1), crop_h.max(1)).to_image();
    let paste_x = ((cw as i64 - cropped.width() as i64) / 2).max(0) as u32;
    let paste_y = ((ch as i64 - cropped.height() as i64) / 2).max(0) as u32;
    blend_image(canvas, &cropped, paste_x, paste_y, opacity);
}

fn blend_image(canvas: &mut RgbaImage, image: &RgbaImage, x: u32, y: u32, opacity: f32) {
    for iy in 0..image.height() {
        for ix in 0..image.width() {
            let dx = x + ix;
            let dy = y + iy;
            if dx >= canvas.width() || dy >= canvas.height() {
                continue;
            }
            let src = *image.get_pixel(ix, iy);
            blend_pixel(canvas, dx, dy, with_opacity(src, opacity));
        }
    }
}

fn blend_image_i32(canvas: &mut RgbaImage, image: &RgbaImage, x: i32, y: i32, opacity: f32) {
    for iy in 0..image.height() as i32 {
        for ix in 0..image.width() as i32 {
            let dx = x + ix;
            let dy = y + iy;
            if dx < 0 || dy < 0 {
                continue;
            }
            let dx = dx as u32;
            let dy = dy as u32;
            if dx >= canvas.width() || dy >= canvas.height() {
                continue;
            }
            let src = *image.get_pixel(ix as u32, iy as u32);
            blend_pixel(canvas, dx, dy, with_opacity(src, opacity));
        }
    }
}

fn fill(canvas: &mut RgbaImage, color: Rgba<u8>) {
    for pixel in canvas.pixels_mut() {
        *pixel = color;
    }
}

fn fill_triangle(canvas: &mut RgbaImage, points: [[f32; 2]; 3], color: Rgba<u8>) {
    let min_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(canvas.width() as f32 - 1.0)
        .max(0.0) as u32;
    let min_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(canvas.height() as f32 - 1.0)
        .max(0.0) as u32;
    let area = edge(points[0], points[1], points[2]);
    if area.abs() <= 0.0001 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge(points[1], points[2], point);
            let w1 = edge(points[2], points[0], point);
            let w2 = edge(points[0], points[1], point);
            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0) {
                blend_pixel(canvas, x, y, color);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_textured_triangle(
    canvas: &mut RgbaImage,
    points: [[f32; 2]; 3],
    uvs: [[f32; 2]; 3],
    texture_width: u32,
    texture_height: u32,
    texture_rgba: &[u8],
    material_factor: [f32; 4],
    shade: f32,
    opacity: f32,
) {
    if texture_width == 0 || texture_height == 0 || texture_rgba.len() < 4 {
        return;
    }
    let min_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(canvas.width() as f32 - 1.0)
        .max(0.0) as u32;
    let min_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(canvas.height() as f32 - 1.0)
        .max(0.0) as u32;
    let area = edge(points[0], points[1], points[2]);
    if area.abs() <= 0.0001 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let e0 = edge(points[1], points[2], point);
            let e1 = edge(points[2], points[0], point);
            let e2 = edge(points[0], points[1], point);
            if !((e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)) {
                continue;
            }
            let w0 = e0 / area;
            let w1 = e1 / area;
            let w2 = e2 / area;
            let u = (uvs[0][0] * w0 + uvs[1][0] * w1 + uvs[2][0] * w2).clamp(0.0, 1.0);
            let v = (uvs[0][1] * w0 + uvs[1][1] * w1 + uvs[2][1] * w2).clamp(0.0, 1.0);
            let tx = (u * texture_width.saturating_sub(1) as f32)
                .round()
                .clamp(0.0, texture_width.saturating_sub(1) as f32) as u32;
            let ty = (v * texture_height.saturating_sub(1) as f32)
                .round()
                .clamp(0.0, texture_height.saturating_sub(1) as f32) as u32;
            let offset = ((ty * texture_width + tx) as usize).saturating_mul(4);
            let Some(texel) = texture_rgba.get(offset..offset + 4) else {
                continue;
            };
            let alpha = texel[3] as f32 / 255.0 * material_factor[3].clamp(0.0, 1.0) * opacity;
            if alpha <= 0.0 {
                continue;
            }
            let color = Rgba([
                (texel[0] as f32 * material_factor[0].clamp(0.0, 1.0) * shade)
                    .round()
                    .clamp(0.0, 255.0) as u8,
                (texel[1] as f32 * material_factor[1].clamp(0.0, 1.0) * shade)
                    .round()
                    .clamp(0.0, 255.0) as u8,
                (texel[2] as f32 * material_factor[2].clamp(0.0, 1.0) * shade)
                    .round()
                    .clamp(0.0, 255.0) as u8,
                (alpha * 255.0).round().clamp(0.0, 255.0) as u8,
            ]);
            blend_pixel(canvas, x, y, color);
        }
    }
}

fn edge(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

fn fill_ellipse(canvas: &mut RgbaImage, cx: f32, cy: f32, rx: f32, ry: f32, color: Rgba<u8>) {
    let rx = rx.max(0.5);
    let ry = ry.max(0.5);
    let min_x = (cx - rx).floor().max(0.0) as u32;
    let max_x = (cx + rx).ceil().min(canvas.width() as f32 - 1.0).max(0.0) as u32;
    let min_y = (cy - ry).floor().max(0.0) as u32;
    let max_y = (cy + ry).ceil().min(canvas.height() as f32 - 1.0).max(0.0) as u32;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let nx = (x as f32 + 0.5 - cx) / rx;
            let ny = (y as f32 + 0.5 - cy) / ry;
            if nx * nx + ny * ny <= 1.0 {
                blend_pixel(canvas, x, y, color);
            }
        }
    }
}

fn draw_line(
    canvas: &mut RgbaImage,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: Rgba<u8>,
    width: f32,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).ceil().max(1.0) as u32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = x0 + dx * t;
        let y = y0 + dy * t;
        fill_ellipse(canvas, x, y, width * 0.5, width * 0.5, color);
    }
}

fn blend_pixel(canvas: &mut RgbaImage, x: u32, y: u32, src: Rgba<u8>) {
    let dst = canvas.get_pixel_mut(x, y);
    let sa = src[3] as f32 / 255.0;
    if sa <= 0.0 {
        return;
    }
    let da = dst[3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= f32::EPSILON {
        *dst = Rgba([0, 0, 0, 0]);
        return;
    }
    for i in 0..3 {
        let sc = src[i] as f32 / 255.0;
        let dc = dst[i] as f32 / 255.0;
        dst[i] = (((sc * sa + dc * da * (1.0 - sa)) / out_a) * 255.0).round() as u8;
    }
    dst[3] = (out_a * 255.0).round() as u8;
}

fn with_opacity(mut color: Rgba<u8>, opacity: f32) -> Rgba<u8> {
    color[3] = (color[3] as f32 * opacity.clamp(0.0, 1.0)).round() as u8;
    color
}

fn parse_rgba(raw: &str) -> Rgba<u8> {
    let text = raw.trim().trim_matches('"').trim();
    let Some(hex) = text.strip_prefix('#') else {
        return Rgba([0, 0, 0, 255]);
    };
    let parse = |range: std::ops::Range<usize>| {
        hex.get(range)
            .and_then(|part| u8::from_str_radix(part, 16).ok())
            .unwrap_or(0)
    };
    match hex.len() {
        6 => Rgba([parse(0..2), parse(2..4), parse(4..6), 255]),
        8 => Rgba([parse(0..2), parse(2..4), parse(4..6), parse(6..8)]),
        _ => Rgba([0, 0, 0, 255]),
    }
}

fn eval_number(expr: &str, default: f32, time: WorldTime) -> Result<f32, WorldRenderError> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Ok(default);
    }
    eval_time_expr(trimmed, time.time_norm(), time.time_sec()).map_err(|message| {
        WorldRenderError::Expression {
            expr: trimmed.to_string(),
            message,
        }
    })
}

fn resolve_asset_path_with_style(
    asset_root: &Path,
    src: &str,
    path_style: WorldPathStyle,
) -> PathBuf {
    let path = Path::new(src);
    match path_style {
        WorldPathStyle::Absolute => path.to_path_buf(),
        WorldPathStyle::Relative => {
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                asset_root.join(path)
            }
        }
    }
}

/// Result of resolving a world asset source, used to load from either the
/// filesystem or an in-memory resolver (e.g. WASM `add_asset`).
enum ResolvedWorldAsset {
    Path(PathBuf),
    Bytes { key: PathBuf, bytes: Vec<u8> },
    Missing { key: PathBuf },
}

#[cfg(not(target_arch = "wasm32"))]
const MAX_REMOTE_WORLD_ASSET_BYTES: u64 = 64 * 1024 * 1024;

fn is_remote_world_asset_source(src: &str) -> bool {
    url::Url::parse(src)
        .map(|url| matches!(url.scheme(), "http" | "https"))
        .unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_remote_world_asset_bytes(src: &str) -> Result<Vec<u8>, WorldRenderError> {
    let response = ureq::get(src)
        .set("User-Agent", "MotionLoom Scene3DRenderer")
        .call()
        .map_err(|err| WorldRenderError::RemoteAsset {
            url: src.to_string(),
            message: match err {
                ureq::Error::Status(code, response) => {
                    format!("HTTP {code} {}", response.status_text())
                }
                other => other.to_string(),
            },
        })?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(MAX_REMOTE_WORLD_ASSET_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| WorldRenderError::RemoteAsset {
            url: src.to_string(),
            message: source.to_string(),
        })?;
    if bytes.is_empty() {
        return Err(WorldRenderError::RemoteAsset {
            url: src.to_string(),
            message: "response body was empty".to_string(),
        });
    }
    if bytes.len() as u64 > MAX_REMOTE_WORLD_ASSET_BYTES {
        return Err(WorldRenderError::RemoteAsset {
            url: src.to_string(),
            message: format!(
                "asset exceeds the {} MiB native download limit",
                MAX_REMOTE_WORLD_ASSET_BYTES / (1024 * 1024)
            ),
        });
    }
    Ok(bytes)
}

fn resolve_remote_world_asset(src: &str) -> Result<ResolvedWorldAsset, WorldRenderError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let bytes = fetch_remote_world_asset_bytes(src)?;
        Ok(ResolvedWorldAsset::Bytes {
            key: PathBuf::from(src),
            bytes,
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        Err(WorldRenderError::RemoteAsset {
            url: src.to_string(),
            message: "browser hosts must preload URL assets into the renderer".to_string(),
        })
    }
}

impl ResolvedWorldAsset {
    /// Cache key used to deduplicate loaded images. For filesystem assets this
    /// is the resolved path; for memory assets it is the original source name.
    fn key(&self) -> &Path {
        match self {
            ResolvedWorldAsset::Path(path) => path,
            ResolvedWorldAsset::Bytes { key, .. } => key,
            ResolvedWorldAsset::Missing { key } => key,
        }
    }
}

/// Resolve an asset source through the global resolver, falling back to the
/// legacy filesystem resolution when no resolver entry exists.
fn resolve_world_asset_source(
    asset_root: &Path,
    src: &str,
    path_style: WorldPathStyle,
    resolver: &dyn AssetResolver,
) -> Result<ResolvedWorldAsset, WorldRenderError> {
    if src.trim_start().to_ascii_lowercase().starts_with("data:") {
        return decode_world_data_uri(src).map(|bytes| ResolvedWorldAsset::Bytes {
            key: PathBuf::from(src),
            bytes,
        });
    }
    match resolver.resolve(src) {
        Ok(AssetSource::Bytes(bytes)) => Ok(ResolvedWorldAsset::Bytes {
            key: PathBuf::from(src),
            bytes,
        }),
        Ok(AssetSource::Path(resolved_path)) => {
            // The resolver may return the raw source path (e.g. PathAssetResolver).
            // A URL returned by that resolver is not a filesystem path: native
            // Scene3DRenderer downloads it, while browser hosts continue to
            // supply asynchronously prefetched bytes through MemoryAssetResolver.
            if is_remote_world_asset_source(src) {
                return resolve_remote_world_asset(src);
            }
            // Resolve relative paths against the asset root and verify existence on
            // native platforms. On WASM there is no filesystem, so treat Path results
            // as missing and rely on memory assets instead.
            let path = if resolved_path.is_absolute() {
                resolved_path
            } else {
                resolve_asset_path_with_style(asset_root, src, path_style)
            };
            #[cfg(not(target_arch = "wasm32"))]
            {
                if path.exists() {
                    Ok(ResolvedWorldAsset::Path(path))
                } else {
                    Ok(ResolvedWorldAsset::Missing {
                        key: PathBuf::from(src),
                    })
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = path;
                Ok(ResolvedWorldAsset::Missing {
                    key: PathBuf::from(src),
                })
            }
        }
        Ok(AssetSource::Url(url)) => resolve_remote_world_asset(&url),
        Err(_) => {
            if is_remote_world_asset_source(src) {
                return resolve_remote_world_asset(src);
            }
            let path = resolve_asset_path_with_style(asset_root, src, path_style);
            if path.exists() {
                Ok(ResolvedWorldAsset::Path(path))
            } else {
                Ok(ResolvedWorldAsset::Missing {
                    key: PathBuf::from(src),
                })
            }
        }
    }
}

/// Decode an inline asset before consulting path or host resolvers. Scene 3D
/// assets use the unchanged data URI as their stable cache key, matching URL
/// and memory-backed assets while keeping self-contained DSL portable.
fn decode_world_data_uri(src: &str) -> Result<Vec<u8>, WorldRenderError> {
    let trimmed = src.trim_start();
    let Some(comma_ix) = trimmed.find(',') else {
        return Err(WorldRenderError::InvalidDataUri {
            message: "missing data payload separator ','".to_string(),
        });
    };
    let (header, payload_with_comma) = trimmed.split_at(comma_ix);
    if !header.to_ascii_lowercase().contains(";base64") {
        return Err(WorldRenderError::InvalidDataUri {
            message: "3D inline assets must use base64 encoding".to_string(),
        });
    }
    base64::engine::general_purpose::STANDARD
        .decode(&payload_with_comma[1..])
        .map_err(|err| WorldRenderError::InvalidDataUri {
            message: format!("base64 decode failed: {err}"),
        })
}

/// Load a dynamic image from a resolved world asset.
fn load_rgba_image_from_resolved(
    resolved: &ResolvedWorldAsset,
    error_ctor: impl Fn(PathBuf, image::ImageError) -> WorldRenderError,
) -> Result<image::DynamicImage, WorldRenderError> {
    match resolved {
        ResolvedWorldAsset::Path(path) => {
            image::open(path).map_err(|source| error_ctor(path.clone(), source))
        }
        ResolvedWorldAsset::Bytes { key, bytes } => {
            image::load_from_memory(bytes).map_err(|source| error_ctor(key.clone(), source))
        }
        ResolvedWorldAsset::Missing { key } => Err(error_ctor(
            key.clone(),
            image::ImageError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "asset not found",
            )),
        )),
    }
}

/// Load a GLB mesh from a resolved world asset. Returns the mesh together with a
/// stable cache key derived from the source resolution.
fn load_glb_mesh_resolved(
    asset_root: &Path,
    src: &str,
    path_style: WorldPathStyle,
    resolver: &dyn AssetResolver,
) -> Result<(PathBuf, GlbMeshData), WorldRenderError> {
    let resolved = resolve_world_asset_source(asset_root, src, path_style, resolver)?;
    let key = resolved.key().to_path_buf();
    let mesh = match resolved {
        ResolvedWorldAsset::Path(path) => load_glb_mesh_data(&path)?,
        ResolvedWorldAsset::Bytes { bytes, .. } => load_glb_mesh_data_from_bytes(&key, &bytes)?,
        ResolvedWorldAsset::Missing { .. } => {
            return Err(GlbLoadError::Io {
                path: key,
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "GLB asset not found"),
            }
            .into());
        }
    };
    Ok((key, mesh))
}

/// Load a GLB motion source. Unlike model loading, this accepts a valid GLB
/// containing nodes, a skeleton, and animation clips without mesh geometry.
fn load_glb_animation_resolved(
    asset_root: &Path,
    src: &str,
    path_style: WorldPathStyle,
    resolver: &dyn AssetResolver,
) -> Result<(PathBuf, GlbMeshData), WorldRenderError> {
    let resolved = resolve_world_asset_source(asset_root, src, path_style, resolver)?;
    let key = resolved.key().to_path_buf();
    let animation = match resolved {
        ResolvedWorldAsset::Path(path) => load_glb_animation_data(&path)?,
        ResolvedWorldAsset::Bytes { bytes, .. } => {
            load_glb_animation_data_from_bytes(&key, &bytes)?
        }
        ResolvedWorldAsset::Missing { .. } => {
            return Err(GlbLoadError::Io {
                path: key,
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "GLB animation asset not found",
                ),
            }
            .into());
        }
    };
    Ok((key, animation))
}

fn glb_mesh_source_cache_key(asset_root: &Path, src: &str, path_style: WorldPathStyle) -> PathBuf {
    if is_remote_world_asset_source(src) {
        PathBuf::from(src)
    } else {
        resolve_asset_path_with_style(asset_root, src, path_style)
    }
}

fn load_cached_glb_mesh_resolved<'a>(
    asset_root: &Path,
    src: &str,
    path_style: WorldPathStyle,
    resolver: &dyn AssetResolver,
    mesh_cache: &'a mut HashMap<PathBuf, GlbMeshData>,
) -> Result<(PathBuf, &'a GlbMeshData), WorldRenderError> {
    let source_key = glb_mesh_source_cache_key(asset_root, src, path_style);
    if !mesh_cache.contains_key(&source_key) {
        let (_, mesh) = load_glb_mesh_resolved(asset_root, src, path_style, resolver)?;
        mesh_cache.insert(source_key.clone(), mesh);
    }
    let mesh = mesh_cache
        .get(&source_key)
        .expect("mesh cache entry inserted before render");
    Ok((source_key, mesh))
}

/// Resolve typed primitives into the same retained mesh cache used by GLB actors.
fn load_cached_actor_mesh<'a>(
    asset_root: &Path,
    actor: &WorldActor,
    resolver: &dyn AssetResolver,
    mesh_cache: &'a mut HashMap<PathBuf, GlbMeshData>,
    primitive_texture_cache: &mut HashMap<PrimitiveTextureSourceKey, Arc<GlbTextureData>>,
    resource_stats: &mut PrimitiveResourceLoadStats,
) -> Result<(PathBuf, &'a GlbMeshData), WorldRenderError> {
    if let Some(terrain) = &actor.terrain {
        let height_source = terrain
            .height_map_src
            .as_deref()
            .unwrap_or(terrain.height_map.as_str());
        let remote_source_key = is_remote_world_asset_source(height_source)
            .then(|| crate::world::terrain::remote_terrain_cache_key(terrain));
        if let Some(source_key) = remote_source_key.as_ref() {
            if mesh_cache.contains_key(source_key) {
                let mesh = mesh_cache
                    .get(source_key)
                    .expect("remote terrain mesh found before render");
                return Ok((source_key.clone(), mesh));
            }
        }
        let height_texture = load_cached_primitive_texture(
            asset_root,
            actor.path_style,
            height_source,
            resolver,
            primitive_texture_cache,
            resource_stats,
        )?;
        let height_map = RgbaImage::from_raw(
            height_texture.width,
            height_texture.height,
            height_texture.rgba.as_ref().clone(),
        )
        .expect("decoded terrain height map has complete RGBA pixels");
        let source_key = remote_source_key
            .unwrap_or_else(|| crate::world::terrain::terrain_cache_key(terrain, &height_map));
        if !mesh_cache.contains_key(&source_key) {
            let texture_set = load_terrain_texture_set(
                asset_root,
                actor.path_style,
                terrain,
                resolver,
                primitive_texture_cache,
                resource_stats,
            )?;
            mesh_cache.insert(
                source_key.clone(),
                crate::world::terrain::generate_terrain_mesh_textured(
                    terrain,
                    &height_map,
                    texture_set,
                ),
            );
        }
        let mesh = mesh_cache
            .get(&source_key)
            .expect("terrain mesh inserted before render");
        return Ok((source_key, mesh));
    }
    if let Some(vegetation) = &actor.vegetation {
        let source_key = crate::world::vegetation::vegetation_cache_key(vegetation);
        if !mesh_cache.contains_key(&source_key) {
            let (primary, secondary) = crate::world::vegetation::vegetation_materials(vegetation);
            let primary_primitive = crate::world::vegetation::vegetation_surface_primitive(
                vegetation, primary, "primary",
            );
            let primary_textures = load_primitive_texture_set(
                asset_root,
                actor.path_style,
                &primary_primitive,
                resolver,
                primitive_texture_cache,
                resource_stats,
            )?;
            let secondary_textures = secondary
                .map(|material| {
                    let primitive = crate::world::vegetation::vegetation_surface_primitive(
                        vegetation,
                        Some(material),
                        "secondary",
                    );
                    load_primitive_texture_set(
                        asset_root,
                        actor.path_style,
                        &primitive,
                        resolver,
                        primitive_texture_cache,
                        resource_stats,
                    )
                })
                .transpose()?;
            mesh_cache.insert(
                source_key.clone(),
                crate::world::vegetation::generate_vegetation_mesh_textured(
                    vegetation,
                    crate::world::vegetation::VegetationTextureSet {
                        primary: primary_textures,
                        secondary: secondary_textures,
                    },
                ),
            );
        }
        let mesh = mesh_cache
            .get(&source_key)
            .expect("vegetation mesh inserted before render");
        return Ok((source_key, mesh));
    }
    if let Some(primitive) = &actor.primitive {
        let source_key = actor.native_skin.as_ref().map_or_else(
            || crate::world::primitive::primitive_cache_key(primitive),
            |skin| crate::world::primitive::native_skinned_primitive_cache_key(primitive, skin),
        );
        if !mesh_cache.contains_key(&source_key) {
            let texture_set = load_primitive_texture_set(
                asset_root,
                actor.path_style,
                primitive,
                resolver,
                primitive_texture_cache,
                resource_stats,
            )?;
            let mut mesh =
                crate::world::primitive::generate_primitive_mesh_textured(primitive, texture_set);
            if let Some(skin) = actor.native_skin.as_ref() {
                crate::world::primitive::apply_native_skin_to_mesh(&mut mesh, skin);
            }
            mesh_cache.insert(source_key.clone(), mesh);
        }
        let mesh = mesh_cache
            .get(&source_key)
            .expect("primitive mesh inserted before render");
        return Ok((source_key, mesh));
    }
    load_cached_glb_mesh_resolved(
        asset_root,
        &actor.model,
        actor.path_style,
        resolver,
        mesh_cache,
    )
}

fn load_terrain_texture_set(
    asset_root: &Path,
    path_style: WorldPathStyle,
    terrain: &crate::dsl::TerrainAssetNode,
    resolver: &dyn AssetResolver,
    texture_cache: &mut HashMap<PrimitiveTextureSourceKey, Arc<GlbTextureData>>,
    resource_stats: &mut PrimitiveResourceLoadStats,
) -> Result<crate::world::primitive::PrimitiveTextureSet, WorldRenderError> {
    if terrain.layer_definitions.is_empty() {
        let primitive = crate::world::terrain::terrain_surface_primitive(terrain);
        return load_primitive_texture_set(
            asset_root,
            path_style,
            &primitive,
            resolver,
            texture_cache,
            resource_stats,
        );
    }

    let blend_source = terrain
        .blend_map_src
        .as_deref()
        .or(terrain.blend_map.as_deref())
        .expect("validated layered terrain has a blend map");
    let blend = load_cached_primitive_texture(
        asset_root,
        path_style,
        blend_source,
        resolver,
        texture_cache,
        resource_stats,
    )?;
    let mut layers = Vec::with_capacity(terrain.layer_definitions.len());
    for material in &terrain.layer_definitions {
        let mut primitive = crate::world::terrain::terrain_surface_primitive(terrain);
        primitive.material_definition = Some(material.clone());
        layers.push((
            material.clone(),
            load_primitive_texture_set(
                asset_root,
                path_style,
                &primitive,
                resolver,
                texture_cache,
                resource_stats,
            )?,
        ));
    }
    Ok(blend_terrain_layers(&blend, &layers))
}

fn blend_terrain_layers(
    blend: &GlbTextureData,
    layers: &[(
        crate::dsl::MaterialAssetNode,
        crate::world::primitive::PrimitiveTextureSet,
    )],
) -> crate::world::primitive::PrimitiveTextureSet {
    let width = blend.width.max(1);
    let height = blend.height.max(1);
    let mut base = vec![0_u8; (width * height * 4) as usize];
    let mut metallic_roughness = vec![0_u8; base.len()];
    let mut normal = vec![0_u8; base.len()];
    let mut emissive = vec![0_u8; base.len()];
    let mut occlusion = vec![255_u8; base.len()];
    for y in 0..height {
        for x in 0..width {
            let index = ((y * width + x) * 4) as usize;
            let weights = terrain_blend_weights(&blend.rgba[index..index + 4], layers.len());
            let mut base_pixel = [0.0_f32; 4];
            let mut mr_pixel = [0.0_f32; 4];
            let mut normal_vector = [0.0_f32; 3];
            let mut emissive_pixel = [0.0_f32; 4];
            let mut occlusion_value = 0.0_f32;
            for (layer_index, (material, textures)) in layers.iter().enumerate() {
                let weight = weights[layer_index];
                let layer_ao = terrain_texture_pixel(
                    textures.occlusion.as_ref(),
                    x,
                    y,
                    width,
                    height,
                    material.texture_scale,
                    material.texture_offset,
                    [255.0; 4],
                )[0] / 255.0;
                occlusion_value += (1.0 - material.occlusion_strength * (1.0 - layer_ao)) * weight;
                let color = terrain_texture_pixel(
                    textures.base_color.as_ref(),
                    x,
                    y,
                    width,
                    height,
                    material.texture_scale,
                    material.texture_offset,
                    material.base_color.map(|value| value * 255.0),
                );
                let mr = terrain_texture_pixel(
                    textures.metallic_roughness.as_ref(),
                    x,
                    y,
                    width,
                    height,
                    material.texture_scale,
                    material.texture_offset,
                    [
                        255.0,
                        material.roughness * 255.0,
                        material.metallic * 255.0,
                        255.0,
                    ],
                );
                let encoded_normal = terrain_texture_pixel(
                    textures.normal.as_ref(),
                    x,
                    y,
                    width,
                    height,
                    material.texture_scale,
                    material.texture_offset,
                    [127.5, 127.5, 255.0, 255.0],
                );
                let layer_normal = [
                    (encoded_normal[0] / 127.5 - 1.0) * material.normal_scale,
                    (encoded_normal[1] / 127.5 - 1.0) * material.normal_scale,
                    encoded_normal[2] / 127.5 - 1.0,
                ];
                let emission_fallback = [
                    material.emissive[0] * material.emissive_strength * 255.0,
                    material.emissive[1] * material.emissive_strength * 255.0,
                    material.emissive[2] * material.emissive_strength * 255.0,
                    255.0,
                ];
                let emission = terrain_texture_pixel(
                    textures.emissive.as_ref(),
                    x,
                    y,
                    width,
                    height,
                    material.texture_scale,
                    material.texture_offset,
                    emission_fallback,
                );
                for channel in 0..4 {
                    base_pixel[channel] += color[channel] * weight;
                    mr_pixel[channel] += mr[channel] * weight;
                    emissive_pixel[channel] += emission[channel] * weight;
                }
                for channel in 0..3 {
                    normal_vector[channel] += layer_normal[channel] * weight;
                }
            }
            occlusion[index] = (occlusion_value * 255.0).round().clamp(0.0, 255.0) as u8;
            let normal_length = normal_vector
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
                .max(1.0e-6);
            let encoded = [
                (normal_vector[0] / normal_length * 0.5 + 0.5) * 255.0,
                (normal_vector[1] / normal_length * 0.5 + 0.5) * 255.0,
                (normal_vector[2] / normal_length * 0.5 + 0.5) * 255.0,
                255.0,
            ];
            for channel in 0..4 {
                base[index + channel] = base_pixel[channel].round().clamp(0.0, 255.0) as u8;
                metallic_roughness[index + channel] =
                    mr_pixel[channel].round().clamp(0.0, 255.0) as u8;
                normal[index + channel] = encoded[channel].round().clamp(0.0, 255.0) as u8;
                emissive[index + channel] = emissive_pixel[channel].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let texture = |rgba| GlbTextureData {
        width,
        height,
        rgba: Arc::new(rgba),
    };
    crate::world::primitive::PrimitiveTextureSet {
        face: HashMap::new(),
        base_color: Some(texture(base)),
        metallic_roughness: Some(texture(metallic_roughness)),
        normal: Some(texture(normal)),
        emissive: Some(texture(emissive)),
        occlusion: Some(texture(occlusion)),
    }
}

fn terrain_blend_weights(pixel: &[u8], layer_count: usize) -> Vec<f32> {
    let mut weights = pixel
        .iter()
        .take(layer_count)
        .map(|value| f32::from(*value) / 255.0)
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    if total <= f32::EPSILON {
        weights.fill(0.0);
        if let Some(first) = weights.first_mut() {
            *first = 1.0;
        }
    } else {
        weights.iter_mut().for_each(|weight| *weight /= total);
    }
    weights
}

#[expect(
    clippy::too_many_arguments,
    reason = "Texture sampling requires independent UV, size, wrap and fallback inputs."
)]
fn terrain_texture_pixel(
    texture: Option<&GlbTextureData>,
    x: u32,
    y: u32,
    target_width: u32,
    target_height: u32,
    texture_scale: [f32; 2],
    texture_offset: [f32; 2],
    fallback: [f32; 4],
) -> [f32; 4] {
    let Some(texture) = texture else {
        return fallback;
    };
    let u = (x as f32 / target_width.max(1) as f32 * texture_scale[0] + texture_offset[0])
        .rem_euclid(1.0);
    let v = (y as f32 / target_height.max(1) as f32 * texture_scale[1] + texture_offset[1])
        .rem_euclid(1.0);
    let source_x = (u * texture.width.max(1) as f32).floor() as u32;
    let source_y = (v * texture.height.max(1) as f32).floor() as u32;
    let index = ((source_y.min(texture.height - 1) * texture.width
        + source_x.min(texture.width - 1))
        * 4) as usize;
    std::array::from_fn(|channel| f32::from(texture.rgba[index + channel]))
}

fn load_primitive_texture_set(
    asset_root: &Path,
    path_style: WorldPathStyle,
    primitive: &crate::dsl::PrimitiveAssetNode,
    resolver: &dyn AssetResolver,
    texture_cache: &mut HashMap<PrimitiveTextureSourceKey, Arc<GlbTextureData>>,
    resource_stats: &mut PrimitiveResourceLoadStats,
) -> Result<crate::world::primitive::PrimitiveTextureSet, WorldRenderError> {
    let Some(material) = primitive.material_definition.as_ref() else {
        return Ok(Default::default());
    };
    let mut load = |source: &Option<String>| -> Result<Option<GlbTextureData>, WorldRenderError> {
        source
            .as_deref()
            .map(|source| {
                load_cached_primitive_texture(
                    asset_root,
                    path_style,
                    source,
                    resolver,
                    texture_cache,
                    resource_stats,
                )
            })
            .transpose()
    };
    let mut face = HashMap::new();
    if let crate::dsl::PrimitiveGeometry::HeadSurface {
        face_layout: Some(layout),
        ..
    } = &primitive.geometry
    {
        for (id, texture) in layout
            .eyes
            .iter()
            .map(|e| (&e.id, &e.texture))
            .chain(
                layout
                    .eyes
                    .iter()
                    .filter_map(|e| e.iris.as_ref().map(|i| (&i.id, &i.texture))),
            )
            .chain(
                layout
                    .eyes
                    .iter()
                    .flat_map(|e| e.eyeliners.iter().map(|l| (&l.id, &l.texture))),
            )
            .chain(layout.noses.iter().map(|e| (&e.id, &e.texture)))
            .chain(layout.mouths.iter().map(|e| (&e.id, &e.texture)))
            .chain(layout.ears.iter().map(|e| (&e.id, &e.texture)))
            .chain(layout.eyebrows.iter().map(|e| (&e.id, &e.texture)))
        {
            if let Some(texture) = texture {
                if let Some(image) = load(&texture.source)? {
                    face.insert(id.clone(), image);
                }
            }
        }
    }
    Ok(crate::world::primitive::PrimitiveTextureSet {
        face,
        base_color: load(&material.base_color_texture_src)?,
        occlusion: load(&material.occlusion_texture_src)?,
        metallic_roughness: load(&material.metallic_roughness_texture_src)?,
        normal: load(&material.normal_texture_src)?,
        emissive: load(&material.emissive_texture_src)?,
    })
}

fn load_cached_primitive_texture(
    asset_root: &Path,
    path_style: WorldPathStyle,
    source: &str,
    resolver: &dyn AssetResolver,
    texture_cache: &mut HashMap<PrimitiveTextureSourceKey, Arc<GlbTextureData>>,
    stats: &mut PrimitiveResourceLoadStats,
) -> Result<GlbTextureData, WorldRenderError> {
    let resolved = resolve_world_asset_source(asset_root, source, path_style, resolver)?;
    let fallback = crate::scene::resource::resolve_local_scene_asset_path(source);
    let key = primitive_texture_source_key(&resolved, &fallback);
    if let Some(texture) = texture_cache.get(&key) {
        stats.texture_cache_hits += 1;
        return Ok(texture.as_ref().clone());
    }
    let started = ProfileClock::now();
    let image = match resolved {
        ResolvedWorldAsset::Path(path) => image::open(path),
        ResolvedWorldAsset::Bytes { bytes, .. } => image::load_from_memory(&bytes),
        ResolvedWorldAsset::Missing { .. } => image::open(fallback),
    }
    .map_err(|source_error| WorldRenderError::PrimitiveMaterialTexture {
        source_ref: source.to_string(),
        source: source_error,
    })?
    .to_rgba8();
    let texture = Arc::new(GlbTextureData {
        width: image.width(),
        height: image.height(),
        rgba: Arc::new(image.into_raw()),
    });
    stats.texture_decode_ms += started.elapsed().as_secs_f64() * 1000.0;
    stats.texture_decode_count += 1;
    stats.texture_decoded_bytes += texture.rgba.len();
    texture_cache.retain(|existing, _| existing.identity != key.identity || *existing == key);
    texture_cache.insert(key, Arc::clone(&texture));
    Ok(texture.as_ref().clone())
}

fn primitive_texture_source_key(
    resolved: &ResolvedWorldAsset,
    missing_fallback: &Path,
) -> PrimitiveTextureSourceKey {
    let (identity, revision) = match resolved {
        ResolvedWorldAsset::Bytes { key, bytes } => {
            let mut hasher = DefaultHasher::new();
            bytes.hash(&mut hasher);
            (key.clone(), hasher.finish())
        }
        ResolvedWorldAsset::Path(path) => (path.clone(), file_revision(path)),
        ResolvedWorldAsset::Missing { .. } => (
            missing_fallback.to_path_buf(),
            file_revision(missing_fallback),
        ),
    };
    PrimitiveTextureSourceKey { identity, revision }
}

fn file_revision(path: &Path) -> u64 {
    let Ok(metadata) = fs::metadata(path) else {
        return 0;
    };
    let mut hasher = DefaultHasher::new();
    metadata.len().hash(&mut hasher);
    if let Ok(modified) = metadata.modified()
        && let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH)
    {
        duration.as_nanos().hash(&mut hasher);
    }
    hasher.finish()
}

fn load_cached_glb_animation_resolved<'a>(
    asset_root: &Path,
    src: &str,
    path_style: WorldPathStyle,
    resolver: &dyn AssetResolver,
    mesh_cache: &'a mut HashMap<PathBuf, GlbMeshData>,
) -> Result<(PathBuf, &'a GlbMeshData), WorldRenderError> {
    let source_key = glb_mesh_source_cache_key(asset_root, src, path_style);
    if !mesh_cache.contains_key(&source_key) {
        let (_, animation) = load_glb_animation_resolved(asset_root, src, path_style, resolver)?;
        mesh_cache.insert(source_key.clone(), animation);
    }
    let animation = mesh_cache
        .get(&source_key)
        .expect("animation cache entry inserted before sampling");
    Ok((source_key, animation))
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests;
