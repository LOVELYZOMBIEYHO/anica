// =========================================
// =========================================
// crates/motionloom/src/weaver/jobs/mod.rs

use super::api::{CancellationToken, FrameDeltaKind, RenderProgress, RenderReport};
use super::{
    WeaverError,
    backend::wgpu::{FILM_BYTES_PER_PIXEL, FILM_FLOATS_PER_PIXEL, Gpu},
    config::*,
    geometry, output,
};
pub(crate) mod sequence;
use crate::{
    AssetResolver, AssetSource,
    scene::{
        compositor::{
            AlphaMode, ColorStage, CompositedFrame, GpuCompositionInput, LinearPremultipliedImage,
            RenderDomain, ResolvedCompositionLayer, SceneCompositionPlan,
            WgpuSceneCompositorExecutor, build_scene_composition_plan,
        },
        render::{SceneRenderProfile, render_scene_graph_frame_with_resolver},
    },
};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

struct RootResolver(PathBuf);
impl AssetResolver for RootResolver {
    // This implements the pre-existing resolver trait; Weaver APIs use typed errors.
    fn resolve(&self, src: &str) -> Result<AssetSource, String> {
        let path = Path::new(src);
        Ok(AssetSource::Path(if path.is_absolute() {
            path.into()
        } else {
            self.0.join(path)
        }))
    }
}

#[derive(Default)]
pub(crate) struct RenderCache {
    source: Option<CachedSource>,
    packed_scene: Option<geometry::PackedScene>,
    textures_changed: bool,
    texture_key: String,
    gpu_scene: Option<(String, Arc<Gpu>)>,
    previous_camera: Option<[[f32; 4]; 4]>,
    temporal_enabled: bool,
    temporal_history: Option<TemporalHistory>,
}

struct CachedSource {
    path: PathBuf,
    len: u64,
    modified: Option<std::time::SystemTime>,
    script: Arc<String>,
    graph: Arc<crate::GraphScript>,
}

struct TemporalHistory {
    size: [u32; 2],
    color: Vec<f32>,
    depth: Vec<f32>,
    normal: Vec<[f32; 3]>,
    albedo: Vec<[f32; 3]>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PackedSceneUpdate {
    Built,
    Refitted,
}

impl RenderCache {
    fn packed_scene(
        &mut self,
        snapshot: &super::scene::Snapshot,
        mipmaps: bool,
        allow_transmission_stopgap: bool,
    ) -> Result<(geometry::PackedScene, PackedSceneUpdate), WeaverError> {
        let (packed, refitted) = match self.packed_scene.as_ref() {
            Some(previous) => {
                geometry::pack_refit(snapshot, mipmaps, allow_transmission_stopgap, previous)?
            }
            None => (
                geometry::pack(snapshot, mipmaps, allow_transmission_stopgap)?,
                false,
            ),
        };
        self.textures_changed = self
            .packed_scene
            .as_ref()
            .is_none_or(|previous| previous.pixels != packed.pixels);
        if self.textures_changed {
            let mut texture_hash = Sha256::new();
            for pixel in &packed.pixels {
                texture_hash.update(pixel.to_le_bytes());
            }
            self.texture_key = format!("{:x}", texture_hash.finalize());
        }
        self.packed_scene = Some(packed.clone());
        Ok((
            packed,
            if refitted {
                PackedSceneUpdate::Refitted
            } else {
                PackedSceneUpdate::Built
            },
        ))
    }

    fn source(
        &mut self,
        path: &Path,
    ) -> Result<(Arc<String>, Arc<crate::GraphScript>, bool), WeaverError> {
        let metadata = std::fs::metadata(path)?;
        let modified = metadata.modified().ok();
        if let Some(source) = &self.source {
            if source.path == path && source.len == metadata.len() && source.modified == modified {
                return Ok((source.script.clone(), source.graph.clone(), true));
            }
        }
        let script = Arc::new(std::fs::read_to_string(path)?);
        let graph = Arc::new(
            crate::parse_graph_script(&script)
                .map_err(|error| WeaverError::Scene(error.to_string()))?,
        );
        self.source = Some(CachedSource {
            path: path.to_path_buf(),
            len: metadata.len(),
            modified,
            script: script.clone(),
            graph: graph.clone(),
        });
        Ok((script, graph, false))
    }

    pub(crate) fn set_temporal_denoise(&mut self, enabled: bool) {
        self.temporal_enabled = enabled;
        if !enabled {
            self.temporal_history = None;
        }
    }

    fn apply_previous_camera(&mut self, params: &mut super::backend::wgpu::CameraParams) {
        if let Some(previous) = self.previous_camera {
            params[22..26].copy_from_slice(&previous);
            params[25][3] = 1.0;
        }
        self.previous_camera = Some([params[2], params[3], params[4], params[5]]);
    }

    async fn gpu(
        &mut self,
        packed: &geometry::PackedScene,
    ) -> Result<(Arc<Gpu>, FrameDeltaKind), WeaverError> {
        let mut hash = Sha256::new();
        hash.update(super::backend::wgpu::bytes(&packed.data));
        hash.update(self.texture_key.as_bytes());
        let key = format!("{:x}", hash.finalize());
        if let Some((cached_key, gpu)) = &mut self.gpu_scene {
            if *cached_key == key {
                return Ok((gpu.clone(), FrameDeltaKind::CameraOrUniforms));
            }
            if gpu.upload_scene(packed, self.textures_changed) {
                *cached_key = key;
                return Ok((gpu.clone(), FrameDeltaKind::SceneBufferUpdate));
            }
        }
        let delta = if self.gpu_scene.is_some() {
            FrameDeltaKind::SceneRebuild
        } else {
            FrameDeltaKind::FirstFrame
        };
        let gpu = Arc::new(Gpu::new(packed).await?);
        self.gpu_scene = Some((key, gpu.clone()));
        Ok((gpu, delta))
    }

    fn temporal_denoise(&mut self, size: [u32; 2], film: &[f32], clean: Vec<f32>) -> Vec<f32> {
        let count = size[0] as usize * size[1] as usize;
        let mut depth = Vec::with_capacity(count);
        let mut normal = Vec::with_capacity(count);
        let mut albedo = Vec::with_capacity(count);
        for pixel in film.chunks_exact(FILM_FLOATS_PER_PIXEL) {
            let samples = pixel[3].max(1.0);
            depth.push(pixel[15] / samples);
            normal.push([
                pixel[12] / samples,
                pixel[13] / samples,
                pixel[14] / samples,
            ]);
            albedo.push([pixel[8] / samples, pixel[9] / samples, pixel[10] / samples]);
        }
        let mut output = clean;
        if self.temporal_enabled {
            if let Some(history) = &self.temporal_history {
                if history.size == size {
                    for (index, pixel) in film.chunks_exact(FILM_FLOATS_PER_PIXEL).enumerate() {
                        let samples = pixel[3].max(1.0);
                        let x = index % size[0] as usize;
                        let y = index / size[0] as usize;
                        let px = x as f32 - pixel[17] / samples;
                        let py = y as f32 - pixel[18] / samples;
                        if px < 0.0 || py < 0.0 || px >= size[0] as f32 || py >= size[1] as f32 {
                            continue;
                        }
                        let previous = py.round() as usize * size[0] as usize + px.round() as usize;
                        if previous >= count
                            || depth[index] <= 0.0
                            || history.depth[previous] <= 0.0
                        {
                            continue;
                        }
                        let dot = normal[index]
                            .iter()
                            .zip(history.normal[previous])
                            .map(|(a, b)| a * b)
                            .sum::<f32>();
                        let albedo_delta = albedo[index]
                            .iter()
                            .zip(history.albedo[previous])
                            .map(|(a, b)| (a - b).abs())
                            .sum::<f32>();
                        if dot < 0.9 || albedo_delta > 0.25 {
                            continue;
                        }
                        for channel in 0..3 {
                            let current = output[index * 3 + channel];
                            let old = history.color[previous * 3 + channel];
                            let limit = current.abs().max(0.02) * 0.5;
                            output[index * 3 + channel] =
                                current * 0.8 + old.clamp(current - limit, current + limit) * 0.2;
                        }
                    }
                }
            }
            self.temporal_history = Some(TemporalHistory {
                size,
                color: output.clone(),
                depth,
                normal,
                albedo,
            });
        }
        output
    }
}

/// Render one evaluated frame. The host owns the executor and cancellation token.
/// Checkpoints are keyed by settings, shader, geometry, lighting and texture bytes.
pub async fn render<F: FnMut(RenderProgress)>(
    job: &RenderJob,
    cancel: &CancellationToken,
    mut progress: F,
) -> Result<RenderReport, WeaverError> {
    render_internal(job, cancel, &mut progress, None).await
}

pub(crate) async fn render_cached<F: FnMut(RenderProgress)>(
    job: &RenderJob,
    cancel: &CancellationToken,
    mut progress: F,
    cache: &mut RenderCache,
) -> Result<RenderReport, WeaverError> {
    render_internal(job, cancel, &mut progress, Some(cache)).await
}

async fn render_internal<F: FnMut(RenderProgress)>(
    job: &RenderJob,
    cancel: &CancellationToken,
    progress: &mut F,
    mut cache: Option<&mut RenderCache>,
) -> Result<RenderReport, WeaverError> {
    job.validate()?;
    let started = Instant::now();
    let parse_started = Instant::now();
    let path = std::fs::canonicalize(&job.scene)?;
    let root = path.parent().unwrap();
    let (script, graph, reused_source) = match cache.as_deref_mut() {
        Some(cache) => cache.source(&path)?,
        None => {
            let script = Arc::new(std::fs::read_to_string(&path)?);
            let graph = Arc::new(
                crate::parse_graph_script(&script)
                    .map_err(|error| WeaverError::Scene(error.to_string()))?,
            );
            (script, graph, false)
        }
    };
    if !graph.fps.is_finite()
        || graph.fps <= 0.0
        || job.frame as f64 / graph.fps as f64 >= graph.duration_ms as f64 / 1000.0
    {
        return Err(WeaverError::Invalid("frame outside scene duration".into()));
    }
    let composition = resolve_composition_plan(&graph, &job.scene_id)?;
    let mut timings = super::api::RenderTimings {
        parse_seconds: parse_started.elapsed().as_secs_f64(),
        ..Default::default()
    };
    let source_diagnostic = reused_source.then_some(
        "Sequence session reused the parsed graph and source text; no per-frame parser rebuild."
            .to_string(),
    );
    if job.output_mode == SceneOutputMode::CompositeScene {
        if composition.output_size != job.resolution {
            return Err(WeaverError::Invalid(format!(
                "composite_scene resolution must match authored output {}x{}",
                composition.output_size[0], composition.output_size[1]
            )));
        }
        if composition
            .layers
            .iter()
            .all(|layer| layer.domain != RenderDomain::ThreeD)
        {
            return render_pure_2d(
                job,
                &graph,
                &script,
                Arc::new(RootResolver(root.into())),
                started,
                progress,
            )
            .await;
        }
    }
    let scene_started = Instant::now();
    let mut snap =
        crate::scene::render::weaver_snapshot(&graph, job, Arc::new(RootResolver(root.into())))
            .await?;
    timings.scene_evaluation_seconds = scene_started.elapsed().as_secs_f64();
    if let Some(diagnostic) = source_diagnostic {
        snap.diagnostics.push(diagnostic);
    }
    snap.diagnostics.push(format!(
        "SceneCompositionPlan v{} contains {} ordered layer(s) and requires coverage={} depth={} motion={}.",
        snap.composition.version,
        snap.composition.layers.len(),
        snap.composition.required_aovs.coverage,
        snap.composition.required_aovs.depth,
        snap.composition.required_aovs.motion,
    ));
    // Apply explicit offline lighting before packing, hashing and rendering.
    for (id, intensity) in &job.lighting.light_intensities {
        let light = snap
            .lighting
            .lights
            .iter_mut()
            .find(|light| light.id.as_ref() == Some(id))
            .ok_or_else(|| WeaverError::Invalid(format!("unknown light override: {id}")))?;
        light.intensity = *intensity;
    }
    if let Some(value) = job.lighting.environment_intensity {
        let env = snap.lighting.environment.as_mut().ok_or_else(|| {
            WeaverError::Invalid("environment override requires an environment".into())
        })?;
        env.intensity = value;
    }
    if let Some(value) = job.lighting.exposure {
        snap.lighting.color_management.exposure = value;
    }
    if job.allow_transmission_stopgap {
        snap.diagnostics.push(
            "Explicit transmission stopgap enabled: transmissive materials use the current opaque/alpha PBR surface and are not physically refractive."
                .into(),
        );
    }
    let pack_started = Instant::now();
    let (mut packed, packed_scene_update) = match cache.as_deref_mut() {
        Some(cache) => {
            cache.packed_scene(&snap, job.texture_mips, job.allow_transmission_stopgap)?
        }
        None => (
            geometry::pack(&snap, job.texture_mips, job.allow_transmission_stopgap)?,
            PackedSceneUpdate::Built,
        ),
    };
    match packed_scene_update {
        PackedSceneUpdate::Refitted => snap.diagnostics.push(
            "Sequence cache retained the SAH BVH topology and refitted bounds for animated geometry."
                .into(),
        ),
        PackedSceneUpdate::Built => {}
    }
    // f32-backed offsets stay exact only up to 2^24; log the packed layout while
    // diagnosing large-scene traversal misses.
    if std::env::var_os("WEAVER_SCENE_DEBUG").is_some() {
        eprintln!(
            "weaver debug: packed triangles={} data={} triangle_offset={} material_offset={} light_offset={} emitter_offset={}",
            packed.triangles,
            packed.data.len(),
            packed.triangle_offset,
            packed.material_offset,
            packed.light_offset,
            packed.emitter_offset
        );
    }
    // Packed GPU data owns the required values; release duplicate world meshes.
    snap.meshes.clear();
    let mut p = [[0.0f32; 4]; 26];
    p[0] = [
        0.0,
        0.0,
        job.sampling.batch_samples as f32,
        f32::from_bits(job.seed),
    ];
    p[1] = [
        // Buffer offsets are u32 addresses carried through f32 uniforms; bitcast
        // keeps them exact beyond f32's 2^24 integer limit on large scenes.
        f32::from_bits(packed.triangle_offset),
        f32::from_bits(packed.material_offset),
        f32::from_bits(packed.light_offset),
        packed.lights as f32,
    ];
    super::camera::configure(&mut p, &snap.camera, job)?;
    if let Some(cache) = cache.as_deref_mut() {
        cache.apply_previous_camera(&mut p);
    }
    snap.diagnostics.push(format!(
        "Physical lens: {:.2} mm equivalent focal length, f/{:.2}, focus {:.2} scene units; assumes one unit is one meter.",
        job.lens.sensor_width_mm / (2.0 * p[3][3] * p[4][3]),
        job.lens.f_stop, job.lens.focus_distance,
    ));
    p[7] = [
        job.sampling.min_samples as f32,
        job.sampling.max_samples as f32,
        job.sampling.noise_threshold,
        0.0,
    ];
    p[8] = [
        job.light_paths.total as f32,
        job.light_paths.diffuse as f32,
        job.light_paths.glossy as f32,
        job.light_paths.roulette_start as f32,
    ];
    p[9][3] = job.light_paths.transparent as f32;
    p[14] = [
        f32::from_bits(packed.emitter_offset),
        packed.emitter_count as f32,
        packed.emitter_area,
        0.0,
    ];
    p[15][0] = job.sun_angular_diameter_degrees.to_radians() * 0.5;
    // B1/B2: hemisphere ambient from the resolved style and normal-based AO.
    let (ambient_intensity, ambient_color) = snap
        .lighting
        .render_style
        .as_ref()
        .map_or((1.0, [1.0; 3]), |style| {
            (style.ambient_intensity, style.ambient_color)
        });
    p[15][1] = snap.lighting.ao_intensity.clamp(0.0, 4.0);
    p[19] = [
        ambient_intensity.max(0.0),
        ambient_color[0],
        ambient_color[1],
        ambient_color[2],
    ];
    super::lighting::pack_atmosphere(
        &mut p,
        snap.lighting.atmosphere_medium.as_ref(),
        &snap.lighting.lights,
        snap.time_seconds,
    );
    if let Some(env) = &snap.lighting.environment {
        let env_path = Path::new(&env.src);
        let env_path = if env_path.is_absolute() {
            env_path.into()
        } else {
            root.join(env_path)
        };
        let (texture, cdf) = super::lighting::environment(&env_path, &mut packed.data)?;
        p[10] = texture;
        p[10][3] = env.rotation_y_degrees.to_radians();
        p[13] = cdf;
        p[11][0] = env.intensity;
        p[11][1] = if env.visible {
            env.background_intensity
        } else {
            0.0
        };
        p[11][2] = env.diffuse_intensity.max(0.0);
        p[11][3] = env.specular_intensity.max(0.0);
        snap.diagnostics.push("Environment uses float radiance, solid-angle importance sampling and MIS. Background blur is not applied.".into());
    }
    if p.iter()
        .flatten()
        .enumerate()
        .any(|(i, v)| i != 3 && !v.is_finite())
    {
        return Err(WeaverError::Scene("degenerate camera basis".into()));
    }
    timings.geometry_pack_seconds = pack_started.elapsed().as_secs_f64();
    let mut hash = Sha256::new();
    hash.update(serde_json::to_vec(job)?);
    hash.update(script.as_bytes());
    hash.update(super::backend::wgpu::bytes(&packed.data));
    hash.update(super::backend::wgpu::bytes(&p));
    for x in &packed.pixels {
        hash.update(x.to_le_bytes());
    }
    hash.update(include_str!("../backend/wgpu/shaders/path_trace.wgsl"));
    let signature = format!("{:x}", hash.finalize());
    let dir = job.output.join(&signature[..16]);
    std::fs::create_dir_all(&dir)?;
    let lock_path = dir.join("render.lock");
    let lock_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .map_err(|e| WeaverError::Invalid(format!("job lock {}: {e}", lock_path.display())))?;
    struct Lock {
        path: PathBuf,
        file: Option<std::fs::File>,
    }
    impl Drop for Lock {
        fn drop(&mut self) {
            // Close before unlinking, including on platforms that lock open files.
            drop(self.file.take());
            let _ = std::fs::remove_file(&self.path);
        }
    }
    let _guard = Lock {
        path: lock_path,
        file: Some(lock_file),
    };
    std::fs::write(
        dir.join("resolved-job.json"),
        serde_json::to_vec_pretty(job)?,
    )?;
    let gpu_started = Instant::now();
    let (gpu, frame_delta) = match cache.as_deref_mut() {
        Some(cache) => cache.gpu(&packed).await?,
        None => (
            Arc::new(Gpu::new(&packed).await?),
            FrameDeltaKind::FirstFrame,
        ),
    };
    timings.gpu_setup_seconds = gpu_started.elapsed().as_secs_f64();
    let shared_gpu = gpu.context();
    let mut report = RenderReport {
        renderer: "MotionLoom Weaver".into(),
        backend: "wgpu path tracer v0.1".into(),
        adapter: gpu.name.clone(),
        status: "rendering".into(),
        output: dir.clone(),
        triangles: packed.triangles,
        elapsed_seconds: 0.0,
        converged_pixels: 0,
        sample_limit_pixels: 0,
        diagnostics: snap.diagnostics,
        timings: timings.clone(),
        frame_delta,
    };
    report.diagnostics.push(format!(
        "Weaver and SceneCompositor share GPU adapter `{}` and one device/queue context.",
        shared_gpu.adapter_info().name
    ));
    report.diagnostics.push(match frame_delta {
        FrameDeltaKind::CameraOrUniforms =>
            "Sequence cache reused path-tracing pipelines and resident scene buffers; only frame uniforms and film changed.".into(),
        FrameDeltaKind::SceneBufferUpdate =>
            "Sequence cache retained the GPU device, pipelines and allocations; animated scene buffers were updated in place.".into(),
        _ =>
            "Sequence cache populated path-tracing pipelines, packed scene buffers and textures for subsequent frames.".into(),
    });
    if job.output_mode == SceneOutputMode::CompositeScene {
        report.diagnostics.push(format!(
            "SceneCompositor retained {} image-plane run(s) and composed linear-premultiplied RGBA16F; no RGBA8 display plate was created.",
            snap.composition_layers.len()
        ));
    }
    drop(packed);
    let [ox, oy, width, height] =
        job.region
            .unwrap_or([0, 0, job.resolution[0], job.resolution[1]]);
    let tiles_x = width.div_ceil(128);
    let tiles_y = height.div_ceil(128);
    let mut full = vec![0.0f32; width as usize * height as usize * FILM_FLOATS_PER_PIXEL];
    let checkpoints = dir.join("checkpoints");
    std::fs::create_dir_all(&checkpoints)?;
    struct TileWork {
        tx: u32,
        ty: u32,
        width: u32,
        height: u32,
        raw: Vec<u8>,
        file: PathBuf,
        tile: super::backend::wgpu::Tile,
        params: super::backend::wgpu::CameraParams,
    }
    let mut work = Vec::with_capacity((tiles_x * tiles_y) as usize);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let w = (width - tx * 128).min(128);
            let h = (height - ty * 128).min(128);
            let count = (w * h) as usize;
            let legacy_file = checkpoints.join(format!("{tx}-{ty}.film"));
            let file = checkpoints.join(format!("{tx}-{ty}.film-v2"));
            if legacy_file.exists() && !file.exists() {
                return Err(WeaverError::Invalid(format!(
                    "legacy 16-float checkpoint {} is incompatible with coverage/motion film v2",
                    legacy_file.display()
                )));
            }
            let raw = if file.exists() {
                std::fs::read(&file)?
            } else {
                vec![0; count * FILM_BYTES_PER_PIXEL]
            };
            if raw.len() != count * FILM_BYTES_PER_PIXEL {
                return Err(WeaverError::Invalid("truncated checkpoint".into()));
            }
            let tile = gpu.tile(count, &raw);
            let mut params = p;
            params[0][0] = w as f32;
            params[0][1] = h as f32;
            params[12] = [
                (ox + tx * 128) as f32,
                (oy + ty * 128) as f32,
                job.resolution[0] as f32,
                job.resolution[1] as f32,
            ];
            work.push(TileWork {
                tx,
                ty,
                width: w,
                height: h,
                raw,
                file,
                tile,
                params,
            });
        }
    }
    // Advance every active tile together. One GPU submit and one collective
    // readback per sample round keeps Metal busy and removes per-tile stalls.
    let trace_started = Instant::now();
    loop {
        if cancel.is_cancelled() {
            report.status = "cancelled".into();
            break;
        }
        let mut active = Vec::new();
        let mut global_minimum = u32::MAX;
        for (index, item) in work.iter().enumerate() {
            let values = output::floats(&item.raw);
            let mut tile_active = false;
            for (pixel_index, f) in values.chunks_exact(FILM_FLOATS_PER_PIXEL).enumerate() {
                if f.iter().any(|v| !v.is_finite())
                    || f[7] > 0.0
                    || f[3] < 0.0
                    || f[3].fract() != 0.0
                    || f[3] > job.sampling.max_samples as f32
                {
                    return Err(WeaverError::Gpu(format!(
                        "invalid sample at ({},{}), film={f:?}",
                        ox + item.tx * 128 + pixel_index as u32 % item.width,
                        oy + item.ty * 128 + pixel_index as u32 / item.width
                    )));
                }
                global_minimum = global_minimum.min(f[3] as u32);
                tile_active |=
                    !converged(f, &job.sampling) && f[3] < job.sampling.max_samples as f32;
            }
            if tile_active {
                active.push(index);
            }
        }
        progress(RenderProgress {
            completed_tiles: tiles_x * tiles_y - active.len() as u32,
            total_tiles: tiles_x * tiles_y,
            tile_min_samples: global_minimum,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        });
        if active.is_empty() {
            break;
        }
        let dispatches = active
            .iter()
            .map(|index| (&work[*index].tile, &work[*index].params))
            .collect::<Vec<_>>();
        gpu.dispatch_all(&dispatches)?;
        let tiles = active
            .iter()
            .map(|index| &work[*index].tile)
            .collect::<Vec<_>>();
        let readbacks = gpu.read_all(&tiles)?;
        for (index, raw) in active.into_iter().zip(readbacks) {
            work[index].raw = raw;
            // Rename only complete checkpoint writes; cancellation retains the last round.
            let tmp = work[index].file.with_extension("tmp");
            std::fs::write(&tmp, &work[index].raw)?;
            std::fs::rename(tmp, &work[index].file)?;
        }
    }
    if report.status != "cancelled" {
        for item in &work {
            let values = output::floats(&item.raw);
            for y in 0..item.height {
                for x in 0..item.width {
                    let source = ((y * item.width + x) as usize) * FILM_FLOATS_PER_PIXEL;
                    let target = (((item.ty * 128 + y) * width + item.tx * 128 + x) as usize)
                        * FILM_FLOATS_PER_PIXEL;
                    full[target..target + FILM_FLOATS_PER_PIXEL]
                        .copy_from_slice(&values[source..source + FILM_FLOATS_PER_PIXEL]);
                    if converged(
                        &values[source..source + FILM_FLOATS_PER_PIXEL],
                        &job.sampling,
                    ) {
                        report.converged_pixels += 1;
                    } else {
                        report.sample_limit_pixels += 1;
                    }
                }
            }
        }
    }
    timings.path_trace_seconds = trace_started.elapsed().as_secs_f64();
    if report.status != "cancelled" {
        let output_started = Instant::now();
        if job.output_mode == SceneOutputMode::CompositeScene {
            let composite = compose_frame(
                &shared_gpu,
                film_beauty([width, height], &full)?,
                &snap.composition_layers,
                &snap.lighting,
            )?;
            output::save_composited(&dir, [width, height], &full, &snap.lighting, &composite)?;
        } else {
            output::save(&dir, [width, height], &full, &snap.lighting)?;
        }
        timings.composition_output_seconds = output_started.elapsed().as_secs_f64();
        let denoise_started = Instant::now();
        if let Some(library) = &job.denoiser_library {
            let mut clean = super::denoise::run(library, [width, height], &full)?;
            if let Some(cache) = cache.as_deref_mut() {
                clean = cache.temporal_denoise([width, height], &full, clean);
            }
            if job.output_mode == SceneOutputMode::CompositeScene {
                let composite = compose_frame(
                    &shared_gpu,
                    denoised_beauty([width, height], &clean)?,
                    &snap.composition_layers,
                    &snap.lighting,
                )?;
                output::save_denoised_composited(
                    &dir.join("denoised"),
                    [width, height],
                    clean,
                    &snap.lighting,
                    &composite,
                )?;
            } else {
                output::save_denoised(
                    &dir.join("denoised"),
                    [width, height],
                    clean,
                    &snap.lighting,
                )?;
            }
            report.diagnostics.push(
                "Native auxiliary-guided HDR denoising applied; original beauty.exr preserved."
                    .into(),
            );
        } else {
            // GPU a-trous denoising is the default; it needs no host library.
            match gpu.denoise(width, height, &full, 4) {
                Ok(mut clean) => {
                    if let Some(cache) = cache.as_deref_mut() {
                        clean = cache.temporal_denoise([width, height], &full, clean);
                    }
                    if job.output_mode == SceneOutputMode::CompositeScene {
                        let composite = compose_frame(
                            &shared_gpu,
                            denoised_beauty([width, height], &clean)?,
                            &snap.composition_layers,
                            &snap.lighting,
                        )?;
                        output::save_denoised_composited(
                            &dir.join("denoised"),
                            [width, height],
                            clean,
                            &snap.lighting,
                            &composite,
                        )?;
                    } else {
                        output::save_denoised(
                            &dir.join("denoised"),
                            [width, height],
                            clean,
                            &snap.lighting,
                        )?;
                    }
                    report.diagnostics.push(
                        "GPU a-trous denoising applied from film AOVs; original beauty.exr preserved."
                            .into(),
                    );
                    if cache.as_ref().is_some_and(|cache| cache.temporal_enabled) {
                        report.diagnostics.push(
                            "Temporal denoising reprojected conservative history through motion, normal and albedo AOV rejection."
                                .into(),
                        );
                    }
                }
                Err(error) => {
                    report
                        .diagnostics
                        .push(format!("GPU denoising unavailable: {error}"));
                }
            }
        }
        timings.denoise_seconds = denoise_started.elapsed().as_secs_f64();
        report.status = if report.sample_limit_pixels > 0 {
            "sample_limit_reached"
        } else {
            "converged"
        }
        .into();
        progress(RenderProgress {
            completed_tiles: tiles_x * tiles_y,
            total_tiles: tiles_x * tiles_y,
            tile_min_samples: job.sampling.min_samples,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        });
        output::save_frame_manifest(
            &dir,
            job.frame,
            graph.fps,
            [width, height],
            job.output_mode == SceneOutputMode::CompositeScene,
            true,
            dir.join("denoised").is_dir(),
        )?;
    }
    report.elapsed_seconds = started.elapsed().as_secs_f64();
    report.timings = timings;
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn film_beauty(size: [u32; 2], film: &[f32]) -> Result<LinearPremultipliedImage, WeaverError> {
    let rgb = film
        .chunks_exact(FILM_FLOATS_PER_PIXEL)
        .map(|pixel| {
            let samples = pixel[3].max(1.0);
            [pixel[0] / samples, pixel[1] / samples, pixel[2] / samples]
        })
        .collect::<Vec<_>>();
    LinearPremultipliedImage::from_opaque_rgb(size, &rgb)
        .map_err(|error| WeaverError::Invalid(error.to_string()))
}

fn denoised_beauty(size: [u32; 2], rgb: &[f32]) -> Result<LinearPremultipliedImage, WeaverError> {
    let rgb = rgb
        .chunks_exact(3)
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect::<Vec<_>>();
    LinearPremultipliedImage::from_opaque_rgb(size, &rgb)
        .map_err(|error| WeaverError::Invalid(error.to_string()))
}

/// Execute scene-linear and display-linear stages on Weaver's own device.
fn compose_frame(
    context: &crate::scene::compositor::SceneGpuContext,
    beauty: LinearPremultipliedImage,
    layers: &[ResolvedCompositionLayer],
    lighting: &crate::world::WorldLighting,
) -> Result<CompositedFrame, WeaverError> {
    if layers.iter().any(|layer| layer.image.size != beauty.size) {
        return Err(WeaverError::Invalid(
            "composition layer size does not match Weaver output".into(),
        ));
    }
    let executor = WgpuSceneCompositorExecutor::new(context.clone());
    let pre = layers
        .iter()
        .filter(|layer| layer.color_stage == ColorStage::SceneLinearPreDisplay)
        .collect::<Vec<_>>();
    let scene_linear = compose_stage(&executor, &beauty, &pre)?;
    let display_pixels = scene_linear
        .pixels
        .iter()
        .map(|pixel| {
            let alpha = pixel[3].clamp(0.0, 1.0);
            let rgb = if alpha > 0.0 {
                [pixel[0] / alpha, pixel[1] / alpha, pixel[2] / alpha]
            } else {
                [0.0; 3]
            };
            let display = super::color::display_linear(rgb, lighting);
            [
                display[0] * alpha,
                display[1] * alpha,
                display[2] * alpha,
                alpha,
            ]
        })
        .collect();
    let display_base = LinearPremultipliedImage::new(beauty.size, display_pixels)
        .map_err(|error| WeaverError::Invalid(error.to_string()))?;
    let post = layers
        .iter()
        .filter(|layer| layer.color_stage == ColorStage::DisplayLinearPostTransform)
        .collect::<Vec<_>>();
    let display_linear = compose_stage(&executor, &display_base, &post)?;
    Ok(CompositedFrame {
        scene_linear,
        display_linear,
    })
}

fn compose_stage(
    executor: &WgpuSceneCompositorExecutor,
    base: &LinearPremultipliedImage,
    layers: &[&ResolvedCompositionLayer],
) -> Result<LinearPremultipliedImage, WeaverError> {
    if layers.is_empty() {
        return Ok(base.clone());
    }
    let mut textures = Vec::with_capacity(layers.len() + 1);
    textures.push(
        executor
            .upload(base)
            .map_err(|error| WeaverError::Gpu(error.to_string()))?,
    );
    for layer in layers {
        textures.push(
            executor
                .upload(&layer.image)
                .map_err(|error| WeaverError::Gpu(error.to_string()))?,
        );
    }
    let inputs = textures
        .iter()
        .map(|texture| GpuCompositionInput {
            texture,
            alpha: AlphaMode::Premultiplied,
        })
        .collect::<Vec<_>>();
    let output = executor.compose(base.size, &inputs);
    executor
        .readback(&output)
        .map_err(|error| WeaverError::Gpu(error.to_string()))
}

fn resolve_composition_plan(
    graph: &crate::GraphScript,
    scene_id: &str,
) -> Result<SceneCompositionPlan, WeaverError> {
    if !scene_id.is_empty() && scene_id != "auto" {
        return build_scene_composition_plan(graph, scene_id)
            .map_err(|error| WeaverError::Scene(error.to_string()));
    }
    let mut fallback = None;
    for scene in &graph.scenes {
        let plan = build_scene_composition_plan(graph, &scene.id)
            .map_err(|error| WeaverError::Scene(error.to_string()))?;
        if plan
            .layers
            .iter()
            .any(|layer| layer.domain == RenderDomain::ThreeD)
        {
            return Ok(plan);
        }
        fallback = Some(plan);
    }
    fallback.ok_or_else(|| WeaverError::Scene("no scene in graph".into()))
}

async fn render_pure_2d<F: FnMut(RenderProgress)>(
    job: &RenderJob,
    graph: &crate::GraphScript,
    script: &str,
    resolver: Arc<dyn AssetResolver>,
    started: Instant,
    progress: &mut F,
) -> Result<RenderReport, WeaverError> {
    let mut hash = Sha256::new();
    hash.update(serde_json::to_vec(job)?);
    hash.update(script.as_bytes());
    let signature = format!("{:x}", hash.finalize());
    let dir = job.output.join(&signature[..16]);
    std::fs::create_dir_all(&dir)?;
    let image =
        render_scene_graph_frame_with_resolver(graph, job.frame, SceneRenderProfile::Gpu, resolver)
            .await
            .map_err(|error| WeaverError::Scene(error.to_string()))?;
    output::save_pure_2d_master(&dir, &image)?;
    output::save_frame_manifest(
        &dir,
        job.frame,
        graph.fps,
        [image.width(), image.height()],
        true,
        false,
        false,
    )?;
    progress(RenderProgress {
        completed_tiles: 1,
        total_tiles: 1,
        tile_min_samples: 0,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    });
    let report = RenderReport {
        renderer: "MotionLoom SceneCompositor".into(),
        backend: "wgpu".into(),
        adapter: "shared scene compositor".into(),
        status: "converged".into(),
        output: dir.clone(),
        triangles: 0,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        converged_pixels: image.width() as u64 * image.height() as u64,
        sample_limit_pixels: 0,
        diagnostics: vec![
            "Pure 2D scene bypassed Weaver path tracing and used SceneCompositor.".into(),
        ],
        timings: super::api::RenderTimings {
            composition_output_seconds: started.elapsed().as_secs_f64(),
            ..Default::default()
        },
        frame_delta: FrameDeltaKind::TwoDOnly,
    };
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub(crate) fn converged(f: &[f32], s: &Sampling) -> bool {
    f[3] >= s.min_samples as f32
        && s.noise_threshold > 0.0
        && (f[5] / (f[3] - 1.0) / f[3]).max(0.0).sqrt() <= s.noise_threshold * f[4].abs().max(0.01)
}
