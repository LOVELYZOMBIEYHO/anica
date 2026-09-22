// =========================================
// =========================================
// crates/motionloom/src/weaver/preview/mod.rs

//! Progressive Weaver preview sessions.
//!
//! A session keeps the evaluated scene, packed geometry, GPU buffers and tile
//! films alive so a host can accumulate samples over time and display the
//! current image. It reuses the same lowering, camera and display transforms as
//! the final `render` job, so a preview differs from a final frame only by
//! resolution and sample count. The final render path is unchanged.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::weaver::backend::wgpu::{FILM_BYTES_PER_PIXEL, FILM_FLOATS_PER_PIXEL, Gpu, Tile, bytes};
use crate::weaver::config::RenderJob;
use crate::weaver::{WeaverError, camera, color, geometry, output};
use crate::world::WorldLighting;
use crate::{AssetResolver, AssetSource};

const TILE: u32 = 128;

// Look-development default keeps preview paths shorter than the authored final
// budget. `set_bounce_budget` raises it when a host wants parity with a final
// job; the final `render` path is never clamped.
const PREVIEW_TOTAL_BOUNCES: u32 = 8;
const PREVIEW_DIFFUSE_BOUNCES: u32 = 4;
const PREVIEW_GLOSSY_BOUNCES: u32 = 6;

struct RootResolver(PathBuf);
impl AssetResolver for RootResolver {
    fn resolve(&self, src: &str) -> Result<AssetSource, String> {
        let path = Path::new(src);
        Ok(AssetSource::Path(if path.is_absolute() {
            path.into()
        } else {
            self.0.join(path)
        }))
    }
}

struct SceneGpu {
    gpu: Gpu,
    p: crate::weaver::backend::wgpu::CameraParams,
    key: String,
    lighting: WorldLighting,
    diagnostics: Vec<String>,
    triangles: usize,
}

/// Build the evaluated scene, packed buffers and GPU resources for one job.
///
/// Mirrors the preparation steps of `jobs::render` without writing checkpoints,
/// a lock file or any output directory.
async fn build_scene_gpu(job: &RenderJob) -> Result<SceneGpu, WeaverError> {
    job.validate()?;
    let path = std::fs::canonicalize(&job.scene)?;
    let root = path.parent().unwrap();
    let script = std::fs::read_to_string(&path)?;
    let graph =
        crate::parse_graph_script(&script).map_err(|e| WeaverError::Scene(e.to_string()))?;
    if !graph.fps.is_finite()
        || graph.fps <= 0.0
        || job.frame as f64 / graph.fps as f64 >= graph.duration_ms as f64 / 1000.0
    {
        return Err(WeaverError::Invalid("frame outside scene duration".into()));
    }
    let mut snap =
        crate::scene::render::weaver_snapshot(&graph, job, Arc::new(RootResolver(root.into())))
            .await?;
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
    let mut packed = geometry::pack(&snap, job.texture_mips, job.allow_transmission_stopgap)?;
    snap.meshes.clear();
    let mut p = [[0.0f32; 4]; 26];
    p[0] = [
        0.0,
        0.0,
        job.sampling.batch_samples as f32,
        f32::from_bits(job.seed),
    ];
    p[1] = [
        // Raw u32 bit patterns; the shader bitcasts them back so large-scene
        // offsets survive the f32 uniform.
        f32::from_bits(packed.triangle_offset),
        f32::from_bits(packed.material_offset),
        f32::from_bits(packed.light_offset),
        packed.lights as f32,
    ];
    camera::configure(&mut p, &snap.camera, job)?;
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
    let preview_total = job.light_paths.total.min(PREVIEW_TOTAL_BOUNCES);
    let preview_diffuse = job.light_paths.diffuse.min(PREVIEW_DIFFUSE_BOUNCES);
    let preview_glossy = job.light_paths.glossy.min(PREVIEW_GLOSSY_BOUNCES);
    p[8] = [
        preview_total as f32,
        preview_diffuse as f32,
        preview_glossy as f32,
        job.light_paths.roulette_start.min(preview_total) as f32,
    ];
    p[9][3] = job.light_paths.transparent as f32;
    snap.diagnostics.push(format!(
        "Preview bounce budget: total {preview_total}, diffuse {preview_diffuse}, glossy {preview_glossy}; final render jobs keep the authored budget."
    ));
    p[14] = [
        f32::from_bits(packed.emitter_offset),
        packed.emitter_count as f32,
        packed.emitter_area,
        0.0,
    ];
    p[15][0] = job.sun_angular_diameter_degrees.to_radians() * 0.5;
    crate::weaver::lighting::pack_atmosphere(
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
        let (texture, cdf) = crate::weaver::lighting::environment(&env_path, &mut packed.data)?;
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
    if p.iter()
        .flatten()
        .enumerate()
        .any(|(i, v)| i != 3 && !v.is_finite())
    {
        return Err(WeaverError::Scene("degenerate camera basis".into()));
    }
    // The GPU key covers only what `Gpu::new` uploads, so camera-only changes can
    // reuse the retained buffers across frames.
    let mut hash = Sha256::new();
    hash.update(bytes(&packed.data));
    for x in &packed.pixels {
        hash.update(x.to_le_bytes());
    }
    let key = format!("{:x}", hash.finalize());
    let gpu = Gpu::new(&packed).await?;
    Ok(SceneGpu {
        gpu,
        p,
        key,
        lighting: snap.lighting,
        diagnostics: snap.diagnostics,
        triangles: packed.triangles,
    })
}

fn tone_map(rgb: &[f32], width: u32, height: u32, lighting: &WorldLighting) -> Vec<u8> {
    let mut out = vec![0u8; width as usize * height as usize * 4];
    for (index, pixel) in rgb.chunks_exact(3).enumerate() {
        let display = color::display([pixel[0], pixel[1], pixel[2]], lighting);
        out[index * 4..index * 4 + 3].copy_from_slice(&display);
        out[index * 4 + 3] = 255;
    }
    out
}

/// Retained progressive preview state for one scene and resolution.
pub struct PreviewSession {
    gpu: Gpu,
    tiles: Vec<Tile>,
    full: Vec<f32>,
    p: crate::weaver::backend::wgpu::CameraParams,
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    minimum_samples: u32,
    key: String,
    lighting: WorldLighting,
    diagnostics: Vec<String>,
    triangles: usize,
    bounce_override: Option<[u32; 3]>,
}

impl PreviewSession {
    /// Prepare a session at `job.resolution`. The host owns the executor.
    pub async fn new(job: &RenderJob) -> Result<Self, WeaverError> {
        let scene = build_scene_gpu(job).await?;
        let width = job.resolution[0].max(1);
        let height = job.resolution[1].max(1);
        let tiles = build_tiles(&scene.gpu, width, height);
        let full = vec![0.0f32; width as usize * height as usize * FILM_FLOATS_PER_PIXEL];
        Ok(Self {
            gpu: scene.gpu,
            tiles,
            full,
            p: scene.p,
            width,
            height,
            tiles_x: width.div_ceil(TILE),
            tiles_y: height.div_ceil(TILE),
            minimum_samples: 0,
            key: scene.key,
            lighting: scene.lighting,
            diagnostics: scene.diagnostics,
            triangles: scene.triangles,
            bounce_override: None,
        })
    }

    /// Raise the preview bounce budget to match a final job when needed.
    ///
    /// Values above the job budget still apply. Changing the estimator resets
    /// accumulated samples because samples from different budgets must not mix.
    pub fn set_bounce_budget(&mut self, total: u32, diffuse: u32, glossy: u32) {
        let total = total.max(1);
        let diffuse = diffuse.min(total);
        let glossy = glossy.min(total);
        let roulette = (self.p[8][3] as u32).min(total);
        self.p[8] = [total as f32, diffuse as f32, glossy as f32, roulette as f32];
        self.bounce_override = Some([total, diffuse, glossy]);
        self.reset_accumulation();
    }

    pub fn resolution(&self) -> [u32; 2] {
        [self.width, self.height]
    }

    /// Minimum completed samples across all preview pixels.
    pub fn samples(&self) -> u32 {
        self.minimum_samples
    }

    pub fn triangles(&self) -> usize {
        self.triangles
    }

    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }

    pub fn gpu_name(&self) -> &str {
        &self.gpu.name
    }

    /// Re-evaluate the document for a new frame, camera or style.
    ///
    /// Returns `true` when the retained GPU geometry had to be rebuilt, and
    /// always resets the accumulated film.
    pub async fn reload(&mut self, job: &RenderJob) -> Result<bool, WeaverError> {
        let scene = build_scene_gpu(job).await?;
        let resolution_changed =
            job.resolution[0].max(1) != self.width || job.resolution[1].max(1) != self.height;
        let geometry_changed = scene.key != self.key;
        if resolution_changed {
            self.width = job.resolution[0].max(1);
            self.height = job.resolution[1].max(1);
            self.tiles_x = self.width.div_ceil(TILE);
            self.tiles_y = self.height.div_ceil(TILE);
            self.full =
                vec![0.0f32; self.width as usize * self.height as usize * FILM_FLOATS_PER_PIXEL];
        }
        if geometry_changed || resolution_changed {
            self.gpu = scene.gpu;
            self.key = scene.key;
        }
        // Camera and environment parameters travel in `p` for every reload.
        self.p = scene.p;
        if let Some([total, diffuse, glossy]) = self.bounce_override {
            let roulette = (self.p[8][3] as u32).min(total);
            self.p[8] = [total as f32, diffuse as f32, glossy as f32, roulette as f32];
        }
        self.lighting = scene.lighting;
        self.diagnostics = scene.diagnostics;
        self.triangles = scene.triangles;
        // Samples from the previous frame or camera must never mix with the new one.
        self.reset_accumulation();
        Ok(geometry_changed || resolution_changed)
    }

    /// Accumulate `samples` more samples per pixel and refresh the CPU image.
    ///
    /// Every tile is dispatched first and the whole frame is read back once, so
    /// GPU idle/readback overhead does not scale with the tile count.
    /// Returns the new minimum sample count across the frame.
    pub fn advance(&mut self, samples: u32) -> Result<u32, WeaverError> {
        if samples == 0 {
            return Ok(self.minimum_samples);
        }
        const MAX_DISPATCH_SAMPLES: u32 = 8;
        const MAX_DISPATCHES: u32 = 8;
        let dispatches = samples
            .div_ceil(MAX_DISPATCH_SAMPLES)
            .clamp(1, MAX_DISPATCHES);
        let per_dispatch = samples.div_ceil(dispatches);
        let debug = std::env::var_os("WEAVER_PREVIEW_DEBUG").is_some();
        let started = std::time::Instant::now();
        let mut p = self.p;
        p[0][2] = per_dispatch as f32;
        for ty in 0..self.tiles_y {
            for tx in 0..self.tiles_x {
                let index = (ty * self.tiles_x + tx) as usize;
                p[0][0] = (self.width - tx * TILE).min(TILE) as f32;
                p[0][1] = (self.height - ty * TILE).min(TILE) as f32;
                p[12] = [
                    (tx * TILE) as f32,
                    (ty * TILE) as f32,
                    self.width as f32,
                    self.height as f32,
                ];
                for _ in 0..dispatches {
                    self.gpu.dispatch(&self.tiles[index], &p)?;
                }
            }
        }
        let dispatched = started.elapsed().as_secs_f64() * 1000.0;
        let raws = {
            let tiles = self.tiles.iter().collect::<Vec<_>>();
            self.gpu.read_all(&tiles)?
        };
        let read = started.elapsed().as_secs_f64() * 1000.0 - dispatched;
        let mut minimum = u32::MAX;
        for ty in 0..self.tiles_y {
            for tx in 0..self.tiles_x {
                let index = (ty * self.tiles_x + tx) as usize;
                let w = (self.width - tx * TILE).min(TILE);
                let values = output::floats(&raws[index]);
                for (pixel, f) in values.chunks_exact(FILM_FLOATS_PER_PIXEL).enumerate() {
                    minimum = minimum.min(f[3] as u32);
                    let x = pixel as u32 % w;
                    let y = pixel as u32 / w;
                    let target = (((ty * TILE + y) * self.width + tx * TILE + x) as usize)
                        * FILM_FLOATS_PER_PIXEL;
                    self.full[target..target + FILM_FLOATS_PER_PIXEL]
                        .copy_from_slice(&f[..FILM_FLOATS_PER_PIXEL]);
                }
            }
        }
        self.minimum_samples = if minimum == u32::MAX { 0 } else { minimum };
        if debug {
            eprintln!(
                "preview advance: spp +{per_dispatch}x{dispatches} dispatch {dispatched:.1} ms read {read:.1} ms tiles {}",
                self.tiles.len()
            );
        }
        Ok(self.minimum_samples)
    }

    /// Drop accumulated samples and start the films over.
    pub fn reset_accumulation(&mut self) {
        self.tiles = build_tiles(&self.gpu, self.width, self.height);
        self.full.fill(0.0);
        self.minimum_samples = 0;
    }

    /// Tone-mapped RGBA image of the current accumulation.
    pub fn display_rgba(&self) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(self.full.len() / FILM_FLOATS_PER_PIXEL * 3);
        for f in self.full.chunks_exact(FILM_FLOATS_PER_PIXEL) {
            let n = f[3].max(1.0);
            rgb.extend_from_slice(&[f[0] / n, f[1] / n, f[2] / n]);
        }
        tone_map(&rgb, self.width, self.height, &self.lighting)
    }

    /// Raw accumulated film (20 floats per pixel) for denoising and AOV export.
    pub fn film(&self) -> &[f32] {
        &self.full
    }

    /// GPU-denoised, tone-mapped RGBA image of the current accumulation.
    ///
    /// Uses the film AOVs (albedo, normal, depth, variance) as edge guides.
    /// `passes` is the number of a-trous iterations; 3 covers steps 1, 2, 4.
    pub fn denoise_rgba(&self, passes: u32) -> Result<Vec<u8>, WeaverError> {
        let rgb = self
            .gpu
            .denoise(self.width, self.height, &self.full, passes)?;
        Ok(tone_map(&rgb, self.width, self.height, &self.lighting))
    }
}

fn build_tiles(gpu: &Gpu, width: u32, height: u32) -> Vec<Tile> {
    let tiles_x = width.div_ceil(TILE);
    let tiles_y = height.div_ceil(TILE);
    let mut tiles = Vec::with_capacity((tiles_x * tiles_y) as usize);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let w = (width - tx * TILE).min(TILE);
            let h = (height - ty * TILE).min(TILE);
            tiles.push(gpu.tile(
                (w * h) as usize,
                &vec![0u8; (w * h) as usize * FILM_BYTES_PER_PIXEL],
            ));
        }
    }
    tiles
}

/// Reusable OIDN denoiser for preview updates.
pub struct PreviewDenoiser(crate::weaver::denoise::Denoiser);

impl PreviewDenoiser {
    /// Load a trusted native OIDN library once for the whole preview session.
    pub fn new(library: &Path) -> Result<Self, WeaverError> {
        Ok(Self(crate::weaver::denoise::Denoiser::new(library)?))
    }

    /// Denoise the session film and return a tone-mapped RGBA image.
    pub fn denoise_rgba(&self, session: &PreviewSession) -> Result<Vec<u8>, WeaverError> {
        let clean = self.0.run(session.resolution(), session.film())?;
        let mut rgb = clean;
        rgb.truncate(session.full.len() / 16 * 3);
        Ok(tone_map(
            &rgb,
            session.width,
            session.height,
            &session.lighting,
        ))
    }
}
