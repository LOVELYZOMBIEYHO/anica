// =========================================
// =========================================
// crates/motionloom/src/weaver/jobs/mod.rs

use super::api::{CancellationToken, RenderProgress, RenderReport};
use super::{WeaverError, backend::wgpu::Gpu, config::*, geometry, output};
pub(crate) mod sequence;
use crate::{AssetResolver, AssetSource};
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

/// Render one evaluated frame. The host owns the executor and cancellation token.
/// Checkpoints are keyed by settings, shader, geometry, lighting and texture bytes.
pub async fn render<F: FnMut(RenderProgress)>(
    job: &RenderJob,
    cancel: &CancellationToken,
    mut progress: F,
) -> Result<RenderReport, WeaverError> {
    job.validate()?;
    let started = Instant::now();
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
    let mut packed = geometry::pack(&snap)?;
    // Packed GPU data owns the required values; release duplicate world meshes.
    snap.meshes.clear();
    let mut p = [[0.0f32; 4]; 20];
    p[0] = [
        0.0,
        0.0,
        job.sampling.batch_samples as f32,
        f32::from_bits(job.seed),
    ];
    p[1] = [
        packed.triangle_offset as f32,
        packed.material_offset as f32,
        packed.light_offset as f32,
        packed.lights as f32,
    ];
    super::camera::configure(&mut p, &snap.camera, job)?;
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
        packed.emitter_offset as f32,
        packed.emitter_count as f32,
        packed.emitter_area,
        0.0,
    ];
    p[15][0] = job.sun_angular_diameter_degrees.to_radians() * 0.5;
    if let Some(v) = &job.volume {
        p[16] = [
            v.bounds_min[0],
            v.bounds_min[1],
            v.bounds_min[2],
            v.extinction,
        ];
        p[17] = [
            v.bounds_max[0],
            v.bounds_max[1],
            v.bounds_max[2],
            v.anisotropy,
        ];
        p[18] = [v.albedo[0], v.albedo[1], v.albedo[2], v.max_bounces as f32];
    }
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
        snap.diagnostics.push("Environment uses float radiance, solid-angle importance sampling and MIS. Background blur is not applied.".into());
    }
    if p.iter()
        .flatten()
        .enumerate()
        .any(|(i, v)| i != 3 && !v.is_finite())
    {
        return Err(WeaverError::Scene("degenerate camera basis".into()));
    }
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
    let gpu = Gpu::new(&packed).await?;
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
    };
    drop(packed);
    let [ox, oy, width, height] =
        job.region
            .unwrap_or([0, 0, job.resolution[0], job.resolution[1]]);
    let tiles_x = width.div_ceil(128);
    let tiles_y = height.div_ceil(128);
    let mut full = vec![0.0f32; width as usize * height as usize * 16];
    let checkpoints = dir.join("checkpoints");
    std::fs::create_dir_all(&checkpoints)?;
    'tiles: for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            if cancel.is_cancelled() {
                report.status = "cancelled".into();
                break 'tiles;
            }
            let w = (width - tx * 128).min(128);
            let h = (height - ty * 128).min(128);
            let count = (w * h) as usize;
            let file = checkpoints.join(format!("{tx}-{ty}.film"));
            let mut raw = if file.exists() {
                std::fs::read(&file)?
            } else {
                vec![0; count * 64]
            };
            if raw.len() != count * 64 {
                return Err(WeaverError::Invalid("truncated checkpoint".into()));
            }
            let tile = gpu.tile(count, &raw);
            p[0][0] = w as f32;
            p[0][1] = h as f32;
            p[12] = [
                (ox + tx * 128) as f32,
                (oy + ty * 128) as f32,
                job.resolution[0] as f32,
                job.resolution[1] as f32,
            ];
            loop {
                let values = output::floats(&raw);
                let mut active = false;
                let mut minimum = u32::MAX;
                for (pixel_index, f) in values.chunks_exact(16).enumerate() {
                    if f.iter().any(|v| !v.is_finite())
                        || f[7] > 0.0
                        || f[3] < 0.0
                        || f[3].fract() != 0.0
                        || f[3] > job.sampling.max_samples as f32
                    {
                        return Err(WeaverError::Gpu(format!(
                            "invalid sample at ({},{}), film={f:?}",
                            ox + tx * 128 + pixel_index as u32 % w,
                            oy + ty * 128 + pixel_index as u32 / w
                        )));
                    }
                    minimum = minimum.min(f[3] as u32);
                    active |=
                        !converged(f, &job.sampling) && f[3] < job.sampling.max_samples as f32;
                }
                progress(RenderProgress {
                    completed_tiles: ty * tiles_x + tx,
                    total_tiles: tiles_x * tiles_y,
                    tile_min_samples: minimum,
                    elapsed_seconds: started.elapsed().as_secs_f64(),
                });
                if !active {
                    break;
                }
                if cancel.is_cancelled() {
                    report.status = "cancelled".into();
                    break 'tiles;
                }
                raw = gpu.batch(&tile, &p)?;
                // Rename only complete checkpoint writes; cancellation retains the last batch.
                let tmp = file.with_extension("tmp");
                std::fs::write(&tmp, &raw)?;
                std::fs::rename(tmp, &file)?;
            }
            let values = output::floats(&raw);
            for y in 0..h {
                for x in 0..w {
                    let source = ((y * w + x) * 16) as usize;
                    let target = (((ty * 128 + y) * width + tx * 128 + x) * 16) as usize;
                    full[target..target + 16].copy_from_slice(&values[source..source + 16]);
                    if converged(&values[source..source + 16], &job.sampling) {
                        report.converged_pixels += 1;
                    } else {
                        report.sample_limit_pixels += 1;
                    }
                }
            }
        }
    }
    if report.status != "cancelled" {
        output::save(&dir, [width, height], &full, &snap.lighting)?;
        if let Some(library) = &job.denoiser_library {
            let clean = super::denoise::run(library, [width, height], &full)?;
            output::save_denoised(
                &dir.join("denoised"),
                [width, height],
                clean,
                &snap.lighting,
            )?;
            report.diagnostics.push(
                "Native auxiliary-guided HDR denoising applied; original beauty.exr preserved."
                    .into(),
            );
        } else {
            report
                .diagnostics
                .push("Denoising disabled: no host-provided denoiser_library.".into());
        }
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
    }
    report.elapsed_seconds = started.elapsed().as_secs_f64();
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub(crate) fn converged(f: &[f32], s: &Sampling) -> bool {
    f[3] >= s.min_samples as f32
        && s.noise_threshold > 0.0
        && (f[5] / (f[3] - 1.0) / f[3]).max(0.0).sqrt() <= s.noise_threshold * f[4].abs().max(0.01)
}
