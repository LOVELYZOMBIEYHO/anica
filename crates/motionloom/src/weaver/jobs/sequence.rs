// =========================================
// =========================================
// crates/motionloom/src/weaver/jobs/sequence.rs

use crate::{
    audio::{compile_audio_plan, prepare_audio},
    weaver::{
        CancellationToken, FrameDeltaKind, MasterColorTarget, MasterSequenceSettings, RenderJob,
        RenderProgress, RenderReport, RenderTimings, SceneOutputMode, SequenceCheckpointRetention,
        SequenceDenoiseMode, SequenceProgress, WeaverError,
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    ops::RangeInclusive,
    path::{Path, PathBuf},
    process::Command,
};

/// Completed sequence paths and the durable manifest used for resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MasterSequenceReport {
    pub status: String,
    pub output: PathBuf,
    pub frame_count: u32,
    pub rendered_frames: u32,
    pub resumed_frames: u32,
    pub display_master_frames: PathBuf,
    pub scene_composite_frames: Option<PathBuf>,
    pub audio_master: Option<PathBuf>,
    pub prores_master: Option<PathBuf>,
    pub preview_movie: Option<PathBuf>,
    pub manifest: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MasterSequenceManifest {
    version: u32,
    status: String,
    source: PathBuf,
    source_sha256: String,
    sequence_sha256: String,
    scene_id: String,
    frame_range: [u32; 2],
    frame_count: u32,
    fps: [u32; 2],
    resolution: [u32; 2],
    color_target: MasterColorTarget,
    master_pixel_format: String,
    alpha_mode: String,
    frames: Vec<MasterFrameRecord>,
    outputs: MasterSequenceOutputs,
    validation: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MasterFrameRecord {
    frame: u32,
    source_time_seconds: f64,
    signature: String,
    status: String,
    resumed: bool,
    display_master: PathBuf,
    scene_composite: Option<PathBuf>,
    render_report: Option<PathBuf>,
    timings: Option<RenderTimings>,
    frame_delta: FrameDeltaKind,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MasterSequenceOutputs {
    display_master_sequence: PathBuf,
    scene_composite_sequence: Option<PathBuf>,
    audio_master: Option<PathBuf>,
    prores_master: Option<PathBuf>,
    preview_movie: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameResumeRecord {
    signature: String,
    report: Option<PathBuf>,
    timings: Option<RenderTimings>,
    #[serde(default)]
    frame_delta: FrameDeltaKind,
}

/// Sequential frame export shares the job contract and cancellation. Per-frame
/// checkpoints remain resumable; cross-frame acceleration caching is separate work.
pub async fn render_sequence<F: FnMut(SequenceProgress)>(
    job: &RenderJob,
    frames: RangeInclusive<u32>,
    cancel: &CancellationToken,
    mut progress: F,
) -> Result<Vec<RenderReport>, WeaverError> {
    let total = checked_frame_count(&frames)?;
    let mut reports = Vec::new();
    let mut render_cache = super::RenderCache::default();
    for frame in frames {
        if cancel.is_cancelled() {
            break;
        }
        let mut frame_job = job.clone();
        frame_job.frame = frame;
        let completed = reports.len() as u32;
        let report = super::render_cached(
            &frame_job,
            cancel,
            |render| {
                progress(SequenceProgress {
                    frame,
                    completed_frames: completed,
                    total_frames: total,
                    render,
                })
            },
            &mut render_cache,
        )
        .await?;
        let cancelled = report.status == "cancelled";
        reports.push(report);
        if cancelled {
            break;
        }
    }
    Ok(reports)
}

/// Render a compositor-complete EXR sequence, resume verified frames, then
/// derive professional and review movies from exactly the same display masters.
pub async fn render_master_sequence<F: FnMut(SequenceProgress)>(
    job: &RenderJob,
    frames: RangeInclusive<u32>,
    settings: &MasterSequenceSettings,
    cancel: &CancellationToken,
    mut progress: F,
) -> Result<MasterSequenceReport, WeaverError> {
    let total = checked_frame_count(&frames)?;
    if job.output_mode != SceneOutputMode::CompositeScene {
        return Err(WeaverError::Invalid(
            "master sequence requires output_mode=composite_scene".into(),
        ));
    }
    if settings.version != 1 {
        return Err(WeaverError::Invalid(format!(
            "unsupported master sequence settings version {}",
            settings.version
        )));
    }
    let canonical_scene = fs::canonicalize(&job.scene)?;
    let script = fs::read_to_string(&canonical_scene)?;
    let graph = crate::parse_graph_script(&script)
        .map_err(|error| WeaverError::Scene(error.to_string()))?;
    let fps = fps_rational(graph.fps)?;
    let source_hash = digest_bytes(script.as_bytes());
    let mut sequence_job = job.clone();
    sequence_job.scene = canonical_scene.clone();
    sequence_job.frame = *frames.start();
    let sequence_hash = digest_serialized(&(sequence_job.clone(), frames.clone(), settings))?;
    let root = job.output.join("sequences").join(&sequence_hash[..16]);
    let display_dir = root.join("frames/display-master");
    let scene_dir = root.join("frames/scene-composite");
    let resume_dir = root.join("frames/resume");
    let checkpoint_dir = root.join("checkpoints");
    fs::create_dir_all(&display_dir)?;
    fs::create_dir_all(&resume_dir)?;
    if settings.write_scene_composite {
        fs::create_dir_all(&scene_dir)?;
    }

    let manifest_path = root.join("sequence-manifest.json");
    let mut manifest = MasterSequenceManifest {
        version: 1,
        status: "rendering".into(),
        source: canonical_scene.clone(),
        source_sha256: source_hash,
        sequence_sha256: sequence_hash,
        scene_id: job.scene_id.clone(),
        frame_range: [*frames.start(), *frames.end()],
        frame_count: total,
        fps,
        resolution: job.resolution,
        color_target: settings.color_target,
        master_pixel_format: "rgba16f".into(),
        alpha_mode: "premultiplied".into(),
        frames: Vec::with_capacity(total as usize),
        outputs: MasterSequenceOutputs {
            display_master_sequence: display_dir.clone(),
            scene_composite_sequence: settings.write_scene_composite.then(|| scene_dir.clone()),
            ..MasterSequenceOutputs::default()
        },
        validation: None,
    };
    write_json_atomic(&manifest_path, &manifest)?;

    let mut rendered_frames = 0;
    let mut resumed_frames = 0;
    let mut render_cache = super::RenderCache::default();
    render_cache.set_temporal_denoise(settings.denoise_mode == SequenceDenoiseMode::Temporal);
    for frame in frames.clone() {
        if cancel.is_cancelled() {
            manifest.status = "cancelled".into();
            write_json_atomic(&manifest_path, &manifest)?;
            return Ok(report_from_manifest(
                &manifest,
                &root,
                rendered_frames,
                resumed_frames,
                &manifest_path,
            ));
        }
        let name = format!("{frame:06}.exr");
        let display_target = display_dir.join(&name);
        let scene_target = scene_dir.join(&name);
        let resume_path = resume_dir.join(format!("{frame:06}.json"));
        let mut frame_job = job.clone();
        frame_job.scene = canonical_scene.clone();
        frame_job.frame = frame;
        frame_job.output = checkpoint_dir.clone();
        let signature = digest_serialized(&(frame_job.clone(), &script))?;
        let resume = read_resume_record(&resume_path);
        let reusable = resume
            .as_ref()
            .is_some_and(|record| record.signature == signature)
            && display_target.is_file()
            && (!settings.write_scene_composite || scene_target.is_file())
            // Temporal history is sequence state. Until history sidecars are
            // persisted, rerender the requested range to keep results exact.
            && settings.denoise_mode == SequenceDenoiseMode::Independent;
        let (render_report, frame_timings, frame_delta) = if reusable {
            resumed_frames += 1;
            progress(SequenceProgress {
                frame,
                completed_frames: manifest.frames.len() as u32,
                total_frames: total,
                render: RenderProgress {
                    completed_tiles: 1,
                    total_tiles: 1,
                    tile_min_samples: frame_job.sampling.min_samples,
                    elapsed_seconds: 0.0,
                },
            });
            resume
                .map(|record| (record.report, record.timings, record.frame_delta))
                .unwrap_or((None, None, FrameDeltaKind::FirstFrame))
        } else {
            let completed = manifest.frames.len() as u32;
            let report = super::render_cached(
                &frame_job,
                cancel,
                |render| {
                    progress(SequenceProgress {
                        frame,
                        completed_frames: completed,
                        total_frames: total,
                        render,
                    })
                },
                &mut render_cache,
            )
            .await?;
            if report.status == "cancelled" {
                manifest.status = "cancelled".into();
                write_json_atomic(&manifest_path, &manifest)?;
                return Ok(report_from_manifest(
                    &manifest,
                    &root,
                    rendered_frames,
                    resumed_frames,
                    &manifest_path,
                ));
            }
            let denoised = report.output.join("denoised");
            let master_source = if denoised.join("display-master.exr").is_file() {
                denoised.as_path()
            } else {
                report.output.as_path()
            };
            atomic_copy(&master_source.join("display-master.exr"), &display_target)?;
            if settings.write_scene_composite {
                atomic_copy(&master_source.join("scene-composite.exr"), &scene_target)?;
            }
            let report_path = report.output.join("report.json");
            let keep_jobs = settings.checkpoint_retention == SequenceCheckpointRetention::All;
            write_json_atomic(
                &resume_path,
                &FrameResumeRecord {
                    signature: signature.clone(),
                    report: keep_jobs.then_some(report_path.clone()),
                    timings: Some(report.timings.clone()),
                    frame_delta: report.frame_delta,
                },
            )?;
            if !keep_jobs {
                fs::remove_dir_all(&report.output)?;
            }
            rendered_frames += 1;
            (
                keep_jobs.then_some(report_path),
                Some(report.timings.clone()),
                report.frame_delta,
            )
        };
        manifest.frames.push(MasterFrameRecord {
            frame,
            source_time_seconds: frame as f64 * fps[1] as f64 / fps[0] as f64,
            signature,
            status: "complete".into(),
            resumed: reusable,
            display_master: display_target,
            scene_composite: settings.write_scene_composite.then_some(scene_target),
            timings: frame_timings
                .or_else(|| render_report.as_deref().and_then(read_render_timings)),
            frame_delta,
            render_report,
        });
        write_json_atomic(&manifest_path, &manifest)?;
    }

    manifest.status = "encoding".into();
    write_json_atomic(&manifest_path, &manifest)?;
    let ffmpeg = resolve_runtime_binary("ANICA_FFMPEG_PATH", "ffmpeg", &canonical_scene);
    let ffprobe = resolve_runtime_binary("ANICA_FFPROBE_PATH", "ffprobe", &canonical_scene);
    let duration = total as f64 * fps[1] as f64 / fps[0] as f64;
    if settings.include_audio {
        let audio_plan =
            compile_audio_plan(&graph).map_err(|error| WeaverError::Scene(error.to_string()))?;
        if !audio_plan.clips.is_empty() {
            let full_duration = (*frames.end() as f64 + 1.0) * fps[1] as f64 / fps[0] as f64;
            let prepared = prepare_audio(
                &ffmpeg.to_string_lossy(),
                audio_plan,
                canonical_scene.parent().unwrap_or_else(|| Path::new(".")),
                full_duration,
            )
            .map_err(|error| WeaverError::Invalid(error.to_string()))?;
            let audio_master = root.join("audio-master.wav");
            encode_audio_master(
                &ffmpeg,
                &prepared.path(),
                &audio_master,
                *frames.start() as f64 * fps[1] as f64 / fps[0] as f64,
                duration,
            )?;
            manifest.outputs.audio_master = Some(audio_master);
        }
    }
    if settings.encode_prores {
        let target = root.join("display-master-prores4444xq.mov");
        encode_video(
            &ffmpeg,
            VideoKind::ProRes4444Xq,
            &display_dir,
            *frames.start(),
            total,
            fps,
            manifest.outputs.audio_master.as_deref(),
            &target,
        )?;
        manifest.outputs.prores_master = Some(target);
    }
    if settings.encode_preview {
        let target = root.join("preview.mp4");
        encode_video(
            &ffmpeg,
            VideoKind::PreviewH264,
            &display_dir,
            *frames.start(),
            total,
            fps,
            manifest.outputs.audio_master.as_deref(),
            &target,
        )?;
        manifest.outputs.preview_movie = Some(target);
    }
    manifest.validation = Some(validate_delivery(&ffmpeg, &ffprobe, &manifest)?);
    manifest.status = "complete".into();
    write_json_atomic(&manifest_path, &manifest)?;
    Ok(report_from_manifest(
        &manifest,
        &root,
        rendered_frames,
        resumed_frames,
        &manifest_path,
    ))
}

fn checked_frame_count(frames: &RangeInclusive<u32>) -> Result<u32, WeaverError> {
    if frames.is_empty() {
        return Err(WeaverError::Invalid("empty frame range".into()));
    }
    frames
        .end()
        .checked_sub(*frames.start())
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| WeaverError::Invalid("frame range overflow".into()))
}

fn fps_rational(fps: f32) -> Result<[u32; 2], WeaverError> {
    if !fps.is_finite() || fps <= 0.0 || fps > 240.0 {
        return Err(WeaverError::Invalid(
            "fps must be finite and in 0..=240".into(),
        ));
    }
    for (value, rational) in [
        (23.976, [24_000, 1_001]),
        (29.97, [30_000, 1_001]),
        (59.94, [60_000, 1_001]),
    ] {
        if (fps - value).abs() < 0.001 {
            return Ok(rational);
        }
    }
    let numerator = (fps * 1_000.0).round() as u32;
    let divisor = gcd(numerator, 1_000);
    Ok([numerator / divisor, 1_000 / divisor])
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest_serialized(value: &impl Serialize) -> Result<String, WeaverError> {
    Ok(digest_bytes(&serde_json::to_vec(value)?))
}

fn read_resume_record(path: &Path) -> Option<FrameResumeRecord> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn read_render_timings(path: &Path) -> Option<RenderTimings> {
    serde_json::from_slice::<RenderReport>(&fs::read(path).ok()?)
        .ok()
        .map(|report| report.timings)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), WeaverError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn atomic_copy(source: &Path, target: &Path) -> Result<(), WeaverError> {
    if !source.is_file() {
        return Err(WeaverError::Invalid(format!(
            "render did not produce {}",
            source.display()
        )));
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = target.with_extension("exr.tmp");
    fs::copy(source, &temporary)?;
    fs::rename(temporary, target)?;
    Ok(())
}

fn resolve_runtime_binary(environment: &str, binary: &str, scene: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os(environment) {
        return PathBuf::from(path);
    }
    for ancestor in scene.ancestors() {
        for candidate in [
            ancestor
                .join("anica/tools/runtime/current/macos/ffmpeg/bin")
                .join(binary),
            ancestor
                .join("tools/runtime/current/macos/ffmpeg/bin")
                .join(binary),
        ] {
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from(binary)
}

fn encode_audio_master(
    ffmpeg: &Path,
    source: &Path,
    target: &Path,
    start: f64,
    duration: f64,
) -> Result<(), WeaverError> {
    let temporary = target.with_file_name("audio-master.tmp.wav");
    run_command(
        Command::new(ffmpeg)
            .args(["-y", "-hide_banner", "-loglevel", "error", "-ss"])
            .arg(format!("{start:.9}"))
            .arg("-i")
            .arg(source)
            .arg("-t")
            .arg(format!("{duration:.9}"))
            .args(["-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le"])
            .arg(&temporary),
        "audio master",
    )?;
    fs::rename(temporary, target)?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum VideoKind {
    ProRes4444Xq,
    PreviewH264,
}

fn encode_video(
    ffmpeg: &Path,
    kind: VideoKind,
    frames: &Path,
    start: u32,
    count: u32,
    fps: [u32; 2],
    audio: Option<&Path>,
    target: &Path,
) -> Result<(), WeaverError> {
    let temporary = match kind {
        VideoKind::ProRes4444Xq => target.with_file_name("display-master-prores4444xq.tmp.mov"),
        VideoKind::PreviewH264 => target.with_file_name("preview.tmp.mp4"),
    };
    let pattern = frames.join("%06d.exr");
    let mut command = Command::new(ffmpeg);
    command
        .args(["-y", "-hide_banner", "-loglevel", "error", "-framerate"])
        .arg(format!("{}/{}", fps[0], fps[1]))
        .args(["-start_number", &start.to_string(), "-i"])
        .arg(pattern);
    if let Some(audio) = audio {
        command.arg("-i").arg(audio);
    }
    command.args(["-frames:v", &count.to_string()]);
    // FFmpeg's colorspace filter performs the linear-light transfer correctly.
    // ProRes reattaches the independently converted alpha plane afterwards.
    let bt709_color = "colorspace=ispace=gbr:iprimaries=bt709:itrc=linear:irange=pc:space=bt709:primaries=bt709:trc=bt709:range=tv";
    let bt709_metadata =
        "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv";
    match kind {
        VideoKind::ProRes4444Xq => {
            let filter = format!(
                "[0:v]split=2[color][alpha];[color]{bt709_color}:format=yuv444p10[color709];[alpha]format=rgba64le,alphaextract,format=gray16le[alpha16];[color709][alpha16]alphamerge,format=yuva444p10le,{bt709_metadata}[v]"
            );
            command
                .args(["-filter_complex", &filter, "-map", "[v]"])
                .args([
                    "-c:v",
                    "prores_ks",
                    "-profile:v",
                    "5",
                    "-vendor",
                    "apl0",
                    "-alpha_bits",
                    "16",
                    "-bits_per_mb",
                    "8000",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-colorspace",
                    "bt709",
                ]);
            if audio.is_some() {
                command.args(["-c:a", "pcm_s24le"]);
            }
        }
        VideoKind::PreviewH264 => {
            command
                .args([
                    "-vf",
                    &format!("{bt709_color}:format=yuv420p,{bt709_metadata}"),
                ])
                .args(["-map", "0:v:0"])
                .args([
                    "-c:v",
                    preview_encoder(),
                    "-b:v",
                    "18M",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-colorspace",
                    "bt709",
                ]);
            #[cfg(target_os = "macos")]
            command.args(["-allow_sw", "1"]);
            if audio.is_some() {
                command.args(["-c:a", "aac", "-b:a", "256k"]);
            }
            command.args(["-movflags", "+faststart"]);
        }
    }
    if audio.is_some() {
        command.args(["-map", "1:a:0", "-ar", "48000", "-ac", "2"]);
    } else {
        command.arg("-an");
    }
    command.arg(&temporary);
    run_command(&mut command, "video encode")?;
    fs::rename(temporary, target)?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn preview_encoder() -> &'static str {
    "h264_videotoolbox"
}
#[cfg(target_os = "windows")]
fn preview_encoder() -> &'static str {
    "h264_mf"
}
#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
fn preview_encoder() -> &'static str {
    "libopenh264"
}

fn run_command(command: &mut Command, label: &str) -> Result<(), WeaverError> {
    let output = command.output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(WeaverError::Invalid(format!(
            "{label} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

fn validate_delivery(
    ffmpeg: &Path,
    ffprobe: &Path,
    manifest: &MasterSequenceManifest,
) -> Result<serde_json::Value, WeaverError> {
    validate_master_frames(manifest)?;
    let mut movies = serde_json::Map::new();
    for (name, path, codec) in [
        (
            "proresMaster",
            manifest.outputs.prores_master.as_ref(),
            "prores",
        ),
        (
            "previewMovie",
            manifest.outputs.preview_movie.as_ref(),
            "h264",
        ),
    ] {
        let Some(path) = path else { continue };
        let output = Command::new(ffprobe).args([
            "-v", "error", "-count_frames", "-show_entries",
            "stream=index,codec_type,codec_name,profile,pix_fmt,width,height,color_space,color_transfer,color_primaries,avg_frame_rate,nb_read_frames,sample_rate,channels:format=duration",
            "-of", "json",
        ]).arg(path).output()?;
        if !output.status.success() {
            return Err(WeaverError::Invalid(format!(
                "ffprobe failed for {}: {}",
                path.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        let mut probe: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let streams = probe["streams"].as_array().ok_or_else(|| {
            WeaverError::Invalid(format!(
                "ffprobe returned no streams for {}",
                path.display()
            ))
        })?;
        let video = streams
            .iter()
            .find(|stream| stream["codec_type"] == "video")
            .ok_or_else(|| {
                WeaverError::Invalid(format!("{} has no video stream", path.display()))
            })?;
        if video["codec_name"] != codec {
            return Err(WeaverError::Invalid(format!(
                "{} codec is {:?}, expected {codec}",
                path.display(),
                video["codec_name"]
            )));
        }
        let pixel_format = video["pix_fmt"].as_str().unwrap_or_default();
        if codec == "prores" && (video["profile"] != "XQ" || !pixel_format.starts_with("yuva444p"))
        {
            return Err(WeaverError::Invalid(format!(
                "{} must be ProRes XQ with a yuva444 alpha pixel format",
                path.display()
            )));
        }
        if codec == "h264" && pixel_format != "yuv420p" {
            return Err(WeaverError::Invalid(format!(
                "{} preview pixel format is {pixel_format}, expected yuv420p",
                path.display()
            )));
        }
        for field in ["color_space", "color_transfer", "color_primaries"] {
            if video[field] != "bt709" {
                return Err(WeaverError::Invalid(format!(
                    "{} {field} is {:?}, expected bt709",
                    path.display(),
                    video[field]
                )));
            }
        }
        let frame_count = video["nb_read_frames"]
            .as_str()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        if frame_count != manifest.frame_count {
            return Err(WeaverError::Invalid(format!(
                "{} contains {frame_count} frames, expected {}",
                path.display(),
                manifest.frame_count
            )));
        }
        let expected_rate = format!("{}/{}", manifest.fps[0], manifest.fps[1]);
        if video["avg_frame_rate"] != expected_rate {
            return Err(WeaverError::Invalid(format!(
                "{} frame rate is {:?}, expected {expected_rate}",
                path.display(),
                video["avg_frame_rate"]
            )));
        }
        let expected_duration =
            manifest.frame_count as f64 * manifest.fps[1] as f64 / manifest.fps[0] as f64;
        let duration = probe["format"]["duration"]
            .as_str()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or_default();
        let frame_duration = manifest.fps[1] as f64 / manifest.fps[0] as f64;
        if (duration - expected_duration).abs() > frame_duration + 0.001 {
            return Err(WeaverError::Invalid(format!(
                "{} duration is {duration:.6}s, expected {expected_duration:.6}s",
                path.display()
            )));
        }
        if video["width"] != manifest.resolution[0] || video["height"] != manifest.resolution[1] {
            return Err(WeaverError::Invalid(format!(
                "{} resolution does not match sequence manifest",
                path.display()
            )));
        }
        let has_audio = streams.iter().any(|stream| stream["codec_type"] == "audio");
        if manifest.outputs.audio_master.is_some() && !has_audio {
            return Err(WeaverError::Invalid(format!(
                "{} is missing authored audio",
                path.display()
            )));
        }
        if let Some(audio) = streams
            .iter()
            .find(|stream| stream["codec_type"] == "audio")
        {
            if audio["sample_rate"] != "48000" || audio["channels"] != 2 {
                return Err(WeaverError::Invalid(format!(
                    "{} audio must be 48 kHz stereo",
                    path.display()
                )));
            }
        }
        let signal = decoded_signal(ffmpeg, path, manifest)?;
        if signal[0] <= 17.0 && signal[1] <= 20.0 {
            return Err(WeaverError::Invalid(format!(
                "{} decodes as an effectively black frame (YAVG {:.3}, YMAX {:.3})",
                path.display(),
                signal[0],
                signal[1]
            )));
        }
        if let Some(object) = probe.as_object_mut() {
            object.insert(
                "decodedSignal".into(),
                serde_json::json!({ "sample": "middle", "yAverage": signal[0], "yMaximum": signal[1] }),
            );
        }
        movies.insert(name.into(), probe);
    }
    Ok(
        serde_json::json!({ "tool": ffprobe, "status": "passed", "expectedFrames": manifest.frame_count, "movies": movies }),
    )
}

fn decoded_signal(
    ffmpeg: &Path,
    movie: &Path,
    manifest: &MasterSequenceManifest,
) -> Result<[f64; 2], WeaverError> {
    let middle_frame = manifest.frame_count / 2;
    let middle_seconds = middle_frame as f64 * manifest.fps[1] as f64 / manifest.fps[0] as f64;
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-ss"])
        .arg(format!("{middle_seconds:.9}"))
        .arg("-i")
        .arg(movie)
        .args([
            "-vf",
            "signalstats,metadata=print:file=-",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ])
        .output()?;
    if !output.status.success() {
        return Err(WeaverError::Invalid(format!(
            "decode validation failed for {}: {}",
            movie.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let value = |name: &str| {
        text.lines().find_map(|line| {
            line.strip_prefix(name)
                .and_then(|value| value.parse::<f64>().ok())
        })
    };
    let average = value("lavfi.signalstats.YAVG=")
        .ok_or_else(|| WeaverError::Invalid("decode validation did not report YAVG".into()))?;
    let maximum = value("lavfi.signalstats.YMAX=")
        .ok_or_else(|| WeaverError::Invalid("decode validation did not report YMAX".into()))?;
    Ok([average, maximum])
}

fn validate_master_frames(manifest: &MasterSequenceManifest) -> Result<(), WeaverError> {
    if manifest.frames.len() != manifest.frame_count as usize {
        return Err(WeaverError::Invalid(format!(
            "manifest contains {} frame records, expected {}",
            manifest.frames.len(),
            manifest.frame_count
        )));
    }
    if manifest.frames.is_empty() {
        return Err(WeaverError::Invalid("master sequence has no frames".into()));
    }
    let sample_indices = [0, manifest.frames.len() / 2, manifest.frames.len() - 1];
    for index in sample_indices {
        let frame = &manifest.frames[index];
        for path in std::iter::once(&frame.display_master).chain(frame.scene_composite.as_ref()) {
            if fs::metadata(path)
                .map(|metadata| metadata.len())
                .unwrap_or(0)
                == 0
            {
                return Err(WeaverError::Invalid(format!(
                    "master acceptance frame is missing or empty: {}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn report_from_manifest(
    manifest: &MasterSequenceManifest,
    root: &Path,
    rendered_frames: u32,
    resumed_frames: u32,
    manifest_path: &Path,
) -> MasterSequenceReport {
    MasterSequenceReport {
        status: manifest.status.clone(),
        output: root.to_path_buf(),
        frame_count: manifest.frame_count,
        rendered_frames,
        resumed_frames,
        display_master_frames: manifest.outputs.display_master_sequence.clone(),
        scene_composite_frames: manifest.outputs.scene_composite_sequence.clone(),
        audio_master: manifest.outputs.audio_master.clone(),
        prores_master: manifest.outputs.prores_master.clone(),
        preview_movie: manifest.outputs.preview_movie.clone(),
        manifest: manifest_path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_preserves_common_ntsc_rates() {
        assert_eq!(fps_rational(23.976).unwrap(), [24_000, 1_001]);
        assert_eq!(fps_rational(29.97).unwrap(), [30_000, 1_001]);
        assert_eq!(fps_rational(24.0).unwrap(), [24, 1]);
    }

    #[test]
    fn frame_count_is_inclusive_and_checked() {
        assert_eq!(checked_frame_count(&(4..=8)).unwrap(), 5);
        assert!(checked_frame_count(&(8..=4)).is_err());
    }
}
