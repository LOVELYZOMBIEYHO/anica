// =========================================
// =========================================
// src/api/export.rs
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::effects::LayerColorBlurEffects;
use crate::core::export::{ExportMode, ExportPreset, ExportRange, ExportSettings};
use crate::core::global_state::{
    AudioTrack, GlobalState, LayerEffectClip, SubtitleGroupTransform, SubtitleTrack, VideoTrack,
};
use crate::export_resolution::{
    format_resolution_label, normalize_export_resolution_hint, parse_resolution_dims,
};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AcpExportRunRequest {
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default)]
    pub output_name: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default)]
    pub fps: Option<u32>,
    #[serde(default)]
    pub crf: Option<u8>,
    #[serde(default)]
    pub encoder_preset: Option<String>,
    #[serde(default)]
    pub audio_bitrate_kbps: Option<u32>,
    #[serde(default)]
    pub range_start_sec: Option<f64>,
    #[serde(default)]
    pub range_end_sec: Option<f64>,
    #[serde(default)]
    pub target_resolution: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AcpExportRunResponse {
    pub ok: bool,
    pub started: bool,
    pub mode: String,
    pub preset: String,
    pub out_path: Option<String>,
    pub layout_resolution: String,
    pub export_resolution: String,
    pub export_width: u32,
    pub export_height: u32,
    pub resolution_source: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedAcpExportRun {
    pub ffmpeg_path: String,
    pub v1: Vec<crate::core::global_state::Clip>,
    pub audio_tracks: Vec<AudioTrack>,
    pub video_tracks: Vec<VideoTrack>,
    pub subtitle_tracks: Vec<SubtitleTrack>,
    pub subtitle_groups: std::collections::HashMap<u64, SubtitleGroupTransform>,
    pub layout_canvas_w: f32,
    pub layout_canvas_h: f32,
    pub export_w: f32,
    pub export_h: f32,
    pub layer_effects: LayerColorBlurEffects,
    pub layer_effect_clips: Vec<LayerEffectClip>,
    pub export_color_mode: crate::core::global_state::ExportColorMode,
    pub export_range: Option<ExportRange>,
    pub export_total: Duration,
    pub export_mode: ExportMode,
    pub export_preset: ExportPreset,
    pub export_settings: ExportSettings,
    pub layout_resolution: String,
    pub export_resolution: String,
    pub resolution_source: String,
    pub out_path: String,
}

#[derive(Debug, Error)]
pub enum AcpExportApiError {
    #[error(
        "MISSING_FFMPEG_FFPROBE: export requires ffmpeg and ffprobe. Install FFmpeg package and retry."
    )]
    MissingMediaTools,
    #[error("Export is already in progress.")]
    ExportAlreadyInProgress,
    #[error("Timeline is empty, nothing to export.")]
    TimelineEmpty,
    #[error("range_end_sec must be greater than range_start_sec")]
    InvalidRange,
}

#[derive(Debug, Clone)]
struct ResolvedTargetResolution {
    width: u32,
    height: u32,
    normalized: String,
    source: &'static str,
}

fn default_export_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        let movies_dir = PathBuf::from(&home).join("Movies");
        if movies_dir.exists() {
            return movies_dir;
        }
        let desktop_dir = PathBuf::from(home).join("Desktop");
        if desktop_dir.exists() {
            return desktop_dir;
        }
    }
    PathBuf::from(".")
}

fn sanitize_output_stem(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "acp_export".to_string();
    }
    let mut out = String::new();
    for ch in trimmed.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "acp_export".to_string()
    } else {
        out
    }
}

pub(crate) fn infer_target_resolution_from_text(raw: &str) -> Option<String> {
    normalize_export_resolution_hint(raw)
}

fn resolve_target_resolution(
    raw: Option<&str>,
    canvas_w: f32,
    canvas_h: f32,
) -> ResolvedTargetResolution {
    let canvas_width = canvas_w.round().max(2.0) as u32;
    let canvas_height = canvas_h.round().max(2.0) as u32;
    let canvas_label = format_resolution_label(canvas_width, canvas_height);

    let Some(raw) = raw else {
        return ResolvedTargetResolution {
            width: canvas_width,
            height: canvas_height,
            normalized: canvas_label,
            source: "canvas",
        };
    };
    let token = raw.trim();
    if token.is_empty() || token.eq_ignore_ascii_case("canvas") {
        return ResolvedTargetResolution {
            width: canvas_width,
            height: canvas_height,
            normalized: canvas_label,
            source: "canvas",
        };
    }

    if let Some(normalized) = infer_target_resolution_from_text(token)
        && let Some((width, height)) = parse_resolution_dims(&normalized)
    {
        return ResolvedTargetResolution {
            width,
            height,
            normalized,
            source: "explicit_target_resolution",
        };
    }

    ResolvedTargetResolution {
        width: canvas_width,
        height: canvas_height,
        normalized: canvas_label,
        source: "canvas_fallback_invalid_target_resolution",
    }
}

fn parse_export_range(
    start_sec: Option<f64>,
    end_sec: Option<f64>,
    timeline_total: Duration,
) -> Result<Option<ExportRange>, AcpExportApiError> {
    let total_secs = timeline_total.as_secs_f64();
    if total_secs <= 0.0 {
        return Err(AcpExportApiError::TimelineEmpty);
    }
    if start_sec.is_none() && end_sec.is_none() {
        return Ok(None);
    }

    let start_secs = start_sec.unwrap_or(0.0).max(0.0).min(total_secs);
    let end_secs = end_sec.unwrap_or(total_secs).max(0.0).min(total_secs);
    if end_secs <= start_secs + 0.001 {
        return Err(AcpExportApiError::InvalidRange);
    }
    Ok(Some(ExportRange {
        start: Duration::from_secs_f64(start_secs),
        end: Duration::from_secs_f64(end_secs),
    }))
}

fn infer_extension_for_mode(mode: ExportMode, preset: ExportPreset, gs: &GlobalState) -> String {
    if (mode == ExportMode::KeepSourceCopy || mode == ExportMode::SmartUniversal)
        && let Some(clip) = gs.v1_clips.first()
        && let Some(ext) = Path::new(&clip.file_path)
            .extension()
            .and_then(|v| v.to_str())
    {
        let normalized = ext.trim().to_ascii_lowercase();
        if !normalized.is_empty() {
            return normalized;
        }
    }
    preset.file_extension().to_string()
}

pub fn resolve_acp_export_run_request(
    request: AcpExportRunRequest,
    gs: &GlobalState,
) -> Result<ResolvedAcpExportRun, AcpExportApiError> {
    if !gs.media_tools_ready_for_export() {
        return Err(AcpExportApiError::MissingMediaTools);
    }

    if gs.export_in_progress {
        return Err(AcpExportApiError::ExportAlreadyInProgress);
    }

    let export_mode = request
        .mode
        .as_deref()
        .and_then(ExportMode::from_id)
        .unwrap_or(ExportMode::SmartUniversal);
    let export_preset = request
        .preset
        .as_deref()
        .and_then(ExportPreset::from_id)
        .unwrap_or(ExportPreset::H264Mp4);

    let mut export_settings = ExportSettings::default();
    if let Some(fps) = request.fps {
        export_settings.fps = fps;
    }
    if let Some(crf) = request.crf {
        export_settings.crf = crf;
    }
    if let Some(preset) = request.encoder_preset {
        export_settings.encoder_preset = preset;
    }
    if let Some(kbps) = request.audio_bitrate_kbps {
        export_settings.audio_bitrate_kbps = kbps;
    }

    let timeline_total = gs.timeline_total();
    let export_range = parse_export_range(
        request.range_start_sec,
        request.range_end_sec,
        timeline_total,
    )?;
    let export_total = export_range
        .map(|v| v.duration())
        .unwrap_or(timeline_total)
        .max(Duration::from_millis(1));
    let resolved_target = resolve_target_resolution(
        request.target_resolution.as_deref(),
        gs.canvas_w,
        gs.canvas_h,
    );
    let export_w = resolved_target.width as f32;
    let export_h = resolved_target.height as f32;
    let layout_resolution = format_resolution_label(
        gs.canvas_w.round().max(2.0) as u32,
        gs.canvas_h.round().max(2.0) as u32,
    );

    let out_path = if let Some(path) = request.output_path {
        path
    } else {
        let stem = request.output_name.unwrap_or_else(|| {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs();
            format!("acp_export_{ts}")
        });
        let ext = infer_extension_for_mode(export_mode, export_preset, gs);
        default_export_dir()
            .join(format!("{}.{}", sanitize_output_stem(&stem), ext))
            .to_string_lossy()
            .to_string()
    };

    Ok(ResolvedAcpExportRun {
        ffmpeg_path: gs.ffmpeg_path.clone(),
        v1: gs.v1_clips.clone(),
        audio_tracks: gs.audio_tracks.clone(),
        video_tracks: gs.video_tracks.clone(),
        subtitle_tracks: gs.subtitle_tracks.clone(),
        subtitle_groups: gs.subtitle_groups.clone(),
        layout_canvas_w: gs.canvas_w,
        layout_canvas_h: gs.canvas_h,
        export_w,
        export_h,
        layer_effects: gs.layer_color_blur_effects(),
        layer_effect_clips: gs.layer_effect_clips().to_vec(),
        export_color_mode: gs.export_color_mode,
        export_range,
        export_total,
        export_mode,
        export_preset,
        export_settings,
        layout_resolution,
        export_resolution: resolved_target.normalized,
        resolution_source: resolved_target.source.to_string(),
        out_path,
    })
}

#[cfg(test)]
mod tests {
    use super::{infer_target_resolution_from_text, resolve_target_resolution};

    #[test]
    fn infer_target_resolution_from_text_maps_orientation_keywords() {
        assert_eq!(
            infer_target_resolution_from_text("portrait"),
            Some("1080x1920".to_string())
        );
        assert_eq!(
            infer_target_resolution_from_text("landscape"),
            Some("1920x1080".to_string())
        );
        assert_eq!(
            infer_target_resolution_from_text("QHD"),
            Some("2560x1440".to_string())
        );
        assert_eq!(
            infer_target_resolution_from_text("DCI 4K"),
            Some("4096x2160".to_string())
        );
        assert_eq!(
            infer_target_resolution_from_text("Vertical (720x1280) 9:16"),
            Some("720x1280".to_string())
        );
    }

    #[test]
    fn resolve_target_resolution_falls_back_to_canvas_for_invalid_hint() {
        let resolved = resolve_target_resolution(Some("unknown-shape"), 1080.0, 1920.0);
        assert_eq!(resolved.normalized, "1080x1920");
        assert_eq!(resolved.source, "canvas_fallback_invalid_target_resolution");
    }
}
