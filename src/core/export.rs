// =========================================
// =========================================
// src/core/export.rs

use gpu_effect_export_engine::{
    SingleClipOpacityVideoToolboxRequest, WgpuOpacityProcessor,
    build_single_clip_opacity_videotoolbox_args,
};
use gpui_video_renderer::{
    BgraGpuEffectParams, BgraProcessEffectInstance, BgraProcessParamValue, BlurMode,
    VideoLocalMaskLayer, process_bgra_effects_with_params,
};
use motionloom::{
    GraphApplyScope, PassNode as MotionloomPassNode, PassTransitionEasing, PassTransitionMode,
    RuntimeFrameOutput, RuntimeProcessEffectInstance, RuntimeProcessParamValue, RuntimeProgram,
    SceneRenderProfile, compile_runtime_program, is_graph_script, parse_graph_script,
    parse_process_graph_script, render_scene_graph_frame, resolve_pass_kernel,
};
use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::Stdio;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{fs, path::Path, path::PathBuf, process::Command};
use thiserror::Error;
use url::Url;
use video_engine::Video;

use crate::core::effects::LayerColorBlurEffects;
use crate::core::global_state::{
    AudioTrack, Clip, ExportColorMode, LayerEffectClip, LocalMaskLayer, ScalarKeyframe,
    SubtitleGroupTransform, SubtitleTrack, VideoEffect, VideoTrack,
};
use crate::core::media_tools::ffprobe_from_ffmpeg;
use crate::core::subtitle_renderer::{
    RenderedSubtitle, SubtitleRenderError, SubtitleRenderOutput, render_subtitle_pngs,
};

#[derive(Debug, Clone, Copy)]
enum OpacityMode {
    AlphaOnly,
    MultiplyRgb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerTransitionEffect {
    FadeIn,
    FadeOut,
    Dip,
}

#[derive(Debug, Clone)]
struct LayerTransitionPlan {
    effect: LayerTransitionEffect,
    mode: PassTransitionMode,
    easing: PassTransitionEasing,
    start_sec: Option<f64>,
    duration_sec: f64,
}

#[derive(Debug, Clone, Copy)]
struct LayerHslaOverlayPlan {
    hue: f64,
    saturation: f64,
    lightness: f64,
    alpha: f64,
}

#[derive(Debug, Clone, Copy)]
struct LayerBloomPlan {
    threshold: f64,
    intensity: f64,
    sigma: f64,
}

#[derive(Debug, Clone, Default)]
struct LayerScriptExportPlan {
    apply: GraphApplyScope,
    graph_duration_sec: f64,
    graph_duration_explicit: bool,
    transitions: Vec<LayerTransitionPlan>,
    blur_sigma: Option<f64>,
    sharpen_sigma: Option<f64>,
    lut_mix: Option<f64>,
    hsla_overlay: Option<LayerHslaOverlayPlan>,
    bloom: Option<LayerBloomPlan>,
    opacity_factor: Option<f64>,
}

#[derive(Debug, Clone)]
struct UnifiedLayerFrameEffects {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    signed_blur_sigma: f32,
    blur_mode: BlurMode,
    lut_mix: f32,
    tint_hue: f32,
    tint_saturation: f32,
    tint_lightness: f32,
    tint_alpha: f32,
    bloom_threshold: f32,
    bloom_intensity: f32,
    bloom_sigma: f32,
    tone_map_enabled: bool,
    tone_map_exposure: f32,
    tone_map_contrast: f32,
    tone_map_shoulder: f32,
    tone_map_gamma: f32,
    tone_map_saturation: f32,
    light_sweep_enabled: bool,
    light_sweep_position: f32,
    light_sweep_angle: f32,
    light_sweep_width: f32,
    light_sweep_softness: f32,
    light_sweep_intensity: f32,
    light_sweep_color: [f32; 4],
    process_effects: Vec<BgraProcessEffectInstance>,
    opacity_factor: f32,
}

impl Default for UnifiedLayerFrameEffects {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            signed_blur_sigma: 0.0,
            blur_mode: BlurMode::Gaussian5tapBlur,
            lut_mix: 0.0,
            tint_hue: 0.0,
            tint_saturation: 0.0,
            tint_lightness: 0.0,
            tint_alpha: 0.0,
            bloom_threshold: 1.0,
            bloom_intensity: 0.0,
            bloom_sigma: 0.0,
            tone_map_enabled: false,
            tone_map_exposure: 0.0,
            tone_map_contrast: 1.0,
            tone_map_shoulder: 1.0,
            tone_map_gamma: 2.2,
            tone_map_saturation: 1.0,
            light_sweep_enabled: false,
            light_sweep_position: 0.5,
            light_sweep_angle: -18.0,
            light_sweep_width: 0.16,
            light_sweep_softness: 0.08,
            light_sweep_intensity: 0.0,
            light_sweep_color: [1.0, 1.0, 1.0, 1.0],
            process_effects: Vec::new(),
            opacity_factor: 1.0,
        }
    }
}

fn motionloom_runtime_effect_to_bgra(
    effect: &RuntimeProcessEffectInstance,
    strength: f32,
) -> BgraProcessEffectInstance {
    let mut params = BTreeMap::new();
    for (key, value) in &effect.params {
        let value = match value {
            RuntimeProcessParamValue::Float(value) => {
                let value = if matches!(key.as_str(), "alpha" | "intensity") {
                    (*value * strength).max(0.0)
                } else {
                    *value
                };
                BgraProcessParamValue::Float(value)
            }
            RuntimeProcessParamValue::Color(color) => BgraProcessParamValue::Color([
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
                color[3] as f32 / 255.0,
            ]),
        };
        params.insert(key.clone(), value);
    }
    BgraProcessEffectInstance {
        effect_id: effect.effect_id.clone(),
        params,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportPreset {
    H264Mp4,
    H264VideotoolboxMp4,
    HevcMp4,
    Gif,
    Vp8Webm,
    Vp9Webm,
    Av1LibaomMkv,
    Av1SvtMkv,
    DnxhrHqMov,
    ProRes422Mov,
    ProRes422HqMov,
    ProRes444Mov,
    ProRes4444Mov,
    AacM4a,
    Mp3,
    Opus,
    Flac,
    WavPcm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportMode {
    SmartUniversal,
    KeepSourceCopy,
    PresetReencode,
}

impl ExportMode {
    pub const fn id(self) -> &'static str {
        match self {
            ExportMode::SmartUniversal => "smart_universal",
            ExportMode::KeepSourceCopy => "keep_source_copy",
            ExportMode::PresetReencode => "preset_reencode",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "smart_universal" => Some(ExportMode::SmartUniversal),
            "keep_source_copy" => Some(ExportMode::KeepSourceCopy),
            "preset_reencode" => Some(ExportMode::PresetReencode),
            _ => None,
        }
    }

    pub const fn all_for_ui() -> &'static [ExportMode] {
        &[
            ExportMode::SmartUniversal,
            ExportMode::KeepSourceCopy,
            ExportMode::PresetReencode,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            ExportMode::SmartUniversal => "Smart Universal (Recommended)",
            ExportMode::KeepSourceCopy => "Keep Source (Copy)",
            ExportMode::PresetReencode => "Preset Re-encode",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ExportMode::SmartUniversal => {
                "Try stream copy first; fallback to preset re-encode when timeline/output needs render."
            }
            ExportMode::KeepSourceCopy => {
                "Trim with stream copy only. Fails if timeline contains effects/overlays/mix."
            }
            ExportMode::PresetReencode => {
                "Always render with selected preset and settings (same behavior as current exporter)."
            }
        }
    }
}

// Hardware H.264 encoders do not reliably support frame rates below ~23.976
// fps. Low fps exports (5, 10, 15) can hang indefinitely because the encoder
// cannot initialise a hardware encoding session at those rates. Only expose
// these low fps options for GIF exports, which use a software pipeline.
pub const UI_EXPORT_FPS_CHOICES: [u32; 11] = [24, 25, 30, 48, 50, 60, 72, 90, 100, 120, 144];
pub const UI_EXPORT_FPS_CHOICES_GIF: [u32; 14] =
    [5, 10, 15, 24, 25, 30, 48, 50, 60, 72, 90, 100, 120, 144];

const EXPORT_SHARPEN_YUV420P_COMP_PCT_ENV: &str = "ANICA_EXPORT_SHARPEN_YUV420P_COMP_PCT";
const EXPORT_SHARPEN_YUV420P_COMP_PCT_DEFAULT: f64 = 0.0;
const EXPORT_SHARPEN_YUV420P_COMP_PCT_MIN: f64 = 0.0;
const EXPORT_SHARPEN_YUV420P_COMP_PCT_MAX: f64 = 30.0;

/// Returns the fps choices visible in the export UI for the given preset.
/// GIF exports show 5/10/15 fps; all video exports hide them because
/// macOS/Windows hardware encoders do not support them reliably.
pub fn export_fps_choices_for_preset(preset: ExportPreset) -> &'static [u32] {
    if preset == ExportPreset::Gif {
        &UI_EXPORT_FPS_CHOICES_GIF
    } else {
        &UI_EXPORT_FPS_CHOICES
    }
}

/// Backward-compatible alias returning the video-only fps list (no 5/10/15).
#[allow(dead_code)]
pub const fn export_fps_choices_for_ui() -> &'static [u32] {
    &UI_EXPORT_FPS_CHOICES
}

pub fn export_resolution_choices_for_ui() -> &'static [(&'static str, &'static str)] {
    crate::export_resolution::export_resolution_choices_for_ui()
}

#[cfg(target_os = "macos")]
const fn platform_h264_label() -> &'static str {
    "H.264 MP4 (VideoToolbox, macOS)"
}

#[cfg(target_os = "windows")]
const fn platform_h264_label() -> &'static str {
    "H.264 MP4 (Media Foundation, Windows)"
}

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
const fn platform_h264_label() -> &'static str {
    "H.264 MP4 (OpenH264)"
}

#[cfg(target_os = "macos")]
const fn platform_h264_description() -> &'static str {
    "macOS hardware H.264 encoder (VideoToolbox). Faster encode with bitrate control."
}

#[cfg(target_os = "windows")]
const fn platform_h264_description() -> &'static str {
    "Windows built-in H.264 encoder (Media Foundation). No x264/GPL dependency."
}

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
const fn platform_h264_description() -> &'static str {
    "OpenH264 compatibility encoder. No x264/GPL dependency."
}

#[cfg(target_os = "macos")]
const fn platform_h264_encoder_name() -> &'static str {
    "h264_videotoolbox"
}

#[cfg(target_os = "windows")]
const fn platform_h264_encoder_name() -> &'static str {
    "h264_mf"
}

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
const fn platform_h264_encoder_name() -> &'static str {
    "libopenh264"
}

impl ExportPreset {
    pub const fn id(self) -> &'static str {
        match self {
            ExportPreset::H264Mp4 => "h264_mp4",
            ExportPreset::H264VideotoolboxMp4 => "h264_videotoolbox_mp4",
            ExportPreset::HevcMp4 => "hevc_mp4",
            ExportPreset::Gif => "gif",
            ExportPreset::Vp8Webm => "vp8_webm",
            ExportPreset::Vp9Webm => "vp9_webm",
            ExportPreset::Av1LibaomMkv => "av1_libaom_mkv",
            ExportPreset::Av1SvtMkv => "av1_svt_mkv",
            ExportPreset::DnxhrHqMov => "dnxhr_hq_mov",
            ExportPreset::ProRes422Mov => "prores_422_mov",
            ExportPreset::ProRes422HqMov => "prores_422_hq_mov",
            ExportPreset::ProRes444Mov => "prores_444_mov",
            ExportPreset::ProRes4444Mov => "prores_4444_mov",
            ExportPreset::AacM4a => "aac_m4a",
            ExportPreset::Mp3 => "mp3",
            ExportPreset::Opus => "opus",
            ExportPreset::Flac => "flac",
            ExportPreset::WavPcm => "wav_pcm",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "h264_mp4" => Some(ExportPreset::H264Mp4),
            "h264_videotoolbox_mp4" => Some(ExportPreset::H264VideotoolboxMp4),
            "hevc_mp4" => Some(ExportPreset::HevcMp4),
            "gif" => Some(ExportPreset::Gif),
            "vp8_webm" => Some(ExportPreset::Vp8Webm),
            "vp9_webm" => Some(ExportPreset::Vp9Webm),
            "av1_libaom_mkv" => Some(ExportPreset::Av1LibaomMkv),
            "av1_svt_mkv" => Some(ExportPreset::Av1SvtMkv),
            "dnxhr_hq_mov" => Some(ExportPreset::DnxhrHqMov),
            "prores_422_mov" => Some(ExportPreset::ProRes422Mov),
            "prores_422_hq_mov" => Some(ExportPreset::ProRes422HqMov),
            "prores_444_mov" => Some(ExportPreset::ProRes444Mov),
            "prores_4444_mov" => Some(ExportPreset::ProRes4444Mov),
            "aac_m4a" => Some(ExportPreset::AacM4a),
            "mp3" => Some(ExportPreset::Mp3),
            "opus" => Some(ExportPreset::Opus),
            "flac" => Some(ExportPreset::Flac),
            "wav_pcm" => Some(ExportPreset::WavPcm),
            _ => None,
        }
    }

    #[cfg(target_os = "macos")]
    pub const fn all_for_ui() -> &'static [ExportPreset] {
        // The bundled macOS FFmpeg build enables the LGPL-compatible external
        // encoders used by these export presets.
        &[
            ExportPreset::H264Mp4,
            ExportPreset::H264VideotoolboxMp4,
            ExportPreset::HevcMp4,
            ExportPreset::Gif,
            ExportPreset::Vp9Webm,
            ExportPreset::Vp8Webm,
            ExportPreset::Av1LibaomMkv,
            ExportPreset::Av1SvtMkv,
            ExportPreset::DnxhrHqMov,
            ExportPreset::ProRes422Mov,
            ExportPreset::ProRes422HqMov,
            ExportPreset::ProRes444Mov,
            ExportPreset::ProRes4444Mov,
            ExportPreset::AacM4a,
            ExportPreset::Mp3,
            ExportPreset::Opus,
            ExportPreset::Flac,
            ExportPreset::WavPcm,
        ]
    }

    pub const fn default_for_platform() -> ExportPreset {
        #[cfg(target_os = "macos")]
        {
            ExportPreset::H264VideotoolboxMp4
        }
        #[cfg(not(target_os = "macos"))]
        {
            ExportPreset::H264Mp4
        }
    }

    pub fn is_available_for_platform(self) -> bool {
        // Keep non-UI export entry points aligned with the presets shown in the UI.
        Self::all_for_ui().contains(&self)
    }

    #[cfg(not(target_os = "macos"))]
    pub const fn all_for_ui() -> &'static [ExportPreset] {
        &[
            ExportPreset::H264Mp4,
            ExportPreset::H264VideotoolboxMp4,
            ExportPreset::HevcMp4,
            ExportPreset::Gif,
            ExportPreset::Vp9Webm,
            ExportPreset::Vp8Webm,
            ExportPreset::Av1LibaomMkv,
            ExportPreset::Av1SvtMkv,
            ExportPreset::DnxhrHqMov,
            ExportPreset::ProRes422Mov,
            ExportPreset::ProRes422HqMov,
            ExportPreset::ProRes444Mov,
            ExportPreset::ProRes4444Mov,
            ExportPreset::AacM4a,
            ExportPreset::Mp3,
            ExportPreset::Opus,
            ExportPreset::Flac,
            ExportPreset::WavPcm,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            ExportPreset::H264Mp4 => "H.264 MP4",
            ExportPreset::H264VideotoolboxMp4 => platform_h264_label(),
            ExportPreset::HevcMp4 => "HEVC MP4",
            ExportPreset::Gif => "Animated GIF",
            ExportPreset::Vp8Webm => "VP8 WEBM",
            ExportPreset::Vp9Webm => "VP9 WEBM",
            ExportPreset::Av1LibaomMkv => "AV1 (libaom+opus) MKV",
            ExportPreset::Av1SvtMkv => "SVT-AV1 (libsvtav1+opus) MKV",
            ExportPreset::DnxhrHqMov => "DNxHR-HQ MOV",
            ExportPreset::ProRes422Mov => "ProRes 422 MOV",
            ExportPreset::ProRes422HqMov => "ProRes 422 HQ MOV",
            ExportPreset::ProRes444Mov => "ProRes 444 MOV",
            ExportPreset::ProRes4444Mov => "ProRes 4444 MOV",
            ExportPreset::AacM4a => "Audio AAC M4A",
            ExportPreset::Mp3 => "Audio MP3",
            ExportPreset::Opus => "Audio OPUS",
            ExportPreset::Flac => "Audio FLAC",
            ExportPreset::WavPcm => "Audio WAV PCM",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ExportPreset::H264Mp4 => {
                "Best compatibility. Uses non-x264 H.264 encoders (VideoToolbox/Media Foundation/OpenH264 path)."
            }
            ExportPreset::H264VideotoolboxMp4 => platform_h264_description(),
            ExportPreset::HevcMp4 => {
                "HEVC via VideoToolbox on macOS. On other platforms falls back to H.264 compatibility export."
            }
            ExportPreset::Gif => {
                "Animated GIF with FFmpeg palette generation. No audio; best for short clips."
            }
            ExportPreset::Vp8Webm => "VP8 video + Opus audio in WebM.",
            ExportPreset::Vp9Webm => "VP9 video + Opus audio in WebM.",
            ExportPreset::Av1LibaomMkv => "AV1 via libaom with Opus audio.",
            ExportPreset::Av1SvtMkv => "AV1 via SVT-AV1 with Opus audio.",
            ExportPreset::DnxhrHqMov => "DNxHR HQ mezzanine format (10-bit 4:2:2).",
            ExportPreset::ProRes422Mov => "ProRes 422 profile for editing workflow.",
            ExportPreset::ProRes422HqMov => {
                "Editing/master format with high quality and large files."
            }
            ExportPreset::ProRes444Mov => "ProRes 444 profile (no alpha channel).",
            ExportPreset::ProRes4444Mov => {
                "High-end master format with alpha support and very large files."
            }
            ExportPreset::AacM4a => "Audio-only AAC in M4A container.",
            ExportPreset::Mp3 => "Audio-only MP3.",
            ExportPreset::Opus => "Audio-only Opus.",
            ExportPreset::Flac => "Audio-only lossless FLAC.",
            ExportPreset::WavPcm => "Audio-only uncompressed PCM WAV.",
        }
    }

    pub fn file_extension(self) -> &'static str {
        match self {
            ExportPreset::H264Mp4 | ExportPreset::H264VideotoolboxMp4 | ExportPreset::HevcMp4 => {
                "mp4"
            }
            ExportPreset::Gif => "gif",
            ExportPreset::Vp8Webm | ExportPreset::Vp9Webm => "webm",
            ExportPreset::Av1LibaomMkv | ExportPreset::Av1SvtMkv => "mkv",
            ExportPreset::DnxhrHqMov
            | ExportPreset::ProRes422Mov
            | ExportPreset::ProRes422HqMov
            | ExportPreset::ProRes444Mov
            | ExportPreset::ProRes4444Mov => "mov",
            ExportPreset::AacM4a => "m4a",
            ExportPreset::Mp3 => "mp3",
            ExportPreset::Opus => "opus",
            ExportPreset::Flac => "flac",
            ExportPreset::WavPcm => "wav",
        }
    }

    pub fn supports_crf(self) -> bool {
        matches!(
            self,
            ExportPreset::Vp8Webm
                | ExportPreset::Vp9Webm
                | ExportPreset::Av1LibaomMkv
                | ExportPreset::Av1SvtMkv
        )
    }

    fn push_non_x264_h264_args(args: &mut Vec<String>, settings: &ExportSettings) {
        // Keep Apache/LGPL path clean: avoid libx264 in generic H.264 export.
        if cfg!(target_os = "macos") {
            // Compatibility preset on macOS allows encoder-side software fallback within VideoToolbox.
            args.push("-c:v".into());
            args.push("h264_videotoolbox".into());
            args.push("-allow_sw".into());
            args.push("1".into());
            args.push("-pix_fmt".into());
            args.push("yuv420p".into());
            args.push("-b:v".into());
            args.push("12M".into());
            args.push("-maxrate".into());
            args.push("16M".into());
            args.push("-bufsize".into());
            args.push("24M".into());
        } else {
            // Non-macOS compatibility path without x264.
            args.push("-c:v".into());
            args.push("libopenh264".into());
            args.push("-pix_fmt".into());
            args.push("yuv420p".into());
            args.push("-b:v".into());
            args.push("8M".into());
            args.push("-maxrate".into());
            args.push("12M".into());
            args.push("-bufsize".into());
            args.push("16M".into());
        }
        args.push("-c:a".into());
        args.push("aac".into());
        args.push("-b:a".into());
        args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
    }

    fn push_non_gpl_hevc_args(args: &mut Vec<String>, settings: &ExportSettings) {
        if cfg!(target_os = "macos") {
            args.push("-c:v".into());
            args.push("hevc_videotoolbox".into());
            args.push("-allow_sw".into());
            args.push("1".into());
            args.push("-pix_fmt".into());
            args.push("yuv420p".into());
            args.push("-b:v".into());
            args.push("12M".into());
            args.push("-maxrate".into());
            args.push("16M".into());
            args.push("-bufsize".into());
            args.push("24M".into());
            args.push("-tag:v".into());
            args.push("hvc1".into());
            args.push("-c:a".into());
            args.push("aac".into());
            args.push("-b:a".into());
            args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
        } else {
            // Keep non-macOS path GPL-free when HEVC encoders are unavailable.
            Self::push_non_x264_h264_args(args, settings);
        }
    }

    pub fn is_audio_only(self) -> bool {
        matches!(
            self,
            ExportPreset::AacM4a
                | ExportPreset::Mp3
                | ExportPreset::Opus
                | ExportPreset::Flac
                | ExportPreset::WavPcm
        )
    }

    pub const fn requires_rendered_video(self) -> bool {
        matches!(self, ExportPreset::Gif)
    }

    pub const fn omits_audio(self) -> bool {
        matches!(self, ExportPreset::Gif)
    }

    fn is_yuv420p_8bit(self) -> bool {
        matches!(
            self,
            ExportPreset::H264Mp4
                | ExportPreset::H264VideotoolboxMp4
                | ExportPreset::Vp8Webm
                | ExportPreset::Vp9Webm
        )
    }

    fn push_output_args(self, args: &mut Vec<String>, settings: &ExportSettings, fps: u32) {
        match self {
            ExportPreset::H264Mp4 => {
                Self::push_non_x264_h264_args(args, settings);
            }
            ExportPreset::H264VideotoolboxMp4 => {
                args.push("-c:v".into());
                args.push(platform_h264_encoder_name().into());
                if cfg!(target_os = "macos") {
                    args.push("-allow_sw".into());
                    // VideoToolbox hardware encoder requires ~23.976+ fps.
                    // Allow software fallback at low fps so the encoder does not hang.
                    args.push(if fps < 24 { "1" } else { "0" }.into());
                }
                args.push("-pix_fmt".into());
                args.push("yuv420p".into());
                args.push("-b:v".into());
                args.push("12M".into());
                args.push("-maxrate".into());
                args.push("16M".into());
                args.push("-bufsize".into());
                args.push("24M".into());
                args.push("-c:a".into());
                args.push("aac".into());
                args.push("-b:a".into());
                args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
            }
            ExportPreset::HevcMp4 => {
                Self::push_non_gpl_hevc_args(args, settings);
            }
            ExportPreset::Gif => {
                args.push("-loop".into());
                args.push("0".into());
            }
            ExportPreset::Vp8Webm => {
                args.push("-c:v".into());
                args.push("libvpx".into());
                args.push("-pix_fmt".into());
                args.push("yuv420p".into());
                args.push("-crf".into());
                args.push(settings.normalized_crf().to_string());
                args.push("-b:v".into());
                args.push("0".into());
                args.push("-deadline".into());
                args.push("good".into());
                args.push("-cpu-used".into());
                args.push("4".into());
                args.push("-c:a".into());
                args.push("libopus".into());
                args.push("-b:a".into());
                args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
            }
            ExportPreset::Vp9Webm => {
                args.push("-c:v".into());
                args.push("libvpx-vp9".into());
                args.push("-pix_fmt".into());
                args.push("yuv420p".into());
                args.push("-crf".into());
                args.push(settings.normalized_crf().to_string());
                args.push("-b:v".into());
                args.push("0".into());
                args.push("-deadline".into());
                args.push("good".into());
                args.push("-row-mt".into());
                args.push("1".into());
                args.push("-tile-columns".into());
                args.push("2".into());
                args.push("-frame-parallel".into());
                args.push("1".into());
                args.push("-cpu-used".into());
                args.push("2".into());
                args.push("-c:a".into());
                args.push("libopus".into());
                args.push("-b:a".into());
                args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
            }
            ExportPreset::Av1LibaomMkv => {
                args.push("-c:v".into());
                args.push("libaom-av1".into());
                args.push("-pix_fmt".into());
                args.push("yuv420p10le".into());
                args.push("-crf".into());
                args.push(settings.normalized_crf().to_string());
                args.push("-b:v".into());
                args.push("0".into());
                args.push("-cpu-used".into());
                args.push("6".into());
                args.push("-row-mt".into());
                args.push("1".into());
                args.push("-tiles".into());
                args.push("2x2".into());
                args.push("-c:a".into());
                args.push("libopus".into());
                args.push("-b:a".into());
                args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
            }
            ExportPreset::Av1SvtMkv => {
                args.push("-c:v".into());
                args.push("libsvtav1".into());
                args.push("-pix_fmt".into());
                args.push("yuv420p10le".into());
                args.push("-crf".into());
                args.push(settings.normalized_crf().to_string());
                args.push("-preset".into());
                args.push("8".into());
                args.push("-c:a".into());
                args.push("libopus".into());
                args.push("-b:a".into());
                args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
            }
            ExportPreset::DnxhrHqMov => {
                args.push("-c:v".into());
                args.push("dnxhd".into());
                args.push("-profile:v".into());
                args.push("dnxhr_hq".into());
                args.push("-pix_fmt".into());
                args.push("yuv422p10le".into());
                args.push("-c:a".into());
                args.push("pcm_s24le".into());
            }
            ExportPreset::ProRes422Mov => {
                args.push("-c:v".into());
                args.push("prores_ks".into());
                args.push("-profile:v".into());
                args.push("2".into());
                args.push("-pix_fmt".into());
                args.push("yuv422p10le".into());
                args.push("-c:a".into());
                args.push("pcm_s16le".into());
            }
            ExportPreset::ProRes422HqMov => {
                args.push("-c:v".into());
                args.push("prores_ks".into());
                args.push("-profile:v".into());
                args.push("3".into());
                args.push("-pix_fmt".into());
                args.push("yuv422p10le".into());
                args.push("-c:a".into());
                args.push("pcm_s16le".into());
            }
            ExportPreset::ProRes444Mov => {
                args.push("-c:v".into());
                args.push("prores_ks".into());
                args.push("-profile:v".into());
                args.push("4".into());
                args.push("-pix_fmt".into());
                args.push("yuv444p10le".into());
                args.push("-c:a".into());
                args.push("pcm_s16le".into());
            }
            ExportPreset::ProRes4444Mov => {
                args.push("-c:v".into());
                args.push("prores_ks".into());
                args.push("-profile:v".into());
                args.push("4".into());
                args.push("-pix_fmt".into());
                args.push("yuva444p10le".into());
                args.push("-c:a".into());
                args.push("pcm_s16le".into());
            }
            ExportPreset::AacM4a => {
                args.push("-vn".into());
                args.push("-c:a".into());
                args.push("aac".into());
                args.push("-b:a".into());
                args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
            }
            ExportPreset::Mp3 => {
                args.push("-vn".into());
                args.push("-c:a".into());
                args.push("libmp3lame".into());
                args.push("-b:a".into());
                args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
            }
            ExportPreset::Opus => {
                args.push("-vn".into());
                args.push("-c:a".into());
                args.push("libopus".into());
                args.push("-b:a".into());
                args.push(format!("{}k", settings.normalized_audio_bitrate_kbps()));
            }
            ExportPreset::Flac => {
                args.push("-vn".into());
                args.push("-c:a".into());
                args.push("flac".into());
            }
            ExportPreset::WavPcm => {
                args.push("-vn".into());
                args.push("-c:a".into());
                args.push("pcm_s16le".into());
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExportSettings {
    pub fps: u32,
    pub crf: u8,
    pub encoder_preset: String,
    pub audio_bitrate_kbps: u32,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            fps: 30,
            crf: 20,
            encoder_preset: "medium".to_string(),
            audio_bitrate_kbps: 192,
        }
    }
}

impl ExportSettings {
    pub fn normalized_fps(&self) -> u32 {
        self.fps.clamp(1, 144)
    }

    pub fn normalized_crf(&self) -> u8 {
        self.crf.clamp(0, 51)
    }

    pub fn normalized_audio_bitrate_kbps(&self) -> u32 {
        self.audio_bitrate_kbps.clamp(64, 512)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExportRange {
    pub start: Duration,
    pub end: Duration,
}

impl ExportRange {
    pub fn duration(self) -> Duration {
        self.end.saturating_sub(self.start)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExportProgress {
    pub rendered: Duration,
    pub total: Duration,
    pub speed: Option<f32>,
}

pub struct FfmpegExporter;
pub const EXPORT_CANCELLED_ERR: &str = "__ANICA_EXPORT_CANCELLED__";

struct SceneLayerExportState {
    layer_id: u64,
    script_hash: u64,
    graph: motionloom::GraphScript,
    fps: f32,
    last_frame: Option<u32>,
    last_rgba: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("Keep Source (Copy) is unavailable for current timeline/output setup.")]
    KeepSourceCopyUnavailable,
    #[error("{reason}")]
    GpuOpacityPathUnavailable { reason: String },
    #[error("Failed to initialize GPU opacity processor: {message}")]
    InitializeGpuOpacityProcessor { message: String },

    #[error("{0}")]
    SubtitleRender(#[from] SubtitleRenderError),
    #[error("Failed to build FFmpeg output path.")]
    MissingOutputPathArg,
    #[error("Failed to execute FFmpeg: {source}")]
    ExecuteFfmpeg { source: std::io::Error },
    #[error("Failed to execute {stage} FFmpeg: {source}")]
    ExecuteStageFfmpeg {
        stage: &'static str,
        source: std::io::Error,
    },
    #[error("Failed to capture FFmpeg progress output.")]
    MissingProgressOutput,
    #[error("Failed to capture FFmpeg error output.")]
    MissingErrorOutput,
    #[error("Missing {stage} FFmpeg {pipe} pipe.")]
    MissingFfmpegPipe {
        stage: &'static str,
        pipe: &'static str,
    },
    #[error("Failed to read FFmpeg progress: {source}")]
    ReadFfmpegProgress { source: std::io::Error },
    #[error("Failed to read {stage} FFmpeg frame stream: {source}")]
    ReadFfmpegFrame {
        stage: &'static str,
        source: std::io::Error,
    },
    #[error("Failed to write {stage} FFmpeg frame stream: {source}")]
    WriteFfmpegFrame {
        stage: &'static str,
        source: std::io::Error,
    },
    #[error("Failed waiting FFmpeg process: {source}")]
    WaitFfmpegProcess { source: std::io::Error },
    #[error("Failed waiting {stage} FFmpeg process: {source}")]
    WaitStageFfmpegProcess {
        stage: &'static str,
        source: std::io::Error,
    },
    #[error("FFmpeg failed.\nSTDERR:\n{stderr}")]
    FfmpegFailed { stderr: String },
    #[error("{stage} FFmpeg failed ({status}).\nSTDERR:\n{stderr}")]
    StageFfmpegFailed {
        stage: &'static str,
        status: String,
        stderr: String,
    },
    #[error("Failed to process GPU opacity frame: {message}")]
    ProcessGpuOpacityFrame { message: String },
    #[error("Failed to load subtitle overlay ({path}): {source}")]
    LoadSubtitleOverlay {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error("Failed to load still frame ({path}): {source}")]
    LoadStillFrame {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error("Internal export state missing: {state}")]
    MissingInternalState { state: &'static str },
    #[error("__ANICA_EXPORT_CANCELLED__")]
    Cancelled,
    #[error("Failed to replace existing output file: {source}")]
    ReplaceExistingOutput { source: std::io::Error },
    #[error("Failed to finalize output file: {source}")]
    FinalizeOutput { source: std::io::Error },
}

pub fn is_cancelled_export_error(err: &str) -> bool {
    err.trim() == EXPORT_CANCELLED_ERR
}

fn temp_output_path(out_path: &str) -> String {
    let path = Path::new(out_path);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("anica_export");
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("mp4");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos();
    let tmp_name = format!(
        "{stem}.anica.{pid}.{nonce}.tmp.{ext}",
        pid = std::process::id()
    );
    if let Some(parent) = path.parent() {
        return parent.join(tmp_name).to_string_lossy().to_string();
    }
    tmp_name
}

impl FfmpegExporter {
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn is_identity_video_effect(effect: &VideoEffect) -> bool {
        match effect {
            VideoEffect::ColorCorrection {
                brightness,
                contrast,
                saturation,
            } => {
                Self::approx_eq(*brightness, 0.0, 0.001)
                    && Self::approx_eq(*contrast, 1.0, 0.001)
                    && Self::approx_eq(*saturation, 1.0, 0.001)
            }
            VideoEffect::Transform {
                scale,
                position_x,
                position_y,
                rotation_deg,
            } => {
                Self::approx_eq(*scale, 1.0, 0.001)
                    && Self::approx_eq(*position_x, 0.0, 0.001)
                    && Self::approx_eq(*position_y, 0.0, 0.001)
                    && Self::approx_eq(*rotation_deg, 0.0, 0.001)
            }
            VideoEffect::Tint {
                hue,
                saturation,
                lightness,
                alpha,
            } => {
                Self::approx_eq(*hue, 0.0, 0.001)
                    && Self::approx_eq(*saturation, 0.0, 0.001)
                    && Self::approx_eq(*lightness, 0.0, 0.001)
                    && Self::approx_eq(*alpha, 0.0, 0.001)
            }
            VideoEffect::Opacity { .. } => false,
            VideoEffect::Fade { fade_in, fade_out } => {
                Self::approx_eq(*fade_in, 0.0, 0.001) && Self::approx_eq(*fade_out, 0.0, 0.001)
            }
            VideoEffect::Dissolve {
                dissolve_in,
                dissolve_out,
            } => {
                Self::approx_eq(*dissolve_in, 0.0, 0.001)
                    && Self::approx_eq(*dissolve_out, 0.0, 0.001)
            }
            VideoEffect::Slide {
                slide_in,
                slide_out,
                ..
            } => Self::approx_eq(*slide_in, 0.0, 0.001) && Self::approx_eq(*slide_out, 0.0, 0.001),
            VideoEffect::Zoom {
                zoom_in, zoom_out, ..
            } => Self::approx_eq(*zoom_in, 0.0, 0.001) && Self::approx_eq(*zoom_out, 0.0, 0.001),
            VideoEffect::ShockZoom {
                shock_in,
                shock_out,
                ..
            } => Self::approx_eq(*shock_in, 0.0, 0.001) && Self::approx_eq(*shock_out, 0.0, 0.001),
            VideoEffect::GaussianBlur { sigma } => Self::approx_eq(*sigma, 0.0, 0.001),
            VideoEffect::LocalMask { enabled, .. } => !enabled,
            VideoEffect::LocalMaskAdjust {
                brightness,
                contrast,
                saturation,
                opacity,
                blur_sigma,
            } => {
                Self::approx_eq(*brightness, 0.0, 0.001)
                    && Self::approx_eq(*contrast, 1.0, 0.001)
                    && Self::approx_eq(*saturation, 1.0, 0.001)
                    && Self::approx_eq(*opacity, 1.0, 0.001)
                    && Self::approx_eq(*blur_sigma, 0.0, 0.001)
            }
            VideoEffect::Pixelate { block_size } => *block_size <= 1,
        }
    }

    fn has_active_local_mask_layers(layers: &[LocalMaskLayer]) -> bool {
        layers.iter().any(|layer| {
            layer.enabled
                && (layer.strength > 0.001)
                && (layer.radius > 0.001 || layer.feather > 0.001)
        })
    }

    fn is_opacity_only_compatible_clip(clip: &Clip) -> bool {
        if Self::has_active_local_mask_layers(&clip.local_mask_layers)
            || !clip.pos_x_keyframes.is_empty()
            || !clip.pos_y_keyframes.is_empty()
            || !clip.scale_keyframes.is_empty()
            || !clip.rotation_keyframes.is_empty()
            || !clip.brightness_keyframes.is_empty()
            || !clip.contrast_keyframes.is_empty()
            || !clip.saturation_keyframes.is_empty()
            || !clip.blur_keyframes.is_empty()
            || clip.dissolve_trim_in > Duration::from_millis(1)
            || clip.dissolve_trim_out > Duration::from_millis(1)
        {
            return false;
        }
        !clip.video_effects.iter().any(|effect| {
            !matches!(effect, VideoEffect::Opacity { .. })
                && !Self::is_identity_video_effect(effect)
        })
    }

    fn clip_has_non_identity_opacity(clip: &Clip) -> bool {
        if !clip.opacity_keyframes.is_empty() {
            return true;
        }
        (clip.get_opacity().clamp(0.0, 1.0) - 1.0).abs() > 0.001
    }

    fn to_renderer_blur_mode(mode: motionloom::BlurSharpenMode) -> BlurMode {
        match mode {
            motionloom::BlurSharpenMode::Gaussian5tapBlur => BlurMode::Gaussian5tapBlur,
            motionloom::BlurSharpenMode::Gaussian5tapH => BlurMode::Gaussian5tapH,
            motionloom::BlurSharpenMode::Gaussian5tapV => BlurMode::Gaussian5tapV,
            motionloom::BlurSharpenMode::Box => BlurMode::Box,
            motionloom::BlurSharpenMode::Unsharp => BlurMode::Unsharp,
        }
    }

    fn clip_requires_unified_gpu_render(clip: &Clip) -> bool {
        Self::has_active_local_mask_layers(&clip.local_mask_layers)
            || !clip.pos_x_keyframes.is_empty()
            || !clip.pos_y_keyframes.is_empty()
            || !clip.scale_keyframes.is_empty()
            || !clip.rotation_keyframes.is_empty()
            || !clip.brightness_keyframes.is_empty()
            || !clip.contrast_keyframes.is_empty()
            || !clip.saturation_keyframes.is_empty()
            || !clip.blur_keyframes.is_empty()
            || clip.dissolve_trim_in > Duration::from_millis(1)
            || clip.dissolve_trim_out > Duration::from_millis(1)
            || Self::clip_has_non_identity_opacity(clip)
            || clip
                .video_effects
                .iter()
                .any(|effect| !Self::is_identity_video_effect(effect))
    }

    fn is_default_linked_audio_layout(audio_tracks: &[AudioTrack], clip: &Clip) -> bool {
        let mut all_audio_clips = audio_tracks.iter().flat_map(|track| track.clips.iter());
        let Some(a) = all_audio_clips.next() else {
            return true;
        };
        if all_audio_clips.next().is_some() {
            return false;
        }
        a.file_path == clip.file_path
            && a.start == clip.start
            && a.duration == clip.duration
            && a.source_in == clip.source_in
            && (a.audio_gain_db.abs() <= 0.001)
    }

    fn should_try_true_gpu_single_clip_opacity_path(
        v1_clips: &[Clip],
        audio_tracks: &[AudioTrack],
        video_tracks: &[VideoTrack],
        subtitle_tracks: &[SubtitleTrack],
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
        preset: ExportPreset,
    ) -> bool {
        if !cfg!(target_os = "macos") {
            return false;
        }
        if preset != ExportPreset::H264VideotoolboxMp4 {
            return false;
        }
        if subtitle_tracks.iter().any(|t| !t.clips.is_empty()) {
            return false;
        }
        if !layer_effects.is_identity() || !layer_effect_clips.is_empty() {
            return false;
        }
        if v1_clips.len() != 1 || video_tracks.iter().any(|t| !t.clips.is_empty()) {
            return false;
        }
        let clip = &v1_clips[0];
        clip.start <= Duration::from_millis(1)
            && Self::is_default_linked_audio_layout(audio_tracks, clip)
            && Self::is_opacity_only_compatible_clip(clip)
            && Self::clip_has_non_identity_opacity(clip)
    }

    fn should_try_true_gpu_multitrack_opacity_path(
        v1_clips: &[Clip],
        audio_tracks: &[AudioTrack],
        video_tracks: &[VideoTrack],
        subtitle_tracks: &[SubtitleTrack],
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
        preset: ExportPreset,
    ) -> bool {
        if !cfg!(target_os = "macos") {
            return false;
        }
        if preset != ExportPreset::H264VideotoolboxMp4 {
            return false;
        }
        if subtitle_tracks.iter().any(|t| !t.clips.is_empty()) {
            return false;
        }
        if !layer_effects.is_identity() || !layer_effect_clips.is_empty() {
            return false;
        }
        if v1_clips.len() != 1 {
            return false;
        }
        if !Self::is_default_linked_audio_layout(audio_tracks, &v1_clips[0]) {
            return false;
        }
        // Phase 2 scope (initial): allow multiple tracks, but each track keeps a
        // single clip to keep stream orchestration stable.
        if video_tracks.iter().any(|t| t.clips.len() > 1) {
            return false;
        }
        let mut all_visual_clips: Vec<&Clip> = Vec::with_capacity(
            v1_clips.len() + video_tracks.iter().map(|t| t.clips.len()).sum::<usize>(),
        );
        all_visual_clips.extend(v1_clips.iter());
        for track in video_tracks {
            all_visual_clips.extend(track.clips.iter());
        }
        if all_visual_clips.is_empty() {
            return false;
        }
        if all_visual_clips
            .iter()
            .any(|clip| !Self::is_opacity_only_compatible_clip(clip))
        {
            return false;
        }
        all_visual_clips
            .iter()
            .any(|clip| Self::clip_has_non_identity_opacity(clip))
    }

    fn should_try_unified_gpu_render_path(
        v1_clips: &[Clip],
        audio_tracks: &[AudioTrack],
        video_tracks: &[VideoTrack],
        subtitle_tracks: &[SubtitleTrack],
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
        preset: ExportPreset,
    ) -> bool {
        if preset.is_audio_only() {
            return false;
        }
        if v1_clips.is_empty() && video_tracks.iter().all(|t| t.clips.is_empty()) {
            return false;
        }

        // GIF requires palettegen/paletteuse which buffers all frames before output.
        // The unified GPU path writes frames through a stdin pipe, which blocks
        // when the internal buffer fills up because palettegen cannot start emitting
        // the palette until every frame has been consumed.  This creates a deadlock
        // where write_all blocks and the cancel path becomes unreachable.
        // Use the standard FFmpeg path instead, which reads from files directly.
        if preset == ExportPreset::Gif {
            return false;
        }
        if subtitle_tracks.iter().any(|t| !t.clips.is_empty()) {
            return true;
        }
        if let Some(base_clip) = v1_clips.first()
            && !Self::is_default_linked_audio_layout(audio_tracks, base_clip)
        {
            return true;
        }
        if !layer_effects.is_identity() {
            return true;
        }
        if v1_clips
            .iter()
            .any(|clip| Self::clip_requires_unified_gpu_render(clip))
            || video_tracks.iter().any(|track| {
                track
                    .clips
                    .iter()
                    .any(|clip| Self::clip_requires_unified_gpu_render(clip))
            })
        {
            return true;
        }

        let mut cache: std::collections::HashMap<u64, Option<LayerScriptExportPlan>> =
            std::collections::HashMap::new();
        layer_effect_clips.iter().any(|layer| {
            layer.duration > Duration::ZERO
                && (layer.has_any_effect_enabled()
                    || Self::layer_script_plan_for_export(layer, &mut cache).is_some())
        })
    }

    fn rgba_bgra_swap_in_place(data: &mut [u8]) {
        for px in data.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }

    fn local_mask_layers_for_gpu(clip: &Clip) -> Vec<VideoLocalMaskLayer> {
        clip.local_mask_layers
            .iter()
            .map(|layer| VideoLocalMaskLayer {
                enabled: layer.enabled,
                center_x: layer.center_x,
                center_y: layer.center_y,
                radius: layer.radius,
                feather: layer.feather,
                strength: layer.strength,
                brightness: layer.brightness,
                contrast: layer.contrast,
                saturation: layer.saturation,
                opacity: layer.opacity,
                blur_sigma: layer.blur_sigma,
            })
            .collect()
    }

    fn runtime_program_for_layer_export(
        layer_clip: &LayerEffectClip,
        cache: &mut std::collections::HashMap<u64, Option<RuntimeProgram>>,
    ) -> Option<RuntimeProgram> {
        let entry = cache.entry(layer_clip.id).or_insert_with(|| {
            if !layer_clip.motionloom_enabled {
                return None;
            }
            let script = layer_clip.motionloom_script.trim();
            if script.is_empty() || !is_graph_script(script) {
                return None;
            }
            parse_process_graph_script(script)
                .ok()
                .and_then(|graph| compile_runtime_program(graph).ok())
        });
        entry.clone()
    }

    fn runtime_output_for_layer_export(
        layer_clip: &LayerEffectClip,
        timeline_time: Duration,
        cache: &mut std::collections::HashMap<u64, Option<RuntimeProgram>>,
    ) -> Option<RuntimeFrameOutput> {
        let runtime = Self::runtime_program_for_layer_export(layer_clip, cache)?;
        let local = layer_clip.local_time(timeline_time)?;
        Some(
            runtime
                .evaluate_at_time_sec(local.as_secs_f32(), Some(layer_clip.duration.as_secs_f32())),
        )
    }

    fn unified_layer_frame_effects_at(
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
        timeline_time: Duration,
        runtime_cache: &mut std::collections::HashMap<u64, Option<RuntimeProgram>>,
    ) -> UnifiedLayerFrameEffects {
        let mut out = UnifiedLayerFrameEffects::default();
        let mut has_active_layer = false;
        let mut opacity_strength = 0.0_f32;
        let mut transition_opacity = 1.0_f32;
        let mut best_lut_mix = 0.0_f32;
        let mut best_hsla_weighted_alpha = 0.0_f32;
        let mut best_bloom_weighted_intensity = 0.0_f32;
        let mut best_light_sweep_weighted_intensity = 0.0_f32;

        for layer_clip in layer_effect_clips {
            if layer_clip.local_time(timeline_time).is_none() {
                continue;
            }
            has_active_layer = true;
            let strength = layer_clip.envelope_factor_at(timeline_time).clamp(0.0, 1.0);
            opacity_strength = opacity_strength.max(strength);

            // When MotionLoom runtime output is available, it owns the blur/color
            // result and legacy scalar fields are ignored. Legacy scalar fields are
            // only used as a fallback when there is no MotionLoom script, compile
            // failure, or for old project compatibility.
            if let Some(runtime_out) =
                Self::runtime_output_for_layer_export(layer_clip, timeline_time, runtime_cache)
            {
                let mut runtime_signed_sigma = 0.0_f32;
                if let Some(v) = runtime_out.layer_blur_sigma {
                    runtime_signed_sigma += v.clamp(0.0, 64.0);
                }
                if let Some(v) = runtime_out.layer_sharpen_sigma {
                    runtime_signed_sigma -= v.clamp(0.0, 64.0);
                }
                out.signed_blur_sigma += runtime_signed_sigma.clamp(-64.0, 64.0);
                if let Some(mode) = runtime_out.blur_sharpen_mode {
                    out.blur_mode = Self::to_renderer_blur_mode(mode);
                }
            } else if let Some(mut sampled) = layer_clip.effects_at(timeline_time) {
                // Match preview semantics: legacy global layer effects are only used
                // through an active Layer FX clip that has no per-clip toggles.
                if !layer_clip.has_any_effect_enabled() && !layer_effects.is_identity() {
                    sampled = layer_effects;
                }
                let sampled = sampled.normalized();
                out.brightness += sampled.brightness;
                out.contrast *= sampled.contrast;
                out.saturation *= sampled.saturation;
                out.signed_blur_sigma += sampled.blur_sigma;
            }

            let Some(runtime_out) =
                Self::runtime_output_for_layer_export(layer_clip, timeline_time, runtime_cache)
            else {
                continue;
            };
            out.process_effects.extend(
                runtime_out
                    .process_effects
                    .iter()
                    .map(|effect| motionloom_runtime_effect_to_bgra(effect, strength)),
            );

            if let Some(v) = runtime_out.layer_transition_opacity {
                transition_opacity = transition_opacity.min(v.clamp(0.0, 1.0));
            }
            if let Some(v) = runtime_out.layer_lut_mix {
                best_lut_mix = best_lut_mix.max(v.clamp(0.0, 1.0) * strength);
            }
            if let (Some(h), Some(s), Some(l), Some(a)) = (
                runtime_out.layer_hsla_hue,
                runtime_out.layer_hsla_saturation,
                runtime_out.layer_hsla_lightness,
                runtime_out.layer_hsla_alpha,
            ) {
                let weighted_alpha = (a.clamp(0.0, 1.0) * strength).clamp(0.0, 1.0);
                if weighted_alpha > best_hsla_weighted_alpha {
                    best_hsla_weighted_alpha = weighted_alpha;
                    out.tint_hue = h;
                    out.tint_saturation = s;
                    out.tint_lightness = l;
                    out.tint_alpha = weighted_alpha.clamp(0.0, 1.0);
                }
            }
            if let (Some(threshold), Some(intensity), Some(sigma)) = (
                runtime_out.layer_bloom_threshold,
                runtime_out.layer_bloom_intensity,
                runtime_out.layer_bloom_sigma,
            ) {
                let weighted_intensity = (intensity.clamp(0.0, 8.0) * strength).clamp(0.0, 8.0);
                if weighted_intensity > best_bloom_weighted_intensity && sigma > 0.001 {
                    best_bloom_weighted_intensity = weighted_intensity;
                    out.bloom_threshold = threshold.clamp(0.0, 1.0);
                    out.bloom_intensity = weighted_intensity.clamp(0.0, 8.0);
                    out.bloom_sigma = sigma.clamp(0.0, 64.0);
                }
            }
            if let (Some(exposure), Some(contrast), Some(shoulder), Some(gamma), Some(saturation)) = (
                runtime_out.layer_tone_map_exposure,
                runtime_out.layer_tone_map_contrast,
                runtime_out.layer_tone_map_shoulder,
                runtime_out.layer_tone_map_gamma,
                runtime_out.layer_tone_map_saturation,
            ) {
                out.tone_map_enabled = true;
                out.tone_map_exposure = exposure.clamp(-8.0, 8.0);
                out.tone_map_contrast = contrast.clamp(0.0, 4.0);
                out.tone_map_shoulder = shoulder.clamp(0.0, 2.0);
                out.tone_map_gamma = gamma.clamp(0.0001, 8.0);
                out.tone_map_saturation = saturation.clamp(0.0, 4.0);
            }
            if let (
                Some(position),
                Some(angle),
                Some(width),
                Some(softness),
                Some(intensity),
                Some(color),
            ) = (
                runtime_out.layer_light_sweep_position,
                runtime_out.layer_light_sweep_angle,
                runtime_out.layer_light_sweep_width,
                runtime_out.layer_light_sweep_softness,
                runtime_out.layer_light_sweep_intensity,
                runtime_out.layer_light_sweep_color,
            ) {
                let weighted_intensity = (intensity.clamp(0.0, 8.0) * strength).clamp(0.0, 8.0);
                if weighted_intensity > best_light_sweep_weighted_intensity {
                    best_light_sweep_weighted_intensity = weighted_intensity;
                    out.light_sweep_enabled = weighted_intensity > 0.001;
                    out.light_sweep_position = position.clamp(-2.0, 3.0);
                    out.light_sweep_angle = angle;
                    out.light_sweep_width = width.clamp(0.0, 2.0);
                    out.light_sweep_softness = softness.clamp(0.0, 2.0);
                    out.light_sweep_intensity = weighted_intensity;
                    out.light_sweep_color = [
                        color[0] as f32 / 255.0,
                        color[1] as f32 / 255.0,
                        color[2] as f32 / 255.0,
                        color[3] as f32 / 255.0,
                    ];
                }
            }
        }

        // Preserve legacy direct layer controls when no Layer FX clip is active.
        if !has_active_layer && !layer_effects.is_identity() {
            let layer = layer_effects.normalized();
            out.brightness += layer.brightness;
            out.contrast *= layer.contrast;
            out.saturation *= layer.saturation;
            out.signed_blur_sigma += layer.blur_sigma;
        }
        if has_active_layer {
            out.opacity_factor = (opacity_strength * transition_opacity).clamp(0.0, 1.0);
            out.lut_mix = best_lut_mix.clamp(0.0, 1.0);
        }

        out.brightness = out.brightness.clamp(-1.0, 1.0);
        out.contrast = out.contrast.clamp(0.0, 2.0);
        out.saturation = out.saturation.clamp(0.0, 2.0);
        out.signed_blur_sigma = out.signed_blur_sigma.clamp(-64.0, 64.0);
        out.opacity_factor = out.opacity_factor.clamp(0.0, 1.0);
        out
    }

    fn alpha_over_rgba(dst: &mut [u8], src: &[u8]) {
        let px_count = dst.len() / 4;
        for i in 0..px_count {
            let off = i * 4;
            let sa = src[off + 3] as u32;
            if sa == 0 {
                continue;
            }
            let inv = 255u32.saturating_sub(sa);

            let dr = dst[off] as u32;
            let dg = dst[off + 1] as u32;
            let db = dst[off + 2] as u32;
            let da = dst[off + 3] as u32;

            let sr = src[off] as u32;
            let sg = src[off + 1] as u32;
            let sb = src[off + 2] as u32;

            dst[off] = ((sr * sa + dr * inv + 127) / 255) as u8;
            dst[off + 1] = ((sg * sa + dg * inv + 127) / 255) as u8;
            dst[off + 2] = ((sb * sa + db * inv + 127) / 255) as u8;
            dst[off + 3] = (sa + ((da * inv + 127) / 255)).min(255) as u8;
        }
    }

    fn key_near_black_to_alpha_rgba(frame: &mut [u8]) {
        for px in frame.chunks_exact_mut(4) {
            if px[3] == 0 {
                continue;
            }
            let r = px[0];
            let g = px[1];
            let b = px[2];
            if r <= 18 && g <= 18 && b <= 18 {
                px[0] = 0;
                px[1] = 0;
                px[2] = 0;
                px[3] = 0;
            }
        }
    }

    fn multiply_alpha_rgba(frame: &mut [u8], opacity: f32) {
        let opacity = opacity.clamp(0.0, 1.0);
        if opacity >= 0.999 {
            return;
        }
        for px in frame.chunks_exact_mut(4) {
            px[3] = ((px[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
        }
    }

    fn scene_layer_script_hash(script: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        script.hash(&mut hasher);
        hasher.finish()
    }

    fn prepare_scene_layer_export_states(
        layer_effect_clips: &[LayerEffectClip],
    ) -> Vec<SceneLayerExportState> {
        let mut states = Vec::new();
        for layer in layer_effect_clips {
            if !layer.motionloom_enabled {
                continue;
            }
            let script = layer.motionloom_script.trim();
            if script.is_empty() || !is_graph_script(script) {
                continue;
            }
            let Ok(graph) = parse_graph_script(script) else {
                continue;
            };
            if !graph.has_scene_nodes() {
                continue;
            }
            states.push(SceneLayerExportState {
                layer_id: layer.id,
                script_hash: Self::scene_layer_script_hash(script),
                fps: graph.fps.max(1.0),
                graph,
                last_frame: None,
                last_rgba: Vec::new(),
            });
        }
        states
    }

    fn scene_layer_state_mut<'a>(
        states: &'a mut [SceneLayerExportState],
        layer: &LayerEffectClip,
    ) -> Option<&'a mut SceneLayerExportState> {
        let script_hash = Self::scene_layer_script_hash(layer.motionloom_script.trim());
        states
            .iter_mut()
            .find(|state| state.layer_id == layer.id && state.script_hash == script_hash)
    }

    fn render_scene_layer_overlay_rgba(
        state: &mut SceneLayerExportState,
        frame: u32,
        width: u32,
        height: u32,
    ) -> Result<&[u8], ExportError> {
        if state.last_frame == Some(frame) && !state.last_rgba.is_empty() {
            return Ok(&state.last_rgba);
        }

        let rendered = pollster::block_on(render_scene_graph_frame(
            &state.graph,
            frame,
            SceneRenderProfile::Cpu,
        ))
        .map_err(|err| ExportError::GpuOpacityPathUnavailable {
            reason: format!("Layer FX scene render failed: {err}"),
        })?;
        let rendered_w = rendered.width();
        let rendered_h = rendered.height();
        let rgba = if rendered_w == width && rendered_h == height {
            rendered.into_raw()
        } else {
            let resized = image::imageops::resize(
                &rendered,
                width,
                height,
                image::imageops::FilterType::Lanczos3,
            );
            resized.into_raw()
        };
        state.last_frame = Some(frame);
        state.last_rgba = rgba;
        Ok(&state.last_rgba)
    }

    fn composite_scene_layer_effects_rgba(
        dst: &mut [u8],
        states: &mut [SceneLayerExportState],
        layer_effect_clips: &[LayerEffectClip],
        t: Duration,
        width: u32,
        height: u32,
    ) -> Result<(), ExportError> {
        if states.is_empty() {
            return Ok(());
        }

        for layer in layer_effect_clips {
            if !layer.motionloom_enabled {
                continue;
            }
            let Some(local) = layer.local_time(t) else {
                continue;
            };
            let opacity = layer.envelope_factor_at(t).clamp(0.0, 1.0);
            if opacity <= 0.001 {
                continue;
            }
            let Some(state) = Self::scene_layer_state_mut(states, layer) else {
                continue;
            };
            let frame = (local.as_secs_f32() * state.fps).floor().max(0.0) as u32;
            let overlay = Self::render_scene_layer_overlay_rgba(state, frame, width, height)?;
            if opacity >= 0.999 {
                Self::alpha_over_rgba(dst, overlay);
            } else {
                let mut overlay = overlay.to_vec();
                Self::multiply_alpha_rgba(&mut overlay, opacity);
                Self::alpha_over_rgba(dst, &overlay);
            }
        }

        Ok(())
    }

    fn load_unified_subtitle_overlays(
        subtitle_overlays: &[RenderedSubtitle],
        width: u32,
        height: u32,
    ) -> Result<Vec<(f64, f64, Vec<u8>)>, ExportError> {
        let mut out = Vec::with_capacity(subtitle_overlays.len());
        for overlay in subtitle_overlays {
            let image = image::open(&overlay.path)
                .map_err(|source| ExportError::LoadSubtitleOverlay {
                    path: overlay.path.clone(),
                    source,
                })?
                .into_rgba8();
            let image = if image.width() == width && image.height() == height {
                image
            } else {
                image::imageops::resize(
                    &image,
                    width,
                    height,
                    image::imageops::FilterType::Lanczos3,
                )
            };
            out.push((overlay.start, overlay.end, image.into_raw()));
        }
        Ok(out)
    }

    fn load_unified_still_frame_rgba(
        path: &str,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, ExportError> {
        // Decode still images once so export does not pipe identical frames through FFmpeg.
        let image_path = PathBuf::from(path);
        let image = image::open(&image_path)
            .map_err(|source| ExportError::LoadStillFrame {
                path: image_path,
                source,
            })?
            .into_rgba8();
        let src_w = image.width().max(1);
        let src_h = image.height().max(1);
        let scale = (width as f32 / src_w as f32).min(height as f32 / src_h as f32);
        let scaled_w = ((src_w as f32 * scale).round() as u32).clamp(1, width.max(1));
        let scaled_h = ((src_h as f32 * scale).round() as u32).clamp(1, height.max(1));
        let resized = if scaled_w == src_w && scaled_h == src_h {
            image
        } else {
            image::imageops::resize(
                &image,
                scaled_w,
                scaled_h,
                image::imageops::FilterType::Lanczos3,
            )
        };
        let mut canvas = image::RgbaImage::from_pixel(width, height, image::Rgba([0, 0, 0, 0]));
        let x = ((width.saturating_sub(scaled_w)) / 2) as i64;
        let y = ((height.saturating_sub(scaled_h)) / 2) as i64;
        image::imageops::overlay(&mut canvas, &resized, x, y);
        Ok(canvas.into_raw())
    }

    fn is_motionloom_scene_generated_path(path: &str) -> bool {
        let normalized = path.replace('\\', "/").to_ascii_lowercase();
        normalized.contains("/motionloom_generated/motionloom_scene_")
            && (normalized.ends_with(".mp4") || normalized.ends_with(".mov"))
    }

    fn build_clip_decode_rgba_filter(
        width: u32,
        height: u32,
        fps: u32,
        preserve_black_as_alpha: bool,
    ) -> String {
        if preserve_black_as_alpha {
            return format!(
                "fps={fps},setsar=1,scale=w={w}:h={h}:force_original_aspect_ratio=decrease:eval=frame,format=rgba,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black@0,colorkey=0x000000:0.025:0.0,format=rgba",
                fps = fps,
                w = width,
                h = height
            );
        }

        format!(
            "fps={fps},setsar=1,scale=w={w}:h={h}:force_original_aspect_ratio=decrease:eval=frame,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,format=rgba",
            fps = fps,
            w = width,
            h = height
        )
    }

    fn build_clip_decode_rgba_args(
        clip: &Clip,
        source_start: Duration,
        duration: Duration,
        width: u32,
        height: u32,
        fps: u32,
        preserve_black_as_alpha: bool,
    ) -> Vec<String> {
        let decode_filter =
            Self::build_clip_decode_rgba_filter(width, height, fps, preserve_black_as_alpha);

        let mut args = vec![
            "-hide_banner".to_string(),
            "-nostats".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
        ];
        if is_image_ext(&clip.file_path) {
            args.push("-loop".to_string());
            args.push("1".to_string());
            args.push("-framerate".to_string());
            args.push(fps.to_string());
            args.push("-t".to_string());
            args.push(format!("{:.6}", duration.as_secs_f64()));
            args.push("-i".to_string());
            args.push(clip.file_path.clone());
        } else {
            args.push("-ss".to_string());
            args.push(format!("{:.6}", source_start.as_secs_f64()));
            args.push("-t".to_string());
            args.push(format!("{:.6}", duration.as_secs_f64()));
            args.push("-i".to_string());
            args.push(clip.file_path.clone());
        }
        args.push("-an".to_string());
        args.push("-vf".to_string());
        args.push(decode_filter);
        args.push("-pix_fmt".to_string());
        args.push("rgba".to_string());
        args.push("-f".to_string());
        args.push("rawvideo".to_string());
        args.push("pipe:1".to_string());
        args
    }

    fn run_unified_gpu_render_export(
        ffmpeg_bin: &str,
        v1_clips: &[Clip],
        audio_tracks: &[AudioTrack],
        video_tracks: &[VideoTrack],
        subtitle_overlays: &[RenderedSubtitle],
        temp_out_path: &str,
        canvas_w: f32,
        canvas_h: f32,
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
        export_preset: ExportPreset,
        export_settings: &ExportSettings,
        timeline_max: Duration,
        export_range: Option<ExportRange>,
        progress_total: Duration,
        cancel_requested: &Arc<AtomicBool>,
        on_progress: &mut impl FnMut(ExportProgress),
    ) -> Result<(), ExportError> {
        #[derive(Clone, Copy)]
        struct TrackClipSpec<'a> {
            clip: &'a Clip,
            active_start: Duration,
            active_end: Duration,
            source_start: Duration,
        }

        struct DecoderProc {
            child: std::process::Child,
            stdout: std::process::ChildStdout,
            stderr_join: std::thread::JoinHandle<String>,
        }

        struct LayerState<'a> {
            spec: TrackClipSpec<'a>,
            decoder: Option<DecoderProc>,
            still_frame_rgba: Option<Vec<u8>>,
            started: bool,
            ended: bool,
            local_mask_layers: Vec<VideoLocalMaskLayer>,
        }

        let fps = export_settings.normalized_fps().clamp(1, 144);
        let width = canvas_w.max(1.0).round() as u32;
        let height = canvas_h.max(1.0).round() as u32;
        let frame_bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        let export_start = export_range.map(|r| r.start).unwrap_or(Duration::ZERO);
        let export_end = export_range
            .map(|r| r.end)
            .unwrap_or_else(|| export_start + progress_total);
        if export_end <= export_start + Duration::from_millis(1) {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "effective duration is too short".to_string(),
            });
        }

        let mut specs: Vec<TrackClipSpec<'_>> = Vec::new();
        for clip in v1_clips {
            let active_start = export_start.max(clip.start);
            let active_end = export_end.min(clip.end());
            if active_end <= active_start + Duration::from_millis(1) {
                continue;
            }
            specs.push(TrackClipSpec {
                clip,
                active_start,
                active_end,
                source_start: clip.source_in + active_start.saturating_sub(clip.start),
            });
        }
        for track in video_tracks {
            for clip in &track.clips {
                let active_start = export_start.max(clip.start);
                let active_end = export_end.min(clip.end());
                if active_end <= active_start + Duration::from_millis(1) {
                    continue;
                }
                specs.push(TrackClipSpec {
                    clip,
                    active_start,
                    active_end,
                    source_start: clip.source_in + active_start.saturating_sub(clip.start),
                });
            }
        }
        if specs.is_empty() {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "no visual clips intersect export range".to_string(),
            });
        }

        let mut encode_args = vec![
            "-y".to_string(),
            "-hide_banner".to_string(),
            "-f".to_string(),
            "rawvideo".to_string(),
            "-pix_fmt".to_string(),
            "rgba".to_string(),
            "-s:v".to_string(),
            format!("{}x{}", width, height),
            "-r".to_string(),
            fps.to_string(),
            "-i".to_string(),
            "pipe:0".to_string(),
        ];

        let mut audio_track_indices: Vec<Vec<Option<usize>>> = Vec::new();
        let mut next_input_idx = 1usize;
        let mut audio_presence_cache: std::collections::HashMap<String, bool> =
            std::collections::HashMap::new();
        if !export_preset.omits_audio() {
            for track in audio_tracks {
                let mut indices = Vec::with_capacity(track.clips.len());
                for clip in &track.clips {
                    let has_audio = *audio_presence_cache
                        .entry(clip.file_path.clone())
                        .or_insert_with(|| has_audio_stream(&clip.file_path));
                    if has_audio {
                        encode_args.push("-i".to_string());
                        encode_args.push(clip.file_path.clone());
                        indices.push(Some(next_input_idx));
                        next_input_idx += 1;
                    } else {
                        indices.push(None);
                    }
                }
                audio_track_indices.push(indices);
            }
        }

        let mut filter_parts = Vec::<String>::new();
        let mut mapped_audio_tag: Option<String> = None;
        if export_preset == ExportPreset::Gif {
            filter_parts.push(format!(
                "[0:v:0]fps={fps},split[v_gif_palette_src][v_gif_use_src];\
                 [v_gif_palette_src]palettegen=stats_mode=diff[p_gif];\
                 [v_gif_use_src][p_gif]paletteuse=dither=sierra2_4a[v_gif]"
            ));
        } else if !export_preset.omits_audio() {
            let mut audio_mix_inputs = Vec::new();
            for (track_idx, track) in audio_tracks.iter().enumerate() {
                if track.clips.is_empty() {
                    continue;
                }
                let Some(indices) = audio_track_indices.get(track_idx) else {
                    continue;
                };
                let label = format!("ugpu_audio_{track_idx}");
                let mut concat_inputs = String::new();
                let mut concat_count = 0usize;
                let mut cursor = Duration::ZERO;

                for (clip_idx, clip) in track.clips.iter().enumerate() {
                    if clip.start > cursor {
                        let gap = (clip.start - cursor).as_secs_f64();
                        filter_parts.push(format!(
                            "anullsrc=channel_layout=stereo:sample_rate=48000:d={gap:.6}[{label}_gap_{clip_idx}]"
                        ));
                        concat_inputs.push_str(&format!("[{label}_gap_{clip_idx}]"));
                        concat_count += 1;
                    }

                    let clip_tag = format!("[{label}_clip_{clip_idx}]");
                    if let Some(Some(input_idx)) = indices.get(clip_idx) {
                        let raw_tag = format!("[{label}_raw_{clip_idx}]");
                        let start = clip.source_in.as_secs_f64();
                        let end = (clip.source_in + clip.duration).as_secs_f64();
                        filter_parts.push(format!(
                            "[{input_idx}:a:0]aresample=48000,atrim=start={start:.6}:end={end:.6},asetpts=PTS-STARTPTS{raw_tag}"
                        ));
                        let gain_linear = 10.0_f64.powf((clip.audio_gain_db as f64) / 20.0);
                        if (gain_linear - 1.0).abs() > 0.0005 {
                            filter_parts
                                .push(format!("{raw_tag}volume={gain_linear:.6}{clip_tag}"));
                        } else {
                            filter_parts.push(format!("{raw_tag}anull{clip_tag}"));
                        }
                    } else {
                        let silence_dur = clip.duration.as_secs_f64().max(0.001);
                        filter_parts.push(format!(
                            "anullsrc=channel_layout=stereo:sample_rate=48000:d={silence_dur:.6}{clip_tag}"
                        ));
                    }
                    concat_inputs.push_str(&clip_tag);
                    concat_count += 1;
                    cursor = clip.start + clip.duration;
                }

                if cursor < timeline_max {
                    let gap = (timeline_max - cursor).as_secs_f64();
                    filter_parts.push(format!(
                        "anullsrc=channel_layout=stereo:sample_rate=48000:d={gap:.6}[{label}_gap_end]"
                    ));
                    concat_inputs.push_str(&format!("[{label}_gap_end]"));
                    concat_count += 1;
                }

                if concat_count == 0 {
                    continue;
                }
                let track_tag = format!("[{label}_track]");
                filter_parts.push(format!(
                    "{concat_inputs}concat=n={concat_count}:v=0:a=1{track_tag}"
                ));
                let gain_linear = 10.0_f64.powf((track.gain_db as f64) / 20.0);
                if (gain_linear - 1.0).abs() > 0.0005 {
                    let gain_tag = format!("[{label}_gain]");
                    filter_parts.push(format!("{track_tag}volume={gain_linear:.6}{gain_tag}"));
                    audio_mix_inputs.push(gain_tag);
                } else {
                    audio_mix_inputs.push(track_tag);
                }
            }

            if !audio_mix_inputs.is_empty() {
                let mixed_tag = if audio_mix_inputs.len() == 1 {
                    let tag = "[ugpu_aout]".to_string();
                    filter_parts.push(format!("{}anull{}", audio_mix_inputs[0], tag));
                    tag
                } else {
                    let tag = "[ugpu_aout]".to_string();
                    filter_parts.push(format!(
                        "{}amix=inputs={}:duration=longest:dropout_transition=0{}",
                        audio_mix_inputs.join(""),
                        audio_mix_inputs.len(),
                        tag
                    ));
                    tag
                };
                let trimmed_tag = "[ugpu_aexport]".to_string();
                filter_parts.push(format!(
                    "{mixed_tag}atrim=start={:.6}:end={:.6},asetpts=PTS-STARTPTS{}",
                    export_start.as_secs_f64(),
                    export_end.as_secs_f64(),
                    trimmed_tag
                ));
                mapped_audio_tag = Some(trimmed_tag);
            }
        }

        if !filter_parts.is_empty() {
            encode_args.push("-filter_complex".to_string());
            encode_args.push(filter_parts.join(";"));
        }
        encode_args.push("-map".to_string());
        if export_preset == ExportPreset::Gif {
            encode_args.push("[v_gif]".to_string());
        } else {
            encode_args.push("0:v:0".to_string());
        }
        if let Some(audio_tag) = mapped_audio_tag.as_ref() {
            encode_args.push("-map".to_string());
            encode_args.push(audio_tag.clone());
        }
        export_preset.push_output_args(&mut encode_args, export_settings, fps);
        if mapped_audio_tag.is_some() {
            encode_args.push("-shortest".to_string());
        }
        encode_args.push("-nostats".to_string());
        encode_args.push("-loglevel".to_string());
        encode_args.push("error".to_string());
        encode_args.push(temp_out_path.to_string());

        println!(
            "[Export][Unified GPU] Encode ffmpeg: {} {:?}",
            ffmpeg_bin, encode_args
        );
        let mut encode_child = Command::new(ffmpeg_bin)
            .args(&encode_args)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| ExportError::ExecuteStageFfmpeg {
                stage: "encode",
                source,
            })?;
        let mut encode_stdin = encode_child
            .stdin
            .take()
            .ok_or(ExportError::MissingFfmpegPipe {
                stage: "encode",
                pipe: "stdin",
            })?;
        let encode_stderr = encode_child
            .stderr
            .take()
            .ok_or(ExportError::MissingFfmpegPipe {
                stage: "encode",
                pipe: "stderr",
            })?;
        let encode_stderr_handle = std::thread::spawn(move || {
            let mut msg = String::new();
            let mut reader = BufReader::new(encode_stderr);
            let _ = reader.read_to_string(&mut msg);
            msg
        });

        let mut layer_states: Vec<LayerState<'_>> = specs
            .into_iter()
            .map(|spec| LayerState {
                local_mask_layers: Self::local_mask_layers_for_gpu(spec.clip),
                spec,
                decoder: None,
                still_frame_rgba: None,
                started: false,
                ended: false,
            })
            .collect();
        let mut motionloom_runtime_cache: std::collections::HashMap<u64, Option<RuntimeProgram>> =
            std::collections::HashMap::new();
        let mut scene_layer_export_states =
            Self::prepare_scene_layer_export_states(layer_effect_clips);

        let mut frame_rgba = vec![0u8; frame_bytes];
        let mut composed_rgba = vec![0u8; frame_bytes];
        let subtitle_overlay_pixels =
            Self::load_unified_subtitle_overlays(subtitle_overlays, width, height)?;
        let total_frames = ((progress_total.as_secs_f64() * f64::from(fps)).ceil() as u64).max(1);

        on_progress(ExportProgress {
            rendered: Duration::ZERO,
            total: progress_total,
            speed: None,
        });

        // Track loop error so we can clean up all spawned processes before returning.
        let mut loop_err: Option<ExportError> = None;

        for frame_idx in 0..total_frames {
            if loop_err.is_some() {
                break;
            }
            if cancel_requested.load(Ordering::Relaxed) {
                loop_err = Some(ExportError::Cancelled);
                break;
            }

            let t = export_start + Duration::from_secs_f64(frame_idx as f64 / f64::from(fps));
            let layer_fx = Self::unified_layer_frame_effects_at(
                layer_effects,
                layer_effect_clips,
                t,
                &mut motionloom_runtime_cache,
            );
            composed_rgba.fill(0);

            for state in &mut layer_states {
                if loop_err.is_some() {
                    break;
                }
                if t < state.spec.active_start || t >= state.spec.active_end {
                    if t >= state.spec.active_end && !state.ended {
                        if let Some(mut d) = state.decoder.take() {
                            let _ = d.child.kill();
                            let _ = d.child.wait();
                            let _ = d.stderr_join.join();
                        }
                        state.ended = true;
                    }
                    continue;
                }

                if !state.started {
                    if is_image_ext(&state.spec.clip.file_path) {
                        println!(
                            "[Export][Unified GPU] Load still RGBA: {}",
                            state.spec.clip.file_path
                        );
                        match Self::load_unified_still_frame_rgba(
                            &state.spec.clip.file_path,
                            width,
                            height,
                        ) {
                            Ok(frame) => state.still_frame_rgba = Some(frame),
                            Err(err) => {
                                loop_err = Some(err);
                                break;
                            }
                        }
                    } else {
                        let decode_duration = state
                            .spec
                            .active_end
                            .saturating_sub(state.spec.active_start);
                        let decode_args = Self::build_clip_decode_rgba_args(
                            state.spec.clip,
                            state.spec.source_start,
                            decode_duration,
                            width,
                            height,
                            fps,
                            Self::is_motionloom_scene_generated_path(&state.spec.clip.file_path),
                        );
                        println!(
                            "[Export][Unified GPU] Decode ffmpeg: {} {:?}",
                            ffmpeg_bin, decode_args
                        );
                        let spawn_result = Command::new(ffmpeg_bin)
                            .args(&decode_args)
                            .stdout(Stdio::piped())
                            .stderr(Stdio::piped())
                            .spawn();
                        let mut child = match spawn_result {
                            Ok(c) => c,
                            Err(source) => {
                                loop_err = Some(ExportError::ExecuteStageFfmpeg {
                                    stage: "decode",
                                    source,
                                });
                                break;
                            }
                        };
                        let stdout = match child.stdout.take() {
                            Some(s) => s,
                            None => {
                                let _ = child.kill();
                                loop_err = Some(ExportError::MissingFfmpegPipe {
                                    stage: "decode",
                                    pipe: "stdout",
                                });
                                break;
                            }
                        };
                        let stderr = match child.stderr.take() {
                            Some(s) => s,
                            None => {
                                let _ = child.kill();
                                loop_err = Some(ExportError::MissingFfmpegPipe {
                                    stage: "decode",
                                    pipe: "stderr",
                                });
                                break;
                            }
                        };
                        let stderr_join = std::thread::spawn(move || {
                            let mut msg = String::new();
                            let mut reader = BufReader::new(stderr);
                            let _ = reader.read_to_string(&mut msg);
                            msg
                        });
                        state.decoder = Some(DecoderProc {
                            child,
                            stdout,
                            stderr_join,
                        });
                    }
                    state.started = true;
                }

                if let Some(still_frame) = state.still_frame_rgba.as_ref() {
                    frame_rgba.copy_from_slice(still_frame);
                } else {
                    let Some(decoder) = state.decoder.as_mut() else {
                        loop_err = Some(ExportError::MissingInternalState {
                            state: "unified gpu decode state",
                        });
                        break;
                    };
                    if let Err(source) = decoder.stdout.read_exact(&mut frame_rgba) {
                        loop_err = Some(ExportError::ReadFfmpegFrame {
                            stage: "decode",
                            source,
                        });
                        break;
                    }
                }
                let key_motionloom_black =
                    Self::is_motionloom_scene_generated_path(&state.spec.clip.file_path);
                if key_motionloom_black {
                    Self::key_near_black_to_alpha_rgba(&mut frame_rgba);
                }

                let local_t = t.saturating_sub(state.spec.clip.start);
                let (clip_tint_h, clip_tint_s, clip_tint_l, clip_tint_a) =
                    state.spec.clip.get_hsla_overlay();
                let use_layer_tint = layer_fx.tint_alpha > 0.001;
                let opacity = state.spec.clip.sample_opacity(local_t)
                    * state.spec.clip.sample_fade_factor(local_t)
                    * state.spec.clip.sample_dissolve_factor(local_t)
                    * layer_fx.opacity_factor;
                let mut params = BgraGpuEffectParams {
                    brightness: (state.spec.clip.sample_brightness(local_t) + layer_fx.brightness)
                        .clamp(-1.0, 1.0),
                    contrast: (state.spec.clip.sample_contrast(local_t) * layer_fx.contrast)
                        .clamp(0.0, 2.0),
                    saturation: (state.spec.clip.sample_saturation(local_t) * layer_fx.saturation)
                        .clamp(0.0, 2.0),
                    lut_mix: layer_fx.lut_mix,
                    opacity: opacity.clamp(0.0, 1.0),
                    rotation_deg: state.spec.clip.sample_rotation(local_t),
                    transform_scale: state.spec.clip.sample_scale(local_t),
                    transform_pos_x: state.spec.clip.sample_pos_x(local_t),
                    transform_pos_y: state.spec.clip.sample_pos_y(local_t),
                    transform_ref_width: width as f32,
                    transform_ref_height: height as f32,
                    tint_hue: if use_layer_tint {
                        layer_fx.tint_hue
                    } else {
                        clip_tint_h
                    },
                    tint_saturation: if use_layer_tint {
                        layer_fx.tint_saturation
                    } else {
                        clip_tint_s
                    },
                    tint_lightness: if use_layer_tint {
                        layer_fx.tint_lightness
                    } else {
                        clip_tint_l
                    },
                    tint_alpha: if use_layer_tint {
                        layer_fx.tint_alpha
                    } else {
                        clip_tint_a
                    },
                    blur_sigma: (state.spec.clip.sample_blur(local_t) + layer_fx.signed_blur_sigma)
                        .clamp(-64.0, 64.0),
                    blur_mode: layer_fx.blur_mode,
                    bloom_threshold: layer_fx.bloom_threshold,
                    bloom_intensity: layer_fx.bloom_intensity,
                    bloom_sigma: layer_fx.bloom_sigma,
                    bloom_tint_color: [1.0, 1.0, 1.0, 1.0],
                    tone_map_enabled: layer_fx.tone_map_enabled,
                    tone_map_exposure: layer_fx.tone_map_exposure,
                    tone_map_contrast: layer_fx.tone_map_contrast,
                    tone_map_shoulder: layer_fx.tone_map_shoulder,
                    tone_map_gamma: layer_fx.tone_map_gamma,
                    tone_map_saturation: layer_fx.tone_map_saturation,
                    light_sweep_enabled: layer_fx.light_sweep_enabled,
                    light_sweep_position: layer_fx.light_sweep_position,
                    light_sweep_angle: layer_fx.light_sweep_angle,
                    light_sweep_width: layer_fx.light_sweep_width,
                    light_sweep_softness: layer_fx.light_sweep_softness,
                    light_sweep_intensity: layer_fx.light_sweep_intensity,
                    light_sweep_color: layer_fx.light_sweep_color,
                    texture_overlay_enabled: false,
                    texture_overlay_kind: 1.0,
                    texture_overlay_scale: 72.0,
                    texture_overlay_strength: 0.0,
                    texture_overlay_contrast: 1.0,
                    texture_overlay_seed: 0.0,
                    texture_overlay_bump_strength: 0.0,
                    texture_overlay_relief: 0.0,
                };
                let unsupported_effects = params.apply_process_effects(&layer_fx.process_effects);
                if !unsupported_effects.is_empty() {
                    log::debug!(
                        "[Export][LayerFX] skipped unsupported MotionLoom process effects: {}",
                        unsupported_effects.join(", ")
                    );
                }

                Self::rgba_bgra_swap_in_place(&mut frame_rgba);
                if !process_bgra_effects_with_params(
                    &mut frame_rgba,
                    width,
                    height,
                    params,
                    &state.local_mask_layers,
                ) {
                    loop_err = Some(ExportError::ProcessGpuOpacityFrame {
                        message: "unified WGPU BGRA effect processing failed".to_string(),
                    });
                    break;
                }
                Self::rgba_bgra_swap_in_place(&mut frame_rgba);
                if key_motionloom_black {
                    Self::key_near_black_to_alpha_rgba(&mut frame_rgba);
                }
                Self::alpha_over_rgba(&mut composed_rgba, &frame_rgba);
            }

            if loop_err.is_some() {
                break;
            }

            if let Err(err) = Self::composite_scene_layer_effects_rgba(
                &mut composed_rgba,
                &mut scene_layer_export_states,
                layer_effect_clips,
                t,
                width,
                height,
            ) {
                loop_err = Some(err);
                break;
            }

            let t_sec = t.as_secs_f64();
            for (start, end, pixels) in &subtitle_overlay_pixels {
                if t_sec >= *start && t_sec < *end {
                    Self::alpha_over_rgba(&mut composed_rgba, pixels);
                }
            }

            if let Err(source) = encode_stdin.write_all(&composed_rgba) {
                loop_err = Some(ExportError::WriteFfmpegFrame {
                    stage: "encode",
                    source,
                });
                break;
            }

            if frame_idx % 8 == 0 {
                let rendered =
                    Duration::from_secs_f64((frame_idx as f64 / f64::from(fps)).max(0.0))
                        .min(progress_total);
                on_progress(ExportProgress {
                    rendered,
                    total: progress_total,
                    speed: None,
                });
            }
        }

        // Error-path cleanup: kill all spawned ffmpeg processes so no orphans remain.
        // On success, the post-loop section below waits for processes normally instead.
        if let Some(err) = loop_err {
            for state in &mut layer_states {
                if let Some(mut d) = state.decoder.take() {
                    let _ = d.child.kill();
                    let _ = d.child.wait();
                    let _ = d.stderr_join.join();
                }
            }
            // Close stdin pipe first so blocked write_all() can fail and unblock
            drop(encode_stdin);
            let _ = encode_child.kill();
            let _ = encode_child.wait();
            let _ = encode_stderr_handle.join();
            return Err(err);
        }

        for state in &mut layer_states {
            if let Some(mut d) = state.decoder.take() {
                let _ = d.child.kill();
                let _ = d.child.wait();
                let _ = d.stderr_join.join();
            }
        }

        drop(encode_stdin);
        let encode_status =
            encode_child
                .wait()
                .map_err(|source| ExportError::WaitStageFfmpegProcess {
                    stage: "encode",
                    source,
                })?;
        let encode_stderr = encode_stderr_handle.join().unwrap_or_default();
        if !encode_status.success() {
            let msg = encode_stderr.trim();
            return Err(ExportError::StageFfmpegFailed {
                stage: "encode",
                status: encode_status.to_string(),
                stderr: if msg.is_empty() {
                    "empty stderr".to_string()
                } else {
                    msg.to_string()
                },
            });
        }
        Ok(())
    }

    fn run_true_gpu_multitrack_opacity_export(
        ffmpeg_bin: &str,
        v1_clips: &[Clip],
        video_tracks: &[VideoTrack],
        temp_out_path: &str,
        canvas_w: f32,
        canvas_h: f32,
        export_settings: &ExportSettings,
        export_range: Option<ExportRange>,
        progress_total: Duration,
        cancel_requested: &Arc<AtomicBool>,
        on_progress: &mut impl FnMut(ExportProgress),
    ) -> Result<(), ExportError> {
        #[derive(Clone, Copy)]
        struct TrackClipSpec<'a> {
            clip: &'a Clip,
            active_start: Duration,
            active_end: Duration,
            source_start: Duration,
        }

        struct DecoderProc {
            child: std::process::Child,
            stdout: std::process::ChildStdout,
            stderr_join: std::thread::JoinHandle<String>,
        }

        struct LayerState<'a> {
            spec: TrackClipSpec<'a>,
            decoder: Option<DecoderProc>,
            started: bool,
            ended: bool,
        }

        let fps = export_settings.normalized_fps().clamp(1, 144);
        let width = canvas_w.max(1.0).round() as u32;
        let height = canvas_h.max(1.0).round() as u32;
        let audio_kbps = export_settings
            .normalized_audio_bitrate_kbps()
            .clamp(64, 512);
        let frame_bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);

        let export_start = export_range.map(|r| r.start).unwrap_or(Duration::ZERO);
        let export_end = export_range
            .map(|r| r.end)
            .unwrap_or_else(|| export_start + progress_total);
        if export_end <= export_start + Duration::from_millis(1) {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "effective duration is too short".to_string(),
            });
        }

        let mut specs: Vec<TrackClipSpec<'_>> = Vec::new();
        let v1 = &v1_clips[0];
        let v1_active_start = export_start.max(v1.start);
        let v1_active_end = export_end.min(v1.end());
        if v1_active_end <= v1_active_start + Duration::from_millis(1) {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "V1 clip does not intersect export range".to_string(),
            });
        }
        specs.push(TrackClipSpec {
            clip: v1,
            active_start: v1_active_start,
            active_end: v1_active_end,
            source_start: v1.source_in + v1_active_start.saturating_sub(v1.start),
        });
        for track in video_tracks {
            if track.clips.is_empty() {
                continue;
            }
            let clip = &track.clips[0];
            let active_start = export_start.max(clip.start);
            let active_end = export_end.min(clip.end());
            if active_end <= active_start + Duration::from_millis(1) {
                continue;
            }
            specs.push(TrackClipSpec {
                clip,
                active_start,
                active_end,
                source_start: clip.source_in + active_start.saturating_sub(clip.start),
            });
        }
        if specs.is_empty() {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "no visual clips intersect export range".to_string(),
            });
        }

        let encode_args = vec![
            "-y".to_string(),
            "-hide_banner".to_string(),
            "-f".to_string(),
            "rawvideo".to_string(),
            "-pix_fmt".to_string(),
            "rgba".to_string(),
            "-s:v".to_string(),
            format!("{}x{}", width, height),
            "-r".to_string(),
            fps.to_string(),
            "-i".to_string(),
            "pipe:0".to_string(),
            "-ss".to_string(),
            format!("{:.6}", export_start.as_secs_f64()),
            "-t".to_string(),
            format!("{:.6}", progress_total.as_secs_f64()),
            "-i".to_string(),
            v1.file_path.clone(),
            "-map".to_string(),
            "0:v:0".to_string(),
            "-map".to_string(),
            "1:a?".to_string(),
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
            "-nostats".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
            temp_out_path.to_string(),
        ];

        println!(
            "[Export][GPU Effect] Encode ffmpeg (phase2): {} {:?}",
            ffmpeg_bin, encode_args
        );

        let mut encode_child = Command::new(ffmpeg_bin)
            .args(&encode_args)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| ExportError::ExecuteStageFfmpeg {
                stage: "encode",
                source,
            })?;
        let mut encode_stdin = encode_child
            .stdin
            .take()
            .ok_or(ExportError::MissingFfmpegPipe {
                stage: "encode",
                pipe: "stdin",
            })?;
        let encode_stderr = encode_child
            .stderr
            .take()
            .ok_or(ExportError::MissingFfmpegPipe {
                stage: "encode",
                pipe: "stderr",
            })?;
        let encode_stderr_handle = std::thread::spawn(move || {
            let mut msg = String::new();
            let mut reader = BufReader::new(encode_stderr);
            let _ = reader.read_to_string(&mut msg);
            msg
        });

        let mut layer_states: Vec<LayerState<'_>> = specs
            .into_iter()
            .map(|spec| LayerState {
                spec,
                decoder: None,
                started: false,
                ended: false,
            })
            .collect();

        let mut opacity_processor = WgpuOpacityProcessor::new(width, height).map_err(|err| {
            ExportError::InitializeGpuOpacityProcessor {
                message: err.to_string(),
            }
        })?;
        let mut frame_rgba = vec![0u8; frame_bytes];
        let mut composed_rgba = vec![0u8; frame_bytes];
        let total_frames = ((progress_total.as_secs_f64() * f64::from(fps)).ceil() as u64).max(1);

        on_progress(ExportProgress {
            rendered: Duration::ZERO,
            total: progress_total,
            speed: None,
        });

        // Track loop error so we can clean up all spawned processes before returning.
        let mut loop_err: Option<ExportError> = None;

        for frame_idx in 0..total_frames {
            if loop_err.is_some() {
                break;
            }
            if cancel_requested.load(Ordering::Relaxed) {
                loop_err = Some(ExportError::Cancelled);
                break;
            }

            let t = export_start + Duration::from_secs_f64(frame_idx as f64 / f64::from(fps));
            composed_rgba.fill(0);

            for state in &mut layer_states {
                if loop_err.is_some() {
                    break;
                }
                if t < state.spec.active_start || t >= state.spec.active_end {
                    if t >= state.spec.active_end && !state.ended {
                        if let Some(mut d) = state.decoder.take() {
                            let _ = d.child.kill();
                            let _ = d.child.wait();
                            let _ = d.stderr_join.join();
                        }
                        state.ended = true;
                    }
                    continue;
                }

                if !state.started {
                    let decode_duration = state
                        .spec
                        .active_end
                        .saturating_sub(state.spec.active_start);
                    let decode_args = Self::build_clip_decode_rgba_args(
                        state.spec.clip,
                        state.spec.source_start,
                        decode_duration,
                        width,
                        height,
                        fps,
                        Self::is_motionloom_scene_generated_path(&state.spec.clip.file_path),
                    );
                    println!(
                        "[Export][GPU Effect] Decode ffmpeg (phase2): {} {:?}",
                        ffmpeg_bin, decode_args
                    );
                    let spawn_result = Command::new(ffmpeg_bin)
                        .args(&decode_args)
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn();
                    let mut child = match spawn_result {
                        Ok(c) => c,
                        Err(source) => {
                            loop_err = Some(ExportError::ExecuteStageFfmpeg {
                                stage: "decode",
                                source,
                            });
                            break;
                        }
                    };
                    let stdout = match child.stdout.take() {
                        Some(s) => s,
                        None => {
                            let _ = child.kill();
                            loop_err = Some(ExportError::MissingFfmpegPipe {
                                stage: "decode",
                                pipe: "stdout",
                            });
                            break;
                        }
                    };
                    let stderr = match child.stderr.take() {
                        Some(s) => s,
                        None => {
                            let _ = child.kill();
                            loop_err = Some(ExportError::MissingFfmpegPipe {
                                stage: "decode",
                                pipe: "stderr",
                            });
                            break;
                        }
                    };
                    let stderr_join = std::thread::spawn(move || {
                        let mut msg = String::new();
                        let mut reader = BufReader::new(stderr);
                        let _ = reader.read_to_string(&mut msg);
                        msg
                    });
                    state.decoder = Some(DecoderProc {
                        child,
                        stdout,
                        stderr_join,
                    });
                    state.started = true;
                }

                let Some(decoder) = state.decoder.as_mut() else {
                    loop_err = Some(ExportError::MissingInternalState {
                        state: "phase2 decode state",
                    });
                    break;
                };
                if let Err(source) = decoder.stdout.read_exact(&mut frame_rgba) {
                    loop_err = Some(ExportError::ReadFfmpegFrame {
                        stage: "decode",
                        source,
                    });
                    break;
                }

                let key_motionloom_black =
                    Self::is_motionloom_scene_generated_path(&state.spec.clip.file_path);
                if key_motionloom_black {
                    Self::key_near_black_to_alpha_rgba(&mut frame_rgba);
                }
                let local_t = t.saturating_sub(state.spec.clip.start);
                let opacity = state.spec.clip.sample_opacity(local_t).clamp(0.0, 1.0);
                match opacity_processor.process_rgba_frame(&frame_rgba, opacity) {
                    Ok(mut layer_frame) => {
                        if key_motionloom_black {
                            Self::key_near_black_to_alpha_rgba(&mut layer_frame);
                        }
                        Self::alpha_over_rgba(&mut composed_rgba, &layer_frame);
                    }
                    Err(err) => {
                        loop_err = Some(ExportError::ProcessGpuOpacityFrame {
                            message: err.to_string(),
                        });
                        break;
                    }
                }
            }

            if loop_err.is_some() {
                break;
            }

            if let Err(source) = encode_stdin.write_all(&composed_rgba) {
                loop_err = Some(ExportError::WriteFfmpegFrame {
                    stage: "encode",
                    source,
                });
                break;
            }

            if frame_idx % 8 == 0 {
                let rendered =
                    Duration::from_secs_f64((frame_idx as f64 / f64::from(fps)).max(0.0))
                        .min(progress_total);
                on_progress(ExportProgress {
                    rendered,
                    total: progress_total,
                    speed: None,
                });
            }
        }

        // Error-path cleanup: kill all spawned ffmpeg processes so no orphans remain.
        // On success, the post-loop section below waits for processes normally instead.
        if let Some(err) = loop_err {
            for state in &mut layer_states {
                if let Some(mut d) = state.decoder.take() {
                    let _ = d.child.kill();
                    let _ = d.child.wait();
                    let _ = d.stderr_join.join();
                }
            }
            drop(encode_stdin);
            let _ = encode_child.kill();
            let _ = encode_child.wait();
            let _ = encode_stderr_handle.join();
            return Err(err);
        }

        for state in &mut layer_states {
            if let Some(mut d) = state.decoder.take() {
                let _ = d.child.kill();
                let _ = d.child.wait();
                let _ = d.stderr_join.join();
            }
        }

        drop(encode_stdin);
        let encode_status =
            encode_child
                .wait()
                .map_err(|source| ExportError::WaitStageFfmpegProcess {
                    stage: "encode",
                    source,
                })?;
        let encode_stderr = encode_stderr_handle.join().unwrap_or_default();
        if !encode_status.success() {
            let msg = encode_stderr.trim();
            if msg.is_empty() {
                return Err(ExportError::StageFfmpegFailed {
                    stage: "encode",
                    status: encode_status.to_string(),
                    stderr: "empty stderr".to_string(),
                });
            }
            return Err(ExportError::StageFfmpegFailed {
                stage: "encode",
                status: encode_status.to_string(),
                stderr: msg.to_string(),
            });
        }
        Ok(())
    }

    fn run_true_gpu_single_clip_opacity_export(
        ffmpeg_bin: &str,
        clip: &Clip,
        temp_out_path: &str,
        canvas_w: f32,
        canvas_h: f32,
        export_settings: &ExportSettings,
        export_range: Option<ExportRange>,
        progress_total: Duration,
        cancel_requested: &Arc<AtomicBool>,
        on_progress: &mut impl FnMut(ExportProgress),
    ) -> Result<(), ExportError> {
        let mut local_start = Duration::ZERO;
        let mut local_end = clip.duration;
        if let Some(range) = export_range {
            let clip_end = clip.start + clip.duration;
            if range.start < clip.start || range.end > clip_end {
                return Err(ExportError::GpuOpacityPathUnavailable {
                    reason: "export range falls outside clip".to_string(),
                });
            }
            local_start = range.start.saturating_sub(clip.start);
            local_end = local_start + range.duration();
        }
        if local_end <= local_start + Duration::from_millis(1) {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "effective duration is too short".to_string(),
            });
        }

        let source_start = clip.source_in + local_start;
        let duration = local_end.saturating_sub(local_start);
        let fps = export_settings.normalized_fps().clamp(1, 144);
        let width = canvas_w.max(1.0).round() as u32;
        let height = canvas_h.max(1.0).round() as u32;
        let audio_kbps = export_settings
            .normalized_audio_bitrate_kbps()
            .clamp(64, 512);
        let frame_bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);

        let decode_filter = format!(
            "fps={fps},setsar=1,scale=w={w}:h={h}:force_original_aspect_ratio=decrease:eval=frame,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,format=rgba",
            fps = fps,
            w = width,
            h = height
        );

        let decode_args = vec![
            "-hide_banner".to_string(),
            "-nostats".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
            "-ss".to_string(),
            format!("{:.6}", source_start.as_secs_f64()),
            "-t".to_string(),
            format!("{:.6}", duration.as_secs_f64()),
            "-i".to_string(),
            clip.file_path.clone(),
            "-an".to_string(),
            "-vf".to_string(),
            decode_filter,
            "-pix_fmt".to_string(),
            "rgba".to_string(),
            "-f".to_string(),
            "rawvideo".to_string(),
            "pipe:1".to_string(),
        ];

        let encode_args = vec![
            "-y".to_string(),
            "-hide_banner".to_string(),
            "-f".to_string(),
            "rawvideo".to_string(),
            "-pix_fmt".to_string(),
            "rgba".to_string(),
            "-s:v".to_string(),
            format!("{}x{}", width, height),
            "-r".to_string(),
            fps.to_string(),
            "-i".to_string(),
            "pipe:0".to_string(),
            "-ss".to_string(),
            format!("{:.6}", source_start.as_secs_f64()),
            "-t".to_string(),
            format!("{:.6}", duration.as_secs_f64()),
            "-i".to_string(),
            clip.file_path.clone(),
            "-map".to_string(),
            "0:v:0".to_string(),
            "-map".to_string(),
            "1:a?".to_string(),
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
            "-nostats".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
            temp_out_path.to_string(),
        ];

        println!(
            "[Export][GPU Effect] Decode ffmpeg: {} {:?}",
            ffmpeg_bin, decode_args
        );
        println!(
            "[Export][GPU Effect] Encode ffmpeg: {} {:?}",
            ffmpeg_bin, encode_args
        );

        let mut decode_child = Command::new(ffmpeg_bin)
            .args(&decode_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| ExportError::ExecuteStageFfmpeg {
                stage: "decode",
                source,
            })?;

        let mut encode_child = Command::new(ffmpeg_bin)
            .args(&encode_args)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| ExportError::ExecuteStageFfmpeg {
                stage: "encode",
                source,
            })?;

        let mut decode_stdout =
            decode_child
                .stdout
                .take()
                .ok_or(ExportError::MissingFfmpegPipe {
                    stage: "decode",
                    pipe: "stdout",
                })?;
        let decode_stderr = decode_child
            .stderr
            .take()
            .ok_or(ExportError::MissingFfmpegPipe {
                stage: "decode",
                pipe: "stderr",
            })?;
        let decode_stderr_handle = std::thread::spawn(move || {
            let mut msg = String::new();
            let mut reader = BufReader::new(decode_stderr);
            let _ = reader.read_to_string(&mut msg);
            msg
        });

        let mut encode_stdin = encode_child
            .stdin
            .take()
            .ok_or(ExportError::MissingFfmpegPipe {
                stage: "encode",
                pipe: "stdin",
            })?;
        let encode_stderr = encode_child
            .stderr
            .take()
            .ok_or(ExportError::MissingFfmpegPipe {
                stage: "encode",
                pipe: "stderr",
            })?;
        let encode_stderr_handle = std::thread::spawn(move || {
            let mut msg = String::new();
            let mut reader = BufReader::new(encode_stderr);
            let _ = reader.read_to_string(&mut msg);
            msg
        });

        let mut processor = match WgpuOpacityProcessor::new(width, height) {
            Ok(processor) => processor,
            Err(err) => {
                let _ = decode_child.kill();
                // Close stdin pipe first so blocked write can fail and unblock
                drop(encode_stdin);
                let _ = encode_child.kill();
                let _ = decode_child.wait();
                let _ = encode_child.wait();
                let _ = decode_stderr_handle.join();
                let _ = encode_stderr_handle.join();
                return Err(ExportError::InitializeGpuOpacityProcessor {
                    message: err.to_string(),
                });
            }
        };

        on_progress(ExportProgress {
            rendered: Duration::ZERO,
            total: progress_total,
            speed: None,
        });

        let mut frame = vec![0u8; frame_bytes];
        let mut frame_idx: u64 = 0;
        loop {
            if cancel_requested.load(Ordering::Relaxed) {
                let _ = decode_child.kill();
                // Close stdin pipe first so blocked write can fail and unblock
                drop(encode_stdin);
                let _ = encode_child.kill();
                let _ = decode_child.wait();
                let _ = encode_child.wait();
                let _ = decode_stderr_handle.join();
                let _ = encode_stderr_handle.join();
                return Err(ExportError::Cancelled);
            }

            match decode_stdout.read_exact(&mut frame) {
                Ok(()) => {
                    let local_t = local_start
                        + Duration::from_secs_f64(frame_idx as f64 / f64::from(fps.max(1)));
                    let opacity = clip.sample_opacity(local_t).clamp(0.0, 1.0);
                    let processed = match processor.process_rgba_frame(&frame, opacity) {
                        Ok(processed) => processed,
                        Err(err) => {
                            let _ = decode_child.kill();
                            // Close stdin pipe first so blocked write can fail and unblock
                            drop(encode_stdin);
                            let _ = encode_child.kill();
                            let _ = decode_child.wait();
                            let _ = encode_child.wait();
                            let _ = decode_stderr_handle.join();
                            let _ = encode_stderr_handle.join();
                            return Err(ExportError::ProcessGpuOpacityFrame {
                                message: err.to_string(),
                            });
                        }
                    };
                    if let Err(err) = encode_stdin.write_all(&processed) {
                        let _ = decode_child.kill();
                        // Close stdin pipe first so blocked write can fail and unblock
                        drop(encode_stdin);
                        let _ = encode_child.kill();
                        let _ = decode_child.wait();
                        let _ = encode_child.wait();
                        let _ = decode_stderr_handle.join();
                        let _ = encode_stderr_handle.join();
                        return Err(ExportError::WriteFfmpegFrame {
                            stage: "encode",
                            source: err,
                        });
                    }
                    frame_idx = frame_idx.saturating_add(1);

                    if frame_idx % 8 == 0 {
                        let rendered =
                            Duration::from_secs_f64(frame_idx as f64 / f64::from(fps.max(1)))
                                .min(progress_total);
                        on_progress(ExportProgress {
                            rendered,
                            total: progress_total,
                            speed: None,
                        });
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(err) => {
                    let _ = decode_child.kill();
                    // Close stdin pipe first so blocked write can fail and unblock
                    drop(encode_stdin);
                    let _ = encode_child.kill();
                    let _ = decode_child.wait();
                    let _ = encode_child.wait();
                    let _ = decode_stderr_handle.join();
                    let _ = encode_stderr_handle.join();
                    return Err(ExportError::ReadFfmpegFrame {
                        stage: "decode",
                        source: err,
                    });
                }
            }
        }
        drop(encode_stdin);

        let decode_status =
            decode_child
                .wait()
                .map_err(|source| ExportError::WaitStageFfmpegProcess {
                    stage: "decode",
                    source,
                })?;
        let decode_stderr = decode_stderr_handle.join().unwrap_or_default();
        if !decode_status.success() {
            let msg = decode_stderr.trim();
            if msg.is_empty() {
                return Err(ExportError::StageFfmpegFailed {
                    stage: "decode",
                    status: decode_status.to_string(),
                    stderr: "empty stderr".to_string(),
                });
            }
            return Err(ExportError::StageFfmpegFailed {
                stage: "decode",
                status: decode_status.to_string(),
                stderr: msg.to_string(),
            });
        }

        let encode_status =
            encode_child
                .wait()
                .map_err(|source| ExportError::WaitStageFfmpegProcess {
                    stage: "encode",
                    source,
                })?;
        let encode_stderr = encode_stderr_handle.join().unwrap_or_default();
        if !encode_status.success() {
            let msg = encode_stderr.trim();
            if msg.is_empty() {
                return Err(ExportError::StageFfmpegFailed {
                    stage: "encode",
                    status: encode_status.to_string(),
                    stderr: "empty stderr".to_string(),
                });
            }
            return Err(ExportError::StageFfmpegFailed {
                stage: "encode",
                status: encode_status.to_string(),
                stderr: msg.to_string(),
            });
        }
        Ok(())
    }

    fn status_is_sigsegv(status: &std::process::ExitStatus) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            status.signal() == Some(11)
        }
        #[cfg(not(unix))]
        {
            let _ = status;
            false
        }
    }

    /// Public entry point
    pub fn export(
        ffmpeg_bin: &str,
        v1_clips: &[Clip],
        audio_tracks: &[AudioTrack],
        video_tracks: &[VideoTrack],
        subtitle_tracks: &[SubtitleTrack],
        subtitle_groups: &std::collections::HashMap<u64, SubtitleGroupTransform>,
        out_path: &str,
        layout_canvas_w: f32,
        layout_canvas_h: f32,
        canvas_w: f32,
        canvas_h: f32,
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
        export_color_mode: ExportColorMode,
        export_mode: ExportMode,
        export_preset: ExportPreset,
        export_settings: ExportSettings,
        export_range: Option<ExportRange>,
        cancel_requested: Arc<AtomicBool>,
        mut on_progress: impl FnMut(ExportProgress),
    ) -> Result<(), ExportError> {
        println!("[Export] Starting export to: {}", out_path);
        println!(
            "[Export] Config: mode={} preset={} fps={} size={}x{}",
            export_mode.id(),
            export_preset.id(),
            export_settings.normalized_fps(),
            canvas_w.max(1.0).round() as u32,
            canvas_h.max(1.0).round() as u32
        );

        let temp_out_path = temp_output_path(out_path);
        let _ = fs::remove_file(&temp_out_path);

        let audio_only_export = export_preset.is_audio_only();
        let timeline_max = if audio_only_export {
            compute_audio_timeline_max(audio_tracks)
        } else {
            compute_timeline_max(v1_clips, audio_tracks, video_tracks, subtitle_tracks)
        };
        let export_range = Self::normalize_export_range(export_range, timeline_max);
        let progress_total = export_range
            .map(|r| r.duration())
            .unwrap_or(timeline_max)
            .max(Duration::from_millis(1));
        let layout_canvas_w_i = layout_canvas_w.max(1.0).round().max(1.0) as u32;
        let layout_canvas_h_i = layout_canvas_h.max(1.0).round().max(1.0) as u32;
        let canvas_w_i = canvas_w.max(1.0).round().max(1.0) as u32;
        let canvas_h_i = canvas_h.max(1.0).round().max(1.0) as u32;
        let mut used_copy_path = false;
        let mut active_export_preset = export_preset;

        if !audio_only_export
            && Self::should_try_true_gpu_single_clip_opacity_path(
                v1_clips,
                audio_tracks,
                video_tracks,
                subtitle_tracks,
                layer_effects,
                layer_effect_clips,
                active_export_preset,
            )
        {
            let clip = &v1_clips[0];
            println!(
                "[Export][GPU Effect] Phase1 true-GPU path selected: single-clip opacity shader + h264_videotoolbox."
            );
            match Self::run_true_gpu_single_clip_opacity_export(
                ffmpeg_bin,
                clip,
                &temp_out_path,
                canvas_w,
                canvas_h,
                &export_settings,
                export_range,
                progress_total,
                &cancel_requested,
                &mut on_progress,
            ) {
                Ok(()) => {
                    if cancel_requested.load(Ordering::Relaxed) {
                        let _ = fs::remove_file(&temp_out_path);
                        return Err(ExportError::Cancelled);
                    }

                    if Path::new(out_path).exists() {
                        fs::remove_file(out_path)
                            .map_err(|source| ExportError::ReplaceExistingOutput { source })?;
                    }
                    fs::rename(&temp_out_path, out_path)
                        .map_err(|source| ExportError::FinalizeOutput { source })?;

                    on_progress(ExportProgress {
                        rendered: progress_total,
                        total: progress_total,
                        speed: None,
                    });
                    println!("[Export] Success!");
                    return Ok(());
                }
                Err(ExportError::Cancelled) => {
                    let _ = fs::remove_file(&temp_out_path);
                    return Err(ExportError::Cancelled);
                }
                Err(reason) => {
                    println!(
                        "[Export][GPU Effect] Phase1 true-GPU path fallback to standard FFmpeg graph: {reason}"
                    );
                    let _ = fs::remove_file(&temp_out_path);
                }
            }
        } else if !audio_only_export
            && Self::should_try_true_gpu_multitrack_opacity_path(
                v1_clips,
                audio_tracks,
                video_tracks,
                subtitle_tracks,
                layer_effects,
                layer_effect_clips,
                active_export_preset,
            )
        {
            println!(
                "[Export][GPU Effect] Phase2 true-GPU path selected: multitrack opacity shader + h264_videotoolbox."
            );
            match Self::run_true_gpu_multitrack_opacity_export(
                ffmpeg_bin,
                v1_clips,
                video_tracks,
                &temp_out_path,
                canvas_w,
                canvas_h,
                &export_settings,
                export_range,
                progress_total,
                &cancel_requested,
                &mut on_progress,
            ) {
                Ok(()) => {
                    if cancel_requested.load(Ordering::Relaxed) {
                        let _ = fs::remove_file(&temp_out_path);
                        return Err(ExportError::Cancelled);
                    }

                    if Path::new(out_path).exists() {
                        fs::remove_file(out_path)
                            .map_err(|source| ExportError::ReplaceExistingOutput { source })?;
                    }
                    fs::rename(&temp_out_path, out_path)
                        .map_err(|source| ExportError::FinalizeOutput { source })?;

                    on_progress(ExportProgress {
                        rendered: progress_total,
                        total: progress_total,
                        speed: None,
                    });
                    println!("[Export] Success!");
                    return Ok(());
                }
                Err(ExportError::Cancelled) => {
                    let _ = fs::remove_file(&temp_out_path);
                    return Err(ExportError::Cancelled);
                }
                Err(reason) => {
                    println!(
                        "[Export][GPU Effect] Phase2 true-GPU path fallback to standard FFmpeg graph: {reason}"
                    );
                    let _ = fs::remove_file(&temp_out_path);
                }
            }
        }

        if !audio_only_export
            && Self::should_try_unified_gpu_render_path(
                v1_clips,
                audio_tracks,
                video_tracks,
                subtitle_tracks,
                layer_effects,
                layer_effect_clips,
                active_export_preset,
            )
        {
            println!(
                "[Export][Unified GPU] Path selected: FFmpeg decode/encode + shared WGPU preview effects."
            );
            let unified_subtitle_renders =
                if subtitle_tracks.iter().any(|track| !track.clips.is_empty()) {
                    Some(render_subtitle_pngs(
                        subtitle_tracks,
                        subtitle_groups,
                        canvas_w_i,
                        canvas_h_i,
                        layout_canvas_w_i,
                        layout_canvas_h_i,
                    )?)
                } else {
                    None
                };
            let unified_subtitle_overlays = unified_subtitle_renders
                .as_ref()
                .map(|renders| renders.overlays.as_slice())
                .unwrap_or(&[]);
            match Self::run_unified_gpu_render_export(
                ffmpeg_bin,
                v1_clips,
                audio_tracks,
                video_tracks,
                unified_subtitle_overlays,
                &temp_out_path,
                canvas_w,
                canvas_h,
                layer_effects,
                layer_effect_clips,
                active_export_preset,
                &export_settings,
                timeline_max,
                export_range,
                progress_total,
                &cancel_requested,
                &mut on_progress,
            ) {
                Ok(()) => {
                    if cancel_requested.load(Ordering::Relaxed) {
                        let _ = fs::remove_file(&temp_out_path);
                        return Err(ExportError::Cancelled);
                    }

                    if Path::new(out_path).exists() {
                        fs::remove_file(out_path)
                            .map_err(|source| ExportError::ReplaceExistingOutput { source })?;
                    }
                    fs::rename(&temp_out_path, out_path)
                        .map_err(|source| ExportError::FinalizeOutput { source })?;

                    on_progress(ExportProgress {
                        rendered: progress_total,
                        total: progress_total,
                        speed: None,
                    });
                    println!("[Export] Success!");
                    return Ok(());
                }
                Err(ExportError::Cancelled) => {
                    let _ = fs::remove_file(&temp_out_path);
                    return Err(ExportError::Cancelled);
                }
                Err(reason) => {
                    println!("[Export][Unified GPU] Fallback to standard FFmpeg graph: {reason}");
                    let _ = fs::remove_file(&temp_out_path);
                }
            }
        }

        // Keep subtitle render output alive until ffmpeg exits.
        // SubtitleRenderOutput drops temp PNG files on Drop.
        // Keep Source/Smart mode can use stream copy only when timeline is a single untouched source clip.
        let build_audio_only_args = |preset: ExportPreset| {
            Self::build_ffmpeg_cmd(
                &[],
                audio_tracks,
                &[],
                &[],
                &temp_out_path,
                canvas_w,
                canvas_h,
                LayerColorBlurEffects::default(),
                &[],
                timeline_max,
                export_range,
                export_color_mode,
                preset,
                &export_settings,
                false,
            )
        };
        let (mut args, subtitle_render_output): (Vec<String>, Option<SubtitleRenderOutput>) =
            match export_mode {
                ExportMode::KeepSourceCopy => {
                    if audio_only_export {
                        (build_audio_only_args(active_export_preset), None)
                    } else if active_export_preset.requires_rendered_video() {
                        return Err(ExportError::KeepSourceCopyUnavailable);
                    } else if let Some(copy_args) = Self::build_copy_ffmpeg_cmd(
                        v1_clips,
                        audio_tracks,
                        video_tracks,
                        subtitle_tracks,
                        &temp_out_path,
                        export_range,
                        layer_effects,
                        layer_effect_clips,
                    ) {
                        used_copy_path = true;
                        (copy_args, None)
                    } else {
                        return Err(ExportError::KeepSourceCopyUnavailable);
                    }
                }
                ExportMode::SmartUniversal => {
                    if audio_only_export {
                        (build_audio_only_args(active_export_preset), None)
                    } else if !active_export_preset.requires_rendered_video()
                        && let Some(copy_args) = Self::build_copy_ffmpeg_cmd(
                            v1_clips,
                            audio_tracks,
                            video_tracks,
                            subtitle_tracks,
                            &temp_out_path,
                            export_range,
                            layer_effects,
                            layer_effect_clips,
                        )
                    {
                        used_copy_path = true;
                        (copy_args, None)
                    } else {
                        match Self::build_gpu_effect_opacity_videotoolbox_cmd(
                            v1_clips,
                            audio_tracks,
                            video_tracks,
                            subtitle_tracks,
                            &temp_out_path,
                            canvas_w,
                            canvas_h,
                            layer_effects,
                            layer_effect_clips,
                            timeline_max,
                            export_range,
                            export_color_mode,
                            active_export_preset,
                            &export_settings,
                        ) {
                            Ok(gpu_effect_args) => {
                                println!(
                                    "[Export][GPU Effect] Path selected: opacity-compatible timeline + h264_videotoolbox."
                                );
                                (gpu_effect_args, None)
                            }
                            Err(reason) => {
                                println!("[Export][GPU Effect] Skip opacity path: {reason}");
                                let subtitle_renders = render_subtitle_pngs(
                                    subtitle_tracks,
                                    subtitle_groups,
                                    canvas_w_i,
                                    canvas_h_i,
                                    layout_canvas_w_i,
                                    layout_canvas_h_i,
                                )?;
                                let subtitle_overlays = subtitle_renders.overlays.as_slice();
                                let args = Self::build_ffmpeg_cmd(
                                    v1_clips,
                                    audio_tracks,
                                    video_tracks,
                                    subtitle_overlays,
                                    &temp_out_path,
                                    canvas_w,
                                    canvas_h,
                                    layer_effects,
                                    layer_effect_clips,
                                    timeline_max,
                                    export_range,
                                    export_color_mode,
                                    active_export_preset,
                                    &export_settings,
                                    false,
                                );
                                (args, Some(subtitle_renders))
                            }
                        }
                    }
                }
                ExportMode::PresetReencode => {
                    if audio_only_export {
                        (build_audio_only_args(active_export_preset), None)
                    } else {
                        match Self::build_gpu_effect_opacity_videotoolbox_cmd(
                            v1_clips,
                            audio_tracks,
                            video_tracks,
                            subtitle_tracks,
                            &temp_out_path,
                            canvas_w,
                            canvas_h,
                            layer_effects,
                            layer_effect_clips,
                            timeline_max,
                            export_range,
                            export_color_mode,
                            active_export_preset,
                            &export_settings,
                        ) {
                            Ok(gpu_effect_args) => {
                                println!(
                                    "[Export][GPU Effect] Path selected: opacity-compatible timeline + h264_videotoolbox."
                                );
                                (gpu_effect_args, None)
                            }
                            Err(reason) => {
                                println!("[Export][GPU Effect] Skip opacity path: {reason}");
                                let subtitle_renders = render_subtitle_pngs(
                                    subtitle_tracks,
                                    subtitle_groups,
                                    canvas_w_i,
                                    canvas_h_i,
                                    layout_canvas_w_i,
                                    layout_canvas_h_i,
                                )?;
                                let subtitle_overlays = subtitle_renders.overlays.as_slice();
                                let args = Self::build_ffmpeg_cmd(
                                    v1_clips,
                                    audio_tracks,
                                    video_tracks,
                                    subtitle_overlays,
                                    &temp_out_path,
                                    canvas_w,
                                    canvas_h,
                                    layer_effects,
                                    layer_effect_clips,
                                    timeline_max,
                                    export_range,
                                    export_color_mode,
                                    active_export_preset,
                                    &export_settings,
                                    false,
                                );
                                (args, Some(subtitle_renders))
                            }
                        }
                    }
                }
            };

        let out_path_arg = args.pop().ok_or(ExportError::MissingOutputPathArg)?;
        args.push("-nostats".into());
        args.push("-loglevel".into());
        args.push("error".into());
        args.push("-progress".into());
        args.push("pipe:1".into());
        args.push(out_path_arg);

        if used_copy_path {
            println!("[Export] Stream-copy path selected.");
        }
        let rebuild_render_args = |preset: ExportPreset,
                                   disable_local_mask_filters: bool|
         -> Result<Vec<String>, ExportError> {
            let subtitle_overlays = subtitle_render_output
                .as_ref()
                .map(|renders| renders.overlays.as_slice())
                .unwrap_or(&[]);
            let mut rebuilt = Self::build_ffmpeg_cmd(
                v1_clips,
                audio_tracks,
                video_tracks,
                subtitle_overlays,
                &temp_out_path,
                canvas_w,
                canvas_h,
                layer_effects,
                layer_effect_clips,
                timeline_max,
                export_range,
                export_color_mode,
                preset,
                &export_settings,
                disable_local_mask_filters,
            );
            let out_path_arg = rebuilt.pop().ok_or(ExportError::MissingOutputPathArg)?;
            rebuilt.push("-nostats".into());
            rebuilt.push("-loglevel".into());
            rebuilt.push("error".into());
            rebuilt.push("-progress".into());
            rebuilt.push("pipe:1".into());
            rebuilt.push(out_path_arg);
            Ok(rebuilt)
        };
        let mut used_safe_local_mask_fallback = false;
        let mut used_videotoolbox_fallback = false;
        let mut speed: Option<f32>;
        loop {
            println!("[Export] Executing: {} {:?}", ffmpeg_bin, args);
            let mut child = Command::new(ffmpeg_bin)
                .args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|source| ExportError::ExecuteFfmpeg { source })?;

            let stdout = child
                .stdout
                .take()
                .ok_or(ExportError::MissingProgressOutput)?;
            let stderr = child.stderr.take().ok_or(ExportError::MissingErrorOutput)?;
            let stderr_handle = std::thread::spawn(move || {
                let mut msg = String::new();
                let mut reader = BufReader::new(stderr);
                let _ = reader.read_to_string(&mut msg);
                msg
            });

            on_progress(ExportProgress {
                rendered: Duration::ZERO,
                total: progress_total,
                speed: None,
            });

            let mut rendered = Duration::ZERO;
            speed = None;
            for line in BufReader::new(stdout).lines() {
                if cancel_requested.load(Ordering::Relaxed) {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stderr_handle.join();
                    let _ = fs::remove_file(&temp_out_path);
                    return Err(ExportError::Cancelled);
                }

                let line = line.map_err(|source| ExportError::ReadFfmpegProgress { source })?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                if let Some(v) = line.strip_prefix("out_time_us=") {
                    if let Ok(us) = v.trim().parse::<u64>() {
                        rendered = Duration::from_micros(us);
                    }
                    continue;
                }

                if let Some(v) = line.strip_prefix("out_time_ms=") {
                    if let Ok(raw) = v.trim().parse::<u64>() {
                        // FFmpeg reports this field in microseconds despite the key name.
                        rendered = Duration::from_micros(raw);
                    }
                    continue;
                }

                if let Some(v) = line.strip_prefix("speed=") {
                    speed = Self::parse_speed(v);
                    continue;
                }

                if let Some(v) = line.strip_prefix("progress=") {
                    let rendered = rendered.min(progress_total);
                    on_progress(ExportProgress {
                        rendered,
                        total: progress_total,
                        speed,
                    });

                    if v.trim() == "end" {
                        break;
                    }
                }
            }

            let status = child
                .wait()
                .map_err(|source| ExportError::WaitFfmpegProcess { source })?;
            let stderr = stderr_handle.join().unwrap_or_default();

            if status.success() {
                break;
            }
            let stderr_trimmed = stderr.trim();

            if !used_videotoolbox_fallback
                && active_export_preset == ExportPreset::H264VideotoolboxMp4
                && !audio_only_export
                && !used_copy_path
            {
                if stderr_trimmed.is_empty() {
                    eprintln!(
                        "[Export] {} failed with status {} and empty stderr. Retrying with H.264 compatibility preset.",
                        platform_h264_label(),
                        status
                    );
                } else {
                    eprintln!(
                        "[Export] {} failed with status {}. Retrying with H.264 compatibility preset.",
                        platform_h264_label(),
                        status
                    );
                    eprintln!("[Export][platform-h264][stderr]\n{}", stderr_trimmed);
                }
                active_export_preset = ExportPreset::H264Mp4;
                used_videotoolbox_fallback = true;
                used_safe_local_mask_fallback = false;
                args = rebuild_render_args(active_export_preset, false)?;
                continue;
            }

            if !used_safe_local_mask_fallback
                && Self::status_is_sigsegv(&status)
                && !audio_only_export
                && !used_copy_path
            {
                eprintln!(
                    "[Export] FFmpeg crashed with SIGSEGV. Retrying with safe local-mask export fallback."
                );
                used_safe_local_mask_fallback = true;
                args = rebuild_render_args(active_export_preset, true)?;
                continue;
            }

            let _ = fs::remove_file(&temp_out_path);
            if stderr_trimmed.is_empty() {
                eprintln!(
                    "[Export] FFmpeg failed with status {} and empty stderr.",
                    status
                );
            } else {
                eprintln!("[Export] FFmpeg failed with status {}.", status);
                eprintln!("[Export][stderr]\n{}", stderr_trimmed);
            }
            return Err(ExportError::FfmpegFailed { stderr });
        }

        if cancel_requested.load(Ordering::Relaxed) {
            let _ = fs::remove_file(&temp_out_path);
            return Err(ExportError::Cancelled);
        }

        if Path::new(out_path).exists() {
            fs::remove_file(out_path)
                .map_err(|source| ExportError::ReplaceExistingOutput { source })?;
        }
        fs::rename(&temp_out_path, out_path)
            .map_err(|source| ExportError::FinalizeOutput { source })?;

        on_progress(ExportProgress {
            rendered: progress_total,
            total: progress_total,
            speed,
        });

        println!("[Export] Success!");
        Ok(())
    }

    fn normalize_export_range(
        range: Option<ExportRange>,
        timeline_max: Duration,
    ) -> Option<ExportRange> {
        let mut range = range?;
        range.start = range.start.min(timeline_max);
        range.end = range.end.min(timeline_max);
        if range.end <= range.start + Duration::from_millis(1) {
            return None;
        }
        if range.start <= Duration::from_millis(1) && range.end >= timeline_max {
            return None;
        }
        Some(range)
    }

    fn parse_speed(value: &str) -> Option<f32> {
        let value = value.trim().trim_end_matches('x');
        let speed = value.parse::<f32>().ok()?;
        if speed.is_finite() && speed > 0.0 {
            Some(speed)
        } else {
            None
        }
    }

    fn export_sharpen_compensation_multiplier(export_preset: ExportPreset) -> f64 {
        if !export_preset.is_yuv420p_8bit() {
            return 1.0;
        }
        // Hidden tuning for 8-bit yuv420p exports where codec/chroma downsampling
        // can soften high-frequency detail versus BGRA preview.
        let pct = std::env::var(EXPORT_SHARPEN_YUV420P_COMP_PCT_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<f64>().ok())
            .unwrap_or(EXPORT_SHARPEN_YUV420P_COMP_PCT_DEFAULT)
            .clamp(
                EXPORT_SHARPEN_YUV420P_COMP_PCT_MIN,
                EXPORT_SHARPEN_YUV420P_COMP_PCT_MAX,
            );
        1.0 + pct / 100.0
    }

    fn build_copy_ffmpeg_cmd(
        v1_clips: &[Clip],
        audio_tracks: &[AudioTrack],
        video_tracks: &[VideoTrack],
        subtitle_tracks: &[SubtitleTrack],
        out_path: &str,
        export_range: Option<ExportRange>,
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
    ) -> Option<Vec<String>> {
        if v1_clips.len() != 1 {
            return None;
        }
        if audio_tracks.iter().any(|t| !t.clips.is_empty()) {
            return None;
        }
        if video_tracks.iter().any(|t| !t.clips.is_empty()) {
            return None;
        }
        if subtitle_tracks.iter().any(|t| !t.clips.is_empty()) {
            return None;
        }

        let clip = &v1_clips[0];
        if clip.start > Duration::from_millis(1) {
            return None;
        }
        if !layer_effects.is_identity() && !layer_effect_clips.is_empty() {
            return None;
        }
        if !clip.video_effects.is_empty()
            || !clip.local_mask_layers.is_empty()
            || !clip.pos_x_keyframes.is_empty()
            || !clip.pos_y_keyframes.is_empty()
            || !clip.scale_keyframes.is_empty()
            || !clip.rotation_keyframes.is_empty()
            || !clip.brightness_keyframes.is_empty()
            || !clip.contrast_keyframes.is_empty()
            || !clip.saturation_keyframes.is_empty()
            || !clip.opacity_keyframes.is_empty()
            || clip.dissolve_trim_in > Duration::from_millis(1)
            || clip.dissolve_trim_out > Duration::from_millis(1)
        {
            return None;
        }

        let mut local_start = Duration::ZERO;
        let mut local_end = clip.duration;
        if let Some(range) = export_range {
            let clip_end = clip.start + clip.duration;
            if range.start < clip.start || range.end > clip_end {
                return None;
            }
            local_start = range.start.saturating_sub(clip.start);
            local_end = local_start + range.duration();
        }
        if local_end <= local_start + Duration::from_millis(1) {
            return None;
        }

        let src_start = clip.source_in + local_start;
        let trim_duration = local_end.saturating_sub(local_start);

        let mut args: Vec<String> = Vec::new();
        args.push("-y".into());
        args.push("-hide_banner".into());
        args.push("-ss".into());
        args.push(format!("{:.6}", src_start.as_secs_f64()));
        args.push("-t".into());
        args.push(format!("{:.6}", trim_duration.as_secs_f64()));
        args.push("-i".into());
        args.push(clip.file_path.clone());
        args.push("-map".into());
        args.push("0:v:0".into());
        args.push("-map".into());
        args.push("0:a?".into());
        args.push("-c".into());
        args.push("copy".into());
        args.push(out_path.to_string());
        Some(args)
    }

    fn build_gpu_effect_opacity_videotoolbox_cmd(
        v1_clips: &[Clip],
        audio_tracks: &[AudioTrack],
        video_tracks: &[VideoTrack],
        subtitle_tracks: &[SubtitleTrack],
        out_path: &str,
        canvas_w: f32,
        canvas_h: f32,
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
        timeline_max: Duration,
        export_range: Option<ExportRange>,
        export_color_mode: ExportColorMode,
        export_preset: ExportPreset,
        export_settings: &ExportSettings,
    ) -> Result<Vec<String>, ExportError> {
        if !cfg!(target_os = "macos") {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "VideoToolbox opacity fast path is macOS-only".to_string(),
            });
        }
        if export_preset != ExportPreset::H264VideotoolboxMp4 {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "preset is not h264_videotoolbox_mp4".to_string(),
            });
        }
        if subtitle_tracks.iter().any(|t| !t.clips.is_empty()) {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "subtitle tracks are present".to_string(),
            });
        }
        if !layer_effects.is_identity() || !layer_effect_clips.is_empty() {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "layer effects are active".to_string(),
            });
        }

        let mut all_visual_clips: Vec<&Clip> = Vec::with_capacity(
            v1_clips.len() + video_tracks.iter().map(|t| t.clips.len()).sum::<usize>(),
        );
        all_visual_clips.extend(v1_clips.iter());
        for track in video_tracks {
            all_visual_clips.extend(track.clips.iter());
        }
        if all_visual_clips.is_empty() {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "no visual clips found".to_string(),
            });
        }
        if all_visual_clips
            .iter()
            .any(|clip| !Self::is_opacity_only_compatible_clip(clip))
        {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "at least one clip has non-opacity active effects".to_string(),
            });
        }
        if !all_visual_clips
            .iter()
            .any(|clip| Self::clip_has_non_identity_opacity(clip))
        {
            return Err(ExportError::GpuOpacityPathUnavailable {
                reason: "all clip opacity values are identity".to_string(),
            });
        }

        // Fast path for one-clip timelines (no overlay tracks) keeps using the
        // lightweight `-vf` builder from gpu-effect-export-engine.
        if v1_clips.len() == 1
            && video_tracks.iter().all(|t| t.clips.is_empty())
            && v1_clips[0].start <= Duration::from_millis(1)
            && Self::is_default_linked_audio_layout(audio_tracks, &v1_clips[0])
        {
            let clip = &v1_clips[0];
            let mut motionloom_plan_cache: std::collections::HashMap<
                u64,
                Option<LayerScriptExportPlan>,
            > = std::collections::HashMap::new();
            let layer_opacity_factor_upper_t = Self::build_layer_opacity_factor_expr_for_clip(
                clip,
                layer_effect_clips,
                &mut motionloom_plan_cache,
                "T",
            );
            let opacity_filter_suffix = Self::build_opacity_filter(
                clip,
                OpacityMode::MultiplyRgb,
                false,
                layer_opacity_factor_upper_t.as_deref(),
            );
            if opacity_filter_suffix.is_empty() {
                return Err(ExportError::GpuOpacityPathUnavailable {
                    reason: "opacity is identity (after keyframe/layer-factor evaluation)"
                        .to_string(),
                });
            }
            let opacity = clip.get_opacity().clamp(0.0, 1.0);

            let mut local_start = Duration::ZERO;
            let mut local_end = clip.duration;
            if let Some(range) = export_range {
                let clip_end = clip.start + clip.duration;
                if range.start < clip.start || range.end > clip_end {
                    return Err(ExportError::GpuOpacityPathUnavailable {
                        reason: "export range falls outside clip".to_string(),
                    });
                }
                local_start = range.start.saturating_sub(clip.start);
                local_end = local_start + range.duration();
            }
            if local_end <= local_start + Duration::from_millis(1) {
                return Err(ExportError::GpuOpacityPathUnavailable {
                    reason: "effective duration is too short".to_string(),
                });
            }

            let source_start = clip.source_in + local_start;
            let duration = local_end.saturating_sub(local_start);
            let request = SingleClipOpacityVideoToolboxRequest {
                source_start,
                duration,
                fps: export_settings.normalized_fps(),
                canvas_width: canvas_w.max(1.0).round() as u32,
                canvas_height: canvas_h.max(1.0).round() as u32,
                opacity,
                opacity_filter_suffix: Some(opacity_filter_suffix),
                audio_bitrate_kbps: export_settings.normalized_audio_bitrate_kbps(),
            };
            return match build_single_clip_opacity_videotoolbox_args(
                &clip.file_path,
                request,
                out_path,
            ) {
                Ok(args) => Ok(args),
                Err(err) => Err(ExportError::GpuOpacityPathUnavailable {
                    reason: format!("builder: {}", err.as_str()),
                }),
            };
        }

        // Multi-track opacity-only route: reuse existing compositor graph builder so
        // all V tracks can run through the same VideoToolbox export preset.
        Ok(Self::build_ffmpeg_cmd(
            v1_clips,
            audio_tracks,
            video_tracks,
            &[],
            out_path,
            canvas_w,
            canvas_h,
            layer_effects,
            layer_effect_clips,
            timeline_max,
            export_range,
            export_color_mode,
            export_preset,
            export_settings,
            false,
        ))
    }

    /// Internal logic to build the FFmpeg command
    fn build_ffmpeg_cmd(
        v1_clips: &[Clip],
        audio_tracks: &[AudioTrack],
        video_tracks: &[VideoTrack],
        subtitle_overlays: &[RenderedSubtitle],
        out_path: &str,
        canvas_w: f32,
        canvas_h: f32,
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
        timeline_max: Duration,
        export_range: Option<ExportRange>,
        export_color_mode: ExportColorMode,
        export_preset: ExportPreset,
        export_settings: &ExportSettings,
        disable_local_mask_filters: bool,
        // This function always succeeds — it only assembles FFmpeg arguments
        // from already-validated inputs, so the return type is plain Vec<String>.
    ) -> Vec<String> {
        let mut args: Vec<String> = Vec::new();
        args.push("-y".into());
        args.push("-hide_banner".into());
        let fps = export_settings.normalized_fps();
        let sharpen_compensation_multiplier =
            Self::export_sharpen_compensation_multiplier(export_preset);

        let mut current_input_idx = 0;
        let mut input_cache: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut add_input = |path: &str, image_duration: Option<f64>| -> usize {
            let key = if let Some(dur) = image_duration {
                format!("img|{}|fps={}|dur={:.6}", path, fps, dur)
            } else {
                format!("media|{}", path)
            };
            if let Some(&idx) = input_cache.get(&key) {
                return idx;
            }
            if let Some(dur) = image_duration {
                args.push("-loop".into());
                args.push("1".into());
                args.push("-framerate".into());
                args.push(fps.to_string());
                args.push("-t".into());
                args.push(dur.to_string());
            }
            args.push("-i".into());
            args.push(path.to_string());
            let idx = current_input_idx;
            current_input_idx += 1;
            input_cache.insert(key, idx);
            idx
        };
        let canvas_w = canvas_w.max(1.0);
        let canvas_h = canvas_h.max(1.0);
        let canvas_w_i = canvas_w.round().max(1.0) as u32;
        let canvas_h_i = canvas_h.round().max(1.0) as u32;
        let mut filter_parts: Vec<String> = Vec::new();
        let mut motionloom_plan_cache: std::collections::HashMap<
            u64,
            Option<LayerScriptExportPlan>,
        > = std::collections::HashMap::new();

        // ---------------------------------------------------------
        // 1. INPUT PROCESSING
        // ---------------------------------------------------------

        // A. V1 Inputs
        let mut v1_indices = Vec::new();
        for c in v1_clips {
            let image_duration = if is_image_ext(&c.file_path) {
                Some(c.duration.as_secs_f64())
            } else {
                None
            };
            v1_indices.push(add_input(&c.file_path, image_duration));
        }

        // B. Video Overlay Inputs
        let mut video_tracks_indices: Vec<Vec<usize>> = Vec::new();
        for track in video_tracks {
            let mut indices = Vec::new();
            for c in &track.clips {
                let image_duration = if is_image_ext(&c.file_path) {
                    Some(c.duration.as_secs_f64())
                } else {
                    None
                };
                indices.push(add_input(&c.file_path, image_duration));
            }
            video_tracks_indices.push(indices);
        }

        // C. Audio Inputs
        let mut audio_tracks_indices: Vec<Vec<usize>> = Vec::new();
        for track in audio_tracks {
            let mut indices = Vec::new();
            for c in &track.clips {
                indices.push(add_input(&c.file_path, None));
            }
            audio_tracks_indices.push(indices);
        }

        // D. Subtitle Overlays (PNG full-canvas)
        let mut subtitle_indices = Vec::new();
        if !subtitle_overlays.is_empty() {
            let timeline_secs = timeline_max.as_secs_f64();
            for overlay in subtitle_overlays {
                args.push("-loop".into());
                args.push("1".into());
                args.push("-framerate".into());
                args.push(fps.to_string());
                args.push("-t".into());
                args.push(timeline_secs.to_string());
                args.push("-i".into());
                args.push(overlay.path.to_string_lossy().to_string());
                subtitle_indices.push(current_input_idx);
                current_input_idx += 1;
            }
        }

        // ---------------------------------------------------------
        // 2. FILTER COMPLEX: VIDEO COMPOSITING (BGRA PIPELINE)
        // ---------------------------------------------------------

        let mut last_video_tag = "[v_base]".to_string();

        // --- Step 2a: Build V1 (The Base) ---
        if !v1_clips.is_empty() {
            let label = "v1";
            let mut concat_str = String::new();
            let mut count = 0;
            let mut cursor = Duration::ZERO;

            for (local_i, &global_i) in v1_indices.iter().enumerate() {
                let c = &v1_clips[local_i];

                // Gap Handling
                if c.start > cursor {
                    let gap = (c.start - cursor).as_secs_f64();
                    // ⚠️ Use BGRA for consistency
                    filter_parts.push(format!(
                        "color=c=black:s={}x{}:r={}:d={},format=bgra[{}_gap_{}]",
                        canvas_w_i, canvas_h_i, fps, gap, label, local_i
                    ));
                    concat_str.push_str(&format!("[{}_gap_{}]", label, local_i));
                    count += 1;
                }

                // Process Clip (Scale, Pos, Effects, Tint)
                let clip_out_tag = Self::build_single_clip_video_filter(
                    &mut filter_parts,
                    c,
                    global_i,
                    label,
                    local_i,
                    canvas_w,
                    canvas_h,
                    layer_effects,
                    layer_effect_clips,
                    export_color_mode,
                    OpacityMode::MultiplyRgb,
                    false,
                    &mut motionloom_plan_cache,
                    fps,
                    sharpen_compensation_multiplier,
                    disable_local_mask_filters,
                );

                concat_str.push_str(&clip_out_tag);
                count += 1;
                cursor = c.start + c.duration;
            }

            // Trailing Gap
            if cursor < timeline_max {
                let gap = (timeline_max - cursor).as_secs_f64();
                filter_parts.push(format!(
                    "color=c=black:s={}x{}:r={}:d={},format=bgra[{}_gap_end]",
                    canvas_w_i, canvas_h_i, fps, gap, label
                ));
                concat_str.push_str(&format!("[{}_gap_end]", label));
                count += 1;
            }

            // Concat V1
            filter_parts.push(format!(
                "{}concat=n={}:v=1:a=0,format=bgra[v_base]",
                concat_str, count
            ));
            last_video_tag = "[v_base]".to_string();

            // --- Optional: V1 Dissolve Overlays (virtual overlap) ---
            if v1_clips.len() >= 2 {
                let mut v1_tag = last_video_tag.clone();
                for idx in 0..(v1_clips.len() - 1) {
                    let left = &v1_clips[idx];
                    let right = &v1_clips[idx + 1];
                    let mut d = left.get_dissolve_out().min(right.get_dissolve_in()) as f64;
                    if d <= 0.001 {
                        continue;
                    }
                    let left_post = left
                        .media_duration
                        .saturating_sub(left.source_in + left.duration)
                        .as_secs_f64();
                    d = d.min(left_post);
                    d = d.min(left.duration.as_secs_f64());
                    d = d.min(right.duration.as_secs_f64());
                    if d <= 0.001 {
                        continue;
                    }

                    let half = d * 0.5;
                    let left_src = left.source_in
                        + left.duration.saturating_sub(Duration::from_secs_f64(half));
                    let right_src = right
                        .source_in
                        .saturating_sub(Duration::from_secs_f64(half));
                    let segment_timeline_start =
                        Duration::from_secs_f64((right.start.as_secs_f64() - half).max(0.0));
                    let left_seg = Self::clip_segment_for_export(
                        left,
                        left_src,
                        segment_timeline_start,
                        d,
                        true,
                    );
                    let right_seg = Self::clip_segment_for_export(
                        right,
                        right_src,
                        segment_timeline_start,
                        d,
                        false,
                    );

                    let left_tag = Self::build_single_clip_video_filter(
                        &mut filter_parts,
                        &left_seg,
                        v1_indices[idx],
                        "v1_dissolve_l",
                        idx,
                        canvas_w,
                        canvas_h,
                        layer_effects,
                        layer_effect_clips,
                        export_color_mode,
                        OpacityMode::MultiplyRgb,
                        false,
                        &mut motionloom_plan_cache,
                        fps,
                        sharpen_compensation_multiplier,
                        disable_local_mask_filters,
                    );
                    let right_tag = Self::build_single_clip_video_filter(
                        &mut filter_parts,
                        &right_seg,
                        v1_indices[idx + 1],
                        "v1_dissolve_r",
                        idx,
                        canvas_w,
                        canvas_h,
                        layer_effects,
                        layer_effect_clips,
                        export_color_mode,
                        OpacityMode::MultiplyRgb,
                        false,
                        &mut motionloom_plan_cache,
                        fps,
                        sharpen_compensation_multiplier,
                        disable_local_mask_filters,
                    );

                    let blend_tag = format!("[v1_dissolve_blend_{}]", idx);
                    filter_parts.push(format!(
                        "{}{}blend=all_expr='A*(1-clip(T/{d:.6},0,1))+B*clip(T/{d:.6},0,1)':shortest=1,format=bgra{}",
                        left_tag, right_tag, blend_tag, d = d
                    ));

                    let start = segment_timeline_start.as_secs_f64();
                    let end = start + d;
                    let blend_timed = format!("[v1_dissolve_timed_{}]", idx);
                    filter_parts.push(format!(
                        "{}setpts=PTS-STARTPTS+{:.6}/TB{}",
                        blend_tag, start, blend_timed
                    ));
                    let next_tag = format!("[v1_dissolve_comp_{}]", idx);
                    filter_parts.push(format!(
                        "{}{}overlay=0:0:format=auto:eof_action=pass:enable='between(t,{:.6},{:.6})'{}",
                        v1_tag, blend_timed, start, end, next_tag
                    ));
                    v1_tag = next_tag;
                }
                last_video_tag = v1_tag;
            }
        } else {
            // No V1 clips, black background
            filter_parts.push(format!(
                "color=c=black:s={}x{}:r={}:d={},format=bgra[v_base]",
                canvas_w_i,
                canvas_h_i,
                fps,
                timeline_max.as_secs_f64()
            ));
        }

        // --- Step 2b: Overlay Video Tracks ---
        for (t_idx, track) in video_tracks.iter().enumerate() {
            if track.clips.is_empty() {
                continue;
            }

            let indices = &video_tracks_indices[t_idx];
            let label = format!("overlay_v{}", t_idx);
            let mut concat_str = String::new();
            let mut count = 0;
            let mut cursor = Duration::ZERO;
            let mut export_starts: Vec<Duration> = Vec::with_capacity(track.clips.len());

            for (local_i, &global_i) in indices.iter().enumerate() {
                let c = &track.clips[local_i];

                // Gap (Must be transparent black)
                let export_start = if c.start > cursor { c.start } else { cursor };
                if c.start > cursor {
                    let gap = (c.start - cursor).as_secs_f64();
                    filter_parts.push(format!(
                        "color=c=black@0.0:s={}x{}:r={}:d={},format=bgra[{}_gap_{}]",
                        canvas_w_i, canvas_h_i, fps, gap, label, local_i
                    ));
                    concat_str.push_str(&format!("[{}_gap_{}]", label, local_i));
                    count += 1;
                }
                export_starts.push(export_start);

                // Process Clip
                let clip_out_tag = Self::build_single_clip_video_filter(
                    &mut filter_parts,
                    c,
                    global_i,
                    &label,
                    local_i,
                    canvas_w,
                    canvas_h,
                    layer_effects,
                    layer_effect_clips,
                    export_color_mode,
                    OpacityMode::AlphaOnly,
                    false,
                    &mut motionloom_plan_cache,
                    fps,
                    sharpen_compensation_multiplier,
                    disable_local_mask_filters,
                );

                concat_str.push_str(&clip_out_tag);
                count += 1;
                cursor = export_start + c.duration;
            }

            // Trailing Gap
            if cursor < timeline_max {
                let gap = (timeline_max - cursor).as_secs_f64();
                filter_parts.push(format!(
                    "color=c=black@0.0:s={}x{}:r={}:d={},format=bgra[{}_gap_end]",
                    canvas_w_i, canvas_h_i, fps, gap, label
                ));
                concat_str.push_str(&format!("[{}_gap_end]", label));
                count += 1;
            }

            // Concat Track
            let layer_tag = format!("[v_layer_{}]", t_idx);
            filter_parts.push(format!(
                "{}concat=n={}:v=1:a=0,format=bgra{}",
                concat_str, count, layer_tag
            ));

            // --- Optional: Overlay Track Dissolves (virtual overlap) ---
            let mut track_tag = layer_tag.clone();
            if track.clips.len() >= 2 {
                for idx in 0..(track.clips.len() - 1) {
                    let left = &track.clips[idx];
                    let right = &track.clips[idx + 1];
                    let mut d = left.get_dissolve_out().min(right.get_dissolve_in()) as f64;
                    if d <= 0.001 {
                        continue;
                    }
                    d = d.min(left.duration.as_secs_f64());
                    d = d.min(right.duration.as_secs_f64());
                    if d <= 0.001 {
                        continue;
                    }

                    let half = d * 0.5;
                    let left_src = left.source_in
                        + left.duration.saturating_sub(Duration::from_secs_f64(half));
                    let right_src = right
                        .source_in
                        .saturating_sub(Duration::from_secs_f64(half));
                    let Some(center_time) = export_starts.get(idx + 1).copied() else {
                        continue;
                    };
                    let center = center_time.as_secs_f64();
                    let segment_timeline_start = Duration::from_secs_f64((center - half).max(0.0));
                    let left_seg = Self::clip_segment_for_export(
                        left,
                        left_src,
                        segment_timeline_start,
                        d,
                        true,
                    );
                    let right_seg = Self::clip_segment_for_export(
                        right,
                        right_src,
                        segment_timeline_start,
                        d,
                        false,
                    );

                    let left_tag = Self::build_single_clip_video_filter(
                        &mut filter_parts,
                        &left_seg,
                        indices[idx],
                        &format!("ov{}_dissolve_l", t_idx),
                        idx,
                        canvas_w,
                        canvas_h,
                        layer_effects,
                        layer_effect_clips,
                        export_color_mode,
                        OpacityMode::AlphaOnly,
                        false,
                        &mut motionloom_plan_cache,
                        fps,
                        sharpen_compensation_multiplier,
                        disable_local_mask_filters,
                    );
                    let right_tag = Self::build_single_clip_video_filter(
                        &mut filter_parts,
                        &right_seg,
                        indices[idx + 1],
                        &format!("ov{}_dissolve_r", t_idx),
                        idx,
                        canvas_w,
                        canvas_h,
                        layer_effects,
                        layer_effect_clips,
                        export_color_mode,
                        OpacityMode::AlphaOnly,
                        false,
                        &mut motionloom_plan_cache,
                        fps,
                        sharpen_compensation_multiplier,
                        disable_local_mask_filters,
                    );

                    let blend_tag = format!("[ov{}_dissolve_blend_{}]", t_idx, idx);
                    filter_parts.push(format!(
                        "{}{}blend=all_expr='A*(1-clip(T/{d:.6},0,1))+B*clip(T/{d:.6},0,1)':shortest=1,format=bgra{}",
                        left_tag, right_tag, blend_tag, d = d
                    ));

                    let start = segment_timeline_start.as_secs_f64();
                    let end = start + d;
                    let blend_timed = format!("[ov{}_dissolve_timed_{}]", t_idx, idx);
                    filter_parts.push(format!(
                        "{}setpts=PTS-STARTPTS+{:.6}/TB{}",
                        blend_tag, start, blend_timed
                    ));
                    let next_tag = format!("[ov{}_dissolve_comp_{}]", t_idx, idx);
                    filter_parts.push(format!(
                        "{}{}overlay=0:0:format=auto:eof_action=pass:enable='between(t,{:.6},{:.6})'{}",
                        track_tag, blend_timed, start, end, next_tag
                    ));
                    track_tag = next_tag;
                }
            }

            // Overlay onto Base
            let next_tag = format!("[v_comp_{}]", t_idx);
            // format=auto is crucial for correct alpha blending
            filter_parts.push(format!(
                "{}{}overlay=0:0:format=auto{}",
                last_video_tag, track_tag, next_tag
            ));

            last_video_tag = next_tag;
        }

        // ---------------------------------------------------------
        // 3. SUBTITLES (pre-rendered PNG overlays)
        // ---------------------------------------------------------
        if !subtitle_overlays.is_empty() {
            for (idx, overlay) in subtitle_overlays.iter().enumerate() {
                let Some(&input_idx) = subtitle_indices.get(idx) else {
                    continue;
                };
                let layer_tag = format!("[v_sub_layer_{}]", idx);
                filter_parts.push(format!("[{}:v]format=bgra{}", input_idx, layer_tag));
                let next_tag = format!("[v_sub_comp_{}]", idx);
                filter_parts.push(format!(
                    "{}{}overlay=0:0:format=auto:enable='between(t,{:.3},{:.3})'{}",
                    last_video_tag, layer_tag, overlay.start, overlay.end, next_tag
                ));
                last_video_tag = next_tag;
            }
        }

        // ---------------------------------------------------------
        // 4. AUDIO MIXING
        // ---------------------------------------------------------

        let mut audio_mix_inputs = Vec::new();
        let mut audio_presence_cache: std::collections::HashMap<String, bool> =
            std::collections::HashMap::new();

        // V1 audio is exported from explicit audio lanes only (A1+), never from embedded V1 sources.
        // This prevents duplicate audio and allows users to mute/remove sound by editing audio clips.

        // Video overlay lanes (V2+) never emit audio in export.
        // Audio output is sourced strictly from explicit audio tracks.

        // Audio Tracks
        for (t_idx, track) in audio_tracks.iter().enumerate() {
            if track.clips.is_empty() {
                continue;
            }
            let indices = &audio_tracks_indices[t_idx];
            let label = format!("audio_{}", t_idx);
            let mut concat_str = String::new();
            let mut count = 0;
            let mut cursor = Duration::ZERO;

            for (local_i, &global_i) in indices.iter().enumerate() {
                let c = &track.clips[local_i];
                if c.start > cursor {
                    let gap = (c.start - cursor).as_secs_f64();
                    filter_parts.push(format!(
                        "anullsrc=channel_layout=stereo:sample_rate=48000:d={}[{}_gap_{}]",
                        gap, label, local_i
                    ));
                    concat_str.push_str(&format!("[{}_gap_{}]", label, local_i));
                    count += 1;
                }
                let start = c.source_in.as_secs_f64();
                let end = (c.source_in + c.duration).as_secs_f64();
                let clip_tag = format!("[{}_clip_{}]", label, local_i);
                let has_audio = *audio_presence_cache
                    .entry(c.file_path.clone())
                    .or_insert_with(|| has_audio_stream(&c.file_path));
                if has_audio {
                    let clip_raw_tag = format!("[{}_clip_raw_{}]", label, local_i);
                    filter_parts.push(format!(
                        "[{}:a:0]aresample=48000,atrim=start={:.6}:end={:.6},asetpts=PTS-STARTPTS{}",
                        global_i, start, end, clip_raw_tag
                    ));
                    let clip_gain_linear = 10.0_f64.powf((c.audio_gain_db as f64) / 20.0);
                    if (clip_gain_linear - 1.0).abs() > 0.0005 {
                        filter_parts.push(format!(
                            "{}volume={:.6}{}",
                            clip_raw_tag, clip_gain_linear, clip_tag
                        ));
                    } else {
                        filter_parts.push(format!("{}anull{}", clip_raw_tag, clip_tag));
                    }
                } else {
                    // Keep timeline sync for clips with no audio stream (e.g. images) by injecting silence.
                    let silence_dur = c.duration.as_secs_f64().max(0.001);
                    filter_parts.push(format!(
                        "anullsrc=channel_layout=stereo:sample_rate=48000:d={:.6}{}",
                        silence_dur, clip_tag
                    ));
                }
                concat_str.push_str(&clip_tag);
                count += 1;
                cursor = c.start + c.duration;
            }
            let out_tag = format!("[a_track_{}]", t_idx);
            filter_parts.push(format!(
                "{}concat=n={}:v=0:a=1{}",
                concat_str, count, out_tag
            ));
            let gain_linear = 10.0_f64.powf((track.gain_db as f64) / 20.0);
            if (gain_linear - 1.0).abs() > 0.0005 {
                let gain_tag = format!("[a_track_gain_{}]", t_idx);
                filter_parts.push(format!("{}volume={:.6}{}", out_tag, gain_linear, gain_tag));
                audio_mix_inputs.push(gain_tag);
            } else {
                audio_mix_inputs.push(out_tag);
            }
        }

        // Final Mix
        if audio_mix_inputs.is_empty() {
            let silence_dur = timeline_max.as_secs_f64().max(0.001);
            filter_parts.push(format!(
                "anullsrc=channel_layout=stereo:sample_rate=48000:d={:.6}[aout]",
                silence_dur
            ));
        } else if audio_mix_inputs.len() == 1 {
            filter_parts.push(format!("{}volume=1.0[aout]", audio_mix_inputs[0]));
        } else {
            let inputs_str = audio_mix_inputs.join("");
            filter_parts.push(format!(
                "{}amix=inputs={}:duration=longest:dropout_transition=0[aout]",
                inputs_str,
                audio_mix_inputs.len()
            ));
        }

        // ---------------------------------------------------------
        // 5. OPTIONAL EXPORT RANGE TRIM
        // ---------------------------------------------------------
        let mut final_video_tag = last_video_tag;
        let mut final_audio_tag = "[aout]".to_string();
        if let Some(range) = export_range {
            let start = range.start.as_secs_f64();
            let end = range.end.as_secs_f64();
            let trimmed_video_tag = "[v_export_trim]".to_string();
            let trimmed_audio_tag = "[a_export_trim]".to_string();
            filter_parts.push(format!(
                "{}trim=start={:.6}:end={:.6},setpts=PTS-STARTPTS{}",
                final_video_tag, start, end, trimmed_video_tag
            ));
            filter_parts.push(format!(
                "{}atrim=start={:.6}:end={:.6},asetpts=PTS-STARTPTS{}",
                final_audio_tag, start, end, trimmed_audio_tag
            ));
            final_video_tag = trimmed_video_tag;
            final_audio_tag = trimmed_audio_tag;
        }

        // ---------------------------------------------------------
        // 6. FINAL ASSEMBLY
        // ---------------------------------------------------------
        if export_preset == ExportPreset::Gif {
            let gif_fps = fps;
            let gif_tag = "[v_gif]".to_string();
            filter_parts.push(format!(
                "{final_video_tag}fps={gif_fps},split[v_gif_palette_src][v_gif_use_src];\
                 [v_gif_palette_src]palettegen=stats_mode=diff[p_gif];\
                 [v_gif_use_src][p_gif]paletteuse=dither=sierra2_4a{gif_tag}"
            ));
            final_video_tag = gif_tag;
        }

        if export_preset.is_audio_only() {
            // Keep video branch connected for ffmpeg graph validation while emitting audio-only output.
            filter_parts.push(format!("{final_video_tag}nullsink"));
        } else if export_preset.omits_audio() {
            // GIF has no audio stream; consume the generated audio branch so filter_complex remains valid.
            filter_parts.push(format!("{final_audio_tag}anullsink"));
        }

        let full_filter = filter_parts.join(";");

        args.push("-filter_complex".into());
        args.push(full_filter);
        if !export_preset.is_audio_only() {
            args.push("-map".into());
            args.push(final_video_tag);
        }
        if !export_preset.omits_audio() {
            args.push("-map".into());
            args.push(final_audio_tag);
        }

        // Output encoding preset
        export_preset.push_output_args(&mut args, export_settings, fps);

        args.push(out_path.to_string());

        args
    }

    /// [Helper] Process a single clip (BGRA Pipeline + Effects + Tint Overlay + Transform)
    /// [Helper] Process a single clip (BGRA pipeline + effects + tint overlay + transform)
    fn source_trim_range_for_path(
        file_path: &str,
        source_in: Duration,
        duration: Duration,
    ) -> (f64, f64) {
        // A still image has no temporal source to seek. Split image clips may
        // carry a non-zero source_in for timeline bookkeeping, but applying it
        // to FFmpeg trim produces an empty stream.
        if is_image_ext(file_path) {
            return (0.0, duration.as_secs_f64());
        }
        (
            source_in.as_secs_f64(),
            (source_in + duration).as_secs_f64(),
        )
    }

    fn clip_canvas_color(opacity_mode: OpacityMode) -> &'static str {
        match opacity_mode {
            OpacityMode::AlphaOnly => "black@0.0",
            OpacityMode::MultiplyRgb => "black",
        }
    }

    fn build_single_clip_video_filter(
        filter_parts: &mut Vec<String>,
        c: &Clip,
        global_i: usize,
        label_prefix: &str,
        local_i: usize,
        canvas_w: f32,
        canvas_h: f32,
        layer_effects: LayerColorBlurEffects,
        layer_effect_clips: &[LayerEffectClip],
        export_color_mode: ExportColorMode,
        opacity_mode: OpacityMode,
        include_dissolve: bool,
        motionloom_plan_cache: &mut std::collections::HashMap<u64, Option<LayerScriptExportPlan>>,
        output_fps: u32,
        sharpen_compensation_multiplier: f64,
        disable_local_mask_filters: bool,
    ) -> String {
        let (start, end) = Self::source_trim_range_for_path(&c.file_path, c.source_in, c.duration);
        let scale = c.get_scale();
        let pos_x = c.get_pos_x();
        let pos_y = c.get_pos_y();
        let rotation = c.get_rotation();
        let scale_expr = Self::build_keyframe_expr(&c.scale_keyframes, scale, "t");
        let rotation_expr = Self::build_keyframe_expr(&c.rotation_keyframes, rotation, "t");
        let zoom_expr = Self::build_zoom_expr(c, "t");
        let shock_zoom_expr = Self::build_shock_zoom_expr(c, "t");
        let scale_expr = format!("({})*({})*({})", scale_expr, zoom_expr, shock_zoom_expr);
        let pos_x_expr = Self::build_keyframe_expr(&c.pos_x_keyframes, pos_x, "t");
        let pos_y_expr = Self::build_keyframe_expr(&c.pos_y_keyframes, pos_y, "t");
        let (slide_x_expr, slide_y_expr) = Self::build_slide_expr(c, "t");
        let pos_x_expr = format!("({})+({})", pos_x_expr, slide_x_expr);
        let pos_y_expr = format!("({})+({})", pos_y_expr, slide_y_expr);

        // 1. Build Effects Filters
        let layer_effect_gate_t = if layer_effects.is_identity() {
            None
        } else {
            Self::build_layer_gate_expr_for_clip(c, layer_effect_clips, "t", false)
        };
        let layer_effect_gate_upper_t = if layer_effects.is_identity() {
            None
        } else {
            Self::build_layer_gate_expr_for_clip(c, layer_effect_clips, "T", false)
        };
        let layer_opacity_factor_upper_t = Self::build_layer_opacity_factor_expr_for_clip(
            c,
            layer_effect_clips,
            motionloom_plan_cache,
            "T",
        );
        let eq_filter = Self::build_eq_filter(
            c,
            layer_effects,
            layer_effect_gate_t.as_deref(),
            layer_effect_gate_upper_t.as_deref(),
            export_color_mode,
        );
        let blur_filter = format!(
            "{}{}",
            Self::build_blur_filter(c, layer_effects, layer_effect_gate_t.as_deref()),
            Self::build_motionloom_blur_filter_for_clip(
                c,
                layer_effect_clips,
                motionloom_plan_cache,
                "t",
            )
        );
        let motionloom_sharpen_filter = Self::build_motionloom_sharpen_filter_for_clip(
            c,
            layer_effect_clips,
            motionloom_plan_cache,
            "t",
            sharpen_compensation_multiplier,
        );
        let motionloom_lut_filter = Self::build_motionloom_lut_filter_for_clip(
            c,
            layer_effect_clips,
            motionloom_plan_cache,
            "T",
        );
        let tint_filter = Self::build_tint_filter(c);
        let motionloom_hsla_filter = Self::build_motionloom_hsla_filter_for_clip(
            c,
            layer_effect_clips,
            motionloom_plan_cache,
            "T",
        );
        let motionloom_bloom_filter = Self::build_motionloom_bloom_filter_for_clip(
            c,
            layer_effect_clips,
            motionloom_plan_cache,
            "T",
            label_prefix,
            local_i,
        );
        let opacity_filter = Self::build_opacity_filter(
            c,
            opacity_mode,
            include_dissolve,
            layer_opacity_factor_upper_t.as_deref(),
        );
        let rotation_filter = if c.rotation_keyframes.is_empty() && rotation.abs() <= 0.0001 {
            String::new()
        } else {
            format!(
                ",rotate='({})*PI/180':ow='rotw(iw)':oh='roth(ih)':c=black",
                rotation_expr
            )
        };
        let pre_opacity_format = if opacity_filter.is_empty() {
            ""
        } else {
            ",format=bgra"
        };

        // 2. Create Transparent Canvas (BGRA)
        let canvas_tag = format!("[{}_canvas_{}]", label_prefix, local_i);
        let canvas_w_i = canvas_w.round().max(1.0) as u32;
        let canvas_h_i = canvas_h.round().max(1.0) as u32;
        let canvas_color = Self::clip_canvas_color(opacity_mode);
        filter_parts.push(format!(
            "color=c={}:s={}x{}:r={}:d={},format=bgra{}",
            canvas_color,
            canvas_w_i,
            canvas_h_i,
            output_fps,
            c.duration.as_secs_f64(),
            canvas_tag
        ));

        // 3. Process Content
        let base_content_tag = format!("[{}_content_base_{}]", label_prefix, local_i);

        // ⚠️ [CRITICAL FIX 1]: Removed `*1.25`
        // To keep export and preview fully consistent, both must use the same scaling logic.
        // Since the GPUI preview uses the standard scale, export must also use the standard scale here.
        // If the image feels too small, adjust the Scale slider in the Inspector instead of hardcoding *1.25 here.
        let scale_w_expr = format!("({:.3})*({})", canvas_w, scale_expr);
        let scale_h_expr = format!("({:.3})*({})", canvas_h, scale_expr);
        filter_parts.push(format!(
            "[{}:v:0]fps={},setsar=1,trim=start={:.6}:end={:.6},setpts=PTS-STARTPTS,format=bgra{},scale=w='{}':h='{}':force_original_aspect_ratio=decrease:eval=frame{}{}{}{}{}{}{}{}{}{}",
            global_i,
            output_fps,
            start,
            end,
            eq_filter,
            scale_w_expr,
            scale_h_expr,
            rotation_filter,
            blur_filter,
            motionloom_sharpen_filter,
            motionloom_lut_filter,
            tint_filter,
            motionloom_hsla_filter,
            if motionloom_bloom_filter.is_empty() {
                String::new()
            } else {
                ",format=bgra".to_string()
            },
            pre_opacity_format,
            opacity_filter,
            base_content_tag
        ));
        let base_content_tag = if motionloom_bloom_filter.is_empty() {
            base_content_tag.clone()
        } else {
            let bloom_tag = format!("[{}_bloom_base_{}]", label_prefix, local_i);
            filter_parts.push(format!(
                "{}{}{}",
                base_content_tag, motionloom_bloom_filter, bloom_tag
            ));
            bloom_tag
        };
        let content_tag = if disable_local_mask_filters {
            base_content_tag.clone()
        } else {
            Self::build_local_mask_filters(
                filter_parts,
                c,
                &base_content_tag,
                label_prefix,
                local_i,
            )
        };

        // 4. Position on Canvas (Overlay)
        // ⚠️ [CRITICAL FIX 2]: Use an absolute coordinate system (unified coordinate system)
        // This matches the new VideoPreview (GPUI) logic exactly:
        // Preview: CanvasCenter + (pos * 1920)
        // Export:  CanvasCenter + (pos * 1920)

        // X Position
        // (W-w)/2 centers the content, and pos_x * 1920 is the offset
        let x_expr = format!("'(W-w)/2 + ({}) * {:.3}'", pos_x_expr, canvas_w);

        // Y Position
        // (H-h)/2 centers the content, and pos_y * 1080 is the offset
        let y_expr = format!("'(H-h)/2 + ({}) * {:.3}'", pos_y_expr, canvas_h);

        let clip_out_tag = format!("[{}_clip_{}]", label_prefix, local_i);

        filter_parts.push(format!(
            "{}{}overlay=x={}:y={}:shortest=1:format=auto{}",
            canvas_tag, content_tag, x_expr, y_expr, clip_out_tag
        ));

        clip_out_tag
    }

    fn local_mask_layer_has_shape(layer: &LocalMaskLayer) -> bool {
        layer.enabled
            && layer.strength >= 0.001
            && layer.radius >= 0.0001
            && (layer.feather >= 0.0001 || layer.radius > 0.0001)
    }

    fn local_mask_layer_has_adjustment(layer: &LocalMaskLayer) -> bool {
        layer.brightness.abs() >= 0.001
            || (layer.contrast - 1.0).abs() >= 0.001
            || (layer.saturation - 1.0).abs() >= 0.001
            || (layer.opacity - 1.0).abs() >= 0.001
            || layer.blur_sigma >= 0.001
    }

    fn build_local_mask_eq_filter(layer: &LocalMaskLayer) -> String {
        let b = layer.brightness.clamp(-1.0, 1.0);
        let c = layer.contrast.clamp(0.0, 2.0);
        let s = layer.saturation.clamp(0.0, 2.0);
        if b.abs() <= 0.001 && (c - 1.0).abs() <= 0.001 && (s - 1.0).abs() <= 0.001 {
            return String::new();
        }
        format!(
            ",eq=brightness='{:.6}':contrast='{:.6}':saturation='{:.6}'",
            b, c, s
        )
    }

    fn build_local_mask_blur_filter(layer: &LocalMaskLayer) -> String {
        let sigma = layer.blur_sigma.clamp(0.0, 64.0);
        if sigma <= 0.001 {
            return String::new();
        }
        format!(",gblur=sigma={:.4}:steps=1", sigma)
    }

    fn build_local_mask_opacity_filter(layer: &LocalMaskLayer) -> String {
        let opacity = layer.opacity.clamp(0.0, 1.0);
        if (opacity - 1.0).abs() <= 0.001 {
            return String::new();
        }
        format!(",colorchannelmixer=aa={:.4}", opacity)
    }

    fn build_local_mask_expr(layer: &LocalMaskLayer) -> String {
        let center_x = layer.center_x.clamp(0.0, 1.0);
        let center_y = layer.center_y.clamp(0.0, 1.0);
        let radius = layer.radius.clamp(0.0, 1.0);
        let feather = layer.feather.clamp(0.0001, 1.0);
        let strength = layer.strength.clamp(0.0, 1.0);
        let dist_expr = format!(
            "hypot((((X+0.5)/W)-{cx:.6})*(W/max(H,1)),(((Y+0.5)/H)-{cy:.6}))",
            cx = center_x,
            cy = center_y
        );
        let edge_expr = format!(
            "clip(({edge:.6}-({dist}))/({feather:.6}),0,1)",
            edge = (radius + feather).clamp(0.0001, 2.0),
            dist = dist_expr,
            feather = feather
        );
        format!(
            "clip(({edge})*{strength:.6},0,1)",
            edge = edge_expr,
            strength = strength
        )
    }

    fn build_local_mask_filters(
        filter_parts: &mut Vec<String>,
        clip: &Clip,
        input_tag: &str,
        label_prefix: &str,
        local_i: usize,
    ) -> String {
        let mut current_tag = input_tag.to_string();
        for (layer_idx, layer) in clip.get_local_mask_layers().iter().enumerate() {
            if !Self::local_mask_layer_has_shape(layer)
                || !Self::local_mask_layer_has_adjustment(layer)
            {
                continue;
            }

            let orig_tag = format!("[{}_lm{}_orig_{}]", label_prefix, layer_idx, local_i);
            let work_tag = format!("[{}_lm{}_work_{}]", label_prefix, layer_idx, local_i);
            filter_parts.push(format!("{}split=2{}{}", current_tag, orig_tag, work_tag));

            let local_eq = Self::build_local_mask_eq_filter(layer);
            let local_blur = Self::build_local_mask_blur_filter(layer);
            let local_opacity = Self::build_local_mask_opacity_filter(layer);
            let work_fx_tag = format!("[{}_lm{}_fx_{}]", label_prefix, layer_idx, local_i);
            filter_parts.push(format!(
                "{}format=bgra{}{}{}{}",
                work_tag, local_eq, local_blur, local_opacity, work_fx_tag
            ));

            let mask_expr = Self::build_local_mask_expr(layer);
            let out_tag = format!("[{}_lm{}_out_{}]", label_prefix, layer_idx, local_i);
            // Avoid st()/ld() state registers in blend expressions; some FFmpeg builds
            // have been observed to crash on complex graphs when registers are used.
            let blend_expr = format!("A*(1-({m}))+B*({m})", m = mask_expr);
            filter_parts.push(format!(
                "{}{}blend=all_expr='{}':shortest=1,format=bgra{}",
                orig_tag, work_fx_tag, blend_expr, out_tag,
            ));
            current_tag = out_tag;
        }
        current_tag
    }

    fn clip_segment_for_export(
        clip: &Clip,
        src_start: Duration,
        timeline_start: Duration,
        duration_secs: f64,
        clear_keys: bool,
    ) -> Clip {
        let mut seg = clip.clone();
        seg.source_in = src_start;
        seg.start = timeline_start;
        seg.duration = Duration::from_secs_f64(duration_secs.max(0.0));
        seg.set_fade_in(0.0);
        seg.set_fade_out(0.0);
        seg.set_dissolve_in(0.0);
        seg.set_dissolve_out(0.0);
        if clear_keys {
            seg.pos_x_keyframes.clear();
            seg.pos_y_keyframes.clear();
            seg.scale_keyframes.clear();
            seg.rotation_keyframes.clear();
            seg.brightness_keyframes.clear();
            seg.contrast_keyframes.clear();
            seg.saturation_keyframes.clear();
            seg.opacity_keyframes.clear();
        }
        seg
    }

    fn build_keyframe_expr(keys: &[ScalarKeyframe], fallback: f32, time_var: &str) -> String {
        if keys.is_empty() {
            return format!("{:.6}", fallback);
        }
        if keys.len() == 1 {
            return format!("{:.6}", keys[0].value);
        }

        let mut expr = format!("{:.6}", keys.last().unwrap().value);
        for idx in (0..keys.len() - 1).rev() {
            let a = &keys[idx];
            let b = &keys[idx + 1];
            let t_a = a.time.as_secs_f64();
            let t_b = b.time.as_secs_f64();
            let span = (t_b - t_a).max(0.000_001);
            let seg_expr = format!(
                "{:.6} + ({:.6}-{:.6})*({}-{:.6})/{:.6}",
                a.value, b.value, a.value, time_var, t_a, span
            );
            expr = format!("if(lt({},{:.6}),{},{})", time_var, t_b, seg_expr, expr);
        }

        let t0 = keys[0].time.as_secs_f64();
        let v0 = keys[0].value;
        format!("if(lt({},{:.6}),{:.6},{})", time_var, t0, v0, expr)
    }

    fn build_slide_expr(clip: &Clip, time_var: &str) -> (String, String) {
        let (in_dir, out_dir, slide_in_raw, slide_out_raw) = clip.get_slide();
        motionloom::experimental::transitions::build_slide_expr(
            clip.duration,
            in_dir,
            out_dir,
            slide_in_raw,
            slide_out_raw,
            time_var,
        )
    }

    fn build_zoom_expr(clip: &Clip, time_var: &str) -> String {
        let (zoom_in_raw, zoom_out_raw, zoom_amount) = clip.get_zoom();
        motionloom::experimental::transitions::build_zoom_expr(
            clip.duration,
            zoom_in_raw,
            zoom_out_raw,
            zoom_amount,
            time_var,
        )
    }

    fn build_shock_zoom_expr(clip: &Clip, time_var: &str) -> String {
        let (shock_in_raw, shock_out_raw, shock_amount) = clip.get_shock_zoom();
        motionloom::experimental::transitions::build_shock_zoom_expr(
            clip.duration,
            shock_in_raw,
            shock_out_raw,
            shock_amount,
            time_var,
        )
    }

    fn build_fade_expr(clip: &Clip, time_var: &str) -> Option<String> {
        let (fade_in_raw, fade_out_raw) = clip.get_fade();
        motionloom::experimental::transitions::build_fade_expr(
            clip.duration,
            fade_in_raw,
            fade_out_raw,
            time_var,
        )
    }

    fn build_dissolve_expr(clip: &Clip, time_var: &str) -> Option<String> {
        let (dissolve_in_raw, dissolve_out_raw) = clip.get_dissolve();
        motionloom::experimental::transitions::build_dissolve_expr(
            clip.duration,
            dissolve_in_raw,
            dissolve_out_raw,
            time_var,
        )
    }

    fn build_layer_gate_expr_for_clip(
        clip: &Clip,
        layer_effect_clips: &[LayerEffectClip],
        time_var: &str,
        with_envelope: bool,
    ) -> Option<String> {
        if clip.duration <= Duration::ZERO || layer_effect_clips.is_empty() {
            return None;
        }

        let clip_start = clip.start;
        let clip_end = clip.start.saturating_add(clip.duration);
        let mut pieces: Vec<String> = Vec::new();
        for layer_clip in layer_effect_clips {
            if layer_clip.duration <= Duration::ZERO {
                continue;
            }
            let layer_start = layer_clip.start;
            let layer_end = layer_clip.start.saturating_add(layer_clip.duration);
            let start = clip_start.max(layer_start);
            let end = clip_end.min(layer_end);
            if end <= start {
                continue;
            }
            let local_start = start.saturating_sub(clip_start).as_secs_f64();
            let local_end = end.saturating_sub(clip_start).as_secs_f64();
            if local_end <= local_start + 0.000_5 {
                continue;
            }

            let layer_start_local = if layer_start >= clip_start {
                layer_start.saturating_sub(clip_start).as_secs_f64()
            } else {
                -(clip_start.saturating_sub(layer_start).as_secs_f64())
            };
            let layer_end_local = layer_start_local + layer_clip.duration.as_secs_f64();
            let fade_in = layer_clip
                .fade_in
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());
            let fade_out = layer_clip
                .fade_out
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());

            if with_envelope {
                let in_expr = if fade_in > 0.000_5 {
                    format!(
                        "clip(({}-({:.6}))/{:.6},0,1)",
                        time_var, layer_start_local, fade_in
                    )
                } else {
                    "1".to_string()
                };
                let out_expr = if fade_out > 0.000_5 {
                    format!(
                        "clip((({:.6})-{})/{:.6},0,1)",
                        layer_end_local, time_var, fade_out
                    )
                } else {
                    "1".to_string()
                };
                let envelope = format!("min({},{})", in_expr, out_expr);
                pieces.push(format!(
                    "between({},{:.6},{:.6})*({})",
                    time_var, local_start, local_end, envelope
                ));
            } else {
                pieces.push(format!(
                    "between({},{:.6},{:.6})",
                    time_var, local_start, local_end
                ));
            }
        }

        if pieces.is_empty() {
            return None;
        }

        let mut expr = pieces[0].clone();
        for piece in pieces.into_iter().skip(1) {
            expr = format!("max({}, {})", expr, piece);
        }

        Some(format!("clip({},0,1)", expr))
    }

    fn pass_param_raw<'a>(pass: &'a MotionloomPassNode, key: &str) -> Option<&'a str> {
        pass.params
            .iter()
            .find(|p| p.key == key)
            .map(|p| p.value.as_str())
    }

    fn normalize_param_value(raw: &str) -> String {
        raw.trim().trim_matches('"').trim().to_string()
    }

    fn pass_param_f64(pass: &MotionloomPassNode, keys: &[&str]) -> Option<f64> {
        for key in keys {
            if let Some(raw) = Self::pass_param_raw(pass, key) {
                let normalized = Self::normalize_param_value(raw);
                if let Ok(v) = normalized.parse::<f64>() {
                    return Some(v);
                }
            }
        }
        None
    }

    fn parse_layer_transition_effect(effect: &str) -> Option<LayerTransitionEffect> {
        let normalized = effect.trim().trim_matches('"').trim().to_ascii_lowercase();
        let normalized = normalized.replace('-', "_");
        match normalized.as_str() {
            "fade_in" => Some(LayerTransitionEffect::FadeIn),
            "fade_out" => Some(LayerTransitionEffect::FadeOut),
            "dip" | "dip_to_black" => Some(LayerTransitionEffect::Dip),
            _ => None,
        }
    }

    fn parse_transition_easing_text(raw: &str) -> Option<PassTransitionEasing> {
        let normalized = raw.trim().trim_matches('"').trim().to_ascii_lowercase();
        match normalized.as_str() {
            "linear" => Some(PassTransitionEasing::Linear),
            "ease-in" | "ease_in" => Some(PassTransitionEasing::EaseIn),
            "ease-out" | "ease_out" => Some(PassTransitionEasing::EaseOut),
            "ease-in-out" | "ease_in_out" => Some(PassTransitionEasing::EaseInOut),
            _ => None,
        }
    }

    fn is_bloom_effect(effect: &str) -> bool {
        matches!(
            effect,
            "bloom"
                | "glow"
                | "glow_bloom"
                | "post.bloom"
                | "post.glow"
                | "post.glow_bloom"
                | "light_atmosphere.bloom"
                | "light_atmosphere.glow"
                | "light_atmosphere.glow_bloom"
        )
    }

    fn build_transition_easing_expr(progress_expr: &str, easing: PassTransitionEasing) -> String {
        match easing {
            PassTransitionEasing::Linear => format!("({})", progress_expr),
            PassTransitionEasing::EaseIn => format!("(({p})*({p}))", p = progress_expr),
            PassTransitionEasing::EaseOut => {
                format!("1-((1-({p}))*(1-({p})))", p = progress_expr)
            }
            PassTransitionEasing::EaseInOut => format!(
                "if(lt({p},0.5),2*({p})*({p}),1-(pow((-2*({p})+2),2)/2))",
                p = progress_expr
            ),
        }
    }

    fn build_layer_transition_expr_from_plan(
        plan: &LayerScriptExportPlan,
        layer_duration_sec: f64,
        layer_local_time_expr: &str,
    ) -> Option<String> {
        let mut opacity_expr: Option<String> = None;
        for transition in &plan.transitions {
            if transition.mode == PassTransitionMode::Off {
                continue;
            }
            let duration_sec = transition.duration_sec.max(0.000_1);
            let start_sec = match transition.start_sec {
                Some(v) => v,
                None => match transition.effect {
                    LayerTransitionEffect::FadeOut => (layer_duration_sec - duration_sec).max(0.0),
                    _ => 0.0,
                },
            };
            let progress = format!(
                "clip((({})-({:.6}))/{:.6},0,1)",
                layer_local_time_expr, start_sec, duration_sec
            );
            let eased = Self::build_transition_easing_expr(&progress, transition.easing.clone());
            let pass_opacity = match transition.effect {
                LayerTransitionEffect::FadeIn => format!("({})", eased),
                LayerTransitionEffect::FadeOut => format!("1-({})", eased),
                LayerTransitionEffect::Dip => format!("abs((2*({}))-1)", eased),
            };
            opacity_expr = Some(match opacity_expr {
                Some(prev) => format!("min({}, {})", prev, pass_opacity),
                None => pass_opacity,
            });
        }
        opacity_expr
    }

    fn analyze_motionloom_script_for_export(script: &str) -> Option<LayerScriptExportPlan> {
        let trimmed = script.trim();
        if trimmed.is_empty() || !is_graph_script(trimmed) {
            return None;
        }
        let graph = parse_process_graph_script(trimmed).ok()?;
        let mut plan = LayerScriptExportPlan {
            apply: graph.apply,
            graph_duration_sec: (graph.duration_ms as f64 / 1000.0).max(0.000_1),
            graph_duration_explicit: graph.duration_explicit,
            ..LayerScriptExportPlan::default()
        };

        for pass in &graph.passes {
            let Some(kernel_name) = resolve_pass_kernel(pass) else {
                continue;
            };
            match kernel_name.as_str() {
                "transition_core.wgsl" => {
                    let Some(effect) = Self::parse_layer_transition_effect(&pass.effect) else {
                        continue;
                    };
                    let mode = pass.transition.clone().unwrap_or(PassTransitionMode::Auto);
                    if mode == PassTransitionMode::Off {
                        continue;
                    }
                    let easing = Self::pass_param_raw(pass, "easing")
                        .and_then(Self::parse_transition_easing_text)
                        .or_else(|| pass.transition_easing.clone())
                        .unwrap_or(PassTransitionEasing::Linear);
                    let start_sec = Self::pass_param_f64(pass, &["startSec", "start_sec"]);
                    let duration_sec = Self::pass_param_f64(pass, &["durationSec", "duration_sec"])
                        .unwrap_or(0.6)
                        .max(0.000_1);
                    plan.transitions.push(LayerTransitionPlan {
                        effect,
                        mode,
                        easing,
                        start_sec,
                        duration_sec,
                    });
                }
                "blur_sharpen_detail_gaussian.wgsl"
                | "blur_sharpen_detail_gaussian_5tap.wgsl"
                | "effect_for_testing_run.wgsl" => {
                    let effect = pass
                        .effect
                        .trim()
                        .trim_matches('"')
                        .trim()
                        .to_ascii_lowercase()
                        .replace('-', "_");
                    let sigma = Self::pass_param_f64(pass, &["sigma"])
                        .unwrap_or(2.0)
                        .clamp(0.0, 64.0);
                    if sigma > 0.001 {
                        if effect == "unsharp" || effect == "sharpen" {
                            plan.sharpen_sigma =
                                Some(plan.sharpen_sigma.map_or(sigma, |prev| prev.max(sigma)));
                        } else {
                            plan.blur_sigma =
                                Some(plan.blur_sigma.map_or(sigma, |prev| prev.max(sigma)));
                        }
                    }
                }
                "composite_core.wgsl" => {
                    let effect = pass
                        .effect
                        .trim()
                        .trim_matches('"')
                        .trim()
                        .to_ascii_lowercase();
                    if (effect == "opacity" || effect == "composite.opacity")
                        && let Some(opacity) = Self::pass_param_f64(pass, &["opacity"])
                    {
                        let opacity = opacity.clamp(0.0, 1.0);
                        plan.opacity_factor = Some(
                            plan.opacity_factor
                                .map_or(opacity, |prev| prev.min(opacity)),
                        );
                    }
                }
                "color_core.wgsl" => {
                    let effect = pass
                        .effect
                        .trim()
                        .trim_matches('"')
                        .trim()
                        .to_ascii_lowercase()
                        .replace('-', "_");
                    if effect == "lut" || effect == "color_tone.lut" {
                        if let Some(mix) = Self::pass_param_f64(pass, &["mix", "lutMix", "lut_mix"])
                        {
                            plan.lut_mix = Some(mix.clamp(0.0, 1.0));
                        }
                        continue;
                    }
                    if Self::is_bloom_effect(&effect) {
                        let threshold = Self::pass_param_f64(pass, &["threshold"])
                            .unwrap_or(0.72)
                            .clamp(0.0, 1.0);
                        let intensity =
                            Self::pass_param_f64(pass, &["intensity", "strength", "amount"])
                                .unwrap_or(1.0)
                                .clamp(0.0, 8.0);
                        let sigma = Self::pass_param_f64(pass, &["sigma", "radius"])
                            .unwrap_or(18.0)
                            .clamp(0.0, 64.0);
                        if intensity > 0.001 && sigma > 0.001 {
                            plan.bloom = Some(LayerBloomPlan {
                                threshold,
                                intensity,
                                sigma,
                            });
                        }
                        continue;
                    }
                    if effect == "hsla_overlay"
                        || effect == "hsla"
                        || effect == "tint_overlay"
                        || effect == "color_tone.hsla_overlay"
                    {
                        let hue = Self::pass_param_f64(pass, &["hue", "h"])
                            .unwrap_or(0.0)
                            .rem_euclid(360.0);
                        let saturation = Self::pass_param_f64(pass, &["saturation", "sat", "s"])
                            .unwrap_or(0.0)
                            .clamp(0.0, 1.0);
                        let lightness = Self::pass_param_f64(pass, &["lightness", "lum", "l"])
                            .unwrap_or(0.0)
                            .clamp(0.0, 1.0);
                        let alpha = Self::pass_param_f64(pass, &["alpha", "a"])
                            .unwrap_or(0.0)
                            .clamp(0.0, 1.0);
                        plan.hsla_overlay = Some(LayerHslaOverlayPlan {
                            hue,
                            saturation,
                            lightness,
                            alpha,
                        });
                    }
                }
                _ => {}
            }
        }

        if plan.transitions.is_empty()
            && plan.blur_sigma.is_none()
            && plan.sharpen_sigma.is_none()
            && plan.lut_mix.is_none()
            && plan.hsla_overlay.is_none()
            && plan.bloom.is_none()
            && plan.opacity_factor.is_none()
        {
            None
        } else {
            Some(plan)
        }
    }

    fn layer_script_plan_for_export(
        layer_clip: &LayerEffectClip,
        cache: &mut std::collections::HashMap<u64, Option<LayerScriptExportPlan>>,
    ) -> Option<LayerScriptExportPlan> {
        let entry = cache.entry(layer_clip.id).or_insert_with(|| {
            if !layer_clip.motionloom_enabled {
                return None;
            }
            Self::analyze_motionloom_script_for_export(&layer_clip.motionloom_script)
        });
        entry.clone()
    }

    fn build_layer_opacity_factor_expr_for_clip(
        clip: &Clip,
        layer_effect_clips: &[LayerEffectClip],
        motionloom_plan_cache: &mut std::collections::HashMap<u64, Option<LayerScriptExportPlan>>,
        time_var: &str,
    ) -> Option<String> {
        if clip.duration <= Duration::ZERO || layer_effect_clips.is_empty() {
            return None;
        }

        let clip_start = clip.start;
        let clip_end = clip.start.saturating_add(clip.duration);
        let mut active_terms = Vec::<String>::new();
        let mut strength_terms = Vec::<String>::new();
        let mut transition_terms = Vec::<String>::new();

        for layer_clip in layer_effect_clips {
            if layer_clip.duration <= Duration::ZERO {
                continue;
            }
            let layer_start = layer_clip.start;
            let layer_end = layer_clip.start.saturating_add(layer_clip.duration);
            let start = clip_start.max(layer_start);
            let end = clip_end.min(layer_end);
            if end <= start {
                continue;
            }
            let local_start = start.saturating_sub(clip_start).as_secs_f64();
            let local_end = end.saturating_sub(clip_start).as_secs_f64();
            if local_end <= local_start + 0.000_5 {
                continue;
            }

            let layer_start_local = if layer_start >= clip_start {
                layer_start.saturating_sub(clip_start).as_secs_f64()
            } else {
                -(clip_start.saturating_sub(layer_start).as_secs_f64())
            };
            let layer_end_local = layer_start_local + layer_clip.duration.as_secs_f64();
            let active_expr = format!("between({},{:.6},{:.6})", time_var, local_start, local_end);

            let fade_in = layer_clip
                .fade_in
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());
            let fade_out = layer_clip
                .fade_out
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());
            let in_expr = if fade_in > 0.000_5 {
                format!(
                    "clip(({}-({:.6}))/{:.6},0,1)",
                    time_var, layer_start_local, fade_in
                )
            } else {
                "1".to_string()
            };
            let out_expr = if fade_out > 0.000_5 {
                format!(
                    "clip((({:.6})-{})/{:.6},0,1)",
                    layer_end_local, time_var, fade_out
                )
            } else {
                "1".to_string()
            };
            let envelope_expr = format!("min({}, {})", in_expr, out_expr);

            let layer_local_expr = format!("({}-({:.6}))", time_var, layer_start_local);
            let script_transition_expr =
                Self::layer_script_plan_for_export(layer_clip, motionloom_plan_cache).and_then(
                    |plan| {
                        let mut effect_exprs = Vec::<String>::new();
                        if let Some(transition_expr) = Self::build_layer_transition_expr_from_plan(
                            &plan,
                            layer_clip.duration.as_secs_f64(),
                            &layer_local_expr,
                        ) {
                            effect_exprs.push(transition_expr);
                        }
                        if let Some(opacity_factor) = plan.opacity_factor {
                            effect_exprs.push(format!("{:.6}", opacity_factor.clamp(0.0, 1.0)));
                        }
                        if effect_exprs.is_empty() {
                            return None;
                        }
                        let mut combined = effect_exprs[0].clone();
                        for expr in effect_exprs.into_iter().skip(1) {
                            combined = format!("min({}, {})", combined, expr);
                        }
                        if plan.apply == GraphApplyScope::Graph && plan.graph_duration_explicit {
                            Some(format!(
                                "if(between({},0,{:.6}),{},1)",
                                layer_local_expr, plan.graph_duration_sec, combined
                            ))
                        } else {
                            Some(combined)
                        }
                    },
                );
            let transition_expr = script_transition_expr.unwrap_or_else(|| "1".to_string());

            active_terms.push(active_expr.clone());
            strength_terms.push(format!("({})*({})", active_expr, envelope_expr));
            transition_terms.push(format!(
                "if(gte({},0.5),{},1)",
                active_expr, transition_expr
            ));
        }

        if active_terms.is_empty() {
            return None;
        }

        let mut active_any = active_terms[0].clone();
        for term in active_terms.iter().skip(1) {
            active_any = format!("max({}, {})", active_any, term);
        }

        let mut strength_max = strength_terms[0].clone();
        for term in strength_terms.iter().skip(1) {
            strength_max = format!("max({}, {})", strength_max, term);
        }

        let mut transition_min = transition_terms[0].clone();
        for term in transition_terms.iter().skip(1) {
            transition_min = format!("min({}, {})", transition_min, term);
        }

        Some(format!(
            "clip(if(gte({},0.5),({})*({}),1),0,1)",
            active_any, strength_max, transition_min
        ))
    }

    fn build_motionloom_blur_filter_for_clip(
        clip: &Clip,
        layer_effect_clips: &[LayerEffectClip],
        motionloom_plan_cache: &mut std::collections::HashMap<u64, Option<LayerScriptExportPlan>>,
        time_var: &str,
    ) -> String {
        if clip.duration <= Duration::ZERO || layer_effect_clips.is_empty() {
            return String::new();
        }

        let clip_start = clip.start;
        let clip_end = clip.start.saturating_add(clip.duration);
        let mut out = String::new();

        for layer_clip in layer_effect_clips {
            if layer_clip.duration <= Duration::ZERO {
                continue;
            }
            let Some(plan) = Self::layer_script_plan_for_export(layer_clip, motionloom_plan_cache)
            else {
                continue;
            };
            let Some(sigma) = plan.blur_sigma else {
                continue;
            };
            if sigma <= 0.001 {
                continue;
            }

            let layer_start = layer_clip.start;
            let layer_end = layer_clip.start.saturating_add(layer_clip.duration);
            let start = clip_start.max(layer_start);
            let end = clip_end.min(layer_end);
            if end <= start {
                continue;
            }
            let local_start = start.saturating_sub(clip_start).as_secs_f64();
            let local_end = end.saturating_sub(clip_start).as_secs_f64();
            if local_end <= local_start + 0.000_5 {
                continue;
            }

            let mut enable_expr =
                format!("between({},{:.6},{:.6})", time_var, local_start, local_end);
            if plan.apply == GraphApplyScope::Graph && plan.graph_duration_explicit {
                let layer_start_local = if layer_start >= clip_start {
                    layer_start.saturating_sub(clip_start).as_secs_f64()
                } else {
                    -(clip_start.saturating_sub(layer_start).as_secs_f64())
                };
                let layer_local_expr = format!("({}-({:.6}))", time_var, layer_start_local);
                enable_expr = format!(
                    "({})*between({},0,{:.6})",
                    enable_expr, layer_local_expr, plan.graph_duration_sec
                );
            }

            out.push_str(&format!(
                ",gblur=sigma={:.4}:steps=1:enable='{}'",
                sigma, enable_expr
            ));
        }

        out
    }

    fn build_motionloom_hsla_filter_for_clip(
        clip: &Clip,
        layer_effect_clips: &[LayerEffectClip],
        motionloom_plan_cache: &mut std::collections::HashMap<u64, Option<LayerScriptExportPlan>>,
        time_var: &str,
    ) -> String {
        if clip.duration <= Duration::ZERO || layer_effect_clips.is_empty() {
            return String::new();
        }

        let clip_start = clip.start;
        let clip_end = clip.start.saturating_add(clip.duration);
        let mut out = String::new();

        for layer_clip in layer_effect_clips {
            if layer_clip.duration <= Duration::ZERO {
                continue;
            }
            let Some(plan) = Self::layer_script_plan_for_export(layer_clip, motionloom_plan_cache)
            else {
                continue;
            };
            let Some(hsla) = plan.hsla_overlay else {
                continue;
            };
            if hsla.alpha <= 0.001 {
                continue;
            }

            let layer_start = layer_clip.start;
            let layer_end = layer_clip.start.saturating_add(layer_clip.duration);
            let start = clip_start.max(layer_start);
            let end = clip_end.min(layer_end);
            if end <= start {
                continue;
            }
            let local_start = start.saturating_sub(clip_start).as_secs_f64();
            let local_end = end.saturating_sub(clip_start).as_secs_f64();
            if local_end <= local_start + 0.000_5 {
                continue;
            }

            let layer_start_local = if layer_start >= clip_start {
                layer_start.saturating_sub(clip_start).as_secs_f64()
            } else {
                -(clip_start.saturating_sub(layer_start).as_secs_f64())
            };
            let layer_end_local = layer_start_local + layer_clip.duration.as_secs_f64();
            let active_expr = format!("between({},{:.6},{:.6})", time_var, local_start, local_end);

            let fade_in = layer_clip
                .fade_in
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());
            let fade_out = layer_clip
                .fade_out
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());
            let in_expr = if fade_in > 0.000_5 {
                format!(
                    "clip(({}-({:.6}))/{:.6},0,1)",
                    time_var, layer_start_local, fade_in
                )
            } else {
                "1".to_string()
            };
            let out_expr = if fade_out > 0.000_5 {
                format!(
                    "clip((({:.6})-{})/{:.6},0,1)",
                    layer_end_local, time_var, fade_out
                )
            } else {
                "1".to_string()
            };
            let envelope_expr = format!("min({}, {})", in_expr, out_expr);

            let layer_local_expr = format!("({}-({:.6}))", time_var, layer_start_local);
            let graph_gate_expr =
                if plan.apply == GraphApplyScope::Graph && plan.graph_duration_explicit {
                    format!(
                        "between({},0,{:.6})",
                        layer_local_expr, plan.graph_duration_sec
                    )
                } else {
                    "1".to_string()
                };
            let alpha_expr = format!(
                "clip(({:.6})*({})*({})*({}),0,1)",
                hsla.alpha, active_expr, envelope_expr, graph_gate_expr
            );
            let (r, g, b) = hsla_to_rgb_components(
                hsla.hue as f32,
                hsla.saturation as f32,
                hsla.lightness as f32,
            );

            out.push_str(&format!(
                ",geq=r='clip(r(X,Y)*(1-({a}))+({r:.6})*({a}),0,255)':g='clip(g(X,Y)*(1-({a}))+({g:.6})*({a}),0,255)':b='clip(b(X,Y)*(1-({a}))+({b:.6})*({a}),0,255)':a='alpha(X,Y)'",
                a = alpha_expr,
                r = r,
                g = g,
                b = b
            ));
        }

        out
    }

    fn build_motionloom_lut_filter_for_clip(
        clip: &Clip,
        layer_effect_clips: &[LayerEffectClip],
        motionloom_plan_cache: &mut std::collections::HashMap<u64, Option<LayerScriptExportPlan>>,
        time_var: &str,
    ) -> String {
        if clip.duration <= Duration::ZERO || layer_effect_clips.is_empty() {
            return String::new();
        }

        let clip_start = clip.start;
        let clip_end = clip.start.saturating_add(clip.duration);
        let mut mix_terms = Vec::<String>::new();

        for layer_clip in layer_effect_clips {
            if layer_clip.duration <= Duration::ZERO {
                continue;
            }
            let Some(plan) = Self::layer_script_plan_for_export(layer_clip, motionloom_plan_cache)
            else {
                continue;
            };
            let Some(mix) = plan.lut_mix else {
                continue;
            };
            if mix <= 0.001 {
                continue;
            }

            let layer_start = layer_clip.start;
            let layer_end = layer_clip.start.saturating_add(layer_clip.duration);
            let start = clip_start.max(layer_start);
            let end = clip_end.min(layer_end);
            if end <= start {
                continue;
            }
            let local_start = start.saturating_sub(clip_start).as_secs_f64();
            let local_end = end.saturating_sub(clip_start).as_secs_f64();
            if local_end <= local_start + 0.000_5 {
                continue;
            }

            let layer_start_local = if layer_start >= clip_start {
                layer_start.saturating_sub(clip_start).as_secs_f64()
            } else {
                -(clip_start.saturating_sub(layer_start).as_secs_f64())
            };
            let layer_end_local = layer_start_local + layer_clip.duration.as_secs_f64();

            let active_expr = format!("between({},{:.6},{:.6})", time_var, local_start, local_end);
            let fade_in = layer_clip
                .fade_in
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());
            let fade_out = layer_clip
                .fade_out
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());
            let in_expr = if fade_in > 0.000_5 {
                format!(
                    "clip(({}-({:.6}))/{:.6},0,1)",
                    time_var, layer_start_local, fade_in
                )
            } else {
                "1".to_string()
            };
            let out_expr = if fade_out > 0.000_5 {
                format!(
                    "clip((({:.6})-{})/{:.6},0,1)",
                    layer_end_local, time_var, fade_out
                )
            } else {
                "1".to_string()
            };
            let envelope_expr = format!("min({}, {})", in_expr, out_expr);
            let layer_local_expr = format!("({}-({:.6}))", time_var, layer_start_local);
            let graph_gate_expr =
                if plan.apply == GraphApplyScope::Graph && plan.graph_duration_explicit {
                    format!(
                        "between({},0,{:.6})",
                        layer_local_expr, plan.graph_duration_sec
                    )
                } else {
                    "1".to_string()
                };

            mix_terms.push(format!(
                "({:.6})*({})*({})*({})",
                mix.clamp(0.0, 1.0),
                active_expr,
                envelope_expr,
                graph_gate_expr
            ));
        }

        if mix_terms.is_empty() {
            return String::new();
        }

        let mut mix_expr = mix_terms[0].clone();
        for term in mix_terms.into_iter().skip(1) {
            mix_expr = format!("max({}, {})", mix_expr, term);
        }
        let mix_expr = format!("clip({},0,1)", mix_expr);

        format!(
            ",geq=r='clip(r(X,Y)*(1+(0.03*({m}))),0,255)':g='g(X,Y)':b='clip(b(X,Y)*(1-(0.03*({m}))),0,255)':a='alpha(X,Y)'",
            m = mix_expr
        )
    }

    fn build_motionloom_sharpen_filter_for_clip(
        clip: &Clip,
        layer_effect_clips: &[LayerEffectClip],
        motionloom_plan_cache: &mut std::collections::HashMap<u64, Option<LayerScriptExportPlan>>,
        time_var: &str,
        sharpen_compensation_multiplier: f64,
    ) -> String {
        if clip.duration <= Duration::ZERO || layer_effect_clips.is_empty() {
            return String::new();
        }

        let clip_start = clip.start;
        let clip_end = clip.start.saturating_add(clip.duration);
        let mut out = String::new();

        for layer_clip in layer_effect_clips {
            if layer_clip.duration <= Duration::ZERO {
                continue;
            }
            let Some(plan) = Self::layer_script_plan_for_export(layer_clip, motionloom_plan_cache)
            else {
                continue;
            };
            let Some(sigma) = plan.sharpen_sigma else {
                continue;
            };
            if sigma <= 0.001 {
                continue;
            }

            let layer_start = layer_clip.start;
            let layer_end = layer_clip.start.saturating_add(layer_clip.duration);
            let start = clip_start.max(layer_start);
            let end = clip_end.min(layer_end);
            if end <= start {
                continue;
            }
            let local_start = start.saturating_sub(clip_start).as_secs_f64();
            let local_end = end.saturating_sub(clip_start).as_secs_f64();
            if local_end <= local_start + 0.000_5 {
                continue;
            }

            let mut enable_expr =
                format!("between({},{:.6},{:.6})", time_var, local_start, local_end);
            if plan.apply == GraphApplyScope::Graph && plan.graph_duration_explicit {
                let layer_start_local = if layer_start >= clip_start {
                    layer_start.saturating_sub(clip_start).as_secs_f64()
                } else {
                    -(clip_start.saturating_sub(layer_start).as_secs_f64())
                };
                let layer_local_expr = format!("({}-({:.6}))", time_var, layer_start_local);
                enable_expr = format!(
                    "({})*between({},0,{:.6})",
                    enable_expr, layer_local_expr, plan.graph_duration_sec
                );
            }

            let sigma = sigma.clamp(0.0, 64.0);
            // For strong sharpen requests, use directional dual-pass kernels.
            // This gives a punchier look than a capped square kernel, while still
            // staying within ffmpeg unsharp's valid range on common builds.
            if sigma >= 7.0 {
                // Scale directional kernel pair by sigma so 12 and 64 no longer look identical.
                // Pair progression (major x minor): 13x13 -> 15x11 -> ... -> 23x3.
                let step = ((sigma - 7.0) / (64.0 - 7.0)).clamp(0.0, 1.0);
                let major_step = (step.sqrt() * 5.0).floor() as i32; // 0..5
                let major = (13 + major_step * 2).clamp(13, 23);
                let minor = (13 - major_step * 2).clamp(3, 13);
                // Lower per-pass strength for dual-pass mode to avoid aggressive halos.
                // We still scale by sigma, but keep the ramp conservative.
                let amount = ((1.00_f64 + step * 0.35_f64) * sharpen_compensation_multiplier)
                    .clamp(0.0, 5.0);
                if major_step == 0 {
                    out.push_str(&format!(
                        ",unsharp=luma_msize_x=13:luma_msize_y=13:luma_amount={a:.4}:chroma_msize_x=13:chroma_msize_y=13:chroma_amount=0.0000:enable='{e}'",
                        a = amount,
                        e = enable_expr
                    ));
                } else {
                    out.push_str(&format!(
                        ",unsharp=luma_msize_x={mx}:luma_msize_y={my}:luma_amount={a:.4}:chroma_msize_x={mx}:chroma_msize_y={my}:chroma_amount=0.0000:enable='{e}'",
                        mx = major,
                        my = minor,
                        a = amount,
                        e = enable_expr
                    ));
                    out.push_str(&format!(
                        ",unsharp=luma_msize_x={mx}:luma_msize_y={my}:luma_amount={a:.4}:chroma_msize_x={mx}:chroma_msize_y={my}:chroma_amount=0.0000:enable='{e}'",
                        mx = minor,
                        my = major,
                        a = amount,
                        e = enable_expr
                    ));
                }
            } else {
                // Keep single-pass close to preview baseline, with only mild yuv420p compensation.
                let amount = (1.05_f64 * sharpen_compensation_multiplier).clamp(0.0, 5.0);
                let mut kernel = (sigma * 2.0).round() as i32;
                // Keep export compatible across ffmpeg builds.
                // In many builds, square unsharp kernels above 13x13 fail with:
                // "(lx/2+ly/2)*2 greater than maximum value 25".
                kernel = kernel.clamp(3, 13);
                if kernel % 2 == 0 {
                    kernel += 1;
                }
                out.push_str(&format!(
                    ",unsharp=luma_msize_x={k}:luma_msize_y={k}:luma_amount={a:.4}:chroma_msize_x={k}:chroma_msize_y={k}:chroma_amount=0.0000:enable='{e}'",
                    k = kernel,
                    a = amount,
                    e = enable_expr
                ));
            }
        }

        out
    }

    fn build_motionloom_bloom_filter_for_clip(
        clip: &Clip,
        layer_effect_clips: &[LayerEffectClip],
        motionloom_plan_cache: &mut std::collections::HashMap<u64, Option<LayerScriptExportPlan>>,
        time_var: &str,
        label_prefix: &str,
        local_i: usize,
    ) -> String {
        if clip.duration <= Duration::ZERO || layer_effect_clips.is_empty() {
            return String::new();
        }

        let clip_start = clip.start;
        let clip_end = clip.start.saturating_add(clip.duration);
        let mut layers = Vec::<(usize, LayerBloomPlan, String)>::new();

        for (layer_idx, layer_clip) in layer_effect_clips.iter().enumerate() {
            if layer_clip.duration <= Duration::ZERO {
                continue;
            }
            let Some(plan) = Self::layer_script_plan_for_export(layer_clip, motionloom_plan_cache)
            else {
                continue;
            };
            let Some(bloom) = plan.bloom else {
                continue;
            };
            if bloom.intensity <= 0.001 || bloom.sigma <= 0.001 {
                continue;
            }

            let layer_start = layer_clip.start;
            let layer_end = layer_clip.start.saturating_add(layer_clip.duration);
            let start = clip_start.max(layer_start);
            let end = clip_end.min(layer_end);
            if end <= start {
                continue;
            }
            let local_start = start.saturating_sub(clip_start).as_secs_f64();
            let local_end = end.saturating_sub(clip_start).as_secs_f64();
            if local_end <= local_start + 0.000_5 {
                continue;
            }

            let layer_start_local = if layer_start >= clip_start {
                layer_start.saturating_sub(clip_start).as_secs_f64()
            } else {
                -(clip_start.saturating_sub(layer_start).as_secs_f64())
            };
            let layer_end_local = layer_start_local + layer_clip.duration.as_secs_f64();
            let active_expr = format!("between({},{:.6},{:.6})", time_var, local_start, local_end);
            let fade_in = layer_clip
                .fade_in
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());
            let fade_out = layer_clip
                .fade_out
                .as_secs_f64()
                .clamp(0.0, layer_clip.duration.as_secs_f64());
            let in_expr = if fade_in > 0.000_5 {
                format!(
                    "clip(({}-({:.6}))/{:.6},0,1)",
                    time_var, layer_start_local, fade_in
                )
            } else {
                "1".to_string()
            };
            let out_expr = if fade_out > 0.000_5 {
                format!(
                    "clip((({:.6})-{})/{:.6},0,1)",
                    layer_end_local, time_var, fade_out
                )
            } else {
                "1".to_string()
            };
            let envelope_expr = format!("min({}, {})", in_expr, out_expr);
            let layer_local_expr = format!("({}-({:.6}))", time_var, layer_start_local);
            let graph_gate_expr =
                if plan.apply == GraphApplyScope::Graph && plan.graph_duration_explicit {
                    format!(
                        "between({},0,{:.6})",
                        layer_local_expr, plan.graph_duration_sec
                    )
                } else {
                    "1".to_string()
                };
            let intensity_expr = format!(
                "clip(({:.6})*({})*({})*({}),0,8)",
                bloom.intensity, active_expr, envelope_expr, graph_gate_expr
            );
            layers.push((layer_idx, bloom, intensity_expr));
        }

        if layers.is_empty() {
            return String::new();
        }

        let mut out = String::new();
        for (i, (layer_idx, bloom, intensity_expr)) in layers.iter().enumerate() {
            let orig_tag = format!("[{}_{}_ml_bloom_{}_orig]", label_prefix, local_i, layer_idx);
            let src_tag = format!("[{}_{}_ml_bloom_{}_src]", label_prefix, local_i, layer_idx);
            let pref_tag = format!("[{}_{}_ml_bloom_{}_pref]", label_prefix, local_i, layer_idx);
            let blur_tag = format!("[{}_{}_ml_bloom_{}_blur]", label_prefix, local_i, layer_idx);
            let out_tag = if i + 1 == layers.len() {
                String::new()
            } else {
                format!("[{}_{}_ml_bloom_{}_out]", label_prefix, local_i, layer_idx)
            };
            if i > 0 {
                out.push(';');
                out.push_str(&format!(
                    "[{}_{}_ml_bloom_{}_out]",
                    label_prefix,
                    local_i,
                    layers[i - 1].0
                ));
            }
            let threshold_255 = (bloom.threshold.clamp(0.0, 1.0) * 255.0).clamp(0.0, 255.0);
            let luma = "(0.2126*r(X,Y)+0.7152*g(X,Y)+0.0722*b(X,Y))";
            let pref_expr = |channel: &str| {
                format!(
                    "if(gte({luma},{thr:.6}),clip({ch}(X,Y)*({intensity}),0,255),0)",
                    luma = luma,
                    thr = threshold_255,
                    ch = channel,
                    intensity = intensity_expr
                )
            };
            out.push_str(&format!(
                "split=2{orig}{src};{src}format=bgra,geq=r='{r}':g='{g}':b='{b}':a='alpha(X,Y)'{pref};{pref}gblur=sigma={sigma:.4}:steps=1,format=bgra{blur};{orig}{blur}blend=all_mode=addition:shortest=1,format=bgra{out_tag}",
                orig = orig_tag,
                src = src_tag,
                r = pref_expr("r"),
                g = pref_expr("g"),
                b = pref_expr("b"),
                pref = pref_tag,
                sigma = bloom.sigma.clamp(0.0, 64.0),
                blur = blur_tag,
                out_tag = out_tag
            ));
        }

        out
    }

    /// [Helper 1] Video Effects (Brightness, Contrast, Saturation) -> GPUI-like correction
    fn build_eq_filter(
        clip: &Clip,
        layer_effects: LayerColorBlurEffects,
        layer_gate_t: Option<&str>,
        layer_gate_upper_t: Option<&str>,
        mode: ExportColorMode,
    ) -> String {
        let b = clip.get_brightness(); // -1.0 ~ 1.0
        let c = clip.get_contrast(); // 0.0 ~ 2.0 (Default 1.0)
        let s = clip.get_saturation(); // 0.0 ~ 2.0 (Default 1.0)
        let layer = layer_effects.normalized();
        let has_keys = !clip.brightness_keyframes.is_empty()
            || !clip.contrast_keyframes.is_empty()
            || !clip.saturation_keyframes.is_empty();
        let has_layer = layer_gate_t.is_some();

        if !has_keys
            && !has_layer
            && b.abs() <= 0.001
            && (c - 1.0).abs() <= 0.001
            && (s - 1.0).abs() <= 0.001
        {
            return String::new();
        }

        let build_geq = |b_expr: String, c_expr: String, s_expr: String| -> String {
            let b_add_expr = format!("({})*255.0", b_expr);
            let l = "(0.2126*r(X,Y)+0.7152*g(X,Y)+0.0722*b(X,Y))";
            let r_expr = format!(
                "clip((({l})+((r(X,Y)-({l}))*({s}))-128)*({c})+128+({b}),0,255)",
                l = l,
                s = s_expr,
                c = c_expr,
                b = b_add_expr
            );
            let g_expr = format!(
                "clip((({l})+((g(X,Y)-({l}))*({s}))-128)*({c})+128+({b}),0,255)",
                l = l,
                s = s_expr,
                c = c_expr,
                b = b_add_expr
            );
            let b_channel_expr = format!(
                "clip((({l})+((b(X,Y)-({l}))*({s}))-128)*({c})+128+({b}),0,255)",
                l = l,
                s = s_expr,
                c = c_expr,
                b = b_add_expr
            );
            format!(
                ",geq=r='{}':g='{}':b='{}':a='alpha(X,Y)'",
                r_expr, g_expr, b_channel_expr
            )
        };

        let merge_layer = |b_expr: String, c_expr: String, s_expr: String, gate: Option<&str>| {
            if let Some(gate) = gate {
                let contrast_gain = layer.contrast - 1.0;
                let saturation_gain = layer.saturation - 1.0;
                (
                    format!(
                        "clip(({})+({:.6})*({}),-1,1)",
                        b_expr, layer.brightness, gate
                    ),
                    format!(
                        "clip(({})*(1+({:.6})*({})),0,2)",
                        c_expr, contrast_gain, gate
                    ),
                    format!(
                        "clip(({})*(1+({:.6})*({})),0,2)",
                        s_expr, saturation_gain, gate
                    ),
                )
            } else {
                (b_expr, c_expr, s_expr)
            }
        };

        match mode {
            ExportColorMode::Fast => {
                let b_expr = if has_keys {
                    Self::build_keyframe_expr(&clip.brightness_keyframes, b, "t")
                } else {
                    format!("{:.6}", b)
                };
                let c_expr = if has_keys {
                    Self::build_keyframe_expr(&clip.contrast_keyframes, c, "t")
                } else {
                    format!("{:.6}", c)
                };
                let s_expr = if has_keys {
                    Self::build_keyframe_expr(&clip.saturation_keyframes, s, "t")
                } else {
                    format!("{:.6}", s)
                };
                let gate = layer_gate_t;
                let (b_expr, c_expr, s_expr) = merge_layer(b_expr, c_expr, s_expr, gate);
                let eval = if has_keys || gate.is_some() {
                    ":eval=frame"
                } else {
                    ""
                };
                format!(
                    ",eq=brightness='{}':contrast='{}':saturation='{}'{}",
                    b_expr, c_expr, s_expr, eval
                )
            }
            ExportColorMode::Hybrid => {
                if has_keys {
                    let b_expr = Self::build_keyframe_expr(&clip.brightness_keyframes, b, "T");
                    let c_expr = Self::build_keyframe_expr(&clip.contrast_keyframes, c, "T");
                    let s_expr = Self::build_keyframe_expr(&clip.saturation_keyframes, s, "T");
                    let (b_expr, c_expr, s_expr) =
                        merge_layer(b_expr, c_expr, s_expr, layer_gate_upper_t);
                    build_geq(b_expr, c_expr, s_expr)
                } else {
                    let b_expr = format!("{:.6}", b);
                    let c_expr = format!("{:.6}", c);
                    let s_expr = format!("{:.6}", s);
                    let gate = layer_gate_t;
                    let (b_expr, c_expr, s_expr) = merge_layer(b_expr, c_expr, s_expr, gate);
                    let eval = if gate.is_some() { ":eval=frame" } else { "" };
                    format!(
                        ",eq=brightness='{}':contrast='{}':saturation='{}'{}",
                        b_expr, c_expr, s_expr, eval
                    )
                }
            }
            ExportColorMode::Exact => {
                let b_expr = if has_keys {
                    Self::build_keyframe_expr(&clip.brightness_keyframes, b, "T")
                } else {
                    format!("{:.6}", b)
                };
                let c_expr = if has_keys {
                    Self::build_keyframe_expr(&clip.contrast_keyframes, c, "T")
                } else {
                    format!("{:.6}", c)
                };
                let s_expr = if has_keys {
                    Self::build_keyframe_expr(&clip.saturation_keyframes, s, "T")
                } else {
                    format!("{:.6}", s)
                };
                let (b_expr, c_expr, s_expr) =
                    merge_layer(b_expr, c_expr, s_expr, layer_gate_upper_t);
                build_geq(b_expr, c_expr, s_expr)
            }
        }
    }

    fn build_opacity_filter(
        clip: &Clip,
        mode: OpacityMode,
        include_dissolve: bool,
        layer_opacity_factor_upper_t: Option<&str>,
    ) -> String {
        let opacity = clip.get_opacity().clamp(0.0, 1.0);
        let has_keys = !clip.opacity_keyframes.is_empty();
        let fade_expr = Self::build_fade_expr(clip, "T");
        let dissolve_expr = if include_dissolve {
            Self::build_dissolve_expr(clip, "T")
        } else {
            None
        };
        let needs_geq = has_keys
            || fade_expr.is_some()
            || dissolve_expr.is_some()
            || layer_opacity_factor_upper_t.is_some();

        if !needs_geq && (opacity - 1.0).abs() <= 0.001 {
            return String::new();
        }

        let opacity_expr = if has_keys {
            Self::build_keyframe_expr(&clip.opacity_keyframes, opacity, "T")
        } else {
            format!("{:.6}", opacity)
        };
        let combined_expr = {
            let mut expr = opacity_expr;
            if let Some(fade_expr) = fade_expr {
                expr = format!("({})*({})", expr, fade_expr);
            }
            if let Some(dissolve_expr) = dissolve_expr {
                expr = format!("({})*({})", expr, dissolve_expr);
            }
            if let Some(layer_factor) = layer_opacity_factor_upper_t {
                expr = format!("({})*({})", expr, layer_factor);
            }
            expr
        };

        match mode {
            OpacityMode::AlphaOnly => {
                if needs_geq {
                    return format!(
                        ",geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='alpha(X,Y)*({})'",
                        combined_expr
                    );
                }
                format!(",colorchannelmixer=aa={:.4}", opacity)
            }
            OpacityMode::MultiplyRgb => {
                if needs_geq {
                    return format!(
                        ",geq=r='r(X,Y)*({e})':g='g(X,Y)*({e})':b='b(X,Y)*({e})':a='alpha(X,Y)'",
                        e = combined_expr
                    );
                }
                format!(
                    ",colorchannelmixer=rr={o:.4}:gg={o:.4}:bb={o:.4}",
                    o = opacity
                )
            }
        }
    }

    fn build_blur_filter(
        clip: &Clip,
        layer_effects: LayerColorBlurEffects,
        layer_gate_t: Option<&str>,
    ) -> String {
        let mut out = String::new();
        let base_sigma = clip.get_blur_sigma().clamp(0.0, 64.0);
        if base_sigma > 0.001 {
            out.push_str(&format!(",gblur=sigma={:.4}:steps=1", base_sigma));
        }

        let layer = layer_effects.normalized();
        if layer.blur_sigma > 0.001
            && let Some(gate) = layer_gate_t
        {
            out.push_str(&format!(
                ",gblur=sigma={:.4}:steps=1:enable='{}'",
                layer.blur_sigma, gate
            ));
        }

        out
    }

    /// [Helper 2] Tint Overlay (Hue, Tint Sat, Lum, Alpha) -> drawbox overlay
    fn build_tint_filter(clip: &Clip) -> String {
        let (h, s, l, a) = clip.get_hsla_overlay();

        // If invisible, skip
        if a < 0.005 {
            return String::new();
        }

        // Use custom hex conversion helper
        let color_hex = hsla_to_hex(h, s, l, a);

        // drawbox=t=fill:replace=0:c=0xRRGGBBAA
        format!(",drawbox=t=fill:replace=0:c={}", color_hex)
    }
}

// Helpers
pub fn has_audio_stream(path: &str) -> bool {
    if is_image_ext(path) {
        return false;
    }
    let ffprobe_bin = resolved_ffprobe_command();
    let out = Command::new(&ffprobe_bin)
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            path,
        ])
        .output();
    match out {
        Ok(o) => !o.stdout.is_empty(),
        Err(_) => false,
    }
}

fn resolved_ffprobe_command() -> String {
    if let Ok(ffprobe_bin) = std::env::var("ANICA_FFPROBE_PATH")
        && !ffprobe_bin.trim().is_empty()
    {
        return ffprobe_bin;
    }
    if let Ok(ffmpeg_bin) = std::env::var("ANICA_FFMPEG_PATH")
        && !ffmpeg_bin.trim().is_empty()
    {
        return ffprobe_from_ffmpeg(&ffmpeg_bin);
    }
    "ffprobe".to_string()
}

pub fn is_supported_media_path(path: &str) -> bool {
    let p = path.to_lowercase();
    p.ends_with(".mp4")
        || p.ends_with(".mov")
        || p.ends_with(".mkv")
        || p.ends_with(".webm")
        || p.ends_with(".avi")
        || p.ends_with(".flv")
        || p.ends_with(".m4v")
        || p.ends_with(".mp3")
        || p.ends_with(".wav")
        || p.ends_with(".m4a")
        || p.ends_with(".aac")
        || p.ends_with(".flac")
        || p.ends_with(".ogg")
        || p.ends_with(".opus")
        || p.ends_with(".jpg")
        || p.ends_with(".jpeg")
        || p.ends_with(".png")
        || p.ends_with(".webp")
        || p.ends_with(".bmp")
        || p.ends_with(".gif")
        || p.ends_with(".tif")
        || p.ends_with(".tiff")
}

fn is_image_ext(path: &str) -> bool {
    let p = path.to_lowercase();
    p.ends_with(".jpg")
        || p.ends_with(".jpeg")
        || p.ends_with(".png")
        || p.ends_with(".webp")
        || p.ends_with(".bmp")
}

pub fn get_media_duration(path: &str) -> Duration {
    if !is_supported_media_path(path) {
        return Duration::ZERO;
    }
    if is_image_ext(path) {
        return Duration::from_secs(5);
    }
    let ffprobe_bin = resolved_ffprobe_command();
    let output = Command::new(&ffprobe_bin)
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ])
        .output();
    if let Ok(out) = output {
        let out_str = String::from_utf8_lossy(&out.stdout);
        if let Ok(secs) = out_str.trim().parse::<f64>() {
            return Duration::from_secs_f64(secs);
        }
    }
    if let Ok(url) = Url::from_file_path(PathBuf::from(path))
        && let Ok(video) = Video::new(&url)
    {
        return video.duration();
    }
    Duration::from_secs(5)
}

fn compute_timeline_max(
    v1_clips: &[Clip],
    audio_tracks: &[AudioTrack],
    video_tracks: &[VideoTrack],
    subtitle_tracks: &[SubtitleTrack],
) -> Duration {
    let mut timeline_max = Duration::ZERO;
    for c in v1_clips {
        timeline_max = timeline_max.max(c.start + c.duration);
    }
    for t in video_tracks {
        for c in &t.clips {
            timeline_max = timeline_max.max(c.start + c.duration);
        }
    }
    for t in audio_tracks {
        for c in &t.clips {
            timeline_max = timeline_max.max(c.start + c.duration);
        }
    }
    for t in subtitle_tracks {
        for c in &t.clips {
            timeline_max = timeline_max.max(c.start + c.duration);
        }
    }
    if timeline_max == Duration::ZERO {
        Duration::from_secs(5)
    } else {
        timeline_max
    }
}

fn compute_audio_timeline_max(audio_tracks: &[AudioTrack]) -> Duration {
    let mut timeline_max = Duration::ZERO;
    for t in audio_tracks {
        for c in &t.clips {
            timeline_max = timeline_max.max(c.start + c.duration);
        }
    }
    if timeline_max == Duration::ZERO {
        Duration::from_secs(5)
    } else {
        timeline_max
    }
}

fn hsla_to_hex(h: f32, s: f32, l: f32, a: f32) -> String {
    let (r, g, b) = hsla_to_rgb_components(h, s, l);
    let alpha = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!(
        "0x{:02X}{:02X}{:02X}{:02X}",
        r as u8, g as u8, b as u8, alpha
    )
}

fn hsla_to_rgb_components(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(360.0);
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r_raw, g_raw, b_raw) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let r = ((r_raw + m) * 255.0).clamp(0.0, 255.0);
    let g = ((g_raw + m) * 255.0).clamp(0.0, 255.0);
    let b = ((b_raw + m) * 255.0).clamp(0.0, 255.0);
    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::{BlurMode, FfmpegExporter, LayerColorBlurEffects, OpacityMode};
    use motionloom::LayerEffectClip;
    use std::collections::HashMap;
    use std::time::Duration;

    #[test]
    fn motionloom_scene_generated_path_is_detected_for_overlay_alpha_decode() {
        assert!(FfmpegExporter::is_motionloom_scene_generated_path(
            "/Users/example/Documents/AnicaProjects/motionloom_generated/motionloom_scene_gpu_123.mp4"
        ));
        assert!(FfmpegExporter::is_motionloom_scene_generated_path(
            "C:\\AnicaProjects\\motionloom_generated\\motionloom_scene_cpu_456.mov"
        ));
        assert!(!FfmpegExporter::is_motionloom_scene_generated_path(
            "/Users/example/Videos/normal_black_background.mp4"
        ));
    }

    #[test]
    fn alpha_preserving_decode_filter_uses_transparent_pad_and_black_key() {
        let filter = FfmpegExporter::build_clip_decode_rgba_filter(1920, 1080, 24, true);
        assert!(filter.contains("pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black@0"));
        assert!(filter.contains("colorkey=0x000000:0.025:0.0"));
        assert!(filter.ends_with("format=rgba"));
    }

    #[test]
    fn normal_decode_filter_keeps_opaque_black_pad() {
        let filter = FfmpegExporter::build_clip_decode_rgba_filter(1920, 1080, 24, false);
        assert!(filter.contains("pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"));
        assert!(!filter.contains("black@0"));
        assert!(!filter.contains("colorkey="));
    }

    #[test]
    fn split_still_image_ignores_source_in_when_building_trim_range() {
        let (start, end) = FfmpegExporter::source_trim_range_for_path(
            "caption.png",
            Duration::from_millis(1_078),
            Duration::from_millis(624),
        );
        assert_eq!(start, 0.0);
        assert!((end - 0.624).abs() < 0.000_001);
    }

    #[test]
    fn video_trim_range_keeps_source_in() {
        let (start, end) = FfmpegExporter::source_trim_range_for_path(
            "clip.mp4",
            Duration::from_millis(1_078),
            Duration::from_millis(624),
        );
        assert!((start - 1.078).abs() < 0.000_001);
        assert!((end - 1.702).abs() < 0.000_001);
    }

    #[test]
    fn overlay_clip_canvas_is_transparent_but_base_canvas_is_opaque() {
        assert_eq!(
            FfmpegExporter::clip_canvas_color(OpacityMode::AlphaOnly),
            "black@0.0"
        );
        assert_eq!(
            FfmpegExporter::clip_canvas_color(OpacityMode::MultiplyRgb),
            "black"
        );
    }

    #[test]
    fn key_near_black_to_alpha_preserves_non_black_pixels() {
        let mut frame = vec![
            0, 0, 0, 255, // pure black background
            12, 10, 9, 255, // near-black compressed background
            220, 64, 44, 255, // red button
            255, 255, 255, 255, // white text
        ];
        FfmpegExporter::key_near_black_to_alpha_rgba(&mut frame);
        assert_eq!(&frame[0..4], &[0, 0, 0, 0]);
        assert_eq!(&frame[4..8], &[0, 0, 0, 0]);
        assert_eq!(&frame[8..12], &[220, 64, 44, 255]);
        assert_eq!(&frame[12..16], &[255, 255, 255, 255]);
    }

    #[test]
    fn unified_still_frame_padding_preserves_alpha() {
        let mut src = image::RgbaImage::new(1, 1);
        src.put_pixel(0, 0, image::Rgba([240, 40, 30, 255]));
        let path = std::env::temp_dir().join(format!(
            "anica_still_alpha_padding_{}_{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        src.save(&path).unwrap();

        let frame =
            FfmpegExporter::load_unified_still_frame_rgba(path.to_str().unwrap(), 4, 2).unwrap();
        let _ = std::fs::remove_file(path);

        assert_eq!(&frame[3..4], &[0], "left padding must stay transparent");
        assert_eq!(
            &frame[(1 * 4)..(1 * 4 + 4)],
            &[240, 40, 30, 255],
            "scaled still content must stay opaque"
        );
        assert_eq!(
            &frame[(3 * 4 + 3)..(3 * 4 + 4)],
            &[0],
            "right padding must stay transparent"
        );
    }

    const PROCESS_LAYER_FX_SCRIPT: &str = r#"
<Graph fps={30} size={[1920,1080]}>
  <Process id="layer_fx">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx_hsla_overlay" kind="compute"
          effect="hsla_overlay"
          in={["src"]} out={["out"]}
          params={{
            hue: "210.0",
            saturation: "0.70",
            lightness: "0.41",
            alpha: "0.45"
          }} />
  </Process>
  <Present from="layer_fx" />
</Graph>
"#;

    const LEGACY_LAYER_FX_SCRIPT: &str = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="clip0" type="video" from="input:clip0" />
  <Tex id="src" fmt="rgba16f" from="clip0" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_hsla_overlay" kind="compute"
        effect="hsla_overlay"
        in={["src"]} out={["out"]}
        params={{ hue: "210.0", saturation: "0.70", lightness: "0.41", alpha: "0.45" }} />
  <Present from="out" />
</Graph>
"#;

    const BLOOM_LAYER_FX_SCRIPT: &str = r#"
<Graph fps={30} size={[1920,1080]}>
  <Process id="layer_fx">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="post_glow_bloom" kind="compute"
          effect="glow_bloom"
          in={["src"]} out={["out"]}
          params={{ threshold: "0.64", intensity: "1.75", sigma: "22.0" }} />
  </Process>
  <Present from="layer_fx" />
</Graph>
"#;

    const TONE_MAP_LAYER_FX_SCRIPT: &str = r#"
<Graph fps={30} size={[1920,1080]}>
  <Process id="layer_fx">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx_tone_map" kind="compute"
          effect="tone_map"
          in={["src"]} out={["out"]}
          params={{
            exposure: "0.35",
            contrast: "1.35",
            shoulder: "0.55",
            gamma: "2.0",
            saturation: "1.22"
          }} />
  </Process>
  <Present from="layer_fx" />
</Graph>
"#;

    const LIGHT_SWEEP_LAYER_FX_SCRIPT: &str = r##"
<Graph fps={30} size={[1920,1080]}>
  <Process id="layer_fx">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx_light_sweep" kind="compute"
          effect="light_sweep"
          in={["src"]} out={["out"]}
          params={{
            position: "0.42",
            angle: "-18.0",
            width: "0.18",
            softness: "0.08",
            intensity: "1.6",
            color: "#80C7FF"
          }} />
  </Process>
  <Present from="layer_fx" />
</Graph>
"##;

    #[test]
    fn export_plan_accepts_process_layer_fx_script() {
        let plan = FfmpegExporter::analyze_motionloom_script_for_export(PROCESS_LAYER_FX_SCRIPT)
            .expect("process layer fx script should produce export plan");
        let hsla = plan.hsla_overlay.expect("HSLA overlay should be analyzed");
        assert_eq!(hsla.hue, 210.0);
        assert_eq!(hsla.saturation, 0.70);
        assert_eq!(hsla.lightness, 0.41);
        assert_eq!(hsla.alpha, 0.45);
    }

    #[test]
    fn export_plan_accepts_process_bloom_layer_fx_script() {
        let plan = FfmpegExporter::analyze_motionloom_script_for_export(BLOOM_LAYER_FX_SCRIPT)
            .expect("process bloom layer fx script should produce export plan");
        let bloom = plan.bloom.expect("bloom should be analyzed");
        assert_eq!(bloom.threshold, 0.64);
        assert_eq!(bloom.intensity, 1.75);
        assert_eq!(bloom.sigma, 22.0);
    }

    #[test]
    fn export_plan_rejects_legacy_root_level_layer_fx_script() {
        assert!(
            FfmpegExporter::analyze_motionloom_script_for_export(LEGACY_LAYER_FX_SCRIPT).is_none()
        );
    }

    const BLUR_LAYER_FX_SCRIPT: &str = r#"
<Graph fps={30} size={[1920,1080]}>
  <Process id="layer_fx">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx_blur" kind="compute"
          effect="gaussian_5tap_blur"
          in={["src"]} out={["out"]}
          params={{ sigma: "10.0" }} />
  </Process>
  <Present from="layer_fx" />
</Graph>
"#;

    const UNSHARP_LAYER_FX_SCRIPT: &str = r#"
<Graph fps={30} size={[1920,1080]}>
  <Process id="layer_fx">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx_unsharp" kind="compute"
          effect="unsharp"
          in={["src"]} out={["out"]}
          params={{ sigma: "2.0" }} />
  </Process>
  <Present from="layer_fx" />
</Graph>
"#;

    fn make_test_layer_effect_clip(script: &str) -> LayerEffectClip {
        LayerEffectClip {
            id: 1,
            start: Duration::ZERO,
            duration: Duration::from_secs(1),
            track_index: 0,
            fade_in: Duration::ZERO,
            fade_out: Duration::ZERO,
            brightness: 0.5,
            contrast: 1.5,
            saturation: 1.5,
            blur_sigma: 5.0,
            brightness_enabled: true,
            contrast_enabled: true,
            saturation_enabled: true,
            blur_enabled: true,
            brightness_keyframes: Vec::new(),
            contrast_keyframes: Vec::new(),
            saturation_keyframes: Vec::new(),
            blur_keyframes: Vec::new(),
            motionloom_enabled: true,
            motionloom_script: script.to_string(),
        }
    }

    #[test]
    fn unified_layer_frame_effects_ignores_legacy_bcs_when_motionloom_runtime_active() {
        let layer_effects = LayerColorBlurEffects {
            brightness: 0.5,
            contrast: 1.5,
            saturation: 1.5,
            blur_sigma: 5.0,
        };
        let clip = make_test_layer_effect_clip(BLUR_LAYER_FX_SCRIPT);
        let mut cache = HashMap::new();

        let fx = FfmpegExporter::unified_layer_frame_effects_at(
            layer_effects,
            &[clip],
            Duration::from_millis(100),
            &mut cache,
        );

        assert!(
            fx.brightness.abs() < 0.001,
            "export brightness should be identity when runtime is active, got {}",
            fx.brightness
        );
        assert!(
            (fx.contrast - 1.0).abs() < 0.001,
            "export contrast should be identity when runtime is active, got {}",
            fx.contrast
        );
        assert!(
            (fx.saturation - 1.0).abs() < 0.001,
            "export saturation should be identity when runtime is active, got {}",
            fx.saturation
        );
        assert!(
            (fx.signed_blur_sigma - 10.0).abs() < 0.001,
            "export blur should come from runtime, got {}",
            fx.signed_blur_sigma
        );
        assert_eq!(fx.blur_mode, BlurMode::Gaussian5tapBlur);
    }

    #[test]
    fn unified_layer_frame_effects_preserves_unsharp_mode_for_sharpen_script() {
        let layer_effects = LayerColorBlurEffects::default();
        let clip = make_test_layer_effect_clip(UNSHARP_LAYER_FX_SCRIPT);
        let mut cache = HashMap::new();

        let fx = FfmpegExporter::unified_layer_frame_effects_at(
            layer_effects,
            &[clip],
            Duration::from_millis(100),
            &mut cache,
        );

        assert_eq!(fx.blur_mode, BlurMode::Unsharp);
        assert!(
            (fx.signed_blur_sigma + 2.0).abs() < 0.001,
            "export sharpen sigma should be negative, got {}",
            fx.signed_blur_sigma
        );
        assert!(
            fx.brightness.abs() < 0.001,
            "export brightness should be identity when runtime is active, got {}",
            fx.brightness
        );
    }

    #[test]
    fn unified_layer_frame_effects_preserves_tone_map_runtime_fields() {
        let layer_effects = LayerColorBlurEffects::default();
        let clip = make_test_layer_effect_clip(TONE_MAP_LAYER_FX_SCRIPT);
        let mut cache = HashMap::new();

        let fx = FfmpegExporter::unified_layer_frame_effects_at(
            layer_effects,
            &[clip],
            Duration::from_millis(100),
            &mut cache,
        );

        assert!(fx.tone_map_enabled);
        assert_eq!(fx.tone_map_exposure, 0.35);
        assert_eq!(fx.tone_map_contrast, 1.35);
        assert_eq!(fx.tone_map_shoulder, 0.55);
        assert_eq!(fx.tone_map_gamma, 2.0);
        assert_eq!(fx.tone_map_saturation, 1.22);
        let effect = fx
            .process_effects
            .iter()
            .find(|effect| effect.effect_id == "tone_map")
            .expect("generic tone_map effect");
        assert_eq!(effect.float("exposure"), Some(0.35));
        assert_eq!(effect.float("contrast"), Some(1.35));
    }

    #[test]
    fn unified_layer_frame_effects_preserves_light_sweep_runtime_fields() {
        let layer_effects = LayerColorBlurEffects::default();
        let clip = make_test_layer_effect_clip(LIGHT_SWEEP_LAYER_FX_SCRIPT);
        let mut cache = HashMap::new();

        let fx = FfmpegExporter::unified_layer_frame_effects_at(
            layer_effects,
            &[clip],
            Duration::from_millis(100),
            &mut cache,
        );

        assert!(fx.light_sweep_enabled);
        assert_eq!(fx.light_sweep_position, 0.42);
        assert_eq!(fx.light_sweep_angle, -18.0);
        assert_eq!(fx.light_sweep_width, 0.18);
        assert_eq!(fx.light_sweep_softness, 0.08);
        assert_eq!(fx.light_sweep_intensity, 1.6);
        assert_eq!(
            fx.light_sweep_color,
            [128.0 / 255.0, 199.0 / 255.0, 1.0, 1.0]
        );
        let effect = fx
            .process_effects
            .iter()
            .find(|effect| effect.effect_id == "light_sweep")
            .expect("generic light_sweep effect");
        assert_eq!(effect.float("position"), Some(0.42));
        assert_eq!(effect.float("intensity"), Some(1.6));
        assert_eq!(
            effect.color("color"),
            Some([128.0 / 255.0, 199.0 / 255.0, 1.0, 1.0])
        );
    }
}
