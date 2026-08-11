// =========================================
// =========================================
// crates/motionloom/src/scene/backend/encoding.rs

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::scene::render::MotionLoomSceneRenderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneRenderProfile {
    Cpu,
    Gpu,
    GpuProRes,
    GpuProRes4444,
    GpuPngSequence,
}

impl SceneRenderProfile {
    pub const fn output_extension(self) -> &'static str {
        match self {
            SceneRenderProfile::Cpu => "mov",
            SceneRenderProfile::Gpu => "mp4",
            SceneRenderProfile::GpuProRes => "mov",
            SceneRenderProfile::GpuProRes4444 => "mov",
            SceneRenderProfile::GpuPngSequence => "png",
        }
    }

    pub const fn output_prefix(self) -> &'static str {
        match self {
            SceneRenderProfile::Cpu => "motionloom_scene",
            SceneRenderProfile::Gpu => "motionloom_scene_gpu",
            SceneRenderProfile::GpuProRes => "motionloom_scene_gpu_prores",
            SceneRenderProfile::GpuProRes4444 => "motionloom_scene_gpu_prores4444",
            SceneRenderProfile::GpuPngSequence => "motionloom_scene_gpu_png",
        }
    }

    pub const fn uses_gpu_compositor(self) -> bool {
        matches!(
            self,
            SceneRenderProfile::Gpu
                | SceneRenderProfile::GpuProRes
                | SceneRenderProfile::GpuProRes4444
                | SceneRenderProfile::GpuPngSequence
        )
    }

    pub const fn is_png_sequence(self) -> bool {
        matches!(self, SceneRenderProfile::GpuPngSequence)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SceneRenderProgress {
    pub rendered_frames: u32,
    pub total_frames: u32,
}

#[allow(dead_code)]
pub fn next_scene_output_path(output_dir: &Path) -> Result<PathBuf, MotionLoomSceneRenderError> {
    next_scene_output_path_for_profile(output_dir, SceneRenderProfile::Cpu)
}

pub fn next_scene_output_path_for_profile(
    output_dir: &Path,
    profile: SceneRenderProfile,
) -> Result<PathBuf, MotionLoomSceneRenderError> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|source| MotionLoomSceneRenderError::ReadTime { source })?
        .as_millis();
    if profile.is_png_sequence() {
        return Ok(output_dir.join(format!("{}_{}", profile.output_prefix(), stamp)));
    }

    Ok(output_dir.join(format!(
        "{}_{}.{}",
        profile.output_prefix(),
        stamp,
        profile.output_extension()
    )))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn scene_encoder_args(
    profile: SceneRenderProfile,
    width: u32,
    height: u32,
    fps: f32,
) -> Vec<String> {
    match profile {
        SceneRenderProfile::Cpu => prores_encoder_args(),
        SceneRenderProfile::Gpu => gpu_h264_encoder_args(width, height, fps),
        SceneRenderProfile::GpuProRes => prores_encoder_args(),
        SceneRenderProfile::GpuProRes4444 => prores_4444_encoder_args(),
        SceneRenderProfile::GpuPngSequence => Vec::new(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct H264RateControl {
    bitrate_mbps: u32,
    maxrate_mbps: u32,
    buffer_mbps: u32,
    gop_frames: u32,
}

#[cfg(not(target_arch = "wasm32"))]
fn h264_rate_control(width: u32, height: u32, fps: f32) -> H264RateControl {
    // Scale from a high-quality 1080p30 baseline while keeping normal delivery
    // bitrates. Square-root FPS scaling avoids an excessive 2x jump at 60 fps.
    let pixel_factor = (width.max(1) as f64 * height.max(1) as f64) / (1920.0 * 1080.0);
    let fps = fps.max(1.0) as f64;
    let fps_factor = (fps / 30.0).clamp(0.5, 2.0).sqrt();
    let bitrate_mbps = (12.0 * pixel_factor * fps_factor).round().clamp(8.0, 80.0) as u32;
    let maxrate_mbps = bitrate_mbps;
    let buffer_mbps = bitrate_mbps.saturating_mul(2);
    let gop_frames = (fps * 2.0).round().clamp(24.0, 240.0) as u32;

    H264RateControl {
        bitrate_mbps,
        maxrate_mbps,
        buffer_mbps,
        gop_frames,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn prores_encoder_args() -> Vec<String> {
    // Keep scene output on an LGPL-safe FFmpeg-friendly path.
    // The app's curated preview runtime does not ship libav, so mp4v/mpeg4
    // decodes poorly there. ProRes MOV is larger but avoids GPL encoders.
    // Use ProRes HQ instead of Proxy: flat anime colors plus fine strokes show
    // visible chroma/luma waves after low-bitrate mezzanine compression.
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
fn prores_4444_encoder_args() -> Vec<String> {
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
fn gpu_h264_encoder_args(width: u32, height: u32, fps: f32) -> Vec<String> {
    let rate = h264_rate_control(width, height, fps);
    vec![
        "-c:v".to_string(),
        "h264_videotoolbox".to_string(),
        "-allow_sw".to_string(),
        "1".to_string(),
        "-profile:v".to_string(),
        "high".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        // VideoToolbox otherwise treats -b:v as a loose VBR hint and can
        // undershoot motion-graphics exports by an order of magnitude.
        "-constant_bit_rate".to_string(),
        "1".to_string(),
        "-b:v".to_string(),
        format!("{}M", rate.bitrate_mbps),
        "-maxrate".to_string(),
        format!("{}M", rate.maxrate_mbps),
        "-bufsize".to_string(),
        format!("{}M", rate.buffer_mbps),
        "-g".to_string(),
        rate.gop_frames.to_string(),
        "-bf".to_string(),
        "2".to_string(),
        "-coder".to_string(),
        "cabac".to_string(),
        "-spatial_aq".to_string(),
        "1".to_string(),
        // Cap the worst per-frame quantizer so flat text and thin lines cannot
        // be sacrificed when VideoToolbox undershoots its bitrate target.
        "-qmax".to_string(),
        "18".to_string(),
        "-realtime".to_string(),
        "0".to_string(),
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
fn gpu_h264_encoder_args(width: u32, height: u32, fps: f32) -> Vec<String> {
    let rate = h264_rate_control(width, height, fps);
    vec![
        "-c:v".to_string(),
        "h264_mf".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-b:v".to_string(),
        format!("{}M", rate.bitrate_mbps),
        "-maxrate".to_string(),
        format!("{}M", rate.maxrate_mbps),
        "-bufsize".to_string(),
        format!("{}M", rate.buffer_mbps),
        "-g".to_string(),
        rate.gop_frames.to_string(),
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
fn gpu_h264_encoder_args(width: u32, height: u32, fps: f32) -> Vec<String> {
    let rate = h264_rate_control(width, height, fps);
    vec![
        "-c:v".to_string(),
        "libopenh264".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-b:v".to_string(),
        format!("{}M", rate.bitrate_mbps),
        "-maxrate".to_string(),
        format!("{}M", rate.maxrate_mbps),
        "-bufsize".to_string(),
        format!("{}M", rate.buffer_mbps),
        "-g".to_string(),
        rate.gop_frames.to_string(),
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::{SceneRenderProfile, h264_rate_control, scene_encoder_args};

    #[test]
    fn h264_quality_scales_with_resolution_and_frame_rate() {
        assert_eq!(h264_rate_control(1920, 1080, 30.0).bitrate_mbps, 12);
        assert_eq!(h264_rate_control(3840, 2160, 30.0).bitrate_mbps, 48);
        assert_eq!(h264_rate_control(3840, 2160, 60.0).bitrate_mbps, 68);
        assert_eq!(h264_rate_control(1280, 720, 30.0).bitrate_mbps, 8);
    }

    #[test]
    fn gpu_h264_uses_two_second_gop_and_universal_pixel_format() {
        let args = scene_encoder_args(SceneRenderProfile::Gpu, 3840, 2160, 30.0);
        let joined = args.join(" ");
        assert!(joined.contains("-b:v 48M"));
        assert!(joined.contains("-g 60"));
        assert!(joined.contains("-pix_fmt yuv420p"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_gpu_h264_prevents_videotoolbox_vbr_undershoot() {
        let args = scene_encoder_args(SceneRenderProfile::Gpu, 3840, 2160, 30.0);
        let joined = args.join(" ");
        assert!(joined.contains("-constant_bit_rate 1"));
        assert!(joined.contains("-coder cabac"));
        assert!(joined.contains("-spatial_aq 1"));
        assert!(joined.contains("-qmax 18"));
        assert!(joined.contains("-bf 2"));
    }
}
