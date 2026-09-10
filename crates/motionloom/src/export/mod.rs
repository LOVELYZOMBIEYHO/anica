// =========================================
// =========================================
// crates/motionloom/src/export/mod.rs

use std::path::Path;

use thiserror::Error;

/// Platform-neutral description of one rendered frame for export.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// RGBA pixels, row-major, non-premultiplied.
    pub rgba: Vec<u8>,
}

impl VideoFrame {
    pub fn from_rgba_image(image: &image::RgbaImage) -> Self {
        let (width, height) = image.dimensions();
        Self {
            width,
            height,
            rgba: image.as_raw().clone(),
        }
    }
}

/// Errors that can occur while encoding a video.
#[derive(Debug, Error)]
pub enum EncodeError {
    #[error(transparent)]
    AudioPlan(#[from] crate::GraphParseError),
    #[error(transparent)]
    Audio(#[from] crate::audio::AudioError),
    #[error("failed to create output directory ({path}): {source}")]
    CreateOutputDir {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("failed to start encoder: {0}")]
    StartEncoder(String),
    #[error("encoder input was not available")]
    MissingEncoderInput,
    #[error("failed to write frame: {0}")]
    WriteFrame(std::io::Error),
    #[error("encoder failed: {0}")]
    EncoderFailed(String),
    #[error("video export is not available on this platform: {0}")]
    NotImplemented(String),
    #[error("encoder not started")]
    NotStarted,
}

/// Platform-neutral video encoder boundary.
///
/// Implementations handle native FFmpeg subprocesses and browser WebCodecs.
///
/// NOTE: The trait is currently synchronous. A future iteration may need an
/// async variant because WebCodecs `flush`/output-chunk assembly is naturally
/// promise-based.
pub trait VideoEncoder {
    /// Configure the encoder for the given output dimensions and frame rate.
    fn begin(&mut self, width: u32, height: u32, fps: f32) -> Result<(), EncodeError>;

    /// Append one RGBA frame. `frame_index` is informational.
    fn push_frame(&mut self, frame_index: u32, rgba: &[u8]) -> Result<(), EncodeError>;

    /// Finalize the output and return any result object (e.g. a Blob URL on
    /// WASM). Native implementations typically write to `output_path`.
    fn finish(&mut self) -> Result<(), EncodeError>;
}

#[cfg(not(target_arch = "wasm32"))]
mod native;

#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(not(target_arch = "wasm32"))]
pub use native::FfmpegVideoEncoder;

#[cfg(target_arch = "wasm32")]
pub use wasm::WebCodecsVideoEncoder;

/// Convenience constructor that picks the correct platform encoder.
pub fn create_encoder(ffmpeg_bin: &str, output_path: &Path) -> Box<dyn VideoEncoder> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Box::new(FfmpegVideoEncoder::new(ffmpeg_bin, output_path))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = ffmpeg_bin;
        let _ = output_path;
        Box::new(WebCodecsVideoEncoder::new())
    }
}
