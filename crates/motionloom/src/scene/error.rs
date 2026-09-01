// =========================================
// =========================================
// crates/motionloom/src/scene/error.rs

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MotionLoomSceneRenderError {
    #[error(
        "MotionLoom scene graph requires at least one node such as <Background>, <Scene>, <Text>, <Image>, <Svg>, <Rect>, <Circle>, <Line>, <Polyline>, <Path>, <FaceJaw>, <Group>, <Mask>, or <Character>."
    )]
    EmptyScene,
    #[error("failed to read system time: {source}")]
    ReadTime { source: std::time::SystemTimeError },
    #[error("failed to create output directory ({path}): {source}")]
    CreateOutputDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to start ffmpeg: {source}")]
    StartFfmpeg { source: std::io::Error },
    #[error("ffmpeg stdin was not available")]
    MissingFfmpegStdin,
    #[error("failed to write raw frame to ffmpeg: {source}. ffmpeg stderr: {stderr}")]
    WriteFrame {
        source: std::io::Error,
        stderr: String,
    },
    #[error("failed to wait for ffmpeg: {source}")]
    WaitFfmpeg { source: std::io::Error },
    #[error("ffmpeg failed: {stderr}")]
    FfmpegFailed { stderr: String },
    #[error("failed to save PNG frame ({path}): {source}")]
    SavePngFrame {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error("invalid color '{value}'")]
    InvalidColor { value: String },
    #[error("invalid scene paint '{value}': {message}")]
    InvalidPaint { value: String, message: String },
    #[error("invalid scene expression '{expr}': {message}")]
    InvalidExpression { expr: String, message: String },
    #[error("invalid scene path data '{value}': {message}")]
    InvalidPathData { value: String, message: String },
    #[error("invalid scene deform grid '{value}': {message}")]
    InvalidDeformGrid { value: String, message: String },
    #[error("failed to open image asset ({path}): {source}")]
    OpenImage {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error("failed to fetch media asset ({url}): {message}")]
    FetchAsset { url: String, message: String },
    #[error("failed to load ActionLibrary ({source_ref}): {message}")]
    LoadActionLibrary { source_ref: String, message: String },
    #[error("failed to parse ActionLibrary ({source_ref}) at line {line}: {message}")]
    ParseActionLibrary {
        source_ref: String,
        line: usize,
        message: String,
    },
    #[error("failed to decode image asset ({source_ref}): {source}")]
    DecodeImage {
        source_ref: String,
        source: image::ImageError,
    },
    #[error("failed to read SVG asset ({path}): {source}")]
    ReadSvg {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse SVG asset ({source_ref}): {source}")]
    ParseSvg {
        source_ref: String,
        source: resvg::usvg::Error,
    },
    #[error("failed to render SVG asset ({source_ref}): invalid SVG size")]
    RenderSvg { source_ref: String },
    #[error("invalid image data URI ({source_ref}): {message}")]
    InvalidImageDataUri { source_ref: String, message: String },
    #[error("invalid SVG data URI ({source_ref}): {message}")]
    InvalidSvgDataUri { source_ref: String, message: String },
    #[error("GPU scene render failed: {message}")]
    GpuRender { message: String },
    #[error("world source render failed: {message}")]
    WorldSource { message: String },
    #[error("video export is not available on this platform: {message}")]
    VideoExportNotAvailable { message: String },
    #[error("scene render cancelled")]
    Cancelled,
    #[error(transparent)]
    Simulation(#[from] crate::simulation::error::SimulationError),
}

impl From<crate::export::EncodeError> for MotionLoomSceneRenderError {
    fn from(err: crate::export::EncodeError) -> Self {
        use crate::export::EncodeError;
        match err {
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

pub type SceneRenderError = MotionLoomSceneRenderError;
