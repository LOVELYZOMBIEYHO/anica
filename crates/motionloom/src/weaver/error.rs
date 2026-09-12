// =========================================
// =========================================
// crates/motionloom/src/weaver/error.rs

/// Errors remain typed until a host chooses a transport representation.
#[derive(Debug, thiserror::Error)]
pub enum WeaverError {
    #[error("Invalid Weaver setting: {0}")]
    Invalid(String),
    #[error("Unsupported Weaver feature: {0}")]
    Unsupported(String),
    #[error("Scene evaluation: {0}")]
    Scene(String),
    #[error("GPU: {0}")]
    Gpu(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
}
