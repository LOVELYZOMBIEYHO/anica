// =========================================
// =========================================
// crates/motionloom/src/scene/compositor/error.rs

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SceneCompositionError {
    #[error("scene composition plan version {0} is unsupported")]
    UnsupportedVersion(u32),
    #[error("scene composition output size must be non-zero")]
    EmptyOutput,
    #[error("composition image has {actual} pixels, expected {expected}")]
    InvalidPixelCount { expected: usize, actual: usize },
    #[error("scene `{0}` is missing from the composition graph")]
    MissingScene(String),
    #[error("composition layer `{layer}` uses unsupported domain `{domain}`")]
    UnsupportedDomain { layer: String, domain: String },
    #[error("composition layer `{layer}` requires unavailable AOV `{aov}`")]
    MissingAov { layer: String, aov: String },
    #[error("composition effect `{effect}` on `{layer}` is unsupported")]
    UnsupportedEffect { layer: String, effect: String },
    #[error("composition layer `{0}` is duplicated")]
    DuplicateLayer(String),
}
