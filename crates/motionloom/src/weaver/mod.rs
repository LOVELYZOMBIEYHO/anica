// =========================================
// =========================================
// crates/motionloom/src/weaver/mod.rs

//! Opt-in native offline rendering, independent of immediate preview.
pub mod api;
mod backend;
mod camera;
mod color;
pub mod config;
mod denoise;
pub mod error;
mod geometry;
mod jobs;
mod lighting;
mod output;
pub mod preview;
pub(crate) mod scene;
pub use api::*;
pub use config::*;
pub use error::WeaverError;

#[cfg(test)]
mod tests;
