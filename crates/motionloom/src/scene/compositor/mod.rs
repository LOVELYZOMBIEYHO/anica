// =========================================
// =========================================
// crates/motionloom/src/scene/compositor/mod.rs

//! Renderer-independent scene composition contracts.
//!
//! The DSL is compiled into [`SceneCompositionPlan`] once. Raster, Weaver,
//! native, and WebGPU executors then bind renderer-owned textures without
//! teaching the plan about a particular GPU API.

mod capabilities;
mod collect;
mod color;
mod domain;
mod error;
mod executor;
#[cfg(not(target_arch = "wasm32"))]
mod gpu;
mod image;
mod plan;

pub use capabilities::*;
pub use collect::*;
pub use color::*;
pub use domain::*;
pub use error::*;
pub use executor::*;
#[cfg(not(target_arch = "wasm32"))]
pub use gpu::*;
pub use image::*;
pub use plan::*;
