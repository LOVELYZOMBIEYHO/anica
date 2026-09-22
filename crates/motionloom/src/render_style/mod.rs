// =========================================
// =========================================
// crates/motionloom/src/render_style/mod.rs

//! Scene-owned visual styles. Authored values stay separate from resolved GPU
//! defaults; no style is an exact opt-out, including for legacy SVG scenes.

mod model;
mod parser;
mod resolve;
mod validation;

pub use model::{
    AntiAliasingStyleNode, CelMaterialSettings, ColorStyleNode, DepthOfFieldStyleNode,
    LightingStyleNode, OutlineStyleNode, PostStyleNode, RenderStyleNode, RenderStyleOverride,
    ResolvedAntiAliasingStyle, ResolvedCelStyle, ResolvedSceneRenderStyle, ResolvedUniversalStyle,
    SurfaceStyleNode, ToneStyleNode,
};
pub(crate) use parser::{parse_cel_material, parse_resource};
pub use resolve::resolve_scene_render_style;
#[cfg(feature = "weaver")]
pub(crate) use resolve::sanitize_scene_style;
pub(crate) use resolve::{apply_scene_style_reference, lower};
pub(crate) use validation::cel_color;

#[cfg(test)]
mod tests;
