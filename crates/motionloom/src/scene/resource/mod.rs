//! Reusable Scene definitions and asset/resource resolution.
//!
//! Owns definitions such as fonts, gradients, brushes, masks, components,
//! precomposes, filters, palettes, and scene asset lookup. Rendering code
//! should resolve resources through this boundary instead of treating `Defs`
//! as drawable content.

mod assets;
mod defs;
mod fonts;

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use assets::scene_asset_relative_suffixes;
pub use assets::{clear_scene_asset_roots, set_scene_asset_roots};
pub(crate) use assets::{
    default_world_asset_root, load_rgba_image_source, load_svg_source, load_utf8_text_source,
    resolve_local_scene_asset_path,
};

pub(crate) use fonts::load_extra_fonts;

pub(crate) use defs::{
    collect_graph_component_defs, collect_graph_filter_defs, collect_graph_font_defs,
    collect_graph_gradient_defs, collect_graph_mask_defs, collect_graph_material_defs,
    collect_graph_noise_defs, collect_graph_palette_defs, collect_graph_precompose_defs,
    collect_graph_texture_defs,
};
