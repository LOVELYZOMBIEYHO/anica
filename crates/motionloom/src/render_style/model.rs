// =========================================
// =========================================
// crates/motionloom/src/render_style/model.rs

//! Authored render-style nodes and their resolved runtime representation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RenderStyleNode {
    pub anti_aliasing: Option<AntiAliasingStyleNode>,
    pub depth_of_field: Option<DepthOfFieldStyleNode>,
    pub color: Option<ColorStyleNode>,
    pub tone: Option<ToneStyleNode>,
    pub outline: Option<OutlineStyleNode>,
    pub id: String,
    pub surface: Option<SurfaceStyleNode>,
    pub lighting: Option<LightingStyleNode>,
    pub post: Option<PostStyleNode>,
}

/// Authored anti-aliasing intent; the host resolves it against GPU capabilities.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AntiAliasingStyleNode {
    pub method: Option<String>,
    pub quality: Option<String>,
    pub fallback: Option<String>,
    pub sharpness: Option<f32>,
}

/// Concrete values reported to renderers after defaults are applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedAntiAliasingStyle {
    pub method: String,
    pub quality: String,
    pub fallback: String,
    pub sharpness: f32,
}

impl Default for ResolvedAntiAliasingStyle {
    fn default() -> Self {
        Self {
            method: "off".into(),
            quality: "medium".into(),
            fallback: "auto".into(),
            sharpness: 0.0,
        }
    }
}

/// Opt-in spatial bokeh; omitted quality resolves to balanced without history.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DepthOfFieldStyleNode {
    pub preset: String,
    pub quality: Option<String>,
    /// Filmic lens controls use normalized UV units, not f-numbers.
    pub aperture: Option<f32>,
    pub max_blur: Option<f32>,
}

/// Universal display-space controls applied after every shading preset.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ColorStyleNode {
    pub tint: Option<String>,
    pub tint_strength: Option<f32>,
    pub saturation: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ToneStyleNode {
    pub exposure: Option<f32>,
    pub contrast: Option<f32>,
    pub shadow_color: Option<String>,
    pub highlight_color: Option<String>,
    pub tone_strength: Option<f32>,
}

/// Neutral defaults preserve old serialized graphs and existing render output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ResolvedUniversalStyle {
    pub enabled: bool,
    pub tint: [f32; 3],
    pub tint_strength: f32,
    pub saturation: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub shadow_color: [f32; 3],
    pub highlight_color: [f32; 3],
    pub tone_strength: f32,
}
impl Default for ResolvedUniversalStyle {
    fn default() -> Self {
        Self {
            enabled: false,
            tint: [1.0; 3],
            tint_strength: 0.0,
            saturation: 1.0,
            exposure: 1.0,
            contrast: 1.0,
            shadow_color: [0.0; 3],
            highlight_color: [1.0; 3],
            tone_strength: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SurfaceStyleNode {
    pub shadow_threshold: Option<f32>,
    pub shadow_feather: Option<f32>,
    pub shadow_color: Option<String>,
    pub shading: Option<String>,
    pub shading_steps: Option<u32>,
    pub diffuse_wrap: Option<f32>,
    pub rim_light: Option<f32>,
    pub rim_power: Option<f32>,
    pub specular: Option<f32>,
    pub roughness_bias: Option<f32>,
    pub saturation: Option<f32>,
    pub outline: Option<String>,
}

/// Screen-width geometry outlines are independent of the base material.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OutlineStyleNode {
    pub enabled: Option<bool>,
    pub method: Option<String>,
    pub color: Option<String>,
    pub width: Option<f32>,
    pub distance_mode: Option<String>,
}

/// Optional material-slot controls travel with the actor, never rewrite a GLB.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CelMaterialSettings {
    pub control_map: Option<String>,
    pub role: Option<String>,
    pub outline_width: Option<f32>,
    pub shadow_color: Option<String>,
    pub hair_highlight: Option<f32>,
}
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LightingStyleNode {
    pub preset: Option<String>,
    pub ambient_intensity: Option<f32>,
    pub ambient_color: Option<String>,
    pub shadow_style: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PostStyleNode {
    pub tone_mapping: Option<String>,
    pub exposure: Option<f32>,
    pub saturation: Option<f32>,
    pub contrast: Option<f32>,
    pub white_balance: Option<f32>,
    pub bloom_threshold: Option<f32>,
    pub bloom_intensity: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedCelStyle {
    pub shadow_threshold: f32,
    pub shadow_feather: f32,
    pub shadow_color: [f32; 3],
    pub outline_width: f32,
    pub outline_color: [f32; 3],
}
impl Default for ResolvedCelStyle {
    fn default() -> Self {
        Self {
            shadow_threshold: 0.5,
            shadow_feather: 0.025,
            shadow_color: [0.4, 0.37, 0.5],
            outline_width: 0.0,
            outline_color: [0.0; 3],
        }
    }
}

/// A serializable report, not an alternative authoring source of truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedSceneRenderStyle {
    /// None means no RenderStyle was authored; host preview policy remains in control.
    #[serde(default)]
    pub anti_aliasing: Option<ResolvedAntiAliasingStyle>,
    #[serde(default)]
    pub depth_of_field: Option<DepthOfFieldStyleNode>,
    #[serde(default)]
    pub universal: ResolvedUniversalStyle,
    #[serde(default)]
    pub cel: ResolvedCelStyle,
    pub scene_id: String,
    pub style_id: Option<String>,
    pub shading: String,
    pub shading_steps: u32,
    pub diffuse_wrap: f32,
    pub rim_light: f32,
    pub rim_power: f32,
    pub specular: f32,
    pub roughness_bias: f32,
    pub surface_saturation: f32,
    pub ambient_intensity: f32,
    pub ambient_color: [f32; 3],
    pub hard_shadows: bool,
    pub lighting_preset: Option<String>,
    pub post: PostStyleNode,
    /// Explicit nodes own their complete setting group, including defaults.
    pub overrides: Vec<RenderStyleOverride>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderStyleOverride {
    pub island_id: Option<String>,
    pub property: String,
    pub style_value: serde_json::Value,
    /// Retains expressions; this is compile-time evidence, not a sampled frame.
    pub final_expression: String,
    pub source: String,
}
