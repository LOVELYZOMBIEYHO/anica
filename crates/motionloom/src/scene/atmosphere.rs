// =========================================
// crates/motionloom/src/scene/atmosphere.rs
// =========================================

use serde::{Deserialize, Serialize};

/// Renderer-independent description of the participating medium authored by
/// `AtmosphereFog`. Density is extinction per world unit and
/// `scattering_color` is single-scattering albedo in linear RGB.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AtmosphereMediumPlan {
    pub density: f32,
    pub scattering_color: [f32; 3],
    pub anisotropy: f32,
    pub base_height: f32,
    pub height_falloff: f32,
    pub bounds_min: Option<[f32; 3]>,
    pub bounds_max: Option<[f32; 3]>,
    pub edge_feather: f32,
    pub affect_environment: bool,
    pub volumetric_scattering: Option<VolumetricScatteringPlan>,
    pub water_caustics: Option<WaterCausticsPlan>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct VolumetricScatteringPlan {
    pub light_ref: String,
    pub shaft_strength: f32,
    pub max_distance: f32,
    pub shadowed: bool,
    pub quality: VolumetricQuality,
    pub max_bounces: u32,
    pub debug_view: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumetricQuality {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WaterCausticsPlan {
    pub intensity: f32,
    pub scale: f32,
    pub speed: f32,
    pub attenuation: f32,
    pub color: [f32; 3],
    pub volume_term: bool,
    pub surface_term: bool,
}
