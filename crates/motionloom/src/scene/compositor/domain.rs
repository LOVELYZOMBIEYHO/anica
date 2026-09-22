// =========================================
// =========================================
// crates/motionloom/src/scene/compositor/domain.rs

use serde::{Deserialize, Serialize};

/// Declares whether authored content belongs to the image plane or 3D world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderDomain {
    /// Display graphics such as titles, subtitles, and HUD elements.
    Screen,
    /// Camera artifacts such as rain, grain, vignette, and chromatic aberration.
    Lens,
    /// A flat image or drawing promoted to 3D geometry and material shading.
    WorldCard,
    /// A decal or projection attached to a 3D surface.
    Surface,
    /// Geometry, lights, volumes, particles, and terrain.
    ThreeD,
}

impl RenderDomain {
    /// World domains must be rendered before image-plane composition because
    /// they participate in visibility, lighting, shadows, and reflections.
    pub const fn requires_world_renderer(self) -> bool {
        matches!(self, Self::WorldCard | Self::Surface | Self::ThreeD)
    }
}

/// Places an operation on one side of the display transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorStage {
    SceneLinearPreDisplay,
    DisplayLinearPostTransform,
}
