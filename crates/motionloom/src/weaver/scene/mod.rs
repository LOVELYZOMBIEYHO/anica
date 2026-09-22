// =========================================
// =========================================
// crates/motionloom/src/weaver/scene/mod.rs

use crate::experimental::geometry::ResolvedMesh;
use crate::scene::compositor::{ResolvedCompositionLayer, SceneCompositionPlan};
use crate::world::{WorldCamera, WorldLighting};

/// Evaluated world-space data; no GPU preview buffers are retained.
pub(crate) struct Snapshot {
    pub meshes: Vec<ResolvedMesh>,
    /// Parallel to `meshes`; false affects primary rays only, so shadows and
    /// reflections retain the authored object.
    pub primary_camera_visibility: Vec<bool>,
    pub camera: WorldCamera,
    pub lighting: WorldLighting,
    pub time_seconds: f32,
    pub diagnostics: Vec<String>,
    pub composition: SceneCompositionPlan,
    /// Evaluated 2D runs, converted at the Raster boundary into the canonical
    /// linear-premultiplied working representation without flattening them.
    pub composition_layers: Vec<ResolvedCompositionLayer>,
}
