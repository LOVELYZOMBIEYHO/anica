// =========================================
// =========================================
// crates/motionloom/src/weaver/scene/mod.rs

use crate::experimental::geometry::ResolvedMesh;
use crate::world::{WorldCamera, WorldLighting};

/// Evaluated world-space data; no GPU preview buffers are retained.
pub(crate) struct Snapshot {
    pub meshes: Vec<ResolvedMesh>,
    pub camera: WorldCamera,
    pub lighting: WorldLighting,
    pub diagnostics: Vec<String>,
}
