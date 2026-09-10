// =========================================
// =========================================
// crates/motionloom/src/geometry/mod.rs

//! Camera-independent, static asset tooling. The DSL remains authoritative.
mod glb;
mod uv;
use crate::world::gltf_loader::{GlbMaterialData, GlbTextureData};
pub use glb::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use uv::*;

#[derive(Debug, thiserror::Error)]
pub enum GeometryError {
    #[error("Geometry evaluation: {0}")]
    Evaluation(String),
    #[error("Unsupported geometry export: {0}")]
    Unsupported(String),
    #[error("Invalid geometry: {0}")]
    Invalid(String),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Default)]
pub struct SceneGeometryOptions {
    pub scene_id: String,
    pub frame: u32,
    pub include_hidden: bool,
}

#[derive(Debug, Clone)]
pub struct ResolvedMesh {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub material: GlbMaterialData,
    pub textures: Vec<GlbTextureData>,
    /// Missing authored UVs must not be disguised by renderer fallback UVs.
    pub uv_source: String,
}

#[derive(Debug, Clone, Default)]
pub struct GeometrySnapshot {
    pub meshes: Vec<ResolvedMesh>,
    pub topology_signature: String,
    pub uv_signature: String,
    pub diagnostics: Vec<String>,
}

impl GeometrySnapshot {
    pub(crate) fn sign(&mut self) {
        // Topology and UV revisions are separate so moving a model is harmless.
        let mut topology = Sha256::new();
        let mut uv = Sha256::new();
        for mesh in &self.meshes {
            topology.update((mesh.positions.len() as u64).to_le_bytes());
            topology.update((mesh.indices.len() as u64).to_le_bytes());
            for i in &mesh.indices {
                topology.update(i.to_le_bytes());
            }
            uv.update((mesh.uvs.len() as u64).to_le_bytes());
            uv.update(mesh.uv_source.as_bytes());
            for v in mesh.uvs.iter().flatten() {
                uv.update(v.to_le_bytes());
            }
        }
        self.topology_signature = format!("topology-v1:{:x}", topology.finalize());
        self.uv_signature = format!("uv-v1:{:x}", uv.finalize());
    }
}

/// Resolve a sampled scene without a GPU or a camera visibility filter.
pub async fn extract_scene_geometry(
    graph: &crate::GraphScript,
    options: &SceneGeometryOptions,
) -> Result<GeometrySnapshot, GeometryError> {
    extract_scene_geometry_with_resolver(
        graph,
        options,
        std::sync::Arc::new(crate::asset::PathAssetResolver),
    )
    .await
}

/// Hosts may supply in-memory assets, including browser/WASM callers.
pub async fn extract_scene_geometry_with_resolver(
    graph: &crate::GraphScript,
    options: &SceneGeometryOptions,
    resolver: std::sync::Arc<dyn crate::asset::AssetResolver>,
) -> Result<GeometrySnapshot, GeometryError> {
    crate::scene::render::extract_geometry_snapshot(graph, options, resolver).await
}
