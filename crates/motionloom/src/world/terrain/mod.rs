// =========================================
// =========================================
// crates/motionloom/src/world/terrain/mod.rs

use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use image::RgbaImage;

use crate::dsl::{PrimitiveAssetNode, PrimitiveCollisionNode, PrimitiveGeometry, TerrainAssetNode};
use crate::world::gltf_loader::GlbMeshData;
use crate::world::primitive::{MeshBuilder, PrimitiveTextureSet};

/// Generate one deterministic heightfield mesh. The renderer can retain this
/// exactly like a generated primitive or loaded GLB on both supported GPU paths.
pub fn generate_terrain_mesh_textured(
    asset: &TerrainAssetNode,
    height_map: &RgbaImage,
    texture_set: PrimitiveTextureSet,
) -> GlbMeshData {
    let x_samples = sample_coordinates(height_map.width(), terrain_stride(asset, height_map));
    let z_samples = sample_coordinates(height_map.height(), terrain_stride(asset, height_map));
    let mut builder = MeshBuilder::default();
    let mut vertices = vec![vec![0_u32; x_samples.len()]; z_samples.len()];
    let mut triangle_chunks = Vec::new();
    let mut minimum_height = f32::INFINITY;
    let mut maximum_height = f32::NEG_INFINITY;

    for (z_index, &source_z) in z_samples.iter().enumerate() {
        for (x_index, &source_x) in x_samples.iter().enumerate() {
            let u = normalized_coordinate(source_x, height_map.width());
            let v = normalized_coordinate(source_z, height_map.height());
            let height = sample_height(asset, height_map, source_x, source_z);
            minimum_height = minimum_height.min(height);
            maximum_height = maximum_height.max(height);
            let normal = terrain_normal(asset, height_map, source_x, source_z);
            vertices[z_index][x_index] = builder.vertex(
                [(u - 0.5) * asset.size[0], height, (v - 0.5) * asset.size[1]],
                normal,
                [u, v],
            );
        }
    }

    // Row-major cells share boundary vertices, so authored chunk counts never
    // introduce cracks even before renderer-level chunk culling is applied.
    for z in 0..z_samples.len().saturating_sub(1) {
        for x in 0..x_samples.len().saturating_sub(1) {
            let a = vertices[z][x];
            let b = vertices[z + 1][x];
            let c = vertices[z + 1][x + 1];
            let d = vertices[z][x + 1];
            builder.triangle(a, b, c);
            builder.triangle(a, c, d);
            let chunk = terrain_chunk_index(
                x,
                z,
                x_samples.len().saturating_sub(1),
                z_samples.len().saturating_sub(1),
                asset.chunks,
            );
            triangle_chunks.extend([chunk, chunk]);
        }
    }

    let material = terrain_surface_primitive(asset);
    let bounds = (
        [-asset.size[0] * 0.5, minimum_height, -asset.size[1] * 0.5],
        [asset.size[0] * 0.5, maximum_height, asset.size[1] * 0.5],
    );
    let mut mesh = builder.finish_with_bounds(&material, texture_set, bounds);
    mesh.path = terrain_cache_key(asset, height_map);
    for (triangle, chunk) in mesh.triangles.iter_mut().zip(triangle_chunks) {
        triangle.mesh = Some(chunk);
    }
    mesh.mesh_names = (0..asset.chunks[0] * asset.chunks[1])
        .map(|chunk| {
            let x = chunk % asset.chunks[0];
            let z = chunk / asset.chunks[0];
            Some(format!("MotionLoom Terrain {} chunk {x},{z}", asset.id))
        })
        .collect();
    mesh
}

pub fn terrain_cache_key(asset: &TerrainAssetNode, height_map: &RgbaImage) -> PathBuf {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_terrain_declaration(asset, &mut hasher);
    height_map.dimensions().hash(&mut hasher);
    height_map.as_raw().hash(&mut hasher);
    PathBuf::from(format!("motionloom-terrain-{:016x}", hasher.finish()))
}

/// Stable renderer-lifetime key for an immutable remote terrain declaration.
///
/// Remote image bytes are expensive to resolve and hash on every frame. The
/// source URL is already part of the declaration, so the renderer can use this
/// key to discover an existing generated mesh before fetching the height map.
/// Local and in-memory sources continue to use `terrain_cache_key`, preserving
/// their content-revision invalidation behaviour.
pub fn remote_terrain_cache_key(asset: &TerrainAssetNode) -> PathBuf {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    "remote".hash(&mut hasher);
    hash_terrain_declaration(asset, &mut hasher);
    PathBuf::from(format!(
        "motionloom-remote-terrain-{:016x}",
        hasher.finish()
    ))
}

fn hash_terrain_declaration(asset: &TerrainAssetNode, hasher: &mut impl Hasher) {
    asset.id.hash(hasher);
    asset.height_map_src.hash(hasher);
    asset.size.map(f32::to_bits).hash(hasher);
    asset.height_scale.to_bits().hash(hasher);
    asset.height_offset.to_bits().hash(hasher);
    asset.layers.hash(hasher);
    format!("{:?}", asset.material_definition).hash(hasher);
    format!("{:?}", asset.layer_definitions).hash(hasher);
    asset.blend_map_src.hash(hasher);
    asset.chunks.hash(hasher);
    asset.lod.hash(hasher);
}

fn terrain_chunk_index(
    x: usize,
    z: usize,
    x_cells: usize,
    z_cells: usize,
    chunks: [u32; 2],
) -> usize {
    let chunk_x = (x * chunks[0] as usize / x_cells.max(1)).min(chunks[0] as usize - 1);
    let chunk_z = (z * chunks[1] as usize / z_cells.max(1)).min(chunks[1] as usize - 1);
    chunk_z * chunks[0] as usize + chunk_x
}

pub fn terrain_bounds(asset: &TerrainAssetNode) -> ([f32; 3], [f32; 3]) {
    let first = asset.height_offset;
    let second = asset.height_offset + asset.height_scale;
    (
        [
            -asset.size[0] * 0.5,
            first.min(second),
            -asset.size[1] * 0.5,
        ],
        [asset.size[0] * 0.5, first.max(second), asset.size[1] * 0.5],
    )
}

pub(crate) fn terrain_collision_triangles(
    asset: &TerrainAssetNode,
    height_map: &RgbaImage,
) -> Vec<crate::world::model_inspection::EnvironmentWalkableTriangle> {
    let mesh = generate_terrain_mesh_textured(asset, height_map, PrimitiveTextureSet::default());
    crate::world::model_inspection::environment_collision_triangles(&mesh)
}

pub(crate) fn terrain_surface_primitive(asset: &TerrainAssetNode) -> PrimitiveAssetNode {
    let mut definition = asset
        .material_definition
        .clone()
        .or_else(|| asset.layer_definitions.first().cloned());
    if !asset.layer_definitions.is_empty()
        && let Some(material) = definition.as_mut()
    {
        // Layer textures and scalar factors are baked before mesh upload.
        material.base_color = [1.0; 4];
        material.metallic = 1.0;
        material.roughness = 1.0;
        material.normal_scale = 1.0;
        material.emissive = [0.0; 3];
        material.emissive_strength = 1.0;
        material.texture_scale = [1.0; 2];
        material.texture_offset = [0.0; 2];
        material.texture_rotation = 0.0;
    }
    PrimitiveAssetNode {
        id: asset.id.clone(),
        geometry: PrimitiveGeometry::Plane {
            size: asset.size,
            segments: 1,
        },
        color: [1.0; 4],
        material: asset
            .material
            .clone()
            .or_else(|| asset.layers.first().cloned()),
        material_definition: definition,
        bevel_radius: 0.0,
        bevel_segments: 0,
        material_seed: None,
        collision: PrimitiveCollisionNode::default(),
        modifiers: Vec::new(),
        mesh_build: Default::default(),
        lod: Default::default(),
    }
}

fn terrain_stride(asset: &TerrainAssetNode, height_map: &RgbaImage) -> u32 {
    match asset.lod.as_str() {
        "full" => 1,
        "half" => 2,
        "quarter" => 4,
        _ => height_map
            .width()
            .max(height_map.height())
            .saturating_sub(1)
            .div_ceil(256)
            .max(1),
    }
}

fn sample_coordinates(length: u32, stride: u32) -> Vec<u32> {
    let last = length.saturating_sub(1);
    let mut coordinates = (0..=last)
        .step_by(stride.max(1) as usize)
        .collect::<Vec<_>>();
    if coordinates.last().copied() != Some(last) {
        coordinates.push(last);
    }
    coordinates
}

fn normalized_coordinate(value: u32, length: u32) -> f32 {
    value as f32 / length.saturating_sub(1).max(1) as f32
}

fn sample_height(asset: &TerrainAssetNode, image: &RgbaImage, x: u32, z: u32) -> f32 {
    let pixel = image.get_pixel(x.min(image.width() - 1), z.min(image.height() - 1));
    let luminance = (0.2126 * f32::from(pixel[0])
        + 0.7152 * f32::from(pixel[1])
        + 0.0722 * f32::from(pixel[2]))
        / 255.0;
    asset.height_offset + luminance * asset.height_scale
}

fn terrain_normal(asset: &TerrainAssetNode, image: &RgbaImage, x: u32, z: u32) -> [f32; 3] {
    let left = sample_height(asset, image, x.saturating_sub(1), z);
    let right = sample_height(asset, image, (x + 1).min(image.width() - 1), z);
    let back = sample_height(asset, image, x, z.saturating_sub(1));
    let front = sample_height(asset, image, x, (z + 1).min(image.height() - 1));
    let cell_x = asset.size[0] / image.width().saturating_sub(1).max(1) as f32;
    let cell_z = asset.size[1] / image.height().saturating_sub(1).max(1) as f32;
    normalize([
        -(right - left) / (2.0 * cell_x),
        1.0,
        -(front - back) / (2.0 * cell_z),
    ])
}

fn normalize(value: [f32; 3]) -> [f32; 3] {
    let length = value
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if length <= f32::EPSILON {
        [0.0, 1.0, 0.0]
    } else {
        value.map(|component| component / length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::PrimitiveCollisionMode;

    fn terrain() -> TerrainAssetNode {
        TerrainAssetNode {
            id: "ground".into(),
            height_map: "height".into(),
            height_map_src: Some("height.png".into()),
            size: [4.0, 4.0],
            height_scale: 2.0,
            height_offset: -0.5,
            material: Some("soil".into()),
            material_definition: None,
            layers: Vec::new(),
            layer_definitions: Vec::new(),
            blend_map: None,
            blend_map_src: None,
            chunks: [2, 2],
            lod: "full".into(),
            collision: PrimitiveCollisionMode::Solid,
        }
    }

    #[test]
    fn heightfield_builds_expected_bounds_and_upward_normals() {
        let image = RgbaImage::from_fn(3, 3, |x, _| image::Rgba([(x * 127) as u8, 0, 0, 255]));
        let mesh = generate_terrain_mesh_textured(&terrain(), &image, Default::default());
        assert_eq!(mesh.positions.len(), 9);
        assert_eq!(mesh.triangles.len(), 8);
        assert!(mesh.bounds_max[1] > mesh.bounds_min[1]);
        assert!(mesh.normals.iter().flatten().all(|normal| normal[1] > 0.0));
    }

    #[test]
    fn heightfield_assigns_triangles_to_authored_chunks() {
        let image = RgbaImage::from_pixel(5, 5, image::Rgba([128, 128, 128, 255]));
        let mesh =
            generate_terrain_mesh_textured(&terrain(), &image, PrimitiveTextureSet::default());
        assert_eq!(mesh.mesh_names.len(), 4);
        assert!(
            mesh.triangles
                .iter()
                .all(|triangle| triangle.mesh.is_some())
        );
        assert_eq!(
            mesh.triangles
                .iter()
                .filter_map(|triangle| triangle.mesh)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            4
        );
    }

    #[test]
    fn remote_cache_key_is_stable_and_tracks_the_height_source() {
        let mut first = terrain();
        first.height_map_src = Some("https://example.com/height-a.png".into());
        let mut second = first.clone();
        assert_eq!(
            remote_terrain_cache_key(&first),
            remote_terrain_cache_key(&second)
        );

        second.height_map_src = Some("https://example.com/height-b.png".into());
        assert_ne!(
            remote_terrain_cache_key(&first),
            remote_terrain_cache_key(&second)
        );
    }
}
