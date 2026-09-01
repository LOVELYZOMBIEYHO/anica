// =========================================
// =========================================
// crates/motionloom/src/world/primitive/mod.rs

mod box_mesh;
mod capsule_mesh;
mod cone_mesh;
mod cylinder_mesh;
mod plane_mesh;
mod sphere_mesh;
mod wedge_mesh;

use std::path::PathBuf;

use crate::dsl::{PrimitiveAssetNode, PrimitiveGeometry};
use crate::world::gltf_loader::{
    GlbAlphaMode, GlbDepthWriteMode, GlbMaterialData, GlbMeshData, GlbTextureData, GlbTriangle,
};

#[derive(Clone, Debug, Default)]
pub struct PrimitiveTextureSet {
    pub base_color: Option<GlbTextureData>,
    pub metallic_roughness: Option<GlbTextureData>,
    pub normal: Option<GlbTextureData>,
    pub emissive: Option<GlbTextureData>,
}

/// Generate the canonical triangle mesh used by both native and WASM renderers.
pub fn generate_primitive_mesh(asset: &PrimitiveAssetNode) -> GlbMeshData {
    generate_primitive_mesh_textured(asset, PrimitiveTextureSet::default())
}

/// Generate a primitive and attach already-resolved PBR textures to its glTF-like mesh.
pub fn generate_primitive_mesh_textured(
    asset: &PrimitiveAssetNode,
    texture_set: PrimitiveTextureSet,
) -> GlbMeshData {
    let mut builder = MeshBuilder::default();
    match &asset.geometry {
        PrimitiveGeometry::Box { size } if asset.bevel_radius > 0.0 => box_mesh::generate_beveled(
            &mut builder,
            *size,
            asset.bevel_radius,
            asset.bevel_segments.max(1),
        ),
        PrimitiveGeometry::Box { size } => box_mesh::generate(&mut builder, *size),
        PrimitiveGeometry::Sphere {
            radius,
            segments,
            rings,
        } => sphere_mesh::generate(&mut builder, *radius, *segments, *rings),
        PrimitiveGeometry::Capsule {
            radius,
            height,
            segments,
            rings,
        } => capsule_mesh::generate(&mut builder, *radius, *height, *segments, *rings),
        PrimitiveGeometry::Plane { size, segments } => {
            plane_mesh::generate(&mut builder, *size, *segments)
        }
        PrimitiveGeometry::Cylinder {
            radius,
            height,
            segments,
        } => cylinder_mesh::generate(&mut builder, *radius, *height, *segments),
        PrimitiveGeometry::Cone {
            radius,
            height,
            segments,
        } => cone_mesh::generate(&mut builder, *radius, *height, *segments),
        PrimitiveGeometry::Wedge { size } => wedge_mesh::generate(&mut builder, *size),
    }
    builder.finish(asset, texture_set)
}

/// Return exact local-space bounds without loading or tessellating an asset.
pub fn primitive_bounds(geometry: &PrimitiveGeometry) -> ([f32; 3], [f32; 3]) {
    match geometry {
        PrimitiveGeometry::Box { size } | PrimitiveGeometry::Wedge { size } => {
            let half = size.map(|value| value * 0.5);
            (half.map(|value| -value), half)
        }
        PrimitiveGeometry::Sphere { radius, .. } => {
            ([-radius, -radius, -radius], [*radius, *radius, *radius])
        }
        PrimitiveGeometry::Capsule { radius, height, .. } => (
            [-radius, -height * 0.5, -radius],
            [*radius, height * 0.5, *radius],
        ),
        PrimitiveGeometry::Plane { size, .. } => (
            [-size[0] * 0.5, 0.0, -size[1] * 0.5],
            [size[0] * 0.5, 0.0, size[1] * 0.5],
        ),
        PrimitiveGeometry::Cylinder { radius, height, .. }
        | PrimitiveGeometry::Cone { radius, height, .. } => (
            [-radius, -height * 0.5, -radius],
            [*radius, height * 0.5, *radius],
        ),
    }
}

/// Build the geometry portion of a typed primitive resource identity.
pub fn primitive_geometry_cache_key(asset: &PrimitiveAssetNode) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    hash_geometry(&asset.geometry, &mut hash);
    hash_bytes(&mut hash, &asset.bevel_radius.to_bits().to_le_bytes());
    hash_bytes(&mut hash, &asset.bevel_segments.to_le_bytes());
    if let Some(material) = &asset.material_definition {
        // Projection changes base vertex UVs; scale/rotation/variation remain
        // material/instance data and must not split geometry identity.
        hash_bytes(&mut hash, material.mapping.as_bytes());
    }
    hash
}

/// Build the material portion of a typed primitive resource identity.
pub fn primitive_material_cache_key(asset: &PrimitiveAssetNode) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    if let Some(material) = &asset.material_definition {
        hash_bytes(&mut hash, material.id.as_bytes());
        hash_bytes(&mut hash, material.shading.as_bytes());
        hash_f32s(&mut hash, &material.base_color);
        for reference in [
            &material.base_color_texture,
            &material.metallic_roughness_texture,
            &material.normal_texture,
            &material.occlusion_texture,
            &material.emissive_texture,
        ] {
            hash_optional_string(&mut hash, reference);
        }
        for source in [
            &material.base_color_texture_src,
            &material.metallic_roughness_texture_src,
            &material.normal_texture_src,
            &material.occlusion_texture_src,
            &material.emissive_texture_src,
        ] {
            hash_optional_string(&mut hash, source);
        }
        hash_f32s(
            &mut hash,
            &[
                material.metallic,
                material.roughness,
                material.normal_scale,
                material.occlusion_strength,
                material.emissive_strength,
                material.specular,
                material.alpha_cutoff,
                material.transmission,
                material.ior,
                material.thickness,
                material.attenuation_distance,
                material.texture_rotation,
            ],
        );
        hash_f32s(&mut hash, &material.emissive);
        hash_f32s(&mut hash, &material.attenuation_color);
        hash_bytes(&mut hash, &[material.double_sided as u8]);
        hash_bytes(&mut hash, material.alpha_mode.as_bytes());
        hash_bytes(&mut hash, material.depth_write.as_bytes());
        hash_bytes(&mut hash, &material.sort_priority.to_le_bytes());
        hash_bytes(&mut hash, material.mapping.as_bytes());
        hash_f32s(&mut hash, &material.texture_scale);
        hash_f32s(&mut hash, &material.texture_offset);
        hash_f32s(&mut hash, &material.variation_amount);
    }
    for channel in asset.color {
        hash_bytes(&mut hash, &channel.to_bits().to_le_bytes());
    }
    hash
}

/// Build a stable retained mesh/material key. Per-instance material variation
/// is deliberately excluded so CompoundAsset children share retained data.
pub fn primitive_cache_key(asset: &PrimitiveAssetNode) -> PathBuf {
    let geometry = primitive_geometry_cache_key(asset);
    let material = primitive_material_cache_key(asset);
    let mut hash = 0xcbf29ce484222325_u64;
    hash_bytes(&mut hash, &geometry.to_le_bytes());
    hash_bytes(&mut hash, &material.to_le_bytes());
    PathBuf::from(format!("motionloom-primitive-{hash:016x}"))
}

/// Resolve deterministic UV translation for one primitive instance.
pub fn primitive_material_seed_offset(asset: &PrimitiveAssetNode) -> [f32; 2] {
    let Some(material) = asset.material_definition.as_ref() else {
        return [0.0; 2];
    };
    let seed = asset.material_seed.unwrap_or_default();
    let random = |salt: u64| {
        let mut value = seed ^ salt.wrapping_mul(0x9e3779b97f4a7c15);
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58476d1ce4e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d049bb133111eb);
        ((value ^ (value >> 31)) as u32) as f32 / u32::MAX as f32
    };
    [
        (random(1) - 0.5) * 2.0 * material.variation_amount[0],
        (random(2) - 0.5) * 2.0 * material.variation_amount[1],
    ]
}

fn hash_f32s(hash: &mut u64, values: &[f32]) {
    for value in values {
        hash_bytes(hash, &value.to_bits().to_le_bytes());
    }
}

fn hash_optional_string(hash: &mut u64, value: &Option<String>) {
    hash_bytes(hash, &[value.is_some() as u8]);
    if let Some(value) = value {
        hash_bytes(hash, value.as_bytes());
    }
}

fn hash_geometry(geometry: &PrimitiveGeometry, hash: &mut u64) {
    match geometry {
        PrimitiveGeometry::Box { size } | PrimitiveGeometry::Wedge { size } => {
            hash_bytes(
                hash,
                &[matches!(geometry, PrimitiveGeometry::Wedge { .. }) as u8],
            );
            size.iter()
                .for_each(|value| hash_bytes(hash, &value.to_bits().to_le_bytes()));
        }
        PrimitiveGeometry::Sphere {
            radius,
            segments,
            rings,
        } => {
            hash_bytes(hash, &[2]);
            hash_bytes(hash, &radius.to_bits().to_le_bytes());
            hash_bytes(hash, &segments.to_le_bytes());
            hash_bytes(hash, &rings.to_le_bytes());
        }
        PrimitiveGeometry::Capsule {
            radius,
            height,
            segments,
            rings,
        } => {
            hash_bytes(hash, &[6]);
            hash_bytes(hash, &radius.to_bits().to_le_bytes());
            hash_bytes(hash, &height.to_bits().to_le_bytes());
            hash_bytes(hash, &segments.to_le_bytes());
            hash_bytes(hash, &rings.to_le_bytes());
        }
        PrimitiveGeometry::Plane { size, segments } => {
            hash_bytes(hash, &[3]);
            size.iter()
                .for_each(|value| hash_bytes(hash, &value.to_bits().to_le_bytes()));
            hash_bytes(hash, &segments.to_le_bytes());
        }
        PrimitiveGeometry::Cylinder {
            radius,
            height,
            segments,
        }
        | PrimitiveGeometry::Cone {
            radius,
            height,
            segments,
        } => {
            hash_bytes(
                hash,
                &[matches!(geometry, PrimitiveGeometry::Cone { .. }) as u8 + 4],
            );
            hash_bytes(hash, &radius.to_bits().to_le_bytes());
            hash_bytes(hash, &height.to_bits().to_le_bytes());
            hash_bytes(hash, &segments.to_le_bytes());
        }
    }
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    // FNV-1a keeps cache identities reproducible across processes and targets.
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

#[derive(Default)]
pub(crate) struct MeshBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<Option<[f32; 3]>>,
    texcoords: Vec<Option<[f32; 2]>>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    pub(crate) fn vertex(&mut self, position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> u32 {
        let index = self.positions.len() as u32;
        self.positions.push(position);
        self.normals.push(Some(normal));
        self.texcoords.push(Some(uv));
        index
    }

    pub(crate) fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend([a, b, c]);
    }

    pub(super) fn quad(&mut self, vertices: [[f32; 3]; 4], normal: [f32; 3]) {
        let ids = [
            self.vertex(vertices[0], normal, [0.0, 0.0]),
            self.vertex(vertices[1], normal, [1.0, 0.0]),
            self.vertex(vertices[2], normal, [1.0, 1.0]),
            self.vertex(vertices[3], normal, [0.0, 1.0]),
        ];
        self.triangle(ids[0], ids[1], ids[2]);
        self.triangle(ids[0], ids[2], ids[3]);
    }

    pub(super) fn triangle_face(&mut self, vertices: [[f32; 3]; 3], normal: [f32; 3]) {
        let a = self.vertex(vertices[0], normal, [0.0, 0.0]);
        let b = self.vertex(vertices[1], normal, [1.0, 0.0]);
        let c = self.vertex(vertices[2], normal, [1.0, 1.0]);
        self.triangle(a, b, c);
    }

    fn finish(self, asset: &PrimitiveAssetNode, texture_set: PrimitiveTextureSet) -> GlbMeshData {
        self.finish_with_bounds(asset, texture_set, primitive_bounds(&asset.geometry))
    }

    pub(crate) fn finish_with_bounds(
        mut self,
        asset: &PrimitiveAssetNode,
        texture_set: PrimitiveTextureSet,
        bounds: ([f32; 3], [f32; 3]),
    ) -> GlbMeshData {
        apply_material_uvs(asset, &self.positions, &self.normals, &mut self.texcoords);
        let triangles = self
            .indices
            .chunks_exact(3)
            .map(|indices| GlbTriangle {
                indices: [indices[0], indices[1], indices[2]],
                material: Some(0),
                mesh: Some(0),
                mesh_node: None,
            })
            .collect();
        let (bounds_min, bounds_max) = bounds;
        let vertex_count = self.positions.len();
        let material_definition = asset.material_definition.as_ref();
        let mut textures = Vec::new();
        let mut push_texture = |texture: Option<GlbTextureData>| {
            texture.map(|texture| {
                let index = textures.len();
                textures.push(Some(texture));
                index
            })
        };
        let base_color_texture = push_texture(texture_set.base_color);
        let metallic_roughness_texture = push_texture(texture_set.metallic_roughness);
        let normal_texture = push_texture(texture_set.normal);
        let emissive_texture = push_texture(texture_set.emissive);
        let material_color = material_definition
            .map(|material| {
                std::array::from_fn(|index| material.base_color[index] * asset.color[index])
            })
            .unwrap_or(asset.color);
        GlbMeshData {
            path: primitive_cache_key(asset),
            positions: self.positions,
            normals: self.normals,
            texcoords: self.texcoords,
            colors: vec![None; vertex_count],
            joints: vec![None; vertex_count],
            weights: vec![None; vertex_count],
            indices: self.indices,
            triangles,
            materials: vec![GlbMaterialData {
                name: Some("MotionLoom Primitive".to_string()),
                base_color_factor: material_color,
                base_color_texture,
                metallic_roughness_texture,
                normal_texture,
                normal_scale: material_definition.map_or(1.0, |material| material.normal_scale),
                emissive_texture,
                emissive_factor: material_definition.map_or([0.0; 3], |material| material.emissive),
                emissive_strength: material_definition
                    .map_or(1.0, |material| material.emissive_strength),
                metallic_factor: material_definition.map_or(0.0, |material| material.metallic),
                roughness_factor: material_definition.map_or(0.82, |material| material.roughness),
                specular_factor: material_definition.map_or(1.0, |material| material.specular),
                alpha_mode: match material_definition.map(|material| material.alpha_mode.as_str()) {
                    Some("mask") => GlbAlphaMode::Mask,
                    Some("blend") => GlbAlphaMode::Blend,
                    _ => GlbAlphaMode::Opaque,
                },
                alpha_cutoff: material_definition.map_or(0.5, |material| material.alpha_cutoff),
                transmission_factor: material_definition
                    .map_or(0.0, |material| material.transmission),
                ior: material_definition.map_or(1.5, |material| material.ior),
                thickness_factor: material_definition.map_or(0.0, |material| material.thickness),
                attenuation_color: material_definition
                    .map_or([1.0; 3], |material| material.attenuation_color),
                attenuation_distance: material_definition
                    .map_or(1_000_000.0, |material| material.attenuation_distance),
                depth_write: match material_definition.map(|material| material.depth_write.as_str())
                {
                    Some("true") => GlbDepthWriteMode::Enabled,
                    Some("false") => GlbDepthWriteMode::Disabled,
                    _ => GlbDepthWriteMode::Auto,
                },
                sort_priority: material_definition.map_or(0, |material| material.sort_priority),
                double_sided: material_definition.is_some_and(|material| material.double_sided),
                ..GlbMaterialData::default()
            }],
            textures,
            mesh_names: vec![Some(format!("MotionLoom {}", asset.geometry.shape_name()))],
            nodes: Vec::new(),
            skin: None,
            animations: Vec::new(),
            bounds_min,
            bounds_max,
        }
    }
}

fn apply_material_uvs(
    asset: &PrimitiveAssetNode,
    positions: &[[f32; 3]],
    normals: &[Option<[f32; 3]>],
    texcoords: &mut [Option<[f32; 2]>],
) {
    let Some(material) = asset.material_definition.as_ref() else {
        return;
    };
    for (index, uv) in texcoords.iter_mut().enumerate() {
        let mapped = if material.mapping == "uv" {
            uv.unwrap_or([0.0; 2])
        } else {
            let position = positions[index];
            let normal = normals[index].unwrap_or([0.0, 1.0, 0.0]);
            let absolute = normal.map(f32::abs);
            if absolute[1] >= absolute[0] && absolute[1] >= absolute[2] {
                [position[0], position[2]]
            } else if absolute[0] >= absolute[2] {
                [position[2], position[1]]
            } else {
                [position[0], position[1]]
            }
        };
        *uv = Some(mapped);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_material() -> crate::dsl::MaterialAssetNode {
        crate::dsl::MaterialAssetNode {
            id: "stone".into(),
            shading: "pbr".into(),
            base_color: [1.0; 4],
            base_color_texture: Some("stone_image".into()),
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            base_color_texture_src: Some("stone.jpg".into()),
            metallic_roughness_texture_src: None,
            normal_texture_src: None,
            occlusion_texture_src: None,
            emissive_texture_src: None,
            metallic: 0.0,
            roughness: 0.8,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            emissive_strength: 1.0,
            specular: 0.3,
            double_sided: false,
            alpha_mode: "opaque".into(),
            alpha_cutoff: 0.5,
            transmission: 0.0,
            ior: 1.5,
            thickness: 0.0,
            attenuation_color: [1.0; 3],
            attenuation_distance: 1_000_000.0,
            depth_write: "auto".into(),
            sort_priority: 0,
            mapping: "triplanar".into(),
            texture_scale: [0.28; 2],
            texture_offset: [0.0; 2],
            texture_rotation: 0.0,
            variation_amount: [0.34, 0.22],
        }
    }

    fn seeded_box(seed: u64) -> PrimitiveAssetNode {
        PrimitiveAssetNode {
            id: format!("step_{seed}"),
            geometry: PrimitiveGeometry::Box {
                size: [4.4, 0.32, 0.9],
            },
            color: [1.0; 4],
            material: Some("stone".into()),
            material_definition: Some(seeded_material()),
            bevel_radius: 0.025,
            bevel_segments: 3,
            material_seed: Some(seed),
            collision: Default::default(),
        }
    }

    #[test]
    fn box_has_expected_topology_and_bounds() {
        let asset = PrimitiveAssetNode {
            id: "box".to_string(),
            geometry: PrimitiveGeometry::Box {
                size: [2.0, 4.0, 6.0],
            },
            color: [1.0, 0.0, 0.0, 1.0],
            material: None,
            material_definition: None,
            bevel_radius: 0.0,
            bevel_segments: 0,
            material_seed: None,
            collision: Default::default(),
        };
        let mesh = generate_primitive_mesh(&asset);
        assert_eq!(mesh.positions.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
        assert_eq!(mesh.bounds_min, [-1.0, -2.0, -3.0]);
        assert_eq!(mesh.bounds_max, [1.0, 2.0, 3.0]);
        assert!(
            mesh.texcoords
                .iter()
                .flatten()
                .all(|uv| uv.iter().all(|v| (0.0..=1.0).contains(v)))
        );
    }

    #[test]
    fn equal_content_reuses_cache_key_across_asset_ids() {
        let geometry = PrimitiveGeometry::Sphere {
            radius: 1.0,
            segments: 16,
            rings: 8,
        };
        let a = PrimitiveAssetNode {
            id: "a".into(),
            geometry: geometry.clone(),
            color: [0.0, 1.0, 1.0, 1.0],
            material: None,
            material_definition: None,
            bevel_radius: 0.0,
            bevel_segments: 0,
            material_seed: None,
            collision: Default::default(),
        };
        let b = PrimitiveAssetNode {
            id: "b".into(),
            geometry,
            color: [0.0, 1.0, 1.0, 1.0],
            material: None,
            material_definition: None,
            bevel_radius: 0.0,
            bevel_segments: 0,
            material_seed: None,
            collision: Default::default(),
        };
        assert_eq!(primitive_cache_key(&a), primitive_cache_key(&b));
    }

    #[test]
    fn material_seed_is_instance_data_not_retained_mesh_identity() {
        let first = seeded_box(76);
        let second = seeded_box(77);
        assert_eq!(primitive_cache_key(&first), primitive_cache_key(&second));
        assert_eq!(
            generate_primitive_mesh(&first).texcoords,
            generate_primitive_mesh(&second).texcoords
        );
        assert_ne!(
            primitive_material_seed_offset(&first),
            primitive_material_seed_offset(&second)
        );
        assert_eq!(
            primitive_material_seed_offset(&first),
            primitive_material_seed_offset(&first)
        );
    }

    #[test]
    fn geometry_and_material_identities_invalidate_independently() {
        let original = seeded_box(76);
        let mut material_change = original.clone();
        material_change
            .material_definition
            .as_mut()
            .expect("test material")
            .roughness = 0.4;
        assert_eq!(
            primitive_geometry_cache_key(&original),
            primitive_geometry_cache_key(&material_change)
        );
        assert_ne!(
            primitive_material_cache_key(&original),
            primitive_material_cache_key(&material_change)
        );

        let mut geometry_change = original.clone();
        geometry_change.bevel_radius = 0.04;
        assert_ne!(
            primitive_geometry_cache_key(&original),
            primitive_geometry_cache_key(&geometry_change)
        );
        assert_eq!(
            primitive_material_cache_key(&original),
            primitive_material_cache_key(&geometry_change)
        );
    }

    #[test]
    fn visual_bevel_adds_geometry_without_expanding_authored_bounds() {
        let asset = PrimitiveAssetNode {
            id: "rounded_step".into(),
            geometry: PrimitiveGeometry::Box {
                size: [4.4, 0.32, 0.9],
            },
            color: [1.0; 4],
            material: None,
            material_definition: None,
            bevel_radius: 0.025,
            bevel_segments: 3,
            material_seed: None,
            collision: Default::default(),
        };
        let mesh = generate_primitive_mesh(&asset);
        assert!(mesh.indices.len() > 36);
        assert_eq!(mesh.bounds_min, [-2.2, -0.16, -0.45]);
        assert_eq!(mesh.bounds_max, [2.2, 0.16, 0.45]);
        assert!(mesh.positions.iter().all(|position| {
            position[0].abs() <= 2.2 && position[1].abs() <= 0.16 && position[2].abs() <= 0.45
        }));
    }

    #[test]
    fn every_v1_shape_has_complete_gpu_geometry() {
        let cases = [
            (
                PrimitiveGeometry::Box {
                    size: [2.0, 4.0, 6.0],
                },
                36,
                [-1.0, -2.0, -3.0],
                [1.0, 2.0, 3.0],
            ),
            (
                PrimitiveGeometry::Sphere {
                    radius: 2.0,
                    segments: 8,
                    rings: 4,
                },
                8 * 4 * 6,
                [-2.0; 3],
                [2.0; 3],
            ),
            (
                PrimitiveGeometry::Plane {
                    size: [4.0, 6.0],
                    segments: 2,
                },
                2 * 2 * 6,
                [-2.0, 0.0, -3.0],
                [2.0, 0.0, 3.0],
            ),
            (
                PrimitiveGeometry::Cylinder {
                    radius: 2.0,
                    height: 6.0,
                    segments: 8,
                },
                8 * 12,
                [-2.0, -3.0, -2.0],
                [2.0, 3.0, 2.0],
            ),
            (
                PrimitiveGeometry::Cone {
                    radius: 2.0,
                    height: 6.0,
                    segments: 8,
                },
                8 * 6,
                [-2.0, -3.0, -2.0],
                [2.0, 3.0, 2.0],
            ),
            (
                PrimitiveGeometry::Capsule {
                    radius: 2.0,
                    height: 6.0,
                    segments: 8,
                    rings: 8,
                },
                432,
                [-2.0, -3.0, -2.0],
                [2.0, 3.0, 2.0],
            ),
            (
                PrimitiveGeometry::Wedge {
                    size: [4.0, 2.0, 6.0],
                },
                24,
                [-2.0, -1.0, -3.0],
                [2.0, 1.0, 3.0],
            ),
        ];
        for (geometry, expected_indices, bounds_min, bounds_max) in cases {
            let asset = PrimitiveAssetNode {
                id: geometry.shape_name().to_string(),
                geometry,
                color: [1.0; 4],
                material: None,
                material_definition: None,
                bevel_radius: 0.0,
                bevel_segments: 0,
                material_seed: None,
                collision: Default::default(),
            };
            let mesh = generate_primitive_mesh(&asset);
            assert_eq!(
                mesh.indices.len(),
                expected_indices,
                "{}",
                asset.geometry.shape_name()
            );
            assert_eq!(mesh.bounds_min, bounds_min);
            assert_eq!(mesh.bounds_max, bounds_max);
            assert_eq!(mesh.normals.len(), mesh.positions.len());
            assert_eq!(mesh.texcoords.len(), mesh.positions.len());
            assert!(
                mesh.indices
                    .iter()
                    .all(|index| (*index as usize) < mesh.positions.len())
            );
            assert!(mesh.normals.iter().flatten().all(|normal| {
                let length =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                (length - 1.0).abs() < 1.0e-4
            }));
            assert!(
                mesh.texcoords
                    .iter()
                    .flatten()
                    .all(|uv| uv.iter().all(|value| (0.0..=1.0).contains(value)))
            );
            for triangle in mesh.indices.chunks_exact(3) {
                let a = mesh.positions[triangle[0] as usize];
                let b = mesh.positions[triangle[1] as usize];
                let c = mesh.positions[triangle[2] as usize];
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let face = [
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                ];
                let normal = mesh.normals[triangle[0] as usize].unwrap();
                let alignment = face[0] * normal[0] + face[1] * normal[1] + face[2] * normal[2];
                assert!(
                    alignment >= -1.0e-5,
                    "{} winding",
                    asset.geometry.shape_name()
                );
            }
        }
    }
}
