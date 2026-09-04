// =========================================
// PROCEDURAL VEGETATION V1
// =========================================
// crates/motionloom/src/world/vegetation/mod.rs

use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use crate::dsl::{
    MaterialAssetNode, PrimitiveAssetNode, PrimitiveCollisionNode, PrimitiveGeometry,
    VegetationAssetNode, VegetationKind, VegetationLod,
};
use crate::world::gltf_loader::GlbMeshData;
use crate::world::primitive::{MeshBuilder, PrimitiveTextureSet};

#[derive(Clone, Debug, Default)]
pub struct VegetationTextureSet {
    pub primary: PrimitiveTextureSet,
    pub secondary: Option<PrimitiveTextureSet>,
}

/// Generate deterministic procedural vegetation into the same retained mesh
/// representation used by primitives, terrain, and imported GLB models.
pub fn generate_vegetation_mesh_textured(
    asset: &VegetationAssetNode,
    textures: VegetationTextureSet,
) -> GlbMeshData {
    let detail = Detail::for_lod(asset.lod);
    let mut primary = MeshBuilder::default();
    let mut secondary = MeshBuilder::default();
    let mut random = StableRandom::new(asset.seed);
    match asset.kind {
        VegetationKind::Tree => generate_tree(
            asset,
            detail,
            &mut random,
            &mut primary,
            Some(&mut secondary),
            false,
        ),
        VegetationKind::Deadwood => {
            generate_tree(asset, detail, &mut random, &mut primary, None, true)
        }
        VegetationKind::Shrub => {
            generate_shrub(asset, detail, &mut random, &mut primary, &mut secondary)
        }
        VegetationKind::Grass => generate_grass(asset, detail, &mut random, &mut primary),
        VegetationKind::Flower => {
            generate_flowers(asset, detail, &mut random, &mut primary, &mut secondary)
        }
        VegetationKind::Fern => generate_fern(asset, detail, &mut random, &mut primary),
    }

    let bounds = vegetation_bounds(asset);
    let (primary_material, secondary_material) = vegetation_materials(asset);
    let mut mesh = finish_part(
        primary,
        asset,
        primary_material,
        textures.primary,
        bounds,
        "primary",
    );
    if let (Some(material), Some(texture_set)) = (secondary_material, textures.secondary) {
        let part = finish_part(
            secondary,
            asset,
            Some(material),
            texture_set,
            bounds,
            "secondary",
        );
        append_mesh(&mut mesh, part);
    }
    mesh.path = vegetation_cache_key(asset);
    mesh.bounds_min = bounds.0;
    mesh.bounds_max = bounds.1;
    mesh
}

pub fn vegetation_cache_key(asset: &VegetationAssetNode) -> PathBuf {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    asset.kind.hash(&mut hasher);
    asset.height.to_bits().hash(&mut hasher);
    asset.material.hash(&mut hasher);
    asset.stem_material.hash(&mut hasher);
    asset.trunk_material.hash(&mut hasher);
    asset.foliage_material.hash(&mut hasher);
    format!("{:?}", asset.material_definition).hash(&mut hasher);
    format!("{:?}", asset.stem_material_definition).hash(&mut hasher);
    format!("{:?}", asset.trunk_material_definition).hash(&mut hasher);
    format!("{:?}", asset.foliage_material_definition).hash(&mut hasher);
    asset.density.hash(&mut hasher);
    asset.branch_levels.hash(&mut hasher);
    asset.seed.hash(&mut hasher);
    asset.lod.hash(&mut hasher);
    asset.wind.hash(&mut hasher);
    PathBuf::from(format!("motionloom-vegetation-{:016x}", hasher.finish()))
}

pub fn vegetation_bounds(asset: &VegetationAssetNode) -> ([f32; 3], [f32; 3]) {
    let radius = match asset.kind {
        VegetationKind::Tree => asset.height * 0.52,
        VegetationKind::Shrub => asset.height * 0.72,
        VegetationKind::Grass | VegetationKind::Flower | VegetationKind::Fern => {
            asset.height * 0.65
        }
        VegetationKind::Deadwood => asset.height * 0.36,
    };
    ([-radius, 0.0, -radius], [radius, asset.height, radius])
}

pub fn vegetation_collision_radius(asset: &VegetationAssetNode) -> f32 {
    match asset.kind {
        VegetationKind::Tree => asset.height * 0.065,
        VegetationKind::Deadwood => asset.height * 0.09,
        _ => 0.0,
    }
}

pub(crate) fn vegetation_collision_triangles(
    asset: &VegetationAssetNode,
) -> Vec<crate::world::model_inspection::EnvironmentWalkableTriangle> {
    if !matches!(asset.kind, VegetationKind::Tree | VegetationKind::Deadwood) {
        return Vec::new();
    }
    let collider_height = asset.height
        * match asset.kind {
            VegetationKind::Tree => 0.72,
            VegetationKind::Deadwood => 0.9,
            _ => 0.0,
        };
    let primitive = PrimitiveAssetNode {
        id: format!("{}-collision", asset.id),
        geometry: PrimitiveGeometry::Cylinder {
            radius: vegetation_collision_radius(asset),
            height: collider_height,
            segments: 10,
        },
        color: [1.0; 4],
        material: None,
        material_definition: None,
        bevel_radius: 0.0,
        bevel_segments: 0,
        material_seed: None,
        collision: PrimitiveCollisionNode::default(),
        modifiers: Vec::new(),
        mesh_build: Default::default(),
        lod: Default::default(),
    };
    let mesh = crate::world::primitive::generate_primitive_mesh(&primitive);
    let mut triangles = crate::world::model_inspection::environment_collision_triangles(&mesh);
    for triangle in &mut triangles {
        for point in &mut triangle.points {
            point[1] += collider_height * 0.5;
        }
    }
    triangles
}

pub(crate) fn vegetation_materials(
    asset: &VegetationAssetNode,
) -> (Option<MaterialAssetNode>, Option<MaterialAssetNode>) {
    match asset.kind {
        VegetationKind::Tree | VegetationKind::Shrub => (
            asset.trunk_material_definition.clone(),
            asset.foliage_material_definition.clone(),
        ),
        VegetationKind::Flower => (
            asset.material_definition.clone(),
            asset
                .stem_material_definition
                .clone()
                .or_else(|| asset.material_definition.clone()),
        ),
        VegetationKind::Grass | VegetationKind::Fern => (asset.material_definition.clone(), None),
        VegetationKind::Deadwood => (asset.trunk_material_definition.clone(), None),
    }
}

fn finish_part(
    builder: MeshBuilder,
    asset: &VegetationAssetNode,
    material: Option<MaterialAssetNode>,
    textures: PrimitiveTextureSet,
    bounds: ([f32; 3], [f32; 3]),
    label: &str,
) -> GlbMeshData {
    let primitive = vegetation_surface_primitive(asset, material, label);
    let mut mesh = builder.finish_with_bounds(&primitive, textures, bounds);
    if let Some(material) = mesh.materials.first_mut() {
        material.name = Some(format!("MotionLoom Vegetation {} {label}", asset.id));
    }
    mesh.mesh_names = vec![Some(format!("MotionLoom Vegetation {} {label}", asset.id))];
    mesh
}

pub(crate) fn vegetation_surface_primitive(
    asset: &VegetationAssetNode,
    material: Option<MaterialAssetNode>,
    label: &str,
) -> PrimitiveAssetNode {
    PrimitiveAssetNode {
        id: format!("{}-{label}", asset.id),
        geometry: PrimitiveGeometry::Plane {
            size: [asset.height, asset.height],
            segments: 1,
        },
        color: [1.0; 4],
        material: material.as_ref().map(|value| value.id.clone()),
        material_definition: material,
        bevel_radius: 0.0,
        bevel_segments: 0,
        material_seed: Some(asset.seed),
        collision: PrimitiveCollisionNode::default(),
        modifiers: Vec::new(),
        mesh_build: Default::default(),
        lod: Default::default(),
    }
}

fn append_mesh(target: &mut GlbMeshData, mut source: GlbMeshData) {
    if source.positions.is_empty() {
        return;
    }
    let vertex_offset = target.positions.len() as u32;
    let material_offset = target.materials.len();
    let texture_offset = target.textures.len();
    let mesh_offset = target.mesh_names.len();
    for material in &mut source.materials {
        material.base_color_texture = material
            .base_color_texture
            .map(|value| value + texture_offset);
        material.metallic_roughness_texture = material
            .metallic_roughness_texture
            .map(|value| value + texture_offset);
        material.normal_texture = material.normal_texture.map(|value| value + texture_offset);
        material.emissive_texture = material
            .emissive_texture
            .map(|value| value + texture_offset);
    }
    for triangle in &mut source.triangles {
        triangle.indices = triangle.indices.map(|value| value + vertex_offset);
        triangle.material = triangle.material.map(|value| value + material_offset);
        triangle.mesh = triangle.mesh.map(|value| value + mesh_offset);
    }
    target.positions.extend(source.positions);
    target.normals.extend(source.normals);
    target.texcoords.extend(source.texcoords);
    target.colors.extend(source.colors);
    target.joints.extend(source.joints);
    target.weights.extend(source.weights);
    target.indices.extend(
        source
            .indices
            .into_iter()
            .map(|value| value + vertex_offset),
    );
    target.triangles.extend(source.triangles);
    target.materials.extend(source.materials);
    target.textures.extend(source.textures);
    target.mesh_names.extend(source.mesh_names);
}

#[derive(Clone, Copy)]
struct Detail {
    radial_segments: u32,
    curve_segments: u32,
    density_divisor: u32,
    branch_level_cap: u32,
}

impl Detail {
    fn for_lod(lod: VegetationLod) -> Self {
        match lod {
            VegetationLod::Full | VegetationLod::Auto => Self {
                radial_segments: 10,
                curve_segments: 5,
                density_divisor: 1,
                branch_level_cap: 5,
            },
            VegetationLod::Half => Self {
                radial_segments: 7,
                curve_segments: 3,
                density_divisor: 2,
                branch_level_cap: 3,
            },
            VegetationLod::Quarter => Self {
                radial_segments: 5,
                curve_segments: 2,
                density_divisor: 4,
                branch_level_cap: 1,
            },
        }
    }

    fn density(self, authored: u32) -> u32 {
        authored.div_ceil(self.density_divisor).max(1)
    }
}

fn generate_tree(
    asset: &VegetationAssetNode,
    detail: Detail,
    random: &mut StableRandom,
    trunk: &mut MeshBuilder,
    mut foliage: Option<&mut MeshBuilder>,
    deadwood: bool,
) {
    let height = asset.height;
    let base_radius = vegetation_collision_radius(asset).max(height * 0.045);
    let mut trunk_points = Vec::new();
    for segment in 0..=detail.curve_segments {
        let t = segment as f32 / detail.curve_segments as f32;
        let sway = t * t * height * 0.035;
        trunk_points.push([sway * random.signed(), height * t, sway * random.signed()]);
    }
    for segment in 0..trunk_points.len() - 1 {
        let t0 = segment as f32 / (trunk_points.len() - 1) as f32;
        let t1 = (segment + 1) as f32 / (trunk_points.len() - 1) as f32;
        add_tapered_tube(
            trunk,
            trunk_points[segment],
            trunk_points[segment + 1],
            base_radius * (1.0 - t0 * 0.78),
            base_radius * (1.0 - t1 * 0.78),
            detail.radial_segments,
        );
    }
    let levels = asset.branch_levels.min(detail.branch_level_cap);
    let branch_count = (4 + levels * 2).max(2);
    let mut tips = Vec::new();
    for branch in 0..branch_count {
        let t = 0.3 + 0.58 * (branch as f32 / branch_count.max(1) as f32);
        let angle = branch as f32 * 2.399_963_1 + random.range(-0.3, 0.3);
        let start = [
            height * 0.02 * random.signed(),
            height * t,
            height * 0.02 * random.signed(),
        ];
        let length = height * random.range(0.2, 0.34) * (1.12 - t * 0.3);
        let end = [
            start[0] + angle.cos() * length,
            start[1] + length * random.range(0.3, 0.62),
            start[2] + angle.sin() * length,
        ];
        add_tapered_tube(
            trunk,
            start,
            end,
            base_radius * (1.0 - t) * 0.72,
            base_radius * 0.08,
            detail.radial_segments.saturating_sub(2).max(4),
        );
        tips.push(end);
        if levels >= 2 {
            for child in 0..2 {
                let child_angle = angle + (child as f32 - 0.5) * random.range(0.75, 1.2);
                let child_end = [
                    end[0] + child_angle.cos() * length * 0.45,
                    end[1] + length * random.range(0.18, 0.34),
                    end[2] + child_angle.sin() * length * 0.45,
                ];
                add_tapered_tube(
                    trunk,
                    end,
                    child_end,
                    base_radius * 0.09,
                    base_radius * 0.025,
                    detail.radial_segments.saturating_sub(3).max(4),
                );
                tips.push(child_end);
            }
        }
    }
    if !deadwood && let Some(foliage) = foliage.as_mut() {
        let count = detail.density(asset.density);
        for index in 0..count {
            let anchor = tips[index as usize % tips.len().max(1)];
            let spread = height * 0.13;
            let center = [
                anchor[0] + random.range(-spread, spread),
                (anchor[1] + random.range(-spread * 0.35, spread * 0.65)).min(height),
                anchor[2] + random.range(-spread, spread),
            ];
            let width = height * random.range(0.1, 0.18);
            add_crossed_cards(
                foliage,
                center,
                width,
                width * 1.35,
                random.range(0.0, 6.283),
                random_atlas_cell(random, 4, 4),
            );
        }
    }
}

fn generate_shrub(
    asset: &VegetationAssetNode,
    detail: Detail,
    random: &mut StableRandom,
    trunk: &mut MeshBuilder,
    foliage: &mut MeshBuilder,
) {
    let stems = detail.density(asset.density).clamp(4, 32);
    for index in 0..stems {
        let angle = index as f32 * 2.399_963_1 + random.range(-0.35, 0.35);
        let length = asset.height * random.range(0.62, 1.0);
        let end = [
            angle.cos() * asset.height * random.range(0.25, 0.55),
            length,
            angle.sin() * asset.height * random.range(0.25, 0.55),
        ];
        add_tapered_tube(
            trunk,
            [0.0, 0.0, 0.0],
            end,
            asset.height * 0.025,
            asset.height * 0.006,
            detail.radial_segments.saturating_sub(3).max(4),
        );
        add_crossed_cards(
            foliage,
            end,
            asset.height * random.range(0.22, 0.36),
            asset.height * random.range(0.28, 0.45),
            angle,
            random_atlas_cell(random, 4, 4),
        );
    }
}

fn generate_grass(
    asset: &VegetationAssetNode,
    detail: Detail,
    random: &mut StableRandom,
    leaves: &mut MeshBuilder,
) {
    for index in 0..detail.density(asset.density) {
        let angle = index as f32 * 2.399_963_1 + random.range(-0.4, 0.4);
        let radius = asset.height * random.range(0.0, 0.42);
        let origin = [angle.cos() * radius, 0.0, angle.sin() * radius];
        let blade_height = asset.height * random.range(0.55, 1.0);
        let bend = [
            angle.cos() * asset.height * random.range(0.08, 0.28),
            0.0,
            angle.sin() * asset.height * random.range(0.08, 0.28),
        ];
        add_curved_ribbon(
            leaves,
            origin,
            blade_height,
            asset.height * random.range(0.025, 0.055),
            bend,
            detail.curve_segments,
            random_atlas_cell(random, 4, 4),
        );
    }
}

fn generate_flowers(
    asset: &VegetationAssetNode,
    detail: Detail,
    random: &mut StableRandom,
    petals: &mut MeshBuilder,
    stems: &mut MeshBuilder,
) {
    for index in 0..detail.density(asset.density) {
        let angle = index as f32 * 2.399_963_1;
        let radius = asset.height * random.range(0.0, 0.48);
        let flower_height = asset.height * random.range(0.55, 1.0);
        let origin = [angle.cos() * radius, 0.0, angle.sin() * radius];
        let top = [
            origin[0] + random.signed() * asset.height * 0.05,
            flower_height,
            origin[2] + random.signed() * asset.height * 0.05,
        ];
        add_tapered_tube(
            stems,
            origin,
            top,
            asset.height * 0.012,
            asset.height * 0.006,
            detail.radial_segments.saturating_sub(4).max(4),
        );
        add_crossed_cards(
            petals,
            top,
            asset.height * 0.13,
            asset.height * 0.13,
            random.range(0.0, 6.283),
            random_atlas_cell(random, 4, 4),
        );
    }
}

fn generate_fern(
    asset: &VegetationAssetNode,
    detail: Detail,
    random: &mut StableRandom,
    fronds: &mut MeshBuilder,
) {
    let count = detail.density(asset.density).max(5);
    for index in 0..count {
        let angle = index as f32 * std::f32::consts::TAU / count as f32 + random.range(-0.16, 0.16);
        let reach = asset.height * random.range(0.55, 0.82);
        add_fern_frond(
            fronds,
            angle,
            reach,
            asset.height * random.range(0.12, 0.2),
            detail.curve_segments + 2,
            random_atlas_cell(random, 5, 2),
        );
    }
}

fn add_tapered_tube(
    builder: &mut MeshBuilder,
    start: [f32; 3],
    end: [f32; 3],
    start_radius: f32,
    end_radius: f32,
    segments: u32,
) {
    let axis = normalize(sub(end, start));
    let reference = if axis[1].abs() < 0.92 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize(cross(axis, reference));
    let bitangent = normalize(cross(axis, tangent));
    let mut rings = [Vec::new(), Vec::new()];
    for (ring_index, (center, radius)) in [(start, start_radius), (end, end_radius)]
        .into_iter()
        .enumerate()
    {
        for segment in 0..segments {
            let angle = segment as f32 * std::f32::consts::TAU / segments as f32;
            let normal = add(scale(tangent, angle.cos()), scale(bitangent, angle.sin()));
            rings[ring_index].push(builder.vertex(
                add(center, scale(normal, radius.max(0.0001))),
                normal,
                [segment as f32 / segments as f32, ring_index as f32],
            ));
        }
    }
    for segment in 0..segments as usize {
        let next = (segment + 1) % segments as usize;
        builder.triangle(rings[0][segment], rings[1][segment], rings[1][next]);
        builder.triangle(rings[0][segment], rings[1][next], rings[0][next]);
    }
}

fn add_crossed_cards(
    builder: &mut MeshBuilder,
    center: [f32; 3],
    width: f32,
    height: f32,
    yaw: f32,
    uv: [f32; 4],
) {
    add_vertical_card(builder, center, width, height, yaw, uv);
    add_vertical_card(
        builder,
        center,
        width,
        height,
        yaw + std::f32::consts::FRAC_PI_2,
        uv,
    );
}

fn add_vertical_card(
    builder: &mut MeshBuilder,
    center: [f32; 3],
    width: f32,
    height: f32,
    yaw: f32,
    uv: [f32; 4],
) {
    let right = [yaw.cos() * width * 0.5, 0.0, yaw.sin() * width * 0.5];
    let up = [0.0, height * 0.5, 0.0];
    let normal = normalize(cross(right, up));
    let ids = [
        builder.vertex(sub(sub(center, right), up), normal, [uv[0], uv[3]]),
        builder.vertex(add(sub(center, right), up), normal, [uv[0], uv[1]]),
        builder.vertex(add(add(center, right), up), normal, [uv[2], uv[1]]),
        builder.vertex(sub(add(center, right), up), normal, [uv[2], uv[3]]),
    ];
    builder.triangle(ids[0], ids[1], ids[2]);
    builder.triangle(ids[0], ids[2], ids[3]);
}

fn add_curved_ribbon(
    builder: &mut MeshBuilder,
    origin: [f32; 3],
    height: f32,
    width: f32,
    bend: [f32; 3],
    segments: u32,
    uv: [f32; 4],
) {
    let sideways = normalize([bend[2], 0.0, -bend[0]]);
    let mut previous: Option<[u32; 2]> = None;
    for segment in 0..=segments {
        let t = segment as f32 / segments.max(1) as f32;
        let center = add(origin, add([0.0, height * t, 0.0], scale(bend, t * t)));
        let half = width * (1.0 - t * 0.88) * 0.5;
        let normal = normalize(cross(
            sideways,
            [bend[0] * 2.0 * t, height, bend[2] * 2.0 * t],
        ));
        let pair = [
            builder.vertex(
                sub(center, scale(sideways, half)),
                normal,
                [uv[0], uv[3] + (uv[1] - uv[3]) * t],
            ),
            builder.vertex(
                add(center, scale(sideways, half)),
                normal,
                [uv[2], uv[3] + (uv[1] - uv[3]) * t],
            ),
        ];
        if let Some(last) = previous {
            builder.triangle(last[0], pair[0], pair[1]);
            builder.triangle(last[0], pair[1], last[1]);
        }
        previous = Some(pair);
    }
}

fn add_fern_frond(
    builder: &mut MeshBuilder,
    angle: f32,
    reach: f32,
    width: f32,
    segments: u32,
    uv: [f32; 4],
) {
    let direction = [angle.cos(), 0.0, angle.sin()];
    let sideways = [-direction[2], 0.0, direction[0]];
    let mut previous: Option<[u32; 2]> = None;
    for segment in 0..=segments {
        let t = segment as f32 / segments.max(1) as f32;
        let center = [
            direction[0] * reach * t,
            reach * (0.18 + 0.64 * t - 0.46 * t * t),
            direction[2] * reach * t,
        ];
        let half = width * (1.0 - t * 0.82) * 0.5;
        let pair = [
            builder.vertex(
                sub(center, scale(sideways, half)),
                [0.0, 1.0, 0.0],
                [uv[0], uv[1] + (uv[3] - uv[1]) * t],
            ),
            builder.vertex(
                add(center, scale(sideways, half)),
                [0.0, 1.0, 0.0],
                [uv[2], uv[1] + (uv[3] - uv[1]) * t],
            ),
        ];
        if let Some(last) = previous {
            builder.triangle(last[0], pair[0], pair[1]);
            builder.triangle(last[0], pair[1], last[1]);
        }
        previous = Some(pair);
    }
}

fn random_atlas_cell(random: &mut StableRandom, columns: u32, rows: u32) -> [f32; 4] {
    let column = (random.next() * columns as f32)
        .floor()
        .min(columns as f32 - 1.0);
    let row = (random.next() * rows as f32).floor().min(rows as f32 - 1.0);
    let gutter = 0.006;
    [
        column / columns as f32 + gutter,
        row / rows as f32 + gutter,
        (column + 1.0) / columns as f32 - gutter,
        (row + 1.0) / rows as f32 - gutter,
    ]
}

fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn sub(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn scale(value: [f32; 3], factor: f32) -> [f32; 3] {
    value.map(|component| component * factor)
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(value: [f32; 3]) -> [f32; 3] {
    let length = value
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if length <= 1.0e-8 {
        [0.0, 1.0, 0.0]
    } else {
        value.map(|component| component / length)
    }
}

struct StableRandom(u64);

impl StableRandom {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let value = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (value >> 40) as f32 / (1_u32 << 24) as f32
    }

    fn range(&mut self, minimum: f32, maximum: f32) -> f32 {
        minimum + (maximum - minimum) * self.next()
    }

    fn signed(&mut self) -> f32 {
        self.range(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::PrimitiveCollisionMode;

    fn asset(kind: VegetationKind, lod: VegetationLod) -> VegetationAssetNode {
        VegetationAssetNode {
            id: "plant".into(),
            kind,
            height: 4.0,
            material: Some("leaf".into()),
            material_definition: None,
            stem_material: None,
            stem_material_definition: None,
            trunk_material: Some("bark".into()),
            trunk_material_definition: None,
            foliage_material: Some("leaf".into()),
            foliage_material_definition: None,
            density: 16,
            branch_levels: 3,
            seed: 77,
            lod,
            wind: true,
            collision: PrimitiveCollisionMode::None,
        }
    }

    #[test]
    fn all_v1_kinds_generate_finite_geometry() {
        for kind in [
            VegetationKind::Tree,
            VegetationKind::Shrub,
            VegetationKind::Grass,
            VegetationKind::Flower,
            VegetationKind::Fern,
            VegetationKind::Deadwood,
        ] {
            let mesh = generate_vegetation_mesh_textured(
                &asset(kind, VegetationLod::Full),
                VegetationTextureSet::default(),
            );
            assert!(!mesh.triangles.is_empty(), "{kind:?}");
            assert!(
                mesh.positions
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
            );
            assert!(
                mesh.normals
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|value| value.is_finite())
            );
        }
    }

    #[test]
    fn seed_is_deterministic_and_lod_reduces_topology() {
        let full = generate_vegetation_mesh_textured(
            &asset(VegetationKind::Tree, VegetationLod::Full),
            VegetationTextureSet::default(),
        );
        let repeated = generate_vegetation_mesh_textured(
            &asset(VegetationKind::Tree, VegetationLod::Full),
            VegetationTextureSet::default(),
        );
        let quarter = generate_vegetation_mesh_textured(
            &asset(VegetationKind::Tree, VegetationLod::Quarter),
            VegetationTextureSet::default(),
        );
        assert_eq!(full.positions, repeated.positions);
        assert!(quarter.triangles.len() < full.triangles.len());
    }

    #[test]
    fn collision_is_coarse_and_limited_to_woody_kinds() {
        let mut tree = asset(VegetationKind::Tree, VegetationLod::Full);
        tree.collision = PrimitiveCollisionMode::Solid;
        assert!(!vegetation_collision_triangles(&tree).is_empty());
        assert!(vegetation_collision_radius(&tree) > 0.0);

        let grass = asset(VegetationKind::Grass, VegetationLod::Full);
        assert!(vegetation_collision_triangles(&grass).is_empty());
        assert_eq!(vegetation_collision_radius(&grass), 0.0);
    }
}
