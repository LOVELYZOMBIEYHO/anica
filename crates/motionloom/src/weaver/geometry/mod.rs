// =========================================
// =========================================
// crates/motionloom/src/weaver/geometry/mod.rs

use super::{WeaverError, scene::Snapshot};
use crate::world::gltf_loader::GlbAlphaMode;
use std::collections::HashMap;

pub(crate) struct PackedScene {
    pub data: Vec<[f32; 4]>,
    pub pixels: Vec<u32>,
    pub triangle_offset: u32,
    pub material_offset: u32,
    pub light_offset: u32,
    pub triangles: usize,
    pub lights: usize,
    pub emitter_offset: u32,
    pub emitter_count: u32,
    pub emitter_area: f32,
}

struct Triangle {
    data: [[f32; 4]; 16],
    min: [f32; 3],
    max: [f32; 3],
}

// Median BVH has bounded depth; all triangles remain eligible for secondary rays.
fn build(tris: &mut [Triangle], start: usize, nodes: &mut Vec<[f32; 4]>) -> usize {
    let id = nodes.len() / 3;
    nodes.extend([[0.0; 4]; 3]);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for t in tris.iter() {
        for a in 0..3 {
            min[a] = min[a].min(t.min[a]);
            max[a] = max[a].max(t.max[a]);
        }
    }
    nodes[id * 3] = [min[0], min[1], min[2], 0.0];
    nodes[id * 3 + 1] = [max[0], max[1], max[2], 0.0];
    if tris.len() <= 4 {
        nodes[id * 3 + 2] = [start as f32, tris.len() as f32, 0.0, 0.0];
    } else {
        let axis = (0..3)
            .max_by(|&a, &b| (max[a] - min[a]).total_cmp(&(max[b] - min[b])))
            .unwrap();
        let mid = tris.len() / 2;
        tris.select_nth_unstable_by(mid, |a, b| {
            (a.min[axis] + a.max[axis]).total_cmp(&(b.min[axis] + b.max[axis]))
        });
        let (l, r) = tris.split_at_mut(mid);
        let left = build(l, start, nodes);
        let right = build(r, start + mid, nodes);
        nodes[id * 3][3] = left as f32;
        nodes[id * 3 + 1][3] = right as f32;
    }
    id
}

pub(crate) fn pack(snapshot: &Snapshot) -> Result<PackedScene, WeaverError> {
    let mut tris = Vec::new();
    let mut materials = Vec::new();
    let mut pixels = Vec::new();
    let mut textures = HashMap::<(usize, u32, u32), [f32; 4]>::new();
    for (mi, mesh) in snapshot.meshes.iter().enumerate() {
        let m = &mesh.material;
        if m.transmission_factor > 0.0 || m.unlit || m.specular_glossiness {
            return Err(WeaverError::Unsupported(format!(
                "material {:?}: transmission/unlit/specular-glossiness requires the next BSDF milestone",
                m.name
            )));
        }
        materials.push([
            m.metallic_factor,
            m.roughness_factor,
            m.normal_scale,
            match m.alpha_mode {
                GlbAlphaMode::Opaque => 0.0,
                GlbAlphaMode::Mask => 1.0,
                GlbAlphaMode::Blend => 2.0,
            },
        ]);
        materials.push([
            m.emissive_factor[0],
            m.emissive_factor[1],
            m.emissive_factor[2],
            m.emissive_strength,
        ]);
        materials.push([
            m.specular_color_factor[0],
            m.specular_color_factor[1],
            m.specular_color_factor[2],
            m.specular_factor,
        ]);
        materials.push([m.alpha_cutoff, 0.0, 0.0, 0.0]);
        for t in mesh.textures.iter().take(5) {
            if t.width == 0
                || t.height == 0
                || t.rgba.len() as u64 != t.width as u64 * t.height as u64 * 4
            {
                return Err(WeaverError::Scene(
                    "invalid texture dimensions/payload".into(),
                ));
            }
            let key = (t.rgba.as_ptr() as usize, t.width, t.height);
            let desc = if let Some(desc) = textures.get(&key) {
                *desc
            } else {
                let offset = u32::try_from(pixels.len()).map_err(|_| {
                    WeaverError::Unsupported("texture pack exceeds u32 index capacity".into())
                })?;
                pixels.extend(
                    t.rgba
                        .chunks_exact(4)
                        .map(|p| u32::from_le_bytes([p[0], p[1], p[2], p[3]])),
                );
                // Preserve the full u32 address in float-backed material storage.
                let desc = [f32::from_bits(offset), t.width as f32, t.height as f32, 0.0];
                textures.insert(key, desc);
                desc
            };
            materials.push(desc);
        }
        if mesh.textures.len() < 5 {
            return Err(WeaverError::Scene(
                "resolved mesh lacks standard textures".into(),
            ));
        }
        materials.push([0.0; 4]);
        if mesh.indices.len() % 3 != 0 {
            return Err(WeaverError::Scene("incomplete triangle indices".into()));
        }
        for idx in mesh.indices.chunks_exact(3) {
            let mut data = [[0.0; 4]; 16];
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for (j, &i) in idx.iter().enumerate() {
                let i = i as usize;
                if i >= mesh.positions.len()
                    || i >= mesh.normals.len()
                    || i >= mesh.tangents.len()
                    || i >= mesh.colors.len()
                    || i >= mesh.uvs.len()
                {
                    return Err(WeaverError::Scene("invalid mesh index".into()));
                }
                let p = mesh.positions[i];
                let n = mesh.normals[i];
                let uv = mesh.uvs[i];
                if p.iter()
                    .chain(n.iter())
                    .chain(mesh.tangents[i].iter())
                    .chain(mesh.colors[i].iter())
                    .chain(uv.iter())
                    .any(|v| !v.is_finite())
                {
                    return Err(WeaverError::Scene("non-finite mesh".into()));
                }
                data[j * 5] = [p[0], p[1], p[2], 0.0];
                data[j * 5 + 1] = [n[0], n[1], n[2], 0.0];
                data[j * 5 + 2] = mesh.tangents[i];
                data[j * 5 + 3] = [uv[0], uv[1], 0.0, 0.0];
                data[j * 5 + 4] = mesh.colors[i];
                for a in 0..3 {
                    min[a] = min[a].min(p[a]);
                    max[a] = max[a].max(p[a]);
                }
            }
            data[15][0] = mi as f32;
            tris.push(Triangle { data, min, max });
        }
    }
    if tris.is_empty() {
        return Err(WeaverError::Scene("empty geometry".into()));
    }
    if pixels.len() > u32::MAX as usize {
        return Err(WeaverError::Unsupported(
            "texture pack exceeds u32 index capacity".into(),
        ));
    }
    let mut data = Vec::new();
    build(&mut tris, 0, &mut data);
    let triangle_offset = data.len() as u32;
    for t in &tris {
        data.extend(t.data);
    }
    let material_offset = data.len() as u32;
    data.extend(materials);
    let light_offset = data.len() as u32;
    for l in &snapshot.lighting.lights {
        use crate::world::WorldLightKind;
        let kind = match l.kind {
            WorldLightKind::Directional => 0.0,
            WorldLightKind::Point => 1.0,
            WorldLightKind::RectArea => 3.0,
            _ => {
                return Err(WeaverError::Unsupported(
                    "spot lights pending light sampler".into(),
                ));
            }
        };
        data.push([l.position[0], l.position[1], l.position[2], kind]);
        data.push([l.direction[0], l.direction[1], l.direction[2], l.intensity]);
        data.push([l.color[0], l.color[1], l.color[2], l.width]);
        data.push([l.height, l.range, 0.0, 0.0]);
    }
    // Area-proportional emitter sampling complements BSDF-hit emission with MIS.
    let emitter_offset = data.len() as u32;
    let mut emitter_area = 0.0f32;
    for (i, t) in tris.iter().enumerate() {
        let m = material_offset as usize + t.data[15][0] as usize * 10;
        if data[m + 1][..3].iter().any(|v| *v > 0.0) && data[m + 1][3] > 0.0 {
            let a = t.data[0];
            let b = t.data[5];
            let c = t.data[10];
            let e = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let f = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                e[1] * f[2] - e[2] * f[1],
                e[2] * f[0] - e[0] * f[2],
                e[0] * f[1] - e[1] * f[0],
            ];
            let area = 0.5 * cross.iter().map(|v| v * v).sum::<f32>().sqrt();
            if area > 1e-10 {
                emitter_area += area;
                data.push([
                    emitter_area,
                    (triangle_offset as usize + i * 16) as f32,
                    0.0,
                    0.0,
                ]);
            }
        }
    }
    let emitter_count = data.len() as u32 - emitter_offset;
    if emitter_area > 0.0 {
        for item in &mut data[emitter_offset as usize..] {
            item[0] /= emitter_area;
        }
    }
    Ok(PackedScene {
        data,
        pixels,
        triangle_offset,
        material_offset,
        light_offset,
        triangles: tris.len(),
        lights: snapshot.lighting.lights.len(),
        emitter_offset,
        emitter_count,
        emitter_area,
    })
}
