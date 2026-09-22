// =========================================
// =========================================
// crates/motionloom/src/weaver/geometry/mod.rs

use super::{WeaverError, scene::Snapshot};
use crate::world::gltf_loader::GlbAlphaMode;
use std::collections::HashMap;

#[derive(Clone)]
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
    triangle_sources: Vec<u32>,
}

struct Triangle {
    data: [[f32; 4]; 16],
    min: [f32; 3],
    max: [f32; 3],
    source: u32,
}

// Binned surface-area-heuristic BVH. Median splits left large scene-spanning
// triangles (sky domes, ground planes) overlapping both children, which made
// traversal visit most of the tree; SAH isolates them instead.
const SAH_BINS: usize = 12;
const MAX_LEAF: usize = 12;

fn surface_area(min: [f32; 3], max: [f32; 3]) -> f32 {
    let d = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
}

fn build(tris: &mut [Triangle], start: usize, nodes: &mut Vec<[f32; 4]>) -> usize {
    let id = nodes.len() / 3;
    nodes.extend([[0.0; 4]; 3]);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut cmin = [f32::INFINITY; 3];
    let mut cmax = [f32::NEG_INFINITY; 3];
    for t in tris.iter() {
        for a in 0..3 {
            min[a] = min[a].min(t.min[a]);
            max[a] = max[a].max(t.max[a]);
            let c = (t.min[a] + t.max[a]) * 0.5;
            cmin[a] = cmin[a].min(c);
            cmax[a] = cmax[a].max(c);
        }
    }
    nodes[id * 3] = [min[0], min[1], min[2], 0.0];
    nodes[id * 3 + 1] = [max[0], max[1], max[2], 0.0];
    let extent = [cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]];
    let axis = (0..3)
        .max_by(|&a, &b| extent[a].total_cmp(&extent[b]))
        .unwrap();
    if tris.len() <= 4 || extent[axis] <= 0.0 {
        nodes[id * 3 + 2] = [start as f32, tris.len() as f32, 0.0, 0.0];
        return id;
    }
    let scale = SAH_BINS as f32 / extent[axis];
    let bin_of = |t: &Triangle| -> usize {
        ((((t.min[axis] + t.max[axis]) * 0.5 - cmin[axis]) * scale) as isize)
            .clamp(0, SAH_BINS as isize - 1) as usize
    };
    let mut bin_min = [[f32::INFINITY; 3]; SAH_BINS];
    let mut bin_max = [[f32::NEG_INFINITY; 3]; SAH_BINS];
    let mut bin_count = [0usize; SAH_BINS];
    for t in tris.iter() {
        let b = bin_of(t);
        bin_count[b] += 1;
        for a in 0..3 {
            bin_min[b][a] = bin_min[b][a].min(t.min[a]);
            bin_max[b][a] = bin_max[b][a].max(t.max[a]);
        }
    }
    // Sweep the 11 candidate splits and rank by area * count on both sides.
    let mut best_split = 0usize;
    let mut best_cost = f32::INFINITY;
    let mut left_min = [f32::INFINITY; 3];
    let mut left_max = [f32::NEG_INFINITY; 3];
    let mut left_count = 0usize;
    for split in 0..SAH_BINS - 1 {
        if bin_count[split] > 0 {
            for a in 0..3 {
                left_min[a] = left_min[a].min(bin_min[split][a]);
                left_max[a] = left_max[a].max(bin_max[split][a]);
            }
            left_count += bin_count[split];
        }
        let right_count = tris.len() - left_count;
        if left_count == 0 || right_count == 0 {
            continue;
        }
        let mut right_min = [f32::INFINITY; 3];
        let mut right_max = [f32::NEG_INFINITY; 3];
        for b in split + 1..SAH_BINS {
            if bin_count[b] > 0 {
                for a in 0..3 {
                    right_min[a] = right_min[a].min(bin_min[b][a]);
                    right_max[a] = right_max[a].max(bin_max[b][a]);
                }
            }
        }
        let cost = surface_area(left_min, left_max) * left_count as f32
            + surface_area(right_min, right_max) * right_count as f32;
        if cost < best_cost {
            best_cost = cost;
            best_split = split;
        }
    }
    let leaf_cost = surface_area(min, max) * tris.len() as f32;
    if tris.len() <= MAX_LEAF && best_cost >= leaf_cost {
        nodes[id * 3 + 2] = [start as f32, tris.len() as f32, 0.0, 0.0];
        return id;
    }
    // Partition by bin; degenerate bins fall back to the median.
    let mut mid = 0usize;
    for count in bin_count.iter().take(best_split + 1) {
        mid += count;
    }
    if mid == 0 || mid >= tris.len() {
        mid = tris.len() / 2;
        tris.select_nth_unstable_by(mid, |a, b| {
            (a.min[axis] + a.max[axis]).total_cmp(&(b.min[axis] + b.max[axis]))
        });
    } else {
        tris.select_nth_unstable_by_key(mid, bin_of);
    }
    let (l, r) = tris.split_at_mut(mid);
    let left = build(l, start, nodes);
    let right = build(r, start + mid, nodes);
    nodes[id * 3][3] = left as f32;
    nodes[id * 3 + 1][3] = right as f32;
    id
}

fn refit(node: usize, nodes: &mut [[f32; 4]], tris: &[Triangle]) -> ([f32; 3], [f32; 3]) {
    let at = node * 3;
    let start = nodes[at + 2][0] as usize;
    let count = nodes[at + 2][1] as usize;
    let (mut min, mut max) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    if count > 0 {
        for triangle in &tris[start..start + count] {
            for axis in 0..3 {
                min[axis] = min[axis].min(triangle.min[axis]);
                max[axis] = max[axis].max(triangle.max[axis]);
            }
        }
    } else {
        let left = nodes[at][3] as usize;
        let right = nodes[at + 1][3] as usize;
        let (left_min, left_max) = refit(left, nodes, tris);
        let (right_min, right_max) = refit(right, nodes, tris);
        for axis in 0..3 {
            min[axis] = left_min[axis].min(right_min[axis]);
            max[axis] = left_max[axis].max(right_max[axis]);
        }
    }
    nodes[at][..3].copy_from_slice(&min);
    nodes[at + 1][..3].copy_from_slice(&max);
    (min, max)
}

fn push_rgba(pixels: &mut Vec<u32>, rgba: &[u8]) {
    pixels.extend(
        rgba.chunks_exact(4)
            .map(|p| u32::from_le_bytes([p[0], p[1], p[2], p[3]])),
    );
}

fn srgb_to_linear(value: u8) -> f32 {
    let c = value as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> u8 {
    let c = value.clamp(0.0, 1.0);
    let encoded = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Box-filtered mip chain pushed contiguously; returns the level count.
fn push_mip_chain(pixels: &mut Vec<u32>, rgba: &[u8], width: u32, height: u32, srgb: bool) -> u32 {
    push_rgba(pixels, rgba);
    let mut levels = 1u32;
    let mut src = rgba.to_vec();
    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        let nw = (w.div_ceil(2)).max(1);
        let nh = (h.div_ceil(2)).max(1);
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 4];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = (x * 2 + dx).min(w - 1);
                        let sy = (y * 2 + dy).min(h - 1);
                        let offset = ((sy * w + sx) * 4) as usize;
                        for channel in 0..4 {
                            let value = src[offset + channel];
                            acc[channel] += if srgb && channel < 3 {
                                srgb_to_linear(value)
                            } else {
                                value as f32 / 255.0
                            };
                        }
                    }
                }
                let offset = ((y * nw + x) * 4) as usize;
                for channel in 0..4 {
                    let value = acc[channel] * 0.25;
                    next[offset + channel] = if srgb && channel < 3 {
                        linear_to_srgb(value)
                    } else {
                        (value * 255.0).round().clamp(0.0, 255.0) as u8
                    };
                }
            }
        }
        push_rgba(pixels, &next);
        src = next;
        w = nw;
        h = nh;
        levels += 1;
    }
    levels
}

pub(crate) fn pack(
    snapshot: &Snapshot,
    mipmaps: bool,
    allow_transmission_stopgap: bool,
) -> Result<PackedScene, WeaverError> {
    pack_with_bvh(snapshot, mipmaps, allow_transmission_stopgap, None).map(|value| value.0)
}

pub(crate) fn pack_refit(
    snapshot: &Snapshot,
    mipmaps: bool,
    allow_transmission_stopgap: bool,
    previous: &PackedScene,
) -> Result<(PackedScene, bool), WeaverError> {
    pack_with_bvh(
        snapshot,
        mipmaps,
        allow_transmission_stopgap,
        Some(previous),
    )
}

fn pack_with_bvh(
    snapshot: &Snapshot,
    mipmaps: bool,
    allow_transmission_stopgap: bool,
    previous: Option<&PackedScene>,
) -> Result<(PackedScene, bool), WeaverError> {
    let mut tris = Vec::new();
    let mut materials = Vec::new();
    let mut pixels = Vec::new();
    let mut textures = HashMap::<(usize, u32, u32, bool), [f32; 4]>::new();
    let mut source = 0u32;
    for (mi, mesh) in snapshot.meshes.iter().enumerate() {
        let m = &mesh.material;
        if (m.transmission_factor > 0.0 && !allow_transmission_stopgap)
            || m.unlit
            || m.specular_glossiness
        {
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
        for (slot, t) in mesh.textures.iter().take(5).enumerate() {
            if t.width == 0
                || t.height == 0
                || t.rgba.len() as u64 != t.width as u64 * t.height as u64 * 4
            {
                return Err(WeaverError::Scene(
                    "invalid texture dimensions/payload".into(),
                ));
            }
            // Slots 0 (base color) and 3 (emissive) are sRGB-encoded; averaging
            // their mips must happen in linear space before re-encoding.
            let srgb = matches!(slot, 0 | 3);
            let key = (t.rgba.as_ptr() as usize, t.width, t.height, srgb);
            let desc = if let Some(desc) = textures.get(&key) {
                *desc
            } else {
                let offset = u32::try_from(pixels.len()).map_err(|_| {
                    WeaverError::Unsupported("texture pack exceeds u32 index capacity".into())
                })?;
                let levels = if mipmaps {
                    push_mip_chain(&mut pixels, &t.rgba, t.width, t.height, srgb)
                } else {
                    push_rgba(&mut pixels, &t.rgba);
                    1
                };
                // Preserve the full u32 address in float-backed material storage;
                // w carries the mip level count.
                let desc = [
                    f32::from_bits(offset),
                    t.width as f32,
                    t.height as f32,
                    levels as f32,
                ];
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
        materials.push([
            crate::world::gltf_loader::material_channel_remap_code(m),
            f32::from(
                snapshot
                    .primary_camera_visibility
                    .get(mi)
                    .copied()
                    .unwrap_or(true),
            ),
            f32::from(m.receive_caustics),
            0.0,
        ]);
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
            tris.push(Triangle {
                data,
                min,
                max,
                source,
            });
            source += 1;
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
    let mut refitted = false;
    let mut data = Vec::new();
    if let Some(previous) = previous.filter(|scene| scene.triangle_sources.len() == tris.len()) {
        let mut by_source = tris.into_iter().map(Some).collect::<Vec<_>>();
        let mut ordered = Vec::with_capacity(by_source.len());
        for source in &previous.triangle_sources {
            let Some(triangle) = by_source.get_mut(*source as usize).and_then(Option::take) else {
                return Err(WeaverError::Scene(
                    "invalid cached BVH triangle permutation".into(),
                ));
            };
            ordered.push(triangle);
        }
        tris = ordered;
        data.extend_from_slice(&previous.data[..previous.triangle_offset as usize]);
        refit(0, &mut data, &tris);
        refitted = true;
    } else {
        build(&mut tris, 0, &mut data);
    }
    let triangle_offset = data.len() as u32;
    let triangle_sources = tris.iter().map(|triangle| triangle.source).collect();
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
                    // Raw u32 bit pattern; the shader bitcasts it back.
                    f32::from_bits((triangle_offset as usize + i * 16) as u32),
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
    Ok((
        PackedScene {
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
            triangle_sources,
        },
        refitted,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_packs_full_levels() {
        // 4x2 red/green checkerboard; sRGB mips average in linear space.
        let mut rgba = Vec::new();
        for y in 0..2 {
            for x in 0..4 {
                let c = if (x + y) % 2 == 0 {
                    [255u8, 0, 0, 255]
                } else {
                    [0u8, 255, 0, 255]
                };
                rgba.extend_from_slice(&c);
            }
        }
        let mut pixels = Vec::new();
        let levels = push_mip_chain(&mut pixels, &rgba, 4, 2, true);
        assert_eq!(levels, 3);
        assert_eq!(pixels.len(), 4 * 2 + 2 * 1 + 1 * 1);
        let expected = linear_to_srgb(0.5);
        let bytes = pixels[pixels.len() - 1].to_le_bytes();
        assert!(bytes[0].abs_diff(expected) <= 1, "{bytes:?} vs {expected}");
        assert!(bytes[1].abs_diff(expected) <= 1, "{bytes:?} vs {expected}");
    }

    #[test]
    fn mip_chain_halves_odd_dimensions() {
        let rgba = vec![128u8; 3 * 3 * 4];
        let mut pixels = Vec::new();
        let levels = push_mip_chain(&mut pixels, &rgba, 3, 3, false);
        assert_eq!(levels, 3);
        assert_eq!(pixels.len(), 9 + 4 + 1);
    }

    #[test]
    fn mip_chain_disabled_keeps_base_level() {
        let rgba = vec![64u8; 2 * 2 * 4];
        let mut pixels = Vec::new();
        push_rgba(&mut pixels, &rgba);
        assert_eq!(pixels.len(), 4);
    }

    #[test]
    fn refit_preserves_tree_and_updates_bounds() {
        let mut triangles = (0..8)
            .map(|source| Triangle {
                data: [[0.0; 4]; 16],
                min: [source as f32, 0.0, 0.0],
                max: [source as f32 + 0.5, 1.0, 1.0],
                source,
            })
            .collect::<Vec<_>>();
        let mut nodes = Vec::new();
        build(&mut triangles, 0, &mut nodes);
        let topology = nodes
            .chunks_exact(3)
            .map(|node| [node[0][3], node[1][3], node[2][0], node[2][1]])
            .collect::<Vec<_>>();
        for triangle in &mut triangles {
            triangle.min[1] += 4.0;
            triangle.max[1] += 4.0;
        }
        refit(0, &mut nodes, &triangles);
        assert_eq!(&nodes[0][..3], &[0.0, 4.0, 0.0]);
        assert_eq!(&nodes[1][..3], &[7.5, 5.0, 1.0]);
        let updated_topology = nodes
            .chunks_exact(3)
            .map(|node| [node[0][3], node[1][3], node[2][0], node[2][1]])
            .collect::<Vec<_>>();
        assert_eq!(updated_topology, topology);
    }
}
