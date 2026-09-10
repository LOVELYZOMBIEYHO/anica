// =========================================
// =========================================
// crates/motionloom/src/geometry/uv.rs

use super::*;
use image::{Rgba, RgbaImage};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct UvCheckOptions {
    pub resolution: u32,
    pub padding: u32,
    /// Mesh indices intended to share one atlas. Empty keeps domains separate.
    pub atlas_meshes: Vec<usize>,
}
impl Default for UvCheckOptions {
    fn default() -> Self {
        Self {
            resolution: 1024,
            padding: 12,
            atlas_meshes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshUvReport {
    pub name: String,
    pub vertices: usize,
    pub triangles: usize,
    pub uv_source: String,
    pub non_finite_uv_vertices: usize,
    pub out_of_range_vertices: usize,
    pub degenerate_uv_triangles: usize,
    pub mirrored_triangles: usize,
    pub island_count: usize,
    pub occupied_pixels: usize,
    pub overlap_pixels: usize,
    pub padding_conflict_pixels: usize,
    pub atlas_coverage: f32,
    pub pixels_per_world_unit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvCheckReport {
    pub topology_signature: String,
    pub uv_signature: String,
    pub resolution: u32,
    pub meshes: Vec<MeshUvReport>,
    pub diagnostics: Vec<String>,
}

pub struct UvCheckOutput {
    pub report: UvCheckReport,
    /// One set per mesh; separate texture domains must not be called overlaps.
    pub images: Vec<UvDiagnosticImages>,
    pub checker: RgbaImage,
}
pub struct UvDiagnosticImages {
    pub wireframe: RgbaImage,
    pub islands: RgbaImage,
    pub overlaps: RgbaImage,
}

/// Pixel-center coverage makes overlap estimates bounded and deterministic.
/// Mirrored/repeating UVs are reported, not rejected: they may be intentional.
pub fn check_scene_uvs(
    snapshot: &GeometrySnapshot,
    options: &UvCheckOptions,
) -> Result<UvCheckOutput, GeometryError> {
    let r = options.resolution;
    if !(32..=4096).contains(&r) || options.padding > r / 4 {
        return Err(GeometryError::Invalid(
            "UV resolution must be 32..4096 and padding <= resolution/4".into(),
        ));
    }
    let mut reports = Vec::new();
    let mut images = Vec::new();
    let combined = if options.atlas_meshes.is_empty() {
        None
    } else {
        let first = *options.atlas_meshes.first().unwrap();
        let mut merged = snapshot
            .meshes
            .get(first)
            .ok_or_else(|| GeometryError::Invalid("invalid atlas mesh index".into()))?
            .clone();
        merged.name = "atlas:selected-meshes".into();
        merged.positions.clear();
        merged.uvs.clear();
        merged.indices.clear();
        merged.normals.clear();
        merged.tangents.clear();
        merged.colors.clear();
        merged.textures.clear();
        let mut seen = HashSet::new();
        for &index in &options.atlas_meshes {
            if !seen.insert(index) {
                return Err(GeometryError::Invalid("duplicate atlas mesh index".into()));
            }
            let m = snapshot
                .meshes
                .get(index)
                .ok_or_else(|| GeometryError::Invalid("invalid atlas mesh index".into()))?;
            let offset = merged.positions.len() as u32;
            merged.positions.extend(&m.positions);
            merged.uvs.extend(&m.uvs);
            merged.indices.extend(m.indices.iter().map(|i| i + offset));
        }
        Some(merged)
    };
    for mesh in snapshot.meshes.iter().chain(combined.iter()) {
        if mesh.indices.len() % 3 != 0
            || mesh
                .indices
                .iter()
                .any(|&i| i as usize >= mesh.uvs.len() || i as usize >= mesh.positions.len())
        {
            return Err(GeometryError::Invalid(format!(
                "invalid UV indices: {}",
                mesh.name
            )));
        }
        let triangles: Vec<_> = mesh
            .indices
            .chunks_exact(3)
            .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
            .collect();
        let mut parent: Vec<_> = (0..triangles.len()).collect();
        let mut edges = HashMap::new();
        // Render vertices are split for material/normals; weld UV-continuous edges.
        for (ti, t) in triangles.iter().enumerate() {
            for k in 0..3 {
                let key = |i: usize| {
                    let p = mesh.positions[i];
                    let u = mesh.uvs[i];
                    [
                        p[0].to_bits(),
                        p[1].to_bits(),
                        p[2].to_bits(),
                        u[0].to_bits(),
                        u[1].to_bits(),
                    ]
                };
                let mut a = key(t[k]);
                let mut b = key(t[(k + 1) % 3]);
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                if let Some(&other) = edges.get(&(a, b)) {
                    let root_a = root(&mut parent, ti);
                    let root_b = root(&mut parent, other);
                    parent[root_a] = root_b;
                } else {
                    edges.insert((a, b), ti);
                }
            }
        }
        let roots: Vec<_> = (0..triangles.len()).map(|i| root(&mut parent, i)).collect();
        let mut ids = HashMap::new();
        let mut island_ids = Vec::new();
        for &v in &roots {
            let next = ids.len() as u32 + 1;
            island_ids.push(*ids.entry(v).or_insert(next));
        }
        let mut coverage = vec![0u16; (r * r) as usize];
        let mut owners = vec![0u32; (r * r) as usize];
        let mut wire = RgbaImage::new(r, r);
        let mut island_image = RgbaImage::new(r, r);
        let mut overlap = RgbaImage::new(r, r);
        let mut degenerate = 0;
        let mut mirrored = 0;
        let mut uv_area = 0.0;
        let mut world_area = 0.0;
        for (ti, t) in triangles.iter().enumerate() {
            let mut p = t.map(|i| mesh.uvs[i]);
            if p.iter().flatten().any(|v| !v.is_finite()) {
                continue;
            }
            let area = edge(p[0], p[1], p[2]);
            if area.abs() < 1e-12 {
                degenerate += 1;
                continue;
            }
            if area < 0.0 {
                mirrored += 1;
                p.swap(1, 2);
            }
            uv_area += area.abs() * 0.5;
            let a = mesh.positions[t[0]];
            let b = mesh.positions[t[1]];
            let c = mesh.positions[t[2]];
            let x: [f32; 3] = std::array::from_fn(|i| b[i] - a[i]);
            let y: [f32; 3] = std::array::from_fn(|i| c[i] - a[i]);
            let cross = [
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ];
            world_area += cross.iter().map(|v| v * v).sum::<f32>().sqrt() * 0.5;
            let bounds = |axis: usize| {
                (
                    (p.iter().map(|v| v[axis]).fold(f32::INFINITY, f32::min) * r as f32)
                        .floor()
                        .clamp(0.0, r as f32) as u32,
                    (p.iter().map(|v| v[axis]).fold(f32::NEG_INFINITY, f32::max) * r as f32)
                        .ceil()
                        .clamp(0.0, r as f32) as u32,
                )
            };
            let (x0, x1) = bounds(0);
            let (y0, y1) = bounds(1);
            for y in y0..y1 {
                for x in x0..x1 {
                    let q = [(x as f32 + 0.5) / r as f32, (y as f32 + 0.5) / r as f32];
                    if (0..3).all(|k| {
                        let a = p[k];
                        let b = p[(k + 1) % 3];
                        let e = edge(a, b, q);
                        e > 0.0 || (e == 0.0 && (b[1] > a[1] || (b[1] == a[1] && b[0] < a[0])))
                    }) {
                        let i = (y * r + x) as usize;
                        coverage[i] = coverage[i].saturating_add(1);
                        owners[i] = island_ids[ti];
                    }
                }
            }
            for k in 0..3 {
                line(&mut wire, p[k], p[(k + 1) % 3], Rgba([225, 225, 225, 255]));
            }
        }
        let mut dilated = owners.clone();
        let mut conflicts = HashSet::new();
        // Expand each island footprint; conflicts are padding estimates in pixels.
        for _ in 0..options.padding {
            let previous = dilated.clone();
            for y in 0..r {
                for x in 0..r {
                    let i = (y * r + x) as usize;
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let xx = x as i32 + dx;
                        let yy = y as i32 + dy;
                        if xx < 0 || yy < 0 || xx >= r as i32 || yy >= r as i32 {
                            continue;
                        }
                        let owner = previous[(yy as u32 * r + xx as u32) as usize];
                        if owner == 0 {
                            continue;
                        }
                        if dilated[i] == 0 {
                            dilated[i] = owner;
                        } else if dilated[i] != owner {
                            conflicts.insert(i);
                        }
                    }
                }
            }
        }
        for y in 0..r {
            for x in 0..r {
                let i = (y * r + x) as usize;
                if owners[i] > 0 {
                    let v = owners[i].wrapping_mul(2654435761);
                    island_image.put_pixel(
                        x,
                        y,
                        Rgba([
                            64 + (v % 192) as u8,
                            64 + ((v >> 8) % 192) as u8,
                            64 + ((v >> 16) % 192) as u8,
                            255,
                        ]),
                    );
                }
                if coverage[i] > 1 {
                    overlap.put_pixel(x, y, Rgba([255, 35, 35, 255]));
                } else if conflicts.contains(&i) {
                    overlap.put_pixel(x, y, Rgba([255, 170, 0, 255]));
                }
            }
        }
        let occupied = coverage.iter().filter(|&&c| c > 0).count();
        reports.push(MeshUvReport {
            name: mesh.name.clone(),
            vertices: mesh.positions.len(),
            triangles: triangles.len(),
            uv_source: mesh.uv_source.clone(),
            non_finite_uv_vertices: mesh
                .uvs
                .iter()
                .filter(|u| u.iter().any(|v| !v.is_finite()))
                .count(),
            out_of_range_vertices: mesh
                .uvs
                .iter()
                .filter(|u| u.iter().any(|v| *v < 0.0 || *v > 1.0))
                .count(),
            degenerate_uv_triangles: degenerate,
            mirrored_triangles: mirrored,
            island_count: ids.len(),
            occupied_pixels: occupied,
            overlap_pixels: coverage.iter().filter(|&&c| c > 1).count(),
            padding_conflict_pixels: conflicts.len(),
            atlas_coverage: occupied as f32 / (r * r) as f32,
            pixels_per_world_unit: if world_area > 0.0 {
                (uv_area / world_area).sqrt() * r as f32
            } else {
                0.0
            },
        });
        images.push(UvDiagnosticImages {
            wireframe: wire,
            islands: island_image,
            overlaps: overlap,
        });
    }
    let checker = RgbaImage::from_fn(r, r, |x, y| {
        if (x / 32 + y / 32) % 2 == 0 {
            Rgba([220, 220, 220, 255])
        } else {
            Rgba([40, 40, 40, 255])
        }
    });
    Ok(UvCheckOutput{report:UvCheckReport{topology_signature:snapshot.topology_signature.clone(),uv_signature:snapshot.uv_signature.clone(),resolution:r,meshes:reports,
        diagnostics:vec!["Overlap/coverage are pixel-center estimates per mesh; subpixel overlaps may not register. Mirroring/repetition can be intentional. Separate meshes are not automatically one atlas.".into()]},images,checker})
}

fn root(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = root(parent, parent[i]);
    }
    parent[i]
}
fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}
fn line(image: &mut RgbaImage, a: [f32; 2], b: [f32; 2], color: Rgba<u8>) {
    let r = image.width() as f32;
    let steps = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) * r)
        .ceil()
        .min(r * 4.0) as u32;
    for i in 0..=steps {
        let t = i as f32 / steps.max(1) as f32;
        let x = ((a[0] + (b[0] - a[0]) * t) * r).round() as i32;
        let y = ((a[1] + (b[1] - a[1]) * t) * r).round() as i32;
        if x >= 0 && y >= 0 && x < r as i32 && y < r as i32 {
            image.put_pixel(x as u32, y as u32, color);
        }
    }
}
