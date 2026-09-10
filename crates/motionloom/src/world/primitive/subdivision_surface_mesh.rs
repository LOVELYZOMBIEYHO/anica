// =========================================
// =========================================
// crates/motionloom/src/world/primitive/subdivision_surface_mesh.rs

use super::{MeshBuilder, normalize, triangle_cross};
use crate::dsl::ControlCageNode;
use std::collections::BTreeMap;

fn average<'a>(items: impl Iterator<Item = &'a [f32; 3]>) -> [f32; 3] {
    let mut sum = [0.0; 3];
    let mut count = 0;
    for p in items {
        for a in 0..3 {
            sum[a] += p[a];
        }
        count += 1;
    }
    if count > 0 {
        for a in &mut sum {
            *a /= count as f32;
        }
    }
    sum
}

fn average_uv<'a>(items: impl Iterator<Item = &'a [f32; 2]>) -> [f32; 2] {
    let mut sum = [0.0; 2];
    let mut count = 0;
    for uv in items {
        sum[0] += uv[0];
        sum[1] += uv[1];
        count += 1;
    }
    if count > 0 {
        sum[0] /= count as f32;
        sum[1] /= count as f32;
    }
    sum
}

/// Catmull-Clark operates on polygon faces before triangulation. Pinned host
/// vertices/edges keep the fitted silhouette fixed while orbital rings relax.
fn subdivide(cage: &ControlCageNode) -> ControlCageNode {
    let points = &cage.positions;
    let face_points: Vec<_> = cage
        .faces
        .iter()
        .map(|f| average(f.iter().map(|&i| &points[i as usize])))
        .collect();
    let face_uvs: Vec<_> = cage
        .faces
        .iter()
        .map(|f| average_uv(f.iter().map(|&i| &cage.uvs[i as usize])))
        .collect();
    let mut edges = BTreeMap::<(u32, u32), Vec<usize>>::new();
    let mut vf = vec![vec![]; points.len()];
    for (fi, face) in cage.faces.iter().enumerate() {
        for j in 0..face.len() {
            let (a, b) = (face[j], face[(j + 1) % face.len()]);
            edges.entry((a.min(b), a.max(b))).or_default().push(fi);
            vf[a as usize].push(fi);
        }
    }
    let mut neighbors = vec![vec![]; points.len()];
    let mut boundary = vec![vec![]; points.len()];
    for (&(a, b), faces) in &edges {
        neighbors[a as usize].push(b);
        neighbors[b as usize].push(a);
        if faces.len() == 1 {
            boundary[a as usize].push(b);
            boundary[b as usize].push(a);
        }
    }
    let mut result = ControlCageNode {
        positions: points.clone(),
        uvs: cage.uvs.clone(),
        pinned: cage.pinned.clone(),
        faces: vec![],
        subdivision: 0,
    };
    for (i, p) in points.iter().enumerate() {
        if cage.pinned[i] || neighbors[i].is_empty() {
            continue;
        }
        if boundary[i].len() == 2 {
            let a = points[boundary[i][0] as usize];
            let b = points[boundary[i][1] as usize];
            result.positions[i] = std::array::from_fn(|k| (6.0 * p[k] + a[k] + b[k]) / 8.0);
            let a = cage.uvs[boundary[i][0] as usize];
            let b = cage.uvs[boundary[i][1] as usize];
            result.uvs[i] = std::array::from_fn(|k| (6.0 * cage.uvs[i][k] + a[k] + b[k]) / 8.0);
        } else {
            let f = average(vf[i].iter().map(|&j| &face_points[j]));
            let n = neighbors[i].len() as f32;
            let r = average(neighbors[i].iter().map(|&j| &points[j as usize]));
            // 2*edge-midpoint average = P + neighbor average.
            result.positions[i] = std::array::from_fn(|k| (f[k] + r[k] + (n - 2.0) * p[k]) / n);
            let f = average_uv(vf[i].iter().map(|&j| &face_uvs[j]));
            let r = average_uv(neighbors[i].iter().map(|&j| &cage.uvs[j as usize]));
            result.uvs[i] = std::array::from_fn(|k| (f[k] + r[k] + (n - 2.0) * cage.uvs[i][k]) / n);
        }
    }
    let mut ei = BTreeMap::new();
    for (&(a, b), faces) in &edges {
        let pin = cage.pinned[a as usize] && cage.pinned[b as usize];
        let mid = average([&points[a as usize], &points[b as usize]].into_iter());
        let value = if pin || faces.len() == 1 {
            mid
        } else {
            average(
                [
                    &points[a as usize],
                    &points[b as usize],
                    &face_points[faces[0]],
                    &face_points[faces[1]],
                ]
                .into_iter(),
            )
        };
        ei.insert((a, b), result.positions.len() as u32);
        result.positions.push(value);
        let uv = if faces.len() == 1 {
            average_uv([&cage.uvs[a as usize], &cage.uvs[b as usize]].into_iter())
        } else {
            average_uv(
                [
                    &cage.uvs[a as usize],
                    &cage.uvs[b as usize],
                    &face_uvs[faces[0]],
                    &face_uvs[faces[1]],
                ]
                .into_iter(),
            )
        };
        result.uvs.push(uv);
        result.pinned.push(pin);
    }
    for (fi, face) in cage.faces.iter().enumerate() {
        let center = result.positions.len() as u32;
        result.positions.push(face_points[fi]);
        result.uvs.push(face_uvs[fi]);
        result
            .pinned
            .push(face.iter().all(|&j| cage.pinned[j as usize]));
        for j in 0..face.len() {
            let a = face[j];
            let b = face[(j + 1) % face.len()];
            let c = face[(j + face.len() - 1) % face.len()];
            result.faces.push(vec![
                a,
                ei[&(a.min(b), a.max(b))],
                center,
                ei[&(a.min(c), a.max(c))],
            ]);
        }
    }
    result
}

pub(super) fn generate(builder: &mut MeshBuilder, cage: &ControlCageNode) {
    let mut mesh = cage.clone();
    if mesh.uvs.len() != mesh.positions.len() {
        mesh.uvs = mesh.positions.iter().map(|p| [p[0], p[1]]).collect();
    }
    for _ in 0..cage.subdivision.min(2) {
        mesh = subdivide(&mesh);
    }
    // Area-weighted normals share the same welded indices across skin/eyelids.
    let mut normals = vec![[0.0; 3]; mesh.positions.len()];
    for f in &mesh.faces {
        for j in 1..f.len() - 1 {
            let ids = [f[0] as usize, f[j] as usize, f[j + 1] as usize];
            let n = triangle_cross(
                mesh.positions[ids[0]],
                mesh.positions[ids[1]],
                mesh.positions[ids[2]],
            );
            for i in ids {
                for (k, v) in n.iter().enumerate() {
                    normals[i][k] += v;
                }
            }
        }
    }
    // Normal-only relaxation hides sampling-density seams in a pinned host;
    // it cannot move the approved silhouette or alter the socket depth.
    normals.iter_mut().for_each(|n| *n = normalize(*n));
    for pass in 0..32 {
        let old = normals.clone();
        let mut counts = vec![1.0; old.len()];
        normals = old.clone();
        for f in &mesh.faces {
            for j in 0..f.len() {
                let a = f[j] as usize;
                let b = f[(j + 1) % f.len()] as usize;
                for k in 0..3 {
                    normals[a][k] += old[b][k];
                    normals[b][k] += old[a][k];
                }
                counts[a] += 1.0;
                counts[b] += 1.0;
            }
        }
        for i in 0..normals.len() {
            normals[i] = if pass >= 3 && !mesh.pinned[i] {
                old[i]
            } else {
                normalize(std::array::from_fn(|k| {
                    old[i][k] * 0.35 + normals[i][k] / counts[i] * 0.65
                }))
            };
        }
    }
    let indices: Vec<_> = mesh
        .positions
        .iter()
        .zip(normals)
        .zip(&mesh.uvs)
        .map(|((&p, n), &uv)| builder.vertex(p, normalize(n), uv))
        .collect();
    for f in &mesh.faces {
        for j in 1..f.len() - 1 {
            builder.triangle(
                indices[f[0] as usize],
                indices[f[j] as usize],
                indices[f[j + 1] as usize],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn subdivision_keeps_host_pins_and_closes_shared_edges() {
        let cage = ControlCageNode {
            positions: vec![[-1., 0., 0.], [1., 0., 0.], [1., 1., 0.], [-1., 1., 0.]],
            uvs: vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
            pinned: vec![true; 4],
            faces: vec![vec![0, 1, 2, 3]],
            subdivision: 1,
        };
        let next = subdivide(&cage);
        assert_eq!(&next.positions[..4], cage.positions.as_slice());
        assert_eq!(next.faces.len(), 4);
        assert_eq!(next.positions.len(), 9);
        assert_eq!(next.positions[8], [0., 0.5, 0.]);
        assert_eq!(next.uvs[8], [0.5, 0.5]);
    }
    #[test]
    fn catmull_clark_cube_vertex_matches_formula() {
        let cage = ControlCageNode {
            positions: vec![
                [-1., -1., -1.],
                [1., -1., -1.],
                [1., 1., -1.],
                [-1., 1., -1.],
                [-1., -1., 1.],
                [1., -1., 1.],
                [1., 1., 1.],
                [-1., 1., 1.],
            ],
            uvs: vec![[0.; 2]; 8],
            pinned: vec![false; 8],
            faces: vec![
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![3, 7, 6, 2],
                vec![0, 4, 7, 3],
                vec![1, 2, 6, 5],
            ],
            subdivision: 1,
        };
        let next = subdivide(&cage);
        for axis in 0..3 {
            assert!((next.positions[0][axis] + 5.0 / 9.0).abs() < 1e-6);
        }
        assert_eq!(next.faces.len(), 24);
    }
}
