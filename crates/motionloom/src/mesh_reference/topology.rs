// =========================================
// =========================================
// crates/motionloom/src/mesh_reference/topology.rs

use super::{MeshProposalValidationOptions, MeshReferenceDiagnostic, MeshTopologyReport};
use crate::ControlCageNode;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub fn validate_mesh_topology(
    cage: &ControlCageNode,
    options: &MeshProposalValidationOptions,
) -> MeshTopologyReport {
    let mut diagnostics = vec![];
    let mut degenerate_faces = vec![];
    let mut edges = BTreeMap::<(u32, u32), Vec<(usize, bool)>>::new();
    let mut valid_faces = vec![];
    for (face_index, face) in cage.faces.iter().enumerate() {
        let distinct: BTreeSet<_> = face.iter().copied().collect();
        let in_range = face
            .iter()
            .all(|&index| (index as usize) < cage.positions.len());
        if face.len() < 3 || distinct.len() < 3 || !in_range {
            degenerate_faces.push(face_index);
            continue;
        }
        let area = face_area(cage, face);
        if !area.is_finite() || area <= 1e-10 {
            degenerate_faces.push(face_index);
            continue;
        }
        valid_faces.push(face_index);
        for index in 0..face.len() {
            let a = face[index];
            let b = face[(index + 1) % face.len()];
            edges
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push((face_index, a < b));
        }
    }
    let open_edges = edges.values().filter(|uses| uses.len() == 1).count();
    let non_manifold_edges = edges.values().filter(|uses| uses.len() > 2).count();
    let inconsistent_winding_edges = edges
        .values()
        .filter(|uses| uses.len() == 2 && uses[0].1 == uses[1].1)
        .count();
    let connected_components = connected_components(cage.faces.len(), &edges, &valid_faces);
    let triangles = triangles(cage);
    let self_intersections = self_intersections(cage, &triangles, 64);
    let signed_volume: Option<f32> = if open_edges == 0 && non_manifold_edges == 0 {
        Some(
            triangles
                .iter()
                .map(|triangle| signed_tetrahedron(cage, *triangle))
                .sum(),
        )
    } else {
        None
    };
    if cage
        .positions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        diagnostics.push(error("NONFINITE_VERTEX", "vertex positions must be finite"));
    }
    if !degenerate_faces.is_empty() {
        diagnostics.push(error(
            "DEGENERATE_FACE",
            &format!(
                "{} faces are invalid or have zero area",
                degenerate_faces.len()
            ),
        ));
    }
    if non_manifold_edges > 0 {
        diagnostics.push(error(
            "NON_MANIFOLD_EDGE",
            &format!("{non_manifold_edges} edges have more than two incident faces"),
        ));
    }
    if open_edges > 0 && !options.allow_boundary {
        diagnostics.push(error(
            "OPEN_BOUNDARY",
            &format!("{open_edges} boundary edges are not allowed"),
        ));
    }
    if connected_components > 1 && !options.allow_multiple_components {
        diagnostics.push(error(
            "DISCONNECTED_COMPONENTS",
            &format!("mesh contains {connected_components} disconnected face components"),
        ));
    }
    if inconsistent_winding_edges > 0 {
        diagnostics.push(error(
            "INCONSISTENT_WINDING",
            &format!("{inconsistent_winding_edges} shared edges use the same direction"),
        ));
    }
    if !self_intersections.is_empty() {
        diagnostics.push(error(
            "SELF_INTERSECTION",
            &format!(
                "at least {} non-adjacent triangle pairs intersect",
                self_intersections.len()
            ),
        ));
    }
    if signed_volume.is_some_and(|volume| volume.abs() <= 1e-10) && open_edges == 0 {
        diagnostics.push(error(
            "COLLAPSED_VOLUME",
            "closed mesh has negligible signed volume",
        ));
    }
    let valid = diagnostics
        .iter()
        .all(|diagnostic| diagnostic.severity != "error");
    MeshTopologyReport {
        valid,
        vertices: cage.positions.len(),
        faces: cage.faces.len(),
        open_edges,
        non_manifold_edges,
        connected_components,
        degenerate_faces,
        inconsistent_winding_edges,
        self_intersections,
        signed_volume,
        diagnostics,
    }
}

fn face_area(cage: &ControlCageNode, face: &[u32]) -> f32 {
    let a = cage.positions[face[0] as usize];
    (1..face.len() - 1)
        .map(|index| {
            triangle_area(
                a,
                cage.positions[face[index] as usize],
                cage.positions[face[index + 1] as usize],
            )
        })
        .sum()
}

fn triangle_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    length(cross(ab, ac)) * 0.5
}

fn connected_components(
    face_count: usize,
    edges: &BTreeMap<(u32, u32), Vec<(usize, bool)>>,
    valid_faces: &[usize],
) -> usize {
    let mut adjacency = vec![vec![]; face_count];
    for uses in edges.values() {
        for a in uses {
            for b in uses {
                if a.0 != b.0 {
                    adjacency[a.0].push(b.0);
                }
            }
        }
    }
    let mut seen = vec![false; face_count];
    let mut count = 0;
    for &start in valid_faces {
        if seen[start] {
            continue;
        }
        count += 1;
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        while let Some(face) = queue.pop_front() {
            for &next in &adjacency[face] {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
    }
    count
}

fn triangles(cage: &ControlCageNode) -> Vec<[usize; 3]> {
    cage.faces
        .iter()
        .filter(|face| {
            face.len() >= 3
                && face
                    .iter()
                    .all(|&index| (index as usize) < cage.positions.len())
        })
        .flat_map(|face| {
            (1..face.len() - 1).map(move |index| {
                [
                    face[0] as usize,
                    face[index] as usize,
                    face[index + 1] as usize,
                ]
            })
        })
        .collect()
}

fn self_intersections(
    cage: &ControlCageNode,
    triangles: &[[usize; 3]],
    maximum: usize,
) -> Vec<[usize; 2]> {
    let mut order: Vec<_> = triangles
        .iter()
        .enumerate()
        .map(|(index, triangle)| (bounds(cage, *triangle), index))
        .collect();
    order.sort_by(|a, b| a.0.0[0].total_cmp(&b.0.0[0]));
    let mut result = vec![];
    for left in 0..order.len() {
        for right in left + 1..order.len() {
            if order[right].0.0[0] > order[left].0.1[0] {
                break;
            }
            let a = triangles[order[left].1];
            let b = triangles[order[right].1];
            if a.iter().any(|vertex| b.contains(vertex)) || !overlap(order[left].0, order[right].0)
            {
                continue;
            }
            if triangle_intersects(cage, a, b) {
                result.push([order[left].1, order[right].1]);
                if result.len() >= maximum {
                    return result;
                }
            }
        }
    }
    result
}

fn bounds(cage: &ControlCageNode, triangle: [usize; 3]) -> ([f32; 3], [f32; 3]) {
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for index in triangle {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(cage.positions[index][axis]);
            maximum[axis] = maximum[axis].max(cage.positions[index][axis]);
        }
    }
    (minimum, maximum)
}

fn overlap(a: ([f32; 3], [f32; 3]), b: ([f32; 3], [f32; 3])) -> bool {
    (0..3).all(|axis| a.0[axis] <= b.1[axis] && b.0[axis] <= a.1[axis])
}

fn triangle_intersects(cage: &ControlCageNode, a: [usize; 3], b: [usize; 3]) -> bool {
    let pa = a.map(|index| cage.positions[index]);
    let pb = b.map(|index| cage.positions[index]);
    triangle_edges(pa)
        .into_iter()
        .any(|(start, end)| segment_triangle(start, end, pb))
        || triangle_edges(pb)
            .into_iter()
            .any(|(start, end)| segment_triangle(start, end, pa))
}

fn triangle_edges(points: [[f32; 3]; 3]) -> [([f32; 3], [f32; 3]); 3] {
    [
        (points[0], points[1]),
        (points[1], points[2]),
        (points[2], points[0]),
    ]
}

fn segment_triangle(start: [f32; 3], end: [f32; 3], triangle: [[f32; 3]; 3]) -> bool {
    let direction = sub(end, start);
    let edge1 = sub(triangle[1], triangle[0]);
    let edge2 = sub(triangle[2], triangle[0]);
    let p = cross(direction, edge2);
    let determinant = dot(edge1, p);
    if determinant.abs() < 1e-7 {
        return false;
    }
    let inverse = 1.0 / determinant;
    let t = sub(start, triangle[0]);
    let u = dot(t, p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = cross(t, edge1);
    let v = dot(direction, q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let distance = dot(edge2, q) * inverse;
    (1e-6..=1.0 - 1e-6).contains(&distance)
}

fn signed_tetrahedron(cage: &ControlCageNode, triangle: [usize; 3]) -> f32 {
    let a = cage.positions[triangle[0]];
    let b = cage.positions[triangle[1]];
    let c = cage.positions[triangle[2]];
    dot(a, cross(b, c)) / 6.0
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn error(code: &str, message: &str) -> MeshReferenceDiagnostic {
    MeshReferenceDiagnostic {
        severity: "error".into(),
        code: code.into(),
        message: message.into(),
    }
}
