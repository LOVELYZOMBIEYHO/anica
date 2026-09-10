// =========================================
// crates/motionloom/src/world/primitive/ribbon_mesh.rs
// =========================================

use crate::dsl::PrimitiveRibbonPointNode;

use super::{MeshBuilder, dot, normalize};

pub(super) fn generate(
    mesh: &mut MeshBuilder,
    default_width: f32,
    default_thickness: f32,
    cap_start: bool,
    cap_end: bool,
    points: &[PrimitiveRibbonPointNode],
) {
    let mut sections = Vec::with_capacity(points.len());
    let mut previous_side = None;
    for (index, point) in points.iter().enumerate() {
        let tangent = point_tangent(points, index);
        let reference = if tangent[1].abs() < 0.92 {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        let mut side = normalize(cross(reference, tangent));
        if previous_side.is_some_and(|previous| dot(previous, side) < 0.0) {
            side = side.map(|value| -value);
        }
        let mut normal = normalize(cross(tangent, side));
        if point.roll != 0.0 {
            side = rotate_about_axis(side, tangent, point.roll.to_radians());
            normal = normalize(cross(tangent, side));
        }
        previous_side = Some(side);
        let half_width = point.width.unwrap_or(default_width) * 0.5;
        let half_thickness = point.thickness.unwrap_or(default_thickness) * 0.5;
        sections.push([
            offset(point.position, side, -half_width, normal, -half_thickness),
            offset(point.position, side, half_width, normal, -half_thickness),
            offset(point.position, side, half_width, normal, half_thickness),
            offset(point.position, side, -half_width, normal, half_thickness),
        ]);
    }

    for pair in sections.windows(2) {
        for face in 0..4 {
            let next = (face + 1) % 4;
            let vertices = [pair[0][face], pair[0][next], pair[1][next], pair[1][face]];
            let normal = normalize(cross(
                subtract(vertices[1], vertices[0]),
                subtract(vertices[3], vertices[0]),
            ));
            mesh.quad(vertices, normal);
        }
    }
    if cap_start {
        cap(mesh, sections[0], false);
    }
    if cap_end {
        cap(mesh, *sections.last().expect("ribbon has sections"), true);
    }
}

fn point_tangent(points: &[PrimitiveRibbonPointNode], index: usize) -> [f32; 3] {
    let before = points[index.saturating_sub(1)].position;
    let after = points[(index + 1).min(points.len() - 1)].position;
    normalize(subtract(after, before))
}

fn offset(
    origin: [f32; 3],
    side: [f32; 3],
    side_scale: f32,
    normal: [f32; 3],
    normal_scale: f32,
) -> [f32; 3] {
    [
        origin[0] + side[0] * side_scale + normal[0] * normal_scale,
        origin[1] + side[1] * side_scale + normal[1] * normal_scale,
        origin[2] + side[2] * side_scale + normal[2] * normal_scale,
    ]
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn rotate_about_axis(vector: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin, cos) = angle.sin_cos();
    let crossed = cross(axis, vector);
    let projected = dot(axis, vector) * (1.0 - cos);
    normalize([
        vector[0] * cos + crossed[0] * sin + axis[0] * projected,
        vector[1] * cos + crossed[1] * sin + axis[1] * projected,
        vector[2] * cos + crossed[2] * sin + axis[2] * projected,
    ])
}

fn cap(mesh: &mut MeshBuilder, section: [[f32; 3]; 4], end: bool) {
    let normal = normalize(cross(
        subtract(section[1], section[0]),
        subtract(section[3], section[0]),
    ));
    if end {
        mesh.quad(section, normal);
    } else {
        mesh.quad(
            [section[3], section[2], section[1], section[0]],
            normal.map(|value| -value),
        );
    }
}
