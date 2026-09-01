// =========================================
// =========================================
// crates/motionloom/src/world/primitive/capsule_mesh.rs

use super::MeshBuilder;
use std::f32::consts::{FRAC_PI_2, TAU};

pub(super) fn generate(
    mesh: &mut MeshBuilder,
    radius: f32,
    height: f32,
    segments: u32,
    rings: u32,
) {
    let hemisphere_rings = (rings.max(4) / 2).max(2);
    let cylinder_half = (height * 0.5 - radius).max(0.0);
    let mut rows = Vec::with_capacity((hemisphere_rings * 2 + 2) as usize);

    for ring in 0..=hemisphere_rings {
        let angle = ring as f32 / hemisphere_rings as f32 * FRAC_PI_2;
        rows.push((
            cylinder_half + angle.cos() * radius,
            angle.sin() * radius,
            angle.cos(),
        ));
    }
    rows.push((-cylinder_half, radius, 0.0));
    for ring in 1..=hemisphere_rings {
        let angle = ring as f32 / hemisphere_rings as f32 * FRAC_PI_2;
        rows.push((
            -cylinder_half - angle.sin() * radius,
            angle.cos() * radius,
            -angle.sin(),
        ));
    }

    let row_count = rows.len();
    for (row, (y, radial, normal_y)) in rows.iter().copied().enumerate() {
        let normal_radial = (1.0 - normal_y * normal_y).max(0.0).sqrt();
        let v = row as f32 / (row_count - 1) as f32;
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let (sin, cos) = (u * TAU).sin_cos();
            mesh.vertex(
                [cos * radial, y, sin * radial],
                [cos * normal_radial, normal_y, sin * normal_radial],
                [u, v],
            );
        }
    }

    let stride = segments + 1;
    for row in 0..(row_count as u32 - 1) {
        for segment in 0..segments {
            let a = row * stride + segment;
            let b = (row + 1) * stride + segment;
            let c = b + 1;
            let d = a + 1;
            mesh.triangle(a, c, b);
            mesh.triangle(a, d, c);
        }
    }
}
