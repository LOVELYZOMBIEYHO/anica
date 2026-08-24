// =========================================
// =========================================
// crates/motionloom/src/world/primitive/cone_mesh.rs

use super::MeshBuilder;
use std::f32::consts::TAU;

pub(super) fn generate(mesh: &mut MeshBuilder, radius: f32, height: f32, segments: u32) {
    let half = height * 0.5;
    let slope = radius / height;
    for segment in 0..segments {
        let u0 = segment as f32 / segments as f32;
        let u1 = (segment + 1) as f32 / segments as f32;
        let (s0, c0) = (u0 * TAU).sin_cos();
        let (s1, c1) = (u1 * TAU).sin_cos();
        let normalize = |v: [f32; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let a = mesh.vertex(
            [0.0, half, 0.0],
            normalize([c0, slope, s0]),
            [(u0 + u1) * 0.5, 1.0],
        );
        let b = mesh.vertex(
            [c0 * radius, -half, s0 * radius],
            normalize([c0, slope, s0]),
            [u0, 0.0],
        );
        let c = mesh.vertex(
            [c1 * radius, -half, s1 * radius],
            normalize([c1, slope, s1]),
            [u1, 0.0],
        );
        mesh.triangle(a, c, b);
        let center = mesh.vertex([0.0, -half, 0.0], [0.0, -1.0, 0.0], [0.5, 0.5]);
        let b0 = mesh.vertex(
            [c0 * radius, -half, s0 * radius],
            [0.0, -1.0, 0.0],
            [(c0 + 1.0) * 0.5, (s0 + 1.0) * 0.5],
        );
        let b1 = mesh.vertex(
            [c1 * radius, -half, s1 * radius],
            [0.0, -1.0, 0.0],
            [(c1 + 1.0) * 0.5, (s1 + 1.0) * 0.5],
        );
        mesh.triangle(center, b0, b1);
    }
}
