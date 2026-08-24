// =========================================
// =========================================
// crates/motionloom/src/world/primitive/cylinder_mesh.rs

use super::MeshBuilder;
use std::f32::consts::TAU;

pub(super) fn generate(mesh: &mut MeshBuilder, radius: f32, height: f32, segments: u32) {
    let half = height * 0.5;
    for segment in 0..segments {
        let u0 = segment as f32 / segments as f32;
        let u1 = (segment + 1) as f32 / segments as f32;
        let (s0, c0) = (u0 * TAU).sin_cos();
        let (s1, c1) = (u1 * TAU).sin_cos();
        let a = mesh.vertex([c0 * radius, -half, s0 * radius], [c0, 0.0, s0], [u0, 0.0]);
        let b = mesh.vertex([c0 * radius, half, s0 * radius], [c0, 0.0, s0], [u0, 1.0]);
        let c = mesh.vertex([c1 * radius, half, s1 * radius], [c1, 0.0, s1], [u1, 1.0]);
        let d = mesh.vertex([c1 * radius, -half, s1 * radius], [c1, 0.0, s1], [u1, 0.0]);
        mesh.triangle(a, b, c);
        mesh.triangle(a, c, d);
        let top = mesh.vertex([0.0, half, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5]);
        let t0 = mesh.vertex(
            [c0 * radius, half, s0 * radius],
            [0.0, 1.0, 0.0],
            [(c0 + 1.0) * 0.5, (s0 + 1.0) * 0.5],
        );
        let t1 = mesh.vertex(
            [c1 * radius, half, s1 * radius],
            [0.0, 1.0, 0.0],
            [(c1 + 1.0) * 0.5, (s1 + 1.0) * 0.5],
        );
        mesh.triangle(top, t1, t0);
        let bottom = mesh.vertex([0.0, -half, 0.0], [0.0, -1.0, 0.0], [0.5, 0.5]);
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
        mesh.triangle(bottom, b0, b1);
    }
}
