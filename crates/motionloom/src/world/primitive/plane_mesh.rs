// =========================================
// =========================================
// crates/motionloom/src/world/primitive/plane_mesh.rs

use super::MeshBuilder;

pub(super) fn generate(mesh: &mut MeshBuilder, size: [f32; 2], segments: u32) {
    let segments = segments.max(1);
    for z in 0..segments {
        for x in 0..segments {
            let u0 = x as f32 / segments as f32;
            let u1 = (x + 1) as f32 / segments as f32;
            let v0 = z as f32 / segments as f32;
            let v1 = (z + 1) as f32 / segments as f32;
            let px = |u: f32| (u - 0.5) * size[0];
            let pz = |v: f32| (v - 0.5) * size[1];
            let a = mesh.vertex([px(u0), 0.0, pz(v0)], [0.0, 1.0, 0.0], [u0, v0]);
            let b = mesh.vertex([px(u0), 0.0, pz(v1)], [0.0, 1.0, 0.0], [u0, v1]);
            let c = mesh.vertex([px(u1), 0.0, pz(v1)], [0.0, 1.0, 0.0], [u1, v1]);
            let d = mesh.vertex([px(u1), 0.0, pz(v0)], [0.0, 1.0, 0.0], [u1, v0]);
            mesh.triangle(a, b, c);
            mesh.triangle(a, c, d);
        }
    }
}
