// =========================================
// =========================================
// crates/motionloom/src/world/primitive/sphere_mesh.rs

use super::MeshBuilder;
use std::f32::consts::{PI, TAU};

pub(super) fn generate(mesh: &mut MeshBuilder, radius: f32, segments: u32, rings: u32) {
    for ring in 0..rings {
        let v0 = ring as f32 / rings as f32;
        let v1 = (ring + 1) as f32 / rings as f32;
        for segment in 0..segments {
            let u0 = segment as f32 / segments as f32;
            let u1 = (segment + 1) as f32 / segments as f32;
            let point = |u: f32, v: f32| {
                let phi = v * PI;
                let theta = u * TAU;
                let normal = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
                (
                    [normal[0] * radius, normal[1] * radius, normal[2] * radius],
                    normal,
                )
            };
            let (pa, na) = point(u0, v0);
            let (pb, nb) = point(u0, v1);
            let (pc, nc) = point(u1, v1);
            let (pd, nd) = point(u1, v0);
            let a = mesh.vertex(pa, na, [u0, v0]);
            let b = mesh.vertex(pb, nb, [u0, v1]);
            let c = mesh.vertex(pc, nc, [u1, v1]);
            let d = mesh.vertex(pd, nd, [u1, v0]);
            mesh.triangle(a, c, b);
            mesh.triangle(a, d, c);
        }
    }
}
