// =========================================
// crates/motionloom/src/world/primitive/loft_mesh.rs
// =========================================

use std::f32::consts::TAU;

use crate::dsl::PrimitiveLoftSectionNode;

use super::{MeshBuilder, normalize};

pub(super) fn generate(
    mesh: &mut MeshBuilder,
    segments: u32,
    closed: bool,
    cap_start: bool,
    cap_end: bool,
    sections: &[PrimitiveLoftSectionNode],
) {
    let ring_samples = segments as usize + usize::from(!closed);
    let mut rings = Vec::with_capacity(sections.len());
    for section in sections {
        let angle = section.rotation.to_radians();
        let (sin_rotation, cos_rotation) = angle.sin_cos();
        let mut ring = Vec::with_capacity(ring_samples);
        for sample in 0..ring_samples {
            let phase = sample as f32 / segments as f32;
            let theta = phase * TAU;
            let [profile_x, profile_z] = profile_point(&section.profile, theta);
            let x = profile_x * section.width * 0.5;
            let z = profile_z * section.depth * 0.5;
            let position = [
                section.offset[0] + x * cos_rotation - z * sin_rotation,
                section.at,
                section.offset[1] + x * sin_rotation + z * cos_rotation,
            ];
            let radial = normalize([
                profile_x * cos_rotation - profile_z * sin_rotation,
                0.0,
                profile_x * sin_rotation + profile_z * cos_rotation,
            ]);
            ring.push(mesh.vertex(position, radial, [phase, section.at]));
        }
        rings.push(ring);
    }

    let edge_count = segments as usize;
    for pair in rings.windows(2) {
        for edge in 0..edge_count {
            let next = if closed {
                (edge + 1) % ring_samples
            } else {
                edge + 1
            };
            mesh.triangle(pair[0][edge], pair[0][next], pair[1][next]);
            mesh.triangle(pair[0][edge], pair[1][next], pair[1][edge]);
        }
    }
    if closed && cap_start {
        cap(mesh, &rings[0], &sections[0], false);
    }
    if closed && cap_end {
        cap(
            mesh,
            rings.last().expect("loft has sections"),
            sections.last().expect("loft has sections"),
            true,
        );
    }
}

fn profile_point(profile: &str, theta: f32) -> [f32; 2] {
    let (sin, cos) = theta.sin_cos();
    match profile {
        "rounded_rect" => [
            cos.signum() * cos.abs().sqrt(),
            sin.signum() * sin.abs().sqrt(),
        ],
        "diamond" => {
            let scale = (cos.abs() + sin.abs()).max(f32::EPSILON);
            [cos / scale, sin / scale]
        }
        "capsule" => {
            let x = cos.clamp(-0.72, 0.72) / 0.72;
            [x, sin]
        }
        _ => [cos, sin],
    }
}

fn cap(mesh: &mut MeshBuilder, ring: &[u32], section: &PrimitiveLoftSectionNode, top: bool) {
    let normal = if top {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, -1.0, 0.0]
    };
    let center = mesh.vertex(
        [section.offset[0], section.at, section.offset[1]],
        normal,
        [0.5, 0.5],
    );
    for edge in 0..ring.len() {
        let next = (edge + 1) % ring.len();
        if top {
            mesh.triangle(center, ring[edge], ring[next]);
        } else {
            mesh.triangle(center, ring[next], ring[edge]);
        }
    }
}
