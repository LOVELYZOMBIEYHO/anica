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
            // Keep triangle winding aligned with the authored outward radial normals.
            mesh.triangle(pair[0][edge], pair[1][next], pair[0][next]);
            mesh.triangle(pair[0][edge], pair[1][edge], pair[1][next]);
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
            mesh.triangle(center, ring[next], ring[edge]);
        } else {
            mesh.triangle(center, ring[edge], ring[next]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(at: f32, width: f32, depth: f32, profile: &str) -> PrimitiveLoftSectionNode {
        PrimitiveLoftSectionNode {
            at,
            width,
            depth,
            profile: profile.into(),
            offset: [0.0, 0.0],
            rotation: 0.0,
        }
    }

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn assert_outward_winding(mesh: &MeshBuilder) {
        for (triangle_index, triangle) in mesh.indices.chunks_exact(3).enumerate() {
            let positions = [
                mesh.positions[triangle[0] as usize],
                mesh.positions[triangle[1] as usize],
                mesh.positions[triangle[2] as usize],
            ];
            let geometric = cross(
                subtract(positions[1], positions[0]),
                subtract(positions[2], positions[0]),
            );
            let area_squared = dot(geometric, geometric);
            assert!(
                area_squared > 1.0e-10,
                "triangle {triangle_index} is degenerate"
            );
            let authored = triangle.iter().fold([0.0; 3], |mut sum, index| {
                let normal = mesh.normals[*index as usize].expect("loft vertex normal");
                for axis in 0..3 {
                    sum[axis] += normal[axis];
                }
                sum
            });
            assert!(
                dot(geometric, authored) > 0.0,
                "triangle {triangle_index} winding opposes its authored normal"
            );
        }
    }

    #[test]
    fn closed_loft_sides_and_caps_use_outward_winding() {
        let mut sections = [
            section(-1.0, 0.9, 0.7, "ellipse"),
            section(0.0, 0.58, 0.75, "rounded_rect"),
            section(1.0, 0.72, 0.5, "capsule"),
        ];
        sections[1].offset = [0.08, -0.04];
        sections[1].rotation = 7.0;
        let mut mesh = MeshBuilder::default();
        generate(&mut mesh, 16, true, true, true, &sections);

        // Winding changes must not alter the canonical vertex, UV, or face counts.
        assert_eq!(mesh.positions.len(), sections.len() * 16 + 2);
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        assert_eq!(mesh.texcoords.len(), mesh.positions.len());
        assert_eq!(mesh.indices.len(), 16 * 6 * 2 + 16 * 3 * 2);
        assert!(
            mesh.positions
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert!(
            mesh.texcoords
                .iter()
                .flatten()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert!(
            mesh.normals
                .iter()
                .flatten()
                .all(|normal| { (dot(*normal, *normal) - 1.0).abs() < 1.0e-5 })
        );
        assert_outward_winding(&mesh);
    }

    #[test]
    fn open_loft_profiles_keep_outward_winding_for_outline_culling() {
        for profile in ["ellipse", "rounded_rect", "capsule", "diamond"] {
            let sections = [
                section(-0.8, 0.95, 0.62, profile),
                section(0.2, 0.64, 0.8, profile),
                section(0.9, 0.5, 0.55, profile),
            ];
            let mut mesh = MeshBuilder::default();
            generate(&mut mesh, 16, false, true, true, &sections);

            assert_eq!(mesh.positions.len(), sections.len() * 17);
            assert_eq!(mesh.indices.len(), 16 * 6 * 2);
            assert_outward_winding(&mesh);
        }
    }
}
