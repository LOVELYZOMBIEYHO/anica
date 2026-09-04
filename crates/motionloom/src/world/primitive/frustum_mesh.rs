// =========================================
// =========================================
// crates/motionloom/src/world/primitive/frustum_mesh.rs

use super::MeshBuilder;

/// Generate a rectangular frustum with independent top and bottom footprints.
pub(super) fn generate(
    mesh: &mut MeshBuilder,
    top_size: [f32; 2],
    bottom_size: [f32; 2],
    height: f32,
) {
    let top = [top_size[0] * 0.5, top_size[1] * 0.5];
    let bottom = [bottom_size[0] * 0.5, bottom_size[1] * 0.5];
    let y = height * 0.5;
    let top_vertices = [
        [-top[0], y, -top[1]],
        [-top[0], y, top[1]],
        [top[0], y, top[1]],
        [top[0], y, -top[1]],
    ];
    let bottom_vertices = [
        [-bottom[0], -y, -bottom[1]],
        [-bottom[0], -y, bottom[1]],
        [bottom[0], -y, bottom[1]],
        [bottom[0], -y, -bottom[1]],
    ];
    mesh.quad(top_vertices, [0.0, 1.0, 0.0]);
    mesh.quad(
        [
            bottom_vertices[1],
            bottom_vertices[0],
            bottom_vertices[3],
            bottom_vertices[2],
        ],
        [0.0, -1.0, 0.0],
    );
    for edge in 0..4 {
        let next = (edge + 1) % 4;
        let a = bottom_vertices[edge];
        let b = bottom_vertices[next];
        let c = top_vertices[next];
        let d = top_vertices[edge];
        let normal = face_normal(a, b, c);
        mesh.quad([a, b, c, d], normal);
    }
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let length = normal.iter().map(|value| value * value).sum::<f32>().sqrt();
    normal.map(|value| value / length.max(f32::EPSILON))
}
