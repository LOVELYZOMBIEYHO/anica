// =========================================
// =========================================
// crates/motionloom/src/world/primitive/box_mesh.rs

use super::MeshBuilder;

pub(super) fn generate(mesh: &mut MeshBuilder, size: [f32; 3]) {
    let [x, y, z] = size.map(|value| value * 0.5);
    mesh.quad(
        [[-x, y, -z], [-x, y, z], [x, y, z], [x, y, -z]],
        [0.0, 1.0, 0.0],
    );
    mesh.quad(
        [[-x, -y, z], [-x, -y, -z], [x, -y, -z], [x, -y, z]],
        [0.0, -1.0, 0.0],
    );
    mesh.quad(
        [[x, -y, -z], [x, y, -z], [x, y, z], [x, -y, z]],
        [1.0, 0.0, 0.0],
    );
    mesh.quad(
        [[-x, -y, z], [-x, y, z], [-x, y, -z], [-x, -y, -z]],
        [-1.0, 0.0, 0.0],
    );
    mesh.quad(
        [[x, -y, z], [x, y, z], [-x, y, z], [-x, -y, z]],
        [0.0, 0.0, 1.0],
    );
    mesh.quad(
        [[-x, -y, -z], [-x, y, -z], [x, y, -z], [x, -y, -z]],
        [0.0, 0.0, -1.0],
    );
}

/// Round a box inward so its authored bounds and simple box collider stay exact.
pub(super) fn generate_beveled(mesh: &mut MeshBuilder, size: [f32; 3], radius: f32, segments: u32) {
    let half = size.map(|value| value * 0.5);
    let inner = half.map(|value| value - radius);
    let axis_samples = |extent: f32, inset: f32| {
        let mut values = Vec::with_capacity((segments * 2 + 2) as usize);
        for index in 0..=segments {
            values.push(-extent + radius * index as f32 / segments as f32);
        }
        for index in (0..=segments).rev() {
            values.push(extent - radius * index as f32 / segments as f32);
        }
        values.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);
        debug_assert!(values.first().is_some_and(|value| *value <= -inset));
        values
    };
    let xs = axis_samples(half[0], inner[0]);
    let ys = axis_samples(half[1], inner[1]);
    let zs = axis_samples(half[2], inner[2]);
    for (axis, sign, us, vs) in [
        (0, 1.0, &zs, &ys),
        (0, -1.0, &zs, &ys),
        (1, 1.0, &xs, &zs),
        (1, -1.0, &xs, &zs),
        (2, 1.0, &xs, &ys),
        (2, -1.0, &xs, &ys),
    ] {
        let mut ids = vec![vec![0_u32; us.len()]; vs.len()];
        for (v_index, v) in vs.iter().enumerate() {
            for (u_index, u) in us.iter().enumerate() {
                let mut cube = [0.0; 3];
                cube[axis] = half[axis] * sign;
                let other = match axis {
                    0 => [2, 1],
                    1 => [0, 2],
                    _ => [0, 1],
                };
                cube[other[0]] = *u;
                cube[other[1]] = *v;
                let core: [f32; 3] =
                    std::array::from_fn(|index| cube[index].clamp(-inner[index], inner[index]));
                let delta = std::array::from_fn::<_, 3, _>(|index| cube[index] - core[index]);
                let length = delta.iter().map(|value| value * value).sum::<f32>().sqrt();
                let normal = delta.map(|value| value / length.max(f32::EPSILON));
                let position = std::array::from_fn(|index| core[index] + normal[index] * radius);
                ids[v_index][u_index] = mesh.vertex(position, normal, [0.0, 0.0]);
            }
        }
        for v in 0..vs.len() - 1 {
            for u in 0..us.len() - 1 {
                let quad = [ids[v][u], ids[v][u + 1], ids[v + 1][u + 1], ids[v + 1][u]];
                let a = mesh.positions[quad[0] as usize];
                let b = mesh.positions[quad[1] as usize];
                let c = mesh.positions[quad[2] as usize];
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let cross = [
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                ];
                let outward = cross[axis] * sign > 0.0;
                if outward {
                    mesh.triangle(quad[0], quad[1], quad[2]);
                    mesh.triangle(quad[0], quad[2], quad[3]);
                } else {
                    mesh.triangle(quad[0], quad[2], quad[1]);
                    mesh.triangle(quad[0], quad[3], quad[2]);
                }
            }
        }
    }
}
