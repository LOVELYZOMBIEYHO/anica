// =========================================
// =========================================
// crates/motionloom/src/world/primitive/wedge_mesh.rs

use super::MeshBuilder;

pub(super) fn generate(mesh: &mut MeshBuilder, size: [f32; 3]) {
    let [x, y, z] = size.map(|value| value * 0.5);
    // The ramp rises from -Z to +Z, giving the public contract one fixed orientation.
    mesh.quad(
        [[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]],
        [0.0, -1.0, 0.0],
    );
    let slope = [0.0, size[2], -size[1]];
    let length = (slope[1] * slope[1] + slope[2] * slope[2]).sqrt();
    mesh.quad(
        [[-x, -y, -z], [-x, y, z], [x, y, z], [x, -y, -z]],
        [0.0, slope[1] / length, slope[2] / length],
    );
    mesh.triangle_face([[x, -y, -z], [x, y, z], [x, -y, z]], [1.0, 0.0, 0.0]);
    mesh.triangle_face([[-x, -y, z], [-x, y, z], [-x, -y, -z]], [-1.0, 0.0, 0.0]);
    mesh.quad(
        [[x, -y, z], [x, y, z], [-x, y, z], [-x, -y, z]],
        [0.0, 0.0, 1.0],
    );
}
