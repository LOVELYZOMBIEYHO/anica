// =========================================
// =========================================
// crates/motionloom/examples/author_textured_cavity.rs

use motionloom::{ControlCageNode, api::mesh_authoring::mesh_asset_element};
use std::{env, fs};

// Create a recessed UV surface for the existing articulated mouth presentation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("expected scene path")?;
    let mut source = fs::read_to_string(&path)?;
    let mut cage = ControlCageNode {
        positions: vec![[-3.14, -0.38, 0.0]],
        uvs: vec![[0.76, 0.30]],
        pinned: vec![false],
        faces: vec![],
        subdivision: 0,
    };
    let segments = 48u32;
    for ring in 1..=8u32 {
        let r = ring as f32 / 8.0;
        for k in 0..segments {
            let a = k as f32 / segments as f32 * std::f32::consts::TAU;
            let y = r * a.cos();
            let z = r * a.sin();
            cage.positions
                .push([-3.14 - 0.06 * r * r, -0.38 + 0.20 * y, 0.30 * z]);
            cage.uvs.push([0.76 + 0.12 * z, 0.30 + 0.13 * y]);
            cage.pinned.push(false);
            let current = 1 + (ring - 1) * segments + k;
            let next = 1 + (ring - 1) * segments + (k + 1) % segments;
            if ring == 1 {
                cage.faces.push(vec![0, next, current]);
            } else {
                cage.faces
                    .push(vec![current - segments, next - segments, next, current]);
            }
        }
    }
    let start = source
        .find("<PrimitiveAsset id=\"oral_shadow\"")
        .ok_or("missing oral asset")?;
    let end = start + source[start..].find("/>").ok_or("missing asset end")? + 2;
    source.replace_range(
        start..end,
        &mesh_asset_element("oral_shadow", "oral", &cage),
    );
    fs::write(path, source)?;
    Ok(())
}
