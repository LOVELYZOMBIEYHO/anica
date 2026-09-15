// =========================================
// =========================================
// crates/motionloom/examples/articulate_mesh_region.rs

// Separate an existing face region into a rigid articulation, preserving UVs.
use motionloom::api::mesh_authoring::mesh_asset_element;
use motionloom::{ControlCageNode, PrimitiveGeometry, parse_graph_script};
use std::{collections::BTreeMap, env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().collect();
    if a.len() != 7 {
        return Err("expected source asset new-asset max-x max-y output".into());
    }
    let source = fs::read_to_string(&a[1])?;
    let graph = parse_graph_script(&source)?;
    let asset = graph
        .assets
        .iter()
        .find(|v| v.id == a[2])
        .ok_or("missing asset")?;
    let primitive = asset.primitive().ok_or("not primitive")?;
    let material = primitive.material.as_deref().ok_or("missing material")?;
    let PrimitiveGeometry::Mesh { cage } = &primitive.geometry else {
        return Err("not MeshAsset".into());
    };
    let max_x: f32 = a[4].parse()?;
    let max_y: f32 = a[5].parse()?;
    let mut selected = Vec::new();
    let mut retained = Vec::new();
    for face in &cage.faces {
        let center = face.iter().fold([0.0; 3], |mut c, &i| {
            for k in 0..3 {
                c[k] += cage.positions[i as usize][k] / face.len() as f32;
            }
            c
        });
        if center[0] < max_x && center[1] < max_y {
            selected.push(face.clone());
        } else {
            retained.push(face.clone());
        }
    }
    if selected.is_empty() || retained.is_empty() {
        return Err("selection must split the mesh".into());
    }
    let compact = |faces: Vec<Vec<u32>>| {
        let mut mapping = BTreeMap::new();
        let mut out = ControlCageNode {
            positions: vec![],
            uvs: vec![],
            pinned: vec![],
            faces: vec![],
            subdivision: cage.subdivision,
        };
        for face in faces {
            let mut indices = vec![];
            for i in face {
                let j = *mapping.entry(i).or_insert_with(|| {
                    let j = out.positions.len() as u32;
                    out.positions.push(cage.positions[i as usize]);
                    out.uvs.push(cage.uvs[i as usize]);
                    out.pinned.push(cage.pinned[i as usize]);
                    j
                });
                indices.push(j);
            }
            out.faces.push(indices);
        }
        out
    };
    let start = source
        .find(&format!("<MeshAsset id=\"{}\"", a[2]))
        .ok_or("missing source tag")?;
    let end = start
        + source[start..]
            .find("</MeshAsset>")
            .ok_or("missing close")?
        + "</MeshAsset>".len();
    let mut result = source.clone();
    result.replace_range(
        start..end,
        &format!(
            "{}\n{}",
            mesh_asset_element(&a[2], material, &compact(retained)),
            mesh_asset_element(&a[3], material, &compact(selected))
        ),
    );
    fs::write(&a[6], result)?;
    Ok(())
}
