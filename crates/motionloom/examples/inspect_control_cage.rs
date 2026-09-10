// =========================================
// =========================================
// crates/motionloom/examples/inspect_control_cage.rs

use motionloom::parse_graph_script;
use std::{
    collections::{BTreeMap, HashSet},
    io::Write,
};

/// Export the actual runtime surface and its explicit or generated control cage.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err(
            "usage: inspect_control_cage <file.motionloom> <asset-id> <output-prefix>".into(),
        );
    }
    let graph = parse_graph_script(&std::fs::read_to_string(&args[1])?)?;
    let asset = graph
        .assets
        .iter()
        .find(|a| a.id == args[2])
        .and_then(|a| a.primitive())
        .ok_or("missing model asset")?;
    let cage = motionloom::experimental::generated_control_cage(asset)
        .ok_or("asset has no inspectable control cage")?;
    let mesh = motionloom::experimental::generate_primitive_mesh(asset);
    let mut surface =
        std::io::BufWriter::new(std::fs::File::create(format!("{}-surface.obj", args[3]))?);
    let mut control =
        std::io::BufWriter::new(std::fs::File::create(format!("{}-cage.obj", args[3]))?);
    for p in &mesh.positions {
        writeln!(surface, "v {} {} {}", p[0], p[1], p[2])?;
    }
    for f in mesh.indices.chunks_exact(3) {
        writeln!(surface, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1)?;
    }
    for p in &cage.positions {
        writeln!(control, "v {} {} {}", p[0], p[1], p[2])?;
    }
    for f in &cage.faces {
        writeln!(
            control,
            "f {}",
            f.iter()
                .map(|i| (i + 1).to_string())
                .collect::<Vec<_>>()
                .join(" ")
        )?;
    }
    let mut edges = BTreeMap::<(u32, u32), usize>::new();
    for f in mesh.indices.chunks_exact(3) {
        for i in 0..3 {
            let (a, b) = (f[i], f[(i + 1) % 3]);
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let positions: HashSet<_> = mesh.positions.iter().map(|p| p.map(f32::to_bits)).collect();
    let missing_pins = cage
        .positions
        .iter()
        .zip(&cage.pinned)
        .filter(|(p, pin)| **pin && !positions.contains(&p.map(f32::to_bits)))
        .count();
    let report = serde_json::json!({"controlVertices":cage.positions.len(),"controlFaces":cage.faces.len(),"pinnedVertices":cage.pinned.iter().filter(|&&pin|pin).count(),"renderVertices":mesh.positions.len(),"renderTriangles":mesh.indices.len()/3,"openEdges":edges.values().filter(|&&n|n==1).count(),"nonManifoldEdges":edges.values().filter(|&&n|n>2).count(),"missingPinnedVertices":missing_pins,"finitePositions":mesh.positions.iter().flatten().all(|x|x.is_finite()),"subdivision":cage.subdivision});
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
