// =========================================
// =========================================
// crates/motionloom/examples/export_scene_glb.rs

use motionloom::experimental::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: export_scene_glb <script> <output.glb> [scene-id] [frame]".into());
    }
    let path = std::path::Path::new(&args[1]);
    motionloom::set_scene_asset_roots(vec![path.parent().unwrap().to_path_buf()]);
    let graph = motionloom::parse_graph_script(&std::fs::read_to_string(path)?)?;
    let snapshot = pollster::block_on(extract_scene_geometry(
        &graph,
        &SceneGeometryOptions {
            scene_id: args
                .get(3)
                .cloned()
                .unwrap_or_else(|| graph.scenes[0].id.clone()),
            frame: args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(0),
            include_hidden: false,
            selected_model_ids: None,
        },
    ))?;
    let bytes = export_scene_glb(&snapshot)?;
    std::fs::write(&args[2], &bytes)?;
    println!(
        "{} meshes, {} vertices, {} triangles; {} bytes\n{}\n{}",
        snapshot.meshes.len(),
        snapshot
            .meshes
            .iter()
            .map(|m| m.positions.len())
            .sum::<usize>(),
        snapshot
            .meshes
            .iter()
            .map(|m| m.indices.len() / 3)
            .sum::<usize>(),
        bytes.len(),
        snapshot.topology_signature,
        snapshot.uv_signature
    );
    Ok(())
}
