// =========================================
// =========================================
// crates/motionloom/examples/check_scene_uvs.rs

use motionloom::experimental::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err(
            "usage: check_scene_uvs <script> <output-directory> [scene-id] [frame] [resolution]"
                .into(),
        );
    }
    let path = std::path::Path::new(&args[1]);
    motionloom::set_scene_asset_roots(vec![path.parent().unwrap().to_path_buf()]);
    let graph = motionloom::parse_graph_script(&std::fs::read_to_string(path)?)?;
    let options = SceneGeometryOptions {
        scene_id: args
            .get(3)
            .cloned()
            .unwrap_or_else(|| graph.scenes[0].id.clone()),
        frame: args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(0),
        include_hidden: false,
        selected_model_ids: None,
    };
    let snapshot = pollster::block_on(extract_scene_geometry(&graph, &options))?;
    let check = check_scene_uvs(
        &snapshot,
        &UvCheckOptions {
            resolution: args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(1024),
            ..Default::default()
        },
    )?;
    let out = std::path::Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    for (i, images) in check.images.iter().enumerate() {
        images
            .wireframe
            .save(out.join(format!("{i:03}-wireframe.png")))?;
        images
            .islands
            .save(out.join(format!("{i:03}-islands.png")))?;
        images
            .overlaps
            .save(out.join(format!("{i:03}-overlaps.png")))?;
    }
    check.checker.save(out.join("checker.png"))?;
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&check.report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&check.report)?);
    Ok(())
}
