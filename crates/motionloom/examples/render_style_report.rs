// =========================================
// =========================================
// crates/motionloom/examples/render_style_report.rs

// Read-only CLI inspection uses the same resolver as WASM and Rust hosts.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let file = args
        .next()
        .ok_or("usage: render_style_report <file> [scene-id]")?;
    let graph = motionloom::api::parse_graph_script(&std::fs::read_to_string(file)?)?;
    let scene = args
        .next()
        .or_else(|| graph.scenes.first().map(|s| s.id.clone()))
        .ok_or("No Scene")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&motionloom::api::resolve_scene_render_style(
            &graph, &scene
        )?)?
    );
    Ok(())
}
