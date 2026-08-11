// =========================================
// =========================================
// crates/motionloom/examples/inspect_glb_skeleton.rs

use std::path::PathBuf;

use motionloom::api::inspect_glb_skeleton_path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: inspect_glb_skeleton <model.glb>")?;
    let report = inspect_glb_skeleton_path(&path)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
