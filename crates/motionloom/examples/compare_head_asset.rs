// =========================================
// =========================================
// crates/motionloom/examples/compare_head_asset.rs

use std::path::PathBuf;

use motionloom::api::compare_glb_head_to_head_asset_path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reference_glb = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: compare_head_asset <reference.glb> <candidate.motionloom> [asset-id]")?;
    let candidate_graph = std::env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .ok_or("usage: compare_head_asset <reference.glb> <candidate.motionloom> [asset-id]")?;
    let candidate_id = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "s86_head".to_string());
    let report =
        compare_glb_head_to_head_asset_path(&reference_glb, &candidate_graph, &candidate_id)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
