// =========================================
// =========================================
// crates/motionloom/examples/apply_mesh_proposal.rs

use motionloom::api::mesh_reference::{MeshAssetProposal, apply_mesh_asset_proposal};
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().collect();
    if arguments.len() != 4 {
        return Err(
            "usage: apply_mesh_proposal SOURCE.motionloom PROPOSAL.json OUTPUT.motionloom".into(),
        );
    }
    let source = fs::read_to_string(&arguments[1])?;
    let proposal: MeshAssetProposal = serde_json::from_str(&fs::read_to_string(&arguments[2])?)?;
    let result = apply_mesh_asset_proposal(&source, &proposal)?;
    fs::write(&arguments[3], &result.source)?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "output": arguments[3],
            "appliedChanges": result.applied_changes,
            "sourceFingerprint": result.source_fingerprint,
            "topologySignature": result.topology_signature,
            "topologyValid": result.topology.valid,
        }))?
    );
    Ok(())
}
