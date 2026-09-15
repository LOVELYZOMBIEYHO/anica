// =========================================
// =========================================
// crates/motionloom/examples/mesh_authoring.rs

use motionloom::api::mesh_authoring::{
    ApplyTopologyProposalResult, GeometryRecipe, apply_mesh_topology_proposal_json,
    execute_geometry_recipe, mesh_asset_element, mesh_authoring_schema_json,
};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().collect();
    match arguments.get(1).map(String::as_str) {
        Some("schema") => println!("{}", mesh_authoring_schema_json()),
        Some("build") if arguments.len() == 6 => {
            let recipe: GeometryRecipe = serde_json::from_str(&fs::read_to_string(&arguments[2])?)?;
            let result = execute_geometry_recipe(&recipe)?;
            let source = mesh_asset_element(&arguments[4], &arguments[5], &result.cage);
            fs::write(&arguments[3], source)?;
            eprintln!(
                "wrote {} vertices and {} faces to {}",
                result.cage.positions.len(),
                result.cage.faces.len(),
                arguments[3]
            );
        }
        Some("apply-topology") if arguments.len() == 7 => {
            let source = fs::read_to_string(&arguments[2])?;
            let result_json = apply_mesh_topology_proposal_json(
                &source,
                &fs::read_to_string(&arguments[3])?,
                &fs::read_to_string(&arguments[4])?,
                &fs::read_to_string(&arguments[5])?,
            )?;
            let result: ApplyTopologyProposalResult = serde_json::from_str(&result_json)?;
            fs::write(&arguments[6], &result.source)?;
            eprintln!(
                "wrote topology candidate {} with signature {}",
                arguments[6], result.topology_signature
            );
        }
        _ => {
            eprintln!("usage: mesh_authoring schema");
            eprintln!(
                "       mesh_authoring build RECIPE.json OUTPUT.motionloom ASSET_ID MATERIAL_ID"
            );
            eprintln!(
                "       mesh_authoring apply-topology SOURCE ANALYSES.json PROPOSAL.json EVALUATION.json OUTPUT"
            );
            std::process::exit(2);
        }
    }
    Ok(())
}
