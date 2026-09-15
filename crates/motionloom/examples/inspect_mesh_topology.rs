// =========================================
// =========================================
// crates/motionloom/examples/inspect_mesh_topology.rs

use motionloom::api::mesh_reference::{MeshProposalValidationOptions, validate_mesh_topology};
use motionloom::{PrimitiveGeometry, parse_graph_script};
use std::env;

// Prints the mesh-reference topology report for one MeshAsset, including the
// exact self-intersecting triangle pairs, so authoring runs can find the
// offending region before proposing position edits.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().collect();
    if arguments.len() != 3 {
        return Err("usage: inspect_mesh_topology <file.motionloom> <asset-id>".into());
    }
    let graph = parse_graph_script(&std::fs::read_to_string(&arguments[1])?)?;
    let asset = graph
        .assets
        .iter()
        .find(|asset| asset.id == arguments[2])
        .ok_or("asset not found")?;
    let primitive = asset.primitive().ok_or("not a primitive asset")?;
    let PrimitiveGeometry::Mesh { cage } = &primitive.geometry else {
        return Err("asset is not a MeshAsset".into());
    };
    let options = MeshProposalValidationOptions {
        allow_boundary: false,
        allow_multiple_components: false,
        ..Default::default()
    };
    let report = validate_mesh_topology(cage, &options);
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "valid": report.valid,
            "vertices": report.vertices,
            "faces": report.faces,
            "openEdges": report.open_edges,
            "nonManifoldEdges": report.non_manifold_edges,
            "connectedComponents": report.connected_components,
            "degenerateFaces": report.degenerate_faces,
            "inconsistentWindingEdges": report.inconsistent_winding_edges,
            "selfIntersections": report.self_intersections.len(),
            "signedVolume": report.signed_volume,
            "diagnostics": report.diagnostics,
        }))?
    );
    // Triangle indices returned by the validator index the fan triangulation
    // of the faces, so rebuild that list before printing the offending pairs.
    let mut triangles: Vec<[u32; 3]> = vec![];
    for face in &cage.faces {
        if face.len() < 3 {
            continue;
        }
        for index in 1..face.len() - 1 {
            triangles.push([face[0], face[index], face[index + 1]]);
        }
    }
    for pair in &report.self_intersections {
        for &index in pair {
            let triangle = triangles[index as usize];
            let points: Vec<_> = triangle
                .iter()
                .map(|&vertex| (vertex, cage.positions[vertex as usize]))
                .collect();
            println!("  tri {index} -> {points:?}");
        }
    }
    Ok(())
}
