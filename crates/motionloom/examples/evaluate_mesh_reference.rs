// =========================================
// =========================================
// crates/motionloom/examples/evaluate_mesh_reference.rs

use motionloom::api::mesh_reference::{
    ImageReferenceAnalysis, MESH_REFERENCE_SCHEMA_VERSION, MeshReferenceOptions, MeshReferenceSet,
    MeshReferenceView, evaluate_mesh_asset_reference, mesh_evaluation_difference_png,
    mesh_evaluation_overlay_png,
};
use std::{env, fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().collect();
    if arguments.len() != 4 && arguments.len() != 8 {
        return Err("usage: evaluate_mesh_reference SOURCE.motionloom REFERENCES.json OUTPUT_DIR\n       evaluate_mesh_reference SOURCE.motionloom --analysis ANALYSIS.json ASSET_ID MODEL_ID FRAME OUTPUT_DIR".into());
    }
    let source = fs::read_to_string(&arguments[1])?;
    let request: MeshReferenceSet = if arguments.len() == 4 {
        serde_json::from_str(&fs::read_to_string(&arguments[2])?)?
    } else {
        if arguments[2] != "--analysis" {
            return Err("the extended form requires --analysis".into());
        }
        let analysis: ImageReferenceAnalysis =
            serde_json::from_str(&fs::read_to_string(&arguments[3])?)?;
        MeshReferenceSet {
            schema_version: MESH_REFERENCE_SCHEMA_VERSION.into(),
            target_asset_id: arguments[4].clone(),
            target_model_id: arguments[5].clone(),
            references: vec![MeshReferenceView {
                id: analysis.view.clone(),
                frame: arguments[6].parse()?,
                analysis,
            }],
            options: MeshReferenceOptions::default(),
        }
    };
    let evaluation = pollster::block_on(evaluate_mesh_asset_reference(&source, &request))?;
    let output = Path::new(if arguments.len() == 4 {
        &arguments[3]
    } else {
        &arguments[7]
    });
    fs::create_dir_all(output)?;
    let report = output.join("evaluation.json");
    fs::write(&report, serde_json::to_vec_pretty(&evaluation)?)?;
    for (reference, view) in request.references.iter().zip(&evaluation.views) {
        let name: String = view
            .id
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                    character
                } else {
                    '_'
                }
            })
            .collect();
        fs::write(
            output.join(format!("{name}-overlay.png")),
            mesh_evaluation_overlay_png(&reference.analysis, view)?,
        )?;
        fs::write(
            output.join(format!("{name}-difference.png")),
            mesh_evaluation_difference_png(&reference.analysis, view)?,
        )?;
    }
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "report": report,
            "objective": evaluation.objective,
            "sourceFingerprint": evaluation.source_fingerprint,
            "topologySignature": evaluation.topology_signature,
            "cameraFingerprint": evaluation.camera_fingerprint,
            "views": evaluation.views.iter().map(|view| serde_json::json!({
                "id": view.id,
                "maskIou": view.mask_iou,
                "meanEdgeDistancePx": view.mean_edge_distance_px,
                "error": view.error,
            })).collect::<Vec<_>>(),
        }))?
    );
    Ok(())
}
