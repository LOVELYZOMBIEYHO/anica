// =========================================
// =========================================
// crates/motionloom/examples/analyze_image_reference.rs

use motionloom::api::mesh_reference::{
    AnalyzeImageReferenceRequest, analysis_internal_edge_png, analysis_mask_png,
    analysis_overlay_png, analyze_image_reference,
};
use std::{env, fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().collect();
    if arguments.len() != 4 {
        return Err("usage: analyze_image_reference IMAGE REQUEST.json OUTPUT_DIR".into());
    }
    let image = fs::read(&arguments[1])?;
    let request: AnalyzeImageReferenceRequest =
        serde_json::from_str(&fs::read_to_string(&arguments[2])?)?;
    let analysis = analyze_image_reference(&image, &request)?;
    let output = Path::new(&arguments[3]);
    fs::create_dir_all(output)?;
    fs::write(
        output.join("analysis.json"),
        serde_json::to_vec_pretty(&analysis)?,
    )?;
    fs::write(output.join("mask.png"), analysis_mask_png(&analysis)?)?;
    fs::write(
        output.join("internal-edges.png"),
        analysis_internal_edge_png(&analysis)?,
    )?;
    fs::write(
        output.join("overlay.png"),
        analysis_overlay_png(&image, &analysis)?,
    )?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "analysis": output.join("analysis.json"),
            "mask": output.join("mask.png"),
            "overlay": output.join("overlay.png"),
            "internalEdges": output.join("internal-edges.png"),
            "regions": analysis.regions.len(),
            "contours": analysis.contours.len(),
            "landmarks": analysis.landmarks.len(),
            "features": analysis.features.len(),
            "confidence": analysis.confidence,
        }))?
    );
    Ok(())
}
