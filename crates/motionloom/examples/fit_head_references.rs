// =========================================
// =========================================
// crates/motionloom/examples/fit_head_references.rs

use motionloom::api::head_fitting::*;
use std::{fs, path::PathBuf};

// Keep filesystem and optional GPU work in the host, outside the reusable fitting core.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err(
            "usage: fit_head_references SOURCE.motionloom REFERENCES.json OUTPUT_DIR [--gpu]"
                .into(),
        );
    }
    let source = fs::read_to_string(&args[1])?;
    let request: HeadReferenceSet = serde_json::from_str(&fs::read_to_string(&args[2])?)?;
    let output = PathBuf::from(&args[3]);
    fs::create_dir_all(&output)?;
    let validation = validate_head_reference_set(&request);
    fs::write(
        output.join("validation.json"),
        serde_json::to_string_pretty(&validation)?,
    )?;
    let proposal = fit_head_asset_to_references(&source, &request, || false)?;
    let candidate = apply_head_fit_proposal(&source, &proposal)?;
    fs::write(output.join("candidate.motionloom"), &candidate)?;
    fs::write(
        output.join("report.json"),
        serde_json::to_string_pretty(&proposal)?,
    )?;
    fs::write(
        output.join("comparison.svg"),
        comparison_svg(&proposal.before_metrics, &proposal.after_metrics),
    )?;
    let previews = head_fit_preview_scripts(&candidate, &request.target_asset_id)?;
    // Write every review scene before optional GPU work so adapter failures preserve the authoring output.
    for (name, script) in &previews {
        fs::write(output.join(format!("preview-{name}.motionloom")), script)?;
    }
    for (name, script) in previews {
        if args.iter().any(|s| s == "--gpu") {
            let graph = motionloom::api::parse_graph_script(&script)?;
            let frame = pollster::block_on(motionloom::api::render_scene_graph_frame(
                &graph,
                0,
                motionloom::api::SceneRenderProfile::Gpu,
            ))?;
            frame.save(output.join(format!("gpu-{name}.png")))?;
        }
    }
    println!(
        "{}: {:.6} -> {:.6}; {}",
        proposal.status,
        proposal.before_metrics.objective,
        proposal.after_metrics.objective,
        proposal.stop_reason
    );
    Ok(())
}
