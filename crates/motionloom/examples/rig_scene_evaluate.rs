// =========================================
// =========================================
// crates/motionloom/examples/rig_scene_evaluate.rs

use motionloom::api::{
    RigEvaluationRequest, RigReportDetail, RigSamplePoint, SceneRenderProfile, SceneRenderer,
    parse_graph_script, rig_evaluation_report_json, set_scene_asset_roots,
};
use std::{env, fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene_path = env::args().nth(1).ok_or(
        "usage: rig_scene_evaluate <scene.motionloom> <actor> <frame|phase:action:value> [output.json]",
    )?;
    let actor_id = env::args().nth(2).ok_or("missing actor id")?;
    let sample = env::args().nth(3).ok_or("missing sample")?;
    let output = env::args().nth(4);
    let script = fs::read_to_string(&scene_path)?;
    if let Some(parent) = Path::new(&scene_path).parent() {
        set_scene_asset_roots(vec![parent.to_path_buf()]);
    }
    let graph = parse_graph_script(&script)?;
    let sample = parse_sample(&sample)?;
    let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Gpu))?;
    let mut report = pollster::block_on(renderer.evaluate_rig(
        &graph,
        &RigEvaluationRequest {
            actor_id,
            sample,
            detail: RigReportDetail::Body,
            include_screen_projection: true,
            include_matrices: false,
        },
    ))?;
    report.provenance.document = Some(scene_path.clone());
    let json = rig_evaluation_report_json(&report);
    if let Some(output) = output {
        fs::write(&output, &json)?;
        println!("saved Scene rig evaluation to {output}");
    } else {
        println!("{json}");
    }
    Ok(())
}

fn parse_sample(value: &str) -> Result<RigSamplePoint, Box<dyn std::error::Error>> {
    if let Some(value) = value.strip_prefix("phase:") {
        let (action_id, phase) = value
            .rsplit_once(':')
            .ok_or("phase sample must use phase:action:value")?;
        return Ok(RigSamplePoint::ActionPhase {
            action_id: action_id.into(),
            phase: phase.parse()?,
        });
    }
    if let Some(value) = value.strip_prefix("time:") {
        return Ok(RigSamplePoint::Time {
            time_sec: value.parse()?,
        });
    }
    Ok(RigSamplePoint::Frame {
        frame: value.parse()?,
    })
}
