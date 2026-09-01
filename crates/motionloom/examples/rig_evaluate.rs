// =========================================
// =========================================
// crates/motionloom/examples/rig_evaluate.rs

use motionloom::api::{
    RigEvaluationRequest, RigReportDetail, RigSamplePoint, evaluate_world_actor_rig,
    parse_world_graph_script, rig_evaluation_report_json,
};
use motionloom::experimental::load_glb_mesh_data;
use std::{env, fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world_path = env::args().nth(1).ok_or(
        "usage: rig_evaluate <world.motionloom> <model.glb> <actor> <frame|phase:action:value> [output.json]",
    )?;
    let glb_path = env::args().nth(2).ok_or("missing model.glb")?;
    let actor_id = env::args().nth(3).ok_or("missing actor id")?;
    let sample_arg = env::args().nth(4).ok_or("missing sample")?;
    let output = env::args().nth(5);
    let script = fs::read_to_string(&world_path)?;
    let graph = parse_world_graph_script(&script)?;
    let mesh = load_glb_mesh_data(Path::new(&glb_path))?;
    let sample = parse_sample(&sample_arg)?;
    let report = evaluate_world_actor_rig(
        &graph,
        &mesh,
        &RigEvaluationRequest {
            actor_id,
            sample,
            detail: RigReportDetail::Body,
            include_screen_projection: false,
            include_matrices: false,
        },
    )?;
    let json = rig_evaluation_report_json(&report);
    if let Some(output) = output {
        fs::write(&output, &json)?;
        println!("saved rig evaluation to {output}");
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
    Ok(RigSamplePoint::Frame {
        frame: value.parse()?,
    })
}
