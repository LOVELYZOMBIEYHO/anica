// =========================================
// =========================================
// crates/motionloom/examples/rig_compare.rs

use motionloom::api::{
    RigComparisonOptions, RigEvaluationReport, compare_humanoid_poses, rig_comparison_report_json,
};
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reference_path = env::args()
        .nth(1)
        .ok_or("usage: rig_compare <reference.json> <candidate.json> [output.json]")?;
    let candidate_path = env::args().nth(2).ok_or("missing candidate report")?;
    let output = env::args().nth(3);
    let reference: RigEvaluationReport =
        serde_json::from_str(&fs::read_to_string(reference_path)?)?;
    let candidate: RigEvaluationReport =
        serde_json::from_str(&fs::read_to_string(candidate_path)?)?;
    let report = compare_humanoid_poses(&reference, &candidate, RigComparisonOptions::default());
    let json = rig_comparison_report_json(&report);
    if let Some(output) = output {
        fs::write(&output, &json)?;
        println!("saved rig comparison to {output}");
    } else {
        println!("RIG COMPARISON: {:?}", report.status);
        println!("Assets: {:?}", report.root_cause.category);
        println!("Compared bones: {}", report.summary.compared_bones);
        println!("Warnings: {}", report.summary.warning_bones);
        println!("Errors: {}", report.summary.error_bones);
        println!(
            "Maximum angular error: {:.3} deg",
            report.summary.max_angular_error_deg
        );
        println!("{json}");
    }
    if matches!(report.status, motionloom::RigComparisonStatus::Match) {
        Ok(())
    } else {
        std::process::exit(1)
    }
}
