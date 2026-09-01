// =========================================
// =========================================
// crates/motionloom/examples/rig_calibrate.rs

use motionloom::api::{RigComparisonReport, propose_rig_calibration};
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let comparison_path = env::args()
        .nth(1)
        .ok_or("usage: rig_calibrate <comparison.json> [output.json]")?;
    let output = env::args().nth(2);
    let comparison: RigComparisonReport =
        serde_json::from_str(&fs::read_to_string(comparison_path)?)?;
    let proposal = propose_rig_calibration(&comparison);
    let json = serde_json::to_string_pretty(&proposal)?;
    if let Some(output) = output {
        fs::write(&output, &json)?;
        println!("saved read-only rig calibration proposal to {output}");
    } else {
        println!("{json}");
    }
    Ok(())
}
