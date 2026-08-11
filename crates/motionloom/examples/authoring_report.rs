// =========================================
// =========================================
// crates/motionloom/examples/authoring_report.rs

use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: authoring_report <file.motionloom> [target]")?;
    let target = env::args().nth(2).unwrap_or_else(|| "auto".to_string());
    let script = fs::read_to_string(path)?;
    println!(
        "{}",
        motionloom::api::motionloom_analyze_script_for_target_json(&script, &target)
    );
    Ok(())
}
