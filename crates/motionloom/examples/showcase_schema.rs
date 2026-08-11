// =========================================
// =========================================
// crates/motionloom/examples/showcase_schema.rs

use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: showcase_schema <file.motionloom>")?;
    let script = fs::read_to_string(path)?;
    println!(
        "{}",
        motionloom::api::motionloom_showcase_schema_json(&script)
    );
    Ok(())
}
