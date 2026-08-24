// =========================================
// =========================================
// crates/motionloom/examples/build_showcase_schemas.rs

use std::{collections::BTreeMap, env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let showcase_root = PathBuf::from(
        env::args()
            .nth(1)
            .unwrap_or_else(|| "../motionloom-example/showcase".to_string()),
    );
    let mut generated = 0usize;
    let mut clean = 0usize;
    let mut needs_review = 0usize;
    let mut needs_repair = 0usize;
    let mut unrenderable = 0usize;
    let mut diagnostic_codes = BTreeMap::<String, usize>::new();
    // Accept either the showcase root or one showcase directory so targeted
    // updates do not rewrite unrelated generated schemas in a dirty worktree.
    let directories = if showcase_root.join("main.motionloom").is_file() {
        vec![showcase_root.clone()]
    } else {
        fs::read_dir(&showcase_root)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect()
    };
    for directory in directories {
        if !directory.is_dir() {
            continue;
        }
        let main = directory.join("main.motionloom");
        if !main.is_file() {
            continue;
        }
        let script = fs::read_to_string(&main)?;
        let schema = motionloom::api::motionloom_showcase_schema_json(&script);
        fs::write(directory.join("schema.json"), format!("{schema}\n"))?;
        let report = motionloom::api::analyze_motionloom_script(&script);
        for diagnostic in &report.diagnostics {
            *diagnostic_codes.entry(diagnostic.code.clone()).or_default() += 1;
        }
        match report.status {
            motionloom::api::AuthoringStatus::Clean => clean += 1,
            motionloom::api::AuthoringStatus::NeedsReview => needs_review += 1,
            motionloom::api::AuthoringStatus::NeedsRepair => needs_repair += 1,
            motionloom::api::AuthoringStatus::Unrenderable => unrenderable += 1,
        }
        if report.summary.errors > 0 {
            eprintln!(
                "{}: {} error(s), {} warning(s)",
                main.display(),
                report.summary.errors,
                report.summary.warnings
            );
            for diagnostic in report.diagnostics.iter().filter(|diagnostic| {
                diagnostic.severity == motionloom::api::AuthoringDiagnosticSeverity::Error
            }) {
                eprintln!(
                    "  {}:{} [{}] {}",
                    diagnostic.line, diagnostic.column, diagnostic.code, diagnostic.message
                );
            }
        }
        generated += 1;
    }
    println!(
        "generated {generated} showcase schema file(s) under {}",
        showcase_root.display()
    );
    println!(
        "analysis: {clean} clean, {needs_review} need review, {needs_repair} need repair, {unrenderable} unrenderable"
    );
    if !diagnostic_codes.is_empty() {
        println!("diagnostic codes: {diagnostic_codes:?}");
    }
    Ok(())
}
