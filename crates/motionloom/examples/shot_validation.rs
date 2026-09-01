use std::{env, fs, path::PathBuf};

use motionloom::api::{
    SceneRenderProfile, SceneRenderer, ShotValidationOptions, parse_graph_script,
    set_scene_asset_roots,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: shot_validation <file.motionloom> [cinematic|realtime|strict] [cpu|gpu]")?;
    let preset = env::args()
        .nth(2)
        .unwrap_or_else(|| "cinematic".to_string());
    let profile = env::args().nth(3).unwrap_or_else(|| "gpu".to_string());
    let options = match preset.as_str() {
        "cinematic" => ShotValidationOptions::cinematic(),
        "realtime" => ShotValidationOptions::realtime_preview(),
        "strict" => ShotValidationOptions::ci_strict(),
        other => return Err(format!("unknown validation preset: {other}").into()),
    };
    let profile = match profile.as_str() {
        "cpu" => SceneRenderProfile::Cpu,
        "gpu" => SceneRenderProfile::Gpu,
        other => return Err(format!("unknown render profile: {other}").into()),
    };
    let script = fs::read_to_string(&path)?;
    if let Some(parent) = PathBuf::from(&path).parent() {
        set_scene_asset_roots(vec![parent.to_path_buf()]);
    }
    let graph = parse_graph_script(&script)?;
    let mut renderer = pollster::block_on(SceneRenderer::new(profile))?;
    let report = pollster::block_on(renderer.validate_shots(&graph, options))?;
    println!("{}", report.to_json()?);
    Ok(())
}
