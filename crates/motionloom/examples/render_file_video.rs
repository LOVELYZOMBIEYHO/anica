use std::{env, fs, path::PathBuf};

use motionloom::{
    SceneRenderProfile, render_motionloom_document_to_video_with_progress, set_scene_asset_roots,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).expect("expected MotionLoom file path");
    let output = env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/private/tmp/motionloom_video_check.mp4"));
    let profile = env::args()
        .nth(3)
        .as_deref()
        .map(parse_profile)
        .transpose()?
        .unwrap_or(SceneRenderProfile::Gpu);
    let ffmpeg = env::var("ANICA_FFMPEG")
        .unwrap_or_else(|_| "tools/runtime/current/macos/ffmpeg/bin/ffmpeg".to_string());

    let script = fs::read_to_string(&path)?;
    if let Some(parent) = PathBuf::from(&path).parent() {
        set_scene_asset_roots(vec![parent.to_path_buf()]);
    }
    pollster::block_on(render_motionloom_document_to_video_with_progress(
        &ffmpeg,
        &script,
        "examples/motionloom/world",
        &output,
        profile,
        15,
        |progress| {
            println!(
                "progress {}/{}",
                progress.rendered_frames(),
                progress.total_frames()
            );
        },
    ))?;

    println!("saved {}", output.display());
    Ok(())
}

fn parse_profile(value: &str) -> Result<SceneRenderProfile, String> {
    match value {
        "cpu" => Ok(SceneRenderProfile::Cpu),
        "gpu" => Ok(SceneRenderProfile::Gpu),
        "prores" => Ok(SceneRenderProfile::GpuProRes),
        "prores4444" | "prores-4444" => Ok(SceneRenderProfile::GpuProRes4444),
        "png" | "png-sequence" => Ok(SceneRenderProfile::GpuPngSequence),
        other => Err(format!("unknown profile: {other}")),
    }
}
