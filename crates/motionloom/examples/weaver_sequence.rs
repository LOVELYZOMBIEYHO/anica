// =========================================
// =========================================
// crates/motionloom/examples/weaver_sequence.rs

#[cfg(all(feature = "weaver", not(target_arch = "wasm32")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use motionloom::api::weaver::{
        CancellationToken, MasterSequenceSettings, QualityPreset, RenderJob, SceneOutputMode,
        SequenceDenoiseMode, render_master_sequence,
    };
    use std::path::PathBuf;

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: weaver_sequence <scene.motionloom> [--frames START:END]\n\
         \x20      [--scene-id <id|auto>] [--style <id|auto>] [--size WxH] [--samples N]\n\
         \x20      [--out DIR] [--mips] [--transmission-stopgap] [--no-prores]\n\
         \x20      [--no-preview] [--no-audio] [--no-scene-composite] [--temporal-denoise]";
    let scene = raw.first().cloned().ok_or(usage)?;
    let mut scene_id = "auto".to_string();
    let mut render_style = "auto".to_string();
    let mut frames = None;
    let mut width = 1920;
    let mut height = 1080;
    let mut samples = None;
    let mut output = PathBuf::from(".render-output/weaver");
    let mut mips = false;
    let mut transmission_stopgap = false;
    let mut settings = MasterSequenceSettings::default();
    let mut it = raw.iter().skip(1);
    while let Some(argument) = it.next() {
        let mut value = || it.next().map(String::as_str).unwrap_or("");
        match argument.as_str() {
            "-h" | "--help" => return Err(usage.into()),
            "--scene-id" => scene_id = value().to_string(),
            "--style" => render_style = value().to_string(),
            "--frames" => {
                let range = value();
                let (start, end) = range.split_once(':').expect("--frames must be START:END");
                frames = Some(start.parse::<u32>()?..=end.parse::<u32>()?);
            }
            "--samples" => samples = Some(value().parse::<u32>()?),
            "--out" => output = PathBuf::from(value()),
            "--mips" => mips = true,
            "--transmission-stopgap" => transmission_stopgap = true,
            "--no-prores" => settings.encode_prores = false,
            "--no-preview" => settings.encode_preview = false,
            "--no-audio" => settings.include_audio = false,
            "--no-scene-composite" => settings.write_scene_composite = false,
            "--temporal-denoise" => settings.denoise_mode = SequenceDenoiseMode::Temporal,
            "--size" => {
                let size = value();
                let (w, h) = size.split_once(['x', 'X']).expect("--size must be WxH");
                width = w.parse()?;
                height = h.parse()?;
            }
            other => return Err(format!("unknown flag `{other}`\n{usage}").into()),
        }
    }
    // Omitted ranges mean the complete authored timeline, which is the normal
    // movie-export behavior; an explicit range remains useful for tests.
    let frames = match frames {
        Some(frames) => frames,
        None => {
            let script = std::fs::read_to_string(&scene)?;
            let graph = motionloom::api::parse_graph_script(&script)?;
            let frame_count = (graph.duration_ms as f64 * graph.fps as f64 / 1_000.0).ceil() as u32;
            if frame_count == 0 {
                return Err("scene duration produces no frames".into());
            }
            0..=frame_count - 1
        }
    };
    let mut job = RenderJob::new(&scene, QualityPreset::Ultra);
    job.scene_id = scene_id;
    job.render_style = render_style;
    job.resolution = [width, height];
    job.output_mode = SceneOutputMode::CompositeScene;
    job.output = output;
    job.memory_budget_mib = 8192;
    job.texture_mips = mips;
    job.allow_transmission_stopgap = transmission_stopgap;
    if let Some(samples) = samples {
        job.sampling.min_samples = samples;
        job.sampling.max_samples = samples;
        job.sampling.batch_samples = samples.min(4).max(1);
    }
    job.validate()?;

    let mut last = (u32::MAX, u32::MAX);
    let report = pollster::block_on(render_master_sequence(
        &job,
        frames,
        &settings,
        &CancellationToken::default(),
        |progress| {
            let current = (progress.frame, progress.render.completed_tiles);
            if current != last {
                last = current;
                eprintln!(
                    "frame {} ({}/{}) tile {}/{} ({:.1}s)",
                    progress.frame,
                    progress.completed_frames,
                    progress.total_frames,
                    progress.render.completed_tiles,
                    progress.render.total_tiles,
                    progress.render.elapsed_seconds,
                );
            }
        },
    ))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(all(feature = "weaver", not(target_arch = "wasm32"))))]
fn main() {
    eprintln!(
        "weaver_sequence requires: cargo run --features weaver --example weaver_sequence -- ..."
    );
}
