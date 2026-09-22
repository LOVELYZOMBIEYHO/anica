// =========================================
// =========================================
// crates/motionloom/examples/weaver_frame.rs

#[cfg(all(feature = "weaver", not(target_arch = "wasm32")))]
/// Prefer the workspace-root `.render-output` so renders never land inside the
/// `anica` package; fall back to the current directory when no workspace marker
/// is found. Overridable with `--out`.
fn default_output_dir() -> std::path::PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    loop {
        let workspace = dir.join("anica").is_dir() && dir.join("motionloom-example").is_dir();
        if workspace || dir.join(".render-output").is_dir() {
            return dir.join(".render-output").join("weaver");
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }
    std::path::PathBuf::from(".render-output/weaver")
}

#[cfg(all(feature = "weaver", not(target_arch = "wasm32")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;

    use motionloom::api::weaver::{CancellationToken, QualityPreset, RenderJob, render};

    // Scene id and style both default to `auto`, so a plain document render
    // never requires knowing authored ids.
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: weaver_frame <scene.motionloom> [--scene-id <id|auto>] [--style <id|auto>]\n\
         \x20      [--frame N] [--size WxH] [--samples N] [--out DIR] [--f-stop F] [--focus D]\n\
         \x20      [--mips] [--composite-scene] [--transmission-stopgap]";
    let scene = raw.first().cloned().ok_or(usage)?;
    let mut scene_id = "auto".to_string();
    let mut render_style = "auto".to_string();
    let mut frame: u32 = 0;
    let mut width: u32 = 1920;
    let mut height: u32 = 1080;
    let mut samples: Option<u32> = None;
    let mut output = default_output_dir();
    let mut f_stop = 4.0f32;
    let mut focus_distance = 20.0f32;
    let mut mips = false;
    let mut composite_scene = false;
    let mut transmission_stopgap = false;
    let mut it = raw.iter().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().map(String::as_str).unwrap_or("");
        match arg.as_str() {
            "-h" | "--help" => return Err(usage.into()),
            "--scene-id" => scene_id = value().to_string(),
            "--style" => render_style = value().to_string(),
            "--frame" => frame = value().parse().expect("--frame must be an integer"),
            "--samples" => samples = Some(value().parse().expect("--samples must be an integer")),
            "--out" => output = PathBuf::from(value()),
            "--f-stop" => f_stop = value().parse().expect("--f-stop must be a number"),
            "--focus" => focus_distance = value().parse().expect("--focus must be a number"),
            "--mips" => mips = true,
            "--composite-scene" => composite_scene = true,
            "--transmission-stopgap" => transmission_stopgap = true,
            "--size" => {
                let size = value();
                let (w, h) = size.split_once(['x', 'X']).expect("--size must be WxH");
                width = w.parse().expect("width must be an integer");
                height = h.parse().expect("height must be an integer");
            }
            other => return Err(format!("unknown flag `{other}`\n{usage}").into()),
        }
    }

    let mut job = RenderJob::new(&scene, QualityPreset::Ultra);
    job.scene_id = scene_id;
    job.render_style = render_style;
    job.frame = frame;
    job.resolution = [width, height];
    job.memory_budget_mib = 8192;
    // Physical lens defaults keep imported scenes legible; override with flags.
    job.lens.f_stop = f_stop;
    job.lens.focus_distance = focus_distance;
    job.texture_mips = mips;
    if composite_scene {
        job.output_mode = motionloom::api::weaver::SceneOutputMode::CompositeScene;
    }
    job.allow_transmission_stopgap = transmission_stopgap;
    if let Some(samples) = samples {
        // Fixed sample budget keeps a preview render bounded; omit it to use the
        // Adaptive Ultra range (256..4096) instead.
        job.sampling.min_samples = samples;
        job.sampling.max_samples = samples;
        job.sampling.batch_samples = 4;
    }
    job.output = output;
    job.validate()?;

    eprintln!(
        "weaver: {} scene={} style={:?} frame={} {}x{} samples={:?} -> {}",
        scene,
        job.scene_id,
        job.render_style,
        job.frame,
        job.resolution[0],
        job.resolution[1],
        samples,
        job.output.display()
    );

    let mut last_tile = u32::MAX;
    let report = pollster::block_on(render(&job, &CancellationToken::default(), |progress| {
        if progress.completed_tiles != last_tile {
            last_tile = progress.completed_tiles;
            eprintln!(
                "weaver tile {}/{} ({:.1}s)",
                progress.completed_tiles, progress.total_tiles, progress.elapsed_seconds
            );
        }
    }))?;

    println!(
        "{} [{}] triangles={} converged={} sample_limit={} elapsed={:.1}s output={}",
        report.status,
        report.renderer,
        report.triangles,
        report.converged_pixels,
        report.sample_limit_pixels,
        report.elapsed_seconds,
        report.output.display()
    );
    for line in &report.diagnostics {
        println!("diagnostic: {line}");
    }
    Ok(())
}

#[cfg(not(all(feature = "weaver", not(target_arch = "wasm32"))))]
fn main() {
    eprintln!("weaver_frame requires: cargo run --features weaver --example weaver_frame -- ...");
}
