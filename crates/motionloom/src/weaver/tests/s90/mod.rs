// =========================================
// =========================================
// crates/motionloom/src/weaver/tests/s90/mod.rs

use crate::weaver::*;

fn job() -> RenderJob {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let mut job = RenderJob::new(
        workspace.join("motionloom-example/showcase/s-000090/main.motionloom"),
        QualityPreset::Ultra,
    );
    job.scene_id = "S90LeatherShoe".into();
    job.render_style = "shoe_filmic_physical".into();
    let width = std::env::var("WEAVER_TEST_WIDTH")
        .map(|value| value.parse().expect("numeric width"))
        .unwrap_or(480);
    let height = std::env::var("WEAVER_TEST_HEIGHT")
        .map(|value| value.parse().expect("numeric height"))
        .unwrap_or(272);
    let samples = std::env::var("WEAVER_TEST_SAMPLES")
        .map(|value| value.parse().expect("numeric samples"))
        .unwrap_or(64);
    job.resolution = [width, height];
    job.sampling.min_samples = samples;
    job.sampling.max_samples = samples;
    job.sampling.batch_samples = 4;
    job.lens.f_stop = 2.8;
    job.lens.focus_distance = 0.685;
    job.frame = 0;
    job.denoiser_library = std::env::var_os("WEAVER_DENOISER_LIBRARY").map(Into::into);
    job.output = workspace.join(".render-output/weaver-s90");
    job
}

#[test]
#[ignore = "GPU and S90 shoe asset; explicit Weaver material benchmark"]
fn s90_shoe_render() {
    let job = job();
    let mut last_tile = u32::MAX;
    let report = pollster::block_on(render(&job, &CancellationToken::default(), |progress| {
        if progress.completed_tiles != last_tile {
            eprintln!(
                "Weaver tile {}/{} ({:.1}s)",
                progress.completed_tiles, progress.total_tiles, progress.elapsed_seconds
            );
            last_tile = progress.completed_tiles;
        }
    }))
    .unwrap();
    assert_ne!(report.status, "cancelled");
    let display = image::open(report.output.join("display.png")).unwrap();
    assert_eq!(display.width(), job.resolution[0]);
    assert_eq!(display.height(), job.resolution[1]);
    let beauty = image::open(report.output.join("beauty.exr"))
        .unwrap()
        .to_rgb32f();
    assert!(
        beauty
            .pixels()
            .flat_map(|pixel| pixel.0)
            .all(f32::is_finite)
    );
    let mean = beauty
        .pixels()
        .map(|pixel| pixel.0.into_iter().sum::<f32>() / 3.0)
        .sum::<f32>()
        / (beauty.width() * beauty.height()) as f32;
    assert!(
        mean > 0.001 && mean < 100.0,
        "unexpected shoe render mean: {mean}"
    );
    let samples = image::open(report.output.join("sample-count.exr"))
        .unwrap()
        .to_rgb32f();
    assert!(
        samples
            .pixels()
            .all(|pixel| pixel[0] == job.sampling.max_samples as f32)
    );
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
}
