// =========================================
// =========================================
// crates/motionloom/src/weaver/tests/s89/mod.rs

mod compare;
mod jobs;
use crate::weaver::*;

#[test]
#[ignore = "GPU/S89 per-dispatch sample accounting regression"]
fn every_pixel_receives_one_sample_batch() {
    let mut job = jobs::baseline();
    job.resolution = [3840, 2160];
    job.region = Some([540, 1390, 128, 128]);
    job.frame = 240;
    job.sampling.min_samples = 128;
    job.sampling.max_samples = 128;
    job.sampling.batch_samples = 4;
    job.denoiser_library = None;
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    job.output = std::env::temp_dir().join(format!("weaver-batch-{}-{unique}", std::process::id()));
    let cancel = CancellationToken::default();
    let mut callbacks = 0;
    let report = pollster::block_on(render(&job, &cancel, |_| {
        callbacks += 1;
        // The first callback is before dispatch; the second observes its result.
        if callbacks == 2 {
            cancel.cancel();
        }
    }))
    .unwrap();
    let raw = std::fs::read(report.output.join("checkpoints/0-0.film")).unwrap();
    let values = crate::weaver::output::floats(&raw);
    let counts: Vec<_> = values.chunks_exact(16).map(|f| f[3]).collect();
    let wrong = counts.iter().filter(|&&count| count != 4.0).count();
    eprintln!(
        "batch accounting: wrong={wrong}, min={}, max={}",
        counts.iter().copied().fold(f32::INFINITY, f32::min),
        counts.iter().copied().fold(0.0, f32::max)
    );
    // Remove only the uniquely named directory owned by this fixture.
    std::fs::remove_dir_all(&job.output).unwrap();
    assert_eq!(wrong, 0);
}

#[test]
#[ignore = "GPU/S89 regression for an environment-pole NaN at 4K frame 240"]
fn environment_pole_remains_finite() {
    let mut job = jobs::baseline();
    job.resolution = [3840, 2160];
    job.region = Some([748, 505, 1, 1]);
    job.frame = 240;
    job.sampling.min_samples = 64;
    job.sampling.max_samples = 64;
    job.denoiser_library = None;
    job.volume = Some(Volume {
        bounds_min: [-200.0, -10.0, -200.0],
        bounds_max: [200.0, 90.0, -16.0],
        extinction: 0.012,
        albedo: [0.9, 0.94, 0.98],
        anisotropy: 0.25,
        max_bounces: 4,
    });
    let report = pollster::block_on(render(&job, &CancellationToken::default(), |_| {})).unwrap();
    let raw = image::open(report.output.join("beauty.exr"))
        .unwrap()
        .to_rgb32f();
    assert!(raw.pixels().flat_map(|p| p.0).all(f32::is_finite));
    let count = image::open(report.output.join("sample-count.exr"))
        .unwrap()
        .to_rgb32f();
    assert_eq!(count.get_pixel(0, 0)[0], 64.0);
}

#[test]
#[ignore = "explicit GPU/S89 benchmark; WEAVER_TEST_WIDTH and WEAVER_TEST_SAMPLES control cost"]
fn s89_gpu_baseline() {
    let job = jobs::baseline();
    let mut last_tile = u32::MAX;
    let result = pollster::block_on(render(&job, &CancellationToken::default(), |p| {
        if p.completed_tiles != last_tile {
            eprintln!(
                "Weaver tile {}/{} ({:.1}s)",
                p.completed_tiles, p.total_tiles, p.elapsed_seconds
            );
            last_tile = p.completed_tiles;
        }
    }))
    .unwrap();
    assert_ne!(result.status, "cancelled");
    let image = image::open(result.output.join("beauty.exr"))
        .unwrap()
        .to_rgb32f();
    let size = job.region.map(|r| [r[2], r[3]]).unwrap_or(job.resolution);
    assert_eq!([image.width(), image.height()], size);
    let samples = image::open(result.output.join("sample-count.exr"))
        .unwrap()
        .to_rgb32f();
    assert!(samples.pixels().all(
        |p| p[0] >= job.sampling.min_samples as f32 && p[0] <= job.sampling.max_samples as f32
    ));
    assert!(image.pixels().flat_map(|p| p.0).all(f32::is_finite));
    let mean = image
        .pixels()
        .map(|p| (p[0] + p[1] + p[2]) / 3.0)
        .sum::<f32>()
        / (image.width() * image.height()) as f32;
    assert!(
        mean > 0.001 && mean < 100.0,
        "unexpected black/exploded image: {mean}"
    );
    eprintln!("{}", serde_json::to_string_pretty(&result).unwrap());
}

#[test]
#[ignore = "GPU and S89 assets; validates cancellation and real disk checkpoint resume"]
fn cancelled_job_resumes_without_changing_samples() {
    let mut job = jobs::baseline();
    job.resolution = [32, 18];
    // Benchmark crop overrides must not alter this fixed resume fixture.
    job.region = None;
    job.sampling.min_samples = 8;
    job.sampling.max_samples = 8;
    job.volume = None;
    job.denoiser_library = None;
    let temp = std::env::temp_dir().join(format!("weaver-resume-{}", std::process::id()));
    job.output = temp.clone();
    let cancel = CancellationToken::default();
    let first = pollster::block_on(render(&job, &cancel, |p| {
        if p.tile_min_samples >= 4 {
            cancel.cancel();
        }
    }))
    .unwrap();
    assert_eq!(first.status, "cancelled");
    assert!(!first.output.join("render.lock").exists());
    let mut first_samples = None;
    let resumed = pollster::block_on(render(&job, &CancellationToken::default(), |p| {
        if first_samples.is_none() {
            first_samples = Some(p.tile_min_samples);
        }
    }))
    .unwrap();
    assert_eq!(first_samples, Some(4));
    assert_eq!(resumed.status, "sample_limit_reached");
    let expected = std::fs::read(resumed.output.join("checkpoints/0-0.film")).unwrap();
    job.output = temp.join("fresh");
    let fresh = pollster::block_on(render(&job, &CancellationToken::default(), |_| {})).unwrap();
    assert_eq!(
        expected,
        std::fs::read(fresh.output.join("checkpoints/0-0.film")).unwrap()
    );
    // Only this test's explicit temporary directory is removed.
    std::fs::remove_dir_all(temp).unwrap();
}
