// =========================================
// =========================================
// crates/motionloom/src/weaver/tests/s89/jobs.rs

use crate::weaver::*;

pub(super) fn baseline() -> RenderJob {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let mut job = RenderJob::new(
        workspace.join("motionloom-example/showcase/s-000089/main.motionloom"),
        QualityPreset::Ultra,
    );
    job.scene_id = "S89MistCourtyard".into();
    job.render_style = "courtyard_filmic_physical".into();
    let width = std::env::var("WEAVER_TEST_WIDTH")
        .map(|s| s.parse::<u32>().expect("numeric width"))
        .unwrap_or(320);
    let samples = std::env::var("WEAVER_TEST_SAMPLES")
        .map(|s| s.parse::<u32>().expect("numeric samples"))
        .unwrap_or(8);
    // Allow exact output dimensions instead of forcing a rounded 16:9 height.
    let height = std::env::var("WEAVER_TEST_HEIGHT")
        .map(|s| s.parse::<u32>().expect("numeric height"))
        .unwrap_or(width * 9 / 16);
    job.resolution = [width, height];
    if let Ok(region) = std::env::var("WEAVER_TEST_REGION") {
        job.region = Some(
            region
                .split(',')
                .map(|s| s.parse::<u32>().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .expect("x,y,w,h"),
        );
    }
    job.sampling.min_samples = samples;
    job.sampling.max_samples = samples;
    if let Ok(value) = std::env::var("WEAVER_TEST_BATCH") {
        job.sampling.batch_samples = value.parse().expect("numeric batch size");
    }
    if std::env::var_os("WEAVER_TEST_CALIBRATED").is_some() {
        // Preserve geometry/camera framing; replace preview fill with world light.
        job.lighting.environment_intensity = Some(0.65);
        job.lighting.light_intensities.insert("sun".into(), 2.5);
        job.lighting
            .light_intensities
            .insert("sky_fill".into(), 0.0);
        job.sun_angular_diameter_degrees = 1.5;
        job.lens.f_stop = 1.4;
        job.lens.focus_distance = 16.2;
    }
    // Hold every other input fixed when isolating direct versus indirect light.
    if let Ok(value) = std::env::var("WEAVER_TEST_BOUNCES") {
        let total = value.parse::<u32>().expect("numeric bounce count");
        job.light_paths.total = total;
        job.light_paths.diffuse = job.light_paths.diffuse.min(total);
        job.light_paths.glossy = job.light_paths.glossy.min(total);
        job.light_paths.transmission = job.light_paths.transmission.min(total);
        job.light_paths.roulette_start = job.light_paths.roulette_start.min(total);
    }
    if let Ok(value) = std::env::var("WEAVER_TEST_FSTOP") {
        job.lens.f_stop = value.parse().expect("numeric f-stop");
    }
    if let Ok(value) = std::env::var("WEAVER_TEST_FOCUS") {
        job.lens.focus_distance = value.parse().expect("numeric focus distance");
    }
    job.frame = std::env::var("WEAVER_TEST_FRAME")
        .map(|s| s.parse::<u32>().expect("numeric frame"))
        .unwrap_or(0);
    job.denoiser_library = std::env::var_os("WEAVER_DENOISER_LIBRARY").map(Into::into);
    if std::env::var_os("WEAVER_TEST_VOLUME").is_some() {
        job.volume = Some(Volume {
            bounds_min: [-200.0, -10.0, -200.0],
            bounds_max: [200.0, 90.0, -16.0],
            extinction: 0.012,
            albedo: [0.9, 0.94, 0.98],
            anisotropy: 0.25,
            max_bounces: 4,
        });
    }
    job.allow_legacy_fog_omission = true;
    job.output = workspace.join(".render-output/weaver-s89");
    job
}
