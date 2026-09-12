// =========================================
// =========================================
// crates/motionloom/src/weaver/tests/mod.rs

use super::*;
mod physics;
mod s89;
mod s90;

#[test]
fn offline_lighting_is_optional_and_validated() {
    let mut job = RenderJob::new("scene", QualityPreset::Ultra);
    job.scene_id = "scene".into();
    let mut json = serde_json::to_value(&job).unwrap();
    json.as_object_mut().unwrap().remove("lighting");
    let legacy: RenderJob = serde_json::from_value(json).unwrap();
    assert!(legacy.lighting.light_intensities.is_empty());
    job.lighting.environment_intensity = Some(-1.0);
    assert!(job.validate().is_err());
    job.lighting.environment_intensity = Some(0.65);
    job.lighting
        .light_intensities
        .insert("sun".into(), f32::NAN);
    assert!(job.validate().is_err());
    job.lighting.light_intensities.insert("sun".into(), 2.5);
    assert!(job.validate().is_ok());
}

#[test]
fn denoised_export_preserves_hdr_without_duplicate_auxiliaries() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("weaver-output-{}-{unique}", std::process::id()));
    let pixels = vec![4.0, 0.5, 0.25, 0.0, 1.0, 2.0];
    super::output::save_denoised(
        &dir,
        [2, 1],
        pixels.clone(),
        &crate::world::WorldLighting::default(),
    )
    .unwrap();
    let hdr = image::open(dir.join("beauty.exr")).unwrap().to_rgb32f();
    assert_eq!(hdr.into_raw(), pixels);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
    assert_eq!(
        image::image_dimensions(dir.join("display.png")).unwrap(),
        (2, 1)
    );
    // This unique directory contains only this test's two generated images.
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn budgets_and_invalid_values() {
    for preset in [
        QualityPreset::Production,
        QualityPreset::Ultra,
        QualityPreset::Reference,
    ] {
        let mut j = RenderJob::new("s89.motionloom", preset);
        j.scene_id = "scene".into();
        j.validate().unwrap();
        let encoded = serde_json::to_string(&j).unwrap();
        let decoded: RenderJob = serde_json::from_str(&encoded).unwrap();
        decoded.validate().unwrap();
    }
    let mut j = RenderJob::new("s89.motionloom", QualityPreset::Ultra);
    j.sampling.noise_threshold = f32::NAN;
    assert!(j.validate().is_err());
    j.sampling.noise_threshold = 0.003;
    j.light_paths.diffuse = 99;
    assert!(j.validate().is_err());
}

#[test]
fn noisy_pixels_do_not_claim_convergence_at_sample_cap() {
    let j = RenderJob::new("s89.motionloom", QualityPreset::Production);
    let mut f = [0.0; 16];
    f[3] = 1024.0;
    f[4] = 0.1;
    f[5] = 1000.0;
    assert!(!api::converged(&f, &j.sampling));
    f[5] = 0.0;
    assert!(api::converged(&f, &j.sampling));
}

#[test]
fn region_bounds_and_physical_ranges_are_checked() {
    let mut job = RenderJob::new("scene", QualityPreset::Ultra);
    job.scene_id = "scene".into();
    job.region = Some([640, 384, 128, 128]);
    assert!(job.validate().is_ok());
    job.region = Some([u32::MAX, 0, 128, 128]);
    assert!(job.validate().is_err());
    job.region = None;
    job.light_paths.transparent = 0;
    assert!(job.validate().is_err());
    job.light_paths.transparent = 64;
    job.sun_angular_diameter_degrees = f32::INFINITY;
    assert!(job.validate().is_err());
    let mut camera = crate::world::WorldCamera::default();
    let mut p = [[0.0; 4]; 20];
    camera.fov = "180".into();
    assert!(super::camera::configure(&mut p, &camera, &job).is_err());
    camera.fov = "57".into();
    camera.distance = "0".into();
    assert!(super::camera::configure(&mut p, &camera, &job).is_err());
}
