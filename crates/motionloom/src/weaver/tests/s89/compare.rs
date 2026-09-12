// =========================================
// =========================================
// crates/motionloom/src/weaver/tests/s89/compare.rs

#[test]
#[ignore = "explicit comparison of two existing render directories"]
fn compare_linear_renders() {
    let a = std::path::PathBuf::from(std::env::var_os("WEAVER_COMPARE_A").expect("first render"));
    let b = std::path::PathBuf::from(std::env::var_os("WEAVER_COMPARE_B").expect("second render"));
    let output = std::path::PathBuf::from(
        std::env::var_os("WEAVER_COMPARE_OUTPUT").expect("comparison output"),
    );
    let x = image::open(a.join("beauty.exr")).unwrap().to_rgb32f();
    let y = image::open(b.join("beauty.exr")).unwrap().to_rgb32f();
    assert_eq!(x.dimensions(), y.dimensions());
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut square = 0.0f64;
    let mut absolute = 0.0f64;
    let mut difference = image::RgbImage::new(x.width(), x.height());
    for (px, py, p) in x.enumerate_pixels() {
        let q = y.get_pixel(px, py);
        for i in 0..3 {
            assert!(p[i].is_finite() && q[i].is_finite());
            let delta = (q[i] - p[i]) as f64;
            sum_a += p[i] as f64;
            sum_b += q[i] as f64;
            square += delta * delta;
            absolute += delta.abs();
        }
        // Amplified absolute linear difference is diagnostic, not beauty grading.
        difference.put_pixel(
            px,
            py,
            image::Rgb(std::array::from_fn(|i| {
                (((q[i] - p[i]).abs() * 4.0).clamp(0.0, 1.0).sqrt() * 255.0) as u8
            })),
        );
    }
    let n = (x.width() as f64) * (x.height() as f64) * 3.0;
    // Sample variance divided by sample count estimates uncertainty in the mean.
    let uncertainty = |dir: &std::path::Path| {
        let variance = image::open(dir.join("variance.exr")).unwrap().to_rgb32f();
        let counts = image::open(dir.join("sample-count.exr"))
            .unwrap()
            .to_rgb32f();
        variance
            .pixels()
            .zip(counts.pixels())
            .map(|(v, c)| (v[0].max(0.0) / c[0].max(1.0)) as f64)
            .sum::<f64>()
            / (variance.width() as f64 * variance.height() as f64)
    };
    let metrics = serde_json::json!({
        "a": a, "b": b, "dimensions": [x.width(),x.height()],
        "mean_linear_rgb_a": sum_a/n, "mean_linear_rgb_b": sum_b/n,
        "mean_absolute_difference": absolute/n, "rmse": (square/n).sqrt(),
        "rms_estimated_standard_error_a": uncertainty(&a).sqrt(),
        "rms_estimated_standard_error_b": uncertainty(&b).sqrt(),
        "note": "Different sample counts include Monte Carlo error; this is not a ground-truth error estimate."
    });
    std::fs::create_dir_all(&output).unwrap();
    std::fs::write(
        output.join("metrics.json"),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap();
    difference.save(output.join("difference-x4.png")).unwrap();
    let first = image::open(a.join("display.png")).unwrap().to_rgb8();
    let second = image::open(b.join("display.png")).unwrap().to_rgb8();
    let mut pair = image::RgbImage::new(x.width() * 2, x.height());
    image::imageops::replace(&mut pair, &first, 0, 0);
    image::imageops::replace(&mut pair, &second, x.width() as i64, 0);
    pair.save(output.join("a-left-b-right.png")).unwrap();
    eprintln!("{metrics}");
}
