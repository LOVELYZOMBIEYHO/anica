// =========================================
// =========================================
// crates/motionloom/src/weaver/output/mod.rs

use crate::scene::compositor::{CompositedFrame, LinearPremultipliedImage};
use crate::weaver::WeaverError;
use crate::weaver::backend::wgpu::FILM_FLOATS_PER_PIXEL;
use std::path::Path;

pub(crate) fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Denoising changes beauty only; keep original auxiliary passes at the job root.
pub(crate) fn save_denoised(
    dir: &Path,
    size: [u32; 2],
    radiance: Vec<f32>,
    lighting: &crate::world::WorldLighting,
) -> Result<(), WeaverError> {
    let beauty = image::Rgb32FImage::from_raw(size[0], size[1], radiance)
        .ok_or_else(|| WeaverError::Invalid("denoised image dimensions mismatch".into()))?;
    std::fs::create_dir_all(dir)?;
    let mut display = image::RgbImage::new(size[0], size[1]);
    for (x, y, pixel) in beauty.enumerate_pixels() {
        display.put_pixel(x, y, image::Rgb(super::color::display(pixel.0, lighting)));
    }
    beauty.save(dir.join("beauty.exr"))?;
    display.save(dir.join("display.png"))?;
    Ok(())
}

/// Save a denoised 3D beauty and the complete compositor result.
pub(crate) fn save_denoised_composited(
    dir: &Path,
    size: [u32; 2],
    radiance: Vec<f32>,
    lighting: &crate::world::WorldLighting,
    composite: &CompositedFrame,
) -> Result<(), WeaverError> {
    save_denoised(dir, size, radiance, lighting)?;
    save_composite_outputs(dir, size, composite)
}

/// Preserve scene-linear radiance; apply display conversion only to the PNG.
pub(crate) fn save(
    dir: &Path,
    size: [u32; 2],
    film: &[f32],
    lighting: &crate::world::WorldLighting,
) -> Result<(), WeaverError> {
    std::fs::create_dir_all(dir)?;
    let mut beauty = image::Rgb32FImage::new(size[0], size[1]);
    let mut albedo = beauty.clone();
    let mut normal = beauty.clone();
    let mut depth = beauty.clone();
    let mut variance = beauty.clone();
    let mut counts = beauty.clone();
    let mut coverage = beauty.clone();
    let mut motion = beauty.clone();
    let mut display = image::RgbImage::new(size[0], size[1]);
    for (i, f) in film.chunks_exact(FILM_FLOATS_PER_PIXEL).enumerate() {
        let x = i as u32 % size[0];
        let y = i as u32 / size[0];
        let n = f[3].max(1.0);
        let rgb = [f[0] / n, f[1] / n, f[2] / n];
        beauty.put_pixel(x, y, image::Rgb(rgb));
        albedo.put_pixel(x, y, image::Rgb([f[8] / n, f[9] / n, f[10] / n]));
        normal.put_pixel(x, y, image::Rgb([f[12] / n, f[13] / n, f[14] / n]));
        depth.put_pixel(x, y, image::Rgb([f[15] / n; 3]));
        variance.put_pixel(x, y, image::Rgb([f[5] / (n - 1.0).max(1.0); 3]));
        counts.put_pixel(x, y, image::Rgb([f[3]; 3]));
        coverage.put_pixel(x, y, image::Rgb([f[16] / n; 3]));
        motion.put_pixel(x, y, image::Rgb([f[17] / n, f[18] / n, 0.0]));
        display.put_pixel(x, y, image::Rgb(super::color::display(rgb, lighting)));
    }
    beauty.save(dir.join("beauty.exr"))?;
    display.save(dir.join("display.png"))?;
    albedo.save(dir.join("albedo.exr"))?;
    normal.save(dir.join("normal.exr"))?;
    depth.save(dir.join("depth.exr"))?;
    variance.save(dir.join("variance.exr"))?;
    counts.save(dir.join("sample-count.exr"))?;
    coverage.save(dir.join("coverage.exr"))?;
    motion.save(dir.join("motion.exr"))?;
    Ok(())
}

/// Keep 3D AOVs untouched while adding the full linear composition.
pub(crate) fn save_composited(
    dir: &Path,
    size: [u32; 2],
    film: &[f32],
    lighting: &crate::world::WorldLighting,
    composite: &CompositedFrame,
) -> Result<(), WeaverError> {
    save(dir, size, film, lighting)?;
    save_composite_outputs(dir, size, composite)
}

pub(crate) fn save_pure_2d_master(dir: &Path, image: &image::RgbaImage) -> Result<(), WeaverError> {
    let size = [image.width(), image.height()];
    let transparent =
        LinearPremultipliedImage::new(size, vec![[0.0; 4]; size[0] as usize * size[1] as usize])
            .map_err(|error| WeaverError::Invalid(error.to_string()))?;
    let frame = CompositedFrame {
        scene_linear: transparent,
        display_linear: LinearPremultipliedImage::from_srgb_straight(image),
    };
    std::fs::create_dir_all(dir)?;
    save_composite_outputs(dir, size, &frame)
}

fn save_composite_outputs(
    dir: &Path,
    size: [u32; 2],
    composite: &CompositedFrame,
) -> Result<(), WeaverError> {
    if composite.scene_linear.size != size || composite.display_linear.size != size {
        return Err(WeaverError::Invalid(format!(
            "composition stages must both be {}x{}",
            size[0], size[1]
        )));
    }
    save_rgba16f_exr(
        &dir.join("scene-composite.exr"),
        &composite.scene_linear,
        "scene-composite",
        "linear_srgb_scene",
    )?;
    save_rgba16f_exr(
        &dir.join("display-master.exr"),
        &composite.display_linear,
        "display-master",
        "linear_srgb_display",
    )?;
    let obsolete = dir.join("composite.exr");
    if obsolete.is_file() {
        std::fs::remove_file(obsolete)?;
    }
    let mut display = image::RgbImage::new(size[0], size[1]);
    for (index, pixel) in composite.display_linear.pixels.iter().enumerate() {
        let alpha = pixel[3].clamp(0.0, 1.0);
        let rgb = if alpha > 0.0 {
            [pixel[0] / alpha, pixel[1] / alpha, pixel[2] / alpha]
        } else {
            [0.0; 3]
        };
        let encoded = rgb.map(|value| {
            (crate::scene::compositor::linear_to_srgb(value).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        });
        display.put_pixel(
            index as u32 % size[0],
            index as u32 / size[0],
            image::Rgb(encoded),
        );
    }
    display.save(dir.join("display.png"))?;
    Ok(())
}

/// Store master images as ZIP-compressed half-float OpenEXR with explicit
/// linear-sRGB and premultiplied-alpha metadata.
fn save_rgba16f_exr(
    path: &Path,
    image: &LinearPremultipliedImage,
    layer_name: &str,
    color_space: &str,
) -> Result<(), WeaverError> {
    use exr::prelude::{
        AttributeValue, Encoding, Image, ImageAttributes, Layer, LayerAttributes, SpecificChannels,
        Text, Vec2, WritableImage, f16,
    };

    let width = image.size[0] as usize;
    let height = image.size[1] as usize;
    let pixels = &image.pixels;
    let channels = SpecificChannels::rgba(|position: Vec2<usize>| {
        let pixel = pixels[position.y() * width + position.x()];
        (
            f16::from_f32(pixel[0]),
            f16::from_f32(pixel[1]),
            f16::from_f32(pixel[2]),
            f16::from_f32(pixel[3]),
        )
    });
    let mut layer_attributes = LayerAttributes::named(layer_name);
    layer_attributes.software_name = Some(Text::from("MotionLoom SceneCompositor"));
    layer_attributes.comments = Some(Text::from(
        "RGBA16F, linear sRGB/Rec.709 primaries, premultiplied alpha",
    ));
    layer_attributes.other.insert(
        Text::from("motionloomColorSpace"),
        AttributeValue::Text(Text::from(color_space)),
    );
    layer_attributes.other.insert(
        Text::from("motionloomAlphaMode"),
        AttributeValue::Text(Text::from("premultiplied")),
    );
    let layer = Layer::new(
        (width, height),
        layer_attributes,
        Encoding::SMALL_LOSSLESS,
        channels,
    );
    let mut output = Image::new(ImageAttributes::with_size((width, height)), layer);
    output.attributes.chromaticities = Some(exr::meta::attribute::Chromaticities {
        red: Vec2(0.640, 0.330),
        green: Vec2(0.300, 0.600),
        blue: Vec2(0.150, 0.060),
        white: Vec2(0.3127, 0.3290),
    });
    output
        .write()
        .to_file(path)
        .map_err(|error| WeaverError::Invalid(format!("write {}: {error}", path.display())))
}

pub(crate) fn save_frame_manifest(
    dir: &Path,
    frame: u32,
    fps: f32,
    size: [u32; 2],
    composited: bool,
    has_beauty: bool,
    denoised: bool,
) -> Result<(), WeaverError> {
    let mut outputs = serde_json::Map::new();
    if has_beauty {
        outputs.insert("beauty".into(), "beauty.exr".into());
    }
    if composited {
        outputs.insert("sceneComposite".into(), "scene-composite.exr".into());
        outputs.insert("displayMaster".into(), "display-master.exr".into());
    }
    outputs.insert("preview".into(), "display.png".into());
    let manifest = serde_json::json!({
        "version": 1,
        "frame": frame,
        "fps": fps,
        "resolution": size,
        "sceneColorSpace": "linear_srgb",
        "displayColorSpace": "linear_srgb_after_display_transform",
        "outputEncoding": "srgb",
        "alphaMode": "premultiplied",
        "masterPixelFormat": "rgba16f",
        "masterCompression": "zip16_lossless",
        "outputs": serde_json::Value::Object(outputs),
        "denoisedOutputs": denoised
    });
    std::fs::write(
        dir.join("frame-manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use exr::prelude::{Compression, MetaData, SampleType, read_first_rgba_layer_from_file};

    fn temp_output(name: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("motionloom-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn master_exr_is_half_float_and_preserves_hdr_values() {
        let dir = temp_output("master-exr");
        std::fs::create_dir_all(&dir).expect("create test output");
        let frame = CompositedFrame {
            scene_linear: LinearPremultipliedImage::new([1, 1], vec![[4.0, 1.5, -0.25, 1.0]])
                .unwrap(),
            display_linear: LinearPremultipliedImage::new([1, 1], vec![[0.25, 0.5, 0.75, 1.0]])
                .unwrap(),
        };
        std::fs::write(dir.join("composite.exr"), b"obsolete").expect("seed old output");
        save_composite_outputs(&dir, [1, 1], &frame).expect("save master outputs");

        let scene_path = dir.join("scene-composite.exr");
        let display_path = dir.join("display-master.exr");
        assert!(scene_path.is_file());
        assert!(display_path.is_file());
        assert!(!dir.join("composite.exr").exists());

        let metadata = MetaData::read_from_file(&scene_path, true).expect("read EXR metadata");
        let header = &metadata.headers[0];
        assert_eq!(header.channels.uniform_sample_type, Some(SampleType::F16));
        assert_eq!(header.compression, Compression::ZIP16);
        assert!(header.shared_attributes.chromaticities.is_some());

        let decoded = read_first_rgba_layer_from_file(
            &scene_path,
            |resolution, _| vec![vec![[0.0; 4]; resolution.width()]; resolution.height()],
            |pixels, position, rgba: (f32, f32, f32, f32)| {
                pixels[position.y()][position.x()] = [rgba.0, rgba.1, rgba.2, rgba.3];
            },
        )
        .expect("read scene master");
        let pixel = decoded.layer_data.channel_data.pixels[0][0];
        assert_eq!(pixel, [4.0, 1.5, -0.25, 1.0]);

        std::fs::remove_dir_all(dir).expect("remove test output");
    }

    #[test]
    fn frame_manifest_names_both_master_stages() {
        let dir = temp_output("frame-manifest");
        std::fs::create_dir_all(&dir).expect("create test output");
        save_frame_manifest(&dir, 12, 24.0, [1920, 1080], true, true, true)
            .expect("save frame manifest");
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("frame-manifest.json")).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(manifest["outputs"]["sceneComposite"], "scene-composite.exr");
        assert_eq!(manifest["outputs"]["displayMaster"], "display-master.exr");
        assert_eq!(manifest["masterPixelFormat"], "rgba16f");
        std::fs::remove_dir_all(dir).expect("remove test output");
    }
}
