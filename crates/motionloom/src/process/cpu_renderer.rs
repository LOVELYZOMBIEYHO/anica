// =========================================
// crates/motionloom/src/process/cpu_renderer.rs
// =========================================

use image::RgbaImage;

use crate::dsl::{GraphScript, PassNode, parse_graph_script};
use crate::error::{GraphParseError, RuntimeCompileError};
use crate::process::cpu_effects::{
    apply_gaussian_blur, apply_glow_bloom, apply_glow_stack, apply_hsla_overlay,
    apply_separable_gaussian_blur,
};
use crate::process::runtime::{RuntimeProgram, compile_runtime_program, eval_time_expr};
use crate::scene::drawable::parse_color;

#[derive(Debug, thiserror::Error)]
pub enum ProcessCpuRenderError {
    #[error(transparent)]
    Parse(#[from] GraphParseError),
    #[error(transparent)]
    Compile(#[from] RuntimeCompileError),
    #[error("invalid RGBA buffer: expected {expected} bytes for {width}x{height}, got {actual}")]
    InvalidRgbaBuffer {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
    #[error("invalid RGBA image buffer")]
    InvalidRgbaImageBuffer,
}

pub struct ProcessCpuRenderer {
    graph: GraphScript,
    _runtime: RuntimeProgram,
}

impl ProcessCpuRenderer {
    pub fn new(graph: GraphScript) -> Result<Self, ProcessCpuRenderError> {
        let runtime = compile_runtime_program(graph.clone())?;
        Ok(Self {
            graph,
            _runtime: runtime,
        })
    }

    pub fn render_frame(
        &self,
        frame: u32,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<RgbaImage, ProcessCpuRenderError> {
        let expected = width as usize * height as usize * 4;
        if width == 0 || height == 0 || rgba.len() != expected {
            return Err(ProcessCpuRenderError::InvalidRgbaBuffer {
                width,
                height,
                expected,
                actual: rgba.len(),
            });
        }
        let image = RgbaImage::from_raw(width, height, rgba.to_vec())
            .ok_or(ProcessCpuRenderError::InvalidRgbaImageBuffer)?;
        Ok(self.render_image(frame, image))
    }

    pub fn render_image(&self, frame: u32, mut image: RgbaImage) -> RgbaImage {
        let time_sec = frame as f32 / self.graph.fps.max(1.0);
        let duration_sec =
            (self.graph.duration_ms as f32 / 1000.0).max(1.0 / self.graph.fps.max(1.0));
        let time_norm = (time_sec / duration_sec).clamp(0.0, 1.0);

        for pass in &self.graph.passes {
            image = apply_process_pass(image, pass, time_norm, time_sec);
        }

        image
    }
}

pub fn render_process_frame_cpu(
    script: &str,
    frame: u32,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<RgbaImage, ProcessCpuRenderError> {
    let graph = parse_graph_script(script)?;
    ProcessCpuRenderer::new(graph)?.render_frame(frame, width, height, rgba)
}

fn apply_process_pass(
    image: RgbaImage,
    pass: &PassNode,
    time_norm: f32,
    time_sec: f32,
) -> RgbaImage {
    use crate::process::effect_kind::{ProcessEffect, resolve_process_effect};
    match resolve_process_effect(&pass.effect) {
        Some(ProcessEffect::HslaOverlay) => {
            let hue = process_param_f32(pass, &["hue", "h"], time_norm, time_sec, 0.0);
            let saturation =
                process_param_f32(pass, &["saturation", "sat", "s"], time_norm, time_sec, 0.0);
            let lightness =
                process_param_f32(pass, &["lightness", "lum", "l"], time_norm, time_sec, 0.0);
            let alpha = process_param_f32(pass, &["alpha", "a"], time_norm, time_sec, 0.0);
            apply_hsla_overlay(&image, hue, saturation, lightness, alpha)
        }
        Some(ProcessEffect::GaussianBlur) => {
            let sigma = process_param_f32(pass, &["sigma"], time_norm, time_sec, 1.0);
            apply_gaussian_blur(&image, sigma.clamp(0.0, 64.0))
        }
        Some(ProcessEffect::GaussianBlurHorizontal) => {
            let sigma = process_param_f32(pass, &["sigma"], time_norm, time_sec, 1.0);
            apply_separable_gaussian_blur(&image, sigma.clamp(0.0, 64.0), true)
        }
        Some(ProcessEffect::GaussianBlurVertical) => {
            let sigma = process_param_f32(pass, &["sigma"], time_norm, time_sec, 1.0);
            apply_separable_gaussian_blur(&image, sigma.clamp(0.0, 64.0), false)
        }
        Some(ProcessEffect::GlowBloom) => {
            let threshold = process_param_f32(pass, &["threshold"], time_norm, time_sec, 0.72);
            let intensity = process_param_f32(
                pass,
                &["intensity", "strength", "amount"],
                time_norm,
                time_sec,
                1.0,
            );
            let sigma = process_param_f32(
                pass,
                &[
                    "sigma",
                    "radius",
                    "radiusLarge",
                    "radiusMedium",
                    "radiusSmall",
                ],
                time_norm,
                time_sec,
                18.0,
            );
            apply_glow_bloom(&image, threshold, intensity, sigma)
        }
        Some(ProcessEffect::GlowStack) => {
            let threshold = process_param_f32(pass, &["threshold"], time_norm, time_sec, 0.62);
            let intensity = process_param_f32(
                pass,
                &["intensity", "strength", "amount"],
                time_norm,
                time_sec,
                1.5,
            );
            let radius_small = process_param_f32(
                pass,
                &["radiusSmall", "radius_small", "small"],
                time_norm,
                time_sec,
                6.0,
            );
            let radius_medium = process_param_f32(
                pass,
                &["radiusMedium", "radius_medium", "medium"],
                time_norm,
                time_sec,
                18.0,
            );
            let radius_large = process_param_f32(
                pass,
                &["radiusLarge", "radius_large", "large"],
                time_norm,
                time_sec,
                48.0,
            );
            let tint = process_param_color(pass, &["tint", "color"], [255, 255, 255, 255]);
            apply_glow_stack(
                &image,
                threshold,
                intensity,
                radius_small,
                radius_medium,
                radius_large,
                tint,
            )
        }
        Some(ProcessEffect::Brightness) => {
            let amount = process_brightness_amount(pass, time_norm, time_sec).clamp(-1.0, 1.0);
            apply_brightness(&image, amount)
        }
        Some(ProcessEffect::Opacity) => {
            let opacity =
                process_param_f32(pass, &["opacity", "alpha", "a"], time_norm, time_sec, 1.0);
            apply_opacity(&image, opacity)
        }
        Some(ProcessEffect::ToneMap)
        | Some(ProcessEffect::LightSweep)
        | Some(ProcessEffect::TextureOverlay)
        | Some(ProcessEffect::MagnifyLens)
        | Some(ProcessEffect::SpectralEnergy) => {
            // CPU renderer does not implement these effects yet; pass through unchanged.
            image
        }
        None => image,
    }
}

fn process_param_color(pass: &PassNode, keys: &[&str], fallback: [u8; 4]) -> [u8; 4] {
    keys.iter()
        .find_map(|key| {
            pass.params
                .iter()
                .find(|param| param.key.eq_ignore_ascii_case(key))
                .and_then(|param| {
                    parse_color(param.value.trim().trim_matches('"').trim_matches('\'')).ok()
                })
        })
        .unwrap_or(fallback)
}

fn apply_opacity(input: &RgbaImage, opacity: f32) -> RgbaImage {
    let opacity = opacity.clamp(0.0, 1.0);
    let mut out = input.clone();
    for pixel in out.pixels_mut() {
        pixel[3] = ((pixel[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
    }
    out
}

fn process_brightness_amount(pass: &PassNode, time_norm: f32, time_sec: f32) -> f32 {
    if pass
        .params
        .iter()
        .any(|param| param.key.eq_ignore_ascii_case("amount"))
    {
        return process_param_f32(pass, &["amount"], time_norm, time_sec, 0.0);
    }
    process_param_f32(pass, &["brightness", "value"], time_norm, time_sec, 0.0)
}

fn apply_brightness(input: &RgbaImage, amount: f32) -> RgbaImage {
    let delta = (amount * 255.0).round() as i16;
    let mut out = input.clone();
    for pixel in out.pixels_mut() {
        pixel[0] = (pixel[0] as i16 + delta).clamp(0, 255) as u8;
        pixel[1] = (pixel[1] as i16 + delta).clamp(0, 255) as u8;
        pixel[2] = (pixel[2] as i16 + delta).clamp(0, 255) as u8;
    }
    out
}

fn process_param_f32(
    pass: &PassNode,
    keys: &[&str],
    time_norm: f32,
    time_sec: f32,
    fallback: f32,
) -> f32 {
    keys.iter()
        .find_map(|key| {
            pass.params
                .iter()
                .find(|param| param.key == *key)
                .and_then(|param| eval_time_expr(&param.value, time_norm, time_sec).ok())
        })
        .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_cpu_renderer_applies_opacity_alpha() {
        let script = r##"
<Graph fps={30} size={[1,1]}>
  <Process id="layer_fx">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1,1]} />
    <Pass id="fx_opacity" kind="compute"
          effect="opacity"
          in={["src"]} out={["out"]}
          params={{ opacity: "0.5" }} />
  </Process>
  <Present from="layer_fx" />
</Graph>
"##;
        let out = render_process_frame_cpu(script, 0, 1, 1, &[20, 40, 60, 200]).unwrap();
        assert_eq!(out.get_pixel(0, 0).0, [20, 40, 60, 100]);
    }
}
