// =========================================
// =========================================
// crates/motionloom/src/process/runtime.rs

use crate::dsl::GraphScript;
pub use crate::error::RuntimeCompileError;
use crate::process::model::{GraphApplyScope, PassNode, PassTransitionEasing, PassTransitionMode};
use crate::process::pass::resolve_pass_kernel;
use crate::process::process_catalog::is_known_process_kernel;
use exmex::Express;
use std::collections::BTreeMap;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "kebab-case")]
pub enum BlurSharpenMode {
    #[default]
    Gaussian5tapBlur,
    Gaussian5tapH,
    Gaussian5tapV,
    Box,
    Unsharp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransitionCoreEffect {
    FadeIn,
    FadeOut,
    Dip,
    Dissolve,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CurveEase {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Ease(f32, f32, f32, f32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CurvePoint {
    t_sec: f32,
    value: f32,
    ease: CurveEase,
}

#[derive(Debug, Clone)]
enum RuntimePass {
    InvertMix {
        mix_expr: String,
    },
    BlurSharpenDetailGaussian {
        effect: BlurSharpenMode,
        sigma_expr: String,
    },
    TransitionCore {
        mode: PassTransitionMode,
        effect: TransitionCoreEffect,
        easing: PassTransitionEasing,
        progress_expr: Option<String>,
        opacity_expr: Option<String>,
        start_sec: Option<f32>,
        duration_sec: f32,
    },
    CompositeOpacity {
        opacity_expr: String,
    },
    ColorCoreLut {
        mix_expr: String,
    },
    ColorCoreBrightness {
        amount_expr: String,
    },
    ColorCoreHslaOverlay {
        hue_expr: String,
        saturation_expr: String,
        lightness_expr: String,
        alpha_expr: String,
    },
    ColorCoreBloom {
        threshold_expr: String,
        intensity_expr: String,
        sigma_expr: String,
    },
    ColorCoreGlowStack {
        threshold_expr: String,
        intensity_expr: String,
        radius_small_expr: String,
        radius_medium_expr: String,
        radius_large_expr: String,
        tint: [u8; 4],
    },
    ColorCoreToneMap {
        exposure_expr: String,
        contrast_expr: String,
        shoulder_expr: String,
        gamma_expr: String,
        saturation_expr: String,
    },
    LightAtmosphereSweep {
        position_expr: String,
        angle_expr: String,
        width_expr: String,
        softness_expr: String,
        intensity_expr: String,
        color: [u8; 4],
    },
    TextureOverlay {
        kind_id: f32,
        scale_expr: String,
        strength_expr: String,
        contrast_expr: String,
        seed_expr: String,
        brush_angle_expr: String,
        bump_strength_expr: String,
        relief_expr: String,
    },
    GpuOnlyKernel {
        kernel: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeProcessParamValue {
    Float(f32),
    Color([u8; 4]),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeProcessEffectInstance {
    pub effect_id: String,
    pub params: BTreeMap<String, RuntimeProcessParamValue>,
}

impl RuntimeProcessEffectInstance {
    fn new(effect_id: impl Into<String>) -> Self {
        Self {
            effect_id: effect_id.into(),
            params: BTreeMap::new(),
        }
    }

    fn with_float(mut self, key: impl Into<String>, value: f32) -> Self {
        self.params
            .insert(key.into(), RuntimeProcessParamValue::Float(value));
        self
    }

    fn with_color(mut self, key: impl Into<String>, value: [u8; 4]) -> Self {
        self.params
            .insert(key.into(), RuntimeProcessParamValue::Color(value));
        self
    }

    pub fn float(&self, key: &str) -> Option<f32> {
        match self.params.get(key) {
            Some(RuntimeProcessParamValue::Float(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn color(&self, key: &str) -> Option<[u8; 4]> {
        match self.params.get(key) {
            Some(RuntimeProcessParamValue::Color(value)) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeFrameOutput {
    pub frame: u32,
    pub time_sec: f32,
    pub time_norm: f32,
    pub invert_mix: f32,
    pub blur_sharpen_mode: Option<BlurSharpenMode>,
    pub layer_blur_sigma: Option<f32>,
    pub layer_sharpen_sigma: Option<f32>,
    pub transition_dissolve_mix: Option<f32>,
    pub layer_transition_opacity: Option<f32>,
    pub layer_lut_mix: Option<f32>,
    pub layer_hsla_hue: Option<f32>,
    pub layer_hsla_saturation: Option<f32>,
    pub layer_hsla_lightness: Option<f32>,
    pub layer_hsla_alpha: Option<f32>,
    pub layer_bloom_threshold: Option<f32>,
    pub layer_bloom_intensity: Option<f32>,
    pub layer_bloom_sigma: Option<f32>,
    pub layer_tone_map_exposure: Option<f32>,
    pub layer_tone_map_contrast: Option<f32>,
    pub layer_tone_map_shoulder: Option<f32>,
    pub layer_tone_map_gamma: Option<f32>,
    pub layer_tone_map_saturation: Option<f32>,
    pub layer_light_sweep_position: Option<f32>,
    pub layer_light_sweep_angle: Option<f32>,
    pub layer_light_sweep_width: Option<f32>,
    pub layer_light_sweep_softness: Option<f32>,
    pub layer_light_sweep_intensity: Option<f32>,
    pub layer_light_sweep_color: Option<[u8; 4]>,
    pub process_effects: Vec<RuntimeProcessEffectInstance>,
}

#[derive(Debug, Clone)]
pub struct RuntimeProgram {
    graph: GraphScript,
    passes: Vec<RuntimePass>,
    unsupported_kernels: Vec<String>,
}

pub fn compile_runtime_program(graph: GraphScript) -> Result<RuntimeProgram, RuntimeCompileError> {
    RuntimeProgram::new(graph)
}

impl RuntimeProgram {
    pub fn new(graph: GraphScript) -> Result<Self, RuntimeCompileError> {
        let mut passes = Vec::<RuntimePass>::new();
        let mut unsupported = Vec::<String>::new();

        for pass in &graph.passes {
            let resolved_kernel = resolve_kernel_for_pass(pass)?;
            match resolved_kernel.as_str() {
                // v1 supported runtime kernel: animatable invert mix.
                "invert_mix.wgsl" => {
                    let mix_expr = pass_param(pass, "mix")
                        .map(normalize_param_expr)
                        .unwrap_or_else(|| "1.0".to_string());
                    validate_expr(&mix_expr).map_err(|e| RuntimeCompileError {
                        message: format!("pass {} invalid mix expression: {}", pass.id, e),
                    })?;
                    passes.push(RuntimePass::InvertMix { mix_expr });
                }
                "blur_sharpen_detail_gaussian.wgsl"
                | "blur_sharpen_detail_gaussian_5tap.wgsl"
                | "effect_for_testing_run.wgsl" => {
                    let effect = parse_blur_sharpen_effect(&pass.effect).map_err(|e| {
                        RuntimeCompileError {
                            message: format!("pass {} invalid effect: {}", pass.id, e),
                        }
                    })?;
                    let sigma_expr = pass_param(pass, "sigma")
                        .map(normalize_param_expr)
                        .unwrap_or_else(|| "2.0".to_string());
                    validate_expr(&sigma_expr).map_err(|e| RuntimeCompileError {
                        message: format!("pass {} invalid sigma expression: {}", pass.id, e),
                    })?;
                    passes.push(RuntimePass::BlurSharpenDetailGaussian { effect, sigma_expr });
                }
                "transition_core.wgsl" => {
                    let mode = pass.transition.clone().unwrap_or(PassTransitionMode::Auto);
                    let effect = parse_transition_core_effect(&pass.effect).map_err(|e| {
                        RuntimeCompileError {
                            message: format!("pass {} invalid effect: {}", pass.id, e),
                        }
                    })?;
                    let easing = pass_param(pass, "easing")
                        .map(normalize_param_expr)
                        .as_deref()
                        .map(parse_transition_easing_param)
                        .transpose()
                        .map_err(|e| RuntimeCompileError {
                            message: format!("pass {} invalid easing: {}", pass.id, e),
                        })?
                        .or_else(|| pass.transition_easing.clone())
                        .unwrap_or(PassTransitionEasing::Linear);

                    let progress_expr = if effect == TransitionCoreEffect::Dissolve {
                        pass_param(pass, "progress")
                            .or_else(|| pass_param(pass, "mix"))
                            .map(normalize_param_expr)
                    } else {
                        None
                    };
                    if let Some(expr) = progress_expr.as_deref() {
                        validate_expr(expr).map_err(|e| RuntimeCompileError {
                            message: format!("pass {} invalid progress expression: {}", pass.id, e),
                        })?;
                    }

                    let start_sec = pass_param_f32(pass, &["startSec", "start_sec"]);
                    let duration_sec =
                        pass_param_f32(pass, &["durationSec", "duration_sec"]).unwrap_or(0.6);
                    let opacity_expr = pass_param(pass, "opacity").map(normalize_param_expr);
                    if let Some(expr) = opacity_expr.as_deref() {
                        validate_expr(expr).map_err(|e| RuntimeCompileError {
                            message: format!("pass {} invalid opacity expression: {}", pass.id, e),
                        })?;
                    }
                    if !duration_sec.is_finite() || duration_sec <= 0.0 {
                        return Err(RuntimeCompileError {
                            message: format!(
                                "pass {} invalid durationSec: expected > 0, got {}",
                                pass.id, duration_sec
                            ),
                        });
                    }

                    passes.push(RuntimePass::TransitionCore {
                        mode,
                        effect,
                        easing,
                        progress_expr,
                        opacity_expr,
                        start_sec,
                        duration_sec,
                    });
                }
                "transition_fade_in_out.wgsl" | "transition_dissolve.wgsl" => {
                    return Err(RuntimeCompileError {
                        message: format!(
                            "pass {} uses removed kernel {}; use transition_core.wgsl + effect instead",
                            pass.id, resolved_kernel
                        ),
                    });
                }
                "composite_core.wgsl" => {
                    let normalized_effect = pass
                        .effect
                        .trim()
                        .trim_matches('"')
                        .trim()
                        .to_ascii_lowercase();
                    if normalized_effect != "opacity" && normalized_effect != "composite.opacity" {
                        passes.push(RuntimePass::GpuOnlyKernel {
                            kernel: resolved_kernel.clone(),
                        });
                        continue;
                    }
                    let opacity_expr = pass_param(pass, "opacity")
                        .map(normalize_param_expr)
                        .unwrap_or_else(|| "1.0".to_string());
                    validate_expr(&opacity_expr).map_err(|e| RuntimeCompileError {
                        message: format!("pass {} invalid opacity expression: {}", pass.id, e),
                    })?;
                    passes.push(RuntimePass::CompositeOpacity { opacity_expr });
                }
                "color_core.wgsl" => {
                    let normalized_effect = pass
                        .effect
                        .trim()
                        .trim_matches('"')
                        .trim()
                        .to_ascii_lowercase();
                    if normalized_effect == "lut" || normalized_effect == "color_tone.lut" {
                        let mix_expr = pass_param(pass, "mix")
                            .or_else(|| pass_param(pass, "lutMix"))
                            .or_else(|| pass_param(pass, "lut_mix"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "1.0".to_string());
                        validate_expr(&mix_expr).map_err(|e| RuntimeCompileError {
                            message: format!("pass {} invalid LUT mix expression: {}", pass.id, e),
                        })?;
                        passes.push(RuntimePass::ColorCoreLut { mix_expr });
                        continue;
                    }
                    if is_brightness_effect(&normalized_effect) {
                        let amount_expr = if let Some(amount) = pass_param(pass, "amount") {
                            normalize_param_expr(amount)
                        } else {
                            match pass_param(pass, "brightness")
                                .or_else(|| pass_param(pass, "value"))
                                .map(normalize_param_expr)
                            {
                                Some(value_expr) => value_expr,
                                None => "0.0".to_string(),
                            }
                        };
                        validate_expr(&amount_expr).map_err(|e| RuntimeCompileError {
                            message: format!(
                                "pass {} invalid brightness expression: {}",
                                pass.id, e
                            ),
                        })?;
                        passes.push(RuntimePass::ColorCoreBrightness { amount_expr });
                        continue;
                    }
                    if is_bloom_effect(&normalized_effect) {
                        if is_glow_stack_effect(&normalized_effect) {
                            let threshold_expr = pass_param(pass, "threshold")
                                .or_else(|| pass_param(pass, "glowThreshold"))
                                .or_else(|| pass_param(pass, "glow_threshold"))
                                .map(normalize_param_expr)
                                .unwrap_or_else(|| "0.62".to_string());
                            let intensity_expr = pass_param(pass, "intensity")
                                .or_else(|| pass_param(pass, "glowIntensity"))
                                .or_else(|| pass_param(pass, "glow_intensity"))
                                .or_else(|| pass_param(pass, "strength"))
                                .or_else(|| pass_param(pass, "amount"))
                                .map(normalize_param_expr)
                                .unwrap_or_else(|| "1.5".to_string());
                            let radius_small_expr = pass_param(pass, "radiusSmall")
                                .or_else(|| pass_param(pass, "radius_small"))
                                .or_else(|| pass_param(pass, "small"))
                                .map(normalize_param_expr)
                                .unwrap_or_else(|| "6.0".to_string());
                            let radius_medium_expr = pass_param(pass, "radiusMedium")
                                .or_else(|| pass_param(pass, "radius_medium"))
                                .or_else(|| pass_param(pass, "medium"))
                                .map(normalize_param_expr)
                                .unwrap_or_else(|| "18.0".to_string());
                            let radius_large_expr = pass_param(pass, "radiusLarge")
                                .or_else(|| pass_param(pass, "radius_large"))
                                .or_else(|| pass_param(pass, "large"))
                                .map(normalize_param_expr)
                                .unwrap_or_else(|| "48.0".to_string());
                            let tint = pass_param(pass, "tint")
                                .or_else(|| pass_param(pass, "color"))
                                .and_then(parse_runtime_color)
                                .unwrap_or([255, 255, 255, 255]);

                            for (name, expr) in [
                                ("threshold", &threshold_expr),
                                ("intensity", &intensity_expr),
                                ("radiusSmall", &radius_small_expr),
                                ("radiusMedium", &radius_medium_expr),
                                ("radiusLarge", &radius_large_expr),
                            ] {
                                validate_expr(expr).map_err(|e| RuntimeCompileError {
                                    message: format!(
                                        "pass {} invalid glow_stack {} expression: {}",
                                        pass.id, name, e
                                    ),
                                })?;
                            }

                            passes.push(RuntimePass::ColorCoreGlowStack {
                                threshold_expr,
                                intensity_expr,
                                radius_small_expr,
                                radius_medium_expr,
                                radius_large_expr,
                                tint,
                            });
                            continue;
                        }

                        let threshold_expr = pass_param(pass, "threshold")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.72".to_string());
                        let intensity_expr = pass_param(pass, "intensity")
                            .or_else(|| pass_param(pass, "strength"))
                            .or_else(|| pass_param(pass, "amount"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "1.0".to_string());
                        let sigma_expr = pass_param(pass, "sigma")
                            .or_else(|| pass_param(pass, "radius"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "18.0".to_string());

                        validate_expr(&threshold_expr).map_err(|e| RuntimeCompileError {
                            message: format!(
                                "pass {} invalid bloom threshold expression: {}",
                                pass.id, e
                            ),
                        })?;
                        validate_expr(&intensity_expr).map_err(|e| RuntimeCompileError {
                            message: format!(
                                "pass {} invalid bloom intensity expression: {}",
                                pass.id, e
                            ),
                        })?;
                        validate_expr(&sigma_expr).map_err(|e| RuntimeCompileError {
                            message: format!(
                                "pass {} invalid bloom sigma expression: {}",
                                pass.id, e
                            ),
                        })?;

                        passes.push(RuntimePass::ColorCoreBloom {
                            threshold_expr,
                            intensity_expr,
                            sigma_expr,
                        });
                        continue;
                    }
                    if is_tone_map_effect(&normalized_effect) {
                        let exposure_expr = pass_param(pass, "exposure")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.0".to_string());
                        let contrast_expr = pass_param(pass, "contrast")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "1.0".to_string());
                        let shoulder_expr = pass_param(pass, "shoulder")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "1.0".to_string());
                        let gamma_expr = pass_param(pass, "gamma")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "2.2".to_string());
                        let saturation_expr = pass_param(pass, "saturation")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "1.0".to_string());

                        for (name, expr) in [
                            ("exposure", &exposure_expr),
                            ("contrast", &contrast_expr),
                            ("shoulder", &shoulder_expr),
                            ("gamma", &gamma_expr),
                            ("saturation", &saturation_expr),
                        ] {
                            validate_expr(expr).map_err(|e| RuntimeCompileError {
                                message: format!(
                                    "pass {} invalid tone_map {} expression: {}",
                                    pass.id, name, e
                                ),
                            })?;
                        }

                        passes.push(RuntimePass::ColorCoreToneMap {
                            exposure_expr,
                            contrast_expr,
                            shoulder_expr,
                            gamma_expr,
                            saturation_expr,
                        });
                        continue;
                    }
                    if is_light_sweep_effect(&normalized_effect) {
                        let position_expr = pass_param(pass, "position")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.5".to_string());
                        let angle_expr = pass_param(pass, "angle")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "-18.0".to_string());
                        let width_expr = pass_param(pass, "width")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.16".to_string());
                        let softness_expr = pass_param(pass, "softness")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.08".to_string());
                        let intensity_expr = pass_param(pass, "intensity")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "1.0".to_string());
                        let color = pass_param(pass, "color")
                            .and_then(parse_runtime_color)
                            .unwrap_or([255, 255, 255, 255]);

                        for (name, expr) in [
                            ("position", &position_expr),
                            ("angle", &angle_expr),
                            ("width", &width_expr),
                            ("softness", &softness_expr),
                            ("intensity", &intensity_expr),
                        ] {
                            validate_expr(expr).map_err(|e| RuntimeCompileError {
                                message: format!(
                                    "pass {} invalid light_sweep {} expression: {}",
                                    pass.id, name, e
                                ),
                            })?;
                        }

                        passes.push(RuntimePass::LightAtmosphereSweep {
                            position_expr,
                            angle_expr,
                            width_expr,
                            softness_expr,
                            intensity_expr,
                            color,
                        });
                        continue;
                    }
                    if is_texture_overlay_effect(&normalized_effect) {
                        let kind_id = pass_param(pass, "kind")
                            .or_else(|| pass_param(pass, "texture"))
                            .map(texture_overlay_kind_id)
                            .unwrap_or(1.0);
                        let scale_expr = pass_param(pass, "scale")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "42.0".to_string());
                        let strength_expr = pass_param(pass, "strength")
                            .or_else(|| pass_param(pass, "amount"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.25".to_string());
                        let contrast_expr = pass_param(pass, "contrast")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.5".to_string());
                        let seed_expr = pass_param(pass, "seed")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.0".to_string());
                        let brush_angle_expr = pass_param(pass, "brush_angle")
                            .or_else(|| pass_param(pass, "angle"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "-8.0".to_string());
                        let bump_strength_expr = pass_param(pass, "bump_strength")
                            .or_else(|| pass_param(pass, "bump"))
                            .or_else(|| pass_param(pass, "impasto_strength"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.35".to_string());
                        let relief_expr = pass_param(pass, "relief")
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.45".to_string());

                        for (name, expr) in [
                            ("scale", &scale_expr),
                            ("strength", &strength_expr),
                            ("contrast", &contrast_expr),
                            ("seed", &seed_expr),
                            ("brush_angle", &brush_angle_expr),
                            ("bump_strength", &bump_strength_expr),
                            ("relief", &relief_expr),
                        ] {
                            validate_expr(expr).map_err(|e| RuntimeCompileError {
                                message: format!(
                                    "pass {} invalid texture_overlay {} expression: {}",
                                    pass.id, name, e
                                ),
                            })?;
                        }

                        passes.push(RuntimePass::TextureOverlay {
                            kind_id,
                            scale_expr,
                            strength_expr,
                            contrast_expr,
                            seed_expr,
                            brush_angle_expr,
                            bump_strength_expr,
                            relief_expr,
                        });
                        continue;
                    }
                    if normalized_effect == "hsla_overlay"
                        || normalized_effect == "hsla"
                        || normalized_effect == "tint_overlay"
                        || normalized_effect == "color_tone.hsla_overlay"
                    {
                        let hue_expr = pass_param(pass, "hue")
                            .or_else(|| pass_param(pass, "h"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.0".to_string());
                        let saturation_expr = pass_param(pass, "saturation")
                            .or_else(|| pass_param(pass, "sat"))
                            .or_else(|| pass_param(pass, "s"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.0".to_string());
                        let lightness_expr = pass_param(pass, "lightness")
                            .or_else(|| pass_param(pass, "lum"))
                            .or_else(|| pass_param(pass, "l"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.0".to_string());
                        let alpha_expr = pass_param(pass, "alpha")
                            .or_else(|| pass_param(pass, "a"))
                            .map(normalize_param_expr)
                            .unwrap_or_else(|| "0.0".to_string());

                        validate_expr(&hue_expr).map_err(|e| RuntimeCompileError {
                            message: format!("pass {} invalid HSLA hue expression: {}", pass.id, e),
                        })?;
                        validate_expr(&saturation_expr).map_err(|e| RuntimeCompileError {
                            message: format!(
                                "pass {} invalid HSLA saturation expression: {}",
                                pass.id, e
                            ),
                        })?;
                        validate_expr(&lightness_expr).map_err(|e| RuntimeCompileError {
                            message: format!(
                                "pass {} invalid HSLA lightness expression: {}",
                                pass.id, e
                            ),
                        })?;
                        validate_expr(&alpha_expr).map_err(|e| RuntimeCompileError {
                            message: format!(
                                "pass {} invalid HSLA alpha expression: {}",
                                pass.id, e
                            ),
                        })?;

                        passes.push(RuntimePass::ColorCoreHslaOverlay {
                            hue_expr,
                            saturation_expr,
                            lightness_expr,
                            alpha_expr,
                        });
                        continue;
                    }
                    passes.push(RuntimePass::GpuOnlyKernel {
                        kernel: resolved_kernel.clone(),
                    });
                    continue;
                }
                other if is_known_process_kernel(other) => {
                    passes.push(RuntimePass::GpuOnlyKernel {
                        kernel: other.to_string(),
                    });
                }
                other if other.ends_with(".wgsl") => {
                    // Explicit kernel path is treated as custom GPU kernel.
                    passes.push(RuntimePass::GpuOnlyKernel {
                        kernel: other.to_string(),
                    });
                }
                other => {
                    unsupported.push(format!("{} ({})", pass.id, other));
                }
            }
        }

        Ok(Self {
            graph,
            passes,
            unsupported_kernels: unsupported,
        })
    }

    pub fn graph(&self) -> &GraphScript {
        &self.graph
    }

    pub fn total_pass_count(&self) -> usize {
        self.graph.passes.len()
    }

    pub fn supported_pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn unsupported_kernels(&self) -> &[String] {
        &self.unsupported_kernels
    }

    pub fn evaluate_frame(&self, frame: u32) -> RuntimeFrameOutput {
        let fps = self.graph.fps.max(1.0);
        let time_sec = frame as f32 / fps;
        self.evaluate_at_time_sec(time_sec, None)
    }

    pub fn evaluate_at_time_sec(
        &self,
        time_sec: f32,
        clip_duration_sec: Option<f32>,
    ) -> RuntimeFrameOutput {
        let fps = self.graph.fps.max(1.0);
        let graph_duration_sec = (self.graph.duration_ms as f32 / 1000.0).max(0.0001);
        let time_sec = time_sec.max(0.0);
        let time_norm = (time_sec / graph_duration_sec).clamp(0.0, 1.0);
        let mut out = RuntimeFrameOutput {
            frame: (time_sec * fps).floor().max(0.0) as u32,
            time_sec,
            time_norm,
            invert_mix: 0.0,
            blur_sharpen_mode: None,
            layer_blur_sigma: None,
            layer_sharpen_sigma: None,
            transition_dissolve_mix: None,
            layer_transition_opacity: None,
            layer_lut_mix: None,
            layer_hsla_hue: None,
            layer_hsla_saturation: None,
            layer_hsla_lightness: None,
            layer_hsla_alpha: None,
            layer_bloom_threshold: None,
            layer_bloom_intensity: None,
            layer_bloom_sigma: None,
            layer_tone_map_exposure: None,
            layer_tone_map_contrast: None,
            layer_tone_map_shoulder: None,
            layer_tone_map_gamma: None,
            layer_tone_map_saturation: None,
            layer_light_sweep_position: None,
            layer_light_sweep_angle: None,
            layer_light_sweep_width: None,
            layer_light_sweep_softness: None,
            layer_light_sweep_intensity: None,
            layer_light_sweep_color: None,
            process_effects: Vec::new(),
        };
        // apply="graph" only gates when duration is explicitly provided on <Graph>.
        if self.graph.apply == GraphApplyScope::Graph
            && self.graph.duration_explicit
            && time_sec >= graph_duration_sec
        {
            return out;
        }

        for pass in &self.passes {
            match pass {
                RuntimePass::InvertMix { mix_expr } => {
                    if let Ok(value) = eval_expr(mix_expr, time_norm, time_sec) {
                        out.invert_mix = value.clamp(0.0, 1.0);
                    }
                }
                RuntimePass::BlurSharpenDetailGaussian { effect, sigma_expr } => {
                    if let Ok(value) = eval_expr(sigma_expr, time_norm, time_sec) {
                        out.blur_sharpen_mode = Some(*effect);
                        let amount = value.clamp(0.0, 64.0);
                        match effect {
                            BlurSharpenMode::Unsharp => {
                                out.layer_sharpen_sigma = Some(amount);
                            }
                            _ => {
                                out.layer_blur_sigma = Some(amount);
                            }
                        }
                    }
                }
                RuntimePass::TransitionCore {
                    mode,
                    effect,
                    easing,
                    progress_expr,
                    opacity_expr,
                    start_sec,
                    duration_sec: pass_duration_sec,
                } => {
                    if *mode == PassTransitionMode::Off {
                        continue;
                    }

                    let progress = if let Some(expr) = progress_expr.as_deref() {
                        match eval_expr(expr, time_norm, time_sec) {
                            Ok(v) => v.clamp(0.0, 1.0),
                            Err(_) => continue,
                        }
                    } else {
                        let clip_span_sec = clip_duration_sec
                            .filter(|v| *v > 0.0)
                            .unwrap_or(graph_duration_sec);
                        let start_sec = match (effect, start_sec) {
                            (_, Some(v)) => *v,
                            (TransitionCoreEffect::FadeOut, None) => {
                                (clip_span_sec - *pass_duration_sec).max(0.0)
                            }
                            _ => 0.0,
                        };
                        ((time_sec - start_sec) / *pass_duration_sec).clamp(0.0, 1.0)
                    };
                    let progress = apply_transition_easing(progress, easing.clone());
                    let curve_opacity = opacity_expr
                        .as_deref()
                        .and_then(|expr| eval_expr(expr, time_norm, time_sec).ok())
                        .map(|v| v.clamp(0.0, 1.0));

                    match effect {
                        TransitionCoreEffect::FadeIn => {
                            let opacity = curve_opacity.unwrap_or(progress.clamp(0.0, 1.0));
                            out.layer_transition_opacity = Some(
                                out.layer_transition_opacity
                                    .map_or(opacity, |prev| prev.min(opacity)),
                            );
                        }
                        TransitionCoreEffect::FadeOut => {
                            let opacity = curve_opacity.unwrap_or((1.0 - progress).clamp(0.0, 1.0));
                            out.layer_transition_opacity = Some(
                                out.layer_transition_opacity
                                    .map_or(opacity, |prev| prev.min(opacity)),
                            );
                        }
                        TransitionCoreEffect::Dip => {
                            let opacity =
                                curve_opacity.unwrap_or(v1_dissolve_opacity_from_mix(progress));
                            out.layer_transition_opacity = Some(
                                out.layer_transition_opacity
                                    .map_or(opacity, |prev| prev.min(opacity)),
                            );
                        }
                        TransitionCoreEffect::Dissolve => {
                            let mix = progress.clamp(0.0, 1.0);
                            out.transition_dissolve_mix = Some(
                                out.transition_dissolve_mix
                                    .map_or(mix, |prev| prev.max(mix)),
                            );
                        }
                    }
                }
                RuntimePass::GpuOnlyKernel { kernel } => {
                    let _ = kernel;
                }
                RuntimePass::CompositeOpacity { opacity_expr } => {
                    if let Ok(value) = eval_expr(opacity_expr, time_norm, time_sec) {
                        let opacity = value.clamp(0.0, 1.0);
                        out.layer_transition_opacity = Some(
                            out.layer_transition_opacity
                                .map_or(opacity, |prev| prev.min(opacity)),
                        );
                    }
                }
                RuntimePass::ColorCoreLut { mix_expr } => {
                    if let Ok(value) = eval_expr(mix_expr, time_norm, time_sec) {
                        out.layer_lut_mix = Some(value.clamp(0.0, 1.0));
                    }
                }
                RuntimePass::ColorCoreBrightness { amount_expr } => {
                    let Ok(amount) = eval_expr(amount_expr, time_norm, time_sec) else {
                        continue;
                    };
                    out.process_effects.push(
                        RuntimeProcessEffectInstance::new("brightness")
                            .with_float("amount", amount.clamp(-1.0, 1.0)),
                    );
                }
                RuntimePass::ColorCoreHslaOverlay {
                    hue_expr,
                    saturation_expr,
                    lightness_expr,
                    alpha_expr,
                } => {
                    let Ok(hue) = eval_expr(hue_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(saturation) = eval_expr(saturation_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(lightness) = eval_expr(lightness_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(alpha) = eval_expr(alpha_expr, time_norm, time_sec) else {
                        continue;
                    };
                    out.layer_hsla_hue = Some(hue.rem_euclid(360.0));
                    out.layer_hsla_saturation = Some(saturation.clamp(0.0, 1.0));
                    out.layer_hsla_lightness = Some(lightness.clamp(0.0, 1.0));
                    out.layer_hsla_alpha = Some(alpha.clamp(0.0, 1.0));
                    out.process_effects.push(
                        RuntimeProcessEffectInstance::new("hsla_overlay")
                            .with_float("hue", hue.rem_euclid(360.0))
                            .with_float("saturation", saturation.clamp(0.0, 1.0))
                            .with_float("lightness", lightness.clamp(0.0, 1.0))
                            .with_float("alpha", alpha.clamp(0.0, 1.0)),
                    );
                }
                RuntimePass::ColorCoreBloom {
                    threshold_expr,
                    intensity_expr,
                    sigma_expr,
                } => {
                    let Ok(threshold) = eval_expr(threshold_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(intensity) = eval_expr(intensity_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(sigma) = eval_expr(sigma_expr, time_norm, time_sec) else {
                        continue;
                    };
                    out.layer_bloom_threshold = Some(threshold.clamp(0.0, 1.0));
                    out.layer_bloom_intensity = Some(intensity.clamp(0.0, 8.0));
                    out.layer_bloom_sigma = Some(sigma.clamp(0.0, 64.0));
                    out.process_effects.push(
                        RuntimeProcessEffectInstance::new("glow_bloom")
                            .with_float("threshold", threshold.clamp(0.0, 1.0))
                            .with_float("intensity", intensity.clamp(0.0, 8.0))
                            .with_float("sigma", sigma.clamp(0.0, 64.0)),
                    );
                }
                RuntimePass::ColorCoreGlowStack {
                    threshold_expr,
                    intensity_expr,
                    radius_small_expr,
                    radius_medium_expr,
                    radius_large_expr,
                    tint,
                } => {
                    let Ok(threshold) = eval_expr(threshold_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(intensity) = eval_expr(intensity_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(radius_small) = eval_expr(radius_small_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(radius_medium) = eval_expr(radius_medium_expr, time_norm, time_sec)
                    else {
                        continue;
                    };
                    let Ok(radius_large) = eval_expr(radius_large_expr, time_norm, time_sec) else {
                        continue;
                    };
                    out.layer_bloom_threshold = Some(threshold.clamp(0.0, 1.0));
                    out.layer_bloom_intensity = Some(intensity.clamp(0.0, 8.0));
                    out.layer_bloom_sigma = Some(radius_medium.clamp(0.0, 96.0));
                    out.process_effects.push(
                        RuntimeProcessEffectInstance::new("glow_stack")
                            .with_float("threshold", threshold.clamp(0.0, 1.0))
                            .with_float("intensity", intensity.clamp(0.0, 8.0))
                            .with_float("radiusSmall", radius_small.clamp(0.0, 64.0))
                            .with_float("radiusMedium", radius_medium.clamp(0.0, 96.0))
                            .with_float("radiusLarge", radius_large.clamp(0.0, 160.0))
                            .with_color("tint", *tint),
                    );
                }
                RuntimePass::ColorCoreToneMap {
                    exposure_expr,
                    contrast_expr,
                    shoulder_expr,
                    gamma_expr,
                    saturation_expr,
                } => {
                    let Ok(exposure) = eval_expr(exposure_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(contrast) = eval_expr(contrast_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(shoulder) = eval_expr(shoulder_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(gamma) = eval_expr(gamma_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(saturation) = eval_expr(saturation_expr, time_norm, time_sec) else {
                        continue;
                    };
                    out.layer_tone_map_exposure = Some(exposure.clamp(-8.0, 8.0));
                    out.layer_tone_map_contrast = Some(contrast.clamp(0.0, 4.0));
                    out.layer_tone_map_shoulder = Some(shoulder.clamp(0.0, 2.0));
                    out.layer_tone_map_gamma = Some(gamma.clamp(0.0001, 8.0));
                    out.layer_tone_map_saturation = Some(saturation.clamp(0.0, 4.0));
                    out.process_effects.push(
                        RuntimeProcessEffectInstance::new("tone_map")
                            .with_float("exposure", exposure.clamp(-8.0, 8.0))
                            .with_float("contrast", contrast.clamp(0.0, 4.0))
                            .with_float("shoulder", shoulder.clamp(0.0, 2.0))
                            .with_float("gamma", gamma.clamp(0.0001, 8.0))
                            .with_float("saturation", saturation.clamp(0.0, 4.0)),
                    );
                }
                RuntimePass::LightAtmosphereSweep {
                    position_expr,
                    angle_expr,
                    width_expr,
                    softness_expr,
                    intensity_expr,
                    color,
                } => {
                    let Ok(position) = eval_expr(position_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(angle) = eval_expr(angle_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(width) = eval_expr(width_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(softness) = eval_expr(softness_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(intensity) = eval_expr(intensity_expr, time_norm, time_sec) else {
                        continue;
                    };
                    out.layer_light_sweep_position = Some(position.clamp(-2.0, 3.0));
                    out.layer_light_sweep_angle = Some(angle);
                    out.layer_light_sweep_width = Some(width.clamp(0.0, 2.0));
                    out.layer_light_sweep_softness = Some(softness.clamp(0.0, 2.0));
                    out.layer_light_sweep_intensity = Some(intensity.clamp(0.0, 8.0));
                    out.layer_light_sweep_color = Some(*color);
                    out.process_effects.push(
                        RuntimeProcessEffectInstance::new("light_sweep")
                            .with_float("position", position.clamp(-2.0, 3.0))
                            .with_float("angle", angle)
                            .with_float("width", width.clamp(0.0, 2.0))
                            .with_float("softness", softness.clamp(0.0, 2.0))
                            .with_float("intensity", intensity.clamp(0.0, 8.0))
                            .with_color("color", *color),
                    );
                }
                RuntimePass::TextureOverlay {
                    kind_id,
                    scale_expr,
                    strength_expr,
                    contrast_expr,
                    seed_expr,
                    brush_angle_expr,
                    bump_strength_expr,
                    relief_expr,
                } => {
                    let Ok(scale) = eval_expr(scale_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(strength) = eval_expr(strength_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(contrast) = eval_expr(contrast_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(seed) = eval_expr(seed_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(brush_angle) = eval_expr(brush_angle_expr, time_norm, time_sec) else {
                        continue;
                    };
                    let Ok(bump_strength) = eval_expr(bump_strength_expr, time_norm, time_sec)
                    else {
                        continue;
                    };
                    let Ok(relief) = eval_expr(relief_expr, time_norm, time_sec) else {
                        continue;
                    };
                    out.process_effects.push(
                        RuntimeProcessEffectInstance::new("texture_overlay")
                            .with_float("kind", *kind_id)
                            .with_float("scale", scale.clamp(0.001, 4096.0))
                            .with_float("strength", strength.clamp(0.0, 1.0))
                            .with_float("contrast", contrast.clamp(0.0, 2.0))
                            .with_float("seed", seed)
                            .with_float("brush_angle", brush_angle)
                            .with_float("bump_strength", bump_strength.clamp(0.0, 2.0))
                            .with_float("relief", relief.clamp(0.0, 2.0)),
                    );
                }
            }
        }

        out
    }

    pub fn summary(&self) -> String {
        format!(
            "Runtime ready: total_passes={}, supported_passes={}, unsupported_passes={}",
            self.total_pass_count(),
            self.supported_pass_count(),
            self.unsupported_kernels.len()
        )
    }
}

fn pass_param<'a>(pass: &'a PassNode, key: &str) -> Option<&'a str> {
    pass.params
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.value.as_str())
}

fn resolve_kernel_for_pass(pass: &PassNode) -> Result<String, RuntimeCompileError> {
    resolve_pass_kernel(pass).ok_or_else(|| RuntimeCompileError {
        message: format!(
            "pass {} missing kernel and no default mapping for effect '{}'",
            pass.id, pass.effect
        ),
    })
}

fn pass_param_f32(pass: &PassNode, keys: &[&str]) -> Option<f32> {
    for key in keys {
        if let Some(v) = pass_param(pass, key)
            && let Ok(parsed) = normalize_param_expr(v).parse::<f32>()
        {
            return Some(parsed);
        }
    }
    None
}

fn is_bloom_effect(effect: &str) -> bool {
    crate::process::effect_kind::is_bloom_family(effect)
}

fn is_glow_stack_effect(effect: &str) -> bool {
    matches!(
        crate::process::effect_kind::resolve_process_effect(effect),
        Some(crate::process::effect_kind::ProcessEffect::GlowStack)
    )
}

fn is_brightness_effect(effect: &str) -> bool {
    matches!(
        crate::process::effect_kind::resolve_process_effect(effect),
        Some(crate::process::effect_kind::ProcessEffect::Brightness)
    )
}

fn is_tone_map_effect(effect: &str) -> bool {
    matches!(
        crate::process::effect_kind::resolve_process_effect(effect),
        Some(crate::process::effect_kind::ProcessEffect::ToneMap)
    )
}

fn is_light_sweep_effect(effect: &str) -> bool {
    matches!(
        crate::process::effect_kind::resolve_process_effect(effect),
        Some(crate::process::effect_kind::ProcessEffect::LightSweep)
    )
}

fn is_texture_overlay_effect(effect: &str) -> bool {
    matches!(
        crate::process::effect_kind::resolve_process_effect(effect),
        Some(crate::process::effect_kind::ProcessEffect::TextureOverlay)
    )
}

fn texture_overlay_kind_id(value: &str) -> f32 {
    match value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_ascii_lowercase()
        .replace(['-', '_'], "")
        .as_str()
    {
        "noise" => 0.0,
        "film" | "grain" | "filmgrain" => 2.0,
        "scanline" | "scanlines" => 3.0,
        "canvas" | "fabric" | "cloth" => 4.0,
        "impasto" | "thickpaint" | "oilpaint" | "oilpainting" => 5.0,
        "brushedpaint" | "brushpaint" | "paintbrush" | "brushed" => 6.0,
        _ => 1.0,
    }
}

fn parse_runtime_color(value: &str) -> Option<[u8; 4]> {
    crate::scene::drawable::parse_color(value.trim().trim_matches('"').trim_matches('\'')).ok()
}

fn normalize_param_expr(value: &str) -> String {
    value.trim().trim_matches('"').trim().to_string()
}

pub fn eval_time_expr(value: &str, time_norm: f32, time_sec: f32) -> Result<f32, String> {
    let expr = normalize_param_expr(value);
    // Static numeric fields dominate large Scene3D graphs. Sending every
    // literal through custom-call folding, variable replacement and exmex
    // parsing on every frame is equivalent but needlessly expensive.
    if let Ok(value) = expr.parse::<f32>() {
        return Ok(value);
    }
    eval_expr(&expr, time_norm, time_sec)
}

#[derive(Debug, Clone)]
pub(crate) enum CompiledTimeExpr {
    Constant(f32),
    Curve(Vec<CurvePoint>),
    Dynamic(String),
}

impl CompiledTimeExpr {
    pub(crate) fn compile(value: &str) -> Result<Self, String> {
        let expr = normalize_param_expr(value);
        if let Ok(value) = expr.parse::<f32>() {
            return Ok(Self::Constant(value));
        }
        if is_curve_expr(&expr) {
            return parse_curve_points(&expr).map(Self::Curve);
        }
        Ok(Self::Dynamic(expr))
    }

    pub(crate) fn evaluate(&self, time_norm: f32, time_sec: f32) -> Result<f32, String> {
        match self {
            Self::Constant(value) => Ok(*value),
            Self::Curve(points) => eval_compiled_curve_points(points, time_sec),
            Self::Dynamic(expr) => eval_expr(expr, time_norm, time_sec),
        }
    }
}

fn validate_expr(expr: &str) -> Result<(), String> {
    eval_expr(expr, 0.5, 1.0).map(|_| ())
}

fn eval_expr(expr: &str, time_norm: f32, time_sec: f32) -> Result<f32, String> {
    if is_curve_expr(expr) {
        return eval_curve_points(expr, time_sec);
    }
    let folded = fold_custom_calls(expr, time_norm, time_sec)?;
    let replaced = replace_time_vars(&folded, time_norm, time_sec);
    let parsed = exmex::FlatEx::<f64>::parse(&replaced).map_err(|e| e.to_string())?;
    let val = parsed.eval(&[]).map_err(|e| e.to_string())?;
    Ok(val as f32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CustomCallKind {
    Floor,
    Min,
    Max,
    Clamp,
    Smoothstep,
    Random,
}

impl CustomCallKind {
    const fn name(self) -> &'static str {
        match self {
            CustomCallKind::Floor => "floor",
            CustomCallKind::Min => "min",
            CustomCallKind::Max => "max",
            CustomCallKind::Clamp => "clamp",
            CustomCallKind::Smoothstep => "smoothstep",
            CustomCallKind::Random => "random",
        }
    }
}

fn fold_custom_calls(expr: &str, time_norm: f32, time_sec: f32) -> Result<String, String> {
    let mut folded = expr.to_string();
    let mut guard = 0usize;
    while let Some((start_ix, end_ix, kind)) = find_next_custom_call(&folded)? {
        guard += 1;
        if guard > 256 {
            return Err("expression is too complex (too many nested function calls).".to_string());
        }
        let call = folded[start_ix..end_ix].to_string();
        let value = eval_custom_call(&call, kind, time_norm, time_sec)?;
        folded.replace_range(start_ix..end_ix, &format!("({value:.9})"));
    }
    Ok(folded)
}

fn eval_custom_call(
    call: &str,
    kind: CustomCallKind,
    time_norm: f32,
    time_sec: f32,
) -> Result<f32, String> {
    match kind {
        CustomCallKind::Floor => {
            let Some(inner) = parse_single_arg_call(call, "floor")? else {
                return Err("invalid floor() call.".to_string());
            };
            Ok(eval_expr(inner, time_norm, time_sec)?.floor())
        }
        CustomCallKind::Clamp => {
            let Some((value_expr, min_expr, max_expr)) = parse_clamp_call(call)? else {
                return Err("invalid clamp() call.".to_string());
            };
            let value = eval_expr(value_expr, time_norm, time_sec)?;
            let min_v = eval_expr(min_expr, time_norm, time_sec)?;
            let max_v = eval_expr(max_expr, time_norm, time_sec)?;
            let lo = min_v.min(max_v);
            let hi = min_v.max(max_v);
            Ok(value.clamp(lo, hi))
        }
        CustomCallKind::Min | CustomCallKind::Max => {
            let Some((parsed_kind, left, right)) = parse_min_max_call(call)? else {
                return Err(format!("invalid {}() call.", kind.name()));
            };
            let a = eval_expr(left, time_norm, time_sec)?;
            let b = eval_expr(right, time_norm, time_sec)?;
            let is_min = parsed_kind == "min";
            Ok(if is_min { a.min(b) } else { a.max(b) })
        }
        CustomCallKind::Smoothstep => {
            let Some((edge0_expr, edge1_expr, x_expr)) = parse_smoothstep_call(call)? else {
                return Err("invalid smoothstep() call.".to_string());
            };
            let edge0 = eval_expr(edge0_expr, time_norm, time_sec)?;
            let edge1 = eval_expr(edge1_expr, time_norm, time_sec)?;
            let x = eval_expr(x_expr, time_norm, time_sec)?;
            let span = edge1 - edge0;
            if span.abs() <= f32::EPSILON {
                return Ok(if x < edge0 { 0.0 } else { 1.0 });
            }
            let t = ((x - edge0) / span).clamp(0.0, 1.0);
            Ok(t * t * (3.0 - 2.0 * t))
        }
        CustomCallKind::Random => {
            let Some(args) = parse_random_call(call)? else {
                return Err("invalid random() call.".to_string());
            };
            match args.as_slice() {
                [seed_expr] => {
                    let seed = eval_expr(seed_expr, time_norm, time_sec)?;
                    Ok(seeded_random_unit(seed))
                }
                [min_expr, max_expr, seed_expr] => {
                    let min_v = eval_expr(min_expr, time_norm, time_sec)?;
                    let max_v = eval_expr(max_expr, time_norm, time_sec)?;
                    let seed = eval_expr(seed_expr, time_norm, time_sec)?;
                    let lo = min_v.min(max_v);
                    let hi = min_v.max(max_v);
                    Ok(lo + (hi - lo) * seeded_random_unit(seed))
                }
                _ => Err(
                    "random() requires either random(seed) or random(min,max,seed).".to_string(),
                ),
            }
        }
    }
}

fn find_next_custom_call(expr: &str) -> Result<Option<(usize, usize, CustomCallKind)>, String> {
    let kinds = [
        CustomCallKind::Floor,
        CustomCallKind::Clamp,
        CustomCallKind::Smoothstep,
        CustomCallKind::Random,
        CustomCallKind::Min,
        CustomCallKind::Max,
    ];
    for (ix, ch) in expr.char_indices() {
        if !matches!(ch, 'c' | 'f' | 'm' | 'r' | 's') {
            continue;
        }
        let prev_is_ident = expr[..ix]
            .chars()
            .next_back()
            .is_some_and(is_identifier_char);
        if prev_is_ident {
            continue;
        }
        for kind in kinds {
            let name = kind.name();
            let call_prefix = format!("{name}(");
            if !expr[ix..].starts_with(&call_prefix) {
                continue;
            }
            let open_ix = ix + name.len();
            let end_ix = matching_close_paren_end(expr, open_ix)?;
            return Ok(Some((ix, end_ix, kind)));
        }
    }
    Ok(None)
}

fn is_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn matching_close_paren_end(expr: &str, open_ix: usize) -> Result<usize, String> {
    let Some(open_ch) = expr[open_ix..].chars().next() else {
        return Err("invalid expression: missing opening parenthesis.".to_string());
    };
    if open_ch != '(' {
        return Err("invalid expression: expected opening parenthesis.".to_string());
    }
    let mut paren_depth = 0_i32;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escape = false;
    for (ix, ch) in expr.char_indices().skip_while(|(ix, _)| *ix < open_ix) {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            continue;
        }
        if in_single_quote {
            if ch == '\'' {
                in_single_quote = false;
            }
            continue;
        }
        if in_double_quote {
            if ch == '"' {
                in_double_quote = false;
            }
            continue;
        }
        match ch {
            '\'' => in_single_quote = true,
            '"' => in_double_quote = true,
            '(' => paren_depth += 1,
            ')' => {
                paren_depth -= 1;
                if paren_depth == 0 {
                    return Ok(ix + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    Err("invalid expression: missing closing parenthesis.".to_string())
}

fn parse_min_max_call(expr: &str) -> Result<Option<(&'static str, &str, &str)>, String> {
    let (kind, inner) = if let Some(inner) = outer_call_inner(expr, "min")? {
        ("min", inner)
    } else if let Some(inner) = outer_call_inner(expr, "max")? {
        ("max", inner)
    } else {
        return Ok(None);
    };
    let Some((left, right)) = split_binary_args(inner) else {
        return Err(format!("{kind}() requires exactly two arguments."));
    };
    Ok(Some((kind, left.trim(), right.trim())))
}

fn parse_single_arg_call<'a>(expr: &'a str, name: &str) -> Result<Option<&'a str>, String> {
    let Some(inner) = outer_call_inner(expr, name)? else {
        return Ok(None);
    };
    if split_top_level_args(inner).len() != 1 {
        return Err(format!("{name}() requires exactly one argument."));
    }
    Ok(Some(inner.trim()))
}

fn parse_clamp_call(expr: &str) -> Result<Option<(&str, &str, &str)>, String> {
    let Some(inner) = outer_call_inner(expr, "clamp")? else {
        return Ok(None);
    };
    let Some((value_expr, min_expr, max_expr)) = split_ternary_args(inner) else {
        return Err("clamp() requires exactly three arguments.".to_string());
    };
    Ok(Some((value_expr.trim(), min_expr.trim(), max_expr.trim())))
}

fn parse_smoothstep_call(expr: &str) -> Result<Option<(&str, &str, &str)>, String> {
    let Some(inner) = outer_call_inner(expr, "smoothstep")? else {
        return Ok(None);
    };
    let Some((edge0_expr, edge1_expr, x_expr)) = split_ternary_args(inner) else {
        return Err("smoothstep() requires exactly three arguments.".to_string());
    };
    Ok(Some((edge0_expr.trim(), edge1_expr.trim(), x_expr.trim())))
}

fn parse_random_call(expr: &str) -> Result<Option<Vec<&str>>, String> {
    let Some(inner) = outer_call_inner(expr, "random")? else {
        return Ok(None);
    };
    let args = split_top_level_args(inner)
        .into_iter()
        .map(str::trim)
        .filter(|arg| !arg.is_empty())
        .collect::<Vec<_>>();
    if matches!(args.len(), 1 | 3) {
        Ok(Some(args))
    } else {
        Err("random() requires either random(seed) or random(min,max,seed).".to_string())
    }
}

fn seeded_random_unit(seed: f32) -> f32 {
    // Deterministic pseudo-randomness keeps preview, export, caching, and tests reproducible.
    let x = ((seed as f64) * 12.9898 + 78.233).sin() * 43_758.545_3;
    x.fract().rem_euclid(1.0) as f32
}

fn outer_call_inner<'a>(expr: &'a str, name: &str) -> Result<Option<&'a str>, String> {
    let trimmed = expr.trim();
    let Some(rest) = trimmed.strip_prefix(name) else {
        return Ok(None);
    };
    let Some(_rest_after_open) = rest.strip_prefix('(') else {
        return Ok(None);
    };
    let open_ix = name.len();
    let end_ix = matching_close_paren_end(trimmed, open_ix)?;
    if end_ix != trimmed.len() {
        return Ok(None);
    }
    Ok(Some(&trimmed[open_ix + 1..trimmed.len() - 1]))
}

fn split_binary_args(input: &str) -> Option<(&str, &str)> {
    let args = split_top_level_args(input);
    if args.len() == 2 {
        Some((args[0], args[1]))
    } else {
        None
    }
}

fn split_ternary_args(input: &str) -> Option<(&str, &str, &str)> {
    let args = split_top_level_args(input);
    if args.len() == 3 {
        Some((args[0], args[1], args[2]))
    } else {
        None
    }
}

fn split_top_level_args(input: &str) -> Vec<&str> {
    let mut paren_depth = 0_i32;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escape = false;
    let mut start_ix = 0usize;
    let mut out = Vec::new();
    for (ix, ch) in input.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            continue;
        }
        if in_single_quote {
            if ch == '\'' {
                in_single_quote = false;
            }
            continue;
        }
        if in_double_quote {
            if ch == '"' {
                in_double_quote = false;
            }
            continue;
        }
        match ch {
            '\'' => in_single_quote = true,
            '"' => in_double_quote = true,
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            ',' if paren_depth == 0 => {
                out.push(&input[start_ix..ix]);
                start_ix = ix + ch.len_utf8();
            }
            _ => {}
        }
    }
    out.push(&input[start_ix..]);
    out
}

fn replace_time_vars(expr: &str, time_norm: f32, time_sec: f32) -> String {
    expr.replace("$time.norm", &format!("{time_norm:.9}"))
        .replace("$time.sec", &format!("{time_sec:.9}"))
}

fn parse_blur_sharpen_effect(raw: &str) -> Result<BlurSharpenMode, String> {
    let normalized = raw.trim().trim_matches('"').trim().to_ascii_lowercase();
    let normalized = normalized.replace('-', "_");
    match normalized.as_str() {
        "gaussian_blur" | "gaussian_5tap_blur" => Ok(BlurSharpenMode::Gaussian5tapBlur),
        "gaussian_5tap_h" => Ok(BlurSharpenMode::Gaussian5tapH),
        "gaussian_5tap_v" => Ok(BlurSharpenMode::Gaussian5tapV),
        "box" => Ok(BlurSharpenMode::Box),
        "sharpen" => Ok(BlurSharpenMode::Unsharp),
        "unsharp" => Ok(BlurSharpenMode::Unsharp),
        other => Err(format!(
            "expected gaussian_blur | gaussian_5tap_blur | gaussian_5tap_h | gaussian_5tap_v | sharpen | unsharp | box, got '{}'",
            other
        )),
    }
}

fn parse_transition_core_effect(raw: &str) -> Result<TransitionCoreEffect, String> {
    let normalized = raw.trim().trim_matches('"').trim().to_ascii_lowercase();
    let normalized = normalized.replace('-', "_");
    match normalized.as_str() {
        "fade_in" => Ok(TransitionCoreEffect::FadeIn),
        "fade_out" => Ok(TransitionCoreEffect::FadeOut),
        "dip" | "dip_to_black" => Ok(TransitionCoreEffect::Dip),
        "dissolve" => Ok(TransitionCoreEffect::Dissolve),
        other => Err(format!(
            "expected fade_in | fade_out | dip | dissolve, got '{}'",
            other
        )),
    }
}

fn parse_transition_easing_param(raw: &str) -> Result<PassTransitionEasing, String> {
    let normalized = raw.trim().trim_matches('"').trim().to_ascii_lowercase();
    match normalized.as_str() {
        "linear" => Ok(PassTransitionEasing::Linear),
        "ease-in" | "ease_in" => Ok(PassTransitionEasing::EaseIn),
        "ease-out" | "ease_out" => Ok(PassTransitionEasing::EaseOut),
        "ease-in-out" | "ease_in_out" => Ok(PassTransitionEasing::EaseInOut),
        other => Err(format!(
            "expected linear | ease-in | ease-out | ease-in-out, got '{}'",
            other
        )),
    }
}

fn v1_dissolve_opacity_from_mix(mix: f32) -> f32 {
    ((mix * 2.0) - 1.0).abs().clamp(0.0, 1.0)
}

fn apply_transition_easing(t: f32, easing: PassTransitionEasing) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match easing {
        PassTransitionEasing::Linear => t,
        PassTransitionEasing::EaseIn => t * t,
        PassTransitionEasing::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
        PassTransitionEasing::EaseInOut => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                1.0 - ((-2.0 * t + 2.0).powi(2) / 2.0)
            }
        }
    }
}

fn is_curve_expr(expr: &str) -> bool {
    let trimmed = expr.trim();
    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("curve(") && trimmed.ends_with(')')
}

fn eval_curve_points(expr: &str, time_sec: f32) -> Result<f32, String> {
    let points = parse_curve_points(expr)?;
    eval_compiled_curve_points(&points, time_sec)
}

fn eval_compiled_curve_points(points: &[CurvePoint], time_sec: f32) -> Result<f32, String> {
    if points.is_empty() {
        return Err("curve() requires at least one point.".to_string());
    }
    if points.len() == 1 {
        return Ok(points[0].value);
    }
    let x = time_sec.max(0.0);
    if x <= points[0].t_sec {
        return Ok(points[0].value);
    }
    if x >= points[points.len() - 1].t_sec {
        return Ok(points[points.len() - 1].value);
    }
    for idx in 0..(points.len() - 1) {
        let a = points[idx];
        let b = points[idx + 1];
        if x >= a.t_sec && x <= b.t_sec {
            let span = (b.t_sec - a.t_sec).max(0.000_001);
            let u = ((x - a.t_sec) / span).clamp(0.0, 1.0);
            let eased = apply_curve_ease(u, a.ease);
            return Ok(a.value + (b.value - a.value) * eased);
        }
    }
    Ok(points[points.len() - 1].value)
}

fn parse_curve_points(expr: &str) -> Result<Vec<CurvePoint>, String> {
    let trimmed = expr.trim();
    let Some(open_ix) = trimmed.find('(') else {
        return Err("curve expression missing '('.".to_string());
    };
    if !trimmed.ends_with(')') {
        return Err("curve expression missing ')'.".to_string());
    }
    let inner_owned = trimmed[open_ix + 1..trimmed.len() - 1]
        .trim()
        .replace("\\\"", "\"")
        .replace("\\'", "'");
    let mut inner = inner_owned.trim();
    if (inner.starts_with('"') && inner.ends_with('"'))
        || (inner.starts_with('\'') && inner.ends_with('\''))
    {
        inner = inner[1..inner.len() - 1].trim();
    }
    if inner.is_empty() {
        return Err("curve() requires at least one point.".to_string());
    }

    let mut points = Vec::<CurvePoint>::new();
    for raw_point in split_curve_point_tokens(inner)? {
        let token = raw_point.trim();
        if token.is_empty() {
            continue;
        }
        let parts: Vec<&str> = token.split(':').map(str::trim).collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(format!(
                "invalid curve point '{}'; expected t:value[:ease]",
                token
            ));
        }
        let t_sec = parts[0]
            .parse::<f32>()
            .map_err(|_| format!("invalid curve time '{}'", parts[0]))?;
        let value = parts[1]
            .parse::<f32>()
            .map_err(|_| format!("invalid curve value '{}'", parts[1]))?;
        if !t_sec.is_finite() || !value.is_finite() {
            return Err(format!("non-finite curve point '{}'", token));
        }
        let ease = if parts.len() == 3 {
            parse_curve_ease(parts[2])?
        } else {
            CurveEase::Linear
        };
        points.push(CurvePoint {
            t_sec: t_sec.max(0.0),
            value,
            ease,
        });
    }

    if points.is_empty() {
        return Err("curve() requires at least one valid point.".to_string());
    }
    points.sort_by(|a, b| a.t_sec.total_cmp(&b.t_sec));
    Ok(points)
}

fn split_curve_point_tokens(inner: &str) -> Result<Vec<&str>, String> {
    let mut tokens = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;

    for (idx, ch) in inner.char_indices() {
        match ch {
            '(' => depth = depth.saturating_add(1),
            ')' => {
                if depth == 0 {
                    return Err("curve expression has unmatched ')'.".to_string());
                }
                depth -= 1;
            }
            ',' if depth == 0 => {
                tokens.push(&inner[start..idx]);
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }

    if depth != 0 {
        return Err("curve expression has unmatched '('.".to_string());
    }

    tokens.push(&inner[start..]);
    Ok(tokens)
}

pub(crate) fn parse_curve_ease(raw: &str) -> Result<CurveEase, String> {
    let normalized = raw
        .trim()
        .trim_matches('"')
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_");
    if let Some(args) = normalized
        .strip_prefix("ease(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let values: Vec<f32> = args
            .split(',')
            .map(str::trim)
            .map(|v| {
                v.parse::<f32>()
                    .map_err(|_| format!("invalid cubic ease value '{}'", v))
            })
            .collect::<Result<_, _>>()?;
        if values.len() != 4 {
            return Err(format!(
                "invalid curve easing '{}'; expected ease(x1,y1,x2,y2)",
                raw.trim()
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("non-finite curve easing '{}'", raw.trim()));
        }
        return Ok(CurveEase::Ease(values[0], values[1], values[2], values[3]));
    }
    match normalized.as_str() {
        "linear" => Ok(CurveEase::Linear),
        "ease_in" => Ok(CurveEase::EaseIn),
        "ease_out" => Ok(CurveEase::EaseOut),
        "ease_in_out" => Ok(CurveEase::EaseInOut),
        other => Err(format!(
            "invalid curve easing '{}'; expected linear | ease_in | ease_out | ease_in_out | ease(x1,y1,x2,y2)",
            other
        )),
    }
}

pub(crate) fn apply_curve_ease(t: f32, easing: CurveEase) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match easing {
        CurveEase::Linear => t,
        CurveEase::EaseIn => t * t,
        CurveEase::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
        CurveEase::EaseInOut => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                1.0 - ((-2.0 * t + 2.0).powi(2) / 2.0)
            }
        }
        CurveEase::Ease(x1, y1, x2, y2) => apply_cubic_bezier_ease(t, x1, y1, x2, y2),
    }
}

fn apply_cubic_bezier_ease(t: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    let x1 = x1.clamp(0.0, 1.0);
    let x2 = x2.clamp(0.0, 1.0);
    let mut u = t;
    for _ in 0..8 {
        let x = cubic_bezier_sample(x1, x2, u) - t;
        if x.abs() < 0.000_01 {
            return cubic_bezier_sample(y1, y2, u);
        }
        let dx = cubic_bezier_derivative(x1, x2, u);
        if dx.abs() < 0.000_001 {
            break;
        }
        u = (u - x / dx).clamp(0.0, 1.0);
    }

    let mut lo = 0.0;
    let mut hi = 1.0;
    u = t;
    for _ in 0..20 {
        let x = cubic_bezier_sample(x1, x2, u);
        if (x - t).abs() < 0.000_01 {
            break;
        }
        if x < t {
            lo = u;
        } else {
            hi = u;
        }
        u = (lo + hi) * 0.5;
    }

    cubic_bezier_sample(y1, y2, u)
}

fn cubic_bezier_sample(a: f32, b: f32, t: f32) -> f32 {
    let inv = 1.0 - t;
    3.0 * inv * inv * t * a + 3.0 * inv * t * t * b + t * t * t
}

fn cubic_bezier_derivative(a: f32, b: f32, t: f32) -> f32 {
    let inv = 1.0 - t;
    3.0 * inv * inv * a + 6.0 * inv * t * (b - a) + 3.0 * t * t * (1.0 - b)
}

#[cfg(test)]
mod tests {
    use crate::dsl::parse_graph_script;

    use super::{
        BlurSharpenMode, RuntimeProcessParamValue, compile_runtime_program, eval_time_expr,
    };

    #[test]
    fn runtime_eval_invert_mix_changes_with_time() {
        let script = r#"
<Graph fps={30} duration="2s" size={[256,256]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[256,256]} />
  <Pass id="invert_pulse" kernel="invert_mix.wgsl" effect="invert_mix"
        in={["src"]}
        out={["out"]}
        params={{
          mix: "0.5 + 0.5*sin($time.sec*6.28318)"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let at_0 = runtime.evaluate_frame(0);
        let at_15 = runtime.evaluate_frame(15);
        assert!(at_0.invert_mix >= 0.0 && at_0.invert_mix <= 1.0);
        assert!(at_15.invert_mix >= 0.0 && at_15.invert_mix <= 1.0);
    }

    #[test]
    fn runtime_eval_time_expr_preserves_static_numeric_literals() {
        assert_eq!(eval_time_expr("  -3.25  ", 0.2, 7.0), Ok(-3.25));
        assert_eq!(eval_time_expr("\"1.5\"", 0.8, 2.0), Ok(1.5));
    }

    #[test]
    fn runtime_eval_time_expr_supports_min_for_fade_in() {
        let at_half =
            eval_time_expr("min($time.sec / 1.0, 1.0)", 0.5, 0.5).expect("fade expression");
        let at_done =
            eval_time_expr("min($time.sec / 1.0, 1.0)", 1.0, 2.0).expect("fade expression");
        assert!((at_half - 0.5).abs() < 0.001);
        assert!((at_done - 1.0).abs() < 0.001);
    }

    #[test]
    fn runtime_eval_time_expr_supports_clamp_with_arithmetic() {
        let expr = "clamp(($time.sec-0.3)/0.8,0,1) * clamp((8-$time.sec)/1.2,0,1)";
        let at_start = eval_time_expr(expr, 0.0, 0.0).expect("clamp expression");
        let at_mid = eval_time_expr(expr, 0.4, 3.5).expect("clamp expression");
        let at_tail = eval_time_expr(expr, 1.0, 8.0).expect("clamp expression");
        assert!(
            at_start.abs() < 0.0001,
            "unexpected start value: {at_start}"
        );
        assert!(at_mid > 0.5, "unexpected mid value: {at_mid}");
        assert!(at_tail.abs() < 0.0001, "unexpected tail value: {at_tail}");
    }

    #[test]
    fn runtime_eval_time_expr_supports_smoothstep() {
        let expr = "smoothstep(0.10,0.72,$time.norm)";
        let at_start = eval_time_expr(expr, 0.0, 0.0).expect("smoothstep expression");
        let at_mid = eval_time_expr(expr, 0.41, 0.0).expect("smoothstep expression");
        let at_end = eval_time_expr(expr, 0.9, 0.0).expect("smoothstep expression");
        assert!(
            at_start.abs() < 0.0001,
            "unexpected start value: {at_start}"
        );
        assert!(
            (at_mid - 0.5).abs() < 0.001,
            "unexpected mid value: {at_mid}"
        );
        assert!(
            (at_end - 1.0).abs() < 0.001,
            "unexpected end value: {at_end}"
        );
    }

    #[test]
    fn runtime_eval_time_expr_supports_deterministic_random() {
        let a = eval_time_expr("random(7)", 0.0, 0.0).expect("random expression");
        let b = eval_time_expr("random(7)", 0.6, 3.0).expect("random expression");
        let ranged = eval_time_expr("random(100,200,floor($time.sec*30)+7)", 0.0, 0.5)
            .expect("ranged random expression");

        assert!(
            (a - b).abs() < 0.0001,
            "same seed should be deterministic: {a} vs {b}"
        );
        assert!(
            (100.0..=200.0).contains(&ranged),
            "ranged random should stay inside bounds, got {ranged}"
        );
    }

    #[test]
    fn runtime_eval_blur_kernel_maps_sigma_to_layer_blur() {
        let script = r#"
<Graph fps={30} duration="1s" size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="fx_blur" kind="compute" kernel="blur_sharpen_detail_gaussian.wgsl" effect="gaussian_5tap_blur"
        in={["src"]}
        out={["out"]}
        params={{
          sigma: "1.5 + 0.5*sin($time.sec*6.28318)"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let at_0 = runtime.evaluate_frame(0);
        let at_8 = runtime.evaluate_frame(8);
        let sigma0 = at_0.layer_blur_sigma.unwrap_or(-1.0);
        let sigma8 = at_8.layer_blur_sigma.unwrap_or(-1.0);
        assert!(sigma0 >= 0.0);
        assert!(sigma8 >= 0.0);
        assert!((sigma0 - sigma8).abs() > 0.0001);
    }

    #[test]
    fn runtime_duration_limits_effect_window() {
        let script = r#"
<Graph fps={30} apply="graph" duration="2s" size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="fx_blur" kind="compute" kernel="blur_sharpen_detail_gaussian.wgsl" effect="gaussian_5tap_blur"
        in={["src"]}
        out={["out"]}
        params={{
          sigma: "10.0"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let in_window = runtime.evaluate_frame(59);
        let out_window = runtime.evaluate_frame(60);
        assert_eq!(in_window.layer_blur_sigma, Some(10.0));
        assert_eq!(out_window.layer_blur_sigma, None);
    }

    #[test]
    fn runtime_default_apply_clip_does_not_gate_by_duration() {
        let script = r#"
<Graph fps={30} duration="2s" size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="fx_blur" kind="compute" kernel="blur_sharpen_detail_gaussian.wgsl" effect="gaussian_5tap_blur"
        in={["src"]}
        out={["out"]}
        params={{
          sigma: "10.0"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out_window = runtime.evaluate_frame(120);
        assert_eq!(out_window.layer_blur_sigma, Some(10.0));
    }

    #[test]
    fn runtime_uses_pass_effect_field_for_blur_sharpen() {
        let script = r##"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="fx_unsharp" kind="compute" kernel="blur_sharpen_detail_gaussian.wgsl" effect="unsharp"
        in={["src"]}
        out={["out"]}
        params={{ sigma: "2.0" }} />
  <Present from="out" />
</Graph>
"##;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        assert_eq!(out.blur_sharpen_mode, Some(BlurSharpenMode::Unsharp));
        assert_eq!(out.layer_blur_sigma, None);
        assert_eq!(out.layer_sharpen_sigma, Some(2.0));
    }

    #[test]
    fn runtime_rejects_invalid_blur_effect() {
        let script = r##"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="fx_bad" kind="compute" kernel="blur_sharpen_detail_gaussian.wgsl" effect="gaussian_9tap"
        in={["src"]}
        out={["out"]}
        params={{ sigma: "2.0" }} />
  <Present from="out" />
</Graph>
"##;
        let graph = parse_graph_script(script).expect("graph parse");
        let err =
            compile_runtime_program(graph).expect_err("compile should fail on invalid effect");
        assert!(
            err.message
                .contains(
                    "expected gaussian_blur | gaussian_5tap_blur | gaussian_5tap_h | gaussian_5tap_v | sharpen | unsharp | box",
                ),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn runtime_transition_core_fade_in_uses_param_window() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fade_in" kind="render" role="transition" kernel="transition_core.wgsl"
        effect="fade_in"
        in={["under"]} out={["out"]}
        params={{
          startSec: "0.5",
          durationSec: "1.0"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        assert_eq!(
            runtime.evaluate_frame(0).layer_transition_opacity,
            Some(0.0)
        );
        assert_eq!(
            runtime.evaluate_frame(15).layer_transition_opacity,
            Some(0.0)
        );
        assert_eq!(
            runtime.evaluate_frame(30).layer_transition_opacity,
            Some(0.5)
        );
        assert_eq!(
            runtime.evaluate_frame(45).layer_transition_opacity,
            Some(1.0)
        );
    }

    #[test]
    fn runtime_transition_core_fade_out_uses_param_window() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fade_out" kind="render" role="transition" kernel="transition_core.wgsl"
        effect="fade_out"
        in={["under"]} out={["out"]}
        params={{
          startSec: "0.0",
          durationSec: "2.0"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        assert_eq!(
            runtime.evaluate_frame(0).layer_transition_opacity,
            Some(1.0)
        );
        assert_eq!(
            runtime.evaluate_frame(30).layer_transition_opacity,
            Some(0.5)
        );
        assert_eq!(
            runtime.evaluate_frame(60).layer_transition_opacity,
            Some(0.0)
        );
    }

    #[test]
    fn runtime_transition_core_fade_out_defaults_to_tail_when_start_missing() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fade_out" kind="render" role="transition" kernel="transition_core.wgsl"
        effect="fade_out"
        in={["under"]} out={["out"]}
        params={{
          durationSec: "0.6"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        // Simulate a 3s layer clip: start fade-out at 2.4s when startSec is omitted.
        assert_eq!(
            runtime
                .evaluate_at_time_sec(1.0, Some(3.0))
                .layer_transition_opacity,
            Some(1.0)
        );
        assert_eq!(
            runtime
                .evaluate_at_time_sec(2.4, Some(3.0))
                .layer_transition_opacity,
            Some(1.0)
        );
        let mid = runtime
            .evaluate_at_time_sec(2.7, Some(3.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);
        assert!((mid - 0.5).abs() < 0.0005, "unexpected mid opacity: {mid}");
        let end = runtime
            .evaluate_at_time_sec(3.0, Some(3.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);
        assert!(end.abs() < 0.0005, "unexpected end opacity: {end}");
    }

    #[test]
    fn runtime_transition_core_fade_out_ignores_progress_expr_and_uses_tail_window() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fade_out" kind="render" role="transition" kernel="transition_core.wgsl"
        effect="fade_out"
        in={["under"]} out={["out"]}
        params={{
          durationSec: "0.6",
          mix: "$time.norm"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        // With a 3s layer clip, fade-out should still happen at tail regardless of mix/progress.
        assert_eq!(
            runtime
                .evaluate_at_time_sec(1.0, Some(3.0))
                .layer_transition_opacity,
            Some(1.0)
        );
        let mid = runtime
            .evaluate_at_time_sec(2.7, Some(3.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);
        assert!((mid - 0.5).abs() < 0.0005, "unexpected mid opacity: {mid}");
    }

    #[test]
    fn runtime_transition_core_fade_in_and_fade_out_compose_for_clip() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fade_in" kind="render" role="transition" kernel="transition_core.wgsl"
        effect="fade_in"
        in={["under"]} out={["out"]}
        params={{
          durationSec: "2.0"
        }} />
  <Pass id="fade_out" kind="render" role="transition" kernel="transition_core.wgsl"
        effect="fade_out"
        in={["under"]} out={["out"]}
        params={{
          durationSec: "2.0"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        // Simulate a 10s layer clip.
        assert_eq!(
            runtime
                .evaluate_at_time_sec(0.0, Some(10.0))
                .layer_transition_opacity,
            Some(0.0)
        );
        let fade_in_mid = runtime
            .evaluate_at_time_sec(1.0, Some(10.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);
        assert!(
            (fade_in_mid - 0.5).abs() < 0.0005,
            "unexpected fade-in midpoint opacity: {fade_in_mid}"
        );
        assert_eq!(
            runtime
                .evaluate_at_time_sec(5.0, Some(10.0))
                .layer_transition_opacity,
            Some(1.0)
        );
        let fade_out_mid = runtime
            .evaluate_at_time_sec(9.0, Some(10.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);
        assert!(
            (fade_out_mid - 0.5).abs() < 0.0005,
            "unexpected fade-out midpoint opacity: {fade_out_mid}"
        );
    }

    #[test]
    fn runtime_transition_core_dissolve_respects_easing() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Input id="prev" type="video" from="input:prev" />
  <Input id="next" type="video" from="input:next" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="dissolve" kind="render" role="transition" kernel="transition_core.wgsl"
        effect="dissolve"
        in={["prev","next"]} out={["out"]}
        params={{
          progress: "$time.sec*0.5",
          easing: "ease-in"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let at_mid = runtime.evaluate_frame(30);
        assert!(
            (at_mid.transition_dissolve_mix.unwrap_or(-1.0) - 0.25).abs() < 0.0001,
            "unexpected mix: {:?}",
            at_mid.transition_dissolve_mix
        );
    }

    #[test]
    fn runtime_transition_core_missing_effect_is_parser_error() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="bad_transition" kind="render" role="transition" kernel="transition_core.wgsl"
        in={["under"]} out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let err = parse_graph_script(script).expect_err("missing effect should fail in parser");
        assert!(
            err.message.contains("Missing required attribute: effect"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn runtime_rejects_removed_transition_kernel_names() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="legacy_transition" kind="render" role="transition" kernel="transition_dissolve.wgsl" effect="dissolve"
        in={["under"]} out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let err = compile_runtime_program(graph).expect_err("legacy transition kernel should fail");
        assert!(
            err.message
                .contains("use transition_core.wgsl + effect instead"),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn runtime_uses_explicit_kernel_with_effect() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="fx_blur" kind="compute" kernel="blur_sharpen_detail_gaussian.wgsl" effect="gaussian_5tap_blur"
        in={["src"]}
        out={["out"]}
        params={{ sigma: "2.0" }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        assert_eq!(
            out.blur_sharpen_mode,
            Some(BlurSharpenMode::Gaussian5tapBlur)
        );
        assert_eq!(out.layer_blur_sigma, Some(2.0));
    }

    #[test]
    fn runtime_opacity_effect_sets_layer_transition_opacity() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_opacity" kind="compute" kernel="composite_core.wgsl" effect="opacity"
        in={["under"]}
        out={["out"]}
        params={{ opacity: "0.7" }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        assert_eq!(out.layer_transition_opacity, Some(0.7));
    }

    #[test]
    fn runtime_transition_opacity_curve_uses_seconds_domain() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fade_in" kind="render" role="transition" kernel="transition_core.wgsl"
        effect="fade_in"
        in={["under"]} out={["out"]}
        params={{
          opacity: "curve(\"0.00:0.0:linear, 1.00:1.0:ease_in_out\")",
          durationSec: "2.0"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        assert_eq!(
            runtime.evaluate_frame(0).layer_transition_opacity,
            Some(0.0)
        );
        assert!(
            runtime
                .evaluate_frame(30)
                .layer_transition_opacity
                .unwrap_or(0.0)
                > 0.0
        );
        assert_eq!(
            runtime.evaluate_frame(60).layer_transition_opacity,
            Some(1.0)
        );
    }

    #[test]
    fn runtime_opacity_curve_matches_points_and_holds_tail_value() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_opacity" kind="compute" effect="opacity"
        in={["under"]} out={["out"]}
        params={{
          opacity: "curve(\"0.00:0.137:ease_in, 1.56:0.929:linear, 2.00:0.700:ease_in_out\")"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");

        let v0 = runtime
            .evaluate_at_time_sec(0.00, Some(30.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);
        let v156 = runtime
            .evaluate_at_time_sec(1.56, Some(30.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);
        let v178 = runtime
            .evaluate_at_time_sec(1.78, Some(30.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);
        let v200 = runtime
            .evaluate_at_time_sec(2.00, Some(30.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);
        let v500 = runtime
            .evaluate_at_time_sec(5.00, Some(30.0))
            .layer_transition_opacity
            .unwrap_or(-1.0);

        assert!((v0 - 0.137).abs() < 1e-4, "unexpected t=0.00 value: {v0}");
        assert!(
            (v156 - 0.929).abs() < 1e-4,
            "unexpected t=1.56 value: {v156}"
        );
        // Segment [1.56, 2.00] is linear (easing comes from the starting point at 1.56).
        assert!(
            (v178 - 0.8145).abs() < 1e-4,
            "unexpected t=1.78 value: {v178}"
        );
        assert!(
            (v200 - 0.700).abs() < 1e-4,
            "unexpected t=2.00 value: {v200}"
        );
        assert!(
            (v500 - 0.700).abs() < 1e-4,
            "unexpected t=5.00 value: {v500}"
        );
    }

    #[test]
    fn runtime_curve_expression_supports_cubic_ease_function() {
        let value = eval_time_expr("curve(\"0:0:ease(0.82,0,0.58,1), 1:100:linear\")", 0.0, 0.5)
            .expect("custom ease curve");

        assert!(
            value > 20.0 && value < 23.0,
            "unexpected cubic ease value at 0.5s: {value}"
        );
    }

    #[test]
    fn runtime_lut_effect_sets_layer_lut_mix() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_lut" kind="compute" kernel="color_core.wgsl" effect="lut"
        in={["under"]}
        out={["out"]}
        params={{ mix: "0.6 + 0.2*sin($time.sec)" }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        let mix = out.layer_lut_mix.expect("lut mix");
        assert!((0.0..=1.0).contains(&mix));
    }

    #[test]
    fn runtime_bloom_alias_sets_layer_bloom_fields() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_bloom" kind="compute" kernel="color_core.wgsl" effect="glow_bloom"
        in={["under"]}
        out={["out"]}
        params={{
          threshold: "0.64",
          intensity: "1.75",
          sigma: "22.0"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        assert_eq!(out.layer_bloom_threshold, Some(0.64));
        assert_eq!(out.layer_bloom_intensity, Some(1.75));
        assert_eq!(out.layer_bloom_sigma, Some(22.0));
    }

    #[test]
    fn runtime_tone_map_effect_sets_layer_tone_map_fields() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_tone_map" kind="compute" kernel="color_core.wgsl" effect="tone_map"
        in={["under"]}
        out={["out"]}
        params={{
          exposure: "0.35",
          contrast: "1.35",
          shoulder: "0.55",
          gamma: "2.0",
          saturation: "1.22"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        assert_eq!(out.layer_tone_map_exposure, Some(0.35));
        assert_eq!(out.layer_tone_map_contrast, Some(1.35));
        assert_eq!(out.layer_tone_map_shoulder, Some(0.55));
        assert_eq!(out.layer_tone_map_gamma, Some(2.0));
        assert_eq!(out.layer_tone_map_saturation, Some(1.22));
        let effect = out
            .process_effects
            .iter()
            .find(|effect| effect.effect_id == "tone_map")
            .expect("generic tone_map effect");
        assert_eq!(effect.float("exposure"), Some(0.35));
        assert_eq!(effect.float("contrast"), Some(1.35));
        assert_eq!(effect.float("shoulder"), Some(0.55));
        assert_eq!(effect.float("gamma"), Some(2.0));
        assert_eq!(effect.float("saturation"), Some(1.22));
    }

    #[test]
    fn runtime_light_sweep_effect_sets_layer_light_sweep_fields() {
        let script = r##"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_light_sweep" kind="compute" kernel="color_core.wgsl" effect="light_sweep"
        in={["under"]}
        out={["out"]}
        params={{
          position: "0.42",
          angle: "-18.0",
          width: "0.18",
          softness: "0.08",
          intensity: "1.6",
          color: "#80C7FF"
        }} />
  <Present from="out" />
</Graph>
"##;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        assert_eq!(out.layer_light_sweep_position, Some(0.42));
        assert_eq!(out.layer_light_sweep_angle, Some(-18.0));
        assert_eq!(out.layer_light_sweep_width, Some(0.18));
        assert_eq!(out.layer_light_sweep_softness, Some(0.08));
        assert_eq!(out.layer_light_sweep_intensity, Some(1.6));
        assert_eq!(out.layer_light_sweep_color, Some([128, 199, 255, 255]));
        let effect = out
            .process_effects
            .iter()
            .find(|effect| effect.effect_id == "light_sweep")
            .expect("generic light_sweep effect");
        assert_eq!(effect.float("position"), Some(0.42));
        assert_eq!(effect.float("angle"), Some(-18.0));
        assert_eq!(effect.float("width"), Some(0.18));
        assert_eq!(effect.float("softness"), Some(0.08));
        assert_eq!(effect.float("intensity"), Some(1.6));
        assert_eq!(effect.color("color"), Some([128, 199, 255, 255]));
    }

    #[test]
    fn runtime_glow_stack_preserves_tint_and_radii_for_process_effect() {
        let script = r##"
<Graph fps={30} size={[800,450]} renderSize={[800,450]}>
  <Process id="GlowStackProcess">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[800,450]} />
    <Pass id="post_glow_stack" kind="compute"
          effect="glow_stack"
          in={["src"]} out={["out"]}
          params={{
            threshold: "0.76",
            intensity: "0.2",
            radiusSmall: "6.0",
            radiusMedium: "1.0",
            radiusLarge: "4.0",
            tint: "#DB2626"
          }} />
  </Process>
  <Present from="GlowStackProcess" />
</Graph>
"##;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        assert_eq!(out.layer_bloom_threshold, Some(0.76));
        assert_eq!(out.layer_bloom_intensity, Some(0.2));
        assert_eq!(out.layer_bloom_sigma, Some(1.0));
        let effect = out
            .process_effects
            .iter()
            .find(|effect| effect.effect_id == "glow_stack")
            .expect("generic glow_stack effect");
        assert_eq!(effect.float("threshold"), Some(0.76));
        assert_eq!(effect.float("intensity"), Some(0.2));
        assert_eq!(effect.float("radiusSmall"), Some(6.0));
        assert_eq!(effect.float("radiusMedium"), Some(1.0));
        assert_eq!(effect.float("radiusLarge"), Some(4.0));
        assert_eq!(effect.color("tint"), Some([219, 38, 38, 255]));
    }

    #[test]
    fn runtime_hsla_overlay_effect_sets_hsla_fields() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_hsla_overlay" kind="compute" kernel="color_core.wgsl" effect="hsla_overlay"
        in={["under"]}
        out={["out"]}
        params={{
          hue: "210.0",
          saturation: "0.70",
          lightness: "0.41",
          alpha: "0.45"
        }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        assert_eq!(out.layer_hsla_hue, Some(210.0));
        assert_eq!(out.layer_hsla_saturation, Some(0.70));
        assert_eq!(out.layer_hsla_lightness, Some(0.41));
        assert_eq!(out.layer_hsla_alpha, Some(0.45));
    }

    #[test]
    fn runtime_blur_sigma_curve_uses_seconds_domain_and_holds_tail() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_blur" kind="compute" effect="gaussian_5tap_blur"
        in={["under"]} out={["out"]}
        params={{ sigma: "curve(\"0.00:2.0:linear, 1.00:10.0:ease_in_out\")" }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");

        let v0 = runtime
            .evaluate_at_time_sec(0.0, Some(8.0))
            .layer_blur_sigma
            .unwrap_or(-1.0);
        let v050 = runtime
            .evaluate_at_time_sec(0.5, Some(8.0))
            .layer_blur_sigma
            .unwrap_or(-1.0);
        let v100 = runtime
            .evaluate_at_time_sec(1.0, Some(8.0))
            .layer_blur_sigma
            .unwrap_or(-1.0);
        let v500 = runtime
            .evaluate_at_time_sec(5.0, Some(8.0))
            .layer_blur_sigma
            .unwrap_or(-1.0);

        assert!((v0 - 2.0).abs() < 1e-4, "unexpected t=0.0 sigma: {v0}");
        assert!(v050 > 2.0 && v050 < 10.0, "unexpected t=0.5 sigma: {v050}");
        assert!((v100 - 10.0).abs() < 1e-4, "unexpected t=1.0 sigma: {v100}");
        assert!((v500 - 10.0).abs() < 1e-4, "unexpected t=5.0 sigma: {v500}");
    }

    #[test]
    fn runtime_rejects_missing_kernel_when_effect_not_mapped() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="fx_unknown" kind="compute" effect="unknown_effect"
        in={["src"]}
        out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let err = compile_runtime_program(graph).expect_err("compile should fail");
        assert!(
            err.message
                .contains("missing kernel and no default mapping for effect"),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn runtime_accepts_explicit_custom_wgsl_kernel() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="fx_custom" kind="compute" kernel="my_custom_shader.wgsl" effect="my_effect"
        in={["src"]}
        out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        assert!(runtime.unsupported_kernels().is_empty());
        assert_eq!(runtime.supported_pass_count(), 1);
    }

    #[test]
    fn runtime_texture_overlay_effect_uses_default_color_core_kernel() {
        let script = r#"
<Graph fps={30} duration="4s" size={[1920,1080]}>
  <Process id="paper_post">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="post_paper_texture" kind="compute" effect="texture_overlay"
          in={["src"]} out={["out"]}
          params={{ kind: "paper", scale: "86.0", strength: "0.24", contrast: "0.58", seed: "101.0" }} />
  </Process>
  <Present from="paper_post" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        assert!(runtime.unsupported_kernels().is_empty());
        let out = runtime.evaluate_frame(0);
        let texture_overlay = out
            .process_effects
            .iter()
            .find(|effect| effect.effect_id == "texture_overlay")
            .expect("texture_overlay runtime effect");
        assert_eq!(
            texture_overlay.params.get("kind"),
            Some(&RuntimeProcessParamValue::Float(1.0))
        );
        assert_eq!(
            texture_overlay.params.get("scale"),
            Some(&RuntimeProcessParamValue::Float(86.0))
        );
        assert_eq!(
            texture_overlay.params.get("strength"),
            Some(&RuntimeProcessParamValue::Float(0.24))
        );
        assert_eq!(
            texture_overlay.params.get("contrast"),
            Some(&RuntimeProcessParamValue::Float(0.58))
        );
        assert_eq!(
            texture_overlay.params.get("seed"),
            Some(&RuntimeProcessParamValue::Float(101.0))
        );
    }

    #[test]
    fn runtime_brightness_one_emits_full_positive_amount() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Process id="brightness_post">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx_brightness" kind="compute" effect="brightness"
          in={["src"]} out={["out"]}
          params={{ brightness: "1.0" }} />
  </Process>
  <Present from="brightness_post" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        let brightness = out
            .process_effects
            .iter()
            .find(|effect| effect.effect_id == "brightness")
            .expect("brightness runtime effect");
        let Some(RuntimeProcessParamValue::Float(amount)) = brightness.params.get("amount") else {
            panic!("brightness runtime effect did not expose amount");
        };
        assert!((*amount - 1.0).abs() < 1e-5, "unexpected amount: {amount}");
    }

    #[test]
    fn runtime_brightness_small_numeric_value_is_additive_amount() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Process id="brightness_post">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx_brightness" kind="compute" effect="brightness"
          in={["src"]} out={["out"]}
          params={{ brightness: "0.3" }} />
  </Process>
  <Present from="brightness_post" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        let brightness = out
            .process_effects
            .iter()
            .find(|effect| effect.effect_id == "brightness")
            .expect("brightness runtime effect");
        let Some(RuntimeProcessParamValue::Float(amount)) = brightness.params.get("amount") else {
            panic!("brightness runtime effect did not expose amount");
        };
        assert!((*amount - 0.3).abs() < 1e-5, "unexpected amount: {amount}");
    }

    #[test]
    fn runtime_brightness_negative_numeric_value_is_additive_amount() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Process id="brightness_post">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx_brightness" kind="compute" effect="brightness"
          in={["src"]} out={["out"]}
          params={{ brightness: "-0.1" }} />
  </Process>
  <Present from="brightness_post" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("graph parse");
        let runtime = compile_runtime_program(graph).expect("runtime compile");
        let out = runtime.evaluate_frame(0);
        let brightness = out
            .process_effects
            .iter()
            .find(|effect| effect.effect_id == "brightness")
            .expect("brightness runtime effect");
        let Some(RuntimeProcessParamValue::Float(amount)) = brightness.params.get("amount") else {
            panic!("brightness runtime effect did not expose amount");
        };
        assert!((*amount + 0.1).abs() < 1e-5, "unexpected amount: {amount}");
    }
}
