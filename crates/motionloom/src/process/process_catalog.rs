#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProcessCategory {
    TransformCamera,
    ColorTone,
    StylizeLook,
    BlurSharpenDetail,
    KeyingMatteMask,
    LightAtmosphere,
    DistortionWarp,
    Composite,
    Transition,
    Testing,
}

impl ProcessCategory {
    pub const fn label(self) -> &'static str {
        match self {
            Self::TransformCamera => "Transform & Camera",
            Self::ColorTone => "Color & Tone",
            Self::StylizeLook => "Stylize & Look",
            Self::BlurSharpenDetail => "Blur, Sharpen & Detail",
            Self::KeyingMatteMask => "Keying, Matte & Mask",
            Self::LightAtmosphere => "Light & Atmosphere",
            Self::DistortionWarp => "Distortion & Warp",
            Self::Composite => "Composite",
            Self::Transition => "Transition",
            Self::Testing => "Testing",
        }
    }

    pub const fn folder(self) -> &'static str {
        match self {
            Self::TransformCamera => "transform_camera",
            Self::ColorTone => "color_tone",
            Self::StylizeLook => "stylize_look",
            Self::BlurSharpenDetail => "blur_sharpen_detail",
            Self::KeyingMatteMask => "keying_matte_mask",
            Self::LightAtmosphere => "light_atmosphere",
            Self::DistortionWarp => "distortion_warp",
            Self::Composite => "composite",
            Self::Transition => "transition",
            Self::Testing => "testing",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessEffectDefinition {
    pub id: &'static str,
    pub display_name: &'static str,
    pub category: ProcessCategory,
    pub kernel: &'static str,
    pub summary: &'static str,
}

pub const PROCESS_CATEGORIES: [ProcessCategory; 10] = [
    ProcessCategory::TransformCamera,
    ProcessCategory::ColorTone,
    ProcessCategory::StylizeLook,
    ProcessCategory::BlurSharpenDetail,
    ProcessCategory::KeyingMatteMask,
    ProcessCategory::LightAtmosphere,
    ProcessCategory::DistortionWarp,
    ProcessCategory::Composite,
    ProcessCategory::Transition,
    ProcessCategory::Testing,
];

pub const PROCESS_EFFECTS: [ProcessEffectDefinition; 17] = [
    ProcessEffectDefinition {
        id: "procedural_surface",
        display_name: "Procedural Surface",
        category: ProcessCategory::StylizeLook,
        kernel: "procedural_surface.wgsl",
        summary: "GPU-only evolving relief with metallic veins and pools; preserves input alpha.",
    },
    ProcessEffectDefinition {
        id: "transform_camera.affine_uv",
        display_name: "Affine UV Transform",
        category: ProcessCategory::TransformCamera,
        kernel: "transform_camera_affine.wgsl",
        summary: "Scale/rotate/translate UV for camera-style pan+zoom motion.",
    },
    ProcessEffectDefinition {
        id: "color_tone.exposure_contrast",
        display_name: "Exposure + Contrast",
        category: ProcessCategory::ColorTone,
        kernel: "color_core.wgsl",
        summary: "Unified color core: exposure/contrast/sat/curves/LUT + look and utility ops.",
    },
    ProcessEffectDefinition {
        id: "stylize_look.posterize",
        display_name: "Posterize",
        category: ProcessCategory::StylizeLook,
        kernel: "stylize_look_posterize.wgsl",
        summary: "Quantize RGB into a fixed number of tonal levels.",
    },
    ProcessEffectDefinition {
        id: "blur_sharpen_detail.gaussian",
        display_name: "Blur/Sharpen Operator",
        category: ProcessCategory::BlurSharpenDetail,
        kernel: "blur_sharpen_detail_gaussian.wgsl",
        summary: "Mode-driven kernel: gaussian_5tap_blur | gaussian_5tap_h | gaussian_5tap_v | box | unsharp.",
    },
    ProcessEffectDefinition {
        id: "keying_matte_mask.luma_key",
        display_name: "Luma Key Matte",
        category: ProcessCategory::KeyingMatteMask,
        kernel: "keying_matte_mask_luma_key.wgsl",
        summary: "Build matte alpha from image luminance with soft edge.",
    },
    ProcessEffectDefinition {
        id: "light_atmosphere.bloom_prefilter",
        display_name: "Bloom Prefilter",
        category: ProcessCategory::LightAtmosphere,
        kernel: "light_atmosphere_bloom_prefilter.wgsl",
        summary: "Extract bright-energy signal before blur and merge.",
    },
    ProcessEffectDefinition {
        id: "light_atmosphere.bloom",
        display_name: "Bloom / Glow Composite",
        category: ProcessCategory::LightAtmosphere,
        kernel: "color_core.wgsl",
        summary: "Full-scene post-process glow: bright-pass, blur, then additive composite.",
    },
    ProcessEffectDefinition {
        id: "light_atmosphere.glow_stack",
        display_name: "Glow Stack",
        category: ProcessCategory::LightAtmosphere,
        kernel: "color_core.wgsl",
        summary: "Cinematic multi-radius glow stack with threshold, tint, and intensity controls.",
    },
    ProcessEffectDefinition {
        id: "color_tone.tone_map",
        display_name: "Tone Map",
        category: ProcessCategory::ColorTone,
        kernel: "color_core.wgsl",
        summary: "Filmic tone mapping with exposure, contrast, shoulder, gamma, and saturation.",
    },
    ProcessEffectDefinition {
        id: "light_atmosphere.light_sweep",
        display_name: "Light Sweep",
        category: ProcessCategory::LightAtmosphere,
        kernel: "color_core.wgsl",
        summary: "Animated directional sweep highlight for text, logos, and energy reveals.",
    },
    ProcessEffectDefinition {
        id: "stylize_look.texture_overlay",
        display_name: "Texture Overlay",
        category: ProcessCategory::StylizeLook,
        kernel: "color_core.wgsl",
        summary: "Procedural paper, canvas, impasto, brushed paint, grain, noise, and scanline texture overlay for tactile surfaces.",
    },
    ProcessEffectDefinition {
        id: "distortion_warp.magnify_lens",
        display_name: "Magnify Lens",
        category: ProcessCategory::DistortionWarp,
        kernel: "color_core.wgsl",
        summary: "Circular pixel-space lens magnification with distortion, feathering, and glass highlight.",
    },
    ProcessEffectDefinition {
        id: "distortion_warp.heat_haze_uv",
        display_name: "Heat Haze UV Warp",
        category: ProcessCategory::DistortionWarp,
        kernel: "distortion_warp_heat_haze.wgsl",
        summary: "Noise-driven UV offset for shimmer/heat refraction looks.",
    },
    ProcessEffectDefinition {
        id: "composite.opacity",
        display_name: "Opacity Composite",
        category: ProcessCategory::Composite,
        kernel: "composite_core.wgsl",
        summary: "Composite operator for layer opacity and related blend staging.",
    },
    ProcessEffectDefinition {
        id: "transition.fade_in",
        display_name: "Transition Core",
        category: ProcessCategory::Transition,
        kernel: "transition_core.wgsl",
        summary: "Transition family entry (fade_in/fade_out/dip/dissolve via effect).",
    },
    ProcessEffectDefinition {
        id: "testing.effect_for_testing_run",
        display_name: "Testing Kernel",
        category: ProcessCategory::Testing,
        kernel: "effect_for_testing_run.wgsl",
        summary: "Testing-only kernel slot for quick validation runs.",
    },
];

pub fn process_effects() -> &'static [ProcessEffectDefinition] {
    &PROCESS_EFFECTS
}

pub fn process_effect_for_id(id: &str) -> Option<&'static ProcessEffectDefinition> {
    PROCESS_EFFECTS.iter().find(|fx| fx.id == id)
}

pub fn process_effects_for_category(
    category: ProcessCategory,
) -> impl Iterator<Item = &'static ProcessEffectDefinition> {
    PROCESS_EFFECTS
        .iter()
        .filter(move |fx| fx.category == category)
}

pub fn kernel_source_by_name(kernel: &str) -> Option<&'static str> {
    match kernel {
        "procedural_surface.wgsl" => {
            Some(include_str!("kernels/stylize_look/procedural_surface.wgsl"))
        }
        "transform_camera_affine.wgsl" => Some(include_str!(
            "kernels/transform_camera/transform_camera_affine.wgsl"
        )),
        "color_core.wgsl" => Some(include_str!("kernels/color_tone/color_core.wgsl")),
        "color_tone_exposure_contrast.wgsl" => Some(include_str!(
            "kernels/color_tone/color_tone_exposure_contrast.wgsl"
        )),
        "stylize_look_posterize.wgsl" => Some(include_str!(
            "kernels/stylize_look/stylize_look_posterize.wgsl"
        )),
        "blur_sharpen_detail_gaussian.wgsl" => Some(include_str!(
            "kernels/blur_sharpen_detail/blur_sharpen_detail_gaussian.wgsl"
        )),
        "keying_matte_mask_luma_key.wgsl" => Some(include_str!(
            "kernels/keying_matte_mask/keying_matte_mask_luma_key.wgsl"
        )),
        "light_atmosphere_bloom_prefilter.wgsl" => Some(include_str!(
            "kernels/light_atmosphere/light_atmosphere_bloom_prefilter.wgsl"
        )),
        "distortion_warp_heat_haze.wgsl" => Some(include_str!(
            "kernels/distortion_warp/distortion_warp_heat_haze.wgsl"
        )),
        "composite_core.wgsl" => Some(include_str!("kernels/composite/composite_core.wgsl")),
        "effect_for_testing_run.wgsl" => {
            Some(include_str!("kernels/testing/effect_for_testing_run.wgsl"))
        }
        "transition_core.wgsl" => Some(include_str!("kernels/transition/transition_core.wgsl")),
        _ => None,
    }
}

pub fn is_known_process_kernel(kernel: &str) -> bool {
    kernel_source_by_name(kernel).is_some()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        PROCESS_CATEGORIES, PROCESS_EFFECTS, is_known_process_kernel, kernel_source_by_name,
        process_effect_for_id,
    };

    #[test]
    fn every_process_category_has_one_seed_effect() {
        let covered: HashSet<_> = PROCESS_EFFECTS.iter().map(|fx| fx.category).collect();
        for category in PROCESS_CATEGORIES {
            assert!(
                covered.contains(&category),
                "missing seed effect: {category:?}"
            );
        }
    }

    #[test]
    fn every_seed_kernel_has_embedded_source() {
        for fx in PROCESS_EFFECTS {
            let src = kernel_source_by_name(fx.kernel).expect("kernel source");
            assert!(
                src.contains("fn ml_"),
                "expected helper function signature in {}",
                fx.kernel
            );
            assert!(is_known_process_kernel(fx.kernel));
            assert_eq!(
                process_effect_for_id(fx.id).map(|it| it.kernel),
                Some(fx.kernel)
            );
        }
    }
}
