// =========================================
// =========================================
// crates/motionloom/src/render_style/validation.rs

//! Render-style value validation shared by parsing and graph lowering.

use super::resolve::resolve_scene_render_style;
use crate::dsl::{GraphParseError, GraphScript};

pub(super) fn error(message: impl Into<String>) -> GraphParseError {
    GraphParseError {
        line: 1,
        message: message.into(),
    }
}

pub(super) fn one_of(
    value: Option<&str>,
    allowed: &[&str],
    property: &str,
) -> Result<(), GraphParseError> {
    if let Some(value) = value {
        if !allowed.contains(&value) {
            return Err(error(format!(
                "Invalid {property}={value}; expected {}",
                allowed.join(" | ")
            )));
        }
    }
    Ok(())
}
pub(super) fn range(
    value: Option<f32>,
    min: f32,
    max: f32,
    property: &str,
) -> Result<(), GraphParseError> {
    if value.is_some_and(|v| !v.is_finite() || v < min || v > max) {
        return Err(error(format!(
            "{property} must be finite in [{min}, {max}]"
        )));
    }
    Ok(())
}
pub(super) fn color(value: &str) -> Result<[f32; 3], GraphParseError> {
    let hex = value
        .strip_prefix('#')
        .filter(|v| v.len() == 6)
        .ok_or_else(|| error("Style color must be #RRGGBB"))?;
    let n = u32::from_str_radix(hex, 16).map_err(|_| error("Invalid style color"))?;
    Ok([
        ((n >> 16) & 255) as f32 / 255.0,
        ((n >> 8) & 255) as f32 / 255.0,
        (n & 255) as f32 / 255.0,
    ])
}
/// Validate every declaration, including unused styles, before GPU work starts.
pub(crate) fn validate(graph: &GraphScript) -> Result<(), GraphParseError> {
    let mut ids = std::collections::HashSet::new();
    for s in &graph.render_styles {
        if let Some(dof) = &s.depth_of_field {
            one_of(
                Some(&dof.preset),
                &["cinematic_bokeh_v1", "filmic_bokeh_v1"],
                "DepthOfFieldStyle.preset",
            )?;
            range(dof.aperture, 0.0, 0.1, "DepthOfFieldStyle.aperture")?;
            range(dof.max_blur, 0.0, 0.1, "DepthOfFieldStyle.maxBlur")?;
            if dof.preset != "filmic_bokeh_v1" && (dof.aperture.is_some() || dof.max_blur.is_some())
            {
                return Err(error(
                    "aperture/maxBlur on DepthOfFieldStyle require filmic_bokeh_v1",
                ));
            }
            one_of(
                dof.quality.as_deref(),
                &["preview", "balanced", "high"],
                "DepthOfFieldStyle.quality",
            )?;
        }
        if !ids.insert(&s.id) {
            return Err(error(format!("Duplicate RenderStyle id {}", s.id)));
        }
        if let Some(s) = &s.surface {
            one_of(
                s.shading.as_deref(),
                &[
                    "physical",
                    "filmic_physical_v1",
                    "stylized",
                    "toon",
                    "clay",
                    "cel",
                    "ink_wash_soft_v1",
                ],
                "shading",
            )?;
            one_of(s.outline.as_deref(), &["none"], "outline (V1)")?;
            range(s.shading_steps.map(|v| v as f32), 2.0, 16.0, "shadingSteps")?;
            for (v, min, max, name) in [
                (s.diffuse_wrap, 0.0, 1.0, "diffuseWrap"),
                (s.rim_light, 0.0, 4.0, "rimLight"),
                (s.rim_power, 0.1, 32.0, "rimPower"),
                (s.specular, 0.0, 4.0, "specular"),
                (s.roughness_bias, -1.0, 1.0, "roughnessBias"),
                (s.saturation, 0.0, 3.0, "saturation"),
            ] {
                range(v, min, max, name)?;
            }
        }
        if let Some(surface) = &s.surface {
            range(surface.shadow_threshold, 0.0, 1.0, "shadowThreshold")?;
            range(surface.shadow_feather, 0.001, 0.5, "shadowFeather")?;
            if let Some(c) = &surface.shadow_color {
                color(c)?;
            }
        }
        if let Some(o) = &s.outline {
            one_of(o.method.as_deref(), &["geometry"], "outline method")?;
            one_of(
                o.distance_mode.as_deref(),
                &["screen"],
                "outline distanceMode",
            )?;
            range(o.width, 0.0, 12.0, "outline width")?;
            if let Some(c) = &o.color {
                color(c)?;
            }
            if o.enabled == Some(true)
                && s.surface.as_ref().and_then(|v| v.outline.as_deref()) == Some("none")
            {
                return Err(error(
                    "OutlineStyle enabled conflicts with SurfaceStyle outline=none",
                ));
            }
        }
        if let Some(l) = &s.lighting {
            one_of(
                l.preset.as_deref(),
                &["neutral", "soft_sunlight", "cinematic", "overcast", "night"],
                "lighting preset",
            )?;
            one_of(l.shadow_style.as_deref(), &["hard", "soft"], "shadowStyle")?;
            range(l.ambient_intensity, 0.0, 10.0, "ambientIntensity")?;
            if let Some(v) = &l.ambient_color {
                color(v)?;
            }
        }
        if let Some(c) = &s.color {
            if let Some(tint) = &c.tint {
                color(tint)?;
            }
            range(c.tint_strength, 0.0, 1.0, "tintStrength")?;
            range(c.saturation, 0.0, 3.0, "ColorStyle saturation")?;
        }
        if let Some(t) = &s.tone {
            for c in [&t.shadow_color, &t.highlight_color].into_iter().flatten() {
                color(c)?;
            }
            range(t.exposure, 0.0, 32.0, "ToneStyle exposure")?;
            range(t.contrast, 0.0, 3.0, "ToneStyle contrast")?;
            range(t.tone_strength, 0.0, 1.0, "toneStrength")?;
        }
        if let Some(p) = &s.post {
            one_of(
                p.tone_mapping.as_deref(),
                &["none", "aces", "filmic_aces_v1", "reinhard"],
                "toneMapping",
            )?;
            for (v, min, max, name) in [
                (p.exposure, 0.0, 32.0, "exposure"),
                (p.saturation, 0.0, 3.0, "saturation"),
                (p.contrast, 0.0, 3.0, "contrast"),
                (p.white_balance, 1000.0, 40000.0, "whiteBalance"),
                (p.bloom_threshold, 0.0, 32.0, "bloomThreshold"),
                (p.bloom_intensity, 0.0, 4.0, "bloomIntensity"),
            ] {
                range(v, min, max, name)?;
            }
        }
    }
    for s in &graph.scenes {
        resolve_scene_render_style(graph, &s.id)?;
    }
    // Discrete Scene style keys must resolve before playback starts.
    for target in graph
        .animation_targets
        .iter()
        .filter(|target| target.property == "renderStyle")
    {
        if !graph.scenes.iter().any(|scene| scene.id == target.node) {
            return Err(error(format!(
                "AnimationTarget renderStyle references unknown Scene {}",
                target.node
            )));
        }
        for key in &target.keys {
            if !graph
                .render_styles
                .iter()
                .any(|style| style.id == key.value)
            {
                return Err(error(format!(
                    "AnimationTarget renderStyle references unknown RenderStyle {}",
                    key.value
                )));
            }
        }
    }
    Ok(())
}

/// Convert a validated authored color into linear runtime components.
pub(crate) fn cel_color(value: &str) -> [f32; 3] {
    color(value).unwrap_or([0.4, 0.37, 0.5])
}
