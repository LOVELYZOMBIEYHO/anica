// =========================================
// =========================================
// crates/motionloom/src/render_style.rs

//! Scene-owned visual styles. Authored values stay separate from resolved GPU
//! defaults; no style is an exact opt-out, including for legacy SVG scenes.

use crate::dsl::{GraphParseError, GraphScript, attr_value, strip_wrappers};
use crate::scene::model::{Scene3DNode, SceneNode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RenderStyleNode {
    pub outline: Option<OutlineStyleNode>,
    pub id: String,
    pub surface: Option<SurfaceStyleNode>,
    pub lighting: Option<LightingStyleNode>,
    pub post: Option<PostStyleNode>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SurfaceStyleNode {
    pub shadow_threshold: Option<f32>,
    pub shadow_feather: Option<f32>,
    pub shadow_color: Option<String>,
    pub shading: Option<String>,
    pub shading_steps: Option<u32>,
    pub diffuse_wrap: Option<f32>,
    pub rim_light: Option<f32>,
    pub rim_power: Option<f32>,
    pub specular: Option<f32>,
    pub roughness_bias: Option<f32>,
    pub saturation: Option<f32>,
    pub outline: Option<String>,
}

/// Screen-width geometry outlines are independent of the base material.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OutlineStyleNode {
    pub enabled: Option<bool>,
    pub method: Option<String>,
    pub color: Option<String>,
    pub width: Option<f32>,
    pub distance_mode: Option<String>,
}

/// Optional material-slot controls travel with the actor, never rewrite a GLB.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CelMaterialSettings {
    pub control_map: Option<String>,
    pub role: Option<String>,
    pub outline_width: Option<f32>,
    pub shadow_color: Option<String>,
    pub hair_highlight: Option<f32>,
}

pub(crate) fn parse_cel_material(
    tag: &str,
    line: usize,
) -> Result<CelMaterialSettings, GraphParseError> {
    let scalar = |key: &str| -> Result<Option<f32>, GraphParseError> {
        attr_value(tag, key)
            .map(|v| {
                strip_wrappers(&v)
                    .parse::<f32>()
                    .map_err(|_| GraphParseError {
                        line,
                        message: format!("{key} must be a number"),
                    })
            })
            .transpose()
    };
    let settings = CelMaterialSettings {
        control_map: attr_value(tag, "celControlMap").map(|v| strip_wrappers(&v).to_owned()),
        role: attr_value(tag, "celRole").map(|v| strip_wrappers(&v).to_owned()),
        shadow_color: attr_value(tag, "celShadowColor").map(|v| strip_wrappers(&v).to_owned()),
        outline_width: scalar("outlineWidth")?,
        hair_highlight: scalar("hairHighlight")?,
    };
    one_of(
        settings.role.as_deref(),
        &["body", "skin", "hair", "face"],
        "celRole",
    )?;
    range(settings.outline_width, 0.0, 12.0, "outlineWidth")?;
    range(settings.hair_highlight, 0.0, 2.0, "hairHighlight")?;
    if let Some(c) = &settings.shadow_color {
        color(c)?;
    }
    if settings.role.as_deref() == Some("face") && settings.control_map.is_none() {
        return Err(GraphParseError {
            line,
            message: "celRole=face requires celControlMap (ImageAsset id)".into(),
        });
    }
    Ok(settings)
}

pub(crate) fn cel_color(value: &str) -> [f32; 3] {
    // Authored colors have already passed validation.
    color(value).unwrap_or([0.4, 0.37, 0.5])
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LightingStyleNode {
    pub preset: Option<String>,
    pub ambient_intensity: Option<f32>,
    pub ambient_color: Option<String>,
    pub shadow_style: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PostStyleNode {
    pub tone_mapping: Option<String>,
    pub exposure: Option<f32>,
    pub saturation: Option<f32>,
    pub contrast: Option<f32>,
    pub white_balance: Option<f32>,
    pub bloom_threshold: Option<f32>,
    pub bloom_intensity: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedCelStyle {
    pub shadow_threshold: f32,
    pub shadow_feather: f32,
    pub shadow_color: [f32; 3],
    pub outline_width: f32,
    pub outline_color: [f32; 3],
}
impl Default for ResolvedCelStyle {
    fn default() -> Self {
        Self {
            shadow_threshold: 0.5,
            shadow_feather: 0.025,
            shadow_color: [0.4, 0.37, 0.5],
            outline_width: 0.0,
            outline_color: [0.0; 3],
        }
    }
}

/// A serializable report, not an alternative authoring source of truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedSceneRenderStyle {
    #[serde(default)]
    pub cel: ResolvedCelStyle,
    pub scene_id: String,
    pub style_id: Option<String>,
    pub shading: String,
    pub shading_steps: u32,
    pub diffuse_wrap: f32,
    pub rim_light: f32,
    pub rim_power: f32,
    pub specular: f32,
    pub roughness_bias: f32,
    pub surface_saturation: f32,
    pub ambient_intensity: f32,
    pub ambient_color: [f32; 3],
    pub hard_shadows: bool,
    pub lighting_preset: Option<String>,
    pub post: PostStyleNode,
    /// Explicit nodes own their complete setting group, including defaults.
    pub overrides: Vec<RenderStyleOverride>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderStyleOverride {
    pub island_id: Option<String>,
    pub property: String,
    pub style_value: serde_json::Value,
    /// Retains expressions; this is compile-time evidence, not a sampled frame.
    pub final_expression: String,
    pub source: String,
}

fn error(message: impl Into<String>) -> GraphParseError {
    GraphParseError {
        line: 1,
        message: message.into(),
    }
}

// Reuse the DSL attribute lexer, then deserialize into strict typed children.
fn attributes<T: serde::de::DeserializeOwned>(
    tag: &str,
    line: usize,
) -> Result<T, GraphParseError> {
    let mut values = serde_json::Map::new();
    for key in crate::dsl::tag_attribute_names(tag) {
        if values.contains_key(&key) {
            return Err(error(format!("Duplicate style attribute {key}")));
        }
        let raw = attr_value(tag, &key).unwrap_or_default();
        let raw = strip_wrappers(&raw);
        let value = raw
            .parse::<serde_json::Number>()
            .map(serde_json::Value::Number)
            .unwrap_or_else(|_| match raw {
                "true" if key == "enabled" => serde_json::Value::Bool(true),
                "false" if key == "enabled" => serde_json::Value::Bool(false),
                _ => serde_json::Value::String(raw.to_string()),
            });
        values.insert(key, value);
    }
    serde_json::from_value(serde_json::Value::Object(values)).map_err(|e| GraphParseError {
        line,
        message: format!("Render style: {e}"),
    })
}

pub(crate) fn parse_resource(
    lines: &[&str],
    start: usize,
) -> Result<(serde_json::Value, usize), GraphParseError> {
    let name = "RenderStyle";
    let (open, end) = crate::dsl::collect_tag_block(lines, start, '>', false)?;
    let mut root: serde_json::Value = attributes(&open, start + 1)?;
    let id = root.get("id").and_then(|v| v.as_str()).unwrap_or("");
    if id.is_empty() || id.starts_with("__ml_style_") {
        return Err(error(
            "Style id must be nonempty and not use reserved __ml_style_ prefix",
        ));
    }
    let mut i = end + 1;
    while i < lines.len() {
        let text = lines[i].trim();
        if text == format!("</{name}>") {
            return Ok((root, i));
        }
        if text.is_empty() || text.starts_with("//") || text.starts_with("<!--") {
            i += 1;
            continue;
        }
        let (tag, last) = crate::dsl::collect_tag_block(lines, i, '>', false)?;
        if !tag.trim_end().ends_with("/>") {
            return Err(error("Style children must be self-closing"));
        }
        let child = tag
            .trim_start_matches('<')
            .split_whitespace()
            .next()
            .unwrap_or("");
        let field = match child {
            "SurfaceStyle" => "surface",
            "OutlineStyle" => "outline",
            "LightingStyle" => "lighting",
            "PostStyle" => "post",
            _ => return Err(error(format!("Unsupported {name} child {child}"))),
        };
        if root.get(field).is_some() {
            return Err(error(format!("Duplicate {child}")));
        }
        root[field] = attributes(&tag, i + 1)?;
        i = last + 1;
    }
    Err(error(format!("Missing </{name}>")))
}

fn one_of(value: Option<&str>, allowed: &[&str], property: &str) -> Result<(), GraphParseError> {
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
fn range(value: Option<f32>, min: f32, max: f32, property: &str) -> Result<(), GraphParseError> {
    if value.is_some_and(|v| !v.is_finite() || v < min || v > max) {
        return Err(error(format!(
            "{property} must be finite in [{min}, {max}]"
        )));
    }
    Ok(())
}
fn color(value: &str) -> Result<[f32; 3], GraphParseError> {
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

/// Resolve one Scene without changing the authored graph.
pub fn resolve_scene_render_style(
    graph: &GraphScript,
    scene_id: &str,
) -> Result<ResolvedSceneRenderStyle, GraphParseError> {
    let scene = graph
        .scenes
        .iter()
        .find(|s| s.id == scene_id)
        .ok_or_else(|| error(format!("Unknown Scene {scene_id}")))?;
    let style = scene
        .render_style
        .as_ref()
        .map(|id| {
            graph
                .render_styles
                .iter()
                .find(|s| &s.id == id)
                .ok_or_else(|| error(format!("Scene {scene_id}: unknown RenderStyle {id}")))
        })
        .transpose()?;
    let empty_surface = SurfaceStyleNode::default();
    let s = style
        .and_then(|s| s.surface.as_ref())
        .unwrap_or(&empty_surface);
    let empty_lighting = LightingStyleNode::default();
    let l = style
        .and_then(|s| s.lighting.as_ref())
        .unwrap_or(&empty_lighting);
    let shading = s.shading.as_deref().unwrap_or("physical");
    let stylized = matches!(shading, "stylized" | "toon" | "cel");
    let outline = style.and_then(|style| style.outline.as_ref());
    let outline_enabled = outline
        .and_then(|o| o.enabled)
        .unwrap_or(shading == "cel" && s.outline.as_deref() != Some("none"));
    let cel = ResolvedCelStyle {
        shadow_threshold: s.shadow_threshold.unwrap_or(0.5),
        shadow_feather: s.shadow_feather.unwrap_or(0.025),
        shadow_color: s
            .shadow_color
            .as_deref()
            .map(color)
            .transpose()?
            .unwrap_or([0.4, 0.37, 0.5]),
        outline_width: if outline_enabled {
            outline.and_then(|o| o.width).unwrap_or(1.5)
        } else {
            0.0
        },
        outline_color: outline
            .and_then(|o| o.color.as_deref())
            .map(color)
            .transpose()?
            .unwrap_or([0.0; 3]),
    };
    let mut r = ResolvedSceneRenderStyle {
        cel,
        scene_id: scene_id.into(),
        style_id: scene.render_style.clone(),
        shading: shading.into(),
        shading_steps: s.shading_steps.unwrap_or(3),
        diffuse_wrap: s.diffuse_wrap.unwrap_or(if stylized { 0.15 } else { 0.0 }),
        rim_light: s.rim_light.unwrap_or(0.0),
        rim_power: s.rim_power.unwrap_or(3.0),
        specular: s.specular.unwrap_or(if stylized { 0.1 } else { 1.0 }),
        roughness_bias: s.roughness_bias.unwrap_or(0.0),
        surface_saturation: s.saturation.unwrap_or(1.0),
        ambient_intensity: l.ambient_intensity.unwrap_or(1.0),
        ambient_color: l
            .ambient_color
            .as_deref()
            .map(color)
            .transpose()?
            .unwrap_or([1.0; 3]),
        hard_shadows: l.shadow_style.as_deref() == Some("hard"),
        lighting_preset: l.preset.clone(),
        post: style.and_then(|s| s.post.clone()).unwrap_or_default(),
        overrides: vec![],
    };
    // Reports expose the same concrete defaults used by the runtime.
    let defaults = crate::world::WorldColorManagement::default();
    r.post.tone_mapping.get_or_insert(defaults.tone_mapping);
    r.post.exposure.get_or_insert(defaults.exposure);
    r.post.contrast.get_or_insert(defaults.contrast);
    r.post
        .white_balance
        .get_or_insert(defaults.white_balance_kelvin);
    r.post.saturation.get_or_insert(1.0);
    r.post.bloom_threshold.get_or_insert(0.9);
    r.post.bloom_intensity.get_or_insert(0.0);
    collect_overrides(&scene.children, &mut r);
    Ok(r)
}

fn collect_overrides(nodes: &[SceneNode], r: &mut ResolvedSceneRenderStyle) {
    for node in nodes {
        let children = match node {
            SceneNode::Group(g) => {
                if let Some(c) = &g.composite {
                    let explicit_light = c.nodes_3d.iter().any(|n| {
                        matches!(
                            n,
                            Scene3DNode::DirectionalLight(_)
                                | Scene3DNode::PointLight(_)
                                | Scene3DNode::SpotLight(_)
                                | Scene3DNode::RectAreaLight(_)
                                | Scene3DNode::EnvironmentLight(_)
                        )
                    });
                    if explicit_light && r.lighting_preset.is_some() {
                        r.overrides.push(RenderStyleOverride {
                            island_id: g.id.clone(),
                            property: "lighting.preset".into(),
                            style_value: serde_json::json!(r.lighting_preset),
                            final_expression: "explicit lights/environment".into(),
                            source: "Scene 3D light nodes".into(),
                        });
                    }
                    for n in &c.nodes_3d {
                        if let Scene3DNode::Model(model) = n {
                            for binding in &model.material_bindings {
                                if binding.cel != CelMaterialSettings::default() {
                                    r.overrides.push(RenderStyleOverride {
                                        island_id: g.id.clone(),
                                        property: "cel.material".into(),
                                        style_value: serde_json::json!(r.cel),
                                        final_expression: serde_json::json!(binding.cel)
                                            .to_string(),
                                        source: format!(
                                            "MaterialBinding:{}:{}",
                                            model.id.as_deref().unwrap_or("anonymous"),
                                            binding.material
                                        ),
                                    });
                                }
                            }
                        }
                        if let Scene3DNode::ColorManagement(v) = n {
                            for (name, value, final_expression) in [
                                (
                                    "toneMapping",
                                    serde_json::json!(r.post.tone_mapping),
                                    v.tone_mapping.clone(),
                                ),
                                (
                                    "exposure",
                                    serde_json::json!(r.post.exposure),
                                    v.exposure.clone(),
                                ),
                                (
                                    "contrast",
                                    serde_json::json!(r.post.contrast),
                                    v.contrast.clone(),
                                ),
                                (
                                    "whiteBalance",
                                    serde_json::json!(r.post.white_balance),
                                    v.white_balance.clone(),
                                ),
                            ] {
                                r.overrides.push(RenderStyleOverride {
                                    island_id: g.id.clone(),
                                    property: format!("post.{name}"),
                                    style_value: value,
                                    final_expression,
                                    source: format!(
                                        "ColorManagement:{}",
                                        v.id.as_deref().unwrap_or("anonymous")
                                    ),
                                });
                            }
                        }
                    }
                }
                &g.children
            }
            SceneNode::Timeline(n) => &n.children,
            SceneNode::Track(n) => &n.children,
            SceneNode::Sequence(n) => &n.children,
            SceneNode::Chain(n) => &n.children,
            SceneNode::Part(n) => &n.children,
            SceneNode::Repeat(n) => &n.children,
            SceneNode::Mask(n) => &n.children,
            SceneNode::Precompose(n) => &n.children,
            SceneNode::Layer(n) => &n.children,
            SceneNode::Camera(n) => &n.children,
            SceneNode::Character(n) => &n.children,
            SceneNode::Puppet(n) => &n.children,
            _ => continue,
        };
        collect_overrides(children, r);
    }
}

/// Validate every declaration, including unused styles, before GPU work starts.
pub(crate) fn validate(graph: &GraphScript) -> Result<(), GraphParseError> {
    let mut ids = std::collections::HashSet::new();
    for s in &graph.render_styles {
        if !ids.insert(&s.id) {
            return Err(error(format!("Duplicate RenderStyle id {}", s.id)));
        }
        if let Some(s) = &s.surface {
            one_of(
                s.shading.as_deref(),
                &["physical", "stylized", "toon", "clay", "cel"],
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
        if let Some(p) = &s.post {
            one_of(
                p.tone_mapping.as_deref(),
                &["none", "aces", "reinhard"],
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

// Lower once into per-island compiler payloads, never into shared materials.
// Explicit ColorManagement owns its complete group; it is evaluated later so
// animated keys always remain authoritative.
pub(crate) fn lower(graph: &mut GraphScript) -> Result<(), GraphParseError> {
    validate(graph)?;
    let reports = graph
        .scenes
        .iter()
        .map(|s| resolve_scene_render_style(graph, &s.id))
        .collect::<Result<Vec<_>, _>>()?;
    let bloom_processes = graph
        .render_styles
        .iter()
        .filter_map(|style| {
            let post = style.post.as_ref()?;
            (post.bloom_intensity.is_some_and(|value| value > 0.0))
                .then(|| (style_bloom_process_id(&style.id), post.clone()))
        })
        .collect::<Vec<_>>();
    for (scene, report) in graph.scenes.iter_mut().zip(reports) {
        if scene.render_style.is_some() {
            visit(&mut scene.children, &report);
            if let Some(id) = report.style_id.as_deref().and_then(|style_id| {
                report
                    .post
                    .bloom_intensity
                    .is_some_and(|value| value > 0.0)
                    .then(|| style_bloom_process_id(style_id))
            }) {
                attach_bloom(&mut scene.children, &id);
            }
        }
    }
    // Compile the same Process language as authored effects. This isolates
    // bloom to styled 3D groups and leaves SVG/title tracks unchanged.
    for (id, p) in bloom_processes {
        if graph.processes.iter().any(|v| v.id == id) {
            return Err(error("Reserved style Process id collision"));
        }
        let source = format!(
            r#"<Graph fps="30" duration="1s" size={{[{w},{h}]}}>
<Process id="{id}">
<Tex id="src" fmt="rgba16f" from="scene" />
<Tex id="out" fmt="rgba16f" size={{[{w},{h}]}} />
<Pass id="bloom" kind="compute" effect="glow_bloom" in={{["src"]}} out={{["out"]}} params={{{{ threshold: "{threshold}", intensity: "{intensity}", sigma: "3.0" }}}} />
</Process>
<Present from="{id}" />
</Graph>"#,
            w = graph.size.0,
            h = graph.size.1,
            threshold = p.bloom_threshold.unwrap_or(0.9),
            intensity = p.bloom_intensity.unwrap_or(0.0)
        );
        let compiled = crate::dsl::parse_graph_script(&source)?;
        graph.processes.extend(compiled.processes);
        graph.textures.extend(compiled.textures);
        graph.passes.extend(compiled.passes);
        graph.outputs.extend(compiled.outputs);
    }
    Ok(())
}

fn style_bloom_process_id(style_id: &str) -> String {
    format!("__ml_style_bloom_{style_id}")
}

/// Replace one Scene's resolved style after a discrete animation cut.
pub(crate) fn apply_scene_style_reference(
    graph: &mut GraphScript,
    scene_id: &str,
    style_id: &str,
) -> Result<(), GraphParseError> {
    let index = graph
        .scenes
        .iter()
        .position(|scene| scene.id == scene_id)
        .ok_or_else(|| error(format!("Unknown Scene {scene_id}")))?;
    graph.scenes[index].render_style = Some(style_id.to_string());
    let report = resolve_scene_render_style(graph, scene_id)?;
    let children = &mut graph.scenes[index].children;
    remove_compiled_bloom(children);
    visit(children, &report);
    if report.post.bloom_intensity.is_some_and(|value| value > 0.0) {
        attach_bloom(children, &style_bloom_process_id(style_id));
    }
    Ok(())
}

fn remove_compiled_bloom(nodes: &mut [SceneNode]) {
    for node in nodes {
        if let SceneNode::Group(group) = node {
            group
                .process_effects
                .retain(|effect| effect.id.as_deref() != Some("__ml_style_bloom"));
        }
        if let Some(children) = children_mut(node) {
            remove_compiled_bloom(children);
        }
    }
}

fn attach_bloom(nodes: &mut [SceneNode], process: &str) {
    for node in nodes {
        if let SceneNode::Group(group) = node {
            if group.composite.as_ref().is_some_and(|c| c.space == "3d") {
                // Explicit group effects follow the default style effect.
                group.process_effects.insert(
                    0,
                    crate::scene::model::SceneEffectRef {
                        process: process.into(),
                        id: Some("__ml_style_bloom".into()),
                        params: vec![],
                    },
                );
            }
        }
        if let Some(children) = children_mut(node) {
            attach_bloom(children, process);
        }
    }
}

fn children_mut(node: &mut SceneNode) -> Option<&mut Vec<SceneNode>> {
    Some(match node {
        SceneNode::Group(n) => &mut n.children,
        SceneNode::Timeline(n) => &mut n.children,
        SceneNode::Track(n) => &mut n.children,
        SceneNode::Sequence(n) => &mut n.children,
        SceneNode::Chain(n) => &mut n.children,
        SceneNode::Part(n) => &mut n.children,
        SceneNode::Repeat(n) => &mut n.children,
        SceneNode::Mask(n) => &mut n.children,
        SceneNode::Precompose(n) => &mut n.children,
        SceneNode::Layer(n) => &mut n.children,
        SceneNode::Camera(n) => &mut n.children,
        SceneNode::Character(n) => &mut n.children,
        SceneNode::Puppet(n) => &mut n.children,
        _ => return None,
    })
}
fn visit(nodes: &mut [SceneNode], report: &ResolvedSceneRenderStyle) {
    for node in nodes {
        let children = match node {
            SceneNode::Group(n) => {
                if let Some(c) = &mut n.composite {
                    if c.space == "3d" {
                        c.render_style = Some(report.clone());
                    }
                }
                &mut n.children
            }
            SceneNode::Timeline(n) => &mut n.children,
            SceneNode::Track(n) => &mut n.children,
            SceneNode::Sequence(n) => &mut n.children,
            SceneNode::Chain(n) => &mut n.children,
            SceneNode::Part(n) => &mut n.children,
            SceneNode::Repeat(n) => &mut n.children,
            SceneNode::Mask(n) => &mut n.children,
            SceneNode::Precompose(n) => &mut n.children,
            SceneNode::Layer(n) => &mut n.children,
            SceneNode::Camera(n) => &mut n.children,
            SceneNode::Character(n) => &mut n.children,
            SceneNode::Puppet(n) => &mut n.children,
            _ => continue,
        };
        visit(children, report);
    }
}
