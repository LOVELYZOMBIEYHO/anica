// =========================================
// =========================================
// crates/motionloom/src/render_style/resolve.rs

//! Resolve authored styles and lower them into scene-owned runtime payloads.

use super::model::*;
use super::validation::{color, error, validate};
use crate::dsl::{GraphParseError, GraphScript};
use crate::scene::model::{Scene3DNode, SceneNode};

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
    let mut universal = ResolvedUniversalStyle::default();
    if let Some(c) = style.and_then(|s| s.color.as_ref()) {
        universal.enabled = true;
        universal.tint = c
            .tint
            .as_deref()
            .map(color)
            .transpose()?
            .unwrap_or([1.0; 3]);
        universal.tint_strength = c.tint_strength.unwrap_or(0.0);
        universal.saturation = c.saturation.unwrap_or(1.0);
    }
    if let Some(t) = style.and_then(|s| s.tone.as_ref()) {
        universal.enabled = true;
        universal.exposure = t.exposure.unwrap_or(1.0);
        universal.contrast = t.contrast.unwrap_or(1.0);
        universal.shadow_color = t
            .shadow_color
            .as_deref()
            .map(color)
            .transpose()?
            .unwrap_or([0.0; 3]);
        universal.highlight_color = t
            .highlight_color
            .as_deref()
            .map(color)
            .transpose()?
            .unwrap_or([1.0; 3]);
        universal.tone_strength = t.tone_strength.unwrap_or(0.0);
    }
    let mut r = ResolvedSceneRenderStyle {
        anti_aliasing: style.map(|style| {
            style
                .anti_aliasing
                .as_ref()
                .map(|aa| ResolvedAntiAliasingStyle {
                    method: aa.method.clone().unwrap_or_else(|| "auto".into()),
                    quality: aa.quality.clone().unwrap_or_else(|| "medium".into()),
                    fallback: aa.fallback.clone().unwrap_or_else(|| "auto".into()),
                    sharpness: aa.sharpness.unwrap_or(0.0),
                })
                .unwrap_or_default()
        }),
        depth_of_field: style.and_then(|s| s.depth_of_field.clone()),
        universal,
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
