// =========================================
// =========================================
// crates/motionloom/src/weaver/scene/bridge.rs

// Compiled under scene::render to reuse private lowering without widening its API.
use super::*;
use crate::weaver::scene::Snapshot;
use crate::weaver::{RenderJob, WeaverError};

pub(crate) async fn weaver_snapshot(
    graph: &GraphScript,
    job: &RenderJob,
    resolver: Arc<dyn AssetResolver>,
) -> Result<Snapshot, WeaverError> {
    let err = |e: MotionLoomSceneRenderError| WeaverError::Scene(e.to_string());
    let mut renderer =
        SceneFrameRenderer::new_for_profile_with_resolver(SceneRenderProfile::Cpu, resolver).await;
    let evaluated = renderer
        .compiled_animation_graph_for_frame(graph, job.frame)
        .map_err(err)?;
    let mut graph = evaluated.as_deref().unwrap_or(graph).clone();
    if !job.render_style.is_empty() {
        crate::render_style::apply_scene_style_reference(
            &mut graph,
            &job.scene_id,
            &job.render_style,
        )
        .map_err(|e| WeaverError::Scene(e.to_string()))?;
    }
    renderer.prepare_frame_caches(&graph);
    let scene = graph
        .scenes
        .iter()
        .find(|s| s.id == job.scene_id)
        .ok_or_else(|| WeaverError::Scene("missing scene".into()))?;
    let sec = job.frame as f32 / graph.fps;
    let mut islands = Vec::new();
    collect(
        &scene.children,
        sec,
        sec / (graph.duration_ms as f32 / 1000.0).max(0.001),
        &mut islands,
    )?;
    if islands.len() != 1 {
        return Err(WeaverError::Unsupported(
            "exactly one active 3D composite is required".into(),
        ));
    }
    let (composite, local_sec, norm) = &islands[0];
    let frame = (local_sec * graph.fps).round() as u32;
    let (world, root, overrides) = renderer
        .prepare_scene_3d_composite(
            composite,
            frame,
            graph.fps,
            graph.duration_ms,
            graph.size,
            *norm,
            *local_sec,
        )
        .map_err(err)?;
    let camera = world
        .presented_world()
        .ok_or_else(|| WeaverError::Scene("no presented world".into()))?
        .camera
        .clone();
    let meshes = renderer
        .scene_3d_renderer
        .extract_asset_meshes(&world, frame, &root, &overrides)
        .map_err(|e| WeaverError::Scene(e.to_string()))?;
    let mut diagnostics = vec!["Thin-lens camera replaces legacy screen-space DoF. Light intensities retain scene units; physical unit calibration is not assumed.".into()];
    diagnostics.push("Path-traced geometry casts physical shadows; preview-only AO, contact-shadow strength and per-light shadow-strength hacks are not applied.".into());
    if world.lighting.atmosphere_fog.is_some() {
        if job.volume.is_some() {
            diagnostics
                .push("Legacy fog replaced by the job's explicit bounded physical medium.".into());
        } else {
            if !job.allow_legacy_fog_omission {
                return Err(WeaverError::Unsupported("legacy fog is not a physical medium; configure volume or explicitly allow_legacy_fog_omission for a lighting baseline".into()));
            }
            diagnostics.push("Legacy atmosphere fog explicitly omitted for baseline.".into());
        }
    }
    if let Some(s) = &world.lighting.render_style {
        if !matches!(s.shading.as_str(), "physical" | "filmic_physical_v1") {
            return Err(WeaverError::Unsupported(format!(
                "surface preset {}",
                s.shading
            )));
        }
        if s.post.bloom_intensity.is_some_and(|v| v > 0.0) || s.cel.outline_width > 0.0 {
            return Err(WeaverError::Unsupported("bloom/outline compositing".into()));
        }
        if s.roughness_bias != 0.0 || s.surface_saturation != 1.0 || s.specular != 1.0 {
            return Err(WeaverError::Unsupported("surface roughnessBias/saturation/specular overrides; use glTF material values for this milestone".into()));
        }
    }
    if world.lighting.color_management.white_balance_kelvin != 6500.0 {
        return Err(WeaverError::Unsupported("non-neutral white balance".into()));
    }
    if !matches!(
        world.lighting.color_management.tone_mapping.as_str(),
        "none" | "linear" | "aces" | "filmic_aces_v1" | "reinhard"
    ) {
        return Err(WeaverError::Unsupported("display transform".into()));
    }
    Ok(Snapshot {
        meshes,
        camera,
        lighting: world.lighting,
        diagnostics,
    })
}

// Reject unimplemented composition instead of silently exporting a different scene.
fn collect(
    nodes: &[SceneNode],
    sec: f32,
    norm: f32,
    out: &mut Vec<(CompositeGroupConfig, f32, f32)>,
) -> Result<(), WeaverError> {
    for n in nodes {
        match n {
            SceneNode::Timeline(v) => collect(&v.children, sec, norm, out)?,
            SceneNode::Track(v) => collect(&v.children, sec, norm, out)?,
            SceneNode::Sequence(v) => {
                if let Some((n, s)) = scene_sequence_local_time(v, None, sec) {
                    collect(&v.children, s, n, out)?;
                }
            }
            SceneNode::Group(v) => {
                let opacity = eval_scene_number(&v.opacity, norm, sec)
                    .map_err(|e| WeaverError::Scene(e.to_string()))?;
                if opacity <= 0.0 {
                    continue;
                }
                if opacity != 1.0 {
                    return Err(WeaverError::Unsupported("partial group opacity".into()));
                }
                let eval = |s: &str| {
                    eval_scene_number(s, norm, sec).map_err(|e| WeaverError::Scene(e.to_string()))
                };
                for s in [
                    &v.x,
                    &v.y,
                    &v.rotation,
                    &v.skew_x,
                    &v.skew_y,
                    &v.deform_amount,
                ] {
                    if eval(s)? != 0.0 {
                        return Err(WeaverError::Unsupported(
                            "2D group transforms/deformation".into(),
                        ));
                    }
                }
                for s in [&v.scale, &v.scale_x, &v.scale_y] {
                    if eval(s)? != 1.0 {
                        return Err(WeaverError::Unsupported("2D group scale".into()));
                    }
                }
                if v.mask.is_some()
                    || v.mask_from.is_some()
                    || !v.effects.is_empty()
                    || !v.process_effects.is_empty()
                {
                    return Err(WeaverError::Unsupported("group masks/effects".into()));
                }
                if let Some(c) = &v.composite {
                    if c.space != "3d" {
                        return Err(WeaverError::Unsupported("2D composite".into()));
                    }
                    out.push((c.clone(), sec, norm));
                }
                collect(&v.children, sec, norm, out)?;
            }
            _ => {
                return Err(WeaverError::Unsupported(
                    "scene container outside Timeline/Track/Sequence/CompositeGroup".into(),
                ));
            }
        }
    }
    Ok(())
}
