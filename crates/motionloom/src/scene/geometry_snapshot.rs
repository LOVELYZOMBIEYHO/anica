// =========================================
// =========================================
// crates/motionloom/src/scene/geometry_snapshot.rs

use super::*;
use crate::experimental::geometry::*;

pub(crate) async fn extract_geometry_snapshot(
    graph: &GraphScript,
    options: &SceneGeometryOptions,
    resolver: Arc<dyn AssetResolver>,
) -> Result<GeometrySnapshot, GeometryError> {
    let error = |e: MotionLoomSceneRenderError| GeometryError::Evaluation(e.to_string());
    if !graph.fps.is_finite() || graph.fps <= 0.0 {
        return Err(GeometryError::Invalid("invalid fps".into()));
    }
    let mut renderer =
        SceneFrameRenderer::new_for_profile_with_resolver(SceneRenderProfile::Cpu, resolver).await;
    let compiled = renderer
        .compiled_animation_graph_for_frame(graph, options.frame)
        .map_err(error)?;
    let graph = compiled.as_deref().unwrap_or(graph);
    renderer.prepare_frame_caches(graph);
    let scene = graph
        .scenes
        .iter()
        .find(|s| s.id == options.scene_id)
        .ok_or_else(|| GeometryError::Invalid(format!("missing Scene '{}'", options.scene_id)))?;
    let seconds = options.frame as f32 / graph.fps;
    let mut snapshot = GeometrySnapshot::default();
    visit(
        &mut renderer,
        &scene.children,
        graph,
        options,
        seconds,
        seconds / (graph.duration_ms as f32 / 1000.0).max(0.001),
        1.0,
        &mut snapshot,
    )?;
    if snapshot.meshes.is_empty() {
        return Err(GeometryError::Invalid("no active 3D models".into()));
    }
    snapshot.diagnostics.push("Static sampled geometry; Model transforms and skin pose are baked into world space. Cameras, lights, 2D presentation transforms and post effects are omitted.".into());
    snapshot.sign();
    Ok(snapshot)
}

// Recursive traversal keeps timing, visibility, and output state explicit at each branch.
#[allow(clippy::too_many_arguments)]
fn visit(
    renderer: &mut SceneFrameRenderer,
    nodes: &[SceneNode],
    graph: &GraphScript,
    options: &SceneGeometryOptions,
    sec: f32,
    norm: f32,
    inherited_opacity: f32,
    snapshot: &mut GeometrySnapshot,
) -> Result<(), GeometryError> {
    for node in nodes {
        let children = match node {
            SceneNode::Timeline(v) => Some(&v.children),
            SceneNode::Track(v) => Some(&v.children),
            SceneNode::Sequence(v) => {
                if let Some((local_norm, local_sec)) = scene_sequence_local_time(v, None, sec) {
                    visit(
                        renderer,
                        &v.children,
                        graph,
                        options,
                        local_sec,
                        local_norm,
                        inherited_opacity,
                        snapshot,
                    )?;
                }
                None
            }
            SceneNode::Group(v) => {
                let opacity = if options.include_hidden {
                    1.0
                } else {
                    inherited_opacity
                        * eval_scene_number(&v.opacity, norm, sec)
                            .map_err(|e| GeometryError::Evaluation(e.to_string()))?
                };
                if opacity <= 0.0 {
                    continue;
                }
                if let Some(c) = &v.composite {
                    if c.space == "3d" {
                        let mut c = c.clone();
                        // Strip only the working copy: the source keeps every camera.
                        c.nodes_3d.retain(|n| {
                            matches!(
                                n,
                                Scene3DNode::Model(_)
                                    | Scene3DNode::Anchor(_)
                                    | Scene3DNode::VolumeRepeat(_)
                            )
                        });
                        c.active_camera = None;
                        c.render_style = None;
                        let frame = (sec * graph.fps).round() as u32;
                        let (mut world, root, overrides) = renderer
                            .prepare_scene_3d_composite(
                                &c,
                                frame,
                                graph.fps,
                                graph.duration_ms,
                                graph.size,
                                norm,
                                sec,
                            )
                            .map_err(|e| GeometryError::Evaluation(e.to_string()))?;
                        for w in &mut world.worlds {
                            w.camera = WorldCamera::default();
                            for a in &mut w.actors {
                                a.camera_hidden_bones.clear();
                                if options.include_hidden {
                                    a.opacity = "1".into();
                                } else {
                                    a.opacity = format!("({})*{}", a.opacity, opacity);
                                }
                            }
                        }
                        snapshot
                            .meshes
                            .extend(renderer.scene_3d_renderer.extract_asset_meshes(
                                &world,
                                frame,
                                &root,
                                &overrides,
                                options.selected_model_ids.as_deref(),
                            )?);
                    }
                }
                visit(
                    renderer,
                    &v.children,
                    graph,
                    options,
                    sec,
                    norm,
                    opacity,
                    snapshot,
                )?;
                None
            }
            SceneNode::Layer(layer) => {
                if layer.source.is_some() {
                    return Err(GeometryError::Unsupported(
                        "Scene-referencing Layer export".into(),
                    ));
                }
                let opacity = if options.include_hidden {
                    1.0
                } else {
                    inherited_opacity
                        * eval_scene_number(&layer.opacity, norm, sec)
                            .map_err(|e| GeometryError::Evaluation(e.to_string()))?
                };
                if opacity > 0.0 {
                    visit(
                        renderer,
                        &layer.children,
                        graph,
                        options,
                        sec,
                        norm,
                        opacity,
                        snapshot,
                    )?;
                }
                None
            }
            SceneNode::Chain(chain) => {
                let mut cursor = chain.from_ms as i64;
                for child in &chain.children {
                    if let SceneNode::Sequence(s) = child {
                        if let Some((local_norm, local_sec)) =
                            scene_sequence_local_time(s, Some(cursor), sec)
                        {
                            visit(
                                renderer,
                                &s.children,
                                graph,
                                options,
                                local_sec,
                                local_norm,
                                inherited_opacity,
                                snapshot,
                            )?;
                        }
                        cursor += s.duration_ms as i64 + chain.gap_ms;
                    }
                }
                None
            }
            SceneNode::Repeat(_) | SceneNode::Use(_) | SceneNode::Precompose(_) => {
                return Err(GeometryError::Unsupported(
                    "Chain/2D Repeat/Use/Precompose export; expand these containers first".into(),
                ));
            }
            _ => None,
        };
        if let Some(children) = children {
            visit(
                renderer,
                children,
                graph,
                options,
                sec,
                norm,
                inherited_opacity,
                snapshot,
            )?;
        }
    }
    Ok(())
}
