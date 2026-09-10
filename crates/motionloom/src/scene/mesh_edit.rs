// =========================================
// =========================================
// crates/motionloom/src/scene/mesh_edit.rs

use super::*;
use crate::experimental::geometry::GeometryError;

// Resolve through the same animation compiler and CompositeGroup preparation as preview.
pub async fn mesh_edit_snapshot(
    graph: &GraphScript,
    model: &str,
    frame: u32,
) -> Result<serde_json::Value, GeometryError> {
    let mut renderer = SceneFrameRenderer::new_for_profile_with_resolver(
        SceneRenderProfile::Cpu,
        Arc::new(crate::asset::PathAssetResolver),
    )
    .await;
    let compiled = renderer
        .compiled_animation_graph_for_frame(graph, frame)
        .map_err(|e| GeometryError::Evaluation(e.to_string()))?;
    let graph = compiled.as_deref().unwrap_or(graph);
    renderer.prepare_frame_caches(graph);
    let sec = frame as f32 / graph.fps;
    for scene in &graph.scenes {
        if let Some(result) = visit(&mut renderer, &scene.children, graph, model, sec)? {
            return Ok(result);
        }
    }
    Err(GeometryError::Invalid(
        "No active MeshAsset model found".into(),
    ))
}

fn visit(
    renderer: &mut SceneFrameRenderer,
    nodes: &[SceneNode],
    graph: &GraphScript,
    model: &str,
    sec: f32,
) -> Result<Option<serde_json::Value>, GeometryError> {
    let norm = sec / (graph.duration_ms as f32 / 1000.0).max(0.001);
    for node in nodes {
        match node {
            SceneNode::Group(g) => {
                if let Some(c) = &g.composite {
                    if c.nodes_3d.iter().any(
                        |n| matches!(n, Scene3DNode::Model(m) if m.id.as_deref() == Some(model)),
                    ) {
                        let frame = (sec * graph.fps).round() as u32;
                        let (world, root, overrides) = renderer
                            .prepare_scene_3d_composite(
                                c,
                                frame,
                                graph.fps,
                                graph.duration_ms,
                                graph.size,
                                norm,
                                sec,
                            )
                            .map_err(|e| GeometryError::Evaluation(e.to_string()))?;
                        check_group(g, norm, sec)?;
                        return renderer
                            .scene_3d_renderer
                            .mesh_edit_snapshot(
                                &world,
                                frame,
                                &root,
                                &overrides,
                                model,
                                graph.size.into(),
                            )
                            .map(Some);
                    }
                }
                if let Some(result) = visit(renderer, &g.children, graph, model, sec)? {
                    check_group(g, norm, sec)?;
                    return Ok(Some(result));
                }
            }
            SceneNode::Timeline(n) => {
                if let Some(v) = visit(renderer, &n.children, graph, model, sec)? {
                    return Ok(Some(v));
                }
            }
            SceneNode::Track(n) => {
                if let Some(v) = visit(renderer, &n.children, graph, model, sec)? {
                    return Ok(Some(v));
                }
            }
            SceneNode::Sequence(n) => {
                if let Some((_, local)) = scene_sequence_local_time(n, None, sec) {
                    if let Some(v) = visit(renderer, &n.children, graph, model, local)? {
                        return Ok(Some(v));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(None)
}

// Refuse presentation-space distortion until the overlay supports the same transform.
fn check_group(g: &GroupNode, norm: f32, sec: f32) -> Result<(), GeometryError> {
    for (value, expected) in [
        (&g.x, 0.0),
        (&g.y, 0.0),
        (&g.rotation, 0.0),
        (&g.scale, 1.0),
        (&g.scale_x, 1.0),
        (&g.scale_y, 1.0),
        (&g.skew_x, 0.0),
        (&g.skew_y, 0.0),
    ] {
        let actual = eval_scene_number(value, norm, sec)
            .map_err(|e| GeometryError::Evaluation(e.to_string()))?;
        if (actual - expected).abs() > 0.00001 {
            return Err(GeometryError::Unsupported("Mesh editing requires identity 2D Group transforms; Model 3D transforms are supported".into()));
        }
    }
    if g.deform_grid.is_some() || g.grid_from.is_some() {
        return Err(GeometryError::Unsupported(
            "deformed presentation group".into(),
        ));
    }
    Ok(())
}
