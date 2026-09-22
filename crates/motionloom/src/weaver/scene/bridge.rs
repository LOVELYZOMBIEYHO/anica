// =========================================
// =========================================
// crates/motionloom/src/weaver/scene/bridge.rs

// Compiled under scene::render to reuse private lowering without widening its API.
use super::*;
use crate::weaver::scene::Snapshot;
use crate::weaver::{RenderJob, SceneOutputMode, WeaverError};

pub(crate) async fn weaver_snapshot(
    graph: &GraphScript,
    job: &RenderJob,
    resolver: Arc<dyn AssetResolver>,
) -> Result<Snapshot, WeaverError> {
    let time_seconds = job.frame as f32 / graph.fps.max(0.001);
    let err = |e: MotionLoomSceneRenderError| WeaverError::Scene(e.to_string());
    let mut renderer =
        SceneFrameRenderer::new_for_profile_with_resolver(SceneRenderProfile::Cpu, resolver).await;
    let evaluated = renderer
        .compiled_animation_graph_for_frame(graph, job.frame)
        .map_err(err)?;
    let mut graph = evaluated.as_deref().unwrap_or(graph).clone();
    let scene_index = select_scene(&graph, &job.scene_id)?;
    let scene_id = graph.scenes[scene_index].id.clone();
    // `render_style = "auto"` picks the first style that Weaver can represent,
    // so hosts do not have to know style ids.
    let style_id = match job.render_style.as_str() {
        "" => None,
        "auto" => match pick_weaver_style(&graph) {
            Some(id) => Some(id),
            // No authored style is directly representable: keep its lighting and
            // tone intent but drop the surface/outline/post fields Weaver cannot
            // use, rather than rejecting the document or blowing out the frame.
            None => {
                crate::render_style::sanitize_scene_style(&mut graph, &scene_id)
                    .map_err(|e| WeaverError::Scene(e.to_string()))?;
                None
            }
        },
        other => Some(other.to_string()),
    };
    if let Some(style_id) = style_id {
        crate::render_style::apply_scene_style_reference(&mut graph, &scene_id, &style_id)
            .map_err(|e| WeaverError::Scene(e.to_string()))?;
    }
    renderer.prepare_frame_caches(&graph);
    let scene = &graph.scenes[scene_index];
    let composition = crate::scene::compositor::build_scene_composition_plan(&graph, &scene_id)
        .map_err(|error| WeaverError::Scene(error.to_string()))?;
    let sec = job.frame as f32 / graph.fps;
    let composition_layers = if job.output_mode == SceneOutputMode::CompositeScene {
        raster_composition_layers(
            &mut renderer,
            scene,
            graph.render_size.unwrap_or(graph.size),
            scene.size.unwrap_or(graph.size),
            sec / (graph.duration_ms as f32 / 1000.0).max(0.001),
            sec,
        )
        .map_err(err)?
    } else {
        Vec::new()
    };
    let mut islands = Vec::new();
    collect(
        &scene.children,
        sec,
        sec / (graph.duration_ms as f32 / 1000.0).max(0.001),
        &mut islands,
    )?;
    if islands.is_empty() {
        return Err(WeaverError::Unsupported(
            "at least one active 3D composite is required".into(),
        ));
    }
    if job.output_mode == SceneOutputMode::CompositeScene {
        let last_three_d = composition
            .layers
            .iter()
            .rposition(|layer| layer.domain == crate::scene::compositor::RenderDomain::ThreeD)
            .unwrap_or(0);
        if composition.layers[..last_three_d]
            .iter()
            .any(|layer| layer.domain != crate::scene::compositor::RenderDomain::ThreeD)
        {
            return Err(WeaverError::Unsupported(
                "2D below or between 3D islands requires independent 3D island textures; the merged Weaver beauty cannot preserve that authored order"
                    .into(),
            ));
        }
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
    let mut meshes = renderer
        .scene_3d_renderer
        .extract_asset_meshes(&world, frame, &root, &overrides, None, true)
        .map_err(|e| WeaverError::Scene(e.to_string()))?;
    let mut primary_camera_visibility = mesh_primary_camera_visibility(&world, &meshes);
    if std::env::var_os("WEAVER_SCENE_DEBUG").is_some() {
        let hidden = meshes
            .iter()
            .zip(&primary_camera_visibility)
            .filter(|(_, visible)| !**visible)
            .map(|(mesh, _)| mesh.name.as_str())
            .collect::<Vec<_>>();
        eprintln!("weaver debug: primary-camera-hidden meshes={hidden:?}");
    }
    // Islands sharing one authored camera can be traced together. This keeps
    // physical visibility/reflections correct while the plan retains their
    // authored ordering for the future layered executor.
    for (composite, local_sec, norm) in islands.iter().skip(1) {
        let island_frame = (local_sec * graph.fps).round() as u32;
        let (island_world, island_root, island_overrides) = renderer
            .prepare_scene_3d_composite(
                composite,
                island_frame,
                graph.fps,
                graph.duration_ms,
                graph.size,
                *norm,
                *local_sec,
            )
            .map_err(err)?;
        let island_camera = &island_world
            .presented_world()
            .ok_or_else(|| WeaverError::Scene("3D island has no presented world".into()))?
            .camera;
        if !compatible_island_camera(&camera, island_camera) {
            return Err(WeaverError::Unsupported(
                "multiple 3D islands currently require the same camera projection".into(),
            ));
        }
        let island_meshes = renderer
            .scene_3d_renderer
            .extract_asset_meshes(
                &island_world,
                island_frame,
                &island_root,
                &island_overrides,
                None,
                true,
            )
            .map_err(|error| WeaverError::Scene(error.to_string()))?;
        primary_camera_visibility.extend(mesh_primary_camera_visibility(
            &island_world,
            &island_meshes,
        ));
        meshes.extend(island_meshes);
    }
    // Opt-in scene/camera dump for diagnosing empty or blown-out frames.
    if std::env::var_os("WEAVER_SCENE_DEBUG").is_some() {
        eprintln!(
            "weaver debug: camera id={:?} control={:?} target=({},{},{}) distance={} yaw={} pitch={} fov={}",
            camera.id,
            camera.control,
            camera.target_x,
            camera.target_y,
            camera.target_z,
            camera.distance,
            camera.yaw,
            camera.pitch,
            camera.fov
        );
    }
    let mut diagnostics = vec!["Thin-lens camera replaces legacy screen-space DoF. Light intensities retain scene units; physical unit calibration is not assumed.".into()];
    diagnostics.push("Path-traced geometry casts physical shadows; preview-only AO, contact-shadow strength and per-light shadow-strength hacks are not applied.".into());
    if islands.len() > 1 {
        diagnostics.push(format!(
            "Merged {} camera-compatible 3D islands into one physical trace.",
            islands.len()
        ));
    }
    if !world.attachments.is_empty() {
        diagnostics.push(format!(
            "Evaluated {} inline attachment(s) after Action and constraint sampling.",
            world.attachments.len()
        ));
    }
    if world.lighting.atmosphere_medium.is_some() {
        diagnostics
            .push("AtmosphereMediumPlan is shared by interactive and final renderers.".into());
    }
    if let Some(s) = &world.lighting.render_style {
        if let Some(aa) = &s.anti_aliasing {
            diagnostics.push(format!(
                "Requested AA {} {} resolves through Weaver pixel sampling ({}-{} samples); spatial preview fallback is not used.",
                aa.method, aa.quality, job.sampling.min_samples, job.sampling.max_samples
            ));
        } else {
            diagnostics.push(
                "RenderStyle omits AntiAliasingStyle: authored AA is off; Weaver still performs the pixel samples required by the render job."
                    .into(),
            );
        }
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
    if !matches!(
        world.lighting.color_management.tone_mapping.as_str(),
        "none" | "linear" | "aces" | "filmic_aces_v1" | "reinhard"
    ) {
        return Err(WeaverError::Unsupported("display transform".into()));
    }
    Ok(Snapshot {
        meshes,
        primary_camera_visibility,
        camera,
        lighting: world.lighting,
        time_seconds,
        diagnostics,
        composition,
        composition_layers,
    })
}

/// Rasterize independent image-plane tracks and normalize them immediately.
/// Defs are repeated as resources only; they do not contribute visible pixels.
fn raster_composition_layers(
    renderer: &mut SceneFrameRenderer,
    scene: &SceneRootNode,
    output: (u32, u32),
    logical: (u32, u32),
    norm: f32,
    sec: f32,
) -> Result<Vec<crate::scene::compositor::ResolvedCompositionLayer>, MotionLoomSceneRenderError> {
    let definitions = scene
        .children
        .iter()
        .filter(|node| matches!(node, SceneNode::Defs(_)))
        .cloned()
        .collect::<Vec<_>>();
    let mut runs = Vec::new();
    collect_image_plane_tracks(&scene.children, &mut runs);
    runs.sort_by_key(|track| track.composite_order.unwrap_or(track.z));

    let mut layers = Vec::with_capacity(runs.len());
    for (index, track) in runs.into_iter().enumerate() {
        let mut nodes = definitions.clone();
        nodes.push(SceneNode::Track(track.clone()));
        let raster = renderer.render_cpu_scene_nodes_scaled(
            &nodes,
            output,
            logical,
            norm,
            sec,
            [0, 0, 0, 0],
        )?;
        let domain = if track.space.eq_ignore_ascii_case("lens") {
            crate::scene::compositor::RenderDomain::Lens
        } else {
            crate::scene::compositor::RenderDomain::Screen
        };
        layers.push(crate::scene::compositor::ResolvedCompositionLayer {
            id: track
                .id
                .clone()
                .unwrap_or_else(|| format!("2d-run-{index}")),
            order: track.composite_order.unwrap_or(track.z),
            domain,
            color_stage: crate::scene::compositor::ColorStage::DisplayLinearPostTransform,
            image: crate::scene::compositor::LinearPremultipliedImage::from_srgb_straight(&raster),
        });
    }
    Ok(layers)
}

fn collect_image_plane_tracks<'a>(nodes: &'a [SceneNode], tracks: &mut Vec<&'a SceneTrackNode>) {
    for node in nodes {
        match node {
            SceneNode::Timeline(value) => collect_image_plane_tracks(&value.children, tracks),
            SceneNode::Track(value) => {
                if value.space != "3d" {
                    tracks.push(value);
                }
            }
            SceneNode::Sequence(value) => collect_image_plane_tracks(&value.children, tracks),
            SceneNode::Chain(value) => collect_image_plane_tracks(&value.children, tracks),
            _ => {}
        }
    }
}

fn mesh_primary_camera_visibility(
    graph: &crate::world::WorldGraph,
    meshes: &[crate::experimental::geometry::ResolvedMesh],
) -> Vec<bool> {
    let Some(world) = graph.presented_world() else {
        return vec![true; meshes.len()];
    };
    meshes
        .iter()
        .map(|mesh| {
            let actor_id = mesh
                .name
                .rsplit_once(':')
                .map_or(mesh.name.as_str(), |(actor, _)| actor);
            !world.actor_slice().iter().any(|actor| {
                actor.id == actor_id && actor.camera_hidden_bones.iter().any(|bone| bone == "hips")
            })
        })
        .collect()
}

fn compatible_island_camera(a: &crate::world::WorldCamera, b: &crate::world::WorldCamera) -> bool {
    a.projection == b.projection
        && a.target_x == b.target_x
        && a.target_y == b.target_y
        && a.target_z == b.target_z
        && a.distance == b.distance
        && a.yaw == b.yaw
        && a.pitch == b.pitch
        && a.fov == b.fov
}

// Resolve an explicit scene id, or auto-pick the first scene that owns a 3D
// composite so hosts can render without knowing authored ids.
fn select_scene(graph: &GraphScript, scene_id: &str) -> Result<usize, WeaverError> {
    // "" and the explicit "auto" marker both mean "pick a scene for me".
    if !scene_id.is_empty() && scene_id != "auto" {
        return graph
            .scenes
            .iter()
            .position(|s| s.id == scene_id)
            .ok_or_else(|| WeaverError::Scene(format!("missing scene `{scene_id}`")));
    }
    graph
        .scenes
        .iter()
        .position(|s| has_3d_composite(&s.children))
        .or(if graph.scenes.is_empty() {
            None
        } else {
            Some(0)
        })
        .ok_or_else(|| WeaverError::Scene("no scene in graph".into()))
}

fn has_3d_composite(nodes: &[SceneNode]) -> bool {
    nodes.iter().any(|n| match n {
        SceneNode::Timeline(v) => has_3d_composite(&v.children),
        SceneNode::Track(v) => has_3d_composite(&v.children),
        SceneNode::Sequence(v) => has_3d_composite(&v.children),
        SceneNode::Group(v) => {
            v.composite.as_ref().is_some_and(|c| c.space == "3d") || has_3d_composite(&v.children)
        }
        _ => false,
    })
}

// `render_style = "auto"` stays honest by only choosing a style Weaver can
// represent instead of failing on the scene's authored 2D-first style.
fn pick_weaver_style(graph: &GraphScript) -> Option<String> {
    graph
        .render_styles
        .iter()
        .find(|s| weaver_style_compatible(s))
        .map(|s| s.id.clone())
}

fn weaver_style_compatible(style: &crate::render_style::RenderStyleNode) -> bool {
    if let Some(surface) = &style.surface {
        let shading = surface.shading.as_deref().unwrap_or("physical");
        if !matches!(shading, "physical" | "filmic_physical_v1") {
            return false;
        }
        if surface.specular.is_some_and(|v| v != 1.0)
            || surface.roughness_bias.is_some_and(|v| v != 0.0)
            || surface.saturation.is_some_and(|v| v != 1.0)
        {
            return false;
        }
    }
    if let Some(outline) = &style.outline {
        if outline.enabled.unwrap_or(false) && outline.width.unwrap_or(0.0) > 0.0 {
            return false;
        }
    }
    if let Some(post) = &style.post {
        if post.bloom_intensity.is_some_and(|v| v > 0.0)
            || post.white_balance.is_some_and(|v| v != 6500.0)
        {
            return false;
        }
    }
    true
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
            // Resource definitions never contribute a render island themselves.
            SceneNode::Defs(_) => {}
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
            // Image-plane content is retained by SceneCompositionPlan and does
            // not enter Weaver's geometry snapshot.
            _ => {}
        }
    }
    Ok(())
}
