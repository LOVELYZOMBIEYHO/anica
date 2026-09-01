// =========================================
// =========================================
// crates/motionloom/src/wasm_api.rs

use wasm_bindgen::prelude::*;

use std::sync::{Arc, Once};

use web_sys::HtmlCanvasElement;

use crate::asset::MemoryAssetResolver;
use crate::authoring::{
    motionloom_analyze_script_for_target_json as analyze_script_for_target_json,
    motionloom_analyze_script_json as analyze_script_json,
    motionloom_dsl_schema_json as dsl_schema_json,
    motionloom_showcase_schema_json as showcase_schema_json,
};
use crate::dsl::{GraphScript, is_graph_script, parse_graph_script};
use crate::process::cpu_renderer::render_process_frame_cpu;
#[cfg(target_arch = "wasm32")]
use crate::process::wasm_webgpu::render_process_frame_to_canvas_gpu as render_process_frame_to_canvas_gpu_impl;
use crate::scene::animation::{animation_property_schema_json, inspect_animation_targets};
use crate::scene::editor_actions::{
    ActionEditCommand, apply_action_edit, extract_editable_action_document,
};
use crate::scene::model::{GroupNode, Scene3DNode, SceneNode};
use crate::scene::render::{SceneRenderProfile, SceneRenderer, render_scene_graph_frame};
use crate::world::{
    WorldFrameRenderer, inspect_glb_environment_json, inspect_glb_humanoid_profile_json,
    inspect_glb_skeleton_json, is_world_graph_script, parse_world_graph_script,
};

fn js_error(message: String) -> JsValue {
    js_sys::Error::new(&message).into()
}

/// CPU-only diagnostic handle; does not change any preview or renderer state.
#[wasm_bindgen]
pub struct WasmPoseDiagnostics {
    graph: crate::WorldGraph,
    mesh: crate::GlbMeshData,
}

#[wasm_bindgen]
impl WasmPoseDiagnostics {
    /// Accept an existing World DSL document and GLB bytes; never fetch assets.
    #[wasm_bindgen(constructor)]
    pub fn new(world_dsl: &str, glb: &[u8]) -> Result<WasmPoseDiagnostics, JsValue> {
        let graph = parse_world_graph_script(world_dsl).map_err(|e| js_error(e.to_string()))?;
        let mesh = crate::experimental::load_glb_mesh_data_from_bytes(
            std::path::Path::new("diagnostic.glb"),
            glb,
        )
        .map_err(|e| js_error(e.to_string()))?;
        Ok(Self { graph, mesh })
    }

    /// Return complete model-global joint matrices using the native evaluator.
    pub fn sample_json(&self, actor_id: &str, frame: u32, fps: f32) -> Result<String, JsValue> {
        let pose = crate::experimental::diagnose_world_actor_pose(
            &self.graph,
            &self.mesh,
            actor_id,
            crate::WorldTime {
                frame,
                fps,
                duration_ms: self.graph.duration_ms,
            },
        )
        .map_err(|e| js_error(e.to_string()))?;
        serde_json::to_string(&pose).map_err(|e| js_error(e.to_string()))
    }

    /// Evaluate the same World pose path into the stable, versioned rig report.
    pub fn evaluate_json(&self, request_json: &str) -> Result<String, JsValue> {
        let request: crate::RigEvaluationRequest = serde_json::from_str(request_json)
            .map_err(|error| js_error(format!("invalid rig evaluation request: {error}")))?;
        let report = crate::evaluate_world_actor_rig(&self.graph, &self.mesh, &request)
            .map_err(|error| js_error(error.to_string()))?;
        Ok(crate::rig_evaluation_report_json(&report))
    }
}

/// Compare two previously evaluated rig reports without loading or rendering assets.
#[wasm_bindgen]
pub fn motionloom_compare_rigs_json(
    reference_json: &str,
    candidate_json: &str,
    options_json: &str,
) -> Result<String, JsValue> {
    let reference: crate::RigEvaluationReport = serde_json::from_str(reference_json)
        .map_err(|error| js_error(format!("invalid reference rig report: {error}")))?;
    let candidate: crate::RigEvaluationReport = serde_json::from_str(candidate_json)
        .map_err(|error| js_error(format!("invalid candidate rig report: {error}")))?;
    let options = if options_json.trim().is_empty() {
        crate::RigComparisonOptions::default()
    } else {
        serde_json::from_str(options_json)
            .map_err(|error| js_error(format!("invalid rig comparison options: {error}")))?
    };
    Ok(crate::rig_comparison_report_json(
        &crate::compare_humanoid_poses(&reference, &candidate, options),
    ))
}

/// Produce read-only calibration suggestions from one comparison report.
#[wasm_bindgen]
pub fn motionloom_propose_rig_calibration_json(comparison_json: &str) -> Result<String, JsValue> {
    let comparison: crate::RigComparisonReport = serde_json::from_str(comparison_json)
        .map_err(|error| js_error(format!("invalid rig comparison report: {error}")))?;
    serde_json::to_string_pretty(&crate::propose_rig_calibration(&comparison))
        .map_err(|error| js_error(error.to_string()))
}

/// Return the versioned JSON Schema envelope for rig diagnostics.
#[wasm_bindgen]
pub fn motionloom_rig_diagnostics_schema_json() -> String {
    crate::rig_diagnostics_schema_json()
}

/// Surface Rust panic locations in the browser instead of an opaque WASM
/// `unreachable`, which makes GPU-path failures actionable for hosts.
fn install_wasm_panic_hook() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            web_sys::console::error_1(&JsValue::from_str(&format!(
                "MotionLoom WASM panic: {info}"
            )));
        }));
    });
}

fn set_group_numeric_attr(group: &mut GroupNode, attr: &str, value: &str) -> bool {
    match attr {
        "x" => group.x = value.to_string(),
        "y" => group.y = value.to_string(),
        "rotation" => group.rotation = value.to_string(),
        "scale" => group.scale = value.to_string(),
        "scaleX" | "scale_x" => group.scale_x = value.to_string(),
        "scaleY" | "scale_y" => group.scale_y = value.to_string(),
        "skewX" | "skew_x" => group.skew_x = value.to_string(),
        "skewY" | "skew_y" => group.skew_y = value.to_string(),
        "transformOriginX" | "transform_origin_x" => group.transform_origin_x = value.to_string(),
        "transformOriginY" | "transform_origin_y" => group.transform_origin_y = value.to_string(),
        "opacity" => group.opacity = value.to_string(),
        _ => return false,
    }
    true
}

fn set_group_attr_in_nodes(
    nodes: &mut [SceneNode],
    group_id: &str,
    attr: &str,
    value: &str,
) -> bool {
    for node in nodes {
        match node {
            SceneNode::Timeline(timeline) => {
                if set_group_attr_in_nodes(&mut timeline.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Track(track) => {
                if set_group_attr_in_nodes(&mut track.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Sequence(sequence) => {
                if set_group_attr_in_nodes(&mut sequence.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Chain(chain) => {
                if set_group_attr_in_nodes(&mut chain.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Group(group) => {
                if group.id.as_deref() == Some(group_id) {
                    return set_group_numeric_attr(group, attr, value);
                }
                if set_group_attr_in_nodes(&mut group.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Part(part) => {
                if set_group_attr_in_nodes(&mut part.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Repeat(repeat) => {
                if set_group_attr_in_nodes(&mut repeat.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Mask(mask) => {
                if set_group_attr_in_nodes(&mut mask.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Precompose(precompose) => {
                if set_group_attr_in_nodes(&mut precompose.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Layer(layer) => {
                if set_group_attr_in_nodes(&mut layer.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Camera(camera) => {
                if set_group_attr_in_nodes(&mut camera.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Character(character) => {
                if set_group_attr_in_nodes(&mut character.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::Puppet(puppet) => {
                if set_group_attr_in_nodes(&mut puppet.children, group_id, attr, value) {
                    return true;
                }
            }
            SceneNode::MeshTopology(mesh) => {
                if set_group_attr_in_nodes(&mut mesh.children, group_id, attr, value) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn set_graph_group_attr(graph: &mut GraphScript, group_id: &str, attr: &str, value: &str) -> bool {
    for scene in &mut graph.scenes {
        if set_group_attr_in_nodes(&mut scene.children, group_id, attr, value) {
            return true;
        }
    }
    set_group_attr_in_nodes(&mut graph.scene_nodes, group_id, attr, value)
}

fn set_camera_pose_in_nodes(
    nodes: &mut [SceneNode],
    camera_id: &str,
    position: &str,
    target: &str,
) -> bool {
    for node in nodes {
        let children = match node {
            SceneNode::Timeline(node) => Some(node.children.as_mut_slice()),
            SceneNode::Track(node) => Some(node.children.as_mut_slice()),
            SceneNode::Sequence(node) => Some(node.children.as_mut_slice()),
            SceneNode::Chain(node) => Some(node.children.as_mut_slice()),
            SceneNode::Part(node) => Some(node.children.as_mut_slice()),
            SceneNode::Repeat(node) => Some(node.children.as_mut_slice()),
            SceneNode::Mask(node) => Some(node.children.as_mut_slice()),
            SceneNode::Precompose(node) => Some(node.children.as_mut_slice()),
            SceneNode::Layer(node) => Some(node.children.as_mut_slice()),
            SceneNode::Camera(node) => Some(node.children.as_mut_slice()),
            SceneNode::Character(node) => Some(node.children.as_mut_slice()),
            SceneNode::Puppet(node) => Some(node.children.as_mut_slice()),
            SceneNode::MeshTopology(node) => Some(node.children.as_mut_slice()),
            SceneNode::Group(group) => {
                if let Some(composite) = group.composite.as_mut()
                    && let Some(camera) =
                        composite.nodes_3d.iter_mut().find_map(|node| match node {
                            Scene3DNode::Camera(camera)
                                if camera.id.as_deref() == Some(camera_id) =>
                            {
                                Some(camera)
                            }
                            _ => None,
                        })
                {
                    camera.position = position.to_string();
                    camera.target = target.to_string();
                    return true;
                }
                Some(group.children.as_mut_slice())
            }
            _ => None,
        };
        if let Some(children) = children
            && set_camera_pose_in_nodes(children, camera_id, position, target)
        {
            return true;
        }
    }
    false
}

fn set_graph_camera_pose(
    graph: &mut GraphScript,
    camera_id: &str,
    position: &str,
    target: &str,
) -> bool {
    for scene in &mut graph.scenes {
        if set_camera_pose_in_nodes(&mut scene.children, camera_id, position, target) {
            return true;
        }
    }
    set_camera_pose_in_nodes(&mut graph.scene_nodes, camera_id, position, target)
}

fn set_action_bone_channel(
    graph: &mut GraphScript,
    action_id: &str,
    pose_ms: u32,
    bone_id: &str,
    channel: &str,
    value: &str,
) -> bool {
    let Some(action) = graph
        .actions
        .iter_mut()
        .find(|action| action.id == action_id)
    else {
        return false;
    };
    let pose_time = pose_ms as f32 / 1_000.0;
    let Some(pose) = action
        .poses
        .iter_mut()
        .find(|pose| (pose.t - pose_time).abs() < 0.000_51)
    else {
        return false;
    };
    let Some(bone) = pose.bones.iter_mut().find(|bone| bone.id == bone_id) else {
        return false;
    };
    let slot = match channel {
        "x" => &mut bone.x,
        "y" => &mut bone.y,
        "z" => &mut bone.z,
        "rotation" => &mut bone.rotation,
        "rotationX" | "rotation_x" => &mut bone.rotation_x,
        "rotationY" | "rotation_y" => &mut bone.rotation_y,
        "rotationZ" | "rotation_z" => &mut bone.rotation_z,
        "forward" => &mut bone.forward,
        "side" => &mut bone.side,
        "twist" => &mut bone.twist,
        "bend" => &mut bone.bend,
        "turn" => &mut bone.turn,
        "scale" => &mut bone.scale,
        "opacity" => &mut bone.opacity,
        _ => return false,
    };
    *slot = Some(value.to_string());
    graph.raw_script = None;
    true
}

/// Parse a MotionLoom script and return a short diagnostic summary.
///
/// Returns an error string if parsing fails.
#[wasm_bindgen]
pub fn motionloom_parse_summary(script: &str) -> Result<String, JsValue> {
    if is_graph_script(script) {
        let graph = parse_graph_script(script).map_err(|err| js_error(err.to_string()))?;
        let frame_count =
            ((graph.duration_ms as f64 / 1000.0) * graph.fps.max(1.0) as f64).round() as u64;
        return Ok(format!(
            "scene graph: {} scene(s), {} frame(s)",
            graph.scenes.len(),
            frame_count
        ));
    }
    if is_world_graph_script(script) {
        let graph = parse_world_graph_script(script).map_err(|err| js_error(err.to_string()))?;
        return Ok(format!(
            "world graph: {} world node(s), {} frame(s)",
            graph.worlds.len(),
            graph.duration_ms
        ));
    }
    Err(js_error(
        "script does not look like a scene or world graph".to_string(),
    ))
}

/// Return the same AnimationTarget capability registry used by native editors.
#[wasm_bindgen]
pub fn motionloom_animation_property_schema_json() -> String {
    animation_property_schema_json()
}

/// Return the complete machine-readable MotionLoom DSL capability catalog.
#[wasm_bindgen]
pub fn motionloom_dsl_schema_json() -> String {
    dsl_schema_json()
}

/// Analyze one DSL revision and return parse, semantic, compatibility, and repair diagnostics.
#[wasm_bindgen]
pub fn motionloom_analyze_script_json(script: &str) -> String {
    analyze_script_json(script)
}

/// Analyze one DSL revision for a concrete renderer such as `wasm-webgpu`.
#[wasm_bindgen]
pub fn motionloom_analyze_script_for_target_json(script: &str, target: &str) -> String {
    analyze_script_for_target_json(script, target)
}

/// Return the machine-readable syntax slice demonstrated by one showcase script.
#[wasm_bindgen]
pub fn motionloom_showcase_schema_json(script: &str) -> String {
    showcase_schema_json(script)
}

/// Analyze host/backend observations without parsing or changing MotionLoom DSL.
/// Empty options select the cinematic defaults.
#[wasm_bindgen]
pub fn motionloom_analyze_shot_observations_json(
    options_json: &str,
    observations_json: &str,
) -> Result<String, JsValue> {
    let options = if options_json.trim().is_empty() {
        crate::ShotValidationOptions::default()
    } else {
        serde_json::from_str(options_json).map_err(|err| js_error(err.to_string()))?
    };
    let observations: Vec<crate::ShotValidationFrameObservation> =
        serde_json::from_str(observations_json).map_err(|err| js_error(err.to_string()))?;
    crate::analyze_shot_observations(options, observations)
        .to_json()
        .map_err(|err| js_error(err.to_string()))
}

/// Inspect GLB bytes and propose humanoid mapping, axes, rest pose, and confidence.
#[wasm_bindgen]
pub fn motionloom_inspect_glb_skeleton_json(
    asset_label: &str,
    bytes: &[u8],
) -> Result<String, JsValue> {
    inspect_glb_skeleton_json(bytes, asset_label).map_err(|err| js_error(err.to_string()))
}

/// Detect declared/known humanoid rigs and propose a compatible profile.
/// This additive API preserves the legacy skeleton-inspection JSON contract.
#[wasm_bindgen]
pub fn motionloom_inspect_glb_humanoid_profile_json(
    asset_label: &str,
    bytes: &[u8],
) -> Result<String, JsValue> {
    inspect_glb_humanoid_profile_json(bytes, asset_label).map_err(|err| js_error(err.to_string()))
}

#[wasm_bindgen]
pub fn motionloom_inspect_glb_environment_json(
    asset_label: &str,
    bytes: &[u8],
) -> Result<String, JsValue> {
    inspect_glb_environment_json(bytes, asset_label).map_err(|err| js_error(err.to_string()))
}

/// Return structured AnimationTarget binding diagnostics for one graph script.
#[wasm_bindgen]
pub fn motionloom_inspect_animation_targets(script: &str) -> Result<String, JsValue> {
    let graph = parse_graph_script(script).map_err(|err| js_error(err.to_string()))?;
    serde_json::to_string_pretty(&inspect_animation_targets(&graph))
        .map_err(|err| js_error(err.to_string()))
}

/// Return the typed Action authoring document used by browser editors.
#[wasm_bindgen]
pub fn motionloom_editable_actions_json(script: &str) -> Result<String, JsValue> {
    let document =
        extract_editable_action_document(script).map_err(|err| js_error(err.to_string()))?;
    serde_json::to_string(&document).map_err(|err| js_error(err.to_string()))
}

/// Apply one JSON-encoded Action edit and return a validated DSL revision.
#[wasm_bindgen]
pub fn motionloom_apply_action_edit(script: &str, command_json: &str) -> Result<String, JsValue> {
    let command = serde_json::from_str::<ActionEditCommand>(command_json)
        .map_err(|err| js_error(format!("Invalid Action edit command: {err}")))?;
    apply_action_edit(script, command).map_err(|err| js_error(err.to_string()))
}

/// Render one frame of a scene graph script to an RGBA byte buffer.
///
/// The returned `Vec<u8>` is row-major RGBA with dimensions `(width, height)`.
/// Hosts can wrap it in `Uint8Array` / `ImageData`.
///
/// This convenience function uses the default path-based asset resolver and
/// tries the GPU profile, falling back to CPU if GPU initialization fails.
/// To supply in-memory assets use `WasmSceneRenderer`.
#[wasm_bindgen]
pub async fn motionloom_render_scene_frame(
    script: &str,
    frame: u32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsValue> {
    motionloom_render_scene_frame_with_profile(script, frame, width, height, "gpu-cpu").await
}

/// Render one frame with an explicit render profile.
///
/// `profile` accepts: `"cpu"`, `"gpu"`, `"gpu-cpu"` (try GPU, fallback to CPU).
#[wasm_bindgen]
pub async fn motionloom_render_scene_frame_with_profile(
    script: &str,
    frame: u32,
    width: u32,
    height: u32,
    profile: &str,
) -> Result<Vec<u8>, JsValue> {
    let mut graph = parse_graph_script(script).map_err(|err| js_error(err.to_string()))?;
    graph.size.0 = width.max(1);
    graph.size.1 = height.max(1);

    let (preferred, fallback) = parse_scene_profile_with_fallback(profile);
    let mut last_err = None;

    for profile in [Some(preferred), fallback].into_iter().flatten() {
        match render_scene_graph_frame(&graph, frame, profile).await {
            Ok(image) => return Ok(image.into_raw()),
            Err(err) => last_err = Some(err),
        }
    }

    Err(js_error(
        last_err
            .map(|err| err.to_string())
            .unwrap_or_else(|| "scene render failed".to_string()),
    ))
}

/// Render one scene frame directly into an HTML canvas using the WASM WebGPU path.
///
/// This is the first no-readback canvas path. It is strict: only GPU-native
/// scene graphs are accepted, and unsupported nodes return an error instead of
/// silently falling back to CPU.
#[wasm_bindgen]
pub async fn motionloom_render_scene_frame_to_canvas_gpu(
    script: &str,
    frame: u32,
    width: u32,
    height: u32,
    canvas: HtmlCanvasElement,
) -> Result<(), JsValue> {
    let mut graph = parse_graph_script(script).map_err(|err| js_error(err.to_string()))?;
    graph.size.0 = width.max(1);
    graph.size.1 = height.max(1);
    graph.render_size = Some((width.max(1), height.max(1)));
    let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu)
        .await
        .map_err(|err| js_error(err.to_string()))?;
    renderer
        .render_frame_to_canvas(&graph, frame, canvas)
        .await
        .map_err(|err| js_error(err.to_string()))
}

/// Draw a solid WebGPU color into an HTML canvas for debugging browser surface presentation.
#[wasm_bindgen]
pub async fn motionloom_webgpu_debug_solid_to_canvas(
    canvas: HtmlCanvasElement,
    width: u32,
    height: u32,
) -> Result<(), JsValue> {
    let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu)
        .await
        .map_err(|err| js_error(err.to_string()))?;
    renderer
        .debug_solid_to_canvas(canvas, width, height, [0.1, 0.85, 0.25, 1.0])
        .await
        .map_err(|err| js_error(err.to_string()))
}

/// Upload a blue WebGPU texture and present it to an HTML canvas for debugging.
#[wasm_bindgen]
pub async fn motionloom_webgpu_debug_uploaded_texture_to_canvas(
    canvas: HtmlCanvasElement,
    width: u32,
    height: u32,
) -> Result<(), JsValue> {
    let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu)
        .await
        .map_err(|err| js_error(err.to_string()))?;
    renderer
        .debug_uploaded_texture_to_canvas(canvas, width, height, [32, 96, 255, 255])
        .await
        .map_err(|err| js_error(err.to_string()))
}

/// Render a white empty scene texture and present it to an HTML canvas for debugging.
#[wasm_bindgen]
pub async fn motionloom_webgpu_debug_empty_scene_texture_to_canvas(
    canvas: HtmlCanvasElement,
    width: u32,
    height: u32,
) -> Result<(), JsValue> {
    let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu)
        .await
        .map_err(|err| js_error(err.to_string()))?;
    renderer
        .debug_empty_scene_texture_to_canvas(canvas, width, height)
        .await
        .map_err(|err| js_error(err.to_string()))
}

/// Render one frame of a process graph over an RGBA source buffer.
#[wasm_bindgen]
pub fn motionloom_render_process_frame(
    script: &str,
    frame: u32,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<Vec<u8>, JsValue> {
    render_process_frame_cpu(script, frame, width, height, rgba)
        .map(|image| image.into_raw())
        .map_err(|err| js_error(err.to_string()))
}

/// Render one frame of a process graph directly to an HTML canvas with WebGPU.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn motionloom_render_process_frame_to_canvas_gpu(
    script: &str,
    frame: u32,
    width: u32,
    height: u32,
    rgba: &[u8],
    canvas: HtmlCanvasElement,
) -> Result<(), JsValue> {
    render_process_frame_to_canvas_gpu_impl(script, frame, width, height, rgba, canvas)
        .await
        .map_err(|err| js_error(err.to_string()))
}

fn parse_scene_profile_with_fallback(
    profile: &str,
) -> (SceneRenderProfile, Option<SceneRenderProfile>) {
    match profile.to_ascii_lowercase().as_str() {
        "cpu" => (SceneRenderProfile::Cpu, None),
        "gpu" => (SceneRenderProfile::Gpu, None),
        "gpu-cpu" => (SceneRenderProfile::Gpu, Some(SceneRenderProfile::Cpu)),
        _ => (SceneRenderProfile::Gpu, Some(SceneRenderProfile::Cpu)),
    }
}

/// Render one frame through the legacy world compatibility path.
///
/// New DSL must use `<Scene>`; `<World>` is no longer a valid authoring tag.
///
/// This convenience function uses the default path-based asset resolver.
/// To supply in-memory assets use `WasmWorldRenderer`.
#[wasm_bindgen]
pub fn motionloom_render_world_frame(
    script: &str,
    frame: u32,
    asset_root: &str,
) -> Result<Vec<u8>, JsValue> {
    let graph = parse_world_graph_script(script).map_err(|err| js_error(err.to_string()))?;
    let mut renderer = WorldFrameRenderer::new();
    let image = renderer
        .render_frame(&graph, frame, asset_root)
        .map_err(|err| js_error(err.to_string()))?;
    Ok(image.into_raw())
}

/// Inspect a script and return the document type as a string.
#[wasm_bindgen]
pub fn motionloom_document_type(script: &str) -> String {
    if is_graph_script(script) {
        "scene".to_string()
    } else if is_world_graph_script(script) {
        "world".to_string()
    } else {
        "unknown".to_string()
    }
}

/// WASM-facing wrapper around a parsed scene graph. Keeps the parsed script
/// alive across JS calls so that repeated frame renders avoid re-parsing.
///
/// Each renderer owns its own `MemoryAssetResolver`; assets added to one
/// renderer do not affect any other renderer or the global state.
#[wasm_bindgen]
pub struct WasmSceneRenderer {
    graph: GraphScript,
    profile: SceneRenderProfile,
    resolver: Arc<MemoryAssetResolver>,
    renderer: Option<SceneRenderer>,
}

/// Compact browser-side environment metadata cache input. Hosts can inspect a
/// remote GLB once while preloading it, then provide the transformed bounds to
/// the renderer before the first presented frame needs semantic coordinates.
fn register_wasm_environment_bounds_asset(
    resolver: &MemoryAssetResolver,
    name: &str,
    bytes: &[u8],
) {
    let Ok(mesh) = crate::world::parse_glb_mesh_data(std::path::Path::new(name), bytes) else {
        return;
    };
    let matrices = crate::world::model_inspection::world_matrices(&mesh.nodes);
    let bounds = crate::world::model_inspection::transformed_environment_bounds(&mesh, &matrices)
        .unwrap_or((mesh.bounds_min, mesh.bounds_max));
    let mut payload = Vec::with_capacity(24);
    for value in bounds.0.into_iter().chain(bounds.1) {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    resolver.insert(format!("__motionloom_environment_bounds__:{name}"), payload);
}

#[wasm_bindgen]
impl WasmSceneRenderer {
    /// Parse `script` and prepare a renderer.
    #[wasm_bindgen(constructor)]
    pub fn new(script: &str, profile: &str) -> Result<WasmSceneRenderer, JsValue> {
        install_wasm_panic_hook();
        let graph = parse_graph_script(script).map_err(|err| js_error(err.to_string()))?;
        let profile = parse_profile(profile)?;
        Ok(Self {
            graph,
            profile,
            resolver: Arc::new(MemoryAssetResolver::new()),
            renderer: None,
        })
    }

    /// Asynchronously parse `script` and initialize the persistent renderer.
    ///
    /// Browser hosts should prefer this factory for animated GPU preview loops
    /// because repeated frame renders reuse the same Rust/WGPU renderer.
    pub async fn create(script: &str, profile: &str) -> Result<WasmSceneRenderer, JsValue> {
        install_wasm_panic_hook();
        let graph = parse_graph_script(script).map_err(|err| js_error(err.to_string()))?;
        let profile = parse_profile(profile)?;
        let resolver = Arc::new(MemoryAssetResolver::new());
        let renderer = SceneRenderer::with_resolver(profile, resolver.clone())
            .await
            .map_err(|err| js_error(err.to_string()))?;
        Ok(Self {
            graph,
            profile,
            resolver,
            renderer: Some(renderer),
        })
    }

    /// Register an in-memory asset for this renderer only.
    ///
    /// The `name` should match the `src` attribute used in `<Image>` or `<Svg>`
    /// nodes (e.g. `"logo.png"`). The `bytes` argument is the raw file content.
    pub fn add_asset(&mut self, name: &str, bytes: &[u8]) {
        self.resolver.insert(name.to_string(), bytes.to_vec());
    }

    /// Register transformed GLB bounds alongside an already preloaded asset.
    /// This is optional and non-breaking; renderers fall back to inspecting
    /// the asset bytes when the hint is absent.
    pub fn add_environment_bounds(&mut self, name: &str, bytes: &[u8]) {
        register_wasm_environment_bounds_asset(self.resolver.as_ref(), name, bytes);
    }

    /// Register an in-memory font for this renderer only.
    ///
    /// Browser hosts should use this for CJK or brand-specific text because
    /// WASM cannot discover OS fonts and browser CSS fonts are not visible to
    /// the Rust text rasterizer.
    pub async fn add_font(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        if self.renderer.is_none() {
            self.renderer = Some(
                SceneRenderer::with_resolver(self.profile, self.resolver.clone())
                    .await
                    .map_err(|err| js_error(err.to_string()))?,
            );
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| js_error("scene renderer was not initialized".to_string()))?;
        renderer.load_font_data(bytes.to_vec());
        Ok(())
    }

    /// Clear all assets previously registered on this renderer.
    pub fn clear_assets(&mut self) {
        self.resolver.clear();
    }

    /// Update a numeric `<Group id="...">` attribute without reparsing the DSL.
    ///
    /// This is intended for editor scrubbing (x/y/rotation/scale/opacity). It
    /// keeps the persistent renderer and its vector/GPU caches alive.
    pub fn set_group_attr(
        &mut self,
        group_id: &str,
        attr: &str,
        value: &str,
    ) -> Result<bool, JsValue> {
        if group_id.trim().is_empty() {
            return Err(js_error("group id is required".to_string()));
        }
        let value_num = value
            .trim()
            .parse::<f32>()
            .map_err(|_| js_error(format!("group attr value must be numeric: {value}")))?;
        if !value_num.is_finite() {
            return Err(js_error(format!(
                "group attr value must be finite: {value}"
            )));
        }
        let updated = set_graph_group_attr(&mut self.graph, group_id, attr, value.trim());
        if updated && let Some(renderer) = self.renderer.as_mut() {
            renderer.invalidate_runtime_scene_transforms();
        }
        Ok(updated)
    }

    /// Update one Camera3D pose in the parsed graph without recreating GPU
    /// pipelines, GLB meshes, textures, or the scene renderer.
    pub fn set_camera3d_pose(
        &mut self,
        camera_id: &str,
        position: &str,
        target: &str,
    ) -> Result<bool, JsValue> {
        if camera_id.trim().is_empty() || position.trim().is_empty() || target.trim().is_empty() {
            return Err(js_error(
                "camera id, position, and target are required".to_string(),
            ));
        }
        let updated =
            set_graph_camera_pose(&mut self.graph, camera_id, position.trim(), target.trim());
        if updated && let Some(renderer) = self.renderer.as_mut() {
            renderer.invalidate_runtime_scene_transforms();
        }
        Ok(updated)
    }

    /// Update one authored Action channel without reconstructing the renderer.
    ///
    /// The source editor remains authoritative: hosts use this method while a
    /// pointer is moving, then commit the same value through the typed Action
    /// edit API when the gesture ends. GLB bytes and GPU mesh caches stay live.
    pub fn set_action_pose_channel(
        &mut self,
        action_id: &str,
        pose_ms: u32,
        bone_id: &str,
        channel: &str,
        value: &str,
    ) -> Result<bool, JsValue> {
        let value_num = value
            .trim()
            .parse::<f32>()
            .map_err(|_| js_error(format!("Action channel value must be numeric: {value}")))?;
        if !value_num.is_finite() {
            return Err(js_error(format!(
                "Action channel value must be finite: {value}"
            )));
        }
        let updated = set_action_bone_channel(
            &mut self.graph,
            action_id,
            pose_ms,
            bone_id,
            channel,
            value.trim(),
        );
        if updated && let Some(renderer) = self.renderer.as_mut() {
            renderer.invalidate_runtime_scene_transforms();
        }
        Ok(updated)
    }

    /// Return true screen-space joints for the most recently rendered frame.
    /// Coordinates use the renderer's authored pixel size and include finger
    /// bones when the active ModelProfile maps them.
    pub fn editor_rig_snapshot_json(&self) -> Result<String, JsValue> {
        let snapshot = self
            .renderer
            .as_ref()
            .and_then(SceneRenderer::last_editor_rig_snapshot);
        serde_json::to_string(&snapshot).map_err(|err| js_error(err.to_string()))
    }

    /// Evaluate one actor through the exact browser Scene renderer and return
    /// the stable, versioned rig report as JSON.
    pub async fn evaluate_rig_frame_json(
        &mut self,
        actor_id: &str,
        frame: u32,
    ) -> Result<String, JsValue> {
        if self.renderer.is_none() {
            self.renderer = Some(
                SceneRenderer::with_resolver(self.profile, self.resolver.clone())
                    .await
                    .map_err(|err| js_error(err.to_string()))?,
            );
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| js_error("scene renderer was not initialized".to_string()))?;
        let report = renderer
            .evaluate_rig_frame(&self.graph, actor_id, frame)
            .await
            .map_err(|err| js_error(err.to_string()))?;
        serde_json::to_string_pretty(&report).map_err(|err| js_error(err.to_string()))
    }

    /// Evaluate a frame, time, or Action phase from a serialized
    /// `RigEvaluationRequest` and return the versioned report JSON.
    pub async fn evaluate_rig_json(&mut self, request_json: &str) -> Result<String, JsValue> {
        let request: crate::RigEvaluationRequest = serde_json::from_str(request_json)
            .map_err(|error| js_error(format!("invalid rig evaluation request: {error}")))?;
        if self.renderer.is_none() {
            self.renderer = Some(
                SceneRenderer::with_resolver(self.profile, self.resolver.clone())
                    .await
                    .map_err(|err| js_error(err.to_string()))?,
            );
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| js_error("scene renderer was not initialized".to_string()))?;
        let report = renderer
            .evaluate_rig(&self.graph, &request)
            .await
            .map_err(|err| js_error(err.to_string()))?;
        serde_json::to_string_pretty(&report).map_err(|err| js_error(err.to_string()))
    }

    /// Render `frame` to an RGBA byte buffer.
    pub async fn render_frame(&mut self, frame: u32) -> Result<Vec<u8>, JsValue> {
        if self.renderer.is_none() {
            self.renderer = Some(
                SceneRenderer::with_resolver(self.profile, self.resolver.clone())
                    .await
                    .map_err(|err| js_error(err.to_string()))?,
            );
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| js_error("scene renderer was not initialized".to_string()))?;
        let image = renderer
            .render_frame(&self.graph, frame)
            .await
            .map_err(|err| js_error(err.to_string()))?;
        Ok(image.into_raw())
    }

    /// Render a sampled range and return a machine-readable shot validation
    /// report. Optional editor/physics observations use the same JSON shape as
    /// `motionloom_analyze_shot_observations_json`.
    pub async fn validate_shots_json(
        &mut self,
        options_json: &str,
        observations_json: &str,
    ) -> Result<String, JsValue> {
        let options = if options_json.trim().is_empty() {
            crate::ShotValidationOptions::default()
        } else {
            serde_json::from_str(options_json).map_err(|err| js_error(err.to_string()))?
        };
        let observations: Vec<crate::ShotValidationFrameObservation> =
            if observations_json.trim().is_empty() {
                Vec::new()
            } else {
                serde_json::from_str(observations_json).map_err(|err| js_error(err.to_string()))?
            };
        if self.renderer.is_none() {
            self.renderer = Some(
                SceneRenderer::with_resolver(self.profile, self.resolver.clone())
                    .await
                    .map_err(|err| js_error(err.to_string()))?,
            );
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| js_error("scene renderer was not initialized".to_string()))?;
        renderer
            .validate_shots_with_observations(&self.graph, options, &observations)
            .await
            .map_err(|err| js_error(err.to_string()))?
            .to_json()
            .map_err(|err| js_error(err.to_string()))
    }

    /// Render `frame` directly into an HTML canvas using the GPU canvas path.
    ///
    /// The renderer profile must be `"gpu"`. CPU profiles continue to use
    /// `render_frame`, which returns RGBA bytes for Canvas2D/ImageData hosts.
    pub async fn render_frame_to_canvas(
        &mut self,
        frame: u32,
        canvas: HtmlCanvasElement,
    ) -> Result<(), JsValue> {
        if self.profile != SceneRenderProfile::Gpu {
            return Err(js_error(
                "render_frame_to_canvas requires a gpu WasmSceneRenderer".to_string(),
            ));
        }
        if self.renderer.is_none() {
            self.renderer = Some(
                SceneRenderer::with_resolver(self.profile, self.resolver.clone())
                    .await
                    .map_err(|err| js_error(err.to_string()))?,
            );
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| js_error("scene renderer was not initialized".to_string()))?;
        renderer
            .render_frame_to_canvas(&self.graph, frame, canvas)
            .await
            .map_err(|err| js_error(err.to_string()))
    }

    /// Draw a solid WebGPU color into the canvas using this renderer's GPU device.
    pub async fn debug_solid_to_canvas(
        &mut self,
        canvas: HtmlCanvasElement,
        width: u32,
        height: u32,
    ) -> Result<(), JsValue> {
        if self.profile != SceneRenderProfile::Gpu {
            return Err(js_error(
                "debug_solid_to_canvas requires a gpu WasmSceneRenderer".to_string(),
            ));
        }
        if self.renderer.is_none() {
            self.renderer = Some(
                SceneRenderer::with_resolver(self.profile, self.resolver.clone())
                    .await
                    .map_err(|err| js_error(err.to_string()))?,
            );
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| js_error("scene renderer was not initialized".to_string()))?;
        renderer
            .debug_solid_to_canvas(canvas, width, height, [0.1, 0.85, 0.25, 1.0])
            .await
            .map_err(|err| js_error(err.to_string()))
    }

    /// Upload a blue WebGPU texture and present it to the canvas.
    pub async fn debug_uploaded_texture_to_canvas(
        &mut self,
        canvas: HtmlCanvasElement,
        width: u32,
        height: u32,
    ) -> Result<(), JsValue> {
        if self.profile != SceneRenderProfile::Gpu {
            return Err(js_error(
                "debug_uploaded_texture_to_canvas requires a gpu WasmSceneRenderer".to_string(),
            ));
        }
        if self.renderer.is_none() {
            self.renderer = Some(
                SceneRenderer::with_resolver(self.profile, self.resolver.clone())
                    .await
                    .map_err(|err| js_error(err.to_string()))?,
            );
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| js_error("scene renderer was not initialized".to_string()))?;
        renderer
            .debug_uploaded_texture_to_canvas(canvas, width, height, [32, 96, 255, 255])
            .await
            .map_err(|err| js_error(err.to_string()))
    }

    /// Render a white empty scene texture and present it to the canvas.
    pub async fn debug_empty_scene_texture_to_canvas(
        &mut self,
        canvas: HtmlCanvasElement,
        width: u32,
        height: u32,
    ) -> Result<(), JsValue> {
        if self.profile != SceneRenderProfile::Gpu {
            return Err(js_error(
                "debug_empty_scene_texture_to_canvas requires a gpu WasmSceneRenderer".to_string(),
            ));
        }
        if self.renderer.is_none() {
            self.renderer = Some(
                SceneRenderer::with_resolver(self.profile, self.resolver.clone())
                    .await
                    .map_err(|err| js_error(err.to_string()))?,
            );
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| js_error("scene renderer was not initialized".to_string()))?;
        renderer
            .debug_empty_scene_texture_to_canvas(canvas, width, height)
            .await
            .map_err(|err| js_error(err.to_string()))
    }

    /// Total number of frames for the graph's duration and fps.
    #[wasm_bindgen(getter)]
    pub fn total_frames(&self) -> u32 {
        let fps = self.graph.fps.max(1.0);
        let duration_sec = (self.graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
        (duration_sec * fps).round() as u32
    }
}

/// WASM-facing wrapper for the legacy world compatibility renderer.
///
/// New DSL must use `<Scene>`; `<World>` is no longer a valid authoring tag.
#[wasm_bindgen]
pub struct WasmWorldRenderer {
    graph: crate::world::WorldGraph,
    resolver: Arc<MemoryAssetResolver>,
}

#[wasm_bindgen]
impl WasmWorldRenderer {
    /// Parse `script` and prepare a renderer.
    #[wasm_bindgen(constructor)]
    pub fn new(script: &str) -> Result<WasmWorldRenderer, JsValue> {
        let graph = parse_world_graph_script(script).map_err(|err| js_error(err.to_string()))?;
        Ok(Self {
            graph,
            resolver: Arc::new(MemoryAssetResolver::new()),
        })
    }

    /// Register an in-memory asset for this renderer only.
    pub fn add_asset(&mut self, name: &str, bytes: &[u8]) {
        self.resolver.insert(name.to_string(), bytes.to_vec());
    }

    /// Clear all assets previously registered on this renderer.
    pub fn clear_assets(&mut self) {
        self.resolver.clear();
    }

    /// Render `frame` to an RGBA byte buffer using the provided asset root for
    /// relative-path fallback.
    pub fn render_frame(&mut self, frame: u32, asset_root: &str) -> Result<Vec<u8>, JsValue> {
        let mut renderer = WorldFrameRenderer::with_resolver(self.resolver.clone());
        let image = renderer
            .render_frame(&self.graph, frame, asset_root)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(image.into_raw())
    }
}

fn parse_profile(profile: &str) -> Result<SceneRenderProfile, JsValue> {
    match profile.to_ascii_lowercase().as_str() {
        "cpu" => Ok(SceneRenderProfile::Cpu),
        "gpu" => Ok(SceneRenderProfile::Gpu),
        _ => Err(js_error(format!("unknown render profile: {profile}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_action_channel_mutates_existing_pose_without_reparse() {
        let source = r#"<Graph fps={30} duration="1s" size={[64,64]}>
  <Action id="wave" skeleton="humanoid_v1" duration="1s">
    <Pose t="0.5s"><Bone id="upper_arm_r" forward="10" /></Pose>
  </Action>
</Graph>"#;
        let mut graph = parse_graph_script(source).expect("Action fixture parses");
        assert!(set_action_bone_channel(
            &mut graph,
            "wave",
            500,
            "upper_arm_r",
            "forward",
            "42"
        ));
        assert_eq!(
            graph.actions[0].poses[0].bones[0].forward.as_deref(),
            Some("42")
        );
        assert!(graph.raw_script.is_none());
    }

    #[test]
    fn runtime_action_channel_rejects_unknown_channel() {
        let source = r#"<Graph fps={30} duration="1s" size={[64,64]}>
  <Action id="idle" skeleton="humanoid_v1" duration="1s">
    <Pose t="0s"><Bone id="hips" turn="0" /></Pose>
  </Action>
</Graph>"#;
        let mut graph = parse_graph_script(source).expect("Action fixture parses");
        assert!(!set_action_bone_channel(
            &mut graph,
            "idle",
            0,
            "hips",
            "notAChannel",
            "1"
        ));
    }

    #[test]
    fn runtime_camera_pose_mutates_composite_camera_without_reparse() {
        let source = r#"<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="scene"><Timeline><Track id="track" space="3d"><Sequence from="0s" duration="1s">
    <CompositeGroup id="world" space="3d"><Camera3D id="editor_camera" position={[0,1,4]} target={[0,1,0]} /></CompositeGroup>
  </Sequence></Track></Timeline></Scene><Present from="scene" />
</Graph>"#;
        let mut graph = parse_graph_script(source).expect("Camera fixture parses");
        assert!(set_graph_camera_pose(
            &mut graph,
            "editor_camera",
            "[1,2,3]",
            "[0,1,0]"
        ));
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("timeline")
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("track")
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("sequence")
        };
        let SceneNode::Group(group) = &sequence.children[0] else {
            panic!("group")
        };
        let Scene3DNode::Camera(camera) = &group.composite.as_ref().unwrap().nodes_3d[0] else {
            panic!("camera")
        };
        assert_eq!(camera.position, "[1,2,3]");
        assert_eq!(camera.target, "[0,1,0]");
    }
}
