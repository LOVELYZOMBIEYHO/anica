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
use crate::scene::model::{GroupNode, SceneNode};
use crate::scene::render::{SceneRenderProfile, SceneRenderer, render_scene_graph_frame};
use crate::world::{
    WorldFrameRenderer, inspect_glb_environment_json, inspect_glb_skeleton_json,
    is_world_graph_script, parse_world_graph_script,
};

fn js_error(message: String) -> JsValue {
    js_sys::Error::new(&message).into()
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

/// Inspect GLB bytes and propose humanoid mapping, axes, rest pose, and confidence.
#[wasm_bindgen]
pub fn motionloom_inspect_glb_skeleton_json(
    asset_label: &str,
    bytes: &[u8],
) -> Result<String, JsValue> {
    inspect_glb_skeleton_json(bytes, asset_label).map_err(|err| js_error(err.to_string()))
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
