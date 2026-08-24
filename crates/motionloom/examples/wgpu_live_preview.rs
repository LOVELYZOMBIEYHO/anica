// =========================================
// =========================================
// crates/motionloom/examples/wgpu_live_preview.rs

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[path = "wgpu_live_preview/preview_host_platform.rs"]
mod preview_host_platform;

use motionloom::{
    PREVIEW_PROTOCOL_VERSION, PreviewCommand, PreviewEvent, PreviewInteractionMode,
    PreviewInteractionNode, Scene3DFrameProfile, SceneCpuFrameProfile,
    WgpuPreviewAdaptiveController, WgpuPreviewEngine, WgpuPreviewPreloadSession,
    WgpuPreviewQuality, parse_graph_script,
};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalSize};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId, WindowLevel};

#[cfg(target_os = "macos")]
use objc2_app_kit::{NSView, NSWindowCollectionBehavior};
#[cfg(target_os = "macos")]
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
#[cfg(target_os = "macos")]
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

const BLIT_SHADER: &str = r#"
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 0.0),
    );

    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(scene_tex, scene_sampler, in.uv);
}
"#;

fn aspect_fit_viewport(
    surface_width: u32,
    surface_height: u32,
    content_width: u32,
    content_height: u32,
) -> (f32, f32, f32, f32) {
    let surface_width = surface_width.max(1) as f32;
    let surface_height = surface_height.max(1) as f32;
    let content_width = content_width.max(1) as f32;
    let content_height = content_height.max(1) as f32;
    let scale = (surface_width / content_width).min(surface_height / content_height);
    let width = content_width * scale;
    let height = content_height * scale;
    (
        (surface_width - width) * 0.5,
        (surface_height - height) * 0.5,
        width,
        height,
    )
}

const OVERLAY_SHADER: &str = r#"
struct OverlayUniforms {
    surface_size: vec2<f32>,
    active_mode: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> overlay: OverlayUniforms;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) instance: u32,
    @location(1) local: vec2<f32>,
};

fn button_rect(instance: u32) -> vec4<f32> {
    if (instance == 0u) {
        return vec4<f32>(12.0, 12.0, 80.0, 28.0);
    }
    if (instance == 1u) {
        return vec4<f32>(100.0, 12.0, 112.0, 28.0);
    }
    if (instance == 2u) {
        return vec4<f32>(220.0, 12.0, 76.0, 28.0);
    }
    if (instance == 3u) {
        return vec4<f32>(304.0, 12.0, 118.0, 28.0);
    }
    return vec4<f32>(430.0, 12.0, 118.0, 28.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance: u32) -> VertexOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );
    let local = corners[vertex_index];
    let rect = button_rect(instance);
    let pixel = rect.xy + local * rect.zw;
    let ndc = vec2<f32>(
        pixel.x / max(overlay.surface_size.x, 1.0) * 2.0 - 1.0,
        1.0 - pixel.y / max(overlay.surface_size.y, 1.0) * 2.0,
    );

    var out: VertexOut;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.instance = instance;
    out.local = local;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let border = in.local.x < 0.04 || in.local.x > 0.96 || in.local.y < 0.10 || in.local.y > 0.90;
    if (in.instance == overlay.active_mode) {
        if (border) {
            return vec4<f32>(0.70, 0.88, 1.0, 0.95);
        }
        return vec4<f32>(0.12, 0.36, 0.60, 0.78);
    }
    if (border) {
        return vec4<f32>(0.42, 0.46, 0.52, 0.86);
    }
    return vec4<f32>(0.05, 0.06, 0.08, 0.72);
}
"#;

const PICK_SHADER: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) shape_kind: f32,
};

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) local: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) shape_kind: f32,
) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.local = local;
    out.color = color;
    out.shape_kind = shape_kind;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if (in.shape_kind > 0.5) {
        let centered = in.local * 2.0 - vec2<f32>(1.0, 1.0);
        if (dot(centered, centered) > 1.0) {
            discard;
        }
    }
    return in.color;
}
"#;

const WAITING_PLACEHOLDER_SCRIPT: &str = r##"
<Graph fps={30} duration="1s" size={[1280,720]} renderSize={[1280,720]}>
  <Background color="#0B1020" />
  <Scene id="waiting_for_anica">
    <Timeline>
      <Track id="main" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="0" y="0" width="1280" height="720" color="#0B1020" />
            <Text value="MotionLoom Preview Host" x="center" y="320" fontSize="52" color="#DDE8FF" />
            <Text value="listening for Anica code block DSL..." x="center" y="382" fontSize="28" color="#7C8AA5" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="waiting_for_anica" />
</Graph>
"##;

const HOST_ATTACH_HEARTBEAT_TIMEOUT: Duration = Duration::from_millis(1500);

fn preview_host_debug_enabled() -> bool {
    std::env::var("MOTIONLOOM_PREVIEW_HOST_DEBUG").is_ok_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    })
}

fn preview_host_debug_log(message: impl AsRef<str>) {
    if preview_host_debug_enabled() {
        eprintln!(
            "[wgpu-preview pid={}] {}",
            std::process::id(),
            message.as_ref()
        );
    }
}

fn preview_feature_enabled(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |value| {
        let value = value.trim();
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    })
}

#[derive(Clone, Debug)]
enum PreviewHostUserEvent {
    Command(PreviewCommand),
}

#[derive(Clone, Default)]
struct PreviewEventBroadcaster {
    clients: Arc<Mutex<Vec<TcpStream>>>,
}

impl PreviewEventBroadcaster {
    fn add_client(&self, stream: TcpStream) {
        let Ok(mut clients) = self.clients.lock() else {
            return;
        };
        clients.push(stream);
    }

    fn broadcast(&self, event: PreviewEvent) {
        let Ok(payload) = serde_json::to_string(&event) else {
            return;
        };
        let line = format!("{payload}\n");
        let Ok(mut clients) = self.clients.lock() else {
            return;
        };
        clients.retain_mut(|client| client.write_all(line.as_bytes()).is_ok());
    }
}

struct LivePreviewApp {
    script_source: String,
    base_script: String,
    base_graph: motionloom::GraphScript,
    graph: Option<motionloom::GraphScript>,
    window: Option<Arc<Window>>,
    instance: Option<wgpu::Instance>,
    surface: Option<wgpu::Surface<'static>>,
    adapter: Option<wgpu::Adapter>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<wgpu::Queue>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    surface_format: Option<wgpu::TextureFormat>,
    preview_engine: Option<WgpuPreviewEngine>,
    target_width: u32,
    target_height: u32,
    sampler: Option<wgpu::Sampler>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    pipeline: Option<wgpu::RenderPipeline>,
    overlay_buffer: Option<wgpu::Buffer>,
    overlay_bind_group: Option<wgpu::BindGroup>,
    overlay_pipeline: Option<wgpu::RenderPipeline>,
    picking_pipeline: Option<wgpu::RenderPipeline>,
    quality: WgpuPreviewQuality,
    adaptive_quality: WgpuPreviewAdaptiveController,
    pending_preload: bool,
    preload_session: Option<WgpuPreviewPreloadSession>,
    last_cursor_pos: Option<(f64, f64)>,
    frame: u32,
    total_frames: u32,
    playback_started_at: Instant,
    playback_start_frame: u32,
    next_redraw_at: Instant,
    last_title_at: Instant,
    last_stats_at: Instant,
    print_stats_enabled: bool,
    auto_advance: bool,
    host_mode: bool,
    host_events: Option<PreviewEventBroadcaster>,
    controller_process_id: Option<u32>,
    overrides: HashMap<(String, String), f32>,
    last_render_ms: f32,
    last_gpu_frame_ms: Option<f64>,
    last_cpu_profile: SceneCpuFrameProfile,
    last_3d_profile: Scene3DFrameProfile,
    last_present_ms: f32,
    render_times: Vec<f32>,
    present_times: Vec<f32>,
    requested_window_bounds: Option<PreviewWindowBounds>,
    last_attach_heartbeat: Option<Instant>,
    needs_redraw: bool,
    // Controller visibility is a hard gate; frame updates must not re-show a hidden host window.
    controller_window_visible_allowed: bool,
    window_visible: bool,
    interaction_mode: PreviewInteractionMode,
    interaction_graph_width: f32,
    interaction_graph_height: f32,
    interaction_targets: Vec<PreviewInteractionNode>,
    interaction_drag: Option<PreviewInteractionDrag>,
    mouse_left_down: bool,
    present_cache: Vec<PresentTextureCache>,
}

struct PresentTextureCache {
    texture: Arc<wgpu::Texture>,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy, Debug)]
struct PickVertex {
    position: [f32; 2],
    local: [f32; 2],
    color: [f32; 4],
    shape_kind: f32,
}

impl PickVertex {
    const STRIDE: wgpu::BufferAddress = 36;

    fn write_bytes(&self, out: &mut Vec<u8>) {
        for value in self
            .position
            .iter()
            .chain(self.local.iter())
            .chain(self.color.iter())
            .chain(std::iter::once(&self.shape_kind))
        {
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
}

#[derive(Clone, Debug)]
struct PreviewInteractionDrag {
    target: PreviewInteractionNode,
    mode: PreviewInteractionMode,
    graph_width: f32,
    graph_height: f32,
    start_cursor_x: f64,
    start_cursor_y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PreviewWindowBounds {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    decorations: bool,
}

impl LivePreviewApp {
    fn new(
        script_source: Option<String>,
        print_stats_enabled: bool,
        auto_advance: bool,
        host_mode: bool,
        host_events: Option<PreviewEventBroadcaster>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (script_source, script) = if let Some(script_source) = script_source {
            configure_scene_asset_root_for_source(&script_source);
            let script = load_script_source(&script_source)?;
            (script_source, script)
        } else {
            (
                "waiting-for-anica-code-block".to_string(),
                WAITING_PLACEHOLDER_SCRIPT.to_string(),
            )
        };
        let graph = parse_graph_script(&script)?;
        let fps = graph.fps.max(1.0);
        let total_frames =
            (((graph.duration_ms as f32 / 1000.0).max(1.0 / fps) * fps).round() as u32).max(1);

        Ok(Self {
            script_source,
            base_script: script,
            base_graph: graph.clone(),
            graph: Some(graph),
            window: None,
            instance: None,
            surface: None,
            adapter: None,
            device: None,
            queue: None,
            surface_config: None,
            surface_format: None,
            preview_engine: None,
            target_width: 0,
            target_height: 0,
            sampler: None,
            bind_group_layout: None,
            pipeline: None,
            overlay_buffer: None,
            overlay_bind_group: None,
            overlay_pipeline: None,
            picking_pipeline: None,
            quality: WgpuPreviewQuality::Full,
            adaptive_quality: {
                let mut adaptive = WgpuPreviewAdaptiveController::for_fps(fps);
                adaptive.set_enabled(
                    auto_advance && preview_feature_enabled("MOTIONLOOM_PREVIEW_ADAPTIVE", false),
                );
                adaptive
            },
            pending_preload: preview_feature_enabled("MOTIONLOOM_PREVIEW_PRELOAD", true),
            preload_session: None,
            last_cursor_pos: None,
            frame: 0,
            total_frames,
            playback_started_at: Instant::now(),
            playback_start_frame: 0,
            next_redraw_at: Instant::now(),
            last_title_at: Instant::now(),
            last_stats_at: Instant::now(),
            print_stats_enabled,
            auto_advance,
            host_mode,
            host_events,
            controller_process_id: None,
            overrides: HashMap::new(),
            last_render_ms: 0.0,
            last_gpu_frame_ms: None,
            last_cpu_profile: SceneCpuFrameProfile::default(),
            last_3d_profile: Scene3DFrameProfile::default(),
            last_present_ms: 0.0,
            render_times: Vec::with_capacity(240),
            present_times: Vec::with_capacity(240),
            requested_window_bounds: None,
            last_attach_heartbeat: None,
            needs_redraw: true,
            controller_window_visible_allowed: !host_mode,
            window_visible: !host_mode,
            interaction_mode: PreviewInteractionMode::Move,
            interaction_graph_width: 1280.0,
            interaction_graph_height: 720.0,
            interaction_targets: Vec::new(),
            interaction_drag: None,
            mouse_left_down: false,
            present_cache: Vec::with_capacity(2),
        })
    }

    fn rebuild_graph_for_quality(&mut self) {
        let base_graph = if self.overrides.is_empty() {
            self.base_graph.clone()
        } else {
            match self.graph_with_overrides() {
                Ok(graph) => graph,
                Err(err) => {
                    self.broadcast_event(PreviewEvent::Error { message: err });
                    self.base_graph.clone()
                }
            }
        };
        let graph = WgpuPreviewEngine::graph_for_quality(&base_graph, self.quality);
        let (target_width, target_height) = graph.render_size.unwrap_or(graph.size);
        self.graph = Some(graph);
        self.target_width = target_width.max(1);
        self.target_height = target_height.max(1);
        self.present_cache.clear();
        self.render_times.clear();
        self.present_times.clear();
        self.update_title_now();
    }

    fn graph_with_overrides(&self) -> Result<motionloom::GraphScript, String> {
        let mut script = self.base_script.clone();
        let mut overrides = self
            .overrides
            .iter()
            .map(|((node, property), value)| (node.clone(), property.clone(), *value))
            .collect::<Vec<_>>();
        overrides.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (node, property, value) in overrides {
            script = patch_scene_tag_attr_number(&script, &node, &property, value)?;
        }
        parse_graph_script(&script).map_err(|err| format!("line {}: {}", err.line, err.message))
    }

    fn set_quality(&mut self, quality: WgpuPreviewQuality) {
        self.adaptive_quality.lock_manual();
        self.set_quality_internal(quality);
    }

    fn set_quality_internal(&mut self, quality: WgpuPreviewQuality) {
        if self.quality == quality {
            return;
        }
        self.quality = quality;
        self.rebuild_graph_for_quality();
        self.request_redraw();
    }

    fn resume_adaptive_quality(&mut self) {
        self.adaptive_quality.resume_auto();
        self.request_redraw();
    }

    fn load_script_text(&mut self, script: String, source: Option<String>) -> Result<(), String> {
        if let Some(source) = source.as_deref() {
            configure_scene_asset_root_for_source(source);
        }
        let graph = parse_graph_script(&script)
            .map_err(|err| format!("line {}: {}", err.line, err.message))?;
        let fps = graph.fps.max(1.0);
        self.total_frames =
            (((graph.duration_ms as f32 / 1000.0).max(1.0 / fps) * fps).round() as u32).max(1);
        self.frame = self.frame.min(self.total_frames.saturating_sub(1));
        self.restart_playback_clock();
        self.script_source = source.unwrap_or_else(|| "preview-host-script".to_string());
        self.base_script = script;
        self.base_graph = graph;
        self.adaptive_quality.set_target_fps(fps);
        self.overrides.clear();
        self.pending_preload = preview_feature_enabled("MOTIONLOOM_PREVIEW_PRELOAD", true);
        self.preload_session = None;
        self.rebuild_graph_for_quality();
        Ok(())
    }

    fn handle_preview_command(&mut self, command: PreviewCommand) {
        preview_host_debug_log(format!("command: {command:?}"));
        match command {
            PreviewCommand::LoadScript { script, source } => {
                preview_host_debug_log(format!("LoadScript start bytes={}", script.len()));
                if let Err(message) = self.load_script_text(script, source) {
                    preview_host_debug_log(format!("LoadScript error: {message}"));
                    self.broadcast_event(PreviewEvent::Error { message });
                } else {
                    preview_host_debug_log("LoadScript ok");
                    self.request_redraw();
                }
            }
            PreviewCommand::SetFrame { frame } => {
                let frame = frame.min(self.total_frames.saturating_sub(1));
                self.keep_attached_visible();
                if self.frame != frame {
                    self.frame = frame;
                    self.restart_playback_clock();
                    self.request_redraw();
                }
            }
            PreviewCommand::SetQuality { quality } => self.set_quality(quality),
            PreviewCommand::SetOverride {
                node,
                property,
                value,
            } => {
                self.keep_attached_visible();
                self.overrides.insert((node, property), value);
                self.rebuild_graph_for_quality();
                self.request_redraw();
            }
            PreviewCommand::ClearOverride { node, property } => {
                self.keep_attached_visible();
                self.overrides.remove(&(node, property));
                self.rebuild_graph_for_quality();
                self.request_redraw();
            }
            PreviewCommand::SetAssetRoots { roots } => {
                if roots.is_empty() {
                    motionloom::clear_scene_asset_roots();
                } else {
                    let roots = roots.into_iter().map(std::path::PathBuf::from).collect();
                    motionloom::set_scene_asset_roots(roots);
                }
                self.request_redraw();
            }
            PreviewCommand::SetWindowBounds {
                x,
                y,
                width,
                height,
                decorations,
            } => {
                let changed = self.set_requested_window_bounds(PreviewWindowBounds {
                    x,
                    y,
                    width,
                    height,
                    decorations,
                });
                if changed {
                    self.request_redraw();
                }
            }
            PreviewCommand::SetWindowVisible { visible } => {
                self.controller_window_visible_allowed = visible;
                let visible = visible && self.frontmost_app_allows_visibility();
                let changed = self.window_visible != visible;
                self.window_visible = visible;
                if changed && let Some(window) = self.window.as_ref() {
                    window.set_visible(visible);
                }
                self.last_attach_heartbeat = visible.then(Instant::now);
                if visible && changed {
                    self.request_redraw();
                }
            }
            PreviewCommand::SetControllerProcessId { pid } => {
                self.controller_process_id = Some(pid);
            }
            PreviewCommand::SetInteractionTarget {
                node,
                mode,
                graph_width,
                graph_height,
                x,
                y,
                rotation,
            } => {
                self.keep_attached_visible();
                self.interaction_mode = mode;
                self.interaction_graph_width = graph_width.max(1.0);
                self.interaction_graph_height = graph_height.max(1.0);
                self.interaction_targets = vec![PreviewInteractionNode {
                    node,
                    tag: "selected".to_string(),
                    x,
                    y,
                    width: 160.0,
                    height: 100.0,
                    rotation,
                }];
            }
            PreviewCommand::SetInteractionTargets {
                mode,
                graph_width,
                graph_height,
                targets,
            } => {
                self.keep_attached_visible();
                self.interaction_mode = mode;
                self.interaction_graph_width = graph_width.max(1.0);
                self.interaction_graph_height = graph_height.max(1.0);
                self.interaction_targets = targets;
            }
        }
    }

    fn request_redraw(&mut self) {
        self.needs_redraw = true;
        self.next_redraw_at = self.next_redraw_at.min(Instant::now());
    }

    fn frame_interval(&self) -> Duration {
        let fps = self
            .graph
            .as_ref()
            .map(|graph| graph.fps)
            .unwrap_or(self.base_graph.fps)
            .clamp(1.0, 60.0);
        Duration::from_secs_f64(1.0 / fps as f64)
    }

    fn restart_playback_clock(&mut self) {
        self.playback_started_at = Instant::now();
        self.playback_start_frame = self.frame;
        self.next_redraw_at = Instant::now();
    }

    fn update_frame_from_wall_clock(&mut self) {
        if !self.auto_advance || self.total_frames <= 1 {
            return;
        }
        let fps = self
            .graph
            .as_ref()
            .map(|graph| graph.fps)
            .unwrap_or(self.base_graph.fps)
            .max(1.0);
        let elapsed_frames =
            (self.playback_started_at.elapsed().as_secs_f64() * fps as f64).floor() as u64;
        self.frame =
            ((self.playback_start_frame as u64 + elapsed_frames) % self.total_frames as u64) as u32;
    }

    fn keep_attached_visible(&mut self) {
        if !self.host_mode {
            return;
        }
        if !self.controller_window_visible_allowed {
            self.window_visible = false;
            self.last_attach_heartbeat = None;
            if let Some(window) = self.window.as_ref() {
                window.set_visible(false);
            }
            return;
        }
        if !self.frontmost_app_allows_visibility() {
            if preview_host_debug_enabled() {
                eprintln!("preview host hide: frontmost app is not controller/host");
            }
            self.window_visible = false;
            self.last_attach_heartbeat = None;
            if let Some(window) = self.window.as_ref() {
                window.set_visible(false);
            }
            return;
        }
        let changed = !self.window_visible;
        self.window_visible = true;
        self.last_attach_heartbeat = Some(Instant::now());
        if let Some(window) = self.window.as_ref() {
            if changed {
                window.set_visible(true);
            }
            if preview_host_platform::should_raise_attached_window_on_heartbeat() {
                window.set_window_level(WindowLevel::AlwaysOnTop);
            }
        }
        if preview_host_platform::should_reapply_bounds_on_heartbeat() {
            self.apply_requested_window_bounds();
        }
    }

    fn frontmost_app_allows_visibility(&self) -> bool {
        if !self.host_mode {
            return true;
        }
        if !preview_host_platform::should_gate_visibility_by_frontmost_process() {
            return true;
        }
        let Some(frontmost_pid) = preview_host_platform::frontmost_process_id() else {
            return true;
        };
        frontmost_pid == std::process::id() || Some(frontmost_pid) == self.controller_process_id
    }

    fn set_requested_window_bounds(&mut self, bounds: PreviewWindowBounds) -> bool {
        if bounds.width < 64.0 || bounds.height < 64.0 {
            return false;
        }
        if self.requested_window_bounds == Some(bounds) {
            return false;
        }
        self.requested_window_bounds = Some(bounds);
        self.last_attach_heartbeat = Some(Instant::now());
        self.apply_requested_window_bounds();
        true
    }

    fn apply_requested_window_bounds(&self) {
        let (Some(window), Some(bounds)) = (self.window.as_ref(), self.requested_window_bounds)
        else {
            return;
        };
        if preview_host_debug_enabled() {
            eprintln!("preview host apply bounds: {bounds:?}");
        }
        window.set_decorations(bounds.decorations);
        window.set_outer_position(LogicalPosition::new(bounds.x, bounds.y));
        let _ = window.request_inner_size(LogicalSize::new(bounds.width, bounds.height));
        let inner_size = window.inner_size();
        self.broadcast_event(PreviewEvent::WindowBounds {
            x: bounds.x,
            y: bounds.y,
            width: f64::from(inner_size.width),
            height: f64::from(inner_size.height),
        });
    }

    fn broadcast_event(&self, event: PreviewEvent) {
        if let Some(host_events) = self.host_events.as_ref() {
            host_events.broadcast(event);
        }
    }

    fn quality_button_at(position: (f64, f64)) -> Option<WgpuPreviewQuality> {
        let (x, y) = position;
        if !(12.0..=40.0).contains(&y) {
            return None;
        }
        if (12.0..=92.0).contains(&x) {
            return Some(WgpuPreviewQuality::Full);
        }
        if (100.0..=212.0).contains(&x) {
            return Some(WgpuPreviewQuality::Balanced);
        }
        if (220.0..=296.0).contains(&x) {
            return Some(WgpuPreviewQuality::Speed);
        }
        if (304.0..=422.0).contains(&x) {
            return Some(WgpuPreviewQuality::HighSpeed);
        }
        if (430.0..=548.0).contains(&x) {
            return Some(WgpuPreviewQuality::UltraSpeed);
        }
        None
    }

    fn current_graph_fit_scale(&self) -> f32 {
        let Some(window) = self.window.as_ref() else {
            return 1.0;
        };
        let inner = window.inner_size();
        let surface_w = inner.width.max(1) as f32;
        let surface_h = inner.height.max(1) as f32;
        (surface_w / self.interaction_graph_width)
            .min(surface_h / self.interaction_graph_height)
            .max(0.001)
    }

    fn cursor_to_graph_position(&self, position: (f64, f64)) -> (f32, f32) {
        let Some(window) = self.window.as_ref() else {
            return (position.0 as f32, position.1 as f32);
        };
        let inner = window.inner_size();
        let scale = self.current_graph_fit_scale();
        let image_w = self.interaction_graph_width * scale;
        let image_h = self.interaction_graph_height * scale;
        let image_left = ((inner.width as f32 - image_w) * 0.5).max(0.0);
        let image_top = ((inner.height as f32 - image_h) * 0.5).max(0.0);
        (
            ((position.0 as f32 - image_left) / scale).clamp(0.0, self.interaction_graph_width),
            ((position.1 as f32 - image_top) / scale).clamp(0.0, self.interaction_graph_height),
        )
    }

    fn cursor_to_graph_pixel(&self, position: (f64, f64)) -> Option<(u32, u32)> {
        let (graph_x, graph_y) = self.cursor_to_graph_position(position);
        if graph_x < 0.0
            || graph_y < 0.0
            || graph_x > self.interaction_graph_width
            || graph_y > self.interaction_graph_height
        {
            return None;
        }
        let target_scale_x = self.target_width as f32 / self.interaction_graph_width.max(1.0);
        let target_scale_y = self.target_height as f32 / self.interaction_graph_height.max(1.0);
        Some((
            (graph_x * target_scale_x)
                .round()
                .clamp(0.0, (self.target_width.saturating_sub(1)) as f32) as u32,
            (graph_y * target_scale_y)
                .round()
                .clamp(0.0, (self.target_height.saturating_sub(1)) as f32) as u32,
        ))
    }

    fn pick_target_graph_bounds(target: &PreviewInteractionNode) -> (f32, f32, f32, f32) {
        let width = target.width.abs().max(12.0);
        let height = target.height.abs().max(12.0);
        if matches!(target.tag.as_str(), "Circle" | "Ellipse") {
            (
                target.x - width * 0.5,
                target.y - height * 0.5,
                width,
                height,
            )
        } else {
            (target.x, target.y, width, height)
        }
    }

    fn pick_shape_kind(target: &PreviewInteractionNode) -> f32 {
        if matches!(target.tag.as_str(), "Circle" | "Ellipse") {
            1.0
        } else {
            0.0
        }
    }

    fn encode_pick_color(index: usize) -> [f32; 4] {
        let id = (index + 1).min(0x00FF_FFFF) as u32;
        [
            ((id & 0x0000_00FF) as f32) / 255.0,
            (((id >> 8) & 0x0000_00FF) as f32) / 255.0,
            (((id >> 16) & 0x0000_00FF) as f32) / 255.0,
            1.0,
        ]
    }

    fn decode_pick_color(bytes: &[u8]) -> Option<usize> {
        if bytes.len() < 3 {
            return None;
        }
        let id = u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
        (id > 0).then_some((id - 1) as usize)
    }

    fn push_pick_quad(
        vertices: &mut Vec<PickVertex>,
        surface_width: f32,
        surface_height: f32,
        rect: (f32, f32, f32, f32),
        color: [f32; 4],
        shape_kind: f32,
    ) {
        let (left, top, width, height) = rect;
        let right = left + width;
        let bottom = top + height;
        let to_ndc = |x: f32, y: f32| {
            [
                x / surface_width.max(1.0) * 2.0 - 1.0,
                1.0 - y / surface_height.max(1.0) * 2.0,
            ]
        };
        let corners = [
            (left, top, [0.0, 0.0]),
            (right, top, [1.0, 0.0]),
            (left, bottom, [0.0, 1.0]),
            (left, bottom, [0.0, 1.0]),
            (right, top, [1.0, 0.0]),
            (right, bottom, [1.0, 1.0]),
        ];
        for (x, y, local) in corners {
            vertices.push(PickVertex {
                position: to_ndc(x, y),
                local,
                color,
                shape_kind,
            });
        }
    }

    fn build_pick_vertices(&self, surface_width: u32, surface_height: u32) -> Vec<PickVertex> {
        let scale = self.current_graph_fit_scale();
        let image_w = self.interaction_graph_width * scale;
        let image_h = self.interaction_graph_height * scale;
        let image_left = ((surface_width as f32 - image_w) * 0.5).max(0.0);
        let image_top = ((surface_height as f32 - image_h) * 0.5).max(0.0);
        let mut vertices = Vec::with_capacity(self.interaction_targets.len() * 6);
        for (index, target) in self.interaction_targets.iter().enumerate() {
            let (graph_x, graph_y, graph_w, graph_h) = Self::pick_target_graph_bounds(target);
            let rect = (
                image_left + graph_x * scale,
                image_top + graph_y * scale,
                graph_w * scale,
                graph_h * scale,
            );
            Self::push_pick_quad(
                &mut vertices,
                surface_width as f32,
                surface_height as f32,
                rect,
                Self::encode_pick_color(index),
                Self::pick_shape_kind(target),
            );
        }
        vertices
    }

    fn pick_interaction_target_gpu(&self, position: (f64, f64)) -> Option<PreviewInteractionNode> {
        let (Some(window), Some(device), Some(queue), Some(picking_pipeline)) = (
            self.window.as_ref(),
            self.device.as_ref(),
            self.queue.as_ref(),
            self.picking_pipeline.as_ref(),
        ) else {
            return None;
        };
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let x = (position.0.round() as i64).clamp(0, i64::from(width.saturating_sub(1))) as u32;
        let y = (position.1.round() as i64).clamp(0, i64::from(height.saturating_sub(1))) as u32;
        let vertices = self.build_pick_vertices(width, height);
        if vertices.is_empty() {
            return None;
        }

        let mut vertex_bytes = Vec::with_capacity(vertices.len() * PickVertex::STRIDE as usize);
        for vertex in &vertices {
            vertex.write_bytes(&mut vertex_bytes);
        }
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motionloom-live-preview-pick-vertices"),
            size: vertex_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(&vertex_bytes);
        vertex_buffer.unmap();

        let pick_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motionloom-live-preview-pick-id-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_view = pick_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motionloom-live-preview-pick-readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motionloom-live-preview-pick-command-encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motionloom-live-preview-pick-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pick_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(picking_pipeline);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.draw(0..vertices.len() as u32, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &pick_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let submission = queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::WaitForSubmissionIndex(submission))
            .ok();

        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::Wait).ok();
        if receiver
            .recv_timeout(Duration::from_millis(50))
            .ok()?
            .is_err()
        {
            return None;
        }
        let mapped = slice.get_mapped_range();
        let index = Self::decode_pick_color(&mapped[0..4])?;
        drop(mapped);
        readback.unmap();
        self.interaction_targets.get(index).cloned()
    }

    fn pick_interaction_target_renderer_id(
        &mut self,
        position: (f64, f64),
    ) -> Option<PreviewInteractionNode> {
        let (x, y) = self.cursor_to_graph_pixel(position)?;
        let graph = self.graph.clone()?;
        let pick_ids = self
            .interaction_targets
            .iter()
            .enumerate()
            .map(|(index, target)| (target.node.clone(), (index + 1).min(0x00FF_FFFF) as u32))
            .collect::<Vec<_>>();
        if pick_ids.is_empty() {
            return None;
        }
        let preview_engine = self.preview_engine.as_mut()?;
        let pick_id = pollster::block_on(
            preview_engine.pick_id_at_wgpu_position(&graph, self.frame, x, y, &pick_ids),
        )
        .ok()
        .flatten()?;
        let index = pick_id.checked_sub(1)? as usize;
        self.interaction_targets.get(index).cloned()
    }

    fn pick_interaction_target(
        &self,
        graph_x: f32,
        graph_y: f32,
    ) -> Option<PreviewInteractionNode> {
        self.interaction_targets
            .iter()
            .rev()
            .find(|target| {
                let width = target.width.abs().max(12.0);
                let height = target.height.abs().max(12.0);
                let centered = matches!(target.tag.as_str(), "Circle" | "Ellipse");
                let left = if centered {
                    target.x - width * 0.5
                } else {
                    target.x
                };
                let top = if centered {
                    target.y - height * 0.5
                } else {
                    target.y
                };
                graph_x >= left
                    && graph_x <= left + width
                    && graph_y >= top
                    && graph_y <= top + height
            })
            .cloned()
    }

    fn begin_interaction_drag(&mut self, position: (f64, f64)) {
        let (graph_x, graph_y) = self.cursor_to_graph_position(position);
        let Some(target) = self
            .pick_interaction_target_gpu(position)
            .or_else(|| self.pick_interaction_target(graph_x, graph_y))
            .or_else(|| self.pick_interaction_target_renderer_id(position))
        else {
            self.broadcast_event(PreviewEvent::PickResult {
                node: None,
                x: graph_x,
                y: graph_y,
            });
            return;
        };
        self.broadcast_event(PreviewEvent::PickResult {
            node: Some(target.node.clone()),
            x: graph_x,
            y: graph_y,
        });
        self.interaction_drag = Some(PreviewInteractionDrag {
            target,
            mode: self.interaction_mode,
            graph_width: self.interaction_graph_width,
            graph_height: self.interaction_graph_height,
            start_cursor_x: position.0,
            start_cursor_y: position.1,
        });
    }

    fn update_interaction_drag(&mut self, position: (f64, f64)) {
        if self.mouse_left_down && self.interaction_drag.is_none() {
            self.begin_interaction_drag(position);
        }
        let Some(drag) = self.interaction_drag.clone() else {
            return;
        };
        self.keep_attached_visible();
        let scale = (self
            .window
            .as_ref()
            .map(|window| {
                let inner = window.inner_size();
                let surface_w = inner.width.max(1) as f32;
                let surface_h = inner.height.max(1) as f32;
                (surface_w / drag.graph_width)
                    .min(surface_h / drag.graph_height)
                    .max(0.001)
            })
            .unwrap_or(1.0))
        .max(0.001);
        let delta_x = (position.0 - drag.start_cursor_x) as f32 / scale;
        let delta_y = (position.1 - drag.start_cursor_y) as f32 / scale;
        match drag.mode {
            PreviewInteractionMode::Move => {
                let x = drag.target.x + delta_x;
                let y = drag.target.y + delta_y;
                self.overrides
                    .insert((drag.target.node.clone(), "x".to_string()), x);
                self.overrides
                    .insert((drag.target.node.clone(), "y".to_string()), y);
                self.broadcast_event(PreviewEvent::TransformDrag {
                    node: drag.target.node.clone(),
                    property: "x".to_string(),
                    value: x,
                });
                self.broadcast_event(PreviewEvent::TransformDrag {
                    node: drag.target.node,
                    property: "y".to_string(),
                    value: y,
                });
            }
            PreviewInteractionMode::Rotate => {
                let rotation = drag.target.rotation + delta_x * 0.5;
                self.overrides
                    .insert((drag.target.node.clone(), "rotation".to_string()), rotation);
                self.broadcast_event(PreviewEvent::TransformDrag {
                    node: drag.target.node,
                    property: "rotation".to_string(),
                    value: rotation,
                });
            }
        }
        self.request_redraw();
    }

    fn end_interaction_drag(&mut self) {
        self.mouse_left_down = false;
        let Some(drag) = self.interaction_drag.take() else {
            return;
        };
        self.broadcast_event(PreviewEvent::TransformDragEnd {
            node: drag.target.node,
        });
    }

    fn init_wgpu(
        &mut self,
        event_loop: &ActiveEventLoop,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let window_attributes = WindowAttributes::default()
            .with_title("MotionLoom wgpu live preview")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window_attributes = if self.host_mode {
            // Controller-owned host windows must stay visible while editing in Anica.
            window_attributes
                .with_window_level(WindowLevel::AlwaysOnTop)
                .with_decorations(false)
        } else {
            window_attributes
        };
        let window = Arc::new(event_loop.create_window(window_attributes)?);
        if self.host_mode {
            configure_host_companion_window(&window);
            window.set_visible(false);
        }
        self.window = Some(window.clone());
        self.apply_requested_window_bounds();
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))?;
        let adapter_features = adapter.features();
        let timestamp_features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let required_features = if adapter_features.contains(timestamp_features) {
            timestamp_features
        } else if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("motionloom-live-preview-device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))?;
        // wgpu's default uncaptured-error handler panics. On macOS this render
        // function is reached through NSView::drawRect, where Rust is not
        // allowed to unwind; a recoverable validation error would therefore
        // abort the entire preview process before its useful message could be
        // reported. Keep the preview alive and preserve the complete error for
        // diagnosis instead.
        device.on_uncaptured_error(Box::new(|error| {
            eprintln!("MotionLoom live preview uncaptured GPU error: {error}");
        }));
        let device = Arc::new(device);
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Fifo)
        {
            wgpu::PresentMode::Fifo
        } else if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::AutoVsync)
        {
            wgpu::PresentMode::AutoVsync
        } else {
            surface_caps.present_modes[0]
        };
        let alpha_mode = surface_caps.alpha_modes[0];
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let preview_engine = pollster::block_on(WgpuPreviewEngine::new_with_device(
            device.clone(),
            queue.clone(),
        ))?;
        let graph = WgpuPreviewEngine::graph_for_quality(&self.base_graph, self.quality);
        let (target_width, target_height) = graph.render_size.unwrap_or(graph.size);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("motionloom-live-preview-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motionloom-live-preview-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motionloom-live-preview-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motionloom-live-preview-blit-shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motionloom-live-preview-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let overlay_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motionloom-live-preview-overlay-buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("motionloom-live-preview-overlay-bind-group-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let overlay_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motionloom-live-preview-overlay-bind-group"),
            layout: &overlay_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: overlay_buffer.as_entire_binding(),
            }],
        });
        let overlay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("motionloom-live-preview-overlay-pipeline-layout"),
                bind_group_layouts: &[&overlay_bind_group_layout],
                push_constant_ranges: &[],
            });
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motionloom-live-preview-overlay-shader"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_SHADER.into()),
        });
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motionloom-live-preview-overlay-pipeline"),
            layout: Some(&overlay_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let picking_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("motionloom-live-preview-picking-pipeline-layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
        let picking_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motionloom-live-preview-picking-shader"),
            source: wgpu::ShaderSource::Wgsl(PICK_SHADER.into()),
        });
        let picking_attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: 32,
                shader_location: 3,
            },
        ];
        let picking_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motionloom-live-preview-picking-pipeline"),
            layout: Some(&picking_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &picking_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: PickVertex::STRIDE,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &picking_attributes,
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &picking_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.instance = Some(instance);
        self.surface = Some(surface);
        self.adapter = Some(adapter);
        self.device = Some(device);
        self.queue = Some(queue);
        self.surface_config = Some(surface_config);
        self.surface_format = Some(surface_format);
        self.preview_engine = Some(preview_engine);
        self.graph = Some(graph);
        self.target_width = target_width.max(1);
        self.target_height = target_height.max(1);
        self.sampler = Some(sampler);
        self.bind_group_layout = Some(bind_group_layout);
        self.pipeline = Some(pipeline);
        self.overlay_buffer = Some(overlay_buffer);
        self.overlay_bind_group = Some(overlay_bind_group);
        self.overlay_pipeline = Some(overlay_pipeline);
        self.picking_pipeline = Some(picking_pipeline);
        self.restart_playback_clock();
        Ok(())
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        let Some(surface) = self.surface.as_ref() else {
            return;
        };
        let Some(device) = self.device.as_ref() else {
            return;
        };
        let Some(config) = self.surface_config.as_mut() else {
            return;
        };
        config.width = size.width.max(1);
        config.height = size.height.max(1);
        surface.configure(device, config);
    }

    fn render(&mut self) {
        if self.pending_preload {
            let preload = {
                let (Some(graph), Some(preview_engine)) =
                    (self.graph.as_ref(), self.preview_engine.as_mut())
                else {
                    return;
                };
                let session = self
                    .preload_session
                    .get_or_insert_with(|| WgpuPreviewEngine::begin_preload_graph_resources(graph));
                pollster::block_on(preview_engine.preload_graph_resources_step(graph, session))
            };
            match preload {
                Ok(progress) => {
                    preview_host_debug_log(format!(
                        "preload progress {}/{} warmed_frame={:?}",
                        progress.completed_frames, progress.total_frames, progress.warmed_frame
                    ));
                    if progress.finished {
                        let report = self
                            .preload_session
                            .as_ref()
                            .expect("preload session exists after successful step")
                            .report();
                        preview_host_debug_log(format!(
                            "preloaded {} declared asset(s) at frame(s) {:?}",
                            report.asset_declarations, report.sampled_frames
                        ));
                        self.pending_preload = false;
                        self.preload_session = None;
                        self.restart_playback_clock();
                    } else {
                        self.request_redraw();
                        return;
                    }
                }
                Err(err) => {
                    eprintln!("preview preload failed: {err}");
                    self.broadcast_event(PreviewEvent::Error {
                        message: format!("preview preload failed: {err}"),
                    });
                    self.pending_preload = false;
                    self.preload_session = None;
                }
            }
        }
        self.update_frame_from_wall_clock();
        let render_start = Instant::now();
        let (frame_texture, gpu_frame_ms, cpu_profile, profile_3d) = {
            let (Some(graph), Some(preview_engine)) =
                (self.graph.as_ref(), self.preview_engine.as_mut())
            else {
                return;
            };
            let texture = match pollster::block_on(
                preview_engine.render_frame_for_native_present(graph, self.frame),
            ) {
                Ok(texture) => texture,
                Err(err) => {
                    eprintln!("render frame {} failed: {err}", self.frame);
                    return;
                }
            };
            (
                texture,
                preview_engine.last_gpu_frame_ms(),
                preview_engine.last_cpu_frame_profile().unwrap_or_default(),
                preview_engine.last_3d_frame_profile().unwrap_or_default(),
            )
        };
        let render_ms = render_start.elapsed().as_secs_f32() * 1000.0;

        self.target_width = frame_texture.width;
        self.target_height = frame_texture.height;
        let cache_index = self
            .present_cache
            .iter()
            .position(|cached| Arc::ptr_eq(&cached.texture, &frame_texture.texture))
            .unwrap_or_else(|| {
                let device = self.device.as_ref().expect("device initialized");
                let sampler = self.sampler.as_ref().expect("sampler initialized");
                let layout = self
                    .bind_group_layout
                    .as_ref()
                    .expect("bind group layout initialized");
                let view = frame_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("motionloom-live-preview-bind-group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                });
                if self.present_cache.len() >= 3 {
                    self.present_cache.remove(0);
                }
                self.present_cache.push(PresentTextureCache {
                    texture: frame_texture.texture.clone(),
                    _view: view,
                    bind_group,
                });
                self.present_cache.len() - 1
            });

        let (
            Some(surface),
            Some(device),
            Some(queue),
            Some(pipeline),
            Some(overlay_buffer),
            Some(overlay_bind_group),
            Some(overlay_pipeline),
            Some(surface_config),
        ) = (
            self.surface.as_ref(),
            self.device.as_ref(),
            self.queue.as_ref(),
            self.pipeline.as_ref(),
            self.overlay_buffer.as_ref(),
            self.overlay_bind_group.as_ref(),
            self.overlay_pipeline.as_ref(),
            self.surface_config.as_ref(),
        )
        else {
            return;
        };

        let surface_texture = match surface.get_current_texture() {
            Ok(texture) => texture,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                if let Some(config) = self.surface_config.as_ref() {
                    surface.configure(device, config);
                }
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => return,
            Err(wgpu::SurfaceError::OutOfMemory) => {
                eprintln!("surface out of memory");
                return;
            }
            Err(wgpu::SurfaceError::Other) => return,
        };

        let present_start = Instant::now();
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = &self.present_cache[cache_index].bind_group;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motionloom-live-preview-command-encoder"),
        });
        let mut overlay_uniforms = [0u8; 16];
        overlay_uniforms[0..4].copy_from_slice(&(surface_config.width as f32).to_ne_bytes());
        overlay_uniforms[4..8].copy_from_slice(&(surface_config.height as f32).to_ne_bytes());
        overlay_uniforms[8..12].copy_from_slice(&self.quality.index().to_ne_bytes());
        queue.write_buffer(overlay_buffer, 0, &overlay_uniforms);
        let (viewport_x, viewport_y, viewport_width, viewport_height) = aspect_fit_viewport(
            surface_config.width,
            surface_config.height,
            frame_texture.width,
            frame_texture.height,
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motionloom-live-preview-blit-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_viewport(
                viewport_x,
                viewport_y,
                viewport_width,
                viewport_height,
                0.0,
                1.0,
            );
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
            pass.set_viewport(
                0.0,
                0.0,
                surface_config.width as f32,
                surface_config.height as f32,
                0.0,
                1.0,
            );
            pass.set_pipeline(overlay_pipeline);
            pass.set_bind_group(0, overlay_bind_group, &[]);
            pass.draw(0..6, 0..5);
        }
        queue.submit(Some(encoder.finish()));
        surface_texture.present();
        let present_ms = present_start.elapsed().as_secs_f32() * 1000.0;
        let rendered_frame = self.frame;

        self.last_render_ms = render_ms;
        self.last_gpu_frame_ms = gpu_frame_ms;
        self.last_cpu_profile = cpu_profile;
        self.last_3d_profile = profile_3d;
        self.last_present_ms = present_ms;
        self.render_times.push(render_ms);
        self.present_times.push(present_ms);
        if self.render_times.len() > 240 {
            self.render_times.remove(0);
        }
        if self.present_times.len() > 240 {
            self.present_times.remove(0);
        }

        self.broadcast_event(PreviewEvent::Rendered {
            frame: rendered_frame,
        });
        if let Some(quality) = self.adaptive_quality.observe(self.quality, render_ms) {
            preview_host_debug_log(format!(
                "adaptive preview quality: {} -> {} after {:.2} ms frame",
                self.quality.label(),
                quality.label(),
                render_ms
            ));
            self.set_quality_internal(quality);
        }
        self.update_title();
        self.print_stats();
    }

    fn hide_stale_attached_window(&mut self) {
        if !self.host_mode {
            return;
        }
        let Some(last_heartbeat) = self.last_attach_heartbeat else {
            return;
        };
        if last_heartbeat.elapsed() < HOST_ATTACH_HEARTBEAT_TIMEOUT {
            return;
        }
        self.last_attach_heartbeat = None;
        self.window_visible = false;
        if let Some(window) = self.window.as_ref() {
            window.set_visible(false);
        }
    }

    fn update_title(&mut self) {
        if self.last_title_at.elapsed() < Duration::from_millis(250) {
            return;
        }
        self.last_title_at = Instant::now();
        self.update_title_now();
    }

    fn update_title_now(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let avg_render = avg(&self.render_times);
        let min_render = min_or_zero(&self.render_times);
        let max_render = max_or_zero(&self.render_times);
        let fps = self.graph.as_ref().map(|graph| graph.fps).unwrap_or(0.0);
        let gpu_frame_label = self
            .last_gpu_frame_ms
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "pending".to_string());
        window.set_title(&format!(
            "MotionLoom wgpu live preview | frame {}/{} | CPU submit {:.2} ms | 3D prep/submit {:.2}/{:.2} ms | GPU {} ms | avg {:.2} ms | min/max {:.2}/{:.2} ms | blit {:.2} ms | timeline {:.1} fps | target {}x{} | surface {:?} | quality {} (0 Auto, 1 Full, 2 Balanced, 3 Speed, 4 High Speed, 5 Ultra Speed) | {}",
            self.frame,
            self.total_frames,
            self.last_render_ms,
            self.last_3d_profile.prepare_ms,
            self.last_3d_profile.submit_ms,
            gpu_frame_label,
            avg_render,
            min_render,
            max_render,
            self.last_present_ms,
            fps,
            self.target_width,
            self.target_height,
            self.surface_format,
            self.quality.label(),
            self.script_source
        ));
    }

    fn print_stats(&mut self) {
        if !self.print_stats_enabled {
            return;
        }
        if self.last_stats_at.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_stats_at = Instant::now();
        let gpu_frame_label = self
            .last_gpu_frame_ms
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "pending".to_string());
        println!(
            "quality={} target={}x{} frame={}/{} cpu_submit_last_ms={:.2} expression_ms={:.3} traversal_ms={:.3} upload_ms={:.3} encode_ms={:.3} wait_ms={:.3} scene3d_prepare_ms={:.3} scene3d_asset_resolve_ms={:.3} texture_decode_ms={:.3} texture_decodes={} texture_cache_hits={} texture_decoded_bytes={} scene3d_submit_ms={:.3} scene3d_draw_calls={} scene3d_resources={} scene3d_texture_resources={} scene3d_geometry_resources={} scene3d_targets={} gpu_last_ms={} gpu_timestamp_supported={} render_avg_ms={:.2} render_min_ms={:.2} render_max_ms={:.2} blit_last_ms={:.2} blit_avg_ms={:.2}",
            self.quality.label(),
            self.target_width,
            self.target_height,
            self.frame,
            self.total_frames,
            self.last_render_ms,
            self.last_cpu_profile.expression_ms,
            self.last_cpu_profile.traversal_ms,
            self.last_cpu_profile.upload_ms,
            self.last_cpu_profile.encode_ms,
            self.last_cpu_profile.wait_ms,
            self.last_3d_profile.prepare_ms,
            self.last_3d_profile.asset_resolve_ms,
            self.last_3d_profile.texture_decode_ms,
            self.last_3d_profile.texture_decode_count,
            self.last_3d_profile.texture_cache_hits,
            self.last_3d_profile.texture_decoded_bytes,
            self.last_3d_profile.submit_ms,
            self.last_3d_profile.draw_calls,
            self.last_3d_profile.gpu_resource_entries,
            self.last_3d_profile.gpu_texture_resources,
            self.last_3d_profile.gpu_geometry_resources,
            self.last_3d_profile.target_pool_size,
            gpu_frame_label,
            self.preview_engine
                .as_ref()
                .is_some_and(WgpuPreviewEngine::gpu_timestamp_supported),
            avg(&self.render_times),
            min_or_zero(&self.render_times),
            max_or_zero(&self.render_times),
            self.last_present_ms,
            avg(&self.present_times)
        );
    }
}

#[cfg(target_os = "macos")]
fn configure_host_companion_window(window: &Window) {
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return;
    };

    // Mission Control should treat the preview as an editor companion, not as a
    // separate app window that moves to its own Space.
    unsafe {
        let view = handle.ns_view.as_ptr() as *mut NSView;
        let Some(ns_window) = view.as_ref().and_then(NSView::window) else {
            return;
        };
        ns_window.setCollectionBehavior(
            NSWindowCollectionBehavior::Transient | NSWindowCollectionBehavior::IgnoresCycle,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn configure_host_companion_window(_window: &Window) {}

impl ApplicationHandler<PreviewHostUserEvent> for LivePreviewApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none()
            && let Err(err) = self.init_wgpu(event_loop)
        {
            eprintln!("failed to initialize wgpu live preview: {err}");
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        PhysicalKey::Code(KeyCode::Digit1) => {
                            self.set_quality(WgpuPreviewQuality::Full);
                        }
                        PhysicalKey::Code(KeyCode::Digit2) => {
                            self.set_quality(WgpuPreviewQuality::Balanced);
                        }
                        PhysicalKey::Code(KeyCode::Digit3) => {
                            self.set_quality(WgpuPreviewQuality::Speed);
                        }
                        PhysicalKey::Code(KeyCode::Digit4) => {
                            self.set_quality(WgpuPreviewQuality::HighSpeed);
                        }
                        PhysicalKey::Code(KeyCode::Digit5) => {
                            self.set_quality(WgpuPreviewQuality::UltraSpeed);
                        }
                        PhysicalKey::Code(KeyCode::Digit0) => {
                            self.resume_adaptive_quality();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.last_cursor_pos = Some((position.x, position.y));
                self.update_interaction_drag((position.x, position.y));
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_left_down = true;
                self.keep_attached_visible();
                if let Some(position) = self.last_cursor_pos
                    && let Some(quality) = Self::quality_button_at(position)
                {
                    self.set_quality(quality);
                    self.mouse_left_down = false;
                    return;
                }
                if let Some(position) = self.last_cursor_pos {
                    self.begin_interaction_drag(position);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => self.end_interaction_drag(),
            WindowEvent::Focused(focused) => {
                if self.host_mode && preview_host_platform::should_emit_host_focus_events() {
                    self.broadcast_event(PreviewEvent::HostFocus { focused });
                }
            }
            WindowEvent::Resized(size) => self.resize(size),
            WindowEvent::RedrawRequested => {
                if self.host_mode && (!self.window_visible || !self.needs_redraw) {
                    return;
                }
                self.needs_redraw = false;
                self.render();
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: PreviewHostUserEvent) {
        match event {
            PreviewHostUserEvent::Command(command) => self.handle_preview_command(command),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.hide_stale_attached_window();
        let now = Instant::now();
        let can_draw = !self.host_mode || self.window_visible;
        if can_draw && (self.auto_advance || self.needs_redraw) && now >= self.next_redraw_at {
            self.needs_redraw = true;
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
            self.next_redraw_at = now + self.frame_interval();
        }
        let heartbeat_at = now + Duration::from_millis(100);
        let wake_at = if self.auto_advance || self.needs_redraw {
            self.next_redraw_at.min(heartbeat_at)
        } else {
            heartbeat_at
        };
        event_loop.set_control_flow(ControlFlow::WaitUntil(wake_at));
    }
}

fn avg(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn min_or_zero(values: &[f32]) -> f32 {
    values.iter().copied().reduce(f32::min).unwrap_or(0.0)
}

fn max_or_zero(values: &[f32]) -> f32 {
    values.iter().copied().reduce(f32::max).unwrap_or(0.0)
}

fn is_http_url(value: &str) -> bool {
    value.starts_with("https://") || value.starts_with("http://")
}

fn local_scene_asset_root_for_source(source: &str) -> Option<std::path::PathBuf> {
    if source.is_empty() || is_http_url(source) {
        return None;
    }
    let parent = std::path::Path::new(source).parent()?;
    if parent.as_os_str().is_empty() {
        return None;
    }
    Some(std::fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf()))
}

fn configure_scene_asset_root_for_source(source: &str) {
    if let Some(root) = local_scene_asset_root_for_source(source) {
        motionloom::set_scene_asset_roots(vec![root]);
    }
}

fn load_script_source(source: &str) -> Result<String, Box<dyn std::error::Error>> {
    if is_http_url(source) {
        let response = ureq::get(source).call()?;
        return Ok(response.into_string()?);
    }
    Ok(std::fs::read_to_string(source)?)
}

fn write_preview_event(stream: &mut TcpStream, event: PreviewEvent) -> std::io::Result<()> {
    let payload = serde_json::to_string(&event).unwrap_or_else(|err| {
        format!("{{\"type\":\"error\",\"message\":\"failed to serialize preview event: {err}\"}}")
    });
    stream.write_all(payload.as_bytes())?;
    stream.write_all(b"\n")
}

fn handle_preview_client(stream: TcpStream, proxy: EventLoopProxy<PreviewHostUserEvent>) {
    let mut writer = match stream.try_clone() {
        Ok(writer) => writer,
        Err(err) => {
            eprintln!("preview host failed to clone client stream: {err}");
            return;
        }
    };
    let _ = write_preview_event(
        &mut writer,
        PreviewEvent::Ready {
            protocol_version: PREVIEW_PROTOCOL_VERSION,
        },
    );
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                eprintln!("preview host client read error: {err}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<PreviewCommand>(&line) {
            Ok(command) => {
                if proxy
                    .send_event(PreviewHostUserEvent::Command(command))
                    .is_err()
                {
                    let _ = write_preview_event(
                        &mut writer,
                        PreviewEvent::Error {
                            message: "preview event loop is closed".to_string(),
                        },
                    );
                    break;
                }
            }
            Err(err) => {
                let _ = write_preview_event(
                    &mut writer,
                    PreviewEvent::Error {
                        message: format!("invalid preview command JSON: {err}"),
                    },
                );
            }
        }
    }
}

fn start_preview_command_server(
    addr: String,
    proxy: EventLoopProxy<PreviewHostUserEvent>,
    broadcaster: PreviewEventBroadcaster,
) -> std::io::Result<()> {
    let listener = TcpListener::bind(&addr)?;
    println!(
        "MotionLoom preview host pid={} listening on {addr}",
        std::process::id()
    );
    thread::Builder::new()
        .name("motionloom-preview-host-tcp".to_string())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let _ = stream.set_nodelay(true);
                        if let Ok(writer) = stream.try_clone() {
                            broadcaster.add_client(writer);
                        }
                        let proxy = proxy.clone();
                        let _ = thread::Builder::new()
                            .name("motionloom-preview-host-client".to_string())
                            .spawn(move || handle_preview_client(stream, proxy));
                    }
                    Err(err) => eprintln!("preview host accept error: {err}"),
                }
            }
        })?;
    Ok(())
}

fn build_event_loop(
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))] host_mode: bool,
) -> Result<EventLoop<PreviewHostUserEvent>, winit::error::EventLoopError> {
    let mut builder = EventLoop::<PreviewHostUserEvent>::with_user_event();
    #[cfg(target_os = "macos")]
    if host_mode {
        // Accessory mode keeps the viewer as an editor companion instead of a
        // separate Mission Control app, which reduces Space/focus disruption.
        builder.with_activation_policy(ActivationPolicy::Accessory);
        builder.with_default_menu(false);
        builder.with_activate_ignoring_other_apps(false);
    }
    builder.build()
}

fn patch_scene_tag_attr_number(
    script: &str,
    node_id: &str,
    attr: &str,
    value: f32,
) -> Result<String, String> {
    let (tag_start, tag_end) = find_scene_tag_range_by_id(script, node_id)
        .ok_or_else(|| format!("Preview override target id=\"{node_id}\" was not found."))?;
    let tag = &script[tag_start..tag_end];

    if let Some((value_start, value_end)) = find_attr_value_range_in_tag(tag, attr) {
        let abs_start = tag_start + value_start;
        let abs_end = tag_start + value_end;
        let value_text = format_live_number(value);
        let mut out = String::with_capacity(script.len() + value_text.len());
        out.push_str(&script[..abs_start]);
        out.push_str(&value_text);
        out.push_str(&script[abs_end..]);
        return Ok(out);
    }

    let insert_at = tag
        .rfind("/>")
        .map(|rel| tag_start + rel)
        .unwrap_or(tag_end.saturating_sub(1));
    let value_text = format_live_number(value);
    let mut out = String::with_capacity(script.len() + attr.len() + value_text.len() + 4);
    out.push_str(&script[..insert_at]);
    out.push(' ');
    out.push_str(attr);
    out.push_str("=\"");
    out.push_str(&value_text);
    out.push('"');
    out.push_str(&script[insert_at..]);
    Ok(out)
}

fn find_scene_tag_range_by_id(script: &str, node_id: &str) -> Option<(usize, usize)> {
    let id_pattern = format!("id=\"{node_id}\"");
    let id_pos = script.find(&id_pattern)?;
    let tag_start = script[..id_pos].rfind('<')?;
    let tag_end = script[id_pos..]
        .find('>')
        .map(|offset| id_pos + offset + 1)?;
    Some((tag_start, tag_end))
}

fn find_attr_value_range_in_tag(tag: &str, attr: &str) -> Option<(usize, usize)> {
    let bytes = tag.as_bytes();
    let attr_bytes = attr.as_bytes();
    let mut index = 0usize;
    while index + attr_bytes.len() <= bytes.len() {
        if &bytes[index..index + attr_bytes.len()] != attr_bytes {
            index += 1;
            continue;
        }
        let before_ok = index == 0 || !is_ident_byte(bytes[index - 1]);
        let after = index + attr_bytes.len();
        let after_ok = after < bytes.len() && !is_ident_byte(bytes[after]);
        if !before_ok || !after_ok {
            index += 1;
            continue;
        }
        let mut cursor = after;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= bytes.len() || bytes[cursor] != b'=' {
            index += 1;
            continue;
        }
        cursor += 1;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= bytes.len() {
            return None;
        }
        match bytes[cursor] {
            b'"' | b'\'' => {
                let quote = bytes[cursor];
                let value_start = cursor + 1;
                let value_end = tag[value_start..]
                    .find(quote as char)
                    .map(|offset| value_start + offset)?;
                return Some((value_start, value_end));
            }
            b'{' => {
                let value_start = cursor + 1;
                let value_end = tag[value_start..]
                    .find('}')
                    .map(|offset| value_start + offset)?;
                return Some((value_start, value_end));
            }
            _ => return None,
        }
    }
    None
}

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
}

fn format_live_number(value: f32) -> String {
    if value.is_finite() && (value.fract()).abs() < 0.0001 {
        return format!("{}", value.round() as i64);
    }
    let mut text = format!("{value:.3}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::{aspect_fit_viewport, local_scene_asset_root_for_source};

    #[test]
    fn aspect_fit_viewport_letterboxes_wide_surface() {
        assert_eq!(
            aspect_fit_viewport(1600, 900, 1200, 900),
            (200.0, 0.0, 1200.0, 900.0)
        );
    }

    #[test]
    fn aspect_fit_viewport_pillarboxes_tall_surface() {
        assert_eq!(
            aspect_fit_viewport(1200, 1200, 1200, 900),
            (0.0, 150.0, 1200.0, 900.0)
        );
    }

    #[test]
    fn aspect_fit_viewport_scales_without_distortion() {
        assert_eq!(
            aspect_fit_viewport(600, 450, 1200, 900),
            (0.0, 0.0, 600.0, 450.0)
        );
    }

    #[test]
    fn local_script_source_uses_its_parent_as_asset_root() {
        let root = local_scene_asset_root_for_source("showcase/s-000070/main.motionloom")
            .expect("local script should have an asset root");
        assert!(root.ends_with("showcase/s-000070"));
    }

    #[test]
    fn remote_script_source_does_not_create_a_filesystem_asset_root() {
        assert!(
            local_scene_asset_root_for_source(
                "https://example.com/showcase/s-000070/main.motionloom"
            )
            .is_none()
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut script_source = None;
    let mut print_stats = false;
    let mut listen_addr = None::<String>;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        if arg == "--stats" || arg == "--print-stats" {
            print_stats = true;
        } else if arg == "--listen" {
            let Some(addr) = args.next() else {
                eprintln!("--listen requires an address, for example 127.0.0.1:49377");
                std::process::exit(2);
            };
            listen_addr = Some(addr);
        } else if script_source.is_none() {
            script_source = Some(arg);
        } else {
            eprintln!("unknown extra argument: {arg}");
            std::process::exit(2);
        }
    }
    if listen_addr.is_none() && script_source.is_none() {
        eprintln!(
            "usage: cargo run -p motionloom --example wgpu_live_preview -- [--stats] path-or-url/to/main.motionloom\n       cargo run -p motionloom --example wgpu_live_preview -- --listen 127.0.0.1:49377 [optional/path.motionloom]"
        );
        std::process::exit(2);
    }
    let host_mode = listen_addr.is_some();
    let event_loop = build_event_loop(host_mode)?;
    let host_events = listen_addr
        .as_ref()
        .map(|_| PreviewEventBroadcaster::default());
    if let (Some(addr), Some(host_events)) = (listen_addr, host_events.clone()) {
        start_preview_command_server(addr, event_loop.create_proxy(), host_events)?;
    }
    let auto_advance = host_events.is_none();
    let mut app = LivePreviewApp::new(
        script_source,
        print_stats,
        auto_advance,
        host_mode,
        host_events,
    )?;
    event_loop.run_app(&mut app)?;
    Ok(())
}
