// =========================================
// =========================================
// src/ui/motionloom_page.rs — MotionLoom Studio page with graph preview and template picker

use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "macos")]
use core_video::pixel_buffer::CVPixelBuffer;

use gpui::{
    ClipboardItem, Context, Element, Entity, Focusable, GlobalElementId, InspectorElementId,
    IntoElement, LayoutId, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    PathPromptOptions, Render, RenderImage, ScrollWheelEvent, SharedString, Style, Subscription,
    Timer, Window, div, prelude::*, px, rgb, rgba,
};
use gpui_component::{
    input::{Input, InputEvent, InputState},
    scroll::ScrollableElement,
    select::{SearchableVec, Select, SelectEvent, SelectItem, SelectState},
    white,
};
use image::{ImageBuffer, Rgba, RgbaImage};
use motionloom::experimental::{Scene3DNode, SceneNode};
use motionloom::{
    EditableAnimationKey, EditableAnimationTarget, GraphScript, MotionLoomDocument,
    MotionLoomRenderProgress, PreviewCommand, PreviewEvent, PreviewInteractionMode,
    PreviewInteractionNode, RuntimeProgram, ScenePlatformPreviewSurface, ScenePreviewBackend,
    ScenePreviewSurface, ScenePreviewSurfaceOptions, SceneRenderProfile, SceneRenderer,
    WgpuPreviewEngine, WgpuPreviewGraphCache, WgpuPreviewQuality, WorldActor, WorldBackground,
    WorldBackgroundFit, WorldCamera, WorldFrameRenderer, WorldGraph, WorldModelProfile, WorldNode,
    WorldPathStyle, WorldPresent, WorldProfileRetarget, WorldRetargetMap, compile_runtime_program,
    extract_editable_animation_timeline, is_graph_script, is_world_graph_script,
    load_glb_mesh_data, next_scene_output_path_for_profile, parse_graph_script,
    parse_motionloom_document, parse_world_graph_script,
    render_motionloom_document_to_video_with_progress_and_cancel, render_scene_graph_frame,
    replace_editable_animation_targets, upsert_editable_animation_target,
};
use smallvec::SmallVec;
use thiserror::Error;

use crate::core::export::get_media_duration;
use crate::core::global_state::{AppPage, GlobalState, MediaPoolUiEvent};
use crate::core::thumbnail;
use crate::ui::motionloom_templates;
use crate::ui::motionloom_templates::LayerEffectTemplateKind;

const THUMB_MAX_DIM: u32 = 640;
const SCENE_RENDER_PROGRESS_EVERY_FRAMES: u32 = 10;
const SCENE_RENDER_PROGRESS_POLL_MS: u64 = 120;
const SCENE_RENDER_LOG_MAX_LINES: usize = 10;
const SCENE_LIVE_PREVIEW_144P_MAX_DIM: u32 = 256;
const SCENE_LIVE_PREVIEW_240P_MAX_DIM: u32 = 426;
const SCENE_LIVE_PREVIEW_360P_MAX_DIM: u32 = 640;
const SCENE_LIVE_PREVIEW_480P_MAX_DIM: u32 = 854;
const SCENE_LIVE_SCROLL_RENDER_DEBOUNCE_MS: u64 = 2000;
const SCENE_LIVE_INPUT_RENDER_DEBOUNCE_MS: u64 = 120;
const SCENE_LIVE_PRERENDER_MAX_FRAMES: u32 = 6000;
const SCENE_LIVE_PREVIEW_FRAME_CACHE_CAPACITY: usize = 6000;
const SCENE_LIVE_PREVIEW_FRAME_CACHE_MAX_BYTES: usize = 768 * 1024 * 1024;
const SCENE_LIVE_RENDER_WORKER_STACK_SIZE: usize = 16 * 1024 * 1024;
const SCENE_LIVE_PRERENDER_POLL_MS: u64 = 16;
const SCENE_LIVE_IDLE_POLL_MS: u64 = 120;
const DEFAULT_SCENE_LIVE_NODE_ID: &str = "iris_outer_soft";
const DEFAULT_SCENE_LIVE_ATTR: &str = "x";
const MOTIONLOOM_EXAMPLE_RAW_ROOT: &str =
    "https://raw.githubusercontent.com/LOVELYZOMBIEYHO/motionloom-example/refs/heads/main";
const DEFAULT_SCENE_TEMPLATE_CATEGORY: &str = "showcase";
const DEFAULT_SCENE_TEMPLATE_NUMBER: &str = "1";

type SceneLivePreviewCacheKey = (u64, u32, SceneLivePreviewQuality, u32, u32);

fn motionloom_external_preview_debug_enabled() -> bool {
    std::env::var("ANICA_DEBUG_MOTIONLOOM_PREVIEW_ATTACH").is_ok_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    })
}

fn motionloom_external_preview_offset() -> (f64, f64) {
    let parse = |name: &str| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.trim().parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    (
        parse("ANICA_MOTIONLOOM_PREVIEW_OFFSET_X"),
        parse("ANICA_MOTIONLOOM_PREVIEW_OFFSET_Y"),
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MotionLoomExternalPreviewProtocol {
    Script,
    Frame,
    Interaction,
    Window,
    Full,
}

impl MotionLoomExternalPreviewProtocol {
    fn current() -> Self {
        let raw = std::env::var("ANICA_MOTIONLOOM_PREVIEW_PROTOCOL")
            .unwrap_or_else(|_| default_motionloom_external_preview_protocol().to_string());
        match raw.trim().to_ascii_lowercase().as_str() {
            "script" | "loadscript" | "load_script" => Self::Script,
            "frame" => Self::Frame,
            "interaction" | "targets" => Self::Interaction,
            "window" | "bounds" | "visible" => Self::Window,
            "full" | "all" => Self::Full,
            _ => default_motionloom_external_preview_protocol_enum(),
        }
    }

    fn allows_frame(self) -> bool {
        matches!(
            self,
            Self::Frame | Self::Interaction | Self::Window | Self::Full
        )
    }

    fn allows_interaction(self) -> bool {
        matches!(self, Self::Interaction | Self::Window | Self::Full)
    }

    fn allows_window(self) -> bool {
        matches!(self, Self::Window | Self::Full)
    }
}

#[cfg(target_os = "windows")]
fn default_motionloom_external_preview_protocol() -> &'static str {
    "window"
}

#[cfg(not(target_os = "windows"))]
fn default_motionloom_external_preview_protocol() -> &'static str {
    "full"
}

fn default_motionloom_external_preview_protocol_enum() -> MotionLoomExternalPreviewProtocol {
    match default_motionloom_external_preview_protocol() {
        "script" => MotionLoomExternalPreviewProtocol::Script,
        _ => MotionLoomExternalPreviewProtocol::Full,
    }
}

fn motionloom_external_preview_log(message: impl AsRef<str>) {
    if motionloom_external_preview_debug_enabled() {
        eprintln!(
            "[anica-preview pid={}] {}",
            std::process::id(),
            message.as_ref()
        );
    }
}

struct SceneExternalPreviewProcess {
    child: Child,
}

impl Drop for SceneExternalPreviewProcess {
    fn drop(&mut self) {
        // The preview host is a UI helper owned by Anica in auto mode.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Clone)]
struct SceneExternalPreviewHost {
    tx: Sender<PreviewCommand>,
    events: Arc<Mutex<VecDeque<PreviewEvent>>>,
}

impl SceneExternalPreviewHost {
    fn from_env_or_auto_spawn() -> (Option<Self>, Option<SceneExternalPreviewProcess>) {
        if let Ok(value) = std::env::var("ANICA_MOTIONLOOM_PREVIEW_HOST") {
            let value = value.trim();
            if value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("false")
                || value == "0"
            {
                return (None, None);
            }
            if !value.is_empty() && !value.eq_ignore_ascii_case("auto") {
                return (Some(Self::spawn(value.to_string())), None);
            }
        }

        match Self::spawn_preview_process() {
            Ok((addr, process)) => (Some(Self::spawn(addr)), Some(process)),
            Err(err) => {
                eprintln!("external MotionLoom preview auto-spawn failed: {err}");
                (None, None)
            }
        }
    }

    fn spawn(addr: String) -> Self {
        let (tx, rx) = mpsc::channel::<PreviewCommand>();
        let events = Arc::new(Mutex::new(VecDeque::new()));
        let controller_events = events.clone();
        let controller_pid = std::process::id();
        let _ = std::thread::Builder::new()
            .name("motionloom-external-preview-controller".to_string())
            .spawn(move || {
                let mut stream = None::<TcpStream>;
                while let Ok(command) = rx.recv() {
                    if stream.is_none() {
                        motionloom_external_preview_log(format!("connect host addr={addr}"));
                        stream = Self::connect_with_retry(&addr, controller_events.clone());
                        if let Some(socket) = stream.as_mut()
                            && !Self::write_command_line(
                                socket,
                                &PreviewCommand::SetControllerProcessId {
                                    pid: controller_pid,
                                },
                            )
                        {
                            stream = None;
                            continue;
                        }
                    }
                    let Some(socket) = stream.as_mut() else {
                        continue;
                    };
                    motionloom_external_preview_log(format!("write command {command:?}"));
                    if !Self::write_command_line(socket, &command) {
                        stream = None;
                    }
                }
            });
        Self { tx, events }
    }

    fn spawn_preview_process() -> std::io::Result<(String, SceneExternalPreviewProcess)> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?.to_string();
        drop(listener);

        let helper = Self::preview_host_binary()
            .or_else(|| Self::build_preview_host_binary().ok().flatten())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "MotionLoom preview host binary was not found and could not be built",
                )
            })?;

        motionloom_external_preview_log(format!(
            "spawn host listen={addr} protocol={:?}",
            MotionLoomExternalPreviewProtocol::current()
        ));

        let mut command = Command::new(helper);
        if motionloom_external_preview_debug_enabled() {
            command.env("MOTIONLOOM_PREVIEW_HOST_DEBUG", "1");
        }

        let child = command
            .arg("--listen")
            .arg(&addr)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        motionloom_external_preview_log(format!("spawned host pid={}", child.id()));

        Ok((addr, SceneExternalPreviewProcess { child }))
    }

    fn preview_host_binary() -> Option<PathBuf> {
        if let Ok(value) = std::env::var("ANICA_MOTIONLOOM_PREVIEW_HOST_BIN") {
            let path = PathBuf::from(value.trim());
            if path.is_file() {
                return Some(path);
            }
        }

        let binary_name = Self::preview_host_binary_name();
        let mut candidates = Vec::new();

        if let Ok(current_exe) = std::env::current_exe()
            && let Some(profile_dir) = current_exe.parent()
        {
            candidates.push(profile_dir.join("examples").join(binary_name));
        }

        let target_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        candidates.push(
            target_dir
                .join("release")
                .join("examples")
                .join(binary_name),
        );
        candidates.push(target_dir.join("debug").join("examples").join(binary_name));

        candidates.into_iter().find(|candidate| candidate.is_file())
    }

    fn build_preview_host_binary() -> std::io::Result<Option<PathBuf>> {
        Self::stop_stale_preview_host_before_build();

        let status = Command::new("cargo")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args([
                "build",
                "--release",
                "-p",
                "motionloom",
                "--example",
                "wgpu_live_preview",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()?;

        if !status.success() {
            return Ok(None);
        }

        Ok(Self::preview_host_binary())
    }

    fn stop_stale_preview_host_before_build() {
        #[cfg(target_os = "windows")]
        {
            let _ = Command::new("taskkill")
                .args(["/IM", "wgpu_live_preview.exe", "/F"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }

    fn preview_host_binary_name() -> &'static str {
        if cfg!(windows) {
            "wgpu_live_preview.exe"
        } else {
            "wgpu_live_preview"
        }
    }

    fn connect_with_retry(
        addr: &str,
        events: Arc<Mutex<VecDeque<PreviewEvent>>>,
    ) -> Option<TcpStream> {
        for _ in 0..1200 {
            if let Some(stream) = Self::connect(addr, events.clone(), false) {
                return Some(stream);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        Self::connect(addr, events, true)
    }

    fn connect(
        addr: &str,
        events: Arc<Mutex<VecDeque<PreviewEvent>>>,
        log_error: bool,
    ) -> Option<TcpStream> {
        let stream = match TcpStream::connect(addr) {
            Ok(stream) => stream,
            Err(err) => {
                if log_error {
                    eprintln!("external MotionLoom preview connect failed ({addr}): {err}");
                }
                return None;
            }
        };
        let _ = stream.set_nodelay(true);
        if let Ok(reader) = stream.try_clone() {
            let _ = std::thread::Builder::new()
                .name("motionloom-external-preview-events".to_string())
                .spawn(move || {
                    let reader = BufReader::new(reader);
                    for line in reader.lines() {
                        let Ok(line) = line else {
                            break;
                        };
                        if line.trim().is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<PreviewEvent>(&line) {
                            Ok(PreviewEvent::Error { message }) => {
                                eprintln!("external MotionLoom preview error: {message}");
                            }
                            Ok(PreviewEvent::Ready { .. }) => {
                                eprintln!("external MotionLoom preview connected");
                            }
                            Ok(PreviewEvent::WindowBounds {
                                x,
                                y,
                                width,
                                height,
                            }) => {
                                if motionloom_external_preview_debug_enabled() {
                                    eprintln!(
                                        "external MotionLoom preview host bounds: x={x:.1} y={y:.1} w={width:.1} h={height:.1}"
                                    );
                                }
                            }
                            Ok(event) => {
                                if let Ok(mut queue) = events.lock() {
                                    queue.push_back(event);
                                    while queue.len() > 240 {
                                        queue.pop_front();
                                    }
                                }
                            }
                            Err(_) => eprintln!("external MotionLoom preview event: {line}"),
                        }
                    }
                });
        }
        Some(stream)
    }

    fn write_command_line(socket: &mut TcpStream, command: &PreviewCommand) -> bool {
        let payload = match serde_json::to_string(command) {
            Ok(payload) => payload,
            Err(err) => {
                eprintln!("external MotionLoom preview command encode error: {err}");
                return true;
            }
        };
        socket.write_all(payload.as_bytes()).is_ok() && socket.write_all(b"\n").is_ok()
    }

    fn send(&self, command: PreviewCommand) {
        let _ = self.tx.send(command);
    }

    fn drain_events(&self) -> Vec<PreviewEvent> {
        let Ok(mut queue) = self.events.lock() else {
            return Vec::new();
        };
        queue.drain(..).collect()
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct SendableCVPixelBuffer(CVPixelBuffer);

#[cfg(target_os = "macos")]
// SAFETY: The worker only transfers a retained CVPixelBuffer handle to the UI
// thread. The buffer contents are fully written before send and are consumed by
// GPUI's surface paint path; no mutable access is shared after transfer.
unsafe impl Send for SendableCVPixelBuffer {}

#[cfg(target_os = "macos")]
// SAFETY: Shared references are used only to keep the retained CVPixelBuffer
// alive while GPUI paints it. The wrapper does not expose mutation.
unsafe impl Sync for SendableCVPixelBuffer {}

enum SceneLivePreviewFrame {
    Bgra {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
    #[cfg(target_os = "macos")]
    MacOsSurface {
        width: u32,
        height: u32,
        surface: SendableCVPixelBuffer,
    },
    #[cfg(target_os = "windows")]
    WindowsSurface {
        width: u32,
        height: u32,
        surface: motionloom::WindowsD3DSharedSurface,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SceneLiveGizmoMode {
    Move,
    Rotate,
}

#[derive(Clone, Copy, Debug)]
enum SceneLiveGizmoDrag {
    Move {
        start_mouse_x: f32,
        start_mouse_y: f32,
        start_x: f32,
        start_y: f32,
    },
    Rotate {
        start_mouse_x: f32,
        start_rotation: f32,
    },
}

#[derive(Clone, Copy, Debug)]
struct SceneLiveGizmoBounds {
    left: f32,
    top: f32,
    width: f32,
    height: f32,
}

impl SceneLivePreviewFrame {
    fn preview_status_label(&self) -> &'static str {
        match self {
            Self::Bgra { .. } => "Preview: CPU BGRA",
            #[cfg(target_os = "macos")]
            Self::MacOsSurface { .. } => "Preview: macOS BGRA surface",
            #[cfg(target_os = "windows")]
            Self::WindowsSurface { .. } => "Preview: Windows D3D BGRA surface",
        }
    }

    fn into_loaded_preview(self) -> Result<LoadedPreview, MotionLoomPageError> {
        match self {
            Self::Bgra {
                width,
                height,
                data,
            } => MotionLoomPage::loaded_preview_from_bgra(width, height, data),
            #[cfg(target_os = "macos")]
            Self::MacOsSurface {
                width,
                height,
                surface,
            } => MotionLoomPage::loaded_preview_from_macos_surface(width, height, surface),
            #[cfg(target_os = "windows")]
            Self::WindowsSurface {
                width,
                height,
                surface,
            } => MotionLoomPage::loaded_preview_from_windows_surface(width, height, surface),
        }
    }
}

struct WorldLivePreviewRequest {
    graph: WorldGraph,
    frame: u32,
    asset_root: PathBuf,
    response_tx: Sender<Result<(u32, u32, Vec<u8>), String>>,
}

struct SceneLivePreviewRequest {
    script: String,
    script_hash: u64,
    render_size: Option<(u32, u32)>,
    frame: u32,
    asset_roots: Vec<PathBuf>,
    response_tx: Sender<Result<(SceneLivePreviewFrame, Option<String>), String>>,
}

#[cfg(test)]
mod tests {
    use super::MotionLoomPage;

    #[test]
    fn scene_live_attr_range_handles_braced_curve_with_spaces() {
        let tag = r#"<Group id="iris_turn_left"
             x={curve("0.00:4:ease_in_out, 0.90:-54:ease_in_out, 2.20:-176:ease_in_out")}
             y="16">"#;

        let (start, end) =
            MotionLoomPage::find_attr_value_range_in_tag(tag, "x").expect("x attr range");
        assert_eq!(
            &tag[start..end],
            r#"{curve("0.00:4:ease_in_out, 0.90:-54:ease_in_out, 2.20:-176:ease_in_out")}"#
        );
    }

    #[test]
    fn scene_live_patch_replaces_whole_braced_curve_attr() {
        let script = r#"<Graph fps={30} duration="3s" size={[1320,768]}>
  <Scene id="scene">
    <Group id="iris_turn_left"
           x={curve("0.00:4:ease_in_out, 0.90:-54:ease_in_out, 2.20:-176:ease_in_out")}
           y="16">
    </Group>
  </Scene>
  <Present from="scene" />
</Graph>"#;

        let updated =
            MotionLoomPage::patch_scene_tag_attr_number(script, "iris_turn_left", "x", -10.0)
                .expect("patch x attr");
        assert!(updated.contains(r#"<Group id="iris_turn_left""#));
        assert!(updated.contains(r#"x=-10"#));
        assert!(!updated.contains("0.90:-54"));
        assert!(updated.contains("</Group>"));
    }
}

fn panic_payload_to_string(payload: Box<dyn Any + Send + 'static>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "non-string panic payload".to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SceneRenderMode {
    CompatibilityCpu,
    GpuNativeH264,
    GpuNativeProRes,
    GpuNativeProRes4444,
    GpuNativePngSequence,
    GpuNativeCurrentFramePng,
}

impl SceneRenderMode {
    const fn label(self) -> &'static str {
        match self {
            SceneRenderMode::CompatibilityCpu => "Compatibility Render (CPU)",
            SceneRenderMode::GpuNativeH264 => "GPU Render",
            SceneRenderMode::GpuNativeProRes => "GPU Render (ProRes)",
            SceneRenderMode::GpuNativeProRes4444 => "GPU Render (ProRes 4444 Alpha)",
            SceneRenderMode::GpuNativePngSequence => "GPU Render (PNG Sequence)",
            SceneRenderMode::GpuNativeCurrentFramePng => "GPU Render Current Frame (PNG)",
        }
    }

    const fn profile(self) -> SceneRenderProfile {
        match self {
            SceneRenderMode::CompatibilityCpu => SceneRenderProfile::Cpu,
            SceneRenderMode::GpuNativeH264 => SceneRenderProfile::Gpu,
            SceneRenderMode::GpuNativeProRes => SceneRenderProfile::GpuProRes,
            SceneRenderMode::GpuNativeProRes4444 => SceneRenderProfile::GpuProRes4444,
            SceneRenderMode::GpuNativePngSequence => SceneRenderProfile::GpuPngSequence,
            SceneRenderMode::GpuNativeCurrentFramePng => SceneRenderProfile::Gpu,
        }
    }

    const fn adds_media_pool_clip(self) -> bool {
        !matches!(self, SceneRenderMode::GpuNativePngSequence)
    }

    const fn preserves_alpha_output(self) -> bool {
        matches!(
            self,
            SceneRenderMode::GpuNativeProRes4444
                | SceneRenderMode::GpuNativePngSequence
                | SceneRenderMode::GpuNativeCurrentFramePng
        )
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum SceneLivePreviewQuality {
    P144,
    P240,
    P360,
    P480,
}

impl SceneLivePreviewQuality {
    const fn label(self) -> &'static str {
        match self {
            SceneLivePreviewQuality::P144 => "144p",
            SceneLivePreviewQuality::P240 => "240p",
            SceneLivePreviewQuality::P360 => "360p",
            SceneLivePreviewQuality::P480 => "480p",
        }
    }

    const fn max_dim(self) -> u32 {
        match self {
            SceneLivePreviewQuality::P144 => SCENE_LIVE_PREVIEW_144P_MAX_DIM,
            SceneLivePreviewQuality::P240 => SCENE_LIVE_PREVIEW_240P_MAX_DIM,
            SceneLivePreviewQuality::P360 => SCENE_LIVE_PREVIEW_360P_MAX_DIM,
            SceneLivePreviewQuality::P480 => SCENE_LIVE_PREVIEW_480P_MAX_DIM,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImportedClipKind {
    Image,
    Video,
}

impl ImportedClipKind {
    const fn label(self) -> &'static str {
        match self {
            ImportedClipKind::Image => "Image",
            ImportedClipKind::Video => "Video",
        }
    }
}

#[derive(Clone)]
struct LoadedPreview {
    image: Arc<RenderImage>,
    bgra: Option<Arc<Vec<u8>>>,
    /// Reusable IOSurface-backed BGRA pixel buffer for `paint_bgra_frame_anica`.
    #[cfg(target_os = "macos")]
    bgra_surface: Option<Arc<SendableCVPixelBuffer>>,
    /// Reusable D3D11 shared texture for `paint_bgra_frame_anica` on Windows.
    #[cfg(target_os = "windows")]
    d3d_surface: Option<motionloom::WindowsD3DSharedSurface>,
    width: u32,
    height: u32,
}

#[derive(Clone)]
struct ImportedClip {
    name: String,
    path: String,
    kind: ImportedClipKind,
    duration: Duration,
}

#[derive(Clone)]
struct VfxAssetItem {
    name: String,
    path: String,
    kind: ImportedClipKind,
    duration: Duration,
    preview: Option<LoadedPreview>,
    error: Option<String>,
}

#[derive(Clone)]
struct VfxAssetContextMenu {
    path: String,
    x: f32,
    y: f32,
}

#[derive(Clone, Debug)]
struct SceneRenderProgressUi {
    label: &'static str,
    rendered_frames: u32,
    total_frames: u32,
}

impl SceneRenderProgressUi {
    fn percent(&self) -> u32 {
        if self.total_frames == 0 {
            return 0;
        }
        ((self.rendered_frames as f32 / self.total_frames as f32) * 100.0)
            .round()
            .clamp(0.0, 100.0) as u32
    }
}

#[derive(Clone, Debug)]
struct SceneLiveTarget {
    id: String,
    tag: String,
    attrs: Vec<String>,
}

#[derive(Clone, Debug)]
struct SceneLiveTargetOption {
    id: String,
    label: String,
}

impl SelectItem for SceneLiveTargetOption {
    type Value = String;

    fn title(&self) -> SharedString {
        SharedString::from(self.label.clone())
    }

    fn value(&self) -> &Self::Value {
        &self.id
    }
}

#[derive(Clone, Debug)]
struct GlbSkeletonInspectReport {
    summary: String,
    retarget_draft: String,
    model_profile_draft: String,
    calibrated_model_profile_draft: String,
    actors: Vec<GlbSkeletonActorReport>,
}

#[derive(Clone, Debug)]
struct GlbSkeletonActorReport {
    actor_id: String,
    model: String,
    resolved_path: String,
    vertex_count: usize,
    triangle_count: usize,
    node_count: usize,
    joint_count: usize,
    weighted_vertex_count: usize,
    has_inverse_bind_matrices: bool,
    mapped_bones: Vec<String>,
    missing_humanoid_bones: Vec<String>,
    guessed_maps: Vec<(String, String)>,
    joint_tree_lines: Vec<String>,
    model_profile_draft: String,
    calibrated_model_profile_draft: String,
    calibration_preview_lines: Vec<String>,
    error: Option<String>,
}

#[derive(Clone, Copy, Debug)]
struct GlbRestPoseBasis {
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
}

#[derive(Clone, Debug)]
struct GlbAxisCalibration {
    axis_lines: Vec<String>,
    preview_lines: Vec<String>,
}

#[derive(Clone, Debug)]
struct GlbAxisBindingScore {
    binding: String,
    score: f32,
}

#[derive(Debug, Error)]
enum MotionLoomPageError {
    #[error("Failed to open preview image: {source}")]
    OpenPreviewImage { source: image::ImageError },
    #[error("Failed to construct runtime preview buffer")]
    BuildRuntimePreviewBuffer,
    #[error(transparent)]
    Thumbnail(#[from] crate::core::thumbnail::ThumbnailError),
}

// Fit-to-container preview image element that renders a source image
// centered inside the available bounds with aspect-ratio preservation.
struct FitPreviewImageElement {
    image: Arc<RenderImage>,
    bgra: Option<Arc<Vec<u8>>>,
    /// Cached IOSurface-backed BGRA surface for the extended surface paint path.
    #[cfg(target_os = "macos")]
    bgra_surface: Option<Arc<SendableCVPixelBuffer>>,
    /// Cached D3D11 shared texture for the extended surface paint path on Windows.
    #[cfg(target_os = "windows")]
    d3d_surface: Option<motionloom::WindowsD3DSharedSurface>,
    width: u32,
    height: u32,
}

impl FitPreviewImageElement {
    fn from_preview(preview: LoadedPreview) -> Self {
        Self {
            image: preview.image,
            bgra: preview.bgra,
            #[cfg(target_os = "macos")]
            bgra_surface: preview.bgra_surface,
            #[cfg(target_os = "windows")]
            d3d_surface: preview.d3d_surface,
            width: preview.width,
            height: preview.height,
        }
    }

    // Calculate destination bounds that fit the image into the container
    // while preserving the original aspect ratio.
    fn fitted_bounds(&self, bounds: gpui::Bounds<gpui::Pixels>) -> gpui::Bounds<gpui::Pixels> {
        let container_w: f32 = bounds.size.width.into();
        let container_h: f32 = bounds.size.height.into();
        let frame_w = self.width as f32;
        let frame_h = self.height as f32;
        if frame_w == 0.0 || frame_h == 0.0 {
            return bounds;
        }

        let fit_scale = (container_w / frame_w).min(container_h / frame_h);
        let dest_w = frame_w * fit_scale;
        let dest_h = frame_h * fit_scale;
        let offset_x = (container_w - dest_w) * 0.5;
        let offset_y = (container_h - dest_h) * 0.5;

        gpui::Bounds::new(
            gpui::point(
                bounds.origin.x + gpui::px(offset_x),
                bounds.origin.y + gpui::px(offset_y),
            ),
            gpui::size(gpui::px(dest_w), gpui::px(dest_h)),
        )
    }
}

impl Element for FitPreviewImageElement {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<gpui::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut gpui::App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let style = Style {
            size: gpui::Size {
                width: gpui::Length::Definite(gpui::DefiniteLength::Fraction(1.0)),
                height: gpui::Length::Definite(gpui::DefiniteLength::Fraction(1.0)),
            },
            ..Default::default()
        };
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: gpui::Bounds<gpui::Pixels>,
        _state: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut gpui::App,
    ) -> Self::PrepaintState {
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: gpui::Bounds<gpui::Pixels>,
        _layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        _cx: &mut gpui::App,
    ) {
        let dest_bounds = self.fitted_bounds(bounds);

        // Prefer the extended BGRA surface path; it avoids CPU-side reallocation
        // per paint and matches the video-preview rendering pipeline.
        #[cfg(target_os = "macos")]
        {
            if let Some(surface) = self.bgra_surface.as_ref() {
                window.paint_bgra_frame_anica(
                    dest_bounds,
                    gpui::BgraFrameSurface::CvPixelBuffer(surface.0.clone()),
                    gpui::SurfaceExParams_anica::default(),
                );
                return;
            }

            if let Some(bgra) = self.bgra.as_ref()
                && let Some(surface) = video_engine::Video::build_surface_bgra_copy_from_data(
                    self.width,
                    self.height,
                    bgra,
                )
            {
                window.paint_bgra_frame_anica(
                    dest_bounds,
                    gpui::BgraFrameSurface::CvPixelBuffer(surface),
                    gpui::SurfaceExParams_anica::default(),
                );
                return;
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Some(surface) = self.d3d_surface.as_ref() {
                if let Some(devices) = window.d3d11_devices_anica() {
                    if let Some(frame_surface) = gpui::BgraFrameSurface::from_shared_handle_bgra(
                        &devices,
                        surface.handle.0,
                        surface.width,
                        surface.height,
                    ) {
                        window.paint_bgra_frame_anica(
                            dest_bounds,
                            frame_surface,
                            gpui::SurfaceExParams_anica::default(),
                        );
                        return;
                    }
                }
            }
        }

        let _ = window.paint_image(
            dest_bounds,
            gpui::Corners::default(),
            self.image.clone(),
            0,
            false,
        );
    }
}

impl IntoElement for FitPreviewImageElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

#[derive(Clone, Copy)]
struct PreviewVisibilityPolicy {
    overlay_open: bool,
    app_active: bool,
    host_focused: bool,
}

impl PreviewVisibilityPolicy {
    fn should_show(self) -> bool {
        if self.overlay_open {
            return false;
        }

        #[cfg(any(target_os = "macos", target_os = "windows"))]
        {
            self.app_active || self.host_focused
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            let _ = (self.app_active, self.host_focused);
            true
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SceneExternalPreviewWindowBounds {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    decorations: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SceneExternalPreviewDesiredWindow {
    visible: bool,
    bounds: Option<SceneExternalPreviewWindowBounds>,
    painted_at: Option<Instant>,
}

impl Default for SceneExternalPreviewDesiredWindow {
    fn default() -> Self {
        Self {
            visible: false,
            bounds: None,
            painted_at: None,
        }
    }
}

struct ExternalPreviewAnchorElement {
    graph_size: (u32, u32),
    visibility: PreviewVisibilityPolicy,
    desired_window: Arc<Mutex<SceneExternalPreviewDesiredWindow>>,
}

impl ExternalPreviewAnchorElement {
    fn new(
        graph_size: (u32, u32),
        visibility: PreviewVisibilityPolicy,
        desired_window: Arc<Mutex<SceneExternalPreviewDesiredWindow>>,
    ) -> Self {
        Self {
            graph_size,
            visibility,
            desired_window,
        }
    }

    fn fitted_bounds(&self, bounds: gpui::Bounds<gpui::Pixels>) -> gpui::Bounds<gpui::Pixels> {
        let container_w: f32 = bounds.size.width.into();
        let container_h: f32 = bounds.size.height.into();
        let graph_w = self.graph_size.0.max(1) as f32;
        let graph_h = self.graph_size.1.max(1) as f32;
        let fit_scale = (container_w / graph_w)
            .min(container_h / graph_h)
            .max(0.001);
        let dest_w = graph_w * fit_scale;
        let dest_h = graph_h * fit_scale;
        let offset_x = (container_w - dest_w) * 0.5;
        let offset_y = (container_h - dest_h) * 0.5;

        gpui::Bounds::new(
            gpui::point(
                bounds.origin.x + gpui::px(offset_x),
                bounds.origin.y + gpui::px(offset_y),
            ),
            gpui::size(gpui::px(dest_w), gpui::px(dest_h)),
        )
    }

    fn clear_desired_window(&self) {
        if let Ok(mut desired) = self.desired_window.lock() {
            *desired = SceneExternalPreviewDesiredWindow::default();
        }
    }

    fn minimum_visible_top(viewport_h: f32) -> f32 {
        // The WGPU preview is an OS window, so GPUI scroll clipping cannot mask it.
        // Keep it below the fixed Studio header when its in-page slot scrolls away.
        (viewport_h * 0.16).max(170.0)
    }

    fn pinned_origin(
        local_x: f32,
        local_y: f32,
        width: f32,
        height: f32,
        viewport_w: f32,
        viewport_h: f32,
    ) -> (f32, f32) {
        let minimum_top = Self::minimum_visible_top(viewport_h);
        let maximum_x = (viewport_w - width).max(0.0);
        let maximum_y = (viewport_h - height).max(minimum_top);
        (
            local_x.clamp(0.0, maximum_x),
            local_y.clamp(minimum_top, maximum_y),
        )
    }
}

impl Element for ExternalPreviewAnchorElement {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<gpui::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut gpui::App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let style = Style {
            size: gpui::Size {
                width: gpui::Length::Definite(gpui::DefiniteLength::Fraction(1.0)),
                height: gpui::Length::Definite(gpui::DefiniteLength::Fraction(1.0)),
            },
            ..Default::default()
        };
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: gpui::Bounds<gpui::Pixels>,
        _state: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut gpui::App,
    ) -> Self::PrepaintState {
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: gpui::Bounds<gpui::Pixels>,
        _layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        _cx: &mut gpui::App,
    ) {
        if !self.visibility.should_show() {
            self.clear_desired_window();
            return;
        }

        let bounds = self.fitted_bounds(bounds);
        let window_bounds = window.bounds();
        let viewport_w = f32::from(window.viewport_size().width);
        let viewport_h = f32::from(window.viewport_size().height);
        let local_x = f32::from(bounds.origin.x);
        let local_y = f32::from(bounds.origin.y);
        let local_w = f32::from(bounds.size.width);
        let local_h = f32::from(bounds.size.height);
        let (pinned_x, pinned_y) =
            Self::pinned_origin(local_x, local_y, local_w, local_h, viewport_w, viewport_h);
        let content_offset_y = (f32::from(window_bounds.size.height)
            - f32::from(window.viewport_size().height))
        .max(0.0);
        let (calibration_x, calibration_y) = motionloom_external_preview_offset();
        let host_x = f64::from(window_bounds.origin.x) + f64::from(pinned_x) + calibration_x;
        let host_y = f64::from(window_bounds.origin.y)
            + f64::from(content_offset_y)
            + f64::from(pinned_y)
            + calibration_y;
        let host_w = f64::from(bounds.size.width);
        let host_h = f64::from(bounds.size.height);
        if motionloom_external_preview_debug_enabled() {
            eprintln!(
                "external MotionLoom preview attach: window=({:.1},{:.1} {:.1}x{:.1}) viewport={:.1}x{:.1} content_offset_y={content_offset_y:.1} local=({:.1},{:.1} {:.1}x{:.1}) pinned=({pinned_x:.1},{pinned_y:.1}) offset=({calibration_x:.1},{calibration_y:.1}) sent=({host_x:.1},{host_y:.1} {host_w:.1}x{host_h:.1}) scale={:.2}",
                f64::from(window_bounds.origin.x),
                f64::from(window_bounds.origin.y),
                f64::from(window_bounds.size.width),
                f64::from(window_bounds.size.height),
                f64::from(window.viewport_size().width),
                f64::from(window.viewport_size().height),
                f64::from(bounds.origin.x),
                f64::from(bounds.origin.y),
                host_w,
                host_h,
                window.scale_factor(),
            );
        }
        if let Ok(mut desired) = self.desired_window.lock() {
            *desired = SceneExternalPreviewDesiredWindow {
                visible: true,
                bounds: Some(SceneExternalPreviewWindowBounds {
                    x: host_x,
                    y: host_y,
                    width: host_w,
                    height: host_h,
                    decorations: false,
                }),
                painted_at: Some(Instant::now()),
            };
        }
    }
}

impl IntoElement for ExternalPreviewAnchorElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

pub struct MotionLoomPage {
    pub global: Entity<GlobalState>,
    clips: Vec<ImportedClip>,
    selected_idx: Option<usize>,
    preview_frame: u32,
    status_line: String,
    script_text: String,
    script_input: Option<Entity<InputState>>,
    script_input_sub: Option<Subscription>,
    script_input_syncing: bool,
    script_input_needs_sync: bool,
    motionloom_script_revision: u64,
    motionloom_apply_revision: u64,
    motionloom_render_revision: u64,
    graph_runtime: Option<RuntimeProgram>,
    scene_live_preview_cache_key: Option<SceneLivePreviewCacheKey>,
    scene_live_preview_cache_image: Option<LoadedPreview>,
    scene_live_preview_frame_cache: HashMap<SceneLivePreviewCacheKey, LoadedPreview>,
    scene_live_preview_frame_cache_order: VecDeque<SceneLivePreviewCacheKey>,
    scene_live_preview_frame_cache_bytes: usize,
    scene_live_parsed_script_hash: Option<u64>,
    scene_live_parsed_graph: Option<GraphScript>,
    scene_live_preview_quality: SceneLivePreviewQuality,
    scene_live_preview_status: String,
    scene_live_render_defer_until: Option<Instant>,
    scene_live_render_defer_token: u64,
    scene_live_async_render_key: Option<SceneLivePreviewCacheKey>,
    scene_live_async_render_token: u64,
    scene_live_preview_tx: Sender<SceneLivePreviewRequest>,
    scene_external_preview_host: Option<SceneExternalPreviewHost>,
    _scene_external_preview_process: Option<SceneExternalPreviewProcess>,
    scene_external_preview_script_hash: Option<u64>,
    scene_external_preview_quality: Option<SceneLivePreviewQuality>,
    scene_external_preview_asset_roots: Vec<PathBuf>,
    scene_external_preview_heartbeat_scheduled: bool,
    scene_external_preview_host_focused: bool,
    scene_external_preview_render_debug_count: u32,
    scene_external_preview_desired_window: Arc<Mutex<SceneExternalPreviewDesiredWindow>>,
    scene_external_preview_last_window: Option<SceneExternalPreviewDesiredWindow>,
    world_live_preview_tx: Sender<WorldLivePreviewRequest>,
    scene_live_prerender_token: u64,
    scene_live_prerendering: bool,
    scene_live_prerender_progress: Option<(u32, u32)>,
    scene_live_prerender_cancel: Option<Arc<AtomicBool>>,
    scene_render_progress: Option<SceneRenderProgressUi>,
    scene_render_cancel: Option<Arc<AtomicBool>>,
    scene_render_log: Vec<String>,
    scene_render_log_collapsed: bool,
    scene_live_knob_node_id: String,
    scene_live_knob_attr: String,
    scene_live_target_select: Option<Entity<SelectState<SearchableVec<SceneLiveTargetOption>>>>,
    scene_live_target_select_sub: Option<Subscription>,
    scene_live_attr_menu_open: bool,
    scene_live_keyframe_show_as_frame: bool,
    scene_live_preview_overrides: HashMap<(String, String), f32>,
    scene_live_gizmo_mode: SceneLiveGizmoMode,
    scene_live_gizmo_drag: Option<SceneLiveGizmoDrag>,
    scene_live_knob_input: Option<Entity<InputState>>,
    scene_live_knob_input_sub: Option<Subscription>,
    scene_live_knob_input_syncing: bool,
    preview_frame_input: Option<Entity<InputState>>,
    preview_frame_input_sub: Option<Subscription>,
    preview_frame_input_syncing: bool,
    scene_live_target_offset: usize,
    scene_live_tag_filters: HashSet<String>,
    preview_playing: bool,
    preview_play_token: u64,
    preview_last_tick: Option<Instant>,
    preview_frame_accum: f32,
    import_modal_open: bool,
    asset_modal_open: bool,
    asset_folder: Option<PathBuf>,
    asset_items: Vec<VfxAssetItem>,
    asset_selected_idx: Option<usize>,
    asset_context_menu: Option<VfxAssetContextMenu>,
    scene_template_select: Option<Entity<SelectState<SearchableVec<String>>>>,
    scene_template_selected_label: String,
    scene_template_number: String,
    scene_template_number_input: Option<Entity<InputState>>,
    scene_template_number_input_sub: Option<Subscription>,
    scene_template_modal_open: bool,
    scene_render_modal_open: bool,
    pending_non_alpha_scene_render_mode: Option<SceneRenderMode>,
    glb_inspector_modal_open: bool,
    glb_inspector_report: Option<GlbSkeletonInspectReport>,
    // Template picker state
    template_modal_open: bool,
    template_selected: Vec<LayerEffectTemplateKind>,
    template_add_time_parameter: bool,
    template_add_curve_parameter: bool,
}

impl MotionLoomPage {
    pub fn new(global: Entity<GlobalState>, cx: &mut Context<Self>) -> Self {
        cx.observe(&global, |_this, _global, cx| {
            cx.notify();
        })
        .detach();

        let (initial_script, script_revision, apply_revision, render_revision) = {
            let gs = global.read(cx);
            (
                gs.motionloom_scene_script().to_string(),
                gs.motionloom_scene_script_revision(),
                gs.motionloom_scene_apply_revision(),
                gs.motionloom_scene_render_revision(),
            )
        };
        let (scene_external_preview_host, scene_external_preview_process) =
            SceneExternalPreviewHost::from_env_or_auto_spawn();

        Self {
            global,
            clips: Vec::new(),
            selected_idx: None,
            preview_frame: 0,
            status_line: "Import a video or still to start building a MotionLoom graph."
                .to_string(),
            script_text: initial_script,
            script_input: None,
            script_input_sub: None,
            script_input_syncing: false,
            script_input_needs_sync: false,
            motionloom_script_revision: script_revision,
            motionloom_apply_revision: apply_revision,
            motionloom_render_revision: render_revision,
            graph_runtime: None,
            scene_live_preview_cache_key: None,
            scene_live_preview_cache_image: None,
            scene_live_preview_frame_cache: HashMap::new(),
            scene_live_preview_frame_cache_order: VecDeque::new(),
            scene_live_preview_frame_cache_bytes: 0,
            scene_live_parsed_script_hash: None,
            scene_live_parsed_graph: None,
            scene_live_preview_quality: SceneLivePreviewQuality::P360,
            scene_live_preview_status: "Preview: idle".to_string(),
            scene_live_render_defer_until: None,
            scene_live_render_defer_token: 0,
            scene_live_async_render_key: None,
            scene_live_async_render_token: 0,
            scene_live_preview_tx: Self::spawn_scene_live_preview_worker(),
            scene_external_preview_host,
            _scene_external_preview_process: scene_external_preview_process,
            scene_external_preview_script_hash: None,
            scene_external_preview_quality: None,
            scene_external_preview_asset_roots: Vec::new(),
            scene_external_preview_heartbeat_scheduled: false,
            scene_external_preview_host_focused: false,
            scene_external_preview_render_debug_count: 0,
            scene_external_preview_desired_window: Arc::new(Mutex::new(
                SceneExternalPreviewDesiredWindow::default(),
            )),
            scene_external_preview_last_window: None,
            world_live_preview_tx: Self::spawn_world_live_preview_worker(),
            scene_live_prerender_token: 0,
            scene_live_prerendering: false,
            scene_live_prerender_progress: None,
            scene_live_prerender_cancel: None,
            scene_render_progress: None,
            scene_render_cancel: None,
            scene_render_log: Vec::new(),
            scene_render_log_collapsed: false,
            scene_live_knob_node_id: DEFAULT_SCENE_LIVE_NODE_ID.to_string(),
            scene_live_knob_attr: DEFAULT_SCENE_LIVE_ATTR.to_string(),
            scene_live_target_select: None,
            scene_live_target_select_sub: None,
            scene_live_attr_menu_open: false,
            scene_live_keyframe_show_as_frame: false,
            scene_live_preview_overrides: HashMap::new(),
            scene_live_gizmo_mode: SceneLiveGizmoMode::Move,
            scene_live_gizmo_drag: None,
            scene_live_knob_input: None,
            scene_live_knob_input_sub: None,
            scene_live_knob_input_syncing: false,
            preview_frame_input: None,
            preview_frame_input_sub: None,
            preview_frame_input_syncing: false,
            scene_live_target_offset: 0,
            scene_live_tag_filters: HashSet::new(),
            preview_playing: false,
            preview_play_token: 0,
            preview_last_tick: None,
            preview_frame_accum: 0.0,
            import_modal_open: false,
            asset_modal_open: false,
            asset_folder: None,
            asset_items: Vec::new(),
            asset_selected_idx: None,
            asset_context_menu: None,
            scene_template_select: None,
            scene_template_selected_label: DEFAULT_SCENE_TEMPLATE_CATEGORY.to_string(),
            scene_template_number: DEFAULT_SCENE_TEMPLATE_NUMBER.to_string(),
            scene_template_number_input: None,
            scene_template_number_input_sub: None,
            scene_template_modal_open: false,
            scene_render_modal_open: false,
            pending_non_alpha_scene_render_mode: None,
            glb_inspector_modal_open: false,
            glb_inspector_report: None,
            template_modal_open: false,
            template_selected: Vec::new(),
            template_add_time_parameter: false,
            template_add_curve_parameter: false,
        }
    }

    fn is_image_path(path: &str) -> bool {
        let p = path.to_ascii_lowercase();
        p.ends_with(".jpg")
            || p.ends_with(".jpeg")
            || p.ends_with(".png")
            || p.ends_with(".webp")
            || p.ends_with(".bmp")
            || p.ends_with(".gif")
    }

    fn is_video_path(path: &str) -> bool {
        let p = path.to_ascii_lowercase();
        p.ends_with(".mp4")
            || p.ends_with(".mov")
            || p.ends_with(".mkv")
            || p.ends_with(".webm")
            || p.ends_with(".avi")
            || p.ends_with(".flv")
            || p.ends_with(".m4v")
    }

    fn is_supported_clip_path(path: &str) -> bool {
        Self::is_image_path(path) || Self::is_video_path(path)
    }

    fn load_render_image(path: &Path) -> Result<LoadedPreview, MotionLoomPageError> {
        let decoded =
            image::open(path).map_err(|source| MotionLoomPageError::OpenPreviewImage { source })?;
        Self::load_render_image_from_dynamic(decoded)
    }

    fn load_render_image_from_dynamic(
        decoded: image::DynamicImage,
    ) -> Result<LoadedPreview, MotionLoomPageError> {
        let rgba = decoded.to_rgba8();
        let (w, h) = rgba.dimensions();
        // Keep a correct RGBA RenderImage for the rare paint_image fallback path.
        let image = {
            let frames = SmallVec::from_elem(image::Frame::new(rgba.clone()), 1);
            Arc::new(RenderImage::new(frames))
        };
        let mut bgra = rgba.into_raw();
        for px in bgra.chunks_mut(4) {
            px.swap(0, 2);
        }
        let bgra = Arc::new(bgra);
        #[cfg(target_os = "macos")]
        let bgra_surface = Self::build_bgra_surface(w, h, bgra.as_ref());
        Ok(LoadedPreview {
            image,
            bgra: Some(bgra),
            #[cfg(target_os = "macos")]
            bgra_surface,
            #[cfg(target_os = "windows")]
            d3d_surface: None,
            width: w,
            height: h,
        })
    }

    fn loaded_preview_from_bgra(
        width: u32,
        height: u32,
        bgra: Vec<u8>,
    ) -> Result<LoadedPreview, MotionLoomPageError> {
        let bgra = Arc::new(bgra);
        let image = Self::render_image_from_bgra(width, height, bgra.as_ref().clone())?;
        #[cfg(target_os = "macos")]
        let bgra_surface = Self::build_bgra_surface(width, height, bgra.as_ref());
        Ok(LoadedPreview {
            image,
            bgra: Some(bgra),
            #[cfg(target_os = "macos")]
            bgra_surface,
            #[cfg(target_os = "windows")]
            d3d_surface: None,
            width,
            height,
        })
    }

    #[cfg(target_os = "macos")]
    fn loaded_preview_from_macos_surface(
        width: u32,
        height: u32,
        surface: SendableCVPixelBuffer,
    ) -> Result<LoadedPreview, MotionLoomPageError> {
        let rgba = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 255]));
        let frames = SmallVec::from_elem(image::Frame::new(rgba), 1);
        Ok(LoadedPreview {
            image: Arc::new(RenderImage::new(frames)),
            bgra: None,
            bgra_surface: Some(Arc::new(surface)),
            #[cfg(target_os = "windows")]
            d3d_surface: None,
            width,
            height,
        })
    }

    #[cfg(target_os = "windows")]
    fn loaded_preview_from_windows_surface(
        width: u32,
        height: u32,
        surface: motionloom::WindowsD3DSharedSurface,
    ) -> Result<LoadedPreview, MotionLoomPageError> {
        let rgba = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 255]));
        let frames = SmallVec::from_elem(image::Frame::new(rgba), 1);
        Ok(LoadedPreview {
            image: Arc::new(RenderImage::new(frames)),
            bgra: None,
            #[cfg(target_os = "macos")]
            bgra_surface: None,
            d3d_surface: Some(surface),
            width,
            height,
        })
    }

    fn render_image_from_bgra(
        width: u32,
        height: u32,
        bgra: Vec<u8>,
    ) -> Result<Arc<RenderImage>, MotionLoomPageError> {
        // Keep BGRA bytes as-is; the app expects BGRA for all preview paths.
        let image_buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, bgra)
            .ok_or(MotionLoomPageError::BuildRuntimePreviewBuffer)?;
        let frames = SmallVec::from_elem(image::Frame::new(image_buffer), 1);
        Ok(Arc::new(RenderImage::new(frames)))
    }

    fn world_asset_root() -> PathBuf {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        for candidate in [
            cwd.join("examples/motionloom/world"),
            cwd.join("anica/examples/motionloom/world"),
            PathBuf::from("examples/motionloom/world"),
            PathBuf::from("anica/examples/motionloom/world"),
        ] {
            if candidate.exists() {
                return candidate;
            }
        }
        PathBuf::from("examples/motionloom/world")
    }

    fn scene_live_asset_roots(&self, cx: &Context<Self>) -> Vec<PathBuf> {
        let gs = self.global.read(cx);
        gs.motionloom_asset_root
            .clone()
            .into_iter()
            .collect::<Vec<_>>()
    }

    fn rgba_image_to_bgra(rgba: RgbaImage) -> (u32, u32, Vec<u8>) {
        let (w, h) = rgba.dimensions();
        let mut bgra = rgba.into_raw();
        for px in bgra.chunks_mut(4) {
            px.swap(0, 2);
        }
        (w, h, bgra)
    }

    /// Create an IOSurface-backed BGRA CVPixelBuffer and copy BGRA data into it.
    /// Returns `None` if allocation fails so callers can fall back to other paths.
    #[cfg(target_os = "macos")]
    fn build_bgra_surface(
        width: u32,
        height: u32,
        bgra: &[u8],
    ) -> Option<Arc<SendableCVPixelBuffer>> {
        let surface = video_engine::Video::create_bgra_surface(width, height)?;
        if video_engine::Video::copy_bgra_into_surface(&surface, width, height, bgra) {
            Some(Arc::new(SendableCVPixelBuffer(surface)))
        } else {
            None
        }
    }

    fn scene_preview_surface_to_live_frame(
        surface: ScenePlatformPreviewSurface,
    ) -> Option<SceneLivePreviewFrame> {
        match surface {
            #[cfg(target_os = "macos")]
            ScenePlatformPreviewSurface::MacOs {
                surface,
                width,
                height,
                ..
            } => Some(SceneLivePreviewFrame::MacOsSurface {
                width,
                height,
                surface: SendableCVPixelBuffer(surface),
            }),
            #[cfg(target_os = "windows")]
            ScenePlatformPreviewSurface::WindowsD3D(surface) => {
                Some(SceneLivePreviewFrame::WindowsSurface {
                    width: surface.width,
                    height: surface.height,
                    surface,
                })
            }
            #[cfg(all(unix, not(target_os = "macos"), not(target_arch = "wasm32")))]
            ScenePlatformPreviewSurface::LinuxDmabuf { .. } => None,
        }
    }

    fn scene_preview_output_to_live_frame(
        surface: ScenePreviewSurface,
    ) -> Result<SceneLivePreviewFrame, String> {
        match surface {
            ScenePreviewSurface::PlatformSurface(surface) => {
                Self::scene_preview_surface_to_live_frame(surface).ok_or_else(|| {
                    "platform preview surface is not displayable on this target yet".to_string()
                })
            }
            ScenePreviewSurface::CpuBgra {
                width,
                height,
                data,
                ..
            } => Ok(SceneLivePreviewFrame::Bgra {
                width,
                height,
                data: (*data).clone(),
            }),
            ScenePreviewSurface::WgpuTexture(_) => {
                Err("wgpu texture preview is not wired to GPUI display yet".to_string())
            }
        }
    }

    fn render_panic_message(panic: Box<dyn Any + Send>) -> String {
        if let Some(message) = panic.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = panic.downcast_ref::<&'static str>() {
            (*message).to_string()
        } else {
            "unknown renderer panic".to_string()
        }
    }

    fn spawn_scene_live_preview_worker() -> Sender<SceneLivePreviewRequest> {
        let (request_tx, request_rx) = mpsc::channel::<SceneLivePreviewRequest>();
        let _ = std::thread::Builder::new()
            .name("motionloom-scene-live-preview".to_string())
            .stack_size(SCENE_LIVE_RENDER_WORKER_STACK_SIZE)
            .spawn(move || {
                let mut preview_engine = std::panic::catch_unwind(|| {
                    pollster::block_on(WgpuPreviewEngine::new_with_cpu_fallback())
                })
                .unwrap_or_else(|_| pollster::block_on(WgpuPreviewEngine::new_cpu_only()));
                let mut graph_cache = WgpuPreviewGraphCache::default();
                while let Ok(mut request) = request_rx.recv() {
                    // Keep preview responsive by discarding stale queued frames.
                    while let Ok(next_request) = request_rx.try_recv() {
                        request = next_request;
                    }
                    if !preview_engine.has_gpu_renderer() {
                        preview_engine = std::panic::catch_unwind(|| {
                            pollster::block_on(WgpuPreviewEngine::new_with_cpu_fallback())
                        })
                        .unwrap_or_else(|_| pollster::block_on(WgpuPreviewEngine::new_cpu_only()));
                    }
                    if request.asset_roots.is_empty() {
                        motionloom::clear_scene_asset_roots();
                    } else {
                        motionloom::set_scene_asset_roots(request.asset_roots.clone());
                    }
                    let script = request.script.clone();
                    let script_hash = request.script_hash;
                    let render_size = request.render_size;
                    let frame = request.frame;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let preview = pollster::block_on(
                            preview_engine.render_script_preview_surface_with_cpu_fallback(
                                &mut graph_cache,
                                &request.script,
                                request.script_hash,
                                request.frame,
                                request.render_size,
                                ScenePreviewSurfaceOptions {
                                    backend: ScenePreviewBackend::PlatformSurface,
                                    ..ScenePreviewSurfaceOptions::default()
                                },
                            ),
                        )
                        .map_err(|err| format!("Scene live render error: {err}"))?;
                        Self::scene_preview_output_to_live_frame(preview.surface)
                            .map(|frame| (frame, preview.warning))
                    }))
                    .unwrap_or_else(|panic| {
                        let panic_message = Self::render_panic_message(panic);
                        preview_engine = pollster::block_on(WgpuPreviewEngine::new_cpu_only());
                        pollster::block_on(
                            preview_engine.render_script_preview_surface_with_cpu_fallback(
                                &mut graph_cache,
                                &script,
                                script_hash,
                                frame,
                                render_size,
                                ScenePreviewSurfaceOptions {
                                    backend: ScenePreviewBackend::CpuBgra,
                                    ..ScenePreviewSurfaceOptions::default()
                                },
                            ),
                        )
                        .map_err(|err| {
                            format!(
                                "Scene live render error: GPU panicked ({panic_message}); CPU fallback failed: {err}"
                            )
                        })
                        .and_then(|preview| {
                            Self::scene_preview_output_to_live_frame(preview.surface).map(|frame| {
                                (
                                    frame,
                                    Some(format!(
                                        "Scene live preview used CPU fallback after GPU panic: {panic_message}"
                                    )),
                                )
                            })
                        })
                    });
                    let _ = request.response_tx.send(result);
                }
            });
        request_tx
    }

    fn spawn_world_live_preview_worker() -> Sender<WorldLivePreviewRequest> {
        let (request_tx, request_rx) = mpsc::channel::<WorldLivePreviewRequest>();
        std::thread::spawn(move || {
            let mut renderer = WorldFrameRenderer::new();
            while let Ok(mut request) = request_rx.recv() {
                while let Ok(next_request) = request_rx.try_recv() {
                    request = next_request;
                }
                let result = pollster::block_on(renderer.render_frame_gpu(
                    &request.graph,
                    request.frame,
                    &request.asset_root,
                ))
                .map_err(|err| format!("World live render error: {err}"))
                .map(Self::rgba_image_to_bgra);
                let _ = request.response_tx.send(result);
            }
        });
        request_tx
    }

    fn uses_pure_world_renderer(raw: &str) -> bool {
        is_world_graph_script(raw)
            && !raw.contains("<Tex")
            && !raw.contains("<Pass")
            && !raw.contains("<Layer")
            && !raw.contains("<Scene")
            && !raw.contains("<Clip")
    }

    fn scene_live_poll_ms(preview_playing: bool) -> u64 {
        if preview_playing {
            SCENE_LIVE_PRERENDER_POLL_MS
        } else {
            SCENE_LIVE_IDLE_POLL_MS
        }
    }

    pub fn suspend_background_rendering(&mut self, cx: &mut Context<Self>) {
        if self.global.read(cx).active_page == AppPage::MotionLoom {
            return;
        }

        self.hide_external_scene_preview();

        let had_preview_work = self.preview_playing
            || self.scene_live_prerendering
            || self.scene_live_async_render_key.is_some()
            || self.scene_live_render_defer_until.is_some();
        if !had_preview_work {
            return;
        }

        self.stop_preview_playback();
        self.cancel_scene_live_prerender();
        self.cancel_scene_live_async_render();
        self.scene_live_render_defer_until = None;
        self.scene_live_render_defer_token = self.scene_live_render_defer_token.wrapping_add(1);
    }

    pub fn hide_external_scene_preview(&mut self) {
        if let Ok(mut desired) = self.scene_external_preview_desired_window.lock() {
            *desired = SceneExternalPreviewDesiredWindow::default();
        }

        let hidden = SceneExternalPreviewDesiredWindow::default();
        if self.scene_external_preview_last_window == Some(hidden) {
            return;
        }

        // Page-level navigation must hard-hide the native WGPU host even when playback stops.
        if let Some(host) = self.scene_external_preview_host.as_ref() {
            motionloom_external_preview_log("send SetWindowVisible false on page leave");
            host.send(PreviewCommand::SetWindowVisible { visible: false });
        }
        self.scene_external_preview_last_window = Some(hidden);
    }

    fn script_hash(script: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        script.hash(&mut hasher);
        hasher.finish()
    }

    fn invalidate_scene_live_preview_cache(&mut self) {
        if let Some(cancel) = self.scene_live_prerender_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.clear_scene_live_preview_overrides();
        self.scene_live_gizmo_drag = None;
        self.clear_scene_live_preview_images();
        self.scene_live_parsed_script_hash = None;
        self.scene_live_parsed_graph = None;
        self.scene_live_async_render_key = None;
        self.scene_live_async_render_token = self.scene_live_async_render_token.wrapping_add(1);
        self.scene_live_prerender_token = self.scene_live_prerender_token.wrapping_add(1);
        self.scene_live_prerendering = false;
        self.scene_live_prerender_progress = None;
        self.scene_live_render_defer_until = None;
        self.scene_live_render_defer_token = self.scene_live_render_defer_token.wrapping_add(1);
    }

    fn invalidate_scene_live_preview_render_only(&mut self) {
        self.cancel_scene_live_prerender();
        self.cancel_scene_live_async_render();
        self.scene_live_preview_cache_key = None;
        self.clear_scene_live_preview_frame_cache();
        self.scene_live_parsed_script_hash = None;
        self.scene_live_parsed_graph = None;
        self.scene_live_render_defer_until = None;
        self.scene_live_render_defer_token = self.scene_live_render_defer_token.wrapping_add(1);
    }

    fn set_scene_live_preview_override(&mut self, node: String, property: String, value: f32) {
        self.scene_live_preview_overrides
            .insert((node.clone(), property.clone()), value);
        let protocol = MotionLoomExternalPreviewProtocol::current();
        if protocol.allows_interaction()
            && self.scene_external_preview_script_hash.is_some()
            && let Some(host) = self.scene_external_preview_host.as_ref()
        {
            host.send(PreviewCommand::SetOverride {
                node,
                property,
                value,
            });
        }
    }

    fn clear_scene_live_preview_overrides(&mut self) {
        let protocol = MotionLoomExternalPreviewProtocol::current();
        if protocol.allows_interaction()
            && self.scene_external_preview_script_hash.is_some()
            && let Some(host) = self.scene_external_preview_host.as_ref()
        {
            for (node, property) in self.scene_live_preview_overrides.keys() {
                host.send(PreviewCommand::ClearOverride {
                    node: node.clone(),
                    property: property.clone(),
                });
            }
        }
        self.scene_live_preview_overrides.clear();
    }

    fn clear_scene_live_preview_images(&mut self) {
        self.scene_live_preview_cache_key = None;
        self.scene_live_preview_cache_image = None;
        self.clear_scene_live_preview_frame_cache();
    }

    fn clear_scene_live_preview_frame_cache(&mut self) {
        self.scene_live_preview_frame_cache.clear();
        self.scene_live_preview_frame_cache_order.clear();
        self.scene_live_preview_frame_cache_bytes = 0;
    }

    fn scene_live_preview_cache_bytes(preview: &LoadedPreview) -> usize {
        preview
            .bgra
            .as_ref()
            .map(|bgra| bgra.len())
            .unwrap_or_else(|| {
                preview.width as usize * preview.height as usize * std::mem::size_of::<u32>()
            })
    }

    fn insert_scene_live_preview_frame_cache(
        &mut self,
        key: SceneLivePreviewCacheKey,
        preview: LoadedPreview,
    ) {
        let preview_bytes = Self::scene_live_preview_cache_bytes(&preview);
        if !self.scene_live_preview_frame_cache.contains_key(&key) {
            self.scene_live_preview_frame_cache_order.push_back(key);
        }
        if let Some(old_preview) = self.scene_live_preview_frame_cache.insert(key, preview) {
            self.scene_live_preview_frame_cache_bytes = self
                .scene_live_preview_frame_cache_bytes
                .saturating_sub(Self::scene_live_preview_cache_bytes(&old_preview));
        }
        self.scene_live_preview_frame_cache_bytes = self
            .scene_live_preview_frame_cache_bytes
            .saturating_add(preview_bytes);
        while (self.scene_live_preview_frame_cache.len() > SCENE_LIVE_PREVIEW_FRAME_CACHE_CAPACITY
            || self.scene_live_preview_frame_cache_bytes > SCENE_LIVE_PREVIEW_FRAME_CACHE_MAX_BYTES)
            && self.scene_live_preview_frame_cache.len() > 1
        {
            let Some(old_key) = self.scene_live_preview_frame_cache_order.pop_front() else {
                break;
            };
            if let Some(old_preview) = self.scene_live_preview_frame_cache.remove(&old_key) {
                self.scene_live_preview_frame_cache_bytes = self
                    .scene_live_preview_frame_cache_bytes
                    .saturating_sub(Self::scene_live_preview_cache_bytes(&old_preview));
            }
        }
    }

    fn scene_live_cached_frame_ready(&self, script_hash: u64, frame: u32) -> bool {
        let quality = self.scene_live_effective_preview_quality();
        self.scene_live_preview_cache_key
            .map(|key| key.0 == script_hash && key.1 == frame && key.2 == quality)
            .unwrap_or(false)
            || self
                .scene_live_preview_frame_cache
                .keys()
                .any(|key| key.0 == script_hash && key.1 == frame && key.2 == quality)
    }

    fn has_scene_live_preview_state(&self) -> bool {
        self.scene_live_preview_cache_key.is_some()
            || self.scene_live_preview_cache_image.is_some()
            || !self.scene_live_preview_frame_cache.is_empty()
            || self.scene_live_async_render_key.is_some()
            || self.scene_live_prerendering
            || self.scene_live_render_defer_until.is_some()
    }

    fn cancel_scene_live_prerender(&mut self) {
        if let Some(cancel) = self.scene_live_prerender_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.scene_live_prerender_token = self.scene_live_prerender_token.wrapping_add(1);
        self.scene_live_prerendering = false;
        self.scene_live_prerender_progress = None;
    }

    fn cancel_scene_live_async_render(&mut self) {
        self.scene_live_async_render_key = None;
        self.scene_live_async_render_token = self.scene_live_async_render_token.wrapping_add(1);
    }

    fn cancel_scene_render_job(&mut self) {
        if let Some(cancel) = self.scene_render_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
            self.push_scene_render_log("Render cancel requested.".to_string());
        }
        self.scene_render_progress = None;
        self.status_line = "Render cancel requested.".to_string();
    }

    fn defer_scene_live_preview_render(&mut self, cx: &mut Context<Self>, delay_ms: u64) {
        self.cancel_scene_live_prerender();
        self.cancel_scene_live_async_render();
        self.scene_live_render_defer_until = Some(Instant::now() + Duration::from_millis(delay_ms));
        self.scene_live_render_defer_token = self.scene_live_render_defer_token.wrapping_add(1);
        let token = self.scene_live_render_defer_token;

        cx.spawn(async move |view, cx| {
            Timer::after(Duration::from_millis(delay_ms)).await;
            let _ = view.update(cx, |this, cx| {
                if this.scene_live_render_defer_token == token {
                    this.scene_live_render_defer_until = None;
                    this.scene_live_preview_cache_key = None;
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn scene_live_preview_output_size(final_size: (u32, u32), max_dim: Option<u32>) -> (u32, u32) {
        let final_w = final_size.0.max(1);
        let final_h = final_size.1.max(1);
        let Some(max_dim) = max_dim.map(|dim| dim.max(1)) else {
            return (final_w, final_h);
        };
        if final_w <= max_dim && final_h <= max_dim {
            return (final_w, final_h);
        }

        let scale = (max_dim as f32 / final_w as f32).min(max_dim as f32 / final_h as f32);
        (
            (final_w as f32 * scale).round().max(1.0) as u32,
            (final_h as f32 * scale).round().max(1.0) as u32,
        )
    }

    fn scene_live_effective_preview_quality(&self) -> SceneLivePreviewQuality {
        self.scene_live_preview_quality
    }

    fn scene_external_preview_active(&self) -> bool {
        self.scene_external_preview_host.is_some()
    }

    fn log_scene_external_render_checkpoint(&mut self, label: &str) {
        // Windows stack overflows happen before panic hooks run, so keep early render breadcrumbs.
        if !cfg!(target_os = "windows") || self.scene_external_preview_render_debug_count >= 80 {
            return;
        }
        self.scene_external_preview_render_debug_count += 1;
        motionloom_external_preview_log(format!(
            "render checkpoint {label} frame={} host={} protocol={:?}",
            self.preview_frame,
            self.scene_external_preview_host.is_some(),
            MotionLoomExternalPreviewProtocol::current()
        ));
    }

    fn scene_external_preview_hidden_by_overlay(&self) -> bool {
        self.import_modal_open
            || self.asset_modal_open
            || self.scene_template_modal_open
            || self.scene_render_modal_open
            || self.glb_inspector_modal_open
            || self.template_modal_open
            || self.asset_context_menu.is_some()
    }

    fn scene_external_preview_visibility_policy(&self, window: &Window) -> PreviewVisibilityPolicy {
        PreviewVisibilityPolicy {
            overlay_open: self.scene_external_preview_hidden_by_overlay(),
            app_active: window.is_window_active(),
            host_focused: self.scene_external_preview_host_focused,
        }
    }

    fn flush_scene_external_preview_window_state(
        &mut self,
        host: &SceneExternalPreviewHost,
        policy: PreviewVisibilityPolicy,
    ) {
        let protocol = MotionLoomExternalPreviewProtocol::current();
        if !protocol.allows_window() {
            self.scene_external_preview_last_window = None;
            return;
        }

        let desired = if policy.should_show() {
            let desired = self
                .scene_external_preview_desired_window
                .lock()
                .map(|desired| *desired)
                .unwrap_or_default();
            if desired
                .painted_at
                .map(|painted_at| painted_at.elapsed() <= Duration::from_millis(300))
                .unwrap_or(false)
            {
                desired
            } else {
                SceneExternalPreviewDesiredWindow::default()
            }
        } else {
            SceneExternalPreviewDesiredWindow::default()
        };

        if self.scene_external_preview_last_window == Some(desired) {
            return;
        }

        let last_visible = self
            .scene_external_preview_last_window
            .map(|last| last.visible)
            .unwrap_or(false);
        let last_bounds = self
            .scene_external_preview_last_window
            .and_then(|last| last.bounds);

        if !desired.visible {
            if last_visible {
                motionloom_external_preview_log("send SetWindowVisible false");
                host.send(PreviewCommand::SetWindowVisible { visible: false });
            }
            self.scene_external_preview_last_window = Some(desired);
            return;
        }

        if desired.bounds != last_bounds {
            if let Some(bounds) = desired.bounds {
                motionloom_external_preview_log(format!("send SetWindowBounds {bounds:?}"));
                host.send(PreviewCommand::SetWindowBounds {
                    x: bounds.x,
                    y: bounds.y,
                    width: bounds.width,
                    height: bounds.height,
                    decorations: bounds.decorations,
                });
            }
        }

        if !last_visible {
            motionloom_external_preview_log("send SetWindowVisible true");
            host.send(PreviewCommand::SetWindowVisible { visible: true });
        }

        self.scene_external_preview_last_window = Some(desired);
    }

    fn scene_external_preview_quality(quality: SceneLivePreviewQuality) -> WgpuPreviewQuality {
        match quality {
            SceneLivePreviewQuality::P480 => WgpuPreviewQuality::Balanced,
            SceneLivePreviewQuality::P360 => WgpuPreviewQuality::Speed,
            SceneLivePreviewQuality::P240 => WgpuPreviewQuality::HighSpeed,
            SceneLivePreviewQuality::P144 => WgpuPreviewQuality::UltraSpeed,
        }
    }

    fn schedule_scene_external_preview_heartbeat(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.scene_external_preview_host.is_none()
            || self.scene_external_preview_heartbeat_scheduled
        {
            return;
        }
        self.scene_external_preview_heartbeat_scheduled = true;
        cx.spawn_in(window, async move |view, window| {
            Timer::after(Duration::from_millis(100)).await;
            view.update_in(window, |this, window, cx| {
                this.scene_external_preview_heartbeat_scheduled = false;
                let Some(host) = this.scene_external_preview_host.clone() else {
                    return;
                };
                this.drain_scene_external_preview_events(&host, window, cx);
                let policy = this.scene_external_preview_visibility_policy(window);
                this.flush_scene_external_preview_window_state(&host, policy);
                if policy.should_show() {
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    fn sync_external_scene_preview(
        &mut self,
        script: &str,
        script_hash: u64,
        frame: u32,
        quality: SceneLivePreviewQuality,
        asset_roots: &[PathBuf],
    ) {
        let Some(host) = self.scene_external_preview_host.clone() else {
            return;
        };
        let protocol = MotionLoomExternalPreviewProtocol::current();
        if protocol != MotionLoomExternalPreviewProtocol::Script
            && self.scene_external_preview_asset_roots != asset_roots
        {
            self.scene_external_preview_asset_roots = asset_roots.to_vec();
            motionloom_external_preview_log(format!(
                "send SetAssetRoots count={}",
                asset_roots.len()
            ));
            host.send(PreviewCommand::SetAssetRoots {
                roots: asset_roots
                    .iter()
                    .map(|path| path.to_string_lossy().to_string())
                    .collect(),
            });
        }
        if self.scene_external_preview_script_hash != Some(script_hash) {
            self.scene_external_preview_script_hash = Some(script_hash);
            motionloom_external_preview_log(format!(
                "send LoadScript bytes={} hash={script_hash}",
                script.len()
            ));
            host.send(PreviewCommand::LoadScript {
                script: script.to_string(),
                source: Some("anica-code-block".to_string()),
            });
        }
        if protocol != MotionLoomExternalPreviewProtocol::Script
            && self.scene_external_preview_quality != Some(quality)
        {
            self.scene_external_preview_quality = Some(quality);
            motionloom_external_preview_log(format!("send SetQuality {quality:?}"));
            host.send(PreviewCommand::SetQuality {
                quality: Self::scene_external_preview_quality(quality),
            });
        }
        if protocol.allows_frame() {
            motionloom_external_preview_log(format!("send SetFrame {frame}"));
            host.send(PreviewCommand::SetFrame { frame });
        }
        for ((node, property), value) in &self.scene_live_preview_overrides {
            if protocol == MotionLoomExternalPreviewProtocol::Script {
                continue;
            }
            motionloom_external_preview_log(format!("send SetOverride {node}.{property}={value}"));
            host.send(PreviewCommand::SetOverride {
                node: node.clone(),
                property: property.clone(),
                value: *value,
            });
        }
        let graph_size = parse_graph_script(script)
            .ok()
            .map(|graph| graph.render_size.unwrap_or(graph.size))
            .unwrap_or((1280, 720));
        let mode = match self.scene_live_gizmo_mode {
            SceneLiveGizmoMode::Move => PreviewInteractionMode::Move,
            SceneLiveGizmoMode::Rotate => PreviewInteractionMode::Rotate,
        };
        let mut targets = Self::extract_scene_live_targets(script);
        if !self.scene_live_tag_filters.is_empty() {
            targets.retain(|target| self.scene_live_tag_filters.contains(target.tag.as_str()));
        }
        let interaction_nodes = self.scene_live_interaction_nodes(
            &targets,
            graph_size.0.max(1) as f32,
            graph_size.1.max(1) as f32,
        );
        if protocol.allows_interaction() {
            motionloom_external_preview_log(format!(
                "send SetInteractionTargets count={}",
                interaction_nodes.len()
            ));
            host.send(PreviewCommand::SetInteractionTargets {
                mode,
                graph_width: graph_size.0.max(1) as f32,
                graph_height: graph_size.1.max(1) as f32,
                targets: interaction_nodes,
            });
        }
    }

    fn drain_scene_external_preview_events(
        &mut self,
        host: &SceneExternalPreviewHost,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        for event in host.drain_events() {
            match event {
                PreviewEvent::PickResult {
                    node: Some(node),
                    x,
                    y,
                } => {
                    if Self::find_scene_tag_range_by_id(&self.script_text, &node).is_some() {
                        self.scene_live_knob_node_id = node.clone();
                    }
                    self.status_line = format!(
                        "External preview picked {node} at {}, {}.",
                        Self::format_live_number(x),
                        Self::format_live_number(y)
                    );
                }
                PreviewEvent::PickResult { node: None, .. } => {
                    self.status_line =
                        "External preview pick missed any editable node.".to_string();
                }
                PreviewEvent::TransformDrag {
                    node,
                    property,
                    value,
                } => {
                    if Self::find_scene_tag_range_by_id(&self.script_text, &node).is_none() {
                        continue;
                    }
                    let value = Self::clamp_scene_live_attr_value(&property, value);
                    self.scene_live_knob_node_id = node.clone();
                    self.scene_live_knob_attr = property.clone();
                    self.set_scene_live_preview_override(node.clone(), property.clone(), value);
                    // The native host renders the drag immediately; avoid
                    // invalidating the slower fallback preview on every mouse move.
                    cx.notify();
                    self.status_line = format!(
                        "External preview drag: {node}.{property} = {}.",
                        Self::format_live_number(value)
                    );
                }
                PreviewEvent::TransformDragEnd { node } => {
                    self.commit_scene_external_drag_keyframes(&node, window, cx);
                }
                PreviewEvent::HostFocus { focused } => {
                    self.scene_external_preview_host_focused = focused;
                }
                _ => {}
            }
        }
    }

    fn commit_scene_external_drag_keyframes(
        &mut self,
        node: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let frame = self.preview_frame;
        let mut keyed = 0usize;
        for property in ["x", "y", "rotation"] {
            let Some(value) = self
                .scene_live_preview_overrides
                .get(&(node.to_string(), property.to_string()))
                .copied()
            else {
                continue;
            };
            self.upsert_scene_live_keyframe_value_for_code_block(
                node,
                property,
                frame,
                Self::format_live_number(value),
                window,
                cx,
            );
            keyed += 1;
        }
        if keyed > 0 {
            self.status_line =
                format!("External preview drag wrote {keyed} keyframe channel(s) for {node}.");
        }
    }

    fn scene_live_preview_label(&self) -> String {
        let quality = self.scene_live_effective_preview_quality();
        let mut label = format!("Preview: {} <= {}px", quality.label(), quality.max_dim());
        if self.scene_live_prerendering
            && let Some((done, total)) = self.scene_live_prerender_progress
        {
            label.push_str(&format!(" · RAM {done}/{total}"));
        }
        label
    }

    fn scene_live_preview_image(
        &mut self,
        frame: u32,
        cx: &mut Context<Self>,
    ) -> Result<Option<LoadedPreview>, String> {
        let external_active = self.scene_external_preview_active();
        let raw = if external_active {
            self.script_text.clone()
        } else {
            self.scene_live_preview_script_text()
        };
        if raw.trim().is_empty() {
            if self.has_scene_live_preview_state() {
                self.invalidate_scene_live_preview_cache();
            }
            return Ok(None);
        }
        if !is_graph_script(&raw) {
            if self.has_scene_live_preview_state() {
                self.invalidate_scene_live_preview_cache();
            }
            return Err(
                "Scene live preview requires a <Graph ...> MotionLoom DSL block.".to_string(),
            );
        }
        if let Some(until) = self.scene_live_render_defer_until
            && Instant::now() < until
        {
            if let Some(preview) = self.scene_live_preview_cache_image.as_ref() {
                return Ok(Some(preview.clone()));
            }
            return Ok(None);
        }
        match parse_motionloom_document(&raw) {
            Ok(MotionLoomDocument::World(_)) => {
                return self.world_live_preview_image(&raw, frame, cx);
            }
            Ok(MotionLoomDocument::Process(_)) => {
                return Err(
                    "Scene live preview cannot render process-only Layer FX graphs without a source clip."
                        .to_string(),
                );
            }
            Ok(MotionLoomDocument::Scene(_)) => {}
            Ok(MotionLoomDocument::Mixed(shell)) if shell.has_scene || shell.has_process => {
                // Mixed Scene/Process graphs use the scene composition renderer so scene: sources
                // can feed post-process passes.
            }
            Ok(MotionLoomDocument::Mixed(shell)) if shell.has_world => {}
            Ok(MotionLoomDocument::Mixed(_)) => {
                return Err(
                    "Scene live preview needs a renderable Scene or World node.".to_string()
                );
            }
            Err(err) => {
                return Err(format!(
                    "Scene live parse error at line {}: {}",
                    err.line, err.message
                ));
            }
        }

        let script_hash = Self::script_hash(&raw);
        let graph = if self.scene_live_parsed_script_hash == Some(script_hash) {
            self.scene_live_parsed_graph
                .clone()
                .ok_or_else(|| "Scene live preview parse cache is missing.".to_string())?
        } else {
            let graph = parse_graph_script(&raw).map_err(|err| {
                format!(
                    "Scene live parse error at line {}: {}",
                    err.line, err.message
                )
            })?;
            self.scene_live_parsed_script_hash = Some(script_hash);
            self.scene_live_parsed_graph = Some(graph.clone());
            graph
        };
        if !graph.has_scene_nodes() {
            return Err("Scene live preview needs at least one scene node.".to_string());
        }

        let final_size = graph.render_size.unwrap_or(graph.size);
        let uses_scene_process_composition = !graph.textures.is_empty()
            || !graph.passes.is_empty()
            || !graph.outputs.is_empty()
            || !graph.layers.is_empty()
            || !graph.world_sources.is_empty();
        let effective_quality = self.scene_live_effective_preview_quality();
        let preview_size = if uses_scene_process_composition {
            final_size
        } else {
            Self::scene_live_preview_output_size(final_size, Some(effective_quality.max_dim()))
        };
        let key = (
            script_hash,
            frame,
            effective_quality,
            preview_size.0,
            preview_size.1,
        );
        let render_size = (preview_size != final_size).then_some(preview_size);
        let asset_roots = self.scene_live_asset_roots(cx);
        self.sync_external_scene_preview(&raw, script_hash, frame, effective_quality, &asset_roots);

        if external_active {
            self.cancel_scene_live_prerender();
            self.cancel_scene_live_async_render();
            self.scene_live_preview_status = "Preview: external WGPU host".to_string();
            return Ok(self.scene_live_preview_cache_image.clone());
        }

        if self.scene_live_preview_cache_key == Some(key)
            && let Some(preview) = self.scene_live_preview_cache_image.as_ref()
        {
            return Ok(Some(preview.clone()));
        }
        if let Some(preview) = self.scene_live_preview_frame_cache.get(&key) {
            let preview = preview.clone();
            self.scene_live_preview_cache_key = Some(key);
            self.scene_live_preview_cache_image = Some(preview.clone());
            return Ok(Some(preview));
        }
        if self.scene_live_async_render_key.is_some() {
            return Ok(self.scene_live_preview_cache_image.clone());
        }
        if self.preview_playing
            && self.scene_live_prerendering
            && let Some(preview) = self.scene_live_preview_cache_image.as_ref()
        {
            return Ok(Some(preview.clone()));
        }

        if self.scene_live_async_render_key != Some(key) {
            self.scene_live_async_render_key = Some(key);
            self.scene_live_preview_status = "Preview: rendering...".to_string();
            self.scene_live_async_render_token = self.scene_live_async_render_token.wrapping_add(1);
            let token = self.scene_live_async_render_token;
            let (tx, rx) =
                mpsc::channel::<Result<(SceneLivePreviewFrame, Option<String>), String>>();
            let request = SceneLivePreviewRequest {
                script: raw,
                script_hash,
                render_size,
                frame,
                asset_roots,
                response_tx: tx,
            };
            if let Err(err) = self.scene_live_preview_tx.send(request) {
                self.scene_live_preview_tx = Self::spawn_scene_live_preview_worker();
                if self.scene_live_preview_tx.send(err.0).is_err() {
                    self.scene_live_async_render_key = None;
                    return Err("Scene live preview worker is unavailable.".to_string());
                }
            }
            cx.spawn(async move |view, cx| {
                let mut poll_ms = SCENE_LIVE_PRERENDER_POLL_MS;
                loop {
                    let mut done = false;
                    let _ = view.update(cx, |this, cx| {
                        poll_ms = Self::scene_live_poll_ms(this.preview_playing);
                        if this.scene_live_async_render_token != token {
                            done = true;
                            return;
                        }
                        match rx.try_recv() {
                            Ok(Ok((frame, warning))) => {
                                this.scene_live_async_render_key = None;
                                let preview_status = if warning.is_some() {
                                    "Preview: CPU fallback".to_string()
                                } else {
                                    frame.preview_status_label().to_string()
                                };
                                if let Ok(preview) = frame.into_loaded_preview() {
                                    this.scene_live_preview_cache_key = Some(key);
                                    this.scene_live_preview_cache_image = Some(preview.clone());
                                    this.insert_scene_live_preview_frame_cache(key, preview);
                                }
                                this.scene_live_preview_status = preview_status;
                                if let Some(warning) = warning {
                                    this.push_scene_render_log(warning);
                                }
                                done = true;
                                cx.notify();
                            }
                            Ok(Err(err)) => {
                                this.scene_live_async_render_key = None;
                                this.scene_live_preview_status = "Preview: error".to_string();
                                this.status_line = err.clone();
                                this.push_scene_render_log(format!("LIVE PREVIEW ERROR: {err}"));
                                done = true;
                                cx.notify();
                            }
                            Err(mpsc::TryRecvError::Empty) => {}
                            Err(mpsc::TryRecvError::Disconnected) => {
                                this.scene_live_async_render_key = None;
                                this.scene_live_preview_status =
                                    "Preview: worker disconnected".to_string();
                                this.push_scene_render_log(
                                    "LIVE PREVIEW ERROR: render worker disconnected.".to_string(),
                                );
                                done = true;
                                cx.notify();
                            }
                        }
                    });
                    if done {
                        break;
                    }
                    Timer::after(Duration::from_millis(poll_ms)).await;
                }
            })
            .detach();
        }
        if let Some(preview) = self.scene_live_preview_cache_image.as_ref() {
            return Ok(Some(preview.clone()));
        }
        Ok(None)
    }

    fn world_live_preview_image(
        &mut self,
        raw: &str,
        frame: u32,
        cx: &mut Context<Self>,
    ) -> Result<Option<LoadedPreview>, String> {
        let graph = parse_world_graph_script(raw).map_err(|err| {
            format!(
                "World live parse error at line {}: {}",
                err.line, err.message
            )
        })?;
        let final_size = graph.render_size.unwrap_or(graph.size);
        let effective_quality = self.scene_live_effective_preview_quality();
        let preview_size =
            Self::scene_live_preview_output_size(final_size, Some(effective_quality.max_dim()));
        let script_hash = Self::script_hash(raw);
        let key = (
            script_hash,
            frame,
            effective_quality,
            preview_size.0,
            preview_size.1,
        );
        if self.scene_live_preview_cache_key == Some(key)
            && let Some(preview) = self.scene_live_preview_cache_image.as_ref()
        {
            return Ok(Some(preview.clone()));
        }
        if let Some(preview) = self.scene_live_preview_frame_cache.get(&key) {
            let preview = preview.clone();
            self.scene_live_preview_cache_key = Some(key);
            self.scene_live_preview_cache_image = Some(preview.clone());
            return Ok(Some(preview));
        }
        if self.preview_playing
            && self.scene_live_prerendering
            && let Some(preview) = self.scene_live_preview_cache_image.as_ref()
        {
            return Ok(Some(preview.clone()));
        }

        let mut preview_graph = graph.clone();
        if preview_size != final_size {
            preview_graph.render_size = Some(preview_size);
        }
        if self.scene_live_async_render_key != Some(key) {
            self.scene_live_async_render_key = Some(key);
            self.scene_live_async_render_token = self.scene_live_async_render_token.wrapping_add(1);
            let token = self.scene_live_async_render_token;
            let asset_root = Self::world_asset_root();
            let (tx, rx) = mpsc::channel::<Result<(u32, u32, Vec<u8>), String>>();
            let request = WorldLivePreviewRequest {
                graph: preview_graph,
                frame,
                asset_root,
                response_tx: tx,
            };
            if let Err(err) = self.world_live_preview_tx.send(request) {
                self.world_live_preview_tx = Self::spawn_world_live_preview_worker();
                let _ = self.world_live_preview_tx.send(err.0);
            }
            cx.spawn(async move |view, cx| {
                let mut poll_ms = SCENE_LIVE_PRERENDER_POLL_MS;
                loop {
                    let mut done = false;
                    let _ = view.update(cx, |this, cx| {
                        poll_ms = Self::scene_live_poll_ms(this.preview_playing);
                        if this.scene_live_async_render_token != token {
                            done = true;
                            return;
                        }
                        match rx.try_recv() {
                            Ok(Ok((w, h, bgra))) => {
                                this.scene_live_async_render_key = None;
                                if let Ok(preview) = Self::loaded_preview_from_bgra(w, h, bgra) {
                                    this.scene_live_preview_cache_key = Some(key);
                                    this.scene_live_preview_cache_image = Some(preview.clone());
                                    this.insert_scene_live_preview_frame_cache(key, preview);
                                }
                                done = true;
                                cx.notify();
                            }
                            Ok(Err(err)) => {
                                this.scene_live_async_render_key = None;
                                this.status_line = err.clone();
                                this.push_scene_render_log(format!("LIVE PREVIEW ERROR: {err}"));
                                done = true;
                                cx.notify();
                            }
                            Err(mpsc::TryRecvError::Empty) => {}
                            Err(mpsc::TryRecvError::Disconnected) => {
                                this.scene_live_async_render_key = None;
                                this.push_scene_render_log(
                                    "LIVE PREVIEW ERROR: render worker disconnected.".to_string(),
                                );
                                done = true;
                                cx.notify();
                            }
                        }
                    });
                    if done {
                        break;
                    }
                    Timer::after(Duration::from_millis(poll_ms)).await;
                }
            })
            .detach();
        }
        if let Some(preview) = self.scene_live_preview_cache_image.as_ref() {
            return Ok(Some(preview.clone()));
        }
        Ok(None)
    }

    fn format_live_number(value: f32) -> String {
        let mut text = format!("{value:.2}");
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        if text == "-0" { "0".to_string() } else { text }
    }

    fn format_live_number_like_existing(value: f32, existing: &str) -> String {
        let value = Self::format_live_number(value);
        let existing = existing.trim();
        if existing.ends_with('s') {
            format!("{value}s")
        } else {
            value
        }
    }

    fn parse_live_number(raw: &str) -> Option<f32> {
        let mut text = raw.trim();
        if let Some(inner) = text.strip_prefix('{').and_then(|v| v.strip_suffix('}')) {
            text = inner.trim();
        }
        if let Some(seconds) = text.strip_suffix('s') {
            text = seconds.trim();
        }
        text.parse::<f32>().ok()
    }

    fn is_attr_ident_byte(byte: u8) -> bool {
        byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || byte == b':'
    }

    fn find_scene_tag_range_by_id(script: &str, node_id: &str) -> Option<(usize, usize)> {
        let id_patterns = [
            format!("id=\"{node_id}\""),
            format!("id='{node_id}'"),
            format!("id={node_id}"),
        ];
        if let Some(id_pos) = id_patterns
            .iter()
            .filter_map(|pattern| script.find(pattern))
            .min()
        {
            let tag_start = script[..id_pos].rfind('<')?;
            let tag_end = id_pos + script[id_pos..].find('>')? + 1;
            return Some((tag_start, tag_end));
        }
        Self::find_synthetic_scene_live_tag_range(script, node_id)
    }

    fn scene_live_filter_tags() -> &'static [&'static str] {
        &["Group", "Part", "Actor", "Action", "Pose", "ApplyAction"]
    }

    fn synthetic_scene_live_target_id(tag_name: &str, ordinal: usize) -> String {
        format!("{tag_name}#{:02}", ordinal + 1)
    }

    fn parse_synthetic_scene_live_target_id(node_id: &str) -> Option<(&str, usize)> {
        let (tag_name, ordinal) = node_id.split_once('#')?;
        if !Self::scene_live_filter_tags().contains(&tag_name) {
            return None;
        }
        let ordinal = ordinal.parse::<usize>().ok()?.checked_sub(1)?;
        Some((tag_name, ordinal))
    }

    fn find_synthetic_scene_live_tag_range(script: &str, node_id: &str) -> Option<(usize, usize)> {
        let (wanted_tag, wanted_ordinal) = Self::parse_synthetic_scene_live_target_id(node_id)?;
        let mut seen = 0usize;
        let mut search_from = 0usize;
        while let Some(rel_start) = script[search_from..].find('<') {
            let tag_start = search_from + rel_start;
            let Some(rel_end) = script[tag_start..].find('>') else {
                break;
            };
            let tag_end = tag_start + rel_end + 1;
            let tag = &script[tag_start..tag_end];
            if Self::tag_name(tag).as_deref() == Some(wanted_tag) {
                if seen == wanted_ordinal {
                    return Some((tag_start, tag_end));
                }
                seen += 1;
            }
            search_from = tag_end;
        }
        None
    }

    fn find_attr_value_range_in_tag(tag: &str, attr: &str) -> Option<(usize, usize)> {
        let bytes = tag.as_bytes();
        let mut search_from = 0;
        while search_from < tag.len() {
            let rel = tag[search_from..].find(attr)?;
            let attr_start = search_from + rel;
            let attr_end = attr_start + attr.len();
            if attr_start > 0 && Self::is_attr_ident_byte(bytes[attr_start - 1]) {
                search_from = attr_end;
                continue;
            }
            if attr_end < bytes.len() && Self::is_attr_ident_byte(bytes[attr_end]) {
                search_from = attr_end;
                continue;
            }

            let mut i = attr_end;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() || bytes[i] != b'=' {
                search_from = attr_end;
                continue;
            }
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() {
                return None;
            }
            if bytes[i] == b'"' || bytes[i] == b'\'' {
                let quote = bytes[i];
                let value_start = i + 1;
                let value_end = tag[value_start..]
                    .bytes()
                    .position(|byte| byte == quote)
                    .map(|pos| value_start + pos)?;
                return Some((value_start, value_end));
            }
            if bytes[i] == b'{' {
                let value_start = i;
                let mut depth = 0usize;
                let mut quote: Option<u8> = None;
                while i < bytes.len() {
                    let byte = bytes[i];
                    if let Some(active_quote) = quote {
                        if byte == active_quote
                            && (i == 0 || bytes.get(i.wrapping_sub(1)) != Some(&b'\\'))
                        {
                            quote = None;
                        }
                        i += 1;
                        continue;
                    }

                    match byte {
                        b'"' | b'\'' => quote = Some(byte),
                        b'{' => depth += 1,
                        b'}' => {
                            depth = depth.saturating_sub(1);
                            i += 1;
                            if depth == 0 {
                                return Some((value_start, i));
                            }
                            continue;
                        }
                        _ => {}
                    }
                    i += 1;
                }
                return None;
            }

            let value_start = i;
            while i < bytes.len()
                && !bytes[i].is_ascii_whitespace()
                && bytes[i] != b'>'
                && bytes[i] != b'/'
            {
                i += 1;
            }
            return Some((value_start, i));
        }
        None
    }

    fn tag_name(tag: &str) -> Option<String> {
        let tag = tag.trim_start();
        let body = tag.strip_prefix('<')?.trim_start();
        if body.starts_with('/') || body.starts_with('!') || body.starts_with('?') {
            return None;
        }
        let end = body
            .find(|ch: char| ch.is_ascii_whitespace() || ch == '/' || ch == '>')
            .unwrap_or(body.len());
        if end == 0 {
            None
        } else {
            Some(body[..end].to_string())
        }
    }

    fn is_scene_live_target_tag(tag: &str) -> bool {
        matches!(
            tag,
            "Character"
                | "Group"
                | "Part"
                | "Actor"
                | "Action"
                | "Pose"
                | "ApplyAction"
                | "Camera"
                | "Mask"
                | "Circle"
                | "Rect"
                | "Path"
                | "FaceJaw"
                | "Line"
                | "Polyline"
                | "Text"
                | "Image"
                | "Svg"
        )
    }

    fn push_scene_live_attr(attrs: &mut Vec<String>, attr: &str) {
        if !attrs.iter().any(|existing| existing == attr) {
            attrs.push(attr.to_string());
        }
    }

    fn scene_live_attrs_for_tag(tag_name: &str, tag: &str) -> Vec<String> {
        let mut attrs = Vec::new();
        match tag_name {
            "Character" | "Group" | "Camera" | "Mask" => {
                for attr in [
                    "x",
                    "y",
                    "rotation",
                    "scale",
                    "scaleX",
                    "scaleY",
                    "skewX",
                    "skewY",
                    "transformOriginX",
                    "transformOriginY",
                    "opacity",
                ] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "Part" => {
                for attr in [
                    "x", "y", "rotation", "scale", "anchorX", "anchorY", "opacity",
                ] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "Actor" => {
                for attr in ["x", "y", "z", "yaw", "pitch", "roll", "scale", "opacity"] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "Action" => {
                for attr in ["duration"] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "Pose" => {
                for attr in ["t"] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "ApplyAction" => {
                for attr in ["at", "weight"] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "Circle" => {
                for attr in [
                    "x",
                    "y",
                    "radius",
                    "rotation",
                    "scale",
                    "scaleX",
                    "scaleY",
                    "skewX",
                    "skewY",
                    "transformOriginX",
                    "transformOriginY",
                    "opacity",
                    "strokeWidth",
                ] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "Rect" | "Text" => {
                for attr in [
                    "x",
                    "y",
                    "width",
                    "height",
                    "rotation",
                    "scale",
                    "scaleX",
                    "scaleY",
                    "skewX",
                    "skewY",
                    "transformOriginX",
                    "transformOriginY",
                    "opacity",
                ] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "Image" | "Svg" => {
                for attr in ["x", "y", "width", "height", "rotation", "scale", "opacity"] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "Line" => {
                for attr in [
                    "x",
                    "y",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "rotation",
                    "scale",
                    "scaleX",
                    "scaleY",
                    "skewX",
                    "skewY",
                    "transformOriginX",
                    "transformOriginY",
                    "strokeWidth",
                    "opacity",
                ] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "Path" | "Polyline" => {
                for attr in [
                    "x",
                    "y",
                    "rotation",
                    "scale",
                    "scaleX",
                    "scaleY",
                    "skewX",
                    "skewY",
                    "transformOriginX",
                    "transformOriginY",
                    "strokeWidth",
                    "opacity",
                    "feather",
                ] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            "FaceJaw" => {
                for attr in [
                    "x",
                    "y",
                    "width",
                    "height",
                    "cheekWidth",
                    "chinWidth",
                    "chinSharpness",
                    "jawEase",
                    "scale",
                    "strokeWidth",
                    "opacity",
                ] {
                    Self::push_scene_live_attr(&mut attrs, attr);
                }
            }
            _ => {}
        }

        for attr in [
            "x",
            "y",
            "x1",
            "y1",
            "x2",
            "y2",
            "z",
            "yaw",
            "pitch",
            "roll",
            "duration",
            "t",
            "at",
            "weight",
            "radius",
            "width",
            "height",
            "cheekWidth",
            "chinWidth",
            "chinSharpness",
            "jawEase",
            "rotation",
            "scale",
            "scaleX",
            "scaleY",
            "skewX",
            "skewY",
            "transformOriginX",
            "transformOriginY",
            "opacity",
            "strokeWidth",
            "feather",
        ] {
            if Self::find_attr_value_range_in_tag(tag, attr).is_some() {
                Self::push_scene_live_attr(&mut attrs, attr);
            }
        }
        attrs
    }

    fn extract_scene_live_targets(script: &str) -> Vec<SceneLiveTarget> {
        let mut out = Vec::new();
        let mut tag_ordinals = HashMap::<String, usize>::new();
        let mut search_from = 0;
        while let Some(rel_start) = script[search_from..].find('<') {
            let tag_start = search_from + rel_start;
            let Some(rel_end) = script[tag_start..].find('>') else {
                break;
            };
            let tag_end = tag_start + rel_end + 1;
            let tag = &script[tag_start..tag_end];
            if let Some(tag_name) = Self::tag_name(tag)
                && Self::is_scene_live_target_tag(&tag_name)
            {
                let ordinal = tag_ordinals.entry(tag_name.clone()).or_insert(0);
                let current_ordinal = *ordinal;
                *ordinal += 1;
                let id = Self::find_attr_value_range_in_tag(tag, "id")
                    .map(|(id_start, id_end)| tag[id_start..id_end].trim().to_string())
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| {
                        Self::synthetic_scene_live_target_id(&tag_name, current_ordinal)
                    });
                if !id.is_empty() && !out.iter().any(|target: &SceneLiveTarget| target.id == id) {
                    let attrs = Self::scene_live_attrs_for_tag(&tag_name, tag);
                    if !attrs.is_empty() {
                        out.push(SceneLiveTarget {
                            id,
                            tag: tag_name,
                            attrs,
                        });
                    }
                }
            }
            search_from = tag_end;
        }
        out
    }

    fn default_scene_live_attr(attrs: &[String]) -> String {
        for preferred in [
            "x",
            "y",
            "radius",
            "rotation",
            "scale",
            "opacity",
            "strokeWidth",
        ] {
            if attrs.iter().any(|attr| attr == preferred) {
                return preferred.to_string();
            }
        }
        attrs
            .first()
            .cloned()
            .unwrap_or_else(|| DEFAULT_SCENE_LIVE_ATTR.to_string())
    }

    fn overall_scene_live_target<'a>(
        targets: &'a [SceneLiveTarget],
    ) -> Option<&'a SceneLiveTarget> {
        targets
            .iter()
            .find(|target| target.tag == "Character" || target.tag == "Group")
            .or_else(|| {
                targets
                    .iter()
                    .find(|target| target.tag == "Camera" || target.tag == "Mask")
            })
            .or_else(|| targets.first())
    }

    fn ensure_scene_live_selection(&mut self, targets: &[SceneLiveTarget]) {
        let Some(mut target) = targets
            .iter()
            .find(|target| target.id == self.scene_live_knob_node_id)
        else {
            if let Some(next) = Self::overall_scene_live_target(targets) {
                self.scene_live_knob_node_id = next.id.clone();
                self.scene_live_knob_attr = Self::default_scene_live_attr(&next.attrs);
            }
            return;
        };

        if !target
            .attrs
            .iter()
            .any(|attr| attr == &self.scene_live_knob_attr)
        {
            self.scene_live_knob_attr = Self::default_scene_live_attr(&target.attrs);
            target = targets
                .iter()
                .find(|target| target.id == self.scene_live_knob_node_id)
                .unwrap_or(target);
        }

        if target.attrs.is_empty() {
            self.scene_live_knob_attr = DEFAULT_SCENE_LIVE_ATTR.to_string();
        }
    }

    fn select_scene_live_target(&mut self, id: String, attrs: Vec<String>) {
        self.scene_live_knob_node_id = id;
        if !attrs.iter().any(|attr| attr == &self.scene_live_knob_attr) {
            self.scene_live_knob_attr = Self::default_scene_live_attr(&attrs);
        }
        self.scene_live_attr_menu_open = false;
        self.status_line = format!(
            "Live target selected: {}.{}.",
            self.scene_live_knob_node_id, self.scene_live_knob_attr
        );
    }

    fn select_scene_live_attr(&mut self, attr: String) {
        self.scene_live_knob_attr = attr;
        self.scene_live_attr_menu_open = false;
        self.status_line = format!(
            "Live attribute selected: {}.{}.",
            self.scene_live_knob_node_id, self.scene_live_knob_attr
        );
    }

    fn scene_live_target_select_items(
        targets: &[SceneLiveTarget],
    ) -> SearchableVec<SceneLiveTargetOption> {
        SearchableVec::new(
            targets
                .iter()
                .map(|target| SceneLiveTargetOption {
                    id: target.id.clone(),
                    label: format!("{} · {}", target.tag, target.id),
                })
                .collect::<Vec<_>>(),
        )
    }

    fn sync_scene_live_dropdowns(
        &mut self,
        targets: &[SceneLiveTarget],
        _attrs: &[String],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let target_items = Self::scene_live_target_select_items(targets);
        if self.scene_live_target_select.is_none() {
            let state = cx.new(|cx| {
                SelectState::new(target_items.clone(), None, window, cx).searchable(false)
            });
            let sub = cx.subscribe(
                &state,
                |this, _, ev: &SelectEvent<SearchableVec<SceneLiveTargetOption>>, cx| {
                    let SelectEvent::Confirm(value) = ev;
                    let Some(value) = value else {
                        return;
                    };
                    let mut targets = Self::extract_scene_live_targets(&this.script_text);
                    if !this.scene_live_tag_filters.is_empty() {
                        targets.retain(|target| {
                            this.scene_live_tag_filters.contains(target.tag.as_str())
                        });
                    }
                    if let Some(target) = targets.iter().find(|target| target.id == *value) {
                        this.select_scene_live_target(value.clone(), target.attrs.clone());
                    }
                    cx.notify();
                },
            );
            self.scene_live_target_select = Some(state);
            self.scene_live_target_select_sub = Some(sub);
        } else if let Some(state) = self.scene_live_target_select.as_ref() {
            state.update(cx, |this, cx| {
                this.set_items(target_items.clone(), window, cx);
            });
        }
        if let Some(state) = self.scene_live_target_select.as_ref() {
            let selected = self.scene_live_knob_node_id.clone();
            state.update(cx, |this, cx| {
                this.set_selected_value(&selected, window, cx);
            });
        }
    }

    fn find_scene_tag_attr_number(script: &str, node_id: &str, attr: &str) -> Option<f32> {
        let (tag_start, tag_end) = Self::find_scene_tag_range_by_id(script, node_id)?;
        let tag = &script[tag_start..tag_end];
        let (value_start, value_end) = Self::find_attr_value_range_in_tag(tag, attr)?;
        Self::parse_live_number(&tag[value_start..value_end])
    }

    fn patch_scene_tag_attr_number(
        script: &str,
        node_id: &str,
        attr: &str,
        value: f32,
    ) -> Result<String, String> {
        let (tag_start, tag_end) = Self::find_scene_tag_range_by_id(script, node_id)
            .ok_or_else(|| format!("Live knob target id=\"{node_id}\" was not found."))?;
        let tag = &script[tag_start..tag_end];

        if let Some((value_start, value_end)) = Self::find_attr_value_range_in_tag(tag, attr) {
            let abs_start = tag_start + value_start;
            let abs_end = tag_start + value_end;
            let value_text =
                Self::format_live_number_like_existing(value, &tag[value_start..value_end]);
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
        let value_text = Self::format_live_number(value);
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

    fn scene_live_preview_script_text(&self) -> String {
        if self.scene_live_preview_overrides.is_empty() {
            return self.script_text.clone();
        }

        let mut script = self.script_text.clone();
        let mut overrides = self
            .scene_live_preview_overrides
            .iter()
            .map(|((node, attr), value)| (node.clone(), attr.clone(), *value))
            .collect::<Vec<_>>();
        overrides.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (node, attr, value) in overrides {
            if let Ok(next) = Self::patch_scene_tag_attr_number(&script, &node, &attr, value) {
                script = next;
            }
        }
        script
    }

    fn scene_live_knob_current_value(&self) -> Option<f32> {
        if let Some(value) = self.scene_live_preview_overrides.get(&(
            self.scene_live_knob_node_id.clone(),
            self.scene_live_knob_attr.clone(),
        )) {
            return Some(*value);
        }
        self.scene_live_value_for_attr(&self.scene_live_knob_node_id, &self.scene_live_knob_attr)
    }

    fn scene_live_value_for_attr(&self, node: &str, attr: &str) -> Option<f32> {
        if let Some(value) = self
            .scene_live_preview_overrides
            .get(&(node.to_string(), attr.to_string()))
        {
            return Some(*value);
        }
        Self::find_scene_tag_attr_number(&self.script_text, node, attr)
    }

    fn scene_live_gizmo_bounds(&self, preview_w: f32, preview_h: f32) -> SceneLiveGizmoBounds {
        let raw = self.scene_live_preview_script_text();
        let graph_size = parse_graph_script(&raw)
            .ok()
            .map(|graph| graph.render_size.unwrap_or(graph.size))
            .unwrap_or((1280, 720));
        let graph_w = graph_size.0.max(1) as f32;
        let graph_h = graph_size.1.max(1) as f32;
        let scale = (preview_w / graph_w).min(preview_h / graph_h).max(0.001);
        let image_w = graph_w * scale;
        let image_h = graph_h * scale;
        let image_left = ((preview_w - image_w) * 0.5).max(0.0);
        let image_top = ((preview_h - image_h) * 0.5).max(0.0);

        let node = self.scene_live_knob_node_id.as_str();
        let x = self
            .scene_live_value_for_attr(node, "x")
            .unwrap_or(graph_w * 0.5 - 80.0);
        let y = self
            .scene_live_value_for_attr(node, "y")
            .unwrap_or(graph_h * 0.5 - 50.0);
        let width = self
            .scene_live_value_for_attr(node, "width")
            .or_else(|| {
                self.scene_live_value_for_attr(node, "radius")
                    .map(|r| r * 2.0)
            })
            .unwrap_or(160.0)
            .abs()
            .max(24.0);
        let height = self
            .scene_live_value_for_attr(node, "height")
            .or_else(|| {
                self.scene_live_value_for_attr(node, "radius")
                    .map(|r| r * 2.0)
            })
            .unwrap_or(100.0)
            .abs()
            .max(24.0);

        SceneLiveGizmoBounds {
            left: image_left + x * scale,
            top: image_top + y * scale,
            width: width * scale,
            height: height * scale,
        }
    }

    fn scene_live_interaction_nodes(
        &self,
        targets: &[SceneLiveTarget],
        graph_w: f32,
        graph_h: f32,
    ) -> Vec<PreviewInteractionNode> {
        targets
            .iter()
            .filter_map(|target| {
                let raw_x = self
                    .scene_live_value_for_attr(&target.id, "x")
                    .unwrap_or(graph_w * 0.5 - 80.0);
                let raw_y = self
                    .scene_live_value_for_attr(&target.id, "y")
                    .unwrap_or(graph_h * 0.5 - 50.0);
                let radius = self.scene_live_value_for_attr(&target.id, "radius");
                let width = self
                    .scene_live_value_for_attr(&target.id, "width")
                    .or_else(|| radius.map(|r| r * 2.0))
                    .or_else(|| {
                        self.scene_live_value_for_attr(&target.id, "rx")
                            .map(|r| r * 2.0)
                    })
                    .unwrap_or(160.0)
                    .abs()
                    .max(24.0);
                let height = self
                    .scene_live_value_for_attr(&target.id, "height")
                    .or_else(|| radius.map(|r| r * 2.0))
                    .or_else(|| {
                        self.scene_live_value_for_attr(&target.id, "ry")
                            .map(|r| r * 2.0)
                    })
                    .unwrap_or(100.0)
                    .abs()
                    .max(24.0);
                Some(PreviewInteractionNode {
                    node: target.id.clone(),
                    tag: target.tag.clone(),
                    x: raw_x,
                    y: raw_y,
                    width,
                    height,
                    rotation: self
                        .scene_live_value_for_attr(&target.id, "rotation")
                        .unwrap_or_else(|| Self::scene_live_attr_default_value("rotation")),
                })
            })
            .collect()
    }

    fn scene_live_knob_input_value(&self, cx: &Context<Self>) -> Option<f32> {
        self.scene_live_knob_input
            .as_ref()
            .and_then(|input| Self::parse_live_number(&input.read(cx).value()))
    }

    fn scene_live_animation_target_property_supported(attr: &str) -> bool {
        matches!(
            attr,
            "x" | "y"
                | "rotation"
                | "scale"
                | "scaleX"
                | "scaleY"
                | "skewX"
                | "skewY"
                | "transformOriginX"
                | "transformOriginY"
                | "opacity"
        )
    }

    fn scene_live_keyframe_summary(&self) -> String {
        match extract_editable_animation_timeline(&self.script_text) {
            Ok(timeline) if timeline.targets.is_empty() => {
                "Keyframes: none in code block DSL.".to_string()
            }
            Ok(timeline) => {
                let key_count: usize = timeline
                    .targets
                    .iter()
                    .map(|target| target.keys.len())
                    .sum();
                let channels = timeline
                    .targets
                    .iter()
                    .take(4)
                    .map(|target| {
                        format!("{}.{}:{}", target.node, target.property, target.keys.len())
                    })
                    .collect::<Vec<_>>()
                    .join(" · ");
                let suffix = if timeline.targets.len() > 4 {
                    format!(" · +{} more", timeline.targets.len() - 4)
                } else {
                    String::new()
                };
                format!(
                    "Keyframes: {} channels · {} keys · {}{}",
                    timeline.targets.len(),
                    key_count,
                    channels,
                    suffix
                )
            }
            Err(err) => format!("Keyframes: unavailable until DSL parses ({err})."),
        }
    }

    fn scene_live_current_animation_keys(&self) -> Vec<EditableAnimationKey> {
        let node = &self.scene_live_knob_node_id;
        let property = &self.scene_live_knob_attr;
        let Ok(timeline) = extract_editable_animation_timeline(&self.script_text) else {
            return Vec::new();
        };
        timeline
            .targets
            .into_iter()
            .find(|target| &target.node == node && &target.property == property)
            .map(|target| target.keys)
            .unwrap_or_default()
    }

    fn scene_live_key_timing_label(key: &EditableAnimationKey) -> String {
        key.time
            .clone()
            .unwrap_or_else(|| format!("f{}", key.frame))
    }

    fn upsert_scene_live_keyframe_for_code_block(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let node = self.scene_live_knob_node_id.clone();
        let property = self.scene_live_knob_attr.clone();
        if !Self::scene_live_animation_target_property_supported(&property) {
            self.status_line = format!(
                "AnimationTarget UI keying does not support selected property yet: {property}."
            );
            return;
        }

        let frame = self.preview_frame;
        let value = self
            .scene_live_knob_input_value(cx)
            .or_else(|| self.scene_live_knob_current_value())
            .unwrap_or_else(|| Self::scene_live_attr_default_value(&property));
        let value_text = Self::format_live_number(value);

        self.upsert_scene_live_keyframe_value_for_code_block(
            &node, &property, frame, value_text, window, cx,
        );
    }

    fn upsert_scene_live_keyframe_value_for_code_block(
        &mut self,
        node: &str,
        property: &str,
        frame: u32,
        value_text: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let timeline = extract_editable_animation_timeline(&self.script_text).ok();
        let fps = timeline
            .as_ref()
            .map(|timeline| timeline.fps)
            .unwrap_or(30.0);
        let mut keys = timeline
            .and_then(|timeline| {
                timeline
                    .targets
                    .into_iter()
                    .find(|target| target.node == node && target.property == property)
                    .map(|target| target.keys)
            })
            .unwrap_or_default();

        if let Some(existing) = keys.iter_mut().find(|key| key.frame == frame) {
            existing.value = value_text.clone();
            if existing.ease.trim().is_empty() {
                existing.ease = "linear".to_string();
            }
        } else {
            let time = (!self.scene_live_keyframe_show_as_frame)
                .then(|| format!("{}s", Self::format_live_number(frame as f32 / fps.max(1.0))));
            keys.push(EditableAnimationKey {
                frame,
                time,
                value: value_text.clone(),
                ease: "linear".to_string(),
            });
        }
        keys.sort_by_key(|key| key.frame);

        let target = EditableAnimationTarget {
            node: node.to_string(),
            property: property.to_string(),
            keys,
        };

        match upsert_editable_animation_target(&self.script_text, target) {
            Ok(script) => {
                self.set_script_text_preserve_preview(script, window, cx);
                self.status_line = format!(
                    "Code block DSL updated: keyed {node}.{property} at frame {frame} = {value_text}."
                );
            }
            Err(err) => {
                self.status_line = format!("Could not write keyframe into code block DSL: {err}");
            }
        }
    }

    fn delete_scene_live_keyframe_for_code_block(
        &mut self,
        frame: u32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let node = self.scene_live_knob_node_id.clone();
        let property = self.scene_live_knob_attr.clone();
        let Ok(timeline) = extract_editable_animation_timeline(&self.script_text) else {
            self.status_line = "Could not delete keyframe: DSL does not parse.".to_string();
            return;
        };

        let mut deleted = false;
        let targets = timeline
            .targets
            .into_iter()
            .filter_map(|mut target| {
                if target.node == node && target.property == property {
                    let before = target.keys.len();
                    target.keys.retain(|key| key.frame != frame);
                    deleted = deleted || target.keys.len() != before;
                }
                (!target.keys.is_empty()).then_some(target)
            })
            .collect::<Vec<_>>();

        if !deleted {
            self.status_line =
                format!("No keyframe to delete for {node}.{property} at frame {frame}.");
            return;
        }

        match replace_editable_animation_targets(&self.script_text, &targets) {
            Ok(script) => {
                self.set_script_text_preserve_preview(script, window, cx);
                self.status_line = format!("Deleted keyframe: {node}.{property} at frame {frame}.");
            }
            Err(err) => {
                self.status_line = format!("Could not delete keyframe: {err}");
            }
        }
    }

    fn set_scene_live_keyframe_ease_for_code_block(
        &mut self,
        frame: u32,
        ease: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let node = self.scene_live_knob_node_id.clone();
        let property = self.scene_live_knob_attr.clone();
        let Ok(timeline) = extract_editable_animation_timeline(&self.script_text) else {
            self.status_line = "Could not update key ease: DSL does not parse.".to_string();
            return;
        };

        let mut changed = false;
        let targets = timeline
            .targets
            .into_iter()
            .map(|mut target| {
                if target.node == node && target.property == property {
                    if let Some(key) = target.keys.iter_mut().find(|key| key.frame == frame) {
                        key.ease = ease.to_string();
                        changed = true;
                    }
                }
                target
            })
            .collect::<Vec<_>>();

        if !changed {
            self.status_line =
                format!("No keyframe to update for {node}.{property} at frame {frame}.");
            return;
        }

        match replace_editable_animation_targets(&self.script_text, &targets) {
            Ok(script) => {
                self.set_script_text_preserve_preview(script, window, cx);
                self.status_line =
                    format!("Updated ease: {node}.{property} at frame {frame} = {ease}.");
            }
            Err(err) => {
                self.status_line = format!("Could not update key ease: {err}");
            }
        }
    }

    fn scene_live_attr_default_value(attr: &str) -> f32 {
        match attr {
            "scale" | "scaleX" | "scaleY" | "opacity" | "duration" | "weight" => 1.0,
            _ => 0.0,
        }
    }

    fn clamp_scene_live_attr_value(attr: &str, value: f32) -> f32 {
        match attr {
            "scale" => value.clamp(0.01, 20.0),
            "scaleX" | "scaleY" => value.clamp(-20.0, 20.0),
            "opacity" | "weight" => value.clamp(0.0, 1.0),
            "chinSharpness" | "jawEase" => value.clamp(0.0, 1.0),
            "radius" | "width" | "height" | "cheekWidth" | "chinWidth" | "strokeWidth"
            | "feather" | "duration" | "t" | "at" => value.max(0.0),
            _ => value,
        }
    }

    fn scene_live_attr_base_step(attr: &str) -> f32 {
        match attr {
            "scale" | "scaleX" | "scaleY" | "opacity" => 0.05,
            "duration" | "t" | "at" => 0.05,
            "weight" => 0.05,
            "chinSharpness" | "jawEase" => 0.02,
            _ => 1.0,
        }
    }

    fn patch_scene_live_knob_value(
        &mut self,
        value: f32,
        _window: Option<&mut Window>,
        cx: &mut Context<Self>,
        defer_preview_ms: Option<u64>,
    ) {
        let next = Self::clamp_scene_live_attr_value(&self.scene_live_knob_attr, value);
        let node = self.scene_live_knob_node_id.clone();
        let attr = self.scene_live_knob_attr.clone();
        if Self::find_scene_tag_range_by_id(&self.script_text, &node).is_none() {
            self.status_line = format!("Live knob target id=\"{node}\" was not found.");
            return;
        }
        self.set_scene_live_preview_override(node.clone(), attr.clone(), next);
        if self.scene_external_preview_active() {
            self.scene_live_preview_status = "Preview: external WGPU host".to_string();
            cx.notify();
            self.status_line = format!(
                "Preview override only: {node}.{attr} = {}. Press Key Frame to write DSL.",
                Self::format_live_number(next)
            );
            return;
        }
        if let Some(defer_ms) = defer_preview_ms {
            self.cancel_scene_live_prerender();
            self.cancel_scene_live_async_render();
            self.scene_live_preview_cache_key = None;
            self.clear_scene_live_preview_frame_cache();
            self.scene_live_parsed_script_hash = None;
            self.scene_live_parsed_graph = None;
            self.defer_scene_live_preview_render(cx, defer_ms);
        } else {
            self.invalidate_scene_live_preview_render_only();
        }
        self.status_line = format!(
            "Preview override only: {node}.{attr} = {}. Press Key Frame to write DSL.",
            Self::format_live_number(next)
        );
    }

    fn nudge_scene_live_knob(
        &mut self,
        delta: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
        defer_preview: bool,
    ) {
        let current = self
            .scene_live_knob_current_value()
            .unwrap_or_else(|| Self::scene_live_attr_default_value(&self.scene_live_knob_attr));
        if defer_preview {
            self.patch_scene_live_knob_value(
                current + delta,
                Some(window),
                cx,
                Some(SCENE_LIVE_SCROLL_RENDER_DEBOUNCE_MS),
            );
        } else {
            self.patch_scene_live_knob_value(current + delta, Some(window), cx, None);
        }
    }

    fn begin_scene_live_gizmo_drag(&mut self, mouse_x: f32, mouse_y: f32) {
        let node = self.scene_live_knob_node_id.clone();
        self.scene_live_gizmo_drag = match self.scene_live_gizmo_mode {
            SceneLiveGizmoMode::Move => Some(SceneLiveGizmoDrag::Move {
                start_mouse_x: mouse_x,
                start_mouse_y: mouse_y,
                start_x: self
                    .scene_live_value_for_attr(&node, "x")
                    .unwrap_or_else(|| Self::scene_live_attr_default_value("x")),
                start_y: self
                    .scene_live_value_for_attr(&node, "y")
                    .unwrap_or_else(|| Self::scene_live_attr_default_value("y")),
            }),
            SceneLiveGizmoMode::Rotate => Some(SceneLiveGizmoDrag::Rotate {
                start_mouse_x: mouse_x,
                start_rotation: self
                    .scene_live_value_for_attr(&node, "rotation")
                    .unwrap_or_else(|| Self::scene_live_attr_default_value("rotation")),
            }),
        };
    }

    fn update_scene_live_gizmo_drag(&mut self, mouse_x: f32, mouse_y: f32) {
        let Some(drag) = self.scene_live_gizmo_drag else {
            return;
        };
        let node = self.scene_live_knob_node_id.clone();
        match drag {
            SceneLiveGizmoDrag::Move {
                start_mouse_x,
                start_mouse_y,
                start_x,
                start_y,
            } => {
                self.set_scene_live_preview_override(
                    node.clone(),
                    "x".to_string(),
                    start_x + mouse_x - start_mouse_x,
                );
                self.set_scene_live_preview_override(
                    node,
                    "y".to_string(),
                    start_y + mouse_y - start_mouse_y,
                );
            }
            SceneLiveGizmoDrag::Rotate {
                start_mouse_x,
                start_rotation,
            } => {
                self.set_scene_live_preview_override(
                    node,
                    "rotation".to_string(),
                    start_rotation + (mouse_x - start_mouse_x) * 0.5,
                );
            }
        }
        if self.scene_external_preview_active() {
            self.scene_live_preview_status = "Preview: external WGPU host".to_string();
        } else {
            self.invalidate_scene_live_preview_render_only();
        }
    }

    fn commit_scene_live_gizmo_drag(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(drag) = self.scene_live_gizmo_drag.take() else {
            return;
        };
        let node = self.scene_live_knob_node_id.clone();
        let frame = self.preview_frame;
        match drag {
            SceneLiveGizmoDrag::Move { .. } => {
                let x = self
                    .scene_live_preview_overrides
                    .get(&(node.clone(), "x".to_string()))
                    .copied();
                let y = self
                    .scene_live_preview_overrides
                    .get(&(node.clone(), "y".to_string()))
                    .copied();
                if let Some(x) = x {
                    self.upsert_scene_live_keyframe_value_for_code_block(
                        &node,
                        "x",
                        frame,
                        Self::format_live_number(x),
                        window,
                        cx,
                    );
                }
                if let Some(y) = y {
                    self.upsert_scene_live_keyframe_value_for_code_block(
                        &node,
                        "y",
                        frame,
                        Self::format_live_number(y),
                        window,
                        cx,
                    );
                }
            }
            SceneLiveGizmoDrag::Rotate { .. } => {
                if let Some(rotation) = self
                    .scene_live_preview_overrides
                    .get(&(node.clone(), "rotation".to_string()))
                    .copied()
                {
                    self.upsert_scene_live_keyframe_value_for_code_block(
                        &node,
                        "rotation",
                        frame,
                        Self::format_live_number(rotation),
                        window,
                        cx,
                    );
                }
            }
        }
    }

    fn scene_live_scroll_delta(evt: &ScrollWheelEvent, attr: &str) -> f32 {
        let delta_y = evt.delta.pixel_delta(px(10.0)).y / px(1.0);
        if delta_y.abs() <= f32::EPSILON {
            return 0.0;
        }
        let base_step = Self::scene_live_attr_base_step(attr);
        let step = if evt.modifiers.shift {
            base_step * 0.1
        } else if evt.modifiers.alt {
            base_step * 10.0
        } else {
            base_step
        };
        if delta_y < 0.0 { step } else { -step }
    }

    fn scene_live_step_labels(attr: &str) -> (f32, f32, String, String, String, String) {
        let small = Self::scene_live_attr_base_step(attr);
        let large = small * 10.0;
        (
            small,
            large,
            format!("-{}", Self::format_live_number(large)),
            format!("-{}", Self::format_live_number(small)),
            format!("+{}", Self::format_live_number(small)),
            format!("+{}", Self::format_live_number(large)),
        )
    }

    fn scene_live_scroll_hint(attr: &str) -> String {
        let base_step = Self::scene_live_attr_base_step(attr);
        format!(
            "Scroll the selected value: normal = {}, Shift = {}, Option/Alt = {}. This edits preview only; press Key Frame to write DSL.",
            Self::format_live_number(base_step),
            Self::format_live_number(base_step * 0.1),
            Self::format_live_number(base_step * 10.0)
        )
    }

    fn playback_fps(&self) -> f32 {
        self.graph_runtime
            .as_ref()
            .map(|runtime| runtime.graph().fps)
            .filter(|fps| fps.is_finite() && *fps > 0.0)
            .or_else(|| {
                self.script_playback_spec()
                    .map(|(fps, _)| fps)
                    .filter(|fps| fps.is_finite() && *fps > 0.0)
            })
            .unwrap_or(30.0)
    }

    fn script_playback_spec(&self) -> Option<(f32, u32)> {
        let graph = match parse_motionloom_document(&self.script_text).ok()? {
            MotionLoomDocument::World(graph) => {
                let fps = graph.fps;
                let total = ((graph.duration_ms as f64 / 1000.0) * fps as f64).round() as u32;
                return Some((fps, total.max(1)));
            }
            MotionLoomDocument::Process(graph) | MotionLoomDocument::Scene(graph) => graph,
            MotionLoomDocument::Mixed(shell) if shell.has_scene || shell.has_process => {
                // Mixed Scene/Process documents are rendered by the scene composition path.
                parse_graph_script(&self.script_text).ok()?
            }
            MotionLoomDocument::Mixed(shell) if shell.has_world => {
                let graph = parse_world_graph_script(&self.script_text).ok()?;
                let fps = graph.fps;
                let total = ((graph.duration_ms as f64 / 1000.0) * fps as f64).round() as u32;
                return Some((fps, total.max(1)));
            }
            MotionLoomDocument::Mixed(_) => return None,
        };
        let fps = graph.fps;
        let total = ((graph.duration_ms as f64 / 1000.0) * fps as f64).round() as u32;
        Some((fps, total.max(1)))
    }

    // Calculate total frame count from graph runtime duration or clip duration.
    fn playback_frame_count(&self) -> u32 {
        if let Some(runtime) = self.graph_runtime.as_ref() {
            let graph = runtime.graph();
            let total = ((graph.duration_ms as f64 / 1000.0) * graph.fps as f64).round() as u32;
            if total > 1 {
                return total;
            }
        }

        if let Some((_, total)) = self.script_playback_spec() {
            if total > 1 {
                return total;
            }
        }

        if let Some(clip) = self.current_clip() {
            let secs = clip.duration.as_secs_f64();
            if secs > 0.0 {
                let fps = self.playback_fps().max(1.0) as f64;
                let total = (secs * fps).round() as u32;
                if total > 1 {
                    return total;
                }
            }
        }

        // Default minimum frame count for still images
        60
    }

    fn has_scene_playback_graph(&self) -> bool {
        self.graph_runtime
            .as_ref()
            .map(|runtime| runtime.graph().has_scene_nodes())
            .or_else(|| self.script_playback_spec().map(|_| true))
            .unwrap_or(false)
    }

    // Schedule timeline playback for imported clips. Graph playback is render-locked by the
    // async frame renderer so the UI never advances to frames that are not ready yet.
    fn schedule_preview_playback(
        &mut self,
        token: u64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.on_next_frame(window, move |this, window, cx| {
            if !this.preview_playing || this.preview_play_token != token {
                return;
            }

            let now = Instant::now();
            let dt = now.saturating_duration_since(this.preview_last_tick.unwrap_or(now));
            this.preview_last_tick = Some(now);

            let fps = this.playback_fps();
            this.preview_frame_accum += dt.as_secs_f32() * fps;
            let mut step = this.preview_frame_accum.floor() as u32;
            if step > 0 {
                let frame_count = this.playback_frame_count().max(1);
                if this.has_scene_playback_graph() && this.scene_live_prerendering {
                    let script_hash = Self::script_hash(&this.script_text);
                    let mut advanced = 0;
                    while step > 0 {
                        let next_frame = this.preview_frame.saturating_add(1) % frame_count;
                        if !this.scene_live_cached_frame_ready(script_hash, next_frame) {
                            break;
                        }
                        this.preview_frame = next_frame;
                        this.clear_scene_live_preview_overrides();
                        this.scene_live_gizmo_drag = None;
                        advanced += 1;
                        step -= 1;
                    }
                    if advanced > 0 {
                        this.preview_frame_accum =
                            (this.preview_frame_accum - advanced as f32).max(0.0);
                        cx.notify();
                    } else {
                        this.preview_frame_accum = 0.0;
                    }
                } else {
                    this.preview_frame_accum -= step as f32;
                    let next_frame = this.preview_frame.saturating_add(step);
                    this.preview_frame = next_frame % frame_count;
                    this.clear_scene_live_preview_overrides();
                    this.scene_live_gizmo_drag = None;
                    cx.notify();
                }
            }

            if this.preview_playing && this.preview_play_token == token {
                this.schedule_preview_playback(token, window, cx);
            }
        });
    }

    fn start_scene_live_prerender(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let raw = self.script_text.clone();
        if Self::uses_pure_world_renderer(&raw) {
            self.start_world_live_prerender(raw, window, cx);
            return;
        }
        let Ok(graph) = parse_graph_script(&raw) else {
            return;
        };
        if !graph.has_scene_nodes() {
            return;
        }

        let final_size = graph.render_size.unwrap_or(graph.size);
        let uses_scene_process_composition = !graph.textures.is_empty()
            || !graph.passes.is_empty()
            || !graph.outputs.is_empty()
            || !graph.layers.is_empty()
            || !graph.world_sources.is_empty();
        let effective_quality = self.scene_live_effective_preview_quality();
        let preview_size = if uses_scene_process_composition {
            final_size
        } else {
            Self::scene_live_preview_output_size(final_size, Some(effective_quality.max_dim()))
        };
        let mut preview_graph = graph.clone();
        if preview_size != final_size {
            preview_graph.render_size = Some(preview_size);
        }

        let total_frames = self
            .playback_frame_count()
            .max(1)
            .min(SCENE_LIVE_PRERENDER_MAX_FRAMES);
        let script_hash = Self::script_hash(&raw);
        if let Some(cancel) = self.scene_live_prerender_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.scene_live_prerender_cancel = Some(cancel.clone());
        self.scene_live_prerender_token = self.scene_live_prerender_token.wrapping_add(1);
        let token = self.scene_live_prerender_token;
        self.scene_live_preview_cache_key = None;
        if !self.preview_playing {
            self.scene_live_preview_cache_image = None;
        }
        self.clear_scene_live_preview_frame_cache();
        self.scene_live_prerendering = true;
        self.scene_live_prerender_progress = Some((0, total_frames));

        enum SceneLivePrerenderEvent {
            Frame {
                frame: u32,
                width: u32,
                height: u32,
                bgra: Vec<u8>,
            },
            Finished(Result<u32, String>),
        }

        let (tx, rx) = mpsc::channel::<SceneLivePrerenderEvent>();
        let _ = std::thread::Builder::new()
            .name("motionloom-scene-live-prerender".to_string())
            .stack_size(SCENE_LIVE_RENDER_WORKER_STACK_SIZE)
            .spawn(move || {
                let mut gpu_renderer = std::panic::catch_unwind(|| {
                    pollster::block_on(SceneRenderer::new(SceneRenderProfile::Gpu)).ok()
                })
                .ok()
                .flatten();
                let mut cpu_renderer = std::panic::catch_unwind(|| {
                    pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).ok()
                })
                .ok()
                .flatten();
                for frame in 0..total_frames {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let rgba = if let Some(renderer) = gpu_renderer.as_mut() {
                        match pollster::block_on(renderer.render_frame(&preview_graph, frame)) {
                            Ok(rgba) => rgba,
                            Err(gpu_err) => {
                                let Some(renderer) = cpu_renderer.as_mut() else {
                                    let _ = tx.send(SceneLivePrerenderEvent::Finished(Err(
                                        gpu_err.to_string(),
                                    )));
                                    return;
                                };
                                match pollster::block_on(
                                    renderer.render_frame(&preview_graph, frame),
                                ) {
                                    Ok(rgba) => rgba,
                                    Err(cpu_err) => {
                                        let _ = tx.send(SceneLivePrerenderEvent::Finished(Err(
                                            format!("{gpu_err}; CPU fallback failed: {cpu_err}"),
                                        )));
                                        return;
                                    }
                                }
                            }
                        }
                    } else {
                        let Some(renderer) = cpu_renderer.as_mut() else {
                            let _ = tx.send(SceneLivePrerenderEvent::Finished(Err(
                                "Scene prerender error: no preview renderer initialized."
                                    .to_string(),
                            )));
                            return;
                        };
                        match pollster::block_on(renderer.render_frame(&preview_graph, frame)) {
                            Ok(rgba) => rgba,
                            Err(err) => {
                                let _ = tx
                                    .send(SceneLivePrerenderEvent::Finished(Err(err.to_string())));
                                return;
                            }
                        }
                    };
                    let (width, height, bgra) = Self::rgba_image_to_bgra(rgba);
                    if tx
                        .send(SceneLivePrerenderEvent::Frame {
                            frame,
                            width,
                            height,
                            bgra,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let _ = tx.send(SceneLivePrerenderEvent::Finished(Ok(total_frames)));
            });

        cx.spawn_in(window, async move |view, window| {
            let mut poll_ms = SCENE_LIVE_PRERENDER_POLL_MS;
            loop {
                let mut done = false;
                let mut should_notify = false;
                let _ = view.update_in(window, |this, _window, cx| {
                    poll_ms = Self::scene_live_poll_ms(this.preview_playing);
                    if this.scene_live_prerender_token != token {
                        done = true;
                        return;
                    }

                    loop {
                        match rx.try_recv() {
                            Ok(SceneLivePrerenderEvent::Frame {
                                frame,
                                width,
                                height,
                                bgra,
                            }) => {
                                let key = (script_hash, frame, effective_quality, width, height);
                                if let Ok(preview) =
                                    Self::loaded_preview_from_bgra(width, height, bgra)
                                {
                                    this.scene_live_preview_frame_cache
                                        .insert(key, preview.clone());
                                    this.scene_live_prerender_progress =
                                        Some((frame.saturating_add(1), total_frames));
                                    if this.preview_frame == frame {
                                        this.scene_live_preview_cache_key = Some(key);
                                        this.scene_live_preview_cache_image = Some(preview);
                                    }
                                    should_notify = true;
                                }
                            }
                            Ok(SceneLivePrerenderEvent::Finished(result)) => {
                                this.scene_live_prerendering = false;
                                this.scene_live_prerender_progress = None;
                                this.scene_live_prerender_cancel = None;
                                let finished_frame_count = result.as_ref().map(|frames| *frames).unwrap_or(total_frames);
                                match result {
                                    Ok(frames) => {
                                        if this.preview_playing {
                                            let frame_count = frames.max(1);
                                            this.preview_frame %= frame_count;
                                            this.preview_last_tick = Some(Instant::now());
                                            this.preview_frame_accum = 0.0;
                                            this.status_line = format!(
                                                "RAM preview cached {} frame(s); loop playback running.",
                                                frames
                                            );
                                        } else {
                                            this.status_line =
                                                format!("RAM preview cached {} frame(s).", frames);
                                        }
                                    }
                                    Err(err) => {
                                        if this.preview_playing {
                                            this.stop_preview_playback();
                                            this.preview_frame = finished_frame_count.saturating_sub(1);
                                        }
                                        this.status_line = format!("RAM preview failed: {err}");
                                    }
                                }
                                should_notify = true;
                                done = true;
                                break;
                            }
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => {
                                this.scene_live_prerendering = false;
                                this.scene_live_prerender_progress = None;
                                this.scene_live_prerender_cancel = None;
                                should_notify = true;
                                done = true;
                                break;
                            }
                        }
                    }

                    if should_notify {
                        cx.notify();
                    }
                });
                if done {
                    break;
                }
                Timer::after(Duration::from_millis(poll_ms)).await;
            }
        })
        .detach();
    }

    fn start_world_live_prerender(
        &mut self,
        raw: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Ok(graph) = parse_world_graph_script(&raw) else {
            return;
        };
        let final_size = graph.render_size.unwrap_or(graph.size);
        let effective_quality = self.scene_live_effective_preview_quality();
        let preview_size =
            Self::scene_live_preview_output_size(final_size, Some(effective_quality.max_dim()));
        let mut preview_graph = graph.clone();
        if preview_size != final_size {
            preview_graph.render_size = Some(preview_size);
        }

        let total_frames = self
            .playback_frame_count()
            .max(1)
            .min(SCENE_LIVE_PRERENDER_MAX_FRAMES);
        let script_hash = Self::script_hash(&raw);
        if let Some(cancel) = self.scene_live_prerender_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.scene_live_prerender_cancel = Some(cancel.clone());
        self.scene_live_prerender_token = self.scene_live_prerender_token.wrapping_add(1);
        let token = self.scene_live_prerender_token;
        self.scene_live_preview_cache_key = None;
        if !self.preview_playing {
            self.scene_live_preview_cache_image = None;
        }
        self.clear_scene_live_preview_frame_cache();
        self.scene_live_prerendering = true;
        self.scene_live_prerender_progress = Some((0, total_frames));

        enum WorldLivePrerenderEvent {
            Frame {
                frame: u32,
                width: u32,
                height: u32,
                bgra: Vec<u8>,
            },
            Finished(Result<u32, String>),
        }

        let asset_root = Self::world_asset_root();
        let (tx, rx) = mpsc::channel::<WorldLivePrerenderEvent>();
        std::thread::spawn(move || {
            let mut renderer = WorldFrameRenderer::new();
            for frame in 0..total_frames {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let rgba = match pollster::block_on(renderer.render_frame_gpu(
                    &preview_graph,
                    frame,
                    &asset_root,
                )) {
                    Ok(rgba) => rgba,
                    Err(err) => {
                        let _ = tx.send(WorldLivePrerenderEvent::Finished(Err(err.to_string())));
                        return;
                    }
                };
                let (width, height) = rgba.dimensions();
                let mut bgra = rgba.into_raw();
                for px in bgra.chunks_mut(4) {
                    px.swap(0, 2);
                }
                if tx
                    .send(WorldLivePrerenderEvent::Frame {
                        frame,
                        width,
                        height,
                        bgra,
                    })
                    .is_err()
                {
                    return;
                }
            }
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let _ = tx.send(WorldLivePrerenderEvent::Finished(Ok(total_frames)));
        });

        cx.spawn_in(window, async move |view, window| {
            let mut poll_ms = SCENE_LIVE_PRERENDER_POLL_MS;
            loop {
                let mut done = false;
                let mut should_notify = false;
                let _ = view.update_in(window, |this, window, cx| {
                    poll_ms = Self::scene_live_poll_ms(this.preview_playing);
                    if this.scene_live_prerender_token != token {
                        done = true;
                        return;
                    }

                    loop {
                        match rx.try_recv() {
                            Ok(WorldLivePrerenderEvent::Frame {
                                frame,
                                width,
                                height,
                                bgra,
                            }) => {
                                let key = (script_hash, frame, effective_quality, width, height);
                                if let Ok(preview) =
                                    Self::loaded_preview_from_bgra(width, height, bgra)
                                {
                                    this.scene_live_preview_frame_cache
                                        .insert(key, preview.clone());
                                    this.scene_live_prerender_progress =
                                        Some((frame.saturating_add(1), total_frames));
                                    if this.preview_playing || this.preview_frame == frame {
                                        this.preview_frame = frame;
                                        this.scene_live_preview_cache_key = Some(key);
                                        this.scene_live_preview_cache_image = Some(preview);
                                    }
                                    should_notify = true;
                                }
                            }
                            Ok(WorldLivePrerenderEvent::Finished(result)) => {
                                this.scene_live_prerendering = false;
                                this.scene_live_prerender_progress = None;
                                this.scene_live_prerender_cancel = None;
                                let finished_frame_count = result.as_ref().map(|frames| *frames).unwrap_or(total_frames);
                                match result {
                                    Ok(frames) => {
                                        if this.preview_playing {
                                            let frame_count = frames.max(1);
                                            this.preview_frame %= frame_count;
                                            this.preview_last_tick = Some(Instant::now());
                                            this.preview_frame_accum = 0.0;
                                            let play_token = this.preview_play_token;
                                            this.schedule_preview_playback(play_token, window, cx);
                                            this.status_line = format!(
                                                "GPU RAM preview cached {} frame(s); loop playback running.",
                                                frames
                                            );
                                        } else {
                                            this.status_line = format!(
                                                "GPU RAM preview cached {} frame(s).",
                                                frames
                                            );
                                        }
                                    }
                                    Err(err) => {
                                        if this.preview_playing {
                                            this.stop_preview_playback();
                                            this.preview_frame = finished_frame_count.saturating_sub(1);
                                        }
                                        this.status_line = format!("GPU RAM preview failed: {err}");
                                    }
                                }
                                should_notify = true;
                                done = true;
                                break;
                            }
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => {
                                this.scene_live_prerendering = false;
                                this.scene_live_prerender_progress = None;
                                this.scene_live_prerender_cancel = None;
                                should_notify = true;
                                done = true;
                                break;
                            }
                        }
                    }

                    if should_notify {
                        cx.notify();
                    }
                });
                if done {
                    break;
                }
                Timer::after(Duration::from_millis(poll_ms)).await;
            }
        })
        .detach();
    }

    fn step_preview_frame(&mut self, delta: i32) {
        self.stop_preview_playback();
        self.cancel_scene_live_prerender();
        self.set_preview_frame_value(if delta >= 0 {
            self.preview_frame.saturating_add(delta as u32)
        } else {
            self.preview_frame.saturating_sub(delta.unsigned_abs())
        });
    }

    fn set_preview_frame_value(&mut self, frame: u32) {
        let frame_count = self.playback_frame_count().max(1);
        let next_frame = frame.min(frame_count.saturating_sub(1));
        if self.preview_frame != next_frame {
            self.clear_scene_live_preview_overrides();
            self.scene_live_gizmo_drag = None;
            self.invalidate_scene_live_preview_render_only();
        }
        self.preview_frame = next_frame;
    }

    fn set_preview_frame(&mut self, frame: u32) {
        self.stop_preview_playback();
        self.cancel_scene_live_prerender();
        self.set_preview_frame_value(frame);
    }

    fn commit_preview_frame_input_value(&mut self, cx: &mut Context<Self>, stop_playback: bool) {
        let Some(input) = self.preview_frame_input.as_ref() else {
            return;
        };
        let raw = input.read(cx).value().trim().to_string();
        let Ok(frame) = raw.parse::<u32>() else {
            return;
        };
        if stop_playback {
            self.set_preview_frame(frame);
        } else {
            self.set_preview_frame_value(frame);
            self.preview_frame_accum = 0.0;
        }
    }

    fn sync_preview_frame_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(input) = self.preview_frame_input.clone() else {
            return;
        };
        let focused = input.read(cx).focus_handle(cx).is_focused(window);
        if focused {
            return;
        }
        let value = self.preview_frame.to_string();
        if input.read(cx).value() == value {
            return;
        }
        self.preview_frame_input_syncing = true;
        input.update(cx, |this, cx| {
            this.set_value(value, window, cx);
        });
        self.preview_frame_input_syncing = false;
    }

    fn stop_preview_playback(&mut self) {
        self.preview_playing = false;
        self.preview_play_token = self.preview_play_token.wrapping_add(1);
        self.preview_last_tick = None;
        self.preview_frame_accum = 0.0;
    }

    fn toggle_preview_playback(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.preview_playing {
            self.preview_playing = false;
            self.preview_last_tick = None;
            self.preview_frame_accum = 0.0;
            self.cancel_scene_live_prerender();
            self.status_line = format!("Paused at frame {}.", self.preview_frame);
            cx.notify();
            return;
        }
        if self.current_clip().is_none() && !self.has_scene_playback_graph() {
            self.status_line =
                "Import/select a clip or write a valid unified <Graph> before playback."
                    .to_string();
            cx.notify();
            return;
        }

        self.commit_preview_frame_input_value(cx, false);
        self.preview_playing = true;
        self.preview_play_token = self.preview_play_token.wrapping_add(1);
        self.preview_last_tick = Some(Instant::now());
        self.preview_frame_accum = 0.0;
        let token = self.preview_play_token;
        self.status_line = format!("Loop playback started at {} fps.", self.playback_fps());
        if self.has_scene_playback_graph() && !self.scene_external_preview_active() {
            self.start_scene_live_prerender(window, cx);
            self.schedule_preview_playback(token, window, cx);
        } else {
            self.schedule_preview_playback(token, window, cx);
        }
        cx.notify();
    }

    fn build_imported_clip(path: &str) -> ImportedClip {
        let pb = PathBuf::from(path);
        let name = pb
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path)
            .to_string();
        let duration = if Self::is_video_path(path) {
            get_media_duration(path)
        } else {
            Duration::ZERO
        };

        ImportedClip {
            name,
            path: path.to_string(),
            kind: if Self::is_image_path(path) {
                ImportedClipKind::Image
            } else {
                ImportedClipKind::Video
            },
            duration,
        }
    }

    fn build_vfx_asset_item(
        path: &str,
        ffmpeg_path: &str,
        cache_root: &Path,
        can_generate_video_thumbnail: bool,
    ) -> VfxAssetItem {
        let pb = PathBuf::from(path);
        let name = pb
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path)
            .to_string();
        let kind = if Self::is_image_path(path) {
            ImportedClipKind::Image
        } else {
            ImportedClipKind::Video
        };
        let duration = if kind == ImportedClipKind::Video {
            get_media_duration(path)
        } else {
            Duration::ZERO
        };

        if kind == ImportedClipKind::Image {
            return match Self::load_render_image(&pb) {
                Ok(preview) => VfxAssetItem {
                    name,
                    path: path.to_string(),
                    kind,
                    duration,
                    preview: Some(preview),
                    error: None,
                },
                Err(err) => VfxAssetItem {
                    name,
                    path: path.to_string(),
                    kind,
                    duration,
                    preview: None,
                    error: Some(err.to_string()),
                },
            };
        }

        let preview = if can_generate_video_thumbnail {
            let thumb_path = thumbnail::thumbnail_path_for_in(cache_root, &pb, THUMB_MAX_DIM);
            thumbnail::run_thumbnail_job(ffmpeg_path, &pb, &thumb_path, THUMB_MAX_DIM)
                .map_err(MotionLoomPageError::from)
                .and_then(|_| Self::load_render_image(&thumb_path))
                .ok()
        } else {
            None
        };

        VfxAssetItem {
            name,
            path: path.to_string(),
            kind,
            duration,
            preview,
            error: None,
        }
    }

    fn load_vfx_asset_folder(&mut self, folder: PathBuf, cx: &mut Context<Self>) {
        let (ffmpeg_path, cache_root, can_generate_video_thumbnail) = {
            let gs = self.global.read(cx);
            (
                gs.ffmpeg_path.clone(),
                gs.cache_root_dir(),
                gs.media_tools_ready_for_preview_gen(),
            )
        };

        let mut paths = match fs::read_dir(&folder) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.is_file())
                .filter(|path| {
                    path.to_str()
                        .map(Self::is_supported_clip_path)
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>(),
            Err(err) => {
                self.status_line = format!("VFX Assets folder error: {err}");
                return;
            }
        };
        paths.sort_by_key(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default()
        });

        self.asset_items = paths
            .iter()
            .filter_map(|path| path.to_str())
            .map(|path| {
                Self::build_vfx_asset_item(
                    path,
                    &ffmpeg_path,
                    &cache_root,
                    can_generate_video_thumbnail,
                )
            })
            .collect();
        self.asset_folder = Some(folder.clone());
        self.asset_selected_idx = if self.asset_items.is_empty() {
            None
        } else {
            Some(0)
        };
        self.asset_context_menu = None;
        self.status_line = if self.asset_items.is_empty() {
            format!(
                "VFX Assets loaded 0 item(s) from {}. No supported files found in this folder root.",
                folder.to_string_lossy()
            )
        } else {
            format!(
                "VFX Assets loaded {} item(s) from {}.",
                self.asset_items.len(),
                folder.to_string_lossy()
            )
        };
    }

    fn refresh_vfx_asset_folder(&mut self, cx: &mut Context<Self>) {
        let Some(folder) = self.asset_folder.clone() else {
            self.status_line = "Open a folder before refreshing VFX Assets.".to_string();
            return;
        };
        self.load_vfx_asset_folder(folder, cx);
    }

    fn current_clip(&self) -> Option<&ImportedClip> {
        let idx = self.selected_idx?;
        self.clips.get(idx)
    }

    fn sync_script_to_global(&mut self, cx: &mut Context<Self>, apply_now: bool) {
        let text = self.script_text.clone();
        let (script_revision, apply_revision) = self.global.update(cx, |gs, _cx| {
            gs.set_motionloom_scene_script(text, apply_now);
            (
                gs.motionloom_scene_script_revision(),
                gs.motionloom_scene_apply_revision(),
            )
        });
        self.motionloom_script_revision = script_revision;
        self.motionloom_apply_revision = apply_revision;
    }

    fn sync_preview_frame_to_global(&mut self, cx: &mut Context<Self>) {
        let frame = self.preview_frame;
        self.global.update(cx, |gs, _cx| {
            gs.set_motionloom_scene_preview_frame(frame);
        });
    }

    fn sync_script_from_global(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (global_script, script_revision, apply_revision) = {
            let gs = self.global.read(cx);
            (
                gs.motionloom_scene_script().to_string(),
                gs.motionloom_scene_script_revision(),
                gs.motionloom_scene_apply_revision(),
            )
        };

        if script_revision != self.motionloom_script_revision {
            self.motionloom_script_revision = script_revision;
            if global_script != self.script_text {
                self.set_script_text(global_script.clone(), window, cx);
            }
        }

        if apply_revision != self.motionloom_apply_revision {
            self.motionloom_apply_revision = apply_revision;
            if global_script != self.script_text {
                self.set_script_text(global_script, window, cx);
            }
            self.apply_script_command(cx);
        }
    }

    fn parse_scene_render_mode_token(token: &str) -> Option<SceneRenderMode> {
        match token.trim().to_ascii_lowercase().as_str() {
            "gpu" | "gpu_render" | "gpu_h264" | "gpu_native_h264" => {
                Some(SceneRenderMode::GpuNativeH264)
            }
            "gpu_prores" | "gpu-prores" | "prores_gpu" => Some(SceneRenderMode::GpuNativeProRes),
            "gpu_prores4444" | "gpu-prores4444" | "gpu_prores_4444" | "prores4444_gpu"
            | "prores_4444" => Some(SceneRenderMode::GpuNativeProRes4444),
            "gpu_png" | "gpu-png" | "png" | "png_sequence" | "png-sequence" => {
                Some(SceneRenderMode::GpuNativePngSequence)
            }
            "gpu_png_frame" | "gpu-png-frame" | "png_frame" | "png-current-frame"
            | "current_frame_png" | "current-frame-png" => {
                Some(SceneRenderMode::GpuNativeCurrentFramePng)
            }
            "compatibility_cpu" | "compatibility-cpu" | "cpu" | "cpu_render" => {
                Some(SceneRenderMode::CompatibilityCpu)
            }
            _ => None,
        }
    }

    fn sync_render_request_from_global(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (render_revision, render_mode) = {
            let gs = self.global.read(cx);
            (
                gs.motionloom_scene_render_revision(),
                gs.motionloom_scene_render_mode().map(ToString::to_string),
            )
        };

        if render_revision == self.motionloom_render_revision {
            return;
        }
        self.motionloom_render_revision = render_revision;

        let Some(mode_token) = render_mode else {
            return;
        };
        let Some(mode) = Self::parse_scene_render_mode_token(&mode_token) else {
            self.status_line = format!(
                "Unsupported MotionLoom render mode token from ACP: {}",
                mode_token
            );
            return;
        };
        self.render_scene_to_media_pool(mode, window, cx);
    }

    fn ensure_script_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.script_input.is_some() {
            return;
        }
        let input = cx.new(|cx| {
            InputState::new(window, cx)
                .code_editor("tsx")
                .rows(8)
                .line_number(true)
                .soft_wrap(true)
                .placeholder("<Graph ...> MotionLoom DSL script")
        });
        let initial = self.script_text.clone();
        input.update(cx, |this, cx| {
            this.set_value(initial.clone(), window, cx);
        });
        let sub = cx.subscribe(&input, |this, input, ev, cx| match ev {
            InputEvent::Change => {
                if this.script_input_syncing {
                    return;
                }
                this.script_text = input.read(cx).value().to_string();
                this.defer_scene_live_preview_render(cx, SCENE_LIVE_INPUT_RENDER_DEBOUNCE_MS);
                this.sync_script_to_global(cx, false);
            }
            InputEvent::PressEnter { secondary } => {
                if this.script_input_syncing {
                    return;
                }
                this.script_text = input.read(cx).value().to_string();
                if *secondary {
                    this.invalidate_scene_live_preview_cache();
                    this.sync_script_to_global(cx, false);
                    this.apply_script_command(cx);
                    cx.notify();
                } else {
                    this.defer_scene_live_preview_render(cx, SCENE_LIVE_INPUT_RENDER_DEBOUNCE_MS);
                    this.sync_script_to_global(cx, false);
                }
            }
            _ => {}
        });
        self.script_input = Some(input);
        self.script_input_sub = Some(sub);
    }

    fn ensure_scene_live_knob_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.scene_live_knob_input.is_some() {
            return;
        }
        let input = cx.new(|cx| InputState::new(window, cx).placeholder("value"));
        let sub = cx.subscribe(&input, |this, input, ev, cx| match ev {
            InputEvent::Change | InputEvent::PressEnter { .. } => {
                if this.scene_live_knob_input_syncing {
                    return;
                }
                let raw = input.read(cx).value().to_string();
                if let Some(value) = Self::parse_live_number(&raw) {
                    this.patch_scene_live_knob_value(
                        value,
                        None,
                        cx,
                        Some(SCENE_LIVE_INPUT_RENDER_DEBOUNCE_MS),
                    );
                    cx.notify();
                }
            }
            _ => {}
        });
        self.scene_live_knob_input = Some(input);
        self.scene_live_knob_input_sub = Some(sub);
    }

    fn ensure_preview_frame_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.preview_frame_input.is_some() {
            return;
        }
        let input = cx.new(|cx| InputState::new(window, cx).placeholder("frame"));
        let initial = self.preview_frame.to_string();
        input.update(cx, |this, cx| {
            this.set_value(initial, window, cx);
        });
        let sub = cx.subscribe(&input, |this, _input, ev, cx| match ev {
            InputEvent::PressEnter { .. } => {
                if this.preview_frame_input_syncing {
                    return;
                }
                this.commit_preview_frame_input_value(cx, true);
                cx.notify();
            }
            _ => {}
        });
        self.preview_frame_input = Some(input);
        self.preview_frame_input_sub = Some(sub);
    }

    fn sync_script_input_if_needed(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.script_input_needs_sync {
            return;
        }
        let Some(input) = self.script_input.as_ref() else {
            return;
        };
        let focused = input.read(cx).focus_handle(cx).is_focused(window);
        if focused {
            return;
        }
        self.script_input_needs_sync = false;
        self.script_input_syncing = true;
        let text = self.script_text.clone();
        input.update(cx, |this, cx| {
            this.set_value(text, window, cx);
        });
        self.script_input_syncing = false;
    }

    fn sync_scene_live_knob_input(
        &mut self,
        value: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(input) = self.scene_live_knob_input.as_ref() else {
            return;
        };
        let focused = input.read(cx).focus_handle(cx).is_focused(window);
        if focused {
            return;
        }
        let current = input.read(cx).value().to_string();
        if current == value {
            return;
        }
        self.scene_live_knob_input_syncing = true;
        input.update(cx, |this, cx| {
            this.set_value(value, window, cx);
        });
        self.scene_live_knob_input_syncing = false;
    }

    // Parse and compile the graph script, activating the runtime for preview.
    fn apply_script_command(&mut self, _cx: &mut Context<Self>) {
        let raw = self.script_text.clone();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            self.status_line =
                "Script is empty. Use the Template Picker or write a <Graph> script.".to_string();
            return;
        }

        // Only accept Graph DSL scripts (XML-based MotionLoom format)
        if !is_graph_script(&raw) {
            self.status_line = "Not a valid Graph script. Use the Template Picker to generate one, or write a <Graph ...> block.".to_string();
            return;
        }

        if Self::uses_pure_world_renderer(&raw) {
            match parse_world_graph_script(&raw) {
                Ok(graph) => {
                    self.graph_runtime = None;
                    self.preview_frame = 0;
                    self.preview_playing = false;
                    self.preview_last_tick = None;
                    self.preview_frame_accum = 0.0;
                    self.invalidate_scene_live_preview_cache();
                    self.status_line = format!(
                        "World DSL active | worlds={} actors={} fps={:.2} duration={}ms",
                        graph.worlds.len(),
                        graph
                            .worlds
                            .iter()
                            .map(|world| world.actors.len())
                            .sum::<usize>(),
                        graph.fps,
                        graph.duration_ms
                    );
                }
                Err(err) => {
                    self.graph_runtime = None;
                    self.status_line =
                        format!("World parse error at line {}: {}", err.line, err.message);
                }
            }
            return;
        }

        let parsed_runtime_graph = match parse_motionloom_document(&raw) {
            Ok(MotionLoomDocument::Process(graph) | MotionLoomDocument::Scene(graph)) => Ok(graph),
            Ok(MotionLoomDocument::Mixed(shell)) if shell.has_scene || shell.has_process => {
                // Mixed documents must stay on the scene composition parser; the process parser is
                // only for process-only Layer FX graphs with an external source clip.
                parse_graph_script(&raw)
            }
            Ok(MotionLoomDocument::Mixed(shell)) if shell.has_world => parse_graph_script(&raw),
            Ok(MotionLoomDocument::World(_)) => parse_graph_script(&raw),
            Ok(MotionLoomDocument::Mixed(_)) => parse_graph_script(&raw),
            Err(err) => Err(err),
        };
        match parsed_runtime_graph {
            Ok(graph) => match compile_runtime_program(graph.clone()) {
                Ok(runtime) => {
                    if !runtime.unsupported_kernels().is_empty() {
                        self.graph_runtime = None;
                        self.status_line = format!(
                            "Unsupported kernel(s): {}",
                            runtime.unsupported_kernels().join(", ")
                        );
                        return;
                    }
                    let graph_summary = graph.summary();
                    let runtime_summary = runtime.summary();
                    self.graph_runtime = Some(runtime);
                    self.preview_frame = 0;
                    self.preview_playing = false;
                    self.preview_last_tick = None;
                    self.preview_frame_accum = 0.0;
                    self.invalidate_scene_live_preview_cache();
                    self.status_line =
                        format!("Runtime ACTIVE | {} | {}", graph_summary, runtime_summary);
                }
                Err(err) => {
                    self.graph_runtime = None;
                    self.status_line = format!("Runtime compile error: {}", err.message);
                }
            },
            Err(err) => {
                self.graph_runtime = None;
                self.status_line =
                    format!("Graph parse error at line {}: {}", err.line, err.message);
            }
        }
    }

    fn push_scene_render_log(&mut self, message: impl Into<String>) {
        let message = message.into();
        if self
            .scene_render_log
            .last()
            .is_some_and(|last| last == &message)
        {
            return;
        }
        self.scene_render_log.push(message);
        if self.scene_render_log.len() > SCENE_RENDER_LOG_MAX_LINES {
            let excess = self.scene_render_log.len() - SCENE_RENDER_LOG_MAX_LINES;
            self.scene_render_log.drain(0..excess);
        }
    }

    fn render_scene_to_media_pool(
        &mut self,
        mode: SceneRenderMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.scene_render_modal_open = false;
        self.pending_non_alpha_scene_render_mode = None;
        let raw = self.script_text.clone();
        if mode == SceneRenderMode::GpuNativeCurrentFramePng {
            self.render_current_frame_png_to_media_pool(raw, window, cx);
            return;
        }
        self.scene_render_log.clear();
        self.push_scene_render_log(format!("{} requested.", mode.label()));
        let document = match parse_motionloom_document(&raw) {
            Ok(document) => document,
            Err(err) => {
                self.status_line = format!(
                    "MotionLoom graph parse error at line {}: {}",
                    err.line, err.message
                );
                self.push_scene_render_log(format!("ERROR: {}", self.status_line));
                cx.notify();
                return;
            }
        };
        self.push_scene_render_log("Graph parsed.".to_string());
        let (duration_ms, fps, world_only) = match &document {
            MotionLoomDocument::Scene(graph) => {
                if !graph.has_scene_nodes() && graph.world_sources.is_empty() {
                    self.status_line =
                        "Graph needs at least one renderable Scene or World node.".to_string();
                    self.push_scene_render_log(format!("ERROR: {}", self.status_line));
                    cx.notify();
                    return;
                }
                (graph.duration_ms, graph.fps, false)
            }
            MotionLoomDocument::World(graph) => (graph.duration_ms, graph.fps, true),
            MotionLoomDocument::Process(graph) => (graph.duration_ms, graph.fps, false),
            MotionLoomDocument::Mixed(shell)
                if shell.has_scene || shell.has_world || shell.has_process =>
            {
                let graph = match parse_graph_script(&raw) {
                    Ok(graph) => graph,
                    Err(err) => {
                        self.status_line = format!(
                            "Scene/World graph parse error at line {}: {}",
                            err.line, err.message
                        );
                        self.push_scene_render_log(format!("ERROR: {}", self.status_line));
                        cx.notify();
                        return;
                    }
                };
                (graph.duration_ms, graph.fps, false)
            }
            MotionLoomDocument::Mixed(_) => {
                self.status_line =
                    "Render supports Scene/World graphs and Scene/World + Process graphs. Process-only Layer FX graphs need a source clip and should be exported from the timeline."
                        .to_string();
                self.push_scene_render_log(format!("ERROR: {}", self.status_line));
                cx.notify();
                return;
            }
        };

        let (ffmpeg_path, output_dir) = {
            let gs = self.global.read(cx);
            if !gs.media_tools_ready_for_preview_gen() {
                self.status_line =
                    "MISSING_FFMPEG: MotionLoom graph render requires FFmpeg.".to_string();
                self.push_scene_render_log(format!("ERROR: {}", self.status_line));
                cx.notify();
                return;
            }
            (
                gs.ffmpeg_path.clone(),
                gs.generated_media_root_dir().join("motionloom_generated"),
            )
        };
        let profile = mode.profile();
        let output_path = if world_only {
            let prefix = match mode {
                SceneRenderMode::CompatibilityCpu => "motionloom_world",
                SceneRenderMode::GpuNativeH264 => "motionloom_world_gpu",
                SceneRenderMode::GpuNativeProRes => "motionloom_world_gpu_prores",
                SceneRenderMode::GpuNativeProRes4444 => "motionloom_world_gpu_prores4444",
                SceneRenderMode::GpuNativePngSequence => "motionloom_world_gpu_png",
                SceneRenderMode::GpuNativeCurrentFramePng => "motionloom_world_frame",
            };
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or(0);
            if profile.is_png_sequence() {
                output_dir.join(format!("{prefix}_{stamp}"))
            } else {
                output_dir.join(format!("{prefix}_{stamp}.{}", profile.output_extension()))
            }
        } else {
            match next_scene_output_path_for_profile(&output_dir, profile) {
                Ok(path) => path,
                Err(err) => {
                    self.status_line = format!("Scene output path error: {err}");
                    self.push_scene_render_log(format!("ERROR: {}", self.status_line));
                    cx.notify();
                    return;
                }
            }
        };
        let duration = Duration::from_millis(duration_ms);
        let global = self.global.clone();
        let ffmpeg_for_render = ffmpeg_path.clone();
        let total_frames = ((duration_ms as f64 / 1000.0) * fps as f64)
            .round()
            .max(1.0) as u32;
        let asset_root = Self::world_asset_root();

        self.status_line = format!(
            "{} started: {}...",
            mode.label(),
            output_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("motionloom_scene.mov")
        );
        self.push_scene_render_log(format!(
            "Starting worker: {} frames @ {:.2} fps.",
            total_frames, fps
        ));
        self.push_scene_render_log(format!("Output: {}", output_path.display()));
        if let Some(cancel) = self.scene_render_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        let render_cancel = Arc::new(AtomicBool::new(false));
        self.scene_render_cancel = Some(render_cancel.clone());
        self.scene_render_progress = Some(SceneRenderProgressUi {
            label: mode.label(),
            rendered_frames: 0,
            total_frames,
        });
        cx.notify();

        enum SceneRenderEvent {
            Log(String),
            Progress(MotionLoomRenderProgress),
            Finished(Result<PathBuf, String>),
        }

        let (tx, rx) = mpsc::channel::<SceneRenderEvent>();
        let output_path_for_thread = output_path.clone();
        let render_cancel_for_thread = render_cancel.clone();
        let render_cancel_for_poll = render_cancel.clone();
        std::thread::spawn(move || {
            let render_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let tx_progress = tx.clone();
                eprintln!(
                    "[motionloom-render] started mode={} output={}",
                    mode.label(),
                    output_path_for_thread.display()
                );
                let _ = tx_progress.send(SceneRenderEvent::Log(format!(
                    "Worker started: {} -> {}",
                    mode.label(),
                    output_path_for_thread.display()
                )));
                pollster::block_on(
                    render_motionloom_document_to_video_with_progress_and_cancel(
                        &ffmpeg_for_render,
                        &raw,
                        &asset_root,
                        &output_path_for_thread,
                        profile,
                        SCENE_RENDER_PROGRESS_EVERY_FRAMES,
                        Some(render_cancel_for_thread),
                        move |progress| {
                            let _ = tx_progress.send(SceneRenderEvent::Progress(progress));
                        },
                    ),
                )
                .map(|_| output_path_for_thread.clone())
                .map_err(|err| err.to_string())
            }))
            .unwrap_or_else(|payload| {
                Err(format!(
                    "MotionLoom render worker panicked: {}",
                    panic_payload_to_string(payload)
                ))
            });
            match &render_result {
                Ok(path) => {
                    eprintln!(
                        "[motionloom-render] finished mode={} output={}",
                        mode.label(),
                        path.display()
                    );
                    let _ = tx.send(SceneRenderEvent::Log(format!(
                        "Worker finished: {}",
                        path.display()
                    )));
                }
                Err(err) => {
                    if err.to_ascii_lowercase().contains("cancelled") {
                        if profile.is_png_sequence() {
                            let _ = fs::remove_dir_all(&output_path_for_thread);
                        } else {
                            let _ = fs::remove_file(&output_path_for_thread);
                        }
                    }
                    eprintln!("[motionloom-render] failed mode={}: {err}", mode.label());
                    let _ = tx.send(SceneRenderEvent::Log(format!("ERROR: {err}")));
                }
            }
            let _ = tx.send(SceneRenderEvent::Finished(render_result));
        });

        cx.spawn_in(window, async move |view, window| {
            loop {
                gpui::Timer::after(Duration::from_millis(SCENE_RENDER_PROGRESS_POLL_MS)).await;

                let mut latest_progress: Option<MotionLoomRenderProgress> = None;
                let mut log_messages: Vec<String> = Vec::new();
                let mut finished: Option<Result<PathBuf, String>> = None;
                loop {
                    match rx.try_recv() {
                        Ok(SceneRenderEvent::Log(message)) => {
                            log_messages.push(message);
                        }
                        Ok(SceneRenderEvent::Progress(progress)) => {
                            latest_progress = Some(progress);
                        }
                        Ok(SceneRenderEvent::Finished(result)) => {
                            finished = Some(result);
                            break;
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            finished = Some(Err("Scene render worker disconnected.".to_string()));
                            log_messages
                                .push("ERROR: Scene render worker disconnected.".to_string());
                            break;
                        }
                    }
                }

                let has_finished = finished.is_some();
                let _ = view.update_in(window, |this, _window, cx| {
                    let active_render = this
                        .scene_render_cancel
                        .as_ref()
                        .is_some_and(|cancel| Arc::ptr_eq(cancel, &render_cancel_for_poll));
                    if !active_render {
                        if has_finished {
                            cx.notify();
                        }
                        return;
                    }
                    for message in log_messages {
                        this.push_scene_render_log(message);
                    }
                    if let Some(progress) = latest_progress {
                        let rendered_frames = progress.rendered_frames();
                        let total_frames = progress.total_frames();
                        let pct = ((rendered_frames as f32 / total_frames as f32) * 100.0)
                            .round()
                            .clamp(0.0, 100.0) as u32;
                        this.scene_render_progress = Some(SceneRenderProgressUi {
                            label: mode.label(),
                            rendered_frames,
                            total_frames,
                        });
                        this.status_line = format!(
                            "{}: {}% ({}/{})",
                            mode.label(),
                            pct,
                            rendered_frames,
                            total_frames
                        );
                        this.push_scene_render_log(format!(
                            "Progress: {}% ({}/{})",
                            pct, rendered_frames, total_frames
                        ));
                    }

                    if let Some(result) = finished {
                        this.scene_render_progress = None;
                        this.scene_render_cancel = None;
                        match result {
                            Ok(path) => {
                                let path_str = path.to_string_lossy().to_string();
                                if mode.adds_media_pool_clip() {
                                    global.update(cx, |gs, cx| {
                                        gs.add_media_pool_item(path.clone(), duration);
                                        gs.ui_notice = Some(format!(
                                            "MotionLoom graph added to Media Pool: {path_str}"
                                        ));
                                        cx.emit(MediaPoolUiEvent::StateChanged);
                                        cx.notify();
                                    });

                                    let clip = Self::build_imported_clip(&path_str);
                                    this.clips.push(clip);
                                    this.selected_idx = Some(this.clips.len().saturating_sub(1));
                                } else {
                                    global.update(cx, |gs, cx| {
                                        gs.ui_notice = Some(format!(
                                            "MotionLoom graph PNG sequence saved: {path_str}"
                                        ));
                                        cx.notify();
                                    });
                                }

                                this.status_line = format!(
                                    "{} done: {}",
                                    mode.label(),
                                    path.file_name()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or("motionloom_scene")
                                );
                                this.push_scene_render_log(format!("Done: {}", path.display()));
                            }
                            Err(err) => {
                                this.status_line = format!("{} failed: {err}", mode.label());
                                this.push_scene_render_log(format!("ERROR: {err}"));
                                global.update(cx, |gs, cx| {
                                    gs.ui_notice =
                                        Some(format!("MotionLoom {} failed: {err}", mode.label()));
                                    cx.notify();
                                });
                            }
                        }
                    }
                    cx.notify();
                });

                if has_finished {
                    break;
                }
            }
        })
        .detach();
    }

    fn request_scene_render_from_modal(
        &mut self,
        mode: SceneRenderMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.scene_script_uses_transparent_background() && !mode.preserves_alpha_output() {
            self.pending_non_alpha_scene_render_mode = Some(mode);
            cx.notify();
            return;
        }
        self.pending_non_alpha_scene_render_mode = None;
        self.render_scene_to_media_pool(mode, window, cx);
    }

    fn scene_script_uses_transparent_background(&self) -> bool {
        let Ok(graph) = parse_graph_script(&self.script_text) else {
            return false;
        };
        graph
            .backgrounds
            .last()
            .is_some_and(|background| Self::is_transparent_color_literal(&background.color))
    }

    fn is_transparent_color_literal(value: &str) -> bool {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("transparent") {
            return true;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let inner = trimmed.trim_start_matches('[').trim_end_matches(']');
            let parts = inner
                .split(',')
                .map(str::trim)
                .filter_map(|part| part.parse::<f32>().ok())
                .collect::<Vec<_>>();
            return parts.len() == 4 && parts[3].abs() <= f32::EPSILON;
        }

        let Some(hex) = trimmed
            .strip_prefix('#')
            .or_else(|| trimmed.strip_prefix("0x"))
        else {
            return false;
        };
        hex.len() == 8 && hex[6..].eq_ignore_ascii_case("00")
    }

    fn render_current_frame_png_to_media_pool(
        &mut self,
        raw: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.scene_render_modal_open = false;
        self.scene_render_log.clear();
        self.push_scene_render_log(format!(
            "{} requested.",
            SceneRenderMode::GpuNativeCurrentFramePng.label()
        ));
        let output_dir = {
            let gs = self.global.read(cx);
            gs.generated_media_root_dir().join("motionloom_generated")
        };
        let requested_frame = self.preview_frame;
        let global = self.global.clone();
        let is_world = Self::uses_pure_world_renderer(&raw);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0);

        enum CurrentFrameRenderEvent {
            Log(String),
            Finished(Result<(PathBuf, u32), String>),
        }

        let (tx, rx) = mpsc::channel::<CurrentFrameRenderEvent>();

        if is_world {
            let graph = match parse_world_graph_script(&raw) {
                Ok(graph) => graph,
                Err(err) => {
                    self.status_line = format!(
                        "World graph parse error at line {}: {}",
                        err.line, err.message
                    );
                    self.push_scene_render_log(format!("ERROR: {}", self.status_line));
                    cx.notify();
                    return;
                }
            };
            let fps = graph.fps.max(1.0);
            let total_frames =
                (((graph.duration_ms as f32 / 1000.0).max(1.0 / fps) * fps).round() as u32).max(1);
            let frame = requested_frame.min(total_frames.saturating_sub(1));
            let output_path =
                output_dir.join(format!("motionloom_world_frame_{stamp}_f{frame:06}.png"));
            let asset_root = Self::world_asset_root();

            self.status_line = format!("PNG current frame started: frame {frame}...");
            self.push_scene_render_log(format!("Frame: {frame}/{total_frames}"));
            self.push_scene_render_log(format!("Output: {}", output_path.display()));
            self.scene_render_progress = Some(SceneRenderProgressUi {
                label: SceneRenderMode::GpuNativeCurrentFramePng.label(),
                rendered_frames: 0,
                total_frames: 1,
            });
            cx.notify();

            std::thread::spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = tx.send(CurrentFrameRenderEvent::Log(format!(
                        "PNG worker started: {}",
                        output_path.display()
                    )));
                    if let Some(parent) = output_path.parent() {
                        fs::create_dir_all(parent)
                            .map_err(|err| format!("Failed to create output directory: {err}"))?;
                    }
                    let mut renderer = WorldFrameRenderer::default();
                    let image =
                        pollster::block_on(renderer.render_frame_gpu(&graph, frame, &asset_root))
                            .map_err(|err| err.to_string())?;
                    image
                        .save(&output_path)
                        .map_err(|err| format!("Failed to save PNG frame: {err}"))?;
                    Ok((output_path, frame))
                }))
                .unwrap_or_else(|payload| {
                    Err(format!(
                        "PNG current frame worker panicked: {}",
                        panic_payload_to_string(payload)
                    ))
                });
                match &result {
                    Ok((path, _)) => {
                        let _ = tx.send(CurrentFrameRenderEvent::Log(format!(
                            "PNG worker finished: {}",
                            path.display()
                        )));
                    }
                    Err(err) => {
                        let _ = tx.send(CurrentFrameRenderEvent::Log(format!("ERROR: {err}")));
                    }
                }
                let _ = tx.send(CurrentFrameRenderEvent::Finished(result));
            });
        } else {
            let graph = match parse_graph_script(&raw) {
                Ok(graph) => graph,
                Err(err) => {
                    self.status_line = format!(
                        "Scene graph parse error at line {}: {}",
                        err.line, err.message
                    );
                    self.push_scene_render_log(format!("ERROR: {}", self.status_line));
                    cx.notify();
                    return;
                }
            };
            if !graph.has_scene_nodes() {
                self.status_line =
                    "Graph needs at least one node before exporting a PNG frame.".to_string();
                self.push_scene_render_log(format!("ERROR: {}", self.status_line));
                cx.notify();
                return;
            }
            let fps = graph.fps.max(1.0);
            let total_frames =
                (((graph.duration_ms as f32 / 1000.0).max(1.0 / fps) * fps).round() as u32).max(1);
            let frame = requested_frame.min(total_frames.saturating_sub(1));
            let output_path =
                output_dir.join(format!("motionloom_scene_frame_{stamp}_f{frame:06}.png"));

            self.status_line = format!("PNG current frame started: frame {frame}...");
            self.push_scene_render_log(format!("Frame: {frame}/{total_frames}"));
            self.push_scene_render_log(format!("Output: {}", output_path.display()));
            self.scene_render_progress = Some(SceneRenderProgressUi {
                label: SceneRenderMode::GpuNativeCurrentFramePng.label(),
                rendered_frames: 0,
                total_frames: 1,
            });
            cx.notify();

            std::thread::spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = tx.send(CurrentFrameRenderEvent::Log(format!(
                        "PNG worker started: {}",
                        output_path.display()
                    )));
                    if let Some(parent) = output_path.parent() {
                        fs::create_dir_all(parent)
                            .map_err(|err| format!("Failed to create output directory: {err}"))?;
                    }
                    let image = pollster::block_on(render_scene_graph_frame(
                        &graph,
                        frame,
                        SceneRenderProfile::Gpu,
                    ))
                    .map_err(|err| err.to_string())?;
                    image
                        .save(&output_path)
                        .map_err(|err| format!("Failed to save PNG frame: {err}"))?;
                    Ok((output_path, frame))
                }))
                .unwrap_or_else(|payload| {
                    Err(format!(
                        "PNG current frame worker panicked: {}",
                        panic_payload_to_string(payload)
                    ))
                });
                match &result {
                    Ok((path, _)) => {
                        let _ = tx.send(CurrentFrameRenderEvent::Log(format!(
                            "PNG worker finished: {}",
                            path.display()
                        )));
                    }
                    Err(err) => {
                        let _ = tx.send(CurrentFrameRenderEvent::Log(format!("ERROR: {err}")));
                    }
                }
                let _ = tx.send(CurrentFrameRenderEvent::Finished(result));
            });
        }

        cx.spawn_in(window, async move |view, window| {
            loop {
                gpui::Timer::after(Duration::from_millis(SCENE_RENDER_PROGRESS_POLL_MS)).await;
                let mut finished: Option<Result<(PathBuf, u32), String>> = None;
                let mut log_messages: Vec<String> = Vec::new();
                loop {
                    match rx.try_recv() {
                        Ok(CurrentFrameRenderEvent::Log(message)) => {
                            log_messages.push(message);
                        }
                        Ok(CurrentFrameRenderEvent::Finished(result)) => {
                            finished = Some(result);
                            break;
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            finished =
                                Some(Err("PNG current frame worker disconnected.".to_string()));
                            log_messages
                                .push("ERROR: PNG current frame worker disconnected.".to_string());
                            break;
                        }
                    }
                }

                let has_finished = finished.is_some();
                let _ = view.update_in(window, |this, _window, cx| {
                    for message in log_messages {
                        this.push_scene_render_log(message);
                    }
                    if let Some(result) = finished {
                        this.scene_render_progress = None;
                        match result {
                            Ok((path, frame)) => {
                                let path_str = path.to_string_lossy().to_string();
                                global.update(cx, |gs, cx| {
                                    gs.add_media_pool_item(path.clone(), Duration::from_secs(1));
                                    gs.ui_notice = Some(format!(
                                        "MotionLoom PNG frame added to Media Pool: {path_str}"
                                    ));
                                    cx.emit(MediaPoolUiEvent::StateChanged);
                                    cx.notify();
                                });

                                let clip = Self::build_imported_clip(&path_str);
                                this.clips.push(clip);
                                this.selected_idx = Some(this.clips.len().saturating_sub(1));
                                this.status_line = format!(
                                    "PNG current frame done: frame {} -> {}",
                                    frame,
                                    path.file_name()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or("motionloom_frame.png")
                                );
                                this.push_scene_render_log(format!("Done: {}", path.display()));
                            }
                            Err(err) => {
                                this.status_line = format!("PNG current frame failed: {err}");
                                this.push_scene_render_log(format!("ERROR: {err}"));
                            }
                        }
                        cx.notify();
                    }
                });

                if has_finished {
                    break;
                }
            }
        })
        .detach();
    }

    // Set the script text and sync into the input widget.
    fn set_script_text(&mut self, text: String, window: &mut Window, cx: &mut Context<Self>) {
        self.script_text = text.clone();
        self.invalidate_scene_live_preview_cache();
        if let Some(input) = self.script_input.as_ref() {
            input.update(cx, |this, cx| {
                this.set_value(text, window, cx);
            });
        }
        self.sync_script_to_global(cx, false);
    }

    fn set_script_text_deferred_sync(&mut self, text: String, cx: &mut Context<Self>) {
        self.script_text = text;
        self.invalidate_scene_live_preview_cache();
        self.script_input_needs_sync = true;
        self.sync_script_to_global(cx, false);
    }

    // Update script while keeping the last preview image visible until the next frame finishes.
    fn set_script_text_preserve_preview(
        &mut self,
        text: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.script_text = text.clone();
        self.clear_scene_live_preview_overrides();
        self.cancel_scene_live_prerender();
        self.cancel_scene_live_async_render();
        self.scene_live_preview_cache_key = None;
        self.clear_scene_live_preview_frame_cache();
        if text.trim().is_empty() || !is_graph_script(&text) {
            self.scene_live_preview_cache_image = None;
            self.preview_playing = false;
            self.preview_last_tick = None;
            self.preview_frame_accum = 0.0;
        }
        self.scene_live_render_defer_until = None;
        self.scene_live_render_defer_token = self.scene_live_render_defer_token.wrapping_add(1);
        if let Some(input) = self.script_input.as_ref() {
            input.update(cx, |this, cx| {
                this.set_value(text, window, cx);
            });
        }
        self.sync_script_to_global(cx, false);
    }

    fn control_button(label: &'static str) -> gpui::Div {
        div()
            .h(px(28.0))
            .flex_shrink_0()
            .px_2()
            .rounded_md()
            .border_1()
            .border_color(white().opacity(0.15))
            .bg(white().opacity(0.06))
            .hover(|s| s.bg(white().opacity(0.1)))
            .cursor_pointer()
            .text_xs()
            .text_color(white().opacity(0.9))
            .flex()
            .items_center()
            .justify_center()
            .child(label)
    }

    fn scene_render_choice_button(mode: SceneRenderMode) -> gpui::Div {
        div()
            .w_full()
            .rounded_lg()
            .border_1()
            .border_color(white().opacity(0.14))
            .bg(white().opacity(0.055))
            .hover(|s| s.bg(white().opacity(0.10)))
            .cursor_pointer()
            .p_3()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_sm()
                    .text_color(white().opacity(0.94))
                    .child(mode.label()),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(white().opacity(0.56))
                    .child(match mode {
                        SceneRenderMode::CompatibilityCpu => {
                            "Compatibility path. Slower, useful when GPU-native output has issues."
                        }
                        SceneRenderMode::GpuNativeH264 => "Fast H.264 MP4. No alpha channel.",
                        SceneRenderMode::GpuNativeProRes => {
                            "High-quality ProRes 422 MOV. No alpha channel."
                        }
                        SceneRenderMode::GpuNativeProRes4444 => {
                            "Alpha-preserving ProRes 4444 MOV for transparent overlays."
                        }
                        SceneRenderMode::GpuNativePngSequence => {
                            "Alpha-preserving PNG frames in an output folder."
                        }
                        SceneRenderMode::GpuNativeCurrentFramePng => {
                            "Alpha-preserving PNG for the current preview frame. Adds the image to Media Pool."
                        }
                    }),
            )
    }

    fn render_scene_render_modal_overlay(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let choice = |mode: SceneRenderMode| {
            Self::scene_render_choice_button(mode).on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, window, cx| {
                    this.request_scene_render_from_modal(mode, window, cx);
                }),
            )
        };
        let pending_non_alpha_mode = self.pending_non_alpha_scene_render_mode;

        div()
            .absolute()
            .top_0()
            .bottom_0()
            .left_0()
            .right_0()
            .bg(rgba(0x0000009e))
            .flex()
            .items_center()
            .justify_center()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.scene_render_modal_open = false;
                    this.pending_non_alpha_scene_render_mode = None;
                    cx.notify();
                }),
            )
            .child(
                div()
                    .w(px(520.0))
                    .rounded_lg()
                    .border_1()
                    .border_color(white().opacity(0.16))
                    .bg(rgb(0x111827))
                    .shadow_2xl()
                    .p_4()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_, _, _, cx| {
                            cx.stop_propagation();
                        }),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(white().opacity(0.96))
                                    .child("Render MotionLoom Scene"),
                            )
                            .child(Self::control_button("Close").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(|this, _, _, cx| {
                                    this.scene_render_modal_open = false;
                                    this.pending_non_alpha_scene_render_mode = None;
                                    cx.notify();
                                }),
                            )),
                    )
                    .when_some(pending_non_alpha_mode, |el, mode| {
                        el.child(
                            div()
                                .rounded_lg()
                                .border_1()
                                .border_color(rgba(0xf59e0baa))
                                .bg(rgba(0x3b250880))
                                .p_3()
                                .flex()
                                .flex_col()
                                .gap_2()
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgba(0xffe0a3ff))
                                        .child("Transparent background warning"),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(white().opacity(0.72))
                                        .child(
                                            "This script uses a transparent background, but the selected render mode does not support alpha output.",
                                        ),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(white().opacity(0.72))
                                        .child(format!(
                                            "Selected mode: {}. The render can continue, but the transparent background will not be preserved.",
                                            mode.label()
                                        )),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(white().opacity(0.72))
                                        .child(
                                            "Use ProRes 4444 Alpha or PNG output if you need transparency.",
                                        ),
                                ),
                        )
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .gap_2()
                                .child(Self::control_button("Back").on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(|this, _, _, cx| {
                                        this.pending_non_alpha_scene_render_mode = None;
                                        cx.notify();
                                    }),
                                ))
                                .child(Self::control_button("Render Anyway").on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, window, cx| {
                                        this.render_scene_to_media_pool(mode, window, cx);
                                    }),
                                )),
                        )
                    })
                    .when(pending_non_alpha_mode.is_none(), |el| {
                        el.child(
                            div()
                                .text_xs()
                                .text_color(white().opacity(0.62))
                                .child("Choose a render mode. Use ProRes 4444 Alpha or PNG output for transparent background output."),
                        )
                        .child(choice(SceneRenderMode::GpuNativeH264))
                        .child(choice(SceneRenderMode::GpuNativeProRes))
                        .child(choice(SceneRenderMode::GpuNativeProRes4444))
                        .child(choice(SceneRenderMode::GpuNativeCurrentFramePng))
                        .child(choice(SceneRenderMode::GpuNativePngSequence))
                        .child(choice(SceneRenderMode::CompatibilityCpu))
                    }),
            )
    }

    fn scene_live_chip(label: String, active: bool) -> gpui::Div {
        let border = if active {
            rgba(0x79c7ffcc)
        } else {
            rgba(0xffffff24)
        };
        let bg = if active {
            rgba(0x1f5c85aa)
        } else {
            rgba(0xffffff0e)
        };
        div()
            .h(px(26.0))
            .flex_shrink_0()
            .px_2()
            .rounded_md()
            .border_1()
            .border_color(border)
            .bg(bg)
            .hover(|s| s.bg(white().opacity(0.11)))
            .cursor_pointer()
            .text_xs()
            .text_color(white().opacity(if active { 0.96 } else { 0.78 }))
            .flex()
            .items_center()
            .justify_center()
            .child(label)
    }

    fn scene_live_checkbox(label: &'static str, checked: bool) -> gpui::Div {
        let border = if checked {
            rgba(0x79c7ffcc)
        } else {
            rgba(0xffffff28)
        };
        let bg = if checked {
            rgba(0x1f5c85aa)
        } else {
            rgba(0xffffff0c)
        };
        div()
            .h(px(28.0))
            .flex_shrink_0()
            .px_2()
            .rounded_md()
            .border_1()
            .border_color(border)
            .bg(bg)
            .hover(|s| s.bg(white().opacity(0.11)))
            .cursor_pointer()
            .flex()
            .items_center()
            .gap_2()
            .child(
                div()
                    .w(px(14.0))
                    .h(px(14.0))
                    .rounded_sm()
                    .border_1()
                    .border_color(if checked {
                        rgba(0x9bd8ffdd)
                    } else {
                        rgba(0xffffff45)
                    })
                    .bg(if checked {
                        rgba(0x2b78aacc)
                    } else {
                        rgba(0xffffff08)
                    })
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(if checked {
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(white().opacity(0.95))
                            .child("x")
                    } else {
                        div()
                    }),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(white().opacity(if checked { 0.95 } else { 0.76 }))
                    .child(label),
            )
    }

    // --- Template picker logic (ported from inspector_panel) ---

    fn template_label(kind: LayerEffectTemplateKind) -> &'static str {
        match kind {
            LayerEffectTemplateKind::BlurGaussian => "Blur Gaussian",
            LayerEffectTemplateKind::BlurGaussianHorizontal => "Blur Horizontal",
            LayerEffectTemplateKind::BlurGaussianVertical => "Blur Vertical",
            LayerEffectTemplateKind::Sharpen => "Sharpen",
            LayerEffectTemplateKind::Opacity => "Opacity",
            LayerEffectTemplateKind::Lut => "LUT",
            LayerEffectTemplateKind::HslaOverlay => "HSLA Overlay",
            LayerEffectTemplateKind::TransitionFadeInOut => "Transition Fade In/Out",
        }
    }

    fn toggle_template_selection(&mut self, kind: LayerEffectTemplateKind) {
        if let Some(idx) = self
            .template_selected
            .iter()
            .position(|selected| *selected == kind)
        {
            self.template_selected.remove(idx);
            return;
        }
        self.template_selected.push(kind);
    }

    fn selected_template_summary(&self) -> String {
        if self.template_selected.is_empty() {
            return "No templates selected.".to_string();
        }
        self.template_selected
            .iter()
            .map(|kind| Self::template_label(*kind))
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    // Apply selected templates: generate a new graph script or append to existing.
    fn apply_selected_templates(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.template_selected.is_empty() {
            self.status_line = "Choose at least one template before pressing OK.".to_string();
            return;
        }

        let add_time = self.template_add_time_parameter;
        let add_curve = self.template_add_curve_parameter;
        let selected = self.template_selected.clone();
        let existing_script = self.script_text.trim().to_string();
        let selection_label = self.selected_template_summary();

        // Build new chain or append to existing graph
        let result = if existing_script.is_empty() || !is_graph_script(&existing_script) {
            motionloom_templates::build_layer_effect_chain_script(&selected, add_time, add_curve)
        } else {
            motionloom_templates::append_layer_effect_template_chain_script(
                &existing_script,
                &selected,
                add_curve,
            )
        };

        let Some(script) = result else {
            self.status_line =
                "Current script is not a standard chainable layer graph. Clear it before applying a multi-template selection."
                    .to_string();
            return;
        };

        self.set_script_text(script, window, cx);
        self.template_modal_open = false;
        self.template_selected.clear();
        self.status_line = if existing_script.is_empty() || !is_graph_script(&existing_script) {
            if add_time && add_curve {
                format!("Inserted template chain: {selection_label} (+apply graph + curve params).")
            } else if add_time {
                format!("Inserted template chain: {selection_label} (+apply graph, duration 5s).")
            } else if add_curve {
                format!("Inserted template chain: {selection_label} (+curve params).")
            } else {
                format!("Inserted template chain: {selection_label}.")
            }
        } else if add_curve {
            format!("Appended template chain: {selection_label} (+curve params).")
        } else {
            format!("Appended template chain: {selection_label}.")
        };
    }

    fn open_template_modal(&mut self) {
        self.import_modal_open = false;
        self.asset_modal_open = false;
        self.scene_template_modal_open = false;
        self.scene_render_modal_open = false;
        self.glb_inspector_modal_open = false;
        self.template_modal_open = true;
        self.template_selected.clear();
        self.status_line = "Template picker opened.".to_string();
    }

    fn scene_template_items() -> SearchableVec<String> {
        SearchableVec::new(vec![
            "showcase".to_string(),
            "process".to_string(),
            "scene".to_string(),
            "text".to_string(),
            "material_and_texture".to_string(),
        ])
    }

    fn ensure_scene_template_select(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.scene_template_select.is_some() {
            return;
        }

        let state = cx.new(|cx| {
            SelectState::new(Self::scene_template_items(), None, window, cx).searchable(false)
        });
        let selected_label = if self.scene_template_selected_label.is_empty() {
            DEFAULT_SCENE_TEMPLATE_CATEGORY.to_string()
        } else {
            self.scene_template_selected_label.clone()
        };
        state.update(cx, |this, cx| {
            this.set_selected_value(&selected_label, window, cx);
        });

        self.scene_template_selected_label = selected_label;
        self.scene_template_select = Some(state);
    }

    fn ensure_scene_template_number_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.scene_template_number_input.is_some() {
            return;
        }
        let initial = if self.scene_template_number.trim().is_empty() {
            DEFAULT_SCENE_TEMPLATE_NUMBER.to_string()
        } else {
            self.scene_template_number.clone()
        };
        let input = cx.new(|cx| InputState::new(window, cx).placeholder("1"));
        input.update(cx, |this, cx| {
            this.set_value(initial.clone(), window, cx);
        });
        let sub = cx.subscribe(&input, |this, input, ev, cx| {
            if matches!(ev, InputEvent::Change | InputEvent::PressEnter { .. }) {
                this.scene_template_number = input.read(cx).value().to_string();
                cx.notify();
            }
        });
        self.scene_template_number_input = Some(input);
        self.scene_template_number_input_sub = Some(sub);
    }

    fn motionloom_example_config(category: &str) -> (&'static str, &'static str, &'static str) {
        match category {
            "process" => ("core/process", "cp", "main_with_scene.motionloom"),
            "scene" => ("core/scene", "cs", "main.motionloom"),
            "text" => ("core/text", "ct", "main.motionloom"),
            "material_and_texture" => ("core/material_and_texture", "cm", "main.motionloom"),
            _ => ("showcase", "s", "main.motionloom"),
        }
    }

    fn motionloom_example_id(category: &str, number: &str) -> Option<String> {
        let number = number.trim().parse::<u32>().ok()?;
        if number == 0 {
            return None;
        }
        let (_, prefix, _) = Self::motionloom_example_config(category);
        Some(format!("{prefix}-{number:06}"))
    }

    fn motionloom_example_url(category: &str, number: &str) -> Option<(String, String)> {
        let id = Self::motionloom_example_id(category, number)?;
        let (folder, _, entry) = Self::motionloom_example_config(category);
        Some((
            id.clone(),
            format!("{MOTIONLOOM_EXAMPLE_RAW_ROOT}/{folder}/{id}/{entry}"),
        ))
    }

    fn load_scene_template_from_github(
        &mut self,
        category: String,
        number: String,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some((id, url)) = Self::motionloom_example_url(&category, &number) else {
            self.status_line = "Enter a valid motionloom-example number.".to_string();
            cx.notify();
            return;
        };
        self.scene_template_selected_label = category.clone();
        self.scene_template_number = number.trim().to_string();
        self.status_line = format!("Loading {category}/{id} from GitHub...");
        cx.notify();

        let fallback_script = motionloom_templates::first_scene_template()
            .map(|template| template.script.to_string())
            .unwrap_or_default();
        cx.spawn(async move |view, cx| {
            let fetch_url = url.clone();
            let result = cx
                .background_spawn(async move {
                    let response = ureq::get(&fetch_url)
                        .set("User-Agent", "Anica MotionLoom Studio")
                        .call()
                        .map_err(|err| err.to_string())?;
                    response.into_string().map_err(|err| err.to_string())
                })
                .await;
            let _ = view.update(cx, |this, cx| {
                match result {
                    Ok(script) => {
                        this.set_script_text_deferred_sync(script, cx);
                        this.status_line = format!("Loaded GitHub motionloom-example: {id}.");
                    }
                    Err(err) => {
                        if fallback_script.is_empty() {
                            this.status_line = format!("Failed to load {id}: {err}");
                        } else {
                            this.set_script_text_deferred_sync(fallback_script, cx);
                            this.status_line =
                                format!("Failed to load {id}; inserted built-in fallback: {err}");
                        }
                    }
                }
                this.scene_template_modal_open = false;
                cx.notify();
            });
        })
        .detach();
    }

    fn open_scene_template_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.import_modal_open = false;
        self.asset_modal_open = false;
        self.template_modal_open = false;
        self.scene_render_modal_open = false;
        self.glb_inspector_modal_open = false;

        self.ensure_scene_template_select(window, cx);
        self.ensure_scene_template_number_input(window, cx);

        if self.scene_template_select.is_none() {
            self.status_line = "No scene templates are available.".to_string();
            cx.notify();
            return;
        }

        self.scene_template_modal_open = true;
        self.status_line = "GitHub motionloom-example loader opened.".to_string();

        cx.notify();
    }
    fn render_scene_template_modal_overlay(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let select_state = self.scene_template_select.as_ref().cloned();
        let number_input = self.scene_template_number_input.as_ref().cloned();

        div()
            .absolute()
            .top_0()
            .bottom_0()
            .left_0()
            .right_0()
            .bg(gpui_component::black().opacity(0.62))
            .flex()
            .items_center()
            .justify_center()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.scene_template_modal_open = false;
                    this.status_line = "Scene template selector closed.".to_string();
                    cx.notify();
                }),
            )
            .child(
                div()
                    .w(px(620.0))
                    .max_h(px(620.0))
                    .rounded_lg()
                    .border_1()
                    .border_color(white().opacity(0.16))
                    .bg(rgb(0x111827))
                    .shadow_2xl()
                    .p_4()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_, _, _, cx| {
                            cx.stop_propagation();
                        }),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(white().opacity(0.96))
                                    .child("GitHub MotionLoom Example"),
                            )
                            .child(
                                div()
                                    .h(px(28.0))
                                    .px_3()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(white().opacity(0.16))
                                    .bg(white().opacity(0.05))
                                    .text_xs()
                                    .text_color(white().opacity(0.85))
                                    .hover(|s| s.bg(white().opacity(0.10)))
                                    .cursor_pointer()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child("Close")
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.scene_template_modal_open = false;
                                            this.status_line =
                                                "Scene template selector closed.".to_string();
                                            cx.notify();
                                        }),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.68))
                            .child(
                                "Load examples from LOVELYZOMBIEYHO/motionloom-example, matching the landing page MotionLoom browser.",
                            ),
                    )
                    .child(
                        div()
                            .grid()
                            .grid_cols(2)
                            .gap_3()
                            .when_some(select_state.clone(), |el, select_state| {
                                el.child(
                                    div()
                                        .flex()
                                        .flex_col()
                                        .gap_1()
                                        .child(
                                            div()
                                                .text_xs()
                                                .text_color(white().opacity(0.62))
                                                .child("Category"),
                                        )
                                        .child(
                                            Select::new(&select_state)
                                                .placeholder("showcase")
                                                .menu_width(px(280.0))
                                                .w_full(),
                                        ),
                                )
                            })
                            .when_some(number_input.clone(), |el, input| {
                                el.child(
                                    div()
                                        .flex()
                                        .flex_col()
                                        .gap_1()
                                        .child(
                                            div()
                                                .text_xs()
                                                .text_color(white().opacity(0.62))
                                                .child("Number"),
                                        )
                                        .child(Input::new(&input).h(px(32.0)).w_full()),
                                )
                            }),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.54))
                            .child(
                                "Examples: showcase 1 -> showcase/s-000001/main.motionloom; process 14 -> core/process/cp-000014/main_with_scene.motionloom.",
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_end()
                            .gap_2()
                            .child(
                                div()
                                    .h(px(30.0))
                                    .px_3()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(white().opacity(0.16))
                                    .bg(white().opacity(0.05))
                                    .text_xs()
                                    .text_color(white().opacity(0.85))
                                    .hover(|s| s.bg(white().opacity(0.10)))
                                    .cursor_pointer()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child("Cancel")
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.scene_template_modal_open = false;
                                            this.status_line =
                                                "Scene template selector closed.".to_string();
                                            cx.notify();
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .h(px(30.0))
                                    .px_3()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(rgba(0x3b82f6cc))
                                    .bg(rgba(0x2563ebdd))
                                    .text_xs()
                                    .text_color(white().opacity(0.96))
                                    .hover(|s| s.bg(rgba(0x3b82f6ee)))
                                    .cursor_pointer()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child("Load Example")
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, window, cx| {
                                            let category = this
                                                .scene_template_select
                                                .as_ref()
                                                .and_then(|select| {
                                                    select.read(cx).selected_value().cloned()
                                                })
                                                .unwrap_or_else(|| {
                                                    DEFAULT_SCENE_TEMPLATE_CATEGORY.to_string()
                                                });
                                            let number = this
                                                .scene_template_number_input
                                                .as_ref()
                                                .map(|input| {
                                                    input.read(cx).value().to_string()
                                                })
                                                .unwrap_or_else(|| {
                                                    this.scene_template_number.clone()
                                                });

                                            this.load_scene_template_from_github(
                                                category, number, window, cx,
                                            );
                                        }),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_h(px(160.0))
                            .max_h(px(320.0))
                            .overflow_y_scrollbar()
                            .rounded_md()
                            .border_1()
                            .border_color(white().opacity(0.10))
                            .bg(white().opacity(0.03))
                            .p_2()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .children(
                                [
                                    "showcase: full public showcase examples",
                                    "process: core process examples, loads main_with_scene.motionloom",
                                    "scene: core scene examples",
                                    "text: core text examples",
                                    "material_and_texture: texture/material examples",
                                ]
                                .into_iter()
                                .map(|line| {
                                    div()
                                        .text_xs()
                                        .text_color(white().opacity(0.58))
                                        .child(line)
                                }),
                            ),
                    ),
            )
    }

    fn open_glb_inspector_modal(&mut self, cx: &mut Context<Self>) {
        self.import_modal_open = false;
        self.asset_modal_open = false;
        self.template_modal_open = false;
        self.scene_template_modal_open = false;
        self.scene_render_modal_open = false;

        match self.build_glb_skeleton_report() {
            Ok(report) => {
                self.status_line = report.summary.clone();
                self.glb_inspector_report = Some(report);
            }
            Err(message) => {
                self.status_line = message.clone();
                self.glb_inspector_report = Some(GlbSkeletonInspectReport {
                    summary: message,
                    retarget_draft: String::new(),
                    model_profile_draft: String::new(),
                    calibrated_model_profile_draft: String::new(),
                    actors: Vec::new(),
                });
            }
        }
        self.glb_inspector_modal_open = true;
        cx.notify();
    }

    fn build_glb_skeleton_report(&self) -> Result<GlbSkeletonInspectReport, String> {
        let raw = self.script_text.trim();
        if raw.is_empty() {
            return Err("Inspect GLB needs a MotionLoom Graph script.".to_string());
        }
        let graph = if is_world_graph_script(raw) {
            parse_world_graph_script(raw).map_err(|err| {
                format!(
                    "Legacy 3D parse error before GLB inspect at line {}: {}",
                    err.line, err.message
                )
            })?
        } else {
            let scene_graph = parse_graph_script(raw).map_err(|err| {
                format!(
                    "Scene parse error before GLB inspect at line {}: {}",
                    err.line, err.message
                )
            })?;
            Self::scene_graph_for_glb_inspector(&scene_graph)?
        };
        let asset_root = Self::world_asset_root();
        let mut actors = Vec::<GlbSkeletonActorReport>::new();

        for world in &graph.worlds {
            for actor in &world.actors {
                let path =
                    Self::resolve_world_model_path(&asset_root, &actor.model, actor.path_style);
                let current_retarget = actor
                    .retarget
                    .as_deref()
                    .and_then(|id| graph.retargets.iter().find(|retarget| retarget.id == id))
                    .or_else(|| {
                        graph
                            .retargets
                            .iter()
                            .find(|retarget| retarget.actor.as_deref() == Some(actor.id.as_str()))
                    });
                let current_profile = actor
                    .profile
                    .as_deref()
                    .and_then(|id| graph.model_profiles.iter().find(|profile| profile.id == id));
                let current_profile_retarget =
                    current_profile.and_then(|profile| profile.retarget.as_ref());
                let mapped_bones = current_retarget
                    .map(|retarget| {
                        retarget
                            .maps
                            .iter()
                            .map(|map| map.to.clone())
                            .collect::<HashSet<_>>()
                    })
                    .or_else(|| {
                        current_profile_retarget.map(|retarget| {
                            retarget
                                .maps
                                .iter()
                                .map(|map| map.to.clone())
                                .collect::<HashSet<_>>()
                        })
                    })
                    .unwrap_or_default();
                let mut mapped_bones_sorted = mapped_bones.iter().cloned().collect::<Vec<_>>();
                mapped_bones_sorted.sort();
                let missing_humanoid_bones = Self::humanoid_bones()
                    .iter()
                    .filter(|bone| !mapped_bones.contains(**bone))
                    .map(|bone| (*bone).to_string())
                    .collect::<Vec<_>>();

                let report = match load_glb_mesh_data(&path) {
                    Ok(mesh) => {
                        let joint_node_indices = mesh
                            .skin
                            .as_ref()
                            .map(|skin| {
                                skin.joints
                                    .iter()
                                    .map(|joint| joint.node_index)
                                    .collect::<HashSet<_>>()
                            })
                            .unwrap_or_default();
                        let joint_names = mesh
                            .skin
                            .as_ref()
                            .map(|skin| {
                                skin.joints
                                    .iter()
                                    .filter_map(|joint| joint.name.clone())
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let guessed_maps = Self::guess_humanoid_retarget_maps(&joint_names);
                        let sorted_guessed_maps = Self::sort_retarget_maps(&guessed_maps);
                        let model_profile_draft = Self::build_model_profile_draft(
                            &actor.id,
                            &actor.model,
                            &mesh,
                            &sorted_guessed_maps,
                            false,
                        );
                        let calibrated_model_profile_draft = Self::build_model_profile_draft(
                            &actor.id,
                            &actor.model,
                            &mesh,
                            &sorted_guessed_maps,
                            true,
                        );
                        let calibration_preview_lines =
                            Self::calibrate_bone_axis_map(&mesh, &sorted_guessed_maps)
                                .preview_lines;
                        let weighted_vertex_count = mesh
                            .weights
                            .iter()
                            .zip(mesh.joints.iter())
                            .filter(|(weights, joints)| {
                                joints.is_some()
                                    && weights
                                        .as_ref()
                                        .is_some_and(|weights| weights.iter().sum::<f32>() > 0.0)
                            })
                            .count();
                        let has_inverse_bind_matrices = mesh.skin.as_ref().is_some_and(|skin| {
                            skin.joints.iter().any(|joint| {
                                joint
                                    .inverse_bind_matrix
                                    .iter()
                                    .zip(Self::identity_mat4().iter())
                                    .any(|(a, b)| (a - b).abs() > 0.0001)
                            })
                        });
                        GlbSkeletonActorReport {
                            actor_id: actor.id.clone(),
                            model: actor.model.clone(),
                            resolved_path: path.display().to_string(),
                            vertex_count: mesh.positions.len(),
                            triangle_count: mesh.triangles.len(),
                            node_count: mesh.nodes.len(),
                            joint_count: joint_node_indices.len(),
                            weighted_vertex_count,
                            has_inverse_bind_matrices,
                            mapped_bones: mapped_bones_sorted,
                            missing_humanoid_bones,
                            guessed_maps,
                            joint_tree_lines: Self::glb_joint_tree_lines(
                                &mesh.nodes,
                                &joint_node_indices,
                            ),
                            model_profile_draft,
                            calibrated_model_profile_draft,
                            calibration_preview_lines,
                            error: None,
                        }
                    }
                    Err(err) => GlbSkeletonActorReport {
                        actor_id: actor.id.clone(),
                        model: actor.model.clone(),
                        resolved_path: path.display().to_string(),
                        vertex_count: 0,
                        triangle_count: 0,
                        node_count: 0,
                        joint_count: 0,
                        weighted_vertex_count: 0,
                        has_inverse_bind_matrices: false,
                        mapped_bones: mapped_bones_sorted,
                        missing_humanoid_bones,
                        guessed_maps: Vec::new(),
                        joint_tree_lines: Vec::new(),
                        model_profile_draft: String::new(),
                        calibrated_model_profile_draft: String::new(),
                        calibration_preview_lines: Vec::new(),
                        error: Some(err.to_string()),
                    },
                };
                actors.push(report);
            }
        }

        if actors.is_empty() {
            return Err("No skinned <Model> was found in the current Scene graph.".to_string());
        }

        let retarget_draft = Self::build_retarget_draft(&actors);
        let model_profile_draft = Self::build_model_profile_drafts(&actors);
        let calibrated_model_profile_draft = Self::build_calibrated_model_profile_drafts(&actors);
        let loaded_count = actors.iter().filter(|actor| actor.error.is_none()).count();
        Ok(GlbSkeletonInspectReport {
            summary: format!(
                "GLB inspect: {} actor(s), {} loaded, {} retarget draft map(s), {} ModelProfile draft(s), {} calibrated profile draft(s).",
                actors.len(),
                loaded_count,
                actors
                    .iter()
                    .map(|actor| actor.guessed_maps.len())
                    .sum::<usize>(),
                actors
                    .iter()
                    .filter(|actor| !actor.model_profile_draft.trim().is_empty())
                    .count(),
                actors
                    .iter()
                    .filter(|actor| !actor.calibrated_model_profile_draft.trim().is_empty())
                    .count()
            ),
            retarget_draft,
            model_profile_draft,
            calibrated_model_profile_draft,
            actors,
        })
    }

    fn scene_graph_for_glb_inspector(graph: &GraphScript) -> Result<WorldGraph, String> {
        let model_assets = graph
            .assets
            .iter()
            .filter(|asset| asset.kind == motionloom::GraphAssetKind::Model)
            .map(|asset| (asset.id.clone(), asset.src.clone()))
            .collect::<HashMap<_, _>>();
        let mut actors = Vec::new();
        for scene in &graph.scenes {
            Self::collect_scene_glb_actors(&scene.children, &model_assets, &mut actors);
        }
        Self::collect_scene_glb_actors(&graph.scene_nodes, &model_assets, &mut actors);
        if actors.is_empty() {
            return Err("No <Model asset=\"...\"> found inside a 3D CompositeGroup.".to_string());
        }
        let model_profiles = graph
            .model_profiles
            .iter()
            .filter(|profile| profile.kind.eq_ignore_ascii_case("3d"))
            .map(|profile| WorldModelProfile {
                id: profile.id.clone(),
                model: profile.model.clone().unwrap_or_default(),
                preset: profile.preset.clone(),
                retarget: profile
                    .retarget
                    .as_ref()
                    .map(|retarget| WorldProfileRetarget {
                        preset: retarget.preset.clone(),
                        maps: retarget
                            .maps
                            .iter()
                            .map(|map| WorldRetargetMap {
                                from: map.from.clone(),
                                to: map.to.clone(),
                            })
                            .collect(),
                    }),
                bone_axis_map: None,
            })
            .collect();
        let world_id = "__scene_glb_inspector".to_string();
        Ok(WorldGraph {
            id: Some(world_id.clone()),
            version: Some("scene-inspector-v1".to_string()),
            fps: graph.fps,
            duration_ms: graph.duration_ms,
            duration_explicit: graph.duration_explicit,
            size: graph.size,
            render_size: graph.render_size,
            model_profiles,
            worlds: vec![WorldNode {
                id: world_id.clone(),
                background: Some(WorldBackground {
                    id: None,
                    src: None,
                    fit: WorldBackgroundFit::Stretch,
                    color: "#00000000".to_string(),
                    opacity: "1".to_string(),
                }),
                camera: WorldCamera::default(),
                actors,
                directional_characters: Vec::new(),
            }],
            retargets: Vec::new(),
            actions: Vec::new(),
            apply_actions: Vec::new(),
            animation_assets: Vec::new(),
            constraints: Vec::new(),
            present: WorldPresent { from: world_id },
        })
    }

    fn collect_scene_glb_actors(
        nodes: &[SceneNode],
        assets: &HashMap<String, String>,
        actors: &mut Vec<WorldActor>,
    ) {
        for node in nodes {
            let children = match node {
                SceneNode::Timeline(node) => Some(node.children.as_slice()),
                SceneNode::Track(node) => Some(node.children.as_slice()),
                SceneNode::Sequence(node) => Some(node.children.as_slice()),
                SceneNode::Chain(node) => Some(node.children.as_slice()),
                SceneNode::Group(node) => {
                    if let Some(composite) = node.composite.as_ref() {
                        for model in composite.nodes_3d.iter().filter_map(|node| match node {
                            Scene3DNode::Model(model) => Some(model),
                            _ => None,
                        }) {
                            let actor_id = model
                                .id
                                .clone()
                                .unwrap_or_else(|| format!("scene_model_{}", actors.len()));
                            let source = assets
                                .get(&model.asset)
                                .cloned()
                                .unwrap_or_else(|| model.asset.clone());
                            actors.push(WorldActor {
                                id: actor_id,
                                model: source,
                                path_style: WorldPathStyle::Relative,
                                hide_meshes: Vec::new(),
                                hide_materials: Vec::new(),
                                profile: model.profile.clone(),
                                rig: model.rig.clone(),
                                retarget: model.retarget.clone(),
                                x: "0".to_string(),
                                y: "0".to_string(),
                                z: "0".to_string(),
                                yaw: "0".to_string(),
                                pitch: "0".to_string(),
                                roll: "0".to_string(),
                                scale: "1".to_string(),
                                opacity: "1".to_string(),
                                material: None,
                                play: None,
                                plays: Vec::new(),
                            });
                        }
                    }
                    Some(node.children.as_slice())
                }
                SceneNode::Part(node) => Some(node.children.as_slice()),
                SceneNode::Repeat(node) => Some(node.children.as_slice()),
                SceneNode::Mask(node) => Some(node.children.as_slice()),
                SceneNode::Precompose(node) => Some(node.children.as_slice()),
                SceneNode::Layer(node) => Some(node.children.as_slice()),
                SceneNode::Camera(node) => Some(node.children.as_slice()),
                SceneNode::Character(node) => Some(node.children.as_slice()),
                SceneNode::Puppet(node) => Some(node.children.as_slice()),
                _ => None,
            };
            if let Some(children) = children {
                Self::collect_scene_glb_actors(children, assets, actors);
            }
        }
    }

    fn resolve_world_model_path(
        asset_root: &Path,
        src: &str,
        path_style: WorldPathStyle,
    ) -> PathBuf {
        let path = Path::new(src);
        match path_style {
            WorldPathStyle::Absolute => path.to_path_buf(),
            WorldPathStyle::Relative => {
                if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    asset_root.join(path)
                }
            }
        }
    }

    fn humanoid_bones() -> &'static [&'static str] {
        &[
            "hips",
            "spine",
            "chest",
            "neck",
            "head",
            "shoulder_l",
            "upper_arm_l",
            "forearm_l",
            "hand_l",
            "shoulder_r",
            "upper_arm_r",
            "forearm_r",
            "hand_r",
            "upper_leg_l",
            "lower_leg_l",
            "foot_l",
            "toe_l",
            "upper_leg_r",
            "lower_leg_r",
            "foot_r",
            "toe_r",
        ]
    }

    fn sort_retarget_maps(maps: &[(String, String)]) -> Vec<(String, String)> {
        let order = Self::humanoid_bones()
            .iter()
            .enumerate()
            .map(|(index, bone)| (*bone, index))
            .collect::<HashMap<_, _>>();
        let mut sorted = maps.to_vec();
        sorted.sort_by(|(_, a), (_, b)| {
            order
                .get(a.as_str())
                .copied()
                .unwrap_or(usize::MAX)
                .cmp(&order.get(b.as_str()).copied().unwrap_or(usize::MAX))
                .then_with(|| a.cmp(b))
        });
        sorted
    }

    fn build_model_profile_drafts(actors: &[GlbSkeletonActorReport]) -> String {
        actors
            .iter()
            .filter_map(|actor| {
                let draft = actor.model_profile_draft.trim();
                if draft.is_empty() {
                    None
                } else {
                    Some(draft.to_string())
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn build_calibrated_model_profile_drafts(actors: &[GlbSkeletonActorReport]) -> String {
        actors
            .iter()
            .filter_map(|actor| {
                let draft = actor.calibrated_model_profile_draft.trim();
                if draft.is_empty() {
                    None
                } else {
                    Some(draft.to_string())
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn build_model_profile_draft(
        actor_id: &str,
        model: &str,
        mesh: &motionloom::GlbMeshData,
        maps: &[(String, String)],
        calibrated: bool,
    ) -> String {
        if maps.is_empty() {
            return String::new();
        }

        let profile_id = format!("{}_profile", Self::sanitize_xml_ident(actor_id));
        let axis_lines = if calibrated {
            Self::calibrate_bone_axis_map(mesh, maps).axis_lines
        } else {
            Self::infer_bone_axis_map_lines(mesh, maps)
        };
        let mut out = String::new();
        out.push_str(&format!(
            "<ModelProfile id=\"{}\"\n",
            Self::xml_attr_escape(&profile_id)
        ));
        out.push_str("              kind=\"3d\"\n");
        out.push_str(&format!(
            "              model=\"{}\"\n",
            Self::xml_attr_escape(model)
        ));
        out.push_str("              preset=\"humanoid_v1\">\n");
        if calibrated {
            out.push_str("  <!-- Auto calibration draft: retarget guess + simulated rotationX/Y/Z endpoint tests. Review once per model. -->\n");
        } else {
            out.push_str("  <!-- Auto draft from GLB joint names + rest-pose geometry. Review once per model. -->\n");
        }
        out.push_str("  <Retarget>\n");
        for (from, to) in maps {
            out.push_str(&format!(
                "    <Map from=\"{}\" to=\"{}\" />\n",
                Self::xml_attr_escape(from),
                Self::xml_attr_escape(to)
            ));
        }
        out.push_str("  </Retarget>\n");

        if !axis_lines.is_empty() {
            out.push_str("\n  <BoneAxisMap>\n");
            for line in axis_lines {
                out.push_str("    ");
                out.push_str(&line);
                out.push('\n');
            }
            out.push_str("  </BoneAxisMap>\n");
        }

        out.push_str("</ModelProfile>");
        out
    }

    fn calibrate_bone_axis_map(
        mesh: &motionloom::GlbMeshData,
        maps: &[(String, String)],
    ) -> GlbAxisCalibration {
        let bone_to_node = Self::bone_to_node_name(maps);
        let node_name_to_index = Self::node_name_to_index(&mesh.nodes);
        let global = Self::glb_global_node_matrices(&mesh.nodes);
        let positions = global
            .iter()
            .map(|matrix| Self::mat4_transform_point(*matrix, [0.0, 0.0, 0.0]))
            .collect::<Vec<_>>();
        let basis = Self::glb_rest_pose_basis(&positions, &bone_to_node, &node_name_to_index);
        let mut axis_lines = Vec::new();
        let mut preview_lines = Vec::new();

        for bone in Self::humanoid_bones() {
            if !bone_to_node.contains_key(*bone) {
                continue;
            }

            let mut attrs = Vec::<(&'static str, GlbAxisBindingScore)>::new();
            match *bone {
                "hips" => {
                    if let Some(score) = Self::score_twist_binding(
                        *bone,
                        basis.up,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("turn", score));
                    }
                }
                "spine" | "chest" | "head" => {
                    if let Some(score) = Self::score_twist_binding(
                        *bone,
                        basis.up,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("turn", score));
                    }
                    if let Some(score) = Self::score_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        basis.forward,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("bend", score));
                    }
                }
                "upper_arm_r" | "upper_arm_l" => {
                    let side_target = if bone.ends_with("_r") {
                        basis.right
                    } else {
                        Self::vec3_scale(basis.right, -1.0)
                    };
                    if let Some(score) = Self::score_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        basis.forward,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("forward", score));
                    }
                    if let Some(score) = Self::score_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        side_target,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("side", score));
                    }
                    if let Some(score) = Self::score_twist_binding(
                        *bone,
                        Self::bone_direction(
                            *bone,
                            Self::humanoid_child_for_axis(*bone),
                            &positions,
                            &bone_to_node,
                            &node_name_to_index,
                        )
                        .unwrap_or(basis.up),
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("twist", score));
                    }
                }
                "forearm_r" | "forearm_l" | "hand_r" | "hand_l" => {
                    if let Some(score) = Self::score_bend_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        Self::humanoid_parent_for_axis(*bone),
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("bend", score));
                    }
                    if let Some(score) = Self::score_twist_binding(
                        *bone,
                        Self::bone_direction(
                            *bone,
                            Self::humanoid_child_for_axis(*bone)
                                .or_else(|| Self::humanoid_parent_for_axis(*bone)),
                            &positions,
                            &bone_to_node,
                            &node_name_to_index,
                        )
                        .unwrap_or(basis.up),
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("twist", score));
                    }
                }
                "upper_leg_r" | "upper_leg_l" => {
                    let side_target = if bone.ends_with("_r") {
                        basis.right
                    } else {
                        Self::vec3_scale(basis.right, -1.0)
                    };
                    if let Some(score) = Self::score_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        basis.forward,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("forward", score));
                    }
                    if let Some(score) = Self::score_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        side_target,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("side", score));
                    }
                }
                "lower_leg_r" | "lower_leg_l" | "foot_r" | "foot_l" => {
                    if let Some(score) = Self::score_bend_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        Self::humanoid_parent_for_axis(*bone),
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("bend", score));
                    }
                }
                _ => {}
            }

            if attrs.is_empty() {
                continue;
            }

            preview_lines.extend(attrs.iter().map(|(semantic, score)| {
                format!(
                    "{}.{} -> {}  score={:.2}",
                    bone, semantic, score.binding, score.score
                )
            }));
            let attrs = attrs
                .into_iter()
                .map(|(key, score)| {
                    format!("{}=\"{}\"", key, Self::xml_attr_escape(&score.binding))
                })
                .collect::<Vec<_>>()
                .join(" ");
            axis_lines.push(format!(
                "<Axis bone=\"{}\" {} />",
                Self::xml_attr_escape(bone),
                attrs
            ));
        }

        GlbAxisCalibration {
            axis_lines,
            preview_lines,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn score_movement_binding(
        bone: &str,
        child: Option<&str>,
        fallback_child: Option<&str>,
        target: [f32; 3],
        global: &[[f32; 16]],
        positions: &[[f32; 3]],
        bone_to_node: &HashMap<String, String>,
        node_name_to_index: &HashMap<String, usize>,
    ) -> Option<GlbAxisBindingScore> {
        let bone_index = Self::node_index_for_bone(bone, bone_to_node, node_name_to_index)?;
        let child_index = child
            .or(fallback_child)
            .and_then(|child| Self::node_index_for_bone(child, bone_to_node, node_name_to_index))?;
        let bone_pos = positions.get(bone_index).copied()?;
        let child_pos = positions.get(child_index).copied()?;
        let rest_vec = Self::vec3_sub(child_pos, bone_pos);
        let target = Self::vec3_normalize(target)?;
        let local_axes = Self::local_axes(global.get(bone_index).copied()?);

        let mut best = None::<GlbAxisBindingScore>;
        for (axis_index, axis_world) in local_axes.into_iter().enumerate() {
            for scale in [1.0, -1.0] {
                let rotated = Self::rotate_vec3_around_axis(
                    rest_vec,
                    axis_world,
                    30.0_f32.to_radians() * scale,
                );
                let Some(movement) = Self::vec3_normalize(Self::vec3_sub(rotated, rest_vec)) else {
                    continue;
                };
                let score = Self::vec3_dot(movement, target).max(0.0);
                if score <= 0.02 {
                    continue;
                }
                let candidate = GlbAxisBindingScore {
                    binding: Self::axis_binding(axis_index, scale),
                    score,
                };
                if best
                    .as_ref()
                    .is_none_or(|best| candidate.score > best.score)
                {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn score_bend_binding(
        bone: &str,
        child: Option<&str>,
        parent: Option<&str>,
        global: &[[f32; 16]],
        positions: &[[f32; 3]],
        bone_to_node: &HashMap<String, String>,
        node_name_to_index: &HashMap<String, usize>,
    ) -> Option<GlbAxisBindingScore> {
        let bone_index = Self::node_index_for_bone(bone, bone_to_node, node_name_to_index)?;
        let child_index = child
            .and_then(|child| Self::node_index_for_bone(child, bone_to_node, node_name_to_index));
        let parent_index = parent
            .and_then(|parent| Self::node_index_for_bone(parent, bone_to_node, node_name_to_index));
        let child_index = child_index.or(parent_index)?;
        let child_pos = positions.get(child_index).copied()?;
        let target = parent_index
            .and_then(|index| positions.get(index).copied())
            .map(|parent_pos| Self::vec3_sub(parent_pos, child_pos))
            .or_else(|| {
                positions
                    .get(bone_index)
                    .copied()
                    .map(|bone_pos| Self::vec3_sub(bone_pos, child_pos))
            })?;
        Self::score_movement_binding(
            bone,
            child,
            parent,
            target,
            global,
            positions,
            bone_to_node,
            node_name_to_index,
        )
    }

    fn score_twist_binding(
        bone: &str,
        target: [f32; 3],
        global: &[[f32; 16]],
        _positions: &[[f32; 3]],
        bone_to_node: &HashMap<String, String>,
        node_name_to_index: &HashMap<String, usize>,
    ) -> Option<GlbAxisBindingScore> {
        let target = Self::vec3_normalize(target)?;
        let node_index = Self::node_index_for_bone(bone, bone_to_node, node_name_to_index)?;
        let local_axes = Self::local_axes(global.get(node_index).copied()?);
        let mut best = None::<GlbAxisBindingScore>;
        for (axis_index, axis_world) in local_axes.into_iter().enumerate() {
            let score = Self::vec3_dot(axis_world, target);
            let (scale, score_abs) = if score >= 0.0 {
                (1.0, score)
            } else {
                (-1.0, -score)
            };
            let candidate = GlbAxisBindingScore {
                binding: Self::axis_binding(axis_index, scale),
                score: score_abs,
            };
            if best
                .as_ref()
                .is_none_or(|best| candidate.score > best.score)
            {
                best = Some(candidate);
            }
        }
        best
    }

    fn infer_bone_axis_map_lines(
        mesh: &motionloom::GlbMeshData,
        maps: &[(String, String)],
    ) -> Vec<String> {
        let bone_to_node = Self::bone_to_node_name(maps);
        let node_name_to_index = Self::node_name_to_index(&mesh.nodes);
        let global = Self::glb_global_node_matrices(&mesh.nodes);
        let positions = global
            .iter()
            .map(|matrix| Self::mat4_transform_point(*matrix, [0.0, 0.0, 0.0]))
            .collect::<Vec<_>>();
        let basis = Self::glb_rest_pose_basis(&positions, &bone_to_node, &node_name_to_index);
        let mut out = Vec::new();

        for bone in Self::humanoid_bones() {
            if !bone_to_node.contains_key(*bone) {
                continue;
            }
            let mut attrs = Vec::<(&'static str, String)>::new();
            match *bone {
                "hips" => {
                    if let Some(binding) = Self::infer_twist_binding(
                        *bone,
                        &basis.up,
                        mesh,
                        &global,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("turn", binding));
                    }
                }
                "spine" | "chest" | "head" => {
                    if let Some(binding) = Self::infer_twist_binding(
                        *bone,
                        &basis.up,
                        mesh,
                        &global,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("turn", binding));
                    }
                    if let Some(binding) = Self::infer_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        basis.forward,
                        mesh,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("bend", binding));
                    }
                }
                "upper_arm_r" | "upper_arm_l" => {
                    let side_target = if bone.ends_with("_r") {
                        basis.right
                    } else {
                        Self::vec3_scale(basis.right, -1.0)
                    };
                    if let Some(binding) = Self::infer_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        basis.forward,
                        mesh,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("forward", binding));
                    }
                    if let Some(binding) = Self::infer_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        side_target,
                        mesh,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("side", binding));
                    }
                    if let Some(binding) = Self::infer_twist_binding(
                        *bone,
                        &Self::bone_direction(
                            *bone,
                            Self::humanoid_child_for_axis(*bone),
                            &positions,
                            &bone_to_node,
                            &node_name_to_index,
                        )
                        .unwrap_or(basis.up),
                        mesh,
                        &global,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("twist", binding));
                    }
                }
                "forearm_r" | "forearm_l" | "hand_r" | "hand_l" => {
                    if let Some(binding) = Self::infer_bend_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        Self::humanoid_parent_for_axis(*bone),
                        mesh,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("bend", binding));
                    }
                    if let Some(binding) = Self::infer_twist_binding(
                        *bone,
                        &Self::bone_direction(
                            *bone,
                            Self::humanoid_child_for_axis(*bone)
                                .or_else(|| Self::humanoid_parent_for_axis(*bone)),
                            &positions,
                            &bone_to_node,
                            &node_name_to_index,
                        )
                        .unwrap_or(basis.up),
                        mesh,
                        &global,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("twist", binding));
                    }
                }
                "upper_leg_r" | "upper_leg_l" => {
                    let side_target = if bone.ends_with("_r") {
                        basis.right
                    } else {
                        Self::vec3_scale(basis.right, -1.0)
                    };
                    if let Some(binding) = Self::infer_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        basis.forward,
                        mesh,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("forward", binding));
                    }
                    if let Some(binding) = Self::infer_movement_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        None,
                        side_target,
                        mesh,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("side", binding));
                    }
                }
                "lower_leg_r" | "lower_leg_l" | "foot_r" | "foot_l" => {
                    if let Some(binding) = Self::infer_bend_binding(
                        *bone,
                        Self::humanoid_child_for_axis(*bone),
                        Self::humanoid_parent_for_axis(*bone),
                        mesh,
                        &global,
                        &positions,
                        &bone_to_node,
                        &node_name_to_index,
                    ) {
                        attrs.push(("bend", binding));
                    }
                }
                _ => {}
            }

            if !attrs.is_empty() {
                let attrs = attrs
                    .into_iter()
                    .map(|(key, value)| format!("{}=\"{}\"", key, Self::xml_attr_escape(&value)))
                    .collect::<Vec<_>>()
                    .join(" ");
                out.push(format!(
                    "<Axis bone=\"{}\" {} />",
                    Self::xml_attr_escape(bone),
                    attrs
                ));
            }
        }

        out
    }

    fn bone_to_node_name(maps: &[(String, String)]) -> HashMap<String, String> {
        maps.iter()
            .map(|(from, to)| (to.clone(), from.clone()))
            .collect()
    }

    fn node_name_to_index(nodes: &[motionloom::GlbNodeData]) -> HashMap<String, usize> {
        nodes
            .iter()
            .filter_map(|node| node.name.as_ref().map(|name| (name.clone(), node.index)))
            .collect()
    }

    fn node_index_for_bone(
        bone: &str,
        bone_to_node: &HashMap<String, String>,
        node_name_to_index: &HashMap<String, usize>,
    ) -> Option<usize> {
        bone_to_node
            .get(bone)
            .and_then(|name| node_name_to_index.get(name))
            .copied()
    }

    fn humanoid_child_for_axis(bone: &str) -> Option<&'static str> {
        match bone {
            "hips" => Some("spine"),
            "spine" => Some("chest"),
            "chest" => Some("neck"),
            "neck" => Some("head"),
            "shoulder_l" => Some("upper_arm_l"),
            "upper_arm_l" => Some("forearm_l"),
            "forearm_l" => Some("hand_l"),
            "hand_l" => None,
            "shoulder_r" => Some("upper_arm_r"),
            "upper_arm_r" => Some("forearm_r"),
            "forearm_r" => Some("hand_r"),
            "hand_r" => None,
            "upper_leg_l" => Some("lower_leg_l"),
            "lower_leg_l" => Some("foot_l"),
            "foot_l" => Some("toe_l"),
            "upper_leg_r" => Some("lower_leg_r"),
            "lower_leg_r" => Some("foot_r"),
            "foot_r" => Some("toe_r"),
            _ => None,
        }
    }

    fn humanoid_parent_for_axis(bone: &str) -> Option<&'static str> {
        match bone {
            "spine" => Some("hips"),
            "chest" => Some("spine"),
            "neck" => Some("chest"),
            "head" => Some("neck"),
            "upper_arm_l" => Some("shoulder_l"),
            "forearm_l" => Some("upper_arm_l"),
            "hand_l" => Some("forearm_l"),
            "upper_arm_r" => Some("shoulder_r"),
            "forearm_r" => Some("upper_arm_r"),
            "hand_r" => Some("forearm_r"),
            "upper_leg_l" => Some("hips"),
            "lower_leg_l" => Some("upper_leg_l"),
            "foot_l" => Some("lower_leg_l"),
            "toe_l" => Some("foot_l"),
            "upper_leg_r" => Some("hips"),
            "lower_leg_r" => Some("upper_leg_r"),
            "foot_r" => Some("lower_leg_r"),
            "toe_r" => Some("foot_r"),
            _ => None,
        }
    }

    fn glb_rest_pose_basis(
        positions: &[[f32; 3]],
        bone_to_node: &HashMap<String, String>,
        node_name_to_index: &HashMap<String, usize>,
    ) -> GlbRestPoseBasis {
        let pos = |bone: &str| {
            Self::node_index_for_bone(bone, bone_to_node, node_name_to_index)
                .and_then(|index| positions.get(index).copied())
        };

        let up = pos("head")
            .zip(pos("hips"))
            .and_then(|(head, hips)| Self::vec3_normalize(Self::vec3_sub(head, hips)))
            .or_else(|| {
                pos("chest")
                    .zip(pos("hips"))
                    .and_then(|(chest, hips)| Self::vec3_normalize(Self::vec3_sub(chest, hips)))
            })
            .unwrap_or([0.0, 1.0, 0.0]);
        let right = pos("shoulder_r")
            .zip(pos("shoulder_l"))
            .and_then(|(right, left)| Self::vec3_normalize(Self::vec3_sub(right, left)))
            .or_else(|| {
                pos("upper_arm_r")
                    .zip(pos("upper_arm_l"))
                    .and_then(|(right, left)| Self::vec3_normalize(Self::vec3_sub(right, left)))
            })
            .unwrap_or([1.0, 0.0, 0.0]);
        let forward = Self::vec3_normalize(Self::vec3_cross(right, up)).unwrap_or([0.0, 0.0, 1.0]);

        GlbRestPoseBasis { right, up, forward }
    }

    #[allow(clippy::too_many_arguments)]
    fn infer_movement_binding(
        bone: &str,
        child: Option<&str>,
        fallback_child: Option<&str>,
        target: [f32; 3],
        _mesh: &motionloom::GlbMeshData,
        global: &[[f32; 16]],
        positions: &[[f32; 3]],
        bone_to_node: &HashMap<String, String>,
        node_name_to_index: &HashMap<String, usize>,
    ) -> Option<String> {
        let bone_dir = Self::bone_direction(
            bone,
            child.or(fallback_child),
            positions,
            bone_to_node,
            node_name_to_index,
        )?;
        let target = Self::vec3_normalize(target)?;
        let node_index = Self::node_index_for_bone(bone, bone_to_node, node_name_to_index)?;
        let local_axes = Self::local_axes(global.get(node_index).copied()?);

        let mut best = None::<(usize, f32, f32)>;
        for (axis_index, axis_world) in local_axes.into_iter().enumerate() {
            let Some(movement) = Self::vec3_normalize(Self::vec3_cross(axis_world, bone_dir))
            else {
                continue;
            };
            let score = Self::vec3_dot(movement, target);
            let (scale, score_abs) = if score >= 0.0 {
                (1.0, score)
            } else {
                (-1.0, -score)
            };
            if best.is_none_or(|(_, _, best_score)| score_abs > best_score) {
                best = Some((axis_index, scale, score_abs));
            }
        }

        let (axis, scale, _) = best?;
        Some(Self::axis_binding(axis, scale))
    }

    #[allow(clippy::too_many_arguments)]
    fn infer_bend_binding(
        bone: &str,
        child: Option<&str>,
        parent: Option<&str>,
        mesh: &motionloom::GlbMeshData,
        global: &[[f32; 16]],
        positions: &[[f32; 3]],
        bone_to_node: &HashMap<String, String>,
        node_name_to_index: &HashMap<String, usize>,
    ) -> Option<String> {
        let bone_index = Self::node_index_for_bone(bone, bone_to_node, node_name_to_index)?;
        let bone_pos = positions.get(bone_index).copied()?;
        let child_pos = child
            .and_then(|child| Self::node_index_for_bone(child, bone_to_node, node_name_to_index))
            .and_then(|index| positions.get(index).copied());
        let parent_pos = parent
            .and_then(|parent| Self::node_index_for_bone(parent, bone_to_node, node_name_to_index))
            .and_then(|index| positions.get(index).copied());
        let target = parent_pos
            .and_then(|parent_pos| Self::vec3_normalize(Self::vec3_sub(parent_pos, bone_pos)))
            .or_else(|| {
                child_pos
                    .and_then(|child_pos| Self::vec3_normalize(Self::vec3_sub(child_pos, bone_pos)))
            })?;

        Self::infer_movement_binding(
            bone,
            child,
            parent,
            target,
            mesh,
            global,
            positions,
            bone_to_node,
            node_name_to_index,
        )
    }

    fn infer_twist_binding(
        bone: &str,
        target: &[f32; 3],
        _mesh: &motionloom::GlbMeshData,
        global: &[[f32; 16]],
        bone_to_node: &HashMap<String, String>,
        node_name_to_index: &HashMap<String, usize>,
    ) -> Option<String> {
        let target = Self::vec3_normalize(*target)?;
        let node_index = Self::node_index_for_bone(bone, bone_to_node, node_name_to_index)?;
        let local_axes = Self::local_axes(global.get(node_index).copied()?);

        let mut best = None::<(usize, f32, f32)>;
        for (axis_index, axis_world) in local_axes.into_iter().enumerate() {
            let score = Self::vec3_dot(axis_world, target);
            let (scale, score_abs) = if score >= 0.0 {
                (1.0, score)
            } else {
                (-1.0, -score)
            };
            if best.is_none_or(|(_, _, best_score)| score_abs > best_score) {
                best = Some((axis_index, scale, score_abs));
            }
        }

        let (axis, scale, _) = best?;
        Some(Self::axis_binding(axis, scale))
    }

    fn bone_direction(
        bone: &str,
        other: Option<&str>,
        positions: &[[f32; 3]],
        bone_to_node: &HashMap<String, String>,
        node_name_to_index: &HashMap<String, usize>,
    ) -> Option<[f32; 3]> {
        let bone_index = Self::node_index_for_bone(bone, bone_to_node, node_name_to_index)?;
        let other_index = Self::node_index_for_bone(other?, bone_to_node, node_name_to_index)?;
        let bone_pos = positions.get(bone_index).copied()?;
        let other_pos = positions.get(other_index).copied()?;
        Self::vec3_normalize(Self::vec3_sub(other_pos, bone_pos))
    }

    fn local_axes(matrix: [f32; 16]) -> [[f32; 3]; 3] {
        [
            Self::vec3_normalize([matrix[0], matrix[1], matrix[2]]).unwrap_or([1.0, 0.0, 0.0]),
            Self::vec3_normalize([matrix[4], matrix[5], matrix[6]]).unwrap_or([0.0, 1.0, 0.0]),
            Self::vec3_normalize([matrix[8], matrix[9], matrix[10]]).unwrap_or([0.0, 0.0, 1.0]),
        ]
    }

    fn axis_binding(axis: usize, scale: f32) -> String {
        let axis = match axis {
            0 => "rotationX",
            1 => "rotationY",
            _ => "rotationZ",
        };
        let sign = if scale >= 0.0 { 1 } else { -1 };
        format!("{axis}:{sign}")
    }

    fn glb_global_node_matrices(nodes: &[motionloom::GlbNodeData]) -> Vec<[f32; 16]> {
        let local = nodes
            .iter()
            .map(|node| {
                node.matrix.unwrap_or_else(|| {
                    Self::mat4_from_trs(node.translation, node.rotation, node.scale)
                })
            })
            .collect::<Vec<_>>();
        let mut global = vec![None; nodes.len()];
        for index in 0..nodes.len() {
            Self::compute_glb_global_node_matrix(index, nodes, &local, &mut global);
        }
        global
            .into_iter()
            .map(|matrix| matrix.unwrap_or_else(Self::identity_mat4))
            .collect()
    }

    fn compute_glb_global_node_matrix(
        index: usize,
        nodes: &[motionloom::GlbNodeData],
        local: &[[f32; 16]],
        global: &mut [Option<[f32; 16]>],
    ) -> [f32; 16] {
        if let Some(matrix) = global.get(index).copied().flatten() {
            return matrix;
        }
        let local_matrix = local
            .get(index)
            .copied()
            .unwrap_or_else(Self::identity_mat4);
        let matrix = nodes
            .get(index)
            .and_then(|node| node.parent)
            .map(|parent| {
                Self::mat4_mul(
                    Self::compute_glb_global_node_matrix(parent, nodes, local, global),
                    local_matrix,
                )
            })
            .unwrap_or(local_matrix);
        if let Some(slot) = global.get_mut(index) {
            *slot = Some(matrix);
        }
        matrix
    }

    fn mat4_from_trs(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> [f32; 16] {
        Self::mat4_mul(
            Self::mat4_mul(
                Self::mat4_translation(translation),
                Self::mat4_from_quat(rotation),
            ),
            Self::mat4_scale(scale),
        )
    }

    fn mat4_translation(translation: [f32; 3]) -> [f32; 16] {
        [
            1.0,
            0.0,
            0.0,
            0.0, //
            0.0,
            1.0,
            0.0,
            0.0, //
            0.0,
            0.0,
            1.0,
            0.0, //
            translation[0],
            translation[1],
            translation[2],
            1.0,
        ]
    }

    fn mat4_scale(scale: [f32; 3]) -> [f32; 16] {
        [
            scale[0], 0.0, 0.0, 0.0, //
            0.0, scale[1], 0.0, 0.0, //
            0.0, 0.0, scale[2], 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn mat4_from_quat(quat: [f32; 4]) -> [f32; 16] {
        let [x, y, z, w] = quat;
        let len = (x * x + y * y + z * z + w * w).sqrt();
        if len <= f32::EPSILON {
            return Self::identity_mat4();
        }
        let x = x / len;
        let y = y / len;
        let z = z / len;
        let w = w / len;
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;
        [
            1.0 - (yy + zz),
            xy + wz,
            xz - wy,
            0.0,
            xy - wz,
            1.0 - (xx + zz),
            yz + wx,
            0.0,
            xz + wy,
            yz - wx,
            1.0 - (xx + yy),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }

    fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum();
            }
        }
        out
    }

    fn mat4_transform_point(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
        [
            matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
            matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
            matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
        ]
    }

    fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn vec3_scale(v: [f32; 3], scale: f32) -> [f32; 3] {
        [v[0] * scale, v[1] * scale, v[2] * scale]
    }

    fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    fn rotate_vec3_around_axis(v: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
        let axis = Self::vec3_normalize(axis).unwrap_or([0.0, 1.0, 0.0]);
        let (sin, cos) = angle.sin_cos();
        let term1 = Self::vec3_scale(v, cos);
        let term2 = Self::vec3_scale(Self::vec3_cross(axis, v), sin);
        let term3 = Self::vec3_scale(axis, Self::vec3_dot(axis, v) * (1.0 - cos));
        Self::vec3_add(Self::vec3_add(term1, term2), term3)
    }

    fn vec3_normalize(v: [f32; 3]) -> Option<[f32; 3]> {
        let len = Self::vec3_dot(v, v).sqrt();
        if len <= 0.000001 {
            return None;
        }
        Some([v[0] / len, v[1] / len, v[2] / len])
    }

    fn sanitize_xml_ident(value: &str) -> String {
        let mut out = value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .collect::<String>();
        if out.is_empty() {
            out = "model".to_string();
        }
        if out
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_digit() || ch == '-')
        {
            out.insert(0, '_');
        }
        out
    }

    fn guess_humanoid_retarget_maps(joint_names: &[String]) -> Vec<(String, String)> {
        let candidates: &[(&str, &[&str])] = &[
            ("hips", &["hips", "pelvis"]),
            ("spine", &["spine"]),
            ("chest", &["chest", "upperchest", "thorax"]),
            ("neck", &["neck"]),
            ("head", &["head"]),
            ("shoulder_l", &["leftshoulder", "lshoulder", "shoulderl"]),
            (
                "upper_arm_l",
                &["leftarm", "leftupperarm", "lupperarm", "upperarml"],
            ),
            (
                "forearm_l",
                &["leftelbow", "leftforearm", "llowerarm", "forearml"],
            ),
            ("hand_l", &["leftwrist", "lefthand", "handl"]),
            ("shoulder_r", &["rightshoulder", "rshoulder", "shoulderr"]),
            (
                "upper_arm_r",
                &["rightarm", "rightupperarm", "rupperarm", "upperarmr"],
            ),
            (
                "forearm_r",
                &["rightelbow", "rightforearm", "rlowerarm", "forearmr"],
            ),
            ("hand_r", &["rightwrist", "righthand", "handr"]),
            (
                "upper_leg_l",
                &["leftleg", "leftupleg", "leftthigh", "thighl"],
            ),
            (
                "lower_leg_l",
                &["leftknee", "leftlowerleg", "leftshin", "calfl"],
            ),
            ("foot_l", &["leftankle", "leftfoot", "footl"]),
            ("toe_l", &["lefttoe", "toel"]),
            (
                "upper_leg_r",
                &["rightleg", "rightupleg", "rightthigh", "thighr"],
            ),
            (
                "lower_leg_r",
                &["rightknee", "rightlowerleg", "rightshin", "calfr"],
            ),
            ("foot_r", &["rightankle", "rightfoot", "footr"]),
            ("toe_r", &["righttoe", "toer"]),
        ];
        let normalized = joint_names
            .iter()
            .map(|name| (name, Self::normalize_joint_name(name)))
            .collect::<Vec<_>>();
        let mut used = HashSet::<String>::new();
        let mut out = Vec::<(String, String)>::new();
        for (bone, patterns) in candidates {
            let Some((name, _)) = normalized
                .iter()
                .filter(|(name, normalized)| {
                    !used.contains(*name)
                        && !Self::is_likely_accessory_joint(normalized)
                        && Self::joint_name_matches_humanoid_bone(normalized, patterns)
                })
                .min_by_key(|(_, normalized)| Self::joint_match_score(normalized, patterns))
            else {
                continue;
            };
            used.insert((*name).clone());
            out.push(((*name).clone(), (*bone).to_string()));
        }
        out
    }

    fn joint_name_matches_humanoid_bone(normalized: &str, patterns: &[&str]) -> bool {
        patterns.iter().any(|pattern| normalized.contains(pattern))
    }

    fn joint_match_score(normalized: &str, patterns: &[&str]) -> usize {
        patterns
            .iter()
            .filter(|pattern| normalized.contains(**pattern))
            .map(|pattern| {
                if normalized == *pattern {
                    0
                } else if normalized.ends_with(*pattern) || normalized.starts_with(*pattern) {
                    1
                } else {
                    2
                }
            })
            .min()
            .unwrap_or(usize::MAX)
    }

    fn is_likely_accessory_joint(normalized: &str) -> bool {
        [
            "ribon", "ribbon", "twist", "hair", "skirt", "wing", "tail", "eye", "front", "kata",
        ]
        .iter()
        .any(|needle| normalized.contains(needle))
    }

    fn normalize_joint_name(name: &str) -> String {
        name.chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    }

    fn glb_joint_tree_lines(
        nodes: &[motionloom::GlbNodeData],
        joint_node_indices: &HashSet<usize>,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        for node in nodes {
            if node.parent.is_none()
                && Self::joint_subtree_contains_joint(node.index, nodes, joint_node_indices)
            {
                Self::push_glb_joint_tree_line(
                    node.index,
                    0,
                    nodes,
                    joint_node_indices,
                    &mut lines,
                );
            }
        }
        if lines.is_empty() {
            return vec!["No skin joint hierarchy found.".to_string()];
        }
        lines.truncate(260);
        lines
    }

    fn joint_subtree_contains_joint(
        index: usize,
        nodes: &[motionloom::GlbNodeData],
        joint_node_indices: &HashSet<usize>,
    ) -> bool {
        joint_node_indices.contains(&index)
            || nodes.get(index).is_some_and(|node| {
                node.children.iter().any(|child| {
                    Self::joint_subtree_contains_joint(*child, nodes, joint_node_indices)
                })
            })
    }

    fn push_glb_joint_tree_line(
        index: usize,
        depth: usize,
        nodes: &[motionloom::GlbNodeData],
        joint_node_indices: &HashSet<usize>,
        lines: &mut Vec<String>,
    ) {
        if lines.len() >= 260 {
            return;
        }
        let Some(node) = nodes.get(index) else {
            return;
        };
        let is_joint = joint_node_indices.contains(&index);
        if is_joint {
            let prefix = "  ".repeat(depth);
            let name = node.name.as_deref().unwrap_or("(unnamed)");
            lines.push(format!("{prefix}- {name} [node {}]", node.index));
        }
        let child_depth = if is_joint { depth + 1 } else { depth };
        for child in &node.children {
            if Self::joint_subtree_contains_joint(*child, nodes, joint_node_indices) {
                Self::push_glb_joint_tree_line(
                    *child,
                    child_depth,
                    nodes,
                    joint_node_indices,
                    lines,
                );
            }
        }
    }

    fn build_retarget_draft(actors: &[GlbSkeletonActorReport]) -> String {
        let mut out = String::new();
        for actor in actors {
            if actor.guessed_maps.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push('\n');
                out.push('\n');
            }
            out.push_str(&format!(
                "<Retarget id=\"{}_humanoid_map\" actor=\"{}\" preset=\"humanoid_v1\">\n",
                Self::xml_attr_escape(&actor.actor_id),
                Self::xml_attr_escape(&actor.actor_id)
            ));
            for (from, to) in &actor.guessed_maps {
                out.push_str(&format!(
                    "  <Map from=\"{}\" to=\"{}\" />\n",
                    Self::xml_attr_escape(from),
                    Self::xml_attr_escape(to)
                ));
            }
            out.push_str("</Retarget>");
        }
        out
    }

    fn xml_attr_escape(value: &str) -> String {
        value
            .replace('&', "&amp;")
            .replace('"', "&quot;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }

    fn identity_mat4() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn render_glb_inspector_modal_overlay(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let report = self.glb_inspector_report.clone();
        let draft = report
            .as_ref()
            .map(|report| report.retarget_draft.clone())
            .unwrap_or_default();
        let profile_draft = report
            .as_ref()
            .map(|report| report.model_profile_draft.clone())
            .unwrap_or_default();
        let calibrated_profile_draft = report
            .as_ref()
            .map(|report| report.calibrated_model_profile_draft.clone())
            .unwrap_or_default();

        div()
            .absolute()
            .top_0()
            .bottom_0()
            .left_0()
            .right_0()
            .bg(gpui_component::black().opacity(0.62))
            .flex()
            .items_center()
            .justify_center()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.glb_inspector_modal_open = false;
                    this.status_line = "GLB inspector closed.".to_string();
                    cx.notify();
                }),
            )
            .child(
                div()
                    .w(px(920.0))
                    .max_h(px(760.0))
                    .rounded_lg()
                    .border_1()
                    .border_color(white().opacity(0.16))
                    .bg(rgb(0x111827))
                    .shadow_2xl()
                    .p_4()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_, _, _, cx| {
                            cx.stop_propagation();
                        }),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_1()
                                    .child(
                                        div()
                                            .text_sm()
                                            .text_color(white().opacity(0.96))
                                            .child("Inspect GLB Skeleton"),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(white().opacity(0.58))
                                            .child(report.as_ref().map(|r| r.summary.clone()).unwrap_or_else(|| "No report loaded.".to_string())),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        Self::control_button("Copy Calibrated Profile")
                                            .on_mouse_down(
                                                MouseButton::Left,
                                                cx.listener(move |this, _, _, cx| {
                                                    if calibrated_profile_draft.trim().is_empty() {
                                                        this.status_line =
                                                            "No calibrated ModelProfile draft available to copy."
                                                                .to_string();
                                                    } else {
                                                        cx.write_to_clipboard(
                                                            ClipboardItem::new_string(
                                                                calibrated_profile_draft.clone(),
                                                            ),
                                                        );
                                                        this.status_line =
                                                            "Copied calibrated GLB ModelProfile draft."
                                                                .to_string();
                                                    }
                                                    cx.notify();
                                                }),
                                            ),
                                    )
                                    .child(Self::control_button("Copy ModelProfile").on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, _, cx| {
                                            if profile_draft.trim().is_empty() {
                                                this.status_line =
                                                    "No ModelProfile draft available to copy."
                                                        .to_string();
                                            } else {
                                                cx.write_to_clipboard(
                                                    ClipboardItem::new_string(
                                                        profile_draft.clone(),
                                                    ),
                                                );
                                                this.status_line =
                                                    "Copied full GLB ModelProfile draft."
                                                        .to_string();
                                            }
                                            cx.notify();
                                        }),
                                    ))
                                    .child(
                                        Self::control_button("Copy Retarget Draft").on_mouse_down(
                                            MouseButton::Left,
                                            cx.listener(move |this, _, _, cx| {
                                                if draft.trim().is_empty() {
                                                    this.status_line =
                                                        "No retarget draft available to copy."
                                                            .to_string();
                                                } else {
                                                    cx.write_to_clipboard(
                                                        ClipboardItem::new_string(draft.clone()),
                                                    );
                                                    this.status_line =
                                                        "Copied GLB retarget draft.".to_string();
                                                }
                                                cx.notify();
                                            }),
                                        ),
                                    )
                                    .child(
                                        Self::control_button("Close").on_mouse_down(
                                            MouseButton::Left,
                                            cx.listener(|this, _, _, cx| {
                                                this.glb_inspector_modal_open = false;
                                                this.status_line =
                                                    "GLB inspector closed.".to_string();
                                                cx.notify();
                                            }),
                                        ),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.66))
                            .child("This reads the current Scene DSL once, lists GLB skin joints, generates a humanoid_v1 Retarget, simulates rotationX/Y/Z endpoint movement for Auto Calibration Preview, and builds copyable kind=\"3d\" ModelProfile drafts. It does not affect frame preview/render speed."),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_h(px(280.0))
                            .max_h(px(610.0))
                            .overflow_y_scrollbar()
                            .rounded_md()
                            .border_1()
                            .border_color(white().opacity(0.10))
                            .bg(white().opacity(0.025))
                            .p_3()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .children(report.map(|report| {
                                report
                                    .actors
                                    .into_iter()
                                    .map(|actor| Self::render_glb_actor_report(actor, cx))
                                    .collect::<Vec<_>>()
                            }).unwrap_or_else(|| {
                                vec![
                                    div()
                                        .text_sm()
                                        .text_color(white().opacity(0.7))
                                        .child("No GLB actor report available.")
                                ]
                            })),
                    ),
            )
    }

    fn render_glb_actor_report(actor: GlbSkeletonActorReport, cx: &mut Context<Self>) -> gpui::Div {
        let mapped = if actor.mapped_bones.is_empty() {
            "none".to_string()
        } else {
            actor.mapped_bones.join(", ")
        };
        let missing = if actor.missing_humanoid_bones.is_empty() {
            "none".to_string()
        } else {
            actor.missing_humanoid_bones.join(", ")
        };
        let actor_id_for_guesses = actor.actor_id.clone();
        let actor_id_for_tree = actor.actor_id.clone();
        let actor_id_for_profile = actor.actor_id.clone();
        let actor_id_for_calibration = actor.actor_id.clone();
        let actor_id_for_calibrated_profile = actor.actor_id.clone();
        let guessed_text = if actor.guessed_maps.is_empty() {
            "No humanoid_v1 map guesses.".to_string()
        } else {
            actor
                .guessed_maps
                .iter()
                .map(|(from, to)| format!("{from} -> {to}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let tree_text = if actor.joint_tree_lines.is_empty() {
            "No skin joint hierarchy found.".to_string()
        } else {
            actor.joint_tree_lines.join("\n")
        };
        let profile_text = if actor.model_profile_draft.trim().is_empty() {
            "No ModelProfile draft generated.".to_string()
        } else {
            actor.model_profile_draft.clone()
        };
        let calibrated_profile_text = if actor.calibrated_model_profile_draft.trim().is_empty() {
            "No calibrated ModelProfile draft generated.".to_string()
        } else {
            actor.calibrated_model_profile_draft.clone()
        };
        let calibration_preview_text = if actor.calibration_preview_lines.is_empty() {
            "No auto calibration preview generated.".to_string()
        } else {
            actor.calibration_preview_lines.join("\n")
        };
        let guessed = if actor.guessed_maps.is_empty() {
            vec![
                div()
                    .font_family("Mono")
                    .text_xs()
                    .text_color(white().opacity(0.52))
                    .child("No humanoid_v1 map guesses."),
            ]
        } else {
            actor
                .guessed_maps
                .iter()
                .map(|(from, to)| {
                    div()
                        .font_family("Mono")
                        .text_xs()
                        .text_color(white().opacity(0.72))
                        .child(format!("{from} -> {to}"))
                })
                .collect::<Vec<_>>()
        };
        let tree_lines = actor
            .joint_tree_lines
            .iter()
            .map(|line| {
                div()
                    .font_family("Mono")
                    .text_xs()
                    .text_color(white().opacity(0.70))
                    .child(line.clone())
            })
            .collect::<Vec<_>>();
        let profile_lines = profile_text
            .lines()
            .map(|line| {
                div()
                    .font_family("Mono")
                    .text_xs()
                    .text_color(white().opacity(0.70))
                    .child(line.to_string())
            })
            .collect::<Vec<_>>();
        let calibration_preview_lines = calibration_preview_text
            .lines()
            .map(|line| {
                div()
                    .font_family("Mono")
                    .text_xs()
                    .text_color(white().opacity(0.70))
                    .child(line.to_string())
            })
            .collect::<Vec<_>>();
        let calibrated_profile_lines = calibrated_profile_text
            .lines()
            .map(|line| {
                div()
                    .font_family("Mono")
                    .text_xs()
                    .text_color(white().opacity(0.70))
                    .child(line.to_string())
            })
            .collect::<Vec<_>>();
        let load_status = actor.error.clone();
        let guessed_text_for_copy = guessed_text.clone();
        let tree_text_for_copy = tree_text.clone();
        let profile_text_for_copy = profile_text.clone();
        let calibration_preview_text_for_copy = calibration_preview_text.clone();
        let calibrated_profile_text_for_copy = calibrated_profile_text.clone();

        div()
            .w_full()
            .rounded_md()
            .border_1()
            .border_color(white().opacity(0.10))
            .bg(white().opacity(0.035))
            .p_3()
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .text_sm()
                            .text_color(white().opacity(0.92))
                            .child(format!("Actor · {}", actor.actor_id)),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(if load_status.is_some() {
                                rgba(0xff8a7acc)
                            } else {
                                rgba(0x8de5bacc)
                            })
                            .child(if load_status.is_some() {
                                "load failed"
                            } else {
                                "loaded"
                            }),
                    ),
            )
            .child(
                div()
                    .font_family("Mono")
                    .text_xs()
                    .text_color(white().opacity(0.58))
                    .child(format!("model={} -> {}", actor.model, actor.resolved_path)),
            )
            .when_some(load_status, |el, error| {
                el.child(div().text_xs().text_color(rgba(0xff8a7acc)).child(error))
            })
            .child(div().flex().flex_wrap().gap_2().children([
                Self::glb_stat_chip(format!("nodes {}", actor.node_count)),
                Self::glb_stat_chip(format!("joints {}", actor.joint_count)),
                Self::glb_stat_chip(format!("vertices {}", actor.vertex_count)),
                Self::glb_stat_chip(format!("triangles {}", actor.triangle_count)),
                Self::glb_stat_chip(format!("weighted {}", actor.weighted_vertex_count)),
                Self::glb_stat_chip(format!(
                    "inverseBind {}",
                    if actor.has_inverse_bind_matrices {
                        "yes"
                    } else {
                        "unknown"
                    }
                )),
            ]))
            .child(
                div()
                    .text_xs()
                    .text_color(white().opacity(0.66))
                    .child(format!("Current mapped humanoid bones: {mapped}")),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(white().opacity(0.54))
                    .child(format!(
                        "Missing humanoid_v1 bones in current DSL: {missing}"
                    )),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.76))
                            .child("Retarget guesses"),
                    )
                    .child(Self::control_button("Copy").on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                guessed_text_for_copy.clone(),
                            ));
                            this.status_line =
                                format!("Copied retarget guesses for {actor_id_for_guesses}.");
                            cx.notify();
                        }),
                    )),
            )
            .child(
                div()
                    .rounded_sm()
                    .bg(rgb(0x090d14))
                    .border_1()
                    .border_color(white().opacity(0.08))
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .children(guessed),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.76))
                            .child("Joint hierarchy"),
                    )
                    .child(Self::control_button("Copy").on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                tree_text_for_copy.clone(),
                            ));
                            this.status_line =
                                format!("Copied joint hierarchy for {actor_id_for_tree}.");
                            cx.notify();
                        }),
                    )),
            )
            .child(
                div()
                    .rounded_sm()
                    .bg(rgb(0x090d14))
                    .border_1()
                    .border_color(white().opacity(0.08))
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .children(tree_lines),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.76))
                            .child("Auto Calibration Preview"),
                    )
                    .child(Self::control_button("Copy").on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                calibration_preview_text_for_copy.clone(),
                            ));
                            this.status_line =
                                format!("Copied auto calibration preview for {actor_id_for_calibration}.");
                            cx.notify();
                        }),
                    )),
            )
            .child(
                div()
                    .rounded_sm()
                    .bg(rgb(0x090d14))
                    .border_1()
                    .border_color(white().opacity(0.08))
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .children(calibration_preview_lines),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.76))
                            .child("Generated ModelProfile"),
                    )
                    .child(Self::control_button("Copy").on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                profile_text_for_copy.clone(),
                            ));
                            this.status_line =
                                format!("Copied ModelProfile draft for {actor_id_for_profile}.");
                            cx.notify();
                        }),
                    )),
            )
            .child(
                div()
                    .rounded_sm()
                    .bg(rgb(0x090d14))
                    .border_1()
                    .border_color(white().opacity(0.08))
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .children(profile_lines),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.76))
                            .child("Calibrated ModelProfile"),
                    )
                    .child(Self::control_button("Copy").on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                calibrated_profile_text_for_copy.clone(),
                            ));
                            this.status_line = format!(
                                "Copied calibrated ModelProfile draft for {actor_id_for_calibrated_profile}."
                            );
                            cx.notify();
                        }),
                    )),
            )
            .child(
                div()
                    .rounded_sm()
                    .bg(rgb(0x090d14))
                    .border_1()
                    .border_color(white().opacity(0.08))
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .children(calibrated_profile_lines),
            )
    }

    fn glb_stat_chip(label: String) -> gpui::Div {
        div()
            .h(px(24.0))
            .px_2()
            .rounded_md()
            .border_1()
            .border_color(white().opacity(0.10))
            .bg(white().opacity(0.055))
            .text_xs()
            .text_color(white().opacity(0.72))
            .flex()
            .items_center()
            .child(label)
    }

    fn open_import_modal(&mut self) {
        self.template_modal_open = false;
        self.asset_modal_open = false;
        self.scene_template_modal_open = false;
        self.scene_render_modal_open = false;
        self.glb_inspector_modal_open = false;
        self.import_modal_open = true;
        self.status_line = "Source/import panel opened.".to_string();
    }

    fn open_asset_modal(&mut self) {
        self.template_modal_open = false;
        self.import_modal_open = false;
        self.scene_template_modal_open = false;
        self.scene_render_modal_open = false;
        self.glb_inspector_modal_open = false;
        self.asset_modal_open = true;
        self.asset_context_menu = None;
        self.status_line = "VFX Assets browser opened.".to_string();
    }

    fn render_vfx_asset_modal_overlay(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let viewport_w = window.viewport_size().width / px(1.0);
        let viewport_h = window.viewport_size().height / px(1.0);
        let modal_w = (viewport_w - 96.0).clamp(760.0, 1180.0);
        let modal_h = (viewport_h - 88.0).clamp(520.0, 820.0);
        let folder_label = self
            .asset_folder
            .as_ref()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_else(|| "No folder opened.".to_string());
        let selected_asset = self
            .asset_selected_idx
            .and_then(|idx| self.asset_items.get(idx))
            .cloned();

        let open_folder_button = Self::control_button("Open Folder").on_mouse_down(
            MouseButton::Left,
            cx.listener(|_this, _, win, cx| {
                let rx = cx.prompt_for_paths(PathPromptOptions {
                    files: false,
                    directories: true,
                    multiple: false,
                    prompt: Some("Open VFX asset folder".into()),
                });
                cx.spawn_in(win, async move |view, window| {
                    let Ok(result) = rx.await else {
                        return;
                    };
                    let Some(paths) = result.ok().flatten() else {
                        return;
                    };
                    let Some(folder) = paths.into_iter().next() else {
                        return;
                    };
                    let _ = view.update_in(window, |this, _window, cx| {
                        this.load_vfx_asset_folder(folder, cx);
                        cx.notify();
                    });
                })
                .detach();
            }),
        );
        let refresh_button = Self::control_button("Refresh").on_mouse_down(
            MouseButton::Left,
            cx.listener(|this, _, _, cx| {
                this.refresh_vfx_asset_folder(cx);
                cx.notify();
            }),
        );

        let context_menu = self.asset_context_menu.clone().map(|menu| {
            let menu_w = 220.0;
            let menu_h = 40.0;
            let menu_x = menu.x.clamp(8.0, (viewport_w - menu_w - 8.0).max(8.0));
            let menu_y = menu.y.clamp(8.0, (viewport_h - menu_h - 8.0).max(8.0));
            let copy_path = menu.path.clone();
            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left_0()
                .right_0()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _, _, cx| {
                        this.asset_context_menu = None;
                        cx.notify();
                    }),
                )
                .on_mouse_down(
                    MouseButton::Right,
                    cx.listener(|this, _, _, cx| {
                        this.asset_context_menu = None;
                        cx.notify();
                    }),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(menu_x))
                        .top(px(menu_y))
                        .w(px(menu_w))
                        .rounded_md()
                        .bg(rgb(0x1f1f23))
                        .border_1()
                        .border_color(white().opacity(0.16))
                        .p_1()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|_, _, _, cx| {
                                cx.stop_propagation();
                            }),
                        )
                        .child(
                            div()
                                .h(px(28.0))
                                .rounded_sm()
                                .px_2()
                                .flex()
                                .items_center()
                                .text_sm()
                                .text_color(white().opacity(0.92))
                                .bg(white().opacity(0.04))
                                .hover(|style| style.bg(white().opacity(0.10)))
                                .cursor_pointer()
                                .child("Copy Path")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        cx.write_to_clipboard(ClipboardItem::new_string(
                                            copy_path.clone(),
                                        ));
                                        this.asset_context_menu = None;
                                        this.status_line =
                                            format!("Copied VFX asset path: {}", copy_path);
                                        cx.notify();
                                    }),
                                ),
                        ),
                )
                .into_any_element()
        });

        let asset_rows = self.asset_items.iter().enumerate().map(|(idx, item)| {
            let active = self.asset_selected_idx == Some(idx);
            let idx_for_select = idx;
            let item_path_for_menu = item.path.clone();
            let duration_label = if item.duration > Duration::ZERO {
                format!("{:.2}s", item.duration.as_secs_f32())
            } else {
                "Still".to_string()
            };
            let thumb = item.preview.clone();
            let mut row = div()
                .rounded_md()
                .border_1()
                .border_color(white().opacity(if active { 0.35 } else { 0.12 }))
                .bg(if active { rgb(0x1f2937) } else { rgb(0x111827) })
                .p_2()
                .flex()
                .items_center()
                .gap_2()
                .cursor_pointer()
                .hover(|s| s.bg(white().opacity(0.09)))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.asset_selected_idx = Some(idx_for_select);
                        this.asset_context_menu = None;
                        cx.notify();
                    }),
                )
                .on_mouse_down(
                    MouseButton::Right,
                    cx.listener(move |this, evt: &MouseDownEvent, _, cx| {
                        cx.stop_propagation();
                        this.asset_selected_idx = Some(idx_for_select);
                        this.asset_context_menu = Some(VfxAssetContextMenu {
                            path: item_path_for_menu.clone(),
                            x: evt.position.x / px(1.0),
                            y: evt.position.y / px(1.0),
                        });
                        cx.notify();
                    }),
                );
            row = row.child(
                div()
                    .w(px(72.0))
                    .h(px(44.0))
                    .flex_shrink_0()
                    .rounded_sm()
                    .border_1()
                    .border_color(white().opacity(0.10))
                    .bg(rgb(0x05070c))
                    .overflow_hidden()
                    .when_some(thumb, |el, preview| {
                        el.child(FitPreviewImageElement::from_preview(preview))
                    }),
            );
            row.child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_sm()
                            .text_color(white().opacity(0.92))
                            .truncate()
                            .child(item.name.clone()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.58))
                            .truncate()
                            .child(format!("{} · {}", item.kind.label(), duration_label)),
                    ),
            )
        });

        let preview_panel = if let Some(asset) = selected_asset {
            let preview = asset.preview.clone();
            div()
                .w(px(330.0))
                .flex_shrink_0()
                .min_h_0()
                .rounded_md()
                .border_1()
                .border_color(white().opacity(0.12))
                .bg(rgb(0x101722))
                .p_3()
                .flex()
                .flex_col()
                .gap_3()
                .child(
                    div()
                        .h(px(210.0))
                        .w_full()
                        .rounded_md()
                        .border_1()
                        .border_color(white().opacity(0.12))
                        .bg(rgb(0x05070c))
                        .overflow_hidden()
                        .when_some(preview, |el, preview| {
                            el.child(FitPreviewImageElement::from_preview(preview))
                        }),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(white().opacity(0.94))
                        .truncate()
                        .child(asset.name),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.62))
                        .child(format!(
                            "{} · {}",
                            asset.kind.label(),
                            if asset.duration > Duration::ZERO {
                                format!("{:.2}s", asset.duration.as_secs_f32())
                            } else {
                                "Still".to_string()
                            }
                        )),
                )
                .child(
                    div().w_full().min_w_0().overflow_x_scrollbar().child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.58))
                            .whitespace_nowrap()
                            .child(asset.path),
                    ),
                )
                .when_some(asset.error, |el, error| {
                    el.child(div().text_xs().text_color(rgba(0xff8a80cc)).child(error))
                })
                .into_any_element()
        } else {
            div()
                .w(px(330.0))
                .flex_shrink_0()
                .min_h_0()
                .rounded_md()
                .border_1()
                .border_color(white().opacity(0.12))
                .bg(rgb(0x101722))
                .p_3()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(white().opacity(0.58))
                .child("Open a folder to browse VFX assets.")
                .into_any_element()
        };

        div()
            .absolute()
            .top_0()
            .bottom_0()
            .left_0()
            .right_0()
            .bg(gpui_component::black().opacity(0.55))
            .flex()
            .items_center()
            .justify_center()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.asset_modal_open = false;
                    this.asset_context_menu = None;
                    cx.notify();
                }),
            )
            .child(
                div()
                    .w(px(modal_w))
                    .h(px(modal_h))
                    .rounded_md()
                    .bg(rgb(0x1a202c))
                    .border_1()
                    .border_color(white().opacity(0.16))
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.asset_context_menu = None;
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    )
                    .on_mouse_down(
                        MouseButton::Right,
                        cx.listener(|_, _, _, cx| {
                            cx.stop_propagation();
                        }),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(white().opacity(0.9))
                                    .child("VFX ASSETS"),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(open_folder_button)
                                    .child(refresh_button)
                                    .child(Self::control_button("Close").on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.asset_modal_open = false;
                                            this.asset_context_menu = None;
                                            cx.notify();
                                        }),
                                    )),
                            ),
                    )
                    .child(
                        div().w_full().min_w_0().overflow_x_scrollbar().child(
                            div()
                                .text_xs()
                                .text_color(white().opacity(0.58))
                                .whitespace_nowrap()
                                .child(folder_label),
                        ),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_h_0()
                            .overflow_hidden()
                            .flex()
                            .gap_3()
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .min_h_0()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(white().opacity(0.12))
                                    .bg(rgb(0x111827))
                                    .p_2()
                                    .overflow_y_scrollbar()
                                    .flex()
                                    .flex_col()
                                    .gap_2()
                                    .when(self.asset_items.is_empty(), |el| {
                                        el.child(
                                            div()
                                                .h_full()
                                                .w_full()
                                                .flex()
                                                .items_center()
                                                .justify_center()
                                                .text_sm()
                                                .text_color(white().opacity(0.55))
                                                .child("No supported image/video files in this folder."),
                                        )
                                    })
                                    .children(asset_rows),
                            )
                            .child(preview_panel),
                    ),
            )
            .when_some(context_menu, |el, menu| el.child(menu))
            .into_any_element()
    }

    fn render_import_modal_overlay(
        &mut self,
        selected_name: String,
        selected_kind_label: String,
        selected_duration_label: String,
        selected_path_label: String,
        imported_count: usize,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let import_button = Self::control_button("Import Clip").on_mouse_down(
            MouseButton::Left,
            cx.listener(move |_this, _, win, cx| {
                let rx = cx.prompt_for_paths(PathPromptOptions {
                    files: true,
                    directories: false,
                    multiple: true,
                    prompt: Some("Import clips into MotionLoom".into()),
                });
                cx.spawn_in(win, async move |view, window| {
                    let Ok(result) = rx.await else {
                        return;
                    };
                    let Some(paths) = result.ok().flatten() else {
                        return;
                    };

                    let _ = view.update_in(window, |this, _window, cx| {
                        let mut imported = 0usize;
                        for path in paths {
                            let path_str = path.to_string_lossy().to_string();
                            if !Self::is_supported_clip_path(&path_str) {
                                continue;
                            }
                            if this.clips.iter().any(|item| item.path == path_str) {
                                continue;
                            }
                            let clip = Self::build_imported_clip(&path_str);
                            this.clips.push(clip);
                            imported += 1;
                        }

                        if imported > 0 {
                            this.selected_idx = Some(this.clips.len().saturating_sub(1));
                            this.status_line =
                                format!("Imported {} clip(s) into MotionLoom Studio.", imported);
                        } else {
                            this.status_line =
                                "No new supported image/video clip was imported.".to_string();
                        }
                        cx.notify();
                    });
                })
                .detach();
            }),
        );

        div()
            .absolute()
            .top_0()
            .bottom_0()
            .left_0()
            .right_0()
            .bg(gpui_component::black().opacity(0.55))
            .flex()
            .items_center()
            .justify_center()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.import_modal_open = false;
                    cx.notify();
                }),
            )
            .child(
                div()
                    .w(px(900.0))
                    .h(px(580.0))
                    .rounded_md()
                    .bg(rgb(0x1a202c))
                    .border_1()
                    .border_color(white().opacity(0.16))
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_, _, _, cx| {
                            cx.stop_propagation();
                        }),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(white().opacity(0.9))
                                    .child("SOURCE / IMPORT PANEL"),
                            )
                            .child(Self::control_button("Close").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(|this, _, _, cx| {
                                    this.import_modal_open = false;
                                    cx.notify();
                                }),
                            )),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.65))
                            .child(
                                "Import clips for media-input graphs. Scene Live Preview can play scene graphs directly.",
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(import_button)
                            .child(
                                div()
                                    .h(px(28.0))
                                    .px_2()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(white().opacity(0.14))
                                    .bg(white().opacity(0.05))
                                    .text_xs()
                                    .text_color(white().opacity(0.78))
                                    .flex()
                                    .items_center()
                                    .child(format!("{imported_count} imported")),
                            ),
                    )
                    .child(
                        div()
                            .rounded_md()
                            .border_1()
                            .border_color(white().opacity(0.1))
                            .bg(rgb(0x0f1726))
                            .p_2()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.58))
                                    .child("Current source"),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(white().opacity(0.94))
                                    .truncate()
                                    .child(selected_name),
                            )
                            .child(div().text_xs().text_color(white().opacity(0.6)).child(
                                format!(
                                    "{} · {} · {} imported",
                                    selected_kind_label, selected_duration_label, imported_count
                                ),
                            )),
                    )
                    .child(
                        div().w_full().min_w_0().overflow_x_scrollbar().child(
                            div()
                                .text_xs()
                                .text_color(white().opacity(0.56))
                                .whitespace_nowrap()
                                .child(selected_path_label),
                        ),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_h_0()
                            .rounded_sm()
                            .border_1()
                            .border_color(white().opacity(0.12))
                            .bg(rgb(0x131722))
                            .p_2()
                            .overflow_y_scrollbar()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .children(self.clips.iter().enumerate().map(|(idx, clip)| {
                                let active = self.selected_idx == Some(idx);
                                let idx_for_select = idx;
                                let duration_label = if clip.duration > Duration::ZERO {
                                    format!("{:.2}s", clip.duration.as_secs_f32())
                                } else {
                                    "Still".to_string()
                                };
                                div()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(white().opacity(if active { 0.35 } else { 0.14 }))
                                    .bg(if active { rgb(0x1f2937) } else { rgb(0x111827) })
                                    .px_2()
                                    .py_2()
                                    .cursor_pointer()
                                    .hover(|s| s.bg(white().opacity(0.09)))
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(white().opacity(0.6))
                                            .child(clip.kind.label()),
                                    )
                                    .child(
                                        div()
                                            .text_sm()
                                            .text_color(white().opacity(0.93))
                                            .truncate()
                                            .child(clip.name.clone()),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(white().opacity(0.6))
                                            .truncate()
                                            .child(duration_label),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, _, cx| {
                                            this.selected_idx = Some(idx_for_select);
                                            cx.notify();
                                        }),
                                    )
                            })),
                    ),
            )
            .into_any_element()
    }

    // Render a single selectable template tile in the picker modal.
    fn render_template_tile(
        &self,
        kind: LayerEffectTemplateKind,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let selected = self.template_selected.contains(&kind);
        let border = if selected {
            rgba(0x4f8fffeb)
        } else {
            rgba(0xffffff3d)
        };
        let bg = if selected {
            rgba(0x253c62c7)
        } else {
            rgba(0xffffff1f)
        };
        let label = Self::template_label(kind);
        div()
            .h(px(34.0))
            .w(px(220.0))
            .px_3()
            .rounded_sm()
            .border_1()
            .border_color(border)
            .bg(bg)
            .text_sm()
            .text_color(white().opacity(0.94))
            .cursor_pointer()
            .overflow_hidden()
            .child(div().w_full().truncate().child(label))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    this.toggle_template_selection(kind);
                    cx.notify();
                }),
            )
    }

    // Render the full-screen template picker modal overlay.
    fn render_template_modal_overlay(&mut self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let add_time_label = if self.template_add_time_parameter {
            "ADD TIME PARAMETER: ON"
        } else {
            "ADD TIME PARAMETER: OFF"
        };
        let add_curve_label = if self.template_add_curve_parameter {
            "ADD CURVE PARAMETER: ON"
        } else {
            "ADD CURVE PARAMETER: OFF"
        };
        let selection_summary = self.selected_template_summary();

        div()
            .absolute()
            .top_0()
            .bottom_0()
            .left_0()
            .right_0()
            .bg(gpui_component::black().opacity(0.55))
            .flex()
            .items_center()
            .justify_center()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.template_modal_open = false;
                    cx.notify();
                }),
            )
            .child(
                div()
                    .w(px(820.0))
                    .h(px(500.0))
                    .rounded_md()
                    .bg(rgb(0x1f1f23))
                    .border_1()
                    .border_color(white().opacity(0.16))
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_3()
                    // Stop click propagation so clicking inside the modal doesn't close it
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_, _, _, cx| {
                            cx.stop_propagation();
                        }),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(white().opacity(0.9))
                            .child("MOTIONLOOM TEMPLATE PICKER"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.65))
                            .child("Select one or more templates, then press OK to generate one graph."),
                    )
                    .child(
                        // Control bar: toggle buttons + OK + Close
                        div()
                            .flex()
                            .items_center()
                            .flex_wrap()
                            .gap_2()
                            .child(
                                div()
                                    .h(px(30.0))
                                    .px_3()
                                    .rounded_sm()
                                    .border_1()
                                    .border_color(white().opacity(0.2))
                                    .bg(white().opacity(0.08))
                                    .text_xs()
                                    .text_color(white().opacity(0.9))
                                    .cursor_pointer()
                                    .child(add_time_label)
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.template_add_time_parameter =
                                                !this.template_add_time_parameter;
                                            cx.notify();
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .h(px(30.0))
                                    .px_3()
                                    .rounded_sm()
                                    .border_1()
                                    .border_color(white().opacity(0.2))
                                    .bg(rgba(0x253c62c7))
                                    .text_xs()
                                    .text_color(white().opacity(0.94))
                                    .cursor_pointer()
                                    .child("OK")
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, window, cx| {
                                            this.apply_selected_templates(window, cx);
                                            cx.notify();
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .h(px(30.0))
                                    .px_3()
                                    .rounded_sm()
                                    .border_1()
                                    .border_color(white().opacity(0.2))
                                    .bg(white().opacity(0.08))
                                    .text_xs()
                                    .text_color(white().opacity(0.9))
                                    .cursor_pointer()
                                    .child(add_curve_label)
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.template_add_curve_parameter =
                                                !this.template_add_curve_parameter;
                                            cx.notify();
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .h(px(30.0))
                                    .px_3()
                                    .rounded_sm()
                                    .border_1()
                                    .border_color(white().opacity(0.2))
                                    .bg(white().opacity(0.06))
                                    .text_xs()
                                    .text_color(white().opacity(0.82))
                                    .cursor_pointer()
                                    .child("Close")
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.template_modal_open = false;
                                            cx.notify();
                                        }),
                                    ),
                            ),
                    )
                    .child(
                        // Template grid organized by category
                        div()
                            .flex_1()
                            .min_h(px(0.0))
                            .rounded_sm()
                            .border_1()
                            .border_color(white().opacity(0.12))
                            .bg(rgb(0x17181d))
                            .p_2()
                            .overflow_y_scrollbar()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.68))
                                    .child(format!("Selection: {selection_summary}")),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.72))
                                    .child("Color Tuning"),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_wrap()
                                    .gap_2()
                                    .child(self.render_template_tile(
                                        LayerEffectTemplateKind::HslaOverlay,
                                        cx,
                                    ))
                                    .child(self.render_template_tile(
                                        LayerEffectTemplateKind::Lut,
                                        cx,
                                    )),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.72))
                                    .child("Blend & Opacity"),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_wrap()
                                    .gap_2()
                                    .child(self.render_template_tile(
                                        LayerEffectTemplateKind::Opacity,
                                        cx,
                                    )),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.72))
                                    .child("Detail & Blur"),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_wrap()
                                    .gap_2()
                                    .child(self.render_template_tile(
                                        LayerEffectTemplateKind::Sharpen,
                                        cx,
                                    ))
                                    .child(self.render_template_tile(
                                        LayerEffectTemplateKind::BlurGaussian,
                                        cx,
                                    ))
                                    .child(self.render_template_tile(
                                        LayerEffectTemplateKind::BlurGaussianHorizontal,
                                        cx,
                                    ))
                                    .child(self.render_template_tile(
                                        LayerEffectTemplateKind::BlurGaussianVertical,
                                        cx,
                                    )),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.72))
                                    .child("Transitions"),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_wrap()
                                    .gap_2()
                                    .child(self.render_template_tile(
                                        LayerEffectTemplateKind::TransitionFadeInOut,
                                        cx,
                                    )),
                            ),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.58))
                            .child(
                                "ADD TIME PARAMETER adds apply=graph + duration(5s). ADD CURVE PARAMETER injects curve(...) into template params. Selected templates are chained in the order shown above.",
                            ),
                    ),
            )
            .into_any_element()
    }
}

impl Render for MotionLoomPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.log_scene_external_render_checkpoint("enter");
        self.sync_script_from_global(window, cx);
        self.sync_render_request_from_global(window, cx);
        self.ensure_script_input(window, cx);
        self.ensure_scene_live_knob_input(window, cx);
        self.ensure_preview_frame_input(window, cx);
        self.sync_script_input_if_needed(window, cx);
        self.sync_preview_frame_to_global(cx);
        self.log_scene_external_render_checkpoint("after_input_sync");
        let scene_template_modal = if self.scene_template_modal_open {
            Some(self.render_scene_template_modal_overlay(cx))
        } else {
            None
        };
        let selected = self.current_clip().cloned();
        let imported_count = self.clips.len();
        let selected_name = selected
            .as_ref()
            .map(|clip| clip.name.clone())
            .unwrap_or_else(|| "No source clip selected".to_string());
        let selected_kind_label = selected
            .as_ref()
            .map(|clip| clip.kind.label().to_string())
            .unwrap_or_else(|| "Source".to_string());
        let selected_duration_label = selected
            .as_ref()
            .map(|clip| {
                if clip.duration > Duration::ZERO {
                    format!("{:.2}s", clip.duration.as_secs_f32())
                } else {
                    "Still".to_string()
                }
            })
            .unwrap_or_else(|| "-".to_string());
        let selected_path_label = selected
            .as_ref()
            .map(|clip| clip.path.clone())
            .unwrap_or_else(|| "Import an image or video clip to begin previewing.".to_string());
        let viewport_w = window.viewport_size().width / px(1.0);
        let viewport_h = window.viewport_size().height / px(1.0);
        if let Some(host) = self.scene_external_preview_host.clone() {
            self.drain_scene_external_preview_events(&host, window, cx);
            let policy = self.scene_external_preview_visibility_policy(window);
            self.flush_scene_external_preview_window_state(&host, policy);
        }
        self.schedule_scene_external_preview_heartbeat(window, cx);
        let content_max_w = if viewport_w >= 1800.0 {
            1560.0
        } else if viewport_w >= 1500.0 {
            1360.0
        } else if viewport_w >= 1280.0 {
            1180.0
        } else if viewport_w >= 1080.0 {
            980.0
        } else {
            860.0
        };
        let scene_live_preview_h = if viewport_w < 1080.0 {
            (viewport_h * 0.34).clamp(220.0, 360.0)
        } else {
            (viewport_h * 0.46).clamp(300.0, 560.0)
        };
        let scene_live_preview_w = content_max_w;
        let scene_live_panel_min_h = (scene_live_preview_h + 190.0).clamp(420.0, 760.0);

        // Build script editor element (expands in left column)
        let script_input_elem = if let Some(input) = self.script_input.as_ref() {
            div()
                .w_full()
                .flex_1()
                .min_h(px(0.0))
                .rounded_md()
                .border_1()
                .border_color(white().opacity(0.18))
                .bg(rgb(0x0b1020))
                .overflow_hidden()
                .child(Input::new(input).h_full().w_full())
                .into_any_element()
        } else {
            div()
                .flex_1()
                .min_h(px(0.0))
                .w_full()
                .rounded_sm()
                .bg(white().opacity(0.05))
                .into_any_element()
        };

        let mut scene_live_targets = Self::extract_scene_live_targets(&self.script_text);
        if !self.scene_live_tag_filters.is_empty() {
            scene_live_targets
                .retain(|target| self.scene_live_tag_filters.contains(target.tag.as_str()));
        }
        self.ensure_scene_live_selection(&scene_live_targets);
        let scene_live_attrs = scene_live_targets
            .iter()
            .find(|target| target.id == self.scene_live_knob_node_id)
            .map(|target| target.attrs.clone())
            .unwrap_or_else(|| vec![self.scene_live_knob_attr.clone()]);

        self.sync_scene_live_dropdowns(&scene_live_targets, &scene_live_attrs, window, cx);
        let scene_live_target_select = self.scene_live_target_select.as_ref().cloned();
        let scene_live_attr_rows = scene_live_attrs
            .iter()
            .map(|attr| {
                let attr_value = attr.clone();
                let active = self.scene_live_knob_attr.as_str() == attr.as_str();
                div()
                    .w_full()
                    .h(px(34.0))
                    .flex_shrink_0()
                    .px_3()
                    .rounded_md()
                    .bg(if active {
                        rgba(0x1f5c85aa)
                    } else {
                        rgba(0xffffff00)
                    })
                    .hover(|style| style.bg(white().opacity(0.08)))
                    .cursor_pointer()
                    .text_sm()
                    .text_color(white().opacity(if active { 0.96 } else { 0.82 }))
                    .flex()
                    .items_center()
                    .child(attr.clone())
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            this.select_scene_live_attr(attr_value.clone());
                            cx.notify();
                        }),
                    )
            })
            .collect::<Vec<_>>();

        let scene_live_tag_filter_chips = Self::scene_live_filter_tags()
            .iter()
            .map(|tag_label| {
                let tag_label = *tag_label;
                let tag_value = tag_label.to_string();
                Self::scene_live_checkbox(
                    tag_label,
                    self.scene_live_tag_filters.contains(tag_label),
                )
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        if !this.scene_live_tag_filters.remove(tag_value.as_str()) {
                            this.scene_live_tag_filters.insert(tag_value.clone());
                        }
                        this.scene_live_target_offset = 0;
                        this.status_line = if this.scene_live_tag_filters.is_empty() {
                            "Scene live tag filter: showing all target tags.".to_string()
                        } else {
                            let mut tags = this
                                .scene_live_tag_filters
                                .iter()
                                .cloned()
                                .collect::<Vec<_>>();
                            tags.sort();
                            format!("Scene live tag filter: showing {}.", tags.join(", "))
                        };
                        cx.notify();
                    }),
                )
            })
            .collect::<Vec<_>>();

        let scene_live_selector_panel = div()
            .w_full()
            .overflow_hidden()
            .rounded_md()
            .border_1()
            .border_color(white().opacity(0.1))
            .bg(white().opacity(0.025))
            .p_2()
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .w(px(48.0))
                            .text_xs()
                            .text_color(white().opacity(0.52))
                            .child("Target"),
                    )
                    .child(if let Some(select) = scene_live_target_select.as_ref() {
                        Select::new(select)
                            .placeholder("Select target")
                            .menu_width(px(360.0))
                            .w(px(300.0))
                            .into_any_element()
                    } else {
                        div()
                            .h(px(28.0))
                            .w(px(300.0))
                            .rounded_md()
                            .border_1()
                            .border_color(white().opacity(0.12))
                            .bg(white().opacity(0.04))
                            .text_xs()
                            .text_color(white().opacity(0.52))
                            .flex()
                            .items_center()
                            .px_2()
                            .child("No targets")
                            .into_any_element()
                    }),
            )
            .child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .w(px(48.0))
                            .text_xs()
                            .text_color(white().opacity(0.52))
                            .child("Attr"),
                    )
                    .child(
                        div()
                            .h(px(28.0))
                            .w(px(240.0))
                            .rounded_md()
                            .border_1()
                            .border_color(if self.scene_live_attr_menu_open {
                                rgba(0x79c7ffcc)
                            } else {
                                rgba(0xffffff24)
                            })
                            .bg(white().opacity(0.06))
                            .hover(|style| style.bg(white().opacity(0.10)))
                            .cursor_pointer()
                            .text_xs()
                            .text_color(white().opacity(0.90))
                            .flex()
                            .items_center()
                            .justify_between()
                            .px_2()
                            .child(self.scene_live_knob_attr.clone())
                            .child(if self.scene_live_attr_menu_open {
                                "▲"
                            } else {
                                "▼"
                            })
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(|this, _, _, cx| {
                                    this.scene_live_attr_menu_open =
                                        !this.scene_live_attr_menu_open;
                                    cx.notify();
                                }),
                            ),
                    ),
            )
            .when(self.scene_live_attr_menu_open, |panel| {
                panel.child(
                    div()
                        .ml(px(56.0))
                        .w(px(320.0))
                        .max_h(px(260.0))
                        .rounded_md()
                        .border_1()
                        .border_color(white().opacity(0.16))
                        .bg(rgb(0x080a0f))
                        .p_1()
                        .overflow_y_scrollbar()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .children(scene_live_attr_rows),
                )
            })
            .child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .w(px(48.0))
                            .text_xs()
                            .text_color(white().opacity(0.52))
                            .child("Tag"),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap_2()
                            .children(scene_live_tag_filter_chips),
                    ),
            );

        let scene_live_preview_label = self.scene_live_preview_label();
        let scene_render_progress = self.scene_render_progress.clone();
        let scene_render_log = self.scene_render_log.clone();
        let scene_render_log_collapsed = self.scene_render_log_collapsed;
        let scene_render_log_has_error = scene_render_log.iter().any(|line| {
            let lower = line.to_ascii_lowercase();
            lower.contains("error") || lower.contains("failed") || lower.contains("panic")
        });
        let scene_render_log_panel = if scene_render_log.is_empty() {
            None
        } else {
            Some(
                div()
                    .w_full()
                    .rounded_md()
                    .border_1()
                    .border_color(if scene_render_log_has_error {
                        rgba(0xff6655bb)
                    } else {
                        rgba(0x79c7ff55)
                    })
                    .bg(if scene_render_log_has_error {
                        rgba(0x1a080bcc)
                    } else {
                        rgba(0x05070dcc)
                    })
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(if scene_render_log_has_error {
                                        rgba(0xffaaa0ff)
                                    } else {
                                        rgba(0xbdeaffff)
                                    })
                                    .child(if scene_render_log_has_error {
                                        "Render Log"
                                    } else {
                                        "Render Log"
                                    }),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_1()
                                    .child(
                                        Self::control_button(if scene_render_log_collapsed {
                                            "+"
                                        } else {
                                            "-"
                                        })
                                        .on_mouse_down(
                                            MouseButton::Left,
                                            cx.listener(|this, _, _, cx| {
                                                this.scene_render_log_collapsed =
                                                    !this.scene_render_log_collapsed;
                                                cx.notify();
                                            }),
                                        ),
                                    )
                                    .child(Self::control_button("X").on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.scene_render_log.clear();
                                            this.scene_render_log_collapsed = false;
                                            cx.notify();
                                        }),
                                    )),
                            ),
                    )
                    .when(scene_render_log_collapsed, |el| {
                        el.child(
                            div()
                                .text_xs()
                                .text_color(white().opacity(0.56))
                                .child(format!("{} line(s) hidden", scene_render_log.len())),
                        )
                    })
                    .when(!scene_render_log_collapsed, |el| {
                        el.children(scene_render_log.into_iter().map(|line| {
                            let lower = line.to_ascii_lowercase();
                            let is_error = lower.contains("error")
                                || lower.contains("failed")
                                || lower.contains("panic");
                            div()
                                .text_xs()
                                .text_color(if is_error {
                                    rgba(0xff8a80ee)
                                } else {
                                    rgba(0xffffffb8)
                                })
                                .child(line)
                        }))
                    }),
            )
        };
        self.log_scene_external_render_checkpoint("before_scene_preview_image");
        let scene_live_preview_result = self.scene_live_preview_image(self.preview_frame, cx);
        self.log_scene_external_render_checkpoint("after_scene_preview_image");
        let scene_live_external_graph_size =
            parse_graph_script(&self.scene_live_preview_script_text())
                .ok()
                .map(|graph| graph.render_size.unwrap_or(graph.size))
                .unwrap_or((1280, 720));
        let scene_live_preview_card = if self.scene_external_preview_host.is_some() {
            let external_preview_visibility = self.scene_external_preview_visibility_policy(window);
            div()
                .w_full()
                .h(px(scene_live_preview_h))
                .flex_shrink_0()
                .rounded_lg()
                .border_1()
                .border_color(white().opacity(0.14))
                .bg(rgb(0x05070c))
                .overflow_hidden()
                .child(ExternalPreviewAnchorElement::new(
                    scene_live_external_graph_size,
                    external_preview_visibility,
                    self.scene_external_preview_desired_window.clone(),
                ))
                .into_any_element()
        } else {
            match scene_live_preview_result {
                Ok(Some(preview)) => div()
                    .w_full()
                    .h(px(scene_live_preview_h))
                    .flex_shrink_0()
                    .rounded_lg()
                    .border_1()
                    .border_color(white().opacity(0.14))
                    .bg(rgb(0x05070c))
                    .overflow_hidden()
                    .child(FitPreviewImageElement::from_preview(preview))
                    .into_any_element(),
                Ok(None) => div()
                    .w_full()
                    .h(px(scene_live_preview_h))
                    .flex_shrink_0()
                    .rounded_lg()
                    .border_1()
                    .border_color(white().opacity(0.14))
                    .bg(rgb(0x05070c))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(
                        div()
                            .text_sm()
                            .text_color(white().opacity(0.62))
                            .child("Preview rendering..."),
                    )
                    .into_any_element(),
                Err(message) => div()
                    .w_full()
                    .h(px(scene_live_preview_h))
                    .flex_shrink_0()
                    .rounded_lg()
                    .border_1()
                    .border_color(rgba(0xff6655cc))
                    .bg(rgb(0x12080a))
                    .p_3()
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(
                        div()
                            .text_sm()
                            .text_color(white().opacity(0.78))
                            .child(message),
                    )
                    .into_any_element(),
            }
        };
        self.log_scene_external_render_checkpoint("after_scene_preview_card");
        let scene_live_gizmo_bounds =
            self.scene_live_gizmo_bounds(scene_live_preview_w, scene_live_preview_h);
        let rotate_handle_x =
            scene_live_gizmo_bounds.left + scene_live_gizmo_bounds.width * 0.5 - 8.0;
        let rotate_handle_y = (scene_live_gizmo_bounds.top - 34.0).max(8.0);
        let scene_live_gizmo_overlay = div()
            .absolute()
            .left_0()
            .top_0()
            .w_full()
            .h_full()
            .child(
                div()
                    .absolute()
                    .left(px(scene_live_gizmo_bounds.left))
                    .top(px(scene_live_gizmo_bounds.top))
                    .w(px(scene_live_gizmo_bounds.width))
                    .h(px(scene_live_gizmo_bounds.height))
                    .border_1()
                    .border_color(rgba(0x38bdf8ee))
                    .bg(rgba(0x38bdf814)),
            )
            .child(
                div()
                    .absolute()
                    .left(px(
                        scene_live_gizmo_bounds.left + scene_live_gizmo_bounds.width * 0.5
                    ))
                    .top(px(rotate_handle_y + 14.0))
                    .w(px(1.0))
                    .h(px(
                        (scene_live_gizmo_bounds.top - rotate_handle_y - 4.0).max(12.0)
                    ))
                    .bg(rgba(0x38bdf8cc)),
            )
            .child(
                div()
                    .absolute()
                    .left(px(rotate_handle_x))
                    .top(px(rotate_handle_y))
                    .w(px(16.0))
                    .h(px(16.0))
                    .rounded_full()
                    .border_1()
                    .border_color(rgba(0xffffffee))
                    .bg(rgba(0x0ea5e9ee))
                    .cursor_pointer()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, evt: &MouseDownEvent, _window, cx| {
                            let mouse_x = f32::from(evt.position.x);
                            let mouse_y = f32::from(evt.position.y);
                            this.scene_live_gizmo_mode = SceneLiveGizmoMode::Rotate;
                            this.begin_scene_live_gizmo_drag(mouse_x, mouse_y);
                            this.status_line =
                                "Scene Live gizmo: release to key rotation.".to_string();
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    ),
            );
        let scene_live_preview_card = div()
            .relative()
            .w_full()
            .h(px(scene_live_preview_h))
            .flex_shrink_0()
            .child(scene_live_preview_card)
            .when(!self.scene_external_preview_active(), |el| {
                el.child(scene_live_gizmo_overlay)
            })
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, evt: &MouseDownEvent, _window, cx| {
                    let mouse_x = f32::from(evt.position.x);
                    let mouse_y = f32::from(evt.position.y);
                    this.begin_scene_live_gizmo_drag(mouse_x, mouse_y);
                    this.status_line = match this.scene_live_gizmo_mode {
                        SceneLiveGizmoMode::Move => "Scene Live gizmo: release to key x/y.",
                        SceneLiveGizmoMode::Rotate => "Scene Live gizmo: release to key rotation.",
                    }
                    .to_string();
                    cx.notify();
                }),
            )
            .on_mouse_move(cx.listener(move |this, evt: &MouseMoveEvent, _window, cx| {
                if !evt.dragging() {
                    return;
                }
                let mouse_x = f32::from(evt.position.x);
                let mouse_y = f32::from(evt.position.y);
                this.update_scene_live_gizmo_drag(mouse_x, mouse_y);
                cx.notify();
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(move |this, _evt: &MouseUpEvent, window, cx| {
                    this.commit_scene_live_gizmo_drag(window, cx);
                    cx.notify();
                }),
            );

        let scene_live_knob_target = format!(
            "{}.{}",
            self.scene_live_knob_node_id, self.scene_live_knob_attr
        );
        let scene_live_knob_value = self
            .scene_live_knob_current_value()
            .map(Self::format_live_number)
            .unwrap_or_else(|| {
                Self::format_live_number(Self::scene_live_attr_default_value(
                    &self.scene_live_knob_attr,
                ))
            });
        self.sync_scene_live_knob_input(scene_live_knob_value.clone(), window, cx);
        let scene_live_knob_input_elem = if let Some(input) = self.scene_live_knob_input.as_ref() {
            let input_for_focus = input.clone();
            div()
                .h(px(24.0))
                .w(px(74.0))
                .rounded_sm()
                .border_1()
                .border_color(white().opacity(0.10))
                .bg(rgb(0x090d14))
                .overflow_hidden()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |_, _, window, cx| {
                        cx.stop_propagation();
                        input_for_focus.read(cx).focus_handle(cx).focus(window);
                    }),
                )
                .child(Input::new(input).h_full().w_full())
                .into_any_element()
        } else {
            div().h(px(24.0)).w(px(74.0)).into_any_element()
        };
        let scene_live_knob_scroll_label = format!("Scroll {}", self.scene_live_knob_attr);
        let scene_live_scroll_attr = self.scene_live_knob_attr.clone();
        let keyframe_summary = self.scene_live_keyframe_summary();
        let keyframe_button_active =
            Self::scene_live_animation_target_property_supported(&self.scene_live_knob_attr);
        let keyframe_time_mode_label = if self.scene_live_keyframe_show_as_frame {
            "Write frame"
        } else {
            "Write time"
        };
        let (
            scene_live_small_step,
            scene_live_large_step,
            scene_live_neg_large_label,
            scene_live_neg_small_label,
            scene_live_pos_small_label,
            scene_live_pos_large_label,
        ) = Self::scene_live_step_labels(&self.scene_live_knob_attr);
        let scene_live_scroll_hint = Self::scene_live_scroll_hint(&self.scene_live_knob_attr);
        let scene_live_preview_quality_chips = [
            SceneLivePreviewQuality::P144,
            SceneLivePreviewQuality::P240,
            SceneLivePreviewQuality::P360,
            SceneLivePreviewQuality::P480,
        ]
        .into_iter()
        .map(|quality| {
            Self::scene_live_chip(
                quality.label().to_string(),
                self.scene_live_preview_quality == quality,
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    this.scene_live_preview_quality = quality;
                    this.invalidate_scene_live_preview_cache();
                    this.status_line = format!(
                        "Scene preview resolution: {} <= {}px.",
                        quality.label(),
                        quality.max_dim()
                    );
                    cx.notify();
                }),
            )
        })
        .collect::<Vec<_>>();
        let scene_live_preview_quality_group = div()
            .h(px(28.0))
            .flex_shrink_0()
            .flex()
            .items_center()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(white().opacity(0.58))
                    .child("Preview"),
            )
            .children(scene_live_preview_quality_chips);
        self.sync_preview_frame_input(window, cx);
        let frame_count = self.playback_frame_count().max(1);
        let preview_frame_input_elem = if let Some(input) = self.preview_frame_input.as_ref() {
            let input_for_focus = input.clone();
            div()
                .h(px(28.0))
                .min_w(px(150.0))
                .px_2()
                .rounded_md()
                .border_1()
                .border_color(white().opacity(0.18))
                .bg(white().opacity(0.055))
                .flex()
                .items_center()
                .gap_2()
                .child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.58))
                        .child(format!("Frame / {}", frame_count.saturating_sub(1))),
                )
                .child(
                    div()
                        .h(px(22.0))
                        .w(px(58.0))
                        .rounded_sm()
                        .border_1()
                        .border_color(white().opacity(0.10))
                        .bg(rgb(0x090d14))
                        .overflow_hidden()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |_, _, window, cx| {
                                cx.stop_propagation();
                                input_for_focus.read(cx).focus_handle(cx).focus(window);
                            }),
                        )
                        .child(Input::new(input).h_full().w_full()),
                )
                .into_any_element()
        } else {
            div().h(px(28.0)).w(px(150.0)).into_any_element()
        };
        let scene_live_knob_scroll = div()
            .h(px(28.0))
            .min_w(px(148.0))
            .px_2()
            .rounded_md()
            .border_1()
            .border_color(white().opacity(0.18))
            .bg(white().opacity(0.055))
            .cursor_pointer()
            .flex()
            .items_center()
            .justify_between()
            .gap_2()
            .on_scroll_wheel(
                cx.listener(move |this, evt: &ScrollWheelEvent, window, cx| {
                    let delta = Self::scene_live_scroll_delta(evt, &scene_live_scroll_attr);
                    if delta.abs() <= f32::EPSILON {
                        return;
                    }
                    cx.stop_propagation();
                    this.nudge_scene_live_knob(delta, window, cx, true);
                    cx.notify();
                }),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(white().opacity(0.58))
                    .child(scene_live_knob_scroll_label),
            )
            .child(scene_live_knob_input_elem);
        let scene_live_status_is_warning = self.scene_live_preview_status.contains("CPU fallback")
            || self.scene_live_preview_status.contains("error");
        let scene_live_preview_status_chip = div()
            .h(px(26.0))
            .px_2()
            .flex_shrink_0()
            .rounded_md()
            .border_1()
            .border_color(if scene_live_status_is_warning {
                rgba(0xff7a66ff)
            } else {
                rgba(0xffffff24)
            })
            .bg(if scene_live_status_is_warning {
                rgba(0x3a1212aa)
            } else {
                rgba(0xffffff0c)
            })
            .flex()
            .items_center()
            .text_xs()
            .text_color(if scene_live_status_is_warning {
                rgba(0xffb4aaff)
            } else {
                rgba(0xffffffad)
            })
            .child(self.scene_live_preview_status.clone());
        let scene_live_gizmo_mode_row = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(white().opacity(0.58))
                    .child("Gizmo"),
            )
            .child(
                Self::scene_live_chip(
                    "Move x/y".to_string(),
                    self.scene_live_gizmo_mode == SceneLiveGizmoMode::Move,
                )
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.scene_live_gizmo_mode = SceneLiveGizmoMode::Move;
                        this.status_line =
                            "Scene Live gizmo: drag preview to move x/y.".to_string();
                        cx.notify();
                    }),
                ),
            )
            .child(
                Self::scene_live_chip(
                    "Rotate".to_string(),
                    self.scene_live_gizmo_mode == SceneLiveGizmoMode::Rotate,
                )
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.scene_live_gizmo_mode = SceneLiveGizmoMode::Rotate;
                        this.status_line =
                            "Scene Live gizmo: drag preview horizontally to rotate.".to_string();
                        cx.notify();
                    }),
                ),
            );
        let current_channel_keys = self.scene_live_current_animation_keys();
        let current_key = current_channel_keys
            .iter()
            .find(|key| key.frame == self.preview_frame)
            .cloned();
        let scene_live_controls_row = div()
            .w_full()
            .min_w_0()
            .flex()
            .flex_wrap()
            .items_center()
            .gap_2()
            .child(scene_live_preview_quality_group)
            .child(scene_live_gizmo_mode_row)
            .child(
                div()
                    .h(px(28.0))
                    .w(px(34.0))
                    .flex_shrink_0()
                    .rounded_md()
                    .border_1()
                    .border_color(white().opacity(0.15))
                    .bg(white().opacity(0.06))
                    .hover(|s| s.bg(white().opacity(0.1)))
                    .cursor_pointer()
                    .text_sm()
                    .text_color(white().opacity(0.92))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(if self.preview_playing { "||" } else { "▶" })
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, window, cx| {
                            this.toggle_preview_playback(window, cx);
                        }),
                    ),
            )
            .child(preview_frame_input_elem)
            .child(Self::control_button("Frame 0").on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    this.set_preview_frame(0);
                    cx.notify();
                }),
            ))
            .child(Self::control_button("F-1").on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    this.step_preview_frame(-1);
                    cx.notify();
                }),
            ))
            .child(Self::control_button("F+1").on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    this.step_preview_frame(1);
                    cx.notify();
                }),
            ))
            .child(
                Self::scene_live_chip(scene_live_neg_large_label, false).on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, window, cx| {
                        this.nudge_scene_live_knob(-scene_live_large_step, window, cx, false);
                        cx.notify();
                    }),
                ),
            )
            .child(
                Self::scene_live_chip(scene_live_neg_small_label, false).on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, window, cx| {
                        this.nudge_scene_live_knob(-scene_live_small_step, window, cx, false);
                        cx.notify();
                    }),
                ),
            )
            .child(scene_live_knob_scroll)
            .child(
                Self::scene_live_chip(
                    "Key Frame".to_string(),
                    keyframe_button_active && current_key.is_some(),
                )
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, window, cx| {
                        this.upsert_scene_live_keyframe_for_code_block(window, cx);
                        cx.notify();
                    }),
                ),
            )
            .child(
                Self::scene_live_chip(
                    keyframe_time_mode_label.to_string(),
                    !self.scene_live_keyframe_show_as_frame,
                )
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.scene_live_keyframe_show_as_frame =
                            !this.scene_live_keyframe_show_as_frame;
                        this.status_line = if this.scene_live_keyframe_show_as_frame {
                            "Scene Live keyframes will be written as frame=\"...\".".to_string()
                        } else {
                            "Scene Live keyframes will be written as time=\"...\".".to_string()
                        };
                        cx.notify();
                    }),
                ),
            )
            .child(
                Self::scene_live_chip(scene_live_pos_small_label, false).on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, window, cx| {
                        this.nudge_scene_live_knob(scene_live_small_step, window, cx, false);
                        cx.notify();
                    }),
                ),
            )
            .child(
                Self::scene_live_chip(scene_live_pos_large_label, false).on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, window, cx| {
                        this.nudge_scene_live_knob(scene_live_large_step, window, cx, false);
                        cx.notify();
                    }),
                ),
            );

        let current_key_label = current_key
            .as_ref()
            .map(|key| {
                format!(
                    "Current key: {} · value {} · ease {}",
                    Self::scene_live_key_timing_label(key),
                    key.value,
                    key.ease
                )
            })
            .unwrap_or_else(|| format!("Current frame {} has no key.", self.preview_frame));
        let keyframe_ease_row = ["linear", "ease_in", "ease_out", "ease_in_out"]
            .into_iter()
            .map(|ease| {
                let active = current_key.as_ref().is_some_and(|key| key.ease == ease);
                let frame = self.preview_frame;
                Self::scene_live_chip(ease.to_string(), active).on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, window, cx| {
                        this.set_scene_live_keyframe_ease_for_code_block(frame, ease, window, cx);
                        cx.notify();
                    }),
                )
            })
            .collect::<Vec<_>>();
        let keyframe_rows = current_channel_keys
            .iter()
            .take(18)
            .cloned()
            .map(|key| {
                let jump_frame = key.frame;
                let delete_frame = key.frame;
                let is_current = jump_frame == self.preview_frame;
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .px_2()
                    .py_1()
                    .rounded_md()
                    .bg(if is_current {
                        rgba(0x2b8cff22)
                    } else {
                        rgba(0xffffff08)
                    })
                    .child(
                        div()
                            .min_w_0()
                            .flex()
                            .items_center()
                            .gap_2()
                            .text_xs()
                            .text_color(white().opacity(0.78))
                            .child(Self::scene_live_key_timing_label(&key))
                            .child("·")
                            .child(format!("f{}", key.frame))
                            .child("·")
                            .child(format!("value {}", key.value))
                            .child("·")
                            .child(format!("ease {}", key.ease)),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .child(Self::control_button("Jump").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    this.set_preview_frame(jump_frame);
                                    cx.notify();
                                }),
                            ))
                            .child(Self::control_button("Delete").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, window, cx| {
                                    this.delete_scene_live_keyframe_for_code_block(
                                        delete_frame,
                                        window,
                                        cx,
                                    );
                                    cx.notify();
                                }),
                            )),
                    )
            })
            .collect::<Vec<_>>();
        let keyframe_hidden_count = current_channel_keys.len().saturating_sub(18);
        let scene_live_keyframe_panel = div()
            .w_full()
            .rounded_lg()
            .border_1()
            .border_color(white().opacity(0.12))
            .bg(rgba(0x060a12cc))
            .p_2()
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(white().opacity(0.88))
                            .child(format!(
                                "Keys · {}.{}",
                                self.scene_live_knob_node_id, self.scene_live_knob_attr
                            )),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.58))
                            .child(format!("{} key(s)", current_channel_keys.len())),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(if current_key.is_some() {
                        rgba(0x9bdcffee)
                    } else {
                        rgba(0xffffff8f)
                    })
                    .child(current_key_label),
            )
            .when(current_key.is_some(), |el| {
                let delete_frame = self.preview_frame;
                el.child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap_1()
                        .child(
                            div()
                                .text_xs()
                                .text_color(white().opacity(0.58))
                                .child("Ease"),
                        )
                        .children(keyframe_ease_row)
                        .child(Self::control_button("Delete Current").on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, _, window, cx| {
                                this.delete_scene_live_keyframe_for_code_block(
                                    delete_frame,
                                    window,
                                    cx,
                                );
                                cx.notify();
                            }),
                        )),
                )
            })
            .when(!keyframe_rows.is_empty(), |el| {
                el.child(
                    div()
                        .w_full()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .children(keyframe_rows),
                )
            })
            .when(current_channel_keys.is_empty(), |el| {
                el.child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.48))
                        .child("No keys for this selected target/property yet."),
                )
            })
            .when(keyframe_hidden_count > 0, |el| {
                el.child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.48))
                        .child(format!("+{} more key(s) hidden", keyframe_hidden_count)),
                )
            });

        let source_button = Self::control_button("Source / Import").on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, _, _, cx| {
                this.open_import_modal();
                cx.notify();
            }),
        );
        let assets_button = Self::control_button("Assets").on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, _, _, cx| {
                this.open_asset_modal();
                cx.notify();
            }),
        );

        // --- Left column: Graph Lab (code) ---
        let graph_lab_panel = div()
            .w_full()
            .flex_1()
            .min_w_0()
            .min_h(px(260.0))
            .rounded_lg()
            .border_1()
            .border_color(white().opacity(0.12))
            .bg(rgb(0x0c111b))
            .p_3()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .text_sm()
                            .text_color(white().opacity(0.94))
                            .child("Graph Lab"),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap_2()
                            .child(source_button)
                            .child(assets_button)
                            .child(Self::control_button("GitHub Example").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, window, cx| {
                                    this.open_scene_template_dialog(window, cx);
                                }),
                            ))
                            .child(Self::control_button("Template Picker").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    this.open_template_modal();
                                    cx.notify();
                                }),
                            ))
                            .child(Self::control_button("Inspect GLB").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    this.open_glb_inspector_modal(cx);
                                }),
                            ))
                            .child(Self::control_button("Apply Effect").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    this.apply_script_command(cx);
                                    cx.notify();
                                }),
                            ))
                            .child(Self::control_button("Render…").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    this.import_modal_open = false;
                                    this.asset_modal_open = false;
                                    this.template_modal_open = false;
                                    this.scene_template_modal_open = false;
                                    this.glb_inspector_modal_open = false;
                                    this.scene_render_modal_open = true;
                                    this.status_line = "Render mode selector opened.".to_string();
                                    cx.notify();
                                }),
                            )),
                    ),
            )
            .child(div().text_xs().text_color(white().opacity(0.68)).child(
                "Use Scene Live Preview for direct frame playback. Use Source / Import only when the graph needs media inputs.",
            ))
            .child(script_input_elem)
            .child(
                div()
                    .text_xs()
                    .text_color(white().opacity(0.56))
                    .child(keyframe_summary),
            );

        // --- Scene Live Preview: direct single-frame scene raster, no FFmpeg ---
        let scene_live_panel = div()
            .w_full()
            .h_auto()
            .flex_shrink_0()
            .min_h(px(scene_live_panel_min_h))
            .rounded_lg()
            .border_1()
            .border_color(white().opacity(0.12))
            .bg(rgb(0x0c111b))
            .p_3()
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .flex()
                            .flex_1()
                            .min_w_0()
                            .items_center()
                            .gap_3()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(white().opacity(0.95))
                                    .child("Scene Live Preview"),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.56))
                                    .truncate()
                                    .child(format!(
                                        "Direct frame {} · no FFmpeg · {} · target {}",
                                        self.preview_frame,
                                        scene_live_preview_label,
                                        scene_live_knob_target
                                    )),
                            ),
                    ),
            )
            .child(scene_live_selector_panel)
            .child(scene_live_controls_row)
            .child(scene_live_keyframe_panel)
            .child(scene_live_preview_card)
            .child(
                div()
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .text_xs()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .text_color(white().opacity(0.58))
                            .truncate()
                            .child(scene_live_scroll_hint),
                    )
                    .when_some(scene_render_progress, |el, progress| {
                        let pct = progress.percent();
                        el.child(
                            div()
                                .flex_shrink_0()
                                .w(px(250.0))
                                .flex()
                                .items_center()
                                .gap_2()
                                .child(
                                    div()
                                        .w(px(88.0))
                                        .text_color(white().opacity(0.72))
                                        .child(format!("{} {}%", progress.label, pct)),
                                )
                                .child(
                                    div()
                                        .h(px(6.0))
                                        .flex_1()
                                        .rounded_full()
                                        .bg(white().opacity(0.10))
                                        .overflow_hidden()
                                        .child(
                                            div()
                                                .h_full()
                                                .w(px((pct as f32 / 100.0) * 96.0))
                                                .rounded_full()
                                                .bg(rgba(0x79c7ffcc)),
                                        ),
                                )
                                .child(div().w(px(58.0)).text_color(white().opacity(0.54)).child(
                                    format!(
                                        "{}/{}",
                                        progress.rendered_frames, progress.total_frames
                                    ),
                                ))
                                .child(
                                    div()
                                        .h(px(22.0))
                                        .w(px(22.0))
                                        .rounded_md()
                                        .border_1()
                                        .border_color(rgba(0xff7a6688))
                                        .bg(rgba(0x3a1212aa))
                                        .hover(|s| s.bg(rgba(0x5a1818cc)))
                                        .cursor_pointer()
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .text_xs()
                                        .text_color(rgba(0xffc0b8ff))
                                        .child("×")
                                        .on_mouse_down(
                                            MouseButton::Left,
                                            cx.listener(|this, _, _, cx| {
                                                cx.stop_propagation();
                                                this.cancel_scene_render_job();
                                                cx.notify();
                                            }),
                                        ),
                                ),
                        )
                    })
                    .child(scene_live_preview_status_chip),
            )
            .when_some(scene_render_log_panel, |el, panel| el.child(panel));

        // --- Template picker modal overlay (rendered on top when open) ---
        let template_modal = if self.template_modal_open {
            Some(self.render_template_modal_overlay(cx))
        } else {
            None
        };

        // --- Import/source modal overlay ---
        let import_modal = if self.import_modal_open {
            Some(self.render_import_modal_overlay(
                selected_name.clone(),
                selected_kind_label.clone(),
                selected_duration_label.clone(),
                selected_path_label.clone(),
                imported_count,
                cx,
            ))
        } else {
            None
        };
        let asset_modal = if self.asset_modal_open {
            Some(self.render_vfx_asset_modal_overlay(window, cx))
        } else {
            None
        };
        let glb_inspector_modal = if self.glb_inspector_modal_open {
            Some(self.render_glb_inspector_modal_overlay(cx))
        } else {
            None
        };
        let scene_render_modal = if self.scene_render_modal_open {
            Some(self.render_scene_render_modal_overlay(cx))
        } else {
            None
        };

        let source_summary_line = format!(
            "Current source: {} · {} · {} · {} imported · {}",
            selected_name,
            selected_kind_label,
            selected_duration_label,
            imported_count,
            selected_path_label
        );

        div()
            .size_full()
            .bg(rgb(0x080a10))
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(
                div()
                    .flex_shrink_0()
                    .border_b_1()
                    .border_color(white().opacity(0.12))
                    .bg(rgb(0x090b12))
                    .px_3()
                    .py_2()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .child(
                                div()
                                    .text_lg()
                                    .text_color(white().opacity(0.96))
                                    .child("MotionLoom · Studio"),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.66))
                                    .truncate()
                                    .child("Import panel is now modal. Main view is fixed two columns."),
                            ),
                    )
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .overflow_hidden()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.56))
                                    .whitespace_normal()
                                    .child(source_summary_line),
                            ),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .min_w_0()
                    .overflow_y_scrollbar()
                    .child(
                        div()
                            .w_full()
                            .max_w(px(content_max_w))
                            .mx_auto()
                            .p_3()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .child(scene_live_panel)
                            .child(graph_lab_panel),
                    ),
            )
            .when(self.import_modal_open, |el| el.child(import_modal.unwrap()))
            .when(self.asset_modal_open, |el| el.child(asset_modal.unwrap()))
            .when(self.glb_inspector_modal_open, |el| {
                el.child(glb_inspector_modal.unwrap())
            })
            .when(self.scene_render_modal_open, |el| {
                el.child(scene_render_modal.unwrap())
            })
            .when(self.template_modal_open, |el| {
                el.child(template_modal.unwrap())
            })
            .when(self.preview_playing, |el| {
                window.request_animation_frame();
                el
            })
            .when_some(scene_template_modal, |el, modal| el.child(modal))
    }
}
