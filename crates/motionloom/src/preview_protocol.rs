// =========================================
// =========================================
// crates/motionloom/src/preview_protocol.rs

use serde::{Deserialize, Serialize};

use crate::WgpuPreviewQuality;

pub const PREVIEW_PROTOCOL_VERSION: u32 = 1;

/// Host-agnostic commands from an editor controller to a preview viewer.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PreviewCommand {
    LoadScript {
        script: String,
        #[serde(default)]
        source: Option<String>,
    },
    SetFrame {
        frame: u32,
    },
    SetQuality {
        quality: WgpuPreviewQuality,
    },
    SetOverride {
        node: String,
        property: String,
        value: f32,
    },
    ClearOverride {
        node: String,
        property: String,
    },
    SetAssetRoots {
        roots: Vec<String>,
    },
    SetWindowBounds {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        decorations: bool,
    },
    SetWindowVisible {
        visible: bool,
    },
    SetControllerProcessId {
        pid: u32,
    },
    SetInteractionTarget {
        node: String,
        mode: PreviewInteractionMode,
        graph_width: f32,
        graph_height: f32,
        x: f32,
        y: f32,
        rotation: f32,
    },
    SetInteractionTargets {
        mode: PreviewInteractionMode,
        graph_width: f32,
        graph_height: f32,
        targets: Vec<PreviewInteractionNode>,
    },
}

/// Interaction tool mode shared by editor controllers and preview viewers.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviewInteractionMode {
    Move,
    Rotate,
}

/// Controller-provided editable node bounds for native preview hit testing.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreviewInteractionNode {
    pub node: String,
    pub tag: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub rotation: f32,
}

/// Viewer events emitted by external hosts or future embedded preview surfaces.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PreviewEvent {
    Ready {
        protocol_version: u32,
    },
    Rendered {
        frame: u32,
    },
    /// Periodic host telemetry makes GPU residency and fallback behavior
    /// visible to editor UIs instead of inferring it from process load.
    FrameMetrics {
        frame: u32,
        backend: String,
        zero_copy: bool,
        cpu_ms: f64,
        gpu_ms: Option<f64>,
        readback_ms: f64,
        draw_calls: usize,
    },
    WindowBounds {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    },
    HostFocus {
        focused: bool,
    },
    Error {
        message: String,
    },
    PickResult {
        node: Option<String>,
        x: f32,
        y: f32,
    },
    TransformDrag {
        node: String,
        property: String,
        value: f32,
    },
    TransformDragEnd {
        node: String,
    },
}
