// =========================================================================
// ============================================================================
// src/ui/timeline_panel.rs
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use gpui::{
    Bounds, Context, Entity, ExternalPaths, FocusHandle, Focusable, IntoElement, KeyDownEvent,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathPromptOptions, Pixels, Render,
    ScrollWheelEvent, Subscription, Window, canvas, div, point, prelude::*, px, quad, rgb, rgba,
    size, transparent_black,
};

use gpui_component::{
    black,
    scroll::ScrollableElement,
    slider::{Slider, SliderEvent, SliderState},
    white,
};

use crate::core::waveform::{WaveformStatus, waveform_key};
use crate::core::{
    export::{get_media_duration, is_supported_media_path},
    global_state::{
        ActiveTool, AudioTrack, Clip, GlobalState, MacPreviewRenderMode, MediaPoolUiEvent,
        PlaybackUiEvent, PreviewQuality, ProxyRenderMode, SemanticClip, SubtitleClip, TrackType,
    },
    proxy::{ProxyStatus, proxy_key, proxy_path_for_in},
};
use crate::ui::display_settings_modal::{DisplaySettingsModalState, display_ratio_label};
use crate::ui::export_modal::ExportModalState;

// ---- Layout constants ----
const APP_NAV_W: f32 = 64.0;
const LEFT_TOOL_W: f32 = 44.0;
const TRACK_LIST_W: f32 = 150.0;
const RIGHT_STRIP_W: f32 = 34.0;

const TIMELINE_HEADER_H: f32 = 66.0;
const TIMELINE_BODY_H: f32 = 320.0;
const TIMELINE_PANEL_H: f32 = TIMELINE_HEADER_H + TIMELINE_BODY_H;

const RULER_H: f32 = 34.0;
const LANE_H: f32 = 28.0;
const AUDIO_WAVEFORM_BUCKETS_MIN: usize = 2048;
const AUDIO_WAVEFORM_BUCKETS_MAX: usize = 65536;
const AUDIO_WAVEFORM_BUCKETS_PER_SEC: f32 = 60.0;

const V1_WINDOW_SECS: u64 = 300;
const V1_HARD_CAP: usize = 5000;
const AUDIO_TRACK_GAIN_MIN_DB: f32 = -60.0;
const AUDIO_TRACK_GAIN_MAX_DB: f32 = 12.0;
const AUDIO_WAVEFORM_MIN_SAMPLES_HIGH: usize = 64;
const AUDIO_WAVEFORM_MIN_SAMPLES_MEDIUM: usize = 32;
const AUDIO_WAVEFORM_MIN_SAMPLES_LOW: usize = 24;
const AUDIO_WAVEFORM_MIN_SAMPLES_ULTRA_LOW: usize = 16;
const AUDIO_WAVEFORM_MAX_SAMPLES_HIGH: usize = 1600;
const AUDIO_WAVEFORM_MAX_SAMPLES_MEDIUM: usize = 960;
const AUDIO_WAVEFORM_MAX_SAMPLES_LOW: usize = 640;
const AUDIO_WAVEFORM_MAX_SAMPLES_ULTRA_LOW: usize = 320;
const AUDIO_WAVEFORM_MIN_SAMPLES_EXTREME: usize = 8;
const AUDIO_WAVEFORM_MAX_SAMPLES_EXTREME: usize = 48;
const AUDIO_WAVEFORM_EXTREME_CLIP_THRESHOLD_DEFAULT: usize = 180;
const AUDIO_CLIP_NODE_VIRTUALIZE_THRESHOLD: usize = 140;
const AUDIO_CLIP_NODE_MIN_PIXEL_WIDTH: f32 = 12.0;
const AUDIO_CLIP_CLUSTER_GAP_PX: f32 = 2.0;
const TIMELINE_LOW_LOAD_MAX_SEGMENTS: usize = 320;
const TIMELINE_LOW_LOAD_MIN_SEGMENT_PX: f32 = 1.5;
const TIMELINE_LOW_LOAD_MERGE_GAP_PX: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AudioWaveformDetailLevel {
    Normal,
    Extreme,
}

#[derive(Clone, Copy, Debug)]
struct AudioWaveformOverlayRect {
    left: f32,
    width: f32,
    amp_norm: f32,
}

#[derive(Clone, Copy, Debug)]
struct LowLoadSegmentRect {
    left: f32,
    width: f32,
}

// DragState stores TrackType directly instead of the old track_idx field.
#[derive(Clone, Debug)]
enum DragState {
    None,
    Scrubbing,
    Pending {
        start_x: Pixels,
        clip_id: u64,
        track_type: TrackType, // Track type replaces the old track_idx field.
        original_start: Duration,
        group_items: Option<Vec<GroupDragItem>>,
    },
    Dragging {
        clip_id: u64,
        track_type: TrackType, // Track type replaces the old track_idx field.
        offset_seconds: f64,
    },
    GroupDragging {
        items: Vec<GroupDragItem>,
        anchor_start: Duration,
        offset_seconds: f64,
    },
    LayerPending {
        start_x: Pixels,
        clip_id: u64,
        original_start: Duration,
    },
    LayerDragging {
        clip_id: u64,
        offset_seconds: f64,
    },
    LayerResizing {
        clip_id: u64,
        start_x: Pixels,
        original_duration: Duration,
    },
    SemanticPending {
        start_x: Pixels,
        clip_id: u64,
        original_start: Duration,
    },
    SemanticDragging {
        clip_id: u64,
        offset_seconds: f64,
    },
    SemanticResizing {
        clip_id: u64,
        start_x: Pixels,
        original_duration: Duration,
    },
    // Resizing state
    Resizing {
        clip_id: u64,
        track_type: TrackType,
        start_x: Pixels,             // Mouse-down position.
        original_duration: Duration, // Original duration at drag start.
    },
    Marquee {
        start: gpui::Point<Pixels>,
        current: gpui::Point<Pixels>,
    },
}

#[derive(Clone, Debug)]
struct GroupDragItem {
    clip_id: u64,
    kind: GroupDragKind,
    original_start: Duration,
}

pub struct TimelinePanel {
    pub global: Entity<GlobalState>,
    pub focus_handle: FocusHandle,
    state_sig: u64,

    pub scroll_offset_x: f32,
    pub scroll_offset_y: f32,

    pub is_scrubbing: bool,
    drag_state: DragState,

    pub px_per_sec: f32,
    pub pending_subtitle_drop: bool,
    show_semantic_lane: bool,
    srt_menu_open: bool,
    srt_menu_anchor: Option<(f32, f32)>,
    media_pool_hover_track: Option<TrackType>,
    media_pool_hover_time: Option<Duration>,

    pub(crate) display_settings_modal: DisplaySettingsModalState,
    pub(crate) export_modal: ExportModalState,
    proxy_modal_open: bool,
    proxy_confirm_quality: Option<PreviewQuality>,
    proxy_confirm_delete_quality: Option<PreviewQuality>,
    proxy_confirm_delete_all: bool,
    proxy_spinner_running: bool,
    proxy_spinner_phase: usize,
    proxy_spinner_token: u64,
    clip_link_menu: Option<ClipLinkContextMenu>,
    timeline_clip_menu: Option<TimelineClipContextMenu>,
    layer_clip_menu: Option<LayerClipContextMenu>,
    subtitle_clip_menu: Option<SubtitleClipContextMenu>,
    semantic_clip_menu: Option<SemanticClipContextMenu>,
    copied_timeline_item: Option<TimelineClipboardItem>,
    audio_gain_sliders: HashMap<usize, Entity<SliderState>>,
    audio_gain_slider_subs: HashMap<usize, Subscription>,
    audio_lane_time_index_token: u64,
    audio_lane_clip_start_index: Vec<Vec<f32>>,
    audio_lane_clip_end_index: Vec<Vec<f32>>,
    ui_fps_last_instant: Option<Instant>,
    ui_fps_ema: f32,
    timeline_load_mode: TimelineLoadMode,
}

impl Focusable for TimelinePanel {
    fn focus_handle(&self, _cx: &gpui::App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

#[derive(Clone, Copy)]
enum TrackHeaderKind {
    V1,
    Audio(usize),
    VideoOverlay(usize),
    Subtitle(usize),
    Semantic,
}

#[derive(Clone, Copy, Debug)]
enum GroupDragKind {
    Timeline(TrackType),
    LayerEffect,
    Semantic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProxyRowState {
    NoSelection,
    Missing,
    Pending,
    Ready,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimelineLoadMode {
    AutoLiteOnPlay,
    Normal,
}

impl TimelineLoadMode {
    fn label(self) -> &'static str {
        match self {
            TimelineLoadMode::AutoLiteOnPlay => "Timeline: Auto Lite",
            TimelineLoadMode::Normal => "Timeline: Normal",
        }
    }

    fn toggled(self) -> Self {
        match self {
            TimelineLoadMode::AutoLiteOnPlay => TimelineLoadMode::Normal,
            TimelineLoadMode::Normal => TimelineLoadMode::AutoLiteOnPlay,
        }
    }

    fn use_low_load(self, playing: bool) -> bool {
        match self {
            TimelineLoadMode::AutoLiteOnPlay => playing,
            TimelineLoadMode::Normal => false,
        }
    }
}

#[derive(Clone)]
struct TrackUi {
    name: String,
    kind: TrackHeaderKind,
}

#[derive(Clone, Copy, Debug)]
struct ClipLinkContextMenu {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug)]
struct TimelineClipContextMenu {
    x: f32,
    y: f32,
    clip_id: u64,
    track_type: TrackType,
}

#[derive(Clone, Copy, Debug)]
struct LayerClipContextMenu {
    x: f32,
    y: f32,
    clip_id: u64,
}

#[derive(Clone, Copy, Debug)]
struct SubtitleClipContextMenu {
    x: f32,
    y: f32,
    clip_id: u64,
    track_index: usize,
}

#[derive(Clone, Copy, Debug)]
struct SemanticClipContextMenu {
    x: f32,
    y: f32,
    clip_id: u64,
}

#[derive(Clone, Copy, Debug)]
enum TimelineClipboardItem {
    Clip { clip_id: u64, track_type: TrackType },
    Subtitle { clip_id: u64, track_index: usize },
    Semantic { clip_id: u64 },
    LayerEffect { clip_id: u64 },
}

fn fmt_mmss(d: Duration) -> String {
    let secs = d.as_secs();
    let m = secs / 60;
    let s = secs % 60;
    format!("{:02}:{:02}", m, s)
}

fn fmt_mmss_millis(d: Duration) -> String {
    let total_ms = d.as_millis() as u64;
    let ms = total_ms % 1000;
    let total_sec = total_ms / 1000;
    let m = total_sec / 60;
    let s = total_sec % 60;
    format!("{:02}:{:02}.{:03}", m, s, ms)
}

fn dur_to_px(d: Duration, px_per_sec: f32) -> f32 {
    d.as_secs_f32() * px_per_sec
}

fn waveform_bucket_count_for_media_duration(media_duration: Duration) -> usize {
    let secs = media_duration.as_secs_f32().max(1.0);
    ((secs * AUDIO_WAVEFORM_BUCKETS_PER_SEC).round() as usize)
        .clamp(AUDIO_WAVEFORM_BUCKETS_MIN, AUDIO_WAVEFORM_BUCKETS_MAX)
}

fn audio_waveform_extreme_mode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Switch:
        // - ANICA_AUDIO_WAVEFORM_EXTREME_MODE=0/off/false => disable
        // - anything else / missing => enable
        std::env::var("ANICA_AUDIO_WAVEFORM_EXTREME_MODE")
            .ok()
            .map(|raw| {
                !matches!(
                    raw.trim().to_ascii_lowercase().as_str(),
                    "0" | "off" | "false"
                )
            })
            .unwrap_or(true)
    })
}

fn audio_waveform_extreme_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("ANICA_AUDIO_WAVEFORM_EXTREME_THRESHOLD")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value >= 16)
            .unwrap_or(AUDIO_WAVEFORM_EXTREME_CLIP_THRESHOLD_DEFAULT)
    })
}

fn waveform_render_profile_for_lane(
    visible_audio_clips: usize,
) -> (usize, usize, AudioWaveformDetailLevel) {
    if audio_waveform_extreme_mode_enabled()
        && visible_audio_clips >= audio_waveform_extreme_threshold()
    {
        return (
            AUDIO_WAVEFORM_MIN_SAMPLES_EXTREME,
            AUDIO_WAVEFORM_MAX_SAMPLES_EXTREME,
            AudioWaveformDetailLevel::Extreme,
        );
    }

    // Dynamically lower waveform density when a lane is visually dense.
    // This keeps UI node count bounded after many razor cuts.
    if visible_audio_clips >= 240 {
        (
            AUDIO_WAVEFORM_MIN_SAMPLES_ULTRA_LOW,
            AUDIO_WAVEFORM_MAX_SAMPLES_ULTRA_LOW,
            AudioWaveformDetailLevel::Normal,
        )
    } else if visible_audio_clips >= 120 {
        (
            AUDIO_WAVEFORM_MIN_SAMPLES_LOW,
            AUDIO_WAVEFORM_MAX_SAMPLES_LOW,
            AudioWaveformDetailLevel::Normal,
        )
    } else if visible_audio_clips >= 60 {
        (
            AUDIO_WAVEFORM_MIN_SAMPLES_MEDIUM,
            AUDIO_WAVEFORM_MAX_SAMPLES_MEDIUM,
            AudioWaveformDetailLevel::Normal,
        )
    } else {
        (
            AUDIO_WAVEFORM_MIN_SAMPLES_HIGH,
            AUDIO_WAVEFORM_MAX_SAMPLES_HIGH,
            AudioWaveformDetailLevel::Normal,
        )
    }
}

fn waveform_slice_for_clip<'a>(peaks: &'a [f32], clip: &Clip) -> Option<&'a [f32]> {
    if peaks.is_empty() {
        return None;
    }

    // Convert clip source-in/duration into a slice over the cached full-file waveform.
    let media_secs = clip.media_duration.as_secs_f32();
    let (slice_start, slice_end) = if media_secs > 0.001 {
        let start_secs = clip.source_in.as_secs_f32().clamp(0.0, media_secs);
        let end_secs = (start_secs + clip.duration.as_secs_f32()).clamp(start_secs, media_secs);
        let start_idx = ((start_secs / media_secs) * peaks.len() as f32).floor() as usize;
        let end_idx = ((end_secs / media_secs) * peaks.len() as f32).ceil() as usize;
        (
            start_idx.min(peaks.len().saturating_sub(1)),
            end_idx.clamp(start_idx.saturating_add(1), peaks.len()),
        )
    } else {
        (0, peaks.len())
    };

    let source = &peaks[slice_start..slice_end];
    if source.is_empty() {
        None
    } else {
        Some(source)
    }
}

fn sample_waveform_for_clip(
    peaks: &[f32],
    clip: &Clip,
    pixel_width: f32,
    min_samples: usize,
    max_samples: usize,
) -> Vec<f32> {
    let Some(source) = waveform_slice_for_clip(peaks, clip) else {
        return Vec::new();
    };

    // Sample at near pixel density so each painted bar keeps the strongest
    // transient in its time slice instead of averaging it away.
    let min_samples = min_samples.max(1);
    let max_samples = max_samples.max(min_samples);
    let sample_count = (pixel_width.ceil() as usize).clamp(min_samples, max_samples);
    let mut out = Vec::with_capacity(sample_count);
    for idx in 0..sample_count {
        let start = (idx * source.len()) / sample_count;
        let mut end = ((idx + 1) * source.len()) / sample_count;
        if end <= start {
            end = (start + 1).min(source.len());
        }
        let window = &source[start..end];
        let peak = window.iter().copied().fold(0.0_f32, f32::max);
        out.push(peak.clamp(0.0, 1.0));
    }

    out
}

fn summarize_waveform_for_clip(peaks: &[f32], clip: &Clip) -> Option<f32> {
    // Extreme mode summary: sample up to ~96 points to avoid heavy per-bar UI construction.
    let source = waveform_slice_for_clip(peaks, clip)?;
    if source.is_empty() {
        return None;
    }

    let stride = (source.len() / 96).max(1);
    let mut count = 0usize;
    let mut peak = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut idx = 0usize;
    while idx < source.len() {
        let value = source[idx].clamp(0.0, 1.0);
        peak = peak.max(value);
        sum += value;
        count = count.saturating_add(1);
        idx = idx.saturating_add(stride);
    }
    if count == 0 {
        return None;
    }
    let mean = sum / count as f32;
    Some((peak * 0.7 + mean * 0.3).clamp(0.0, 1.0))
}

fn get_dynamic_ruler_steps(px_per_sec: f32) -> (u64, u64, u64) {
    const MIN_LABEL_WIDTH_PX: f32 = 80.0;
    let min_ms = (((MIN_LABEL_WIDTH_PX / px_per_sec.max(0.01)) * 1000.0).ceil()).max(1.0) as u64;
    let nice_intervals_ms = [
        1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 15_000, 30_000, 60_000,
        120_000, 300_000, 600_000, 900_000, 1_800_000, 3_600_000,
    ];
    let label_step = *nice_intervals_ms
        .iter()
        .find(|&&x| x >= min_ms)
        .unwrap_or(&3_600_000);

    let major = (label_step / 2).max(1);
    let minor = (major / 5).max(1);

    (minor, major, label_step)
}

fn timeline_state_sig(gs: &GlobalState) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::mem::discriminant;

    let mut h = DefaultHasher::new();
    // Clip / track structure — covers clip edits, additions, removals, track changes
    gs.timeline_edit_token().hash(&mut h);
    // Selection state
    gs.selected_clip_ids.hash(&mut h);
    gs.selected_subtitle_ids.hash(&mut h);
    gs.selected_layer_effect_clip_id().hash(&mut h);
    gs.selected_semantic_clip_id().hash(&mut h);
    // Playback control (play/pause transitions not covered by Tick)
    gs.is_playing.hash(&mut h);
    // Tool / display mode
    discriminant(&gs.active_tool).hash(&mut h);
    discriminant(&gs.v1_move_mode).hash(&mut h);
    discriminant(&gs.preview_quality).hash(&mut h);
    discriminant(&gs.preview_fps).hash(&mut h);
    // Canvas dimensions
    gs.canvas_w.to_bits().hash(&mut h);
    gs.canvas_h.to_bits().hash(&mut h);
    gs.active_source_name.hash(&mut h);
    // Export progress (no Tick fires during export)
    gs.export_in_progress.hash(&mut h);
    gs.export_progress_ratio.to_bits().hash(&mut h);
    gs.export_eta.hash(&mut h);
    gs.export_last_error.hash(&mut h);
    gs.export_last_out_path.hash(&mut h);
    // UI notices
    gs.ui_notice.hash(&mut h);
    gs.pending_trim_to_fit.is_some().hash(&mut h);
    h.finish()
}

impl TimelinePanel {
    pub fn new(global: Entity<GlobalState>, cx: &mut Context<Self>) -> Self {
        let initial_sig = {
            let gs = global.read(cx);
            timeline_state_sig(gs)
        };
        cx.observe(&global, |this, global, cx| {
            let sig = timeline_state_sig(global.read(cx));
            if sig != this.state_sig {
                this.state_sig = sig;
                cx.notify();
            }
        })
        .detach();
        cx.subscribe(&global, |_, _, evt: &MediaPoolUiEvent, cx| {
            if matches!(evt, MediaPoolUiEvent::StateChanged) {
                cx.notify();
            }
        })
        .detach();
        cx.subscribe(&global, |_, _, evt: &PlaybackUiEvent, cx| {
            if matches!(evt, PlaybackUiEvent::Tick) {
                cx.notify();
            }
        })
        .detach();

        Self {
            global,
            focus_handle: cx.focus_handle(),
            state_sig: initial_sig,
            scroll_offset_x: 0.0,
            scroll_offset_y: 0.0,
            is_scrubbing: false,
            drag_state: DragState::None,
            px_per_sec: 12.0,
            pending_subtitle_drop: false,
            show_semantic_lane: false,
            srt_menu_open: false,
            srt_menu_anchor: None,
            media_pool_hover_track: None,
            media_pool_hover_time: None,
            display_settings_modal: DisplaySettingsModalState::new(),
            export_modal: ExportModalState::new(),
            proxy_modal_open: false,
            proxy_confirm_quality: None,
            proxy_confirm_delete_quality: None,
            proxy_confirm_delete_all: false,
            proxy_spinner_running: false,
            proxy_spinner_phase: 0,
            proxy_spinner_token: 0,
            clip_link_menu: None,
            timeline_clip_menu: None,
            layer_clip_menu: None,
            subtitle_clip_menu: None,
            semantic_clip_menu: None,
            copied_timeline_item: None,
            audio_gain_sliders: HashMap::new(),
            audio_gain_slider_subs: HashMap::new(),
            audio_lane_time_index_token: u64::MAX,
            audio_lane_clip_start_index: Vec::new(),
            audio_lane_clip_end_index: Vec::new(),
            ui_fps_last_instant: None,
            ui_fps_ema: 0.0,
            timeline_load_mode: TimelineLoadMode::Normal,
        }
    }

    pub fn is_low_load_mode_effective(&self, cx: &gpui::App) -> bool {
        let playing = self.global.read(cx).is_playing;
        self.timeline_load_mode.use_low_load(playing)
    }

    fn ensure_audio_lane_time_index(&mut self, tracks: &[AudioTrack], timeline_edit_token: u64) {
        if self.audio_lane_time_index_token == timeline_edit_token
            && self.audio_lane_clip_start_index.len() == tracks.len()
            && self.audio_lane_clip_end_index.len() == tracks.len()
        {
            let lens_match = tracks.iter().enumerate().all(|(idx, track)| {
                self.audio_lane_clip_start_index
                    .get(idx)
                    .map(|v| v.len())
                    .unwrap_or(0)
                    == track.clips.len()
                    && self
                        .audio_lane_clip_end_index
                        .get(idx)
                        .map(|v| v.len())
                        .unwrap_or(0)
                        == track.clips.len()
            });
            if lens_match {
                return;
            }
        }

        self.audio_lane_clip_start_index.clear();
        self.audio_lane_clip_end_index.clear();
        self.audio_lane_clip_start_index.reserve(tracks.len());
        self.audio_lane_clip_end_index.reserve(tracks.len());

        for track in tracks {
            let mut starts = Vec::with_capacity(track.clips.len());
            let mut ends = Vec::with_capacity(track.clips.len());
            for clip in &track.clips {
                let start = clip.start.as_secs_f32();
                starts.push(start);
                ends.push(start + clip.duration.as_secs_f32());
            }
            self.audio_lane_clip_start_index.push(starts);
            self.audio_lane_clip_end_index.push(ends);
        }

        self.audio_lane_time_index_token = timeline_edit_token;
    }

    fn transport_btn(label: &'static str) -> gpui::Div {
        div()
            .h(px(28.0))
            .px_3()
            .rounded_lg()
            .border_1()
            .border_color(white().opacity(0.12))
            .bg(white().opacity(0.05))
            .text_color(white().opacity(0.85))
            .hover(|s| s.bg(white().opacity(0.10)))
            .cursor_pointer()
            .flex()
            .items_center()
            .justify_center()
            .child(label)
    }

    fn queue_selected_proxies(gs: &mut GlobalState, max_dim: u32) -> bool {
        if !gs.media_tools_ready_for_preview_gen() {
            gs.ui_notice =
                Some("Proxy generation requires FFmpeg. Install tools first.".to_string());
            gs.show_media_dependency_modal();
            return false;
        }

        let mut selected_ids = gs.selected_clip_ids.clone();
        if selected_ids.is_empty()
            && let Some(id) = gs.selected_clip_id
        {
            selected_ids.push(id);
        }
        if selected_ids.is_empty() {
            return false;
        }

        let mut paths = std::collections::HashSet::new();
        let mut push_clip = |clip: &crate::core::global_state::Clip| {
            if selected_ids.contains(&clip.id) {
                let p = clip.file_path.to_lowercase();
                if p.ends_with(".jpg")
                    || p.ends_with(".jpeg")
                    || p.ends_with(".png")
                    || p.ends_with(".webp")
                    || p.ends_with(".bmp")
                    || p.ends_with(".gif")
                {
                    return;
                }
                paths.insert(clip.file_path.clone());
            }
        };
        for c in &gs.v1_clips {
            push_clip(c);
        }
        for track in &gs.video_tracks {
            for c in &track.clips {
                push_clip(c);
            }
        }

        if paths.is_empty() {
            return false;
        }

        for path in paths {
            gs.ensure_proxy_for_path(&path, max_dim);
        }

        true
    }

    fn selected_video_paths_for_proxy(gs: &GlobalState) -> Vec<String> {
        let mut selected_ids = gs.selected_clip_ids.clone();
        if selected_ids.is_empty()
            && let Some(id) = gs.selected_clip_id
        {
            selected_ids.push(id);
        }
        if selected_ids.is_empty() {
            return Vec::new();
        }

        let mut paths = std::collections::HashSet::new();
        let mut push_clip = |clip: &crate::core::global_state::Clip| {
            if selected_ids.contains(&clip.id) {
                let p = clip.file_path.to_lowercase();
                if p.ends_with(".jpg")
                    || p.ends_with(".jpeg")
                    || p.ends_with(".png")
                    || p.ends_with(".webp")
                    || p.ends_with(".bmp")
                    || p.ends_with(".gif")
                {
                    return;
                }
                paths.insert(clip.file_path.clone());
            }
        };
        for c in &gs.v1_clips {
            push_clip(c);
        }
        for track in &gs.video_tracks {
            for c in &track.clips {
                push_clip(c);
            }
        }

        paths.into_iter().collect()
    }

    fn proxy_state_for_quality(gs: &GlobalState, quality: PreviewQuality) -> ProxyRowState {
        let Some(max_dim) = quality.max_dim() else {
            return ProxyRowState::NoSelection;
        };

        let paths = Self::selected_video_paths_for_proxy(gs);
        if paths.is_empty() {
            return ProxyRowState::NoSelection;
        }
        // Use the same cache root policy as proxy generation.
        let cache_root = gs.cache_root_dir();

        let mut any_pending = false;
        let mut all_ready = true;

        for path in paths {
            let src = PathBuf::from(&path);
            if !src.exists() {
                all_ready = false;
                continue;
            }
            let key = proxy_key(&src, max_dim);

            if gs.proxy_active.as_ref().is_some_and(|job| job.key == key)
                || gs.proxy_queue.iter().any(|job| job.key == key)
            {
                any_pending = true;
                all_ready = false;
                continue;
            }

            if let Some(entry) = gs.proxy_entries.get(&key) {
                match entry.status {
                    ProxyStatus::Ready => {
                        if !entry.path.exists() {
                            all_ready = false;
                        }
                    }
                    ProxyStatus::Pending => {
                        any_pending = true;
                        all_ready = false;
                    }
                    ProxyStatus::Missing | ProxyStatus::Failed => {
                        all_ready = false;
                    }
                }
                continue;
            }

            let dst = proxy_path_for_in(&cache_root, &src, max_dim);
            if !dst.exists() {
                all_ready = false;
            }
        }

        if any_pending {
            ProxyRowState::Pending
        } else if all_ready {
            ProxyRowState::Ready
        } else {
            ProxyRowState::Missing
        }
    }

    fn ensure_proxy_spinner_running(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.proxy_spinner_running {
            return;
        }

        self.proxy_spinner_running = true;
        self.proxy_spinner_token = self.proxy_spinner_token.wrapping_add(1);
        let token = self.proxy_spinner_token;

        cx.spawn_in(window, async move |view, window| {
            loop {
                gpui::Timer::after(Duration::from_millis(120)).await;
                let keep = view
                    .update_in(window, |this, _window, cx| {
                        if this.proxy_spinner_token != token || !this.proxy_modal_open {
                            this.proxy_spinner_running = false;
                            return false;
                        }

                        let has_pending = {
                            let gs = this.global.read(cx);
                            matches!(
                                Self::proxy_state_for_quality(gs, PreviewQuality::High),
                                ProxyRowState::Pending
                            ) || matches!(
                                Self::proxy_state_for_quality(gs, PreviewQuality::Medium),
                                ProxyRowState::Pending
                            ) || matches!(
                                Self::proxy_state_for_quality(gs, PreviewQuality::Low),
                                ProxyRowState::Pending
                            )
                        };

                        if !has_pending {
                            this.proxy_spinner_running = false;
                            return false;
                        }

                        this.proxy_spinner_phase = (this.proxy_spinner_phase + 1) % 4;
                        cx.notify();
                        true
                    })
                    .unwrap_or(false);

                if !keep {
                    break;
                }
            }
        })
        .detach();
    }

    fn tool_btn(label: &'static str, active: bool) -> gpui::Div {
        let bg = if active {
            rgba(0x2563eb40)
        } else {
            rgba(0xffffff08)
        };
        let border = if active {
            rgba(0x60a5fa80)
        } else {
            rgba(0xffffff14)
        };
        div()
            .w(px(32.0))
            .h(px(32.0))
            .rounded_lg()
            .border_1()
            .border_color(border)
            .bg(bg)
            .hover(|s| s.bg(white().opacity(0.06)))
            .cursor_pointer()
            .flex()
            .items_center()
            .justify_center()
            .text_color(white().opacity(0.75))
            .child(label)
    }

    fn track_sweep_direction(tool: ActiveTool, alt_pressed: bool) -> Option<bool> {
        if tool == ActiveTool::TrackSweep {
            Some(!alt_pressed)
        } else {
            None
        }
    }

    fn timeline_time_from_mouse_x(&self, x: Pixels) -> Duration {
        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
        let window_x_px = x - px(offset_w);
        let window_x_f32 = window_x_px / px(1.0);
        let local_x_f32 = (window_x_f32 + self.scroll_offset_x).max(0.0);
        Duration::from_secs_f32((local_x_f32 / self.px_per_sec).max(0.0))
    }

    fn ensure_audio_gain_sliders(&mut self, cx: &mut Context<Self>) {
        let gains: Vec<f32> = {
            let gs = self.global.read(cx);
            gs.audio_tracks.iter().map(|track| track.gain_db).collect()
        };
        let track_len = gains.len();

        self.audio_gain_sliders.retain(|idx, _| *idx < track_len);
        self.audio_gain_slider_subs
            .retain(|idx, _| *idx < track_len);

        for (idx, gain_db) in gains.into_iter().enumerate() {
            if self.audio_gain_sliders.contains_key(&idx) {
                continue;
            }
            let slider = cx.new(|_| {
                SliderState::new()
                    .min(AUDIO_TRACK_GAIN_MIN_DB)
                    .max(AUDIO_TRACK_GAIN_MAX_DB)
                    .default_value(gain_db)
                    .step(0.1)
            });
            let global = self.global.clone();
            let sub = cx.subscribe(&slider, move |_this, _, ev, cx| {
                let SliderEvent::Change(value) = ev;
                global.update(cx, |gs, cx| {
                    let _ = gs.set_audio_track_gain_db(idx, value.start());
                    cx.notify();
                });
                cx.notify();
            });
            self.audio_gain_sliders.insert(idx, slider);
            self.audio_gain_slider_subs.insert(idx, sub);
        }
    }

    fn sweep_select_from_anchor(
        &mut self,
        track_type: TrackType,
        pivot: Duration,
        forward: bool,
        include_all_tracks: bool,
        cx: &mut Context<Self>,
    ) {
        self.global.update(cx, |gs, cx| {
            let mut clip_hits: Vec<(Duration, u64)> = Vec::new();
            let mut subtitle_hits: Vec<(Duration, u64)> = Vec::new();

            let include_clip = |clip_start: Duration, clip_end: Duration| -> bool {
                if forward {
                    clip_end > pivot
                } else {
                    clip_start < pivot
                }
            };

            let mut push_clip = |clip: &Clip| {
                if include_clip(clip.start, clip.end()) {
                    clip_hits.push((clip.start, clip.id));
                }
            };
            let mut push_subtitle = |clip: &SubtitleClip| {
                if include_clip(clip.start, clip.end()) {
                    subtitle_hits.push((clip.start, clip.id));
                }
            };

            if include_all_tracks {
                for clip in &gs.v1_clips {
                    push_clip(clip);
                }
                for track in &gs.video_tracks {
                    for clip in &track.clips {
                        push_clip(clip);
                    }
                }
                for track in &gs.audio_tracks {
                    for clip in &track.clips {
                        push_clip(clip);
                    }
                }
                for track in &gs.subtitle_tracks {
                    for clip in &track.clips {
                        push_subtitle(clip);
                    }
                }
            } else {
                match track_type {
                    TrackType::V1 => {
                        for clip in &gs.v1_clips {
                            push_clip(clip);
                        }
                    }
                    TrackType::VideoOverlay(track_index) => {
                        if let Some(track) = gs.video_tracks.get(track_index) {
                            for clip in &track.clips {
                                push_clip(clip);
                            }
                        }
                    }
                    TrackType::Audio(track_index) => {
                        if let Some(track) = gs.audio_tracks.get(track_index) {
                            for clip in &track.clips {
                                push_clip(clip);
                            }
                        }
                    }
                    TrackType::Subtitle(track_index) => {
                        if let Some(track) = gs.subtitle_tracks.get(track_index) {
                            for clip in &track.clips {
                                push_subtitle(clip);
                            }
                        }
                    }
                }
            }

            clip_hits.sort_by_key(|(start, id)| (*start, *id));
            subtitle_hits.sort_by_key(|(start, id)| (*start, *id));

            if gs.effective_v1_magnetic(false) {
                let v1_ids: HashSet<u64> = gs.v1_clips.iter().map(|clip| clip.id).collect();
                let v1_link_groups: HashSet<u64> = gs
                    .v1_clips
                    .iter()
                    .filter_map(|clip| clip.link_group_id)
                    .collect();
                let mut blocked_ids = v1_ids;
                for track in &gs.video_tracks {
                    for clip in &track.clips {
                        if clip
                            .link_group_id
                            .is_some_and(|link_group_id| v1_link_groups.contains(&link_group_id))
                        {
                            blocked_ids.insert(clip.id);
                        }
                    }
                }
                for track in &gs.audio_tracks {
                    for clip in &track.clips {
                        if clip
                            .link_group_id
                            .is_some_and(|link_group_id| v1_link_groups.contains(&link_group_id))
                        {
                            blocked_ids.insert(clip.id);
                        }
                    }
                }
                clip_hits.retain(|(_, id)| !blocked_ids.contains(id));
            }

            gs.set_playhead(pivot);
            gs.selected_layer_effect_clip_id = None;
            gs.clear_semantic_clip_selection();
            gs.selected_clip_ids = clip_hits.iter().map(|(_, id)| *id).collect();
            gs.selected_subtitle_ids = subtitle_hits.iter().map(|(_, id)| *id).collect();

            gs.selected_clip_id = if forward {
                clip_hits
                    .iter()
                    .find(|(start, _)| *start >= pivot)
                    .map(|(_, id)| *id)
                    .or_else(|| clip_hits.last().map(|(_, id)| *id))
            } else {
                clip_hits
                    .iter()
                    .rev()
                    .find(|(start, _)| *start <= pivot)
                    .map(|(_, id)| *id)
                    .or_else(|| clip_hits.first().map(|(_, id)| *id))
            };
            gs.selected_subtitle_id = if forward {
                subtitle_hits
                    .iter()
                    .find(|(start, _)| *start >= pivot)
                    .map(|(_, id)| *id)
                    .or_else(|| subtitle_hits.last().map(|(_, id)| *id))
            } else {
                subtitle_hits
                    .iter()
                    .rev()
                    .find(|(start, _)| *start <= pivot)
                    .map(|(_, id)| *id)
                    .or_else(|| subtitle_hits.first().map(|(_, id)| *id))
            };

            let clip_count = gs.selected_clip_ids.len();
            let subtitle_count = gs.selected_subtitle_ids.len();
            let scope = if include_all_tracks {
                "all tracks".to_string()
            } else {
                match track_type {
                    TrackType::V1 => "V1".to_string(),
                    TrackType::VideoOverlay(index) => format!("V{}", index.saturating_add(2)),
                    TrackType::Audio(index) => format!("A{}", index.saturating_add(1)),
                    TrackType::Subtitle(index) => format!("S{}", index.saturating_add(1)),
                }
            };
            if clip_count == 0 && subtitle_count == 0 {
                gs.ui_notice = Some(if forward {
                    format!("Sweep Forward ({scope}): no timeline items after cursor.")
                } else {
                    format!("Sweep Backward ({scope}): no timeline items before cursor.")
                });
            } else {
                gs.ui_notice = Some(format!(
                    "{} ({scope}) selected {} clip(s), {} subtitle clip(s).",
                    if forward {
                        "Sweep Forward"
                    } else {
                        "Sweep Backward"
                    },
                    clip_count,
                    subtitle_count
                ));
            }

            cx.notify();
        });
    }

    fn clip_track_and_start(gs: &GlobalState, clip_id: u64) -> Option<(TrackType, Duration)> {
        if let Some(clip) = gs.v1_clips.iter().find(|clip| clip.id == clip_id) {
            return Some((TrackType::V1, clip.start));
        }
        for (idx, track) in gs.video_tracks.iter().enumerate() {
            if let Some(clip) = track.clips.iter().find(|clip| clip.id == clip_id) {
                return Some((TrackType::VideoOverlay(idx), clip.start));
            }
        }
        for (idx, track) in gs.audio_tracks.iter().enumerate() {
            if let Some(clip) = track.clips.iter().find(|clip| clip.id == clip_id) {
                return Some((TrackType::Audio(idx), clip.start));
            }
        }
        None
    }

    fn subtitle_track_and_start(gs: &GlobalState, clip_id: u64) -> Option<(usize, Duration)> {
        for (idx, track) in gs.subtitle_tracks.iter().enumerate() {
            if let Some(clip) = track.clips.iter().find(|clip| clip.id == clip_id) {
                return Some((idx, clip.start));
            }
        }
        None
    }

    fn build_track_sweep_group_items(
        &self,
        gs: &GlobalState,
        pivot: Duration,
        forward: bool,
    ) -> Option<Vec<GroupDragItem>> {
        let include_range = |start: Duration, duration: Duration| -> bool {
            let end = start.saturating_add(duration);
            if forward { end > pivot } else { start < pivot }
        };

        let mut items = Vec::new();

        for clip in &gs.v1_clips {
            if gs.selected_clip_ids.contains(&clip.id) {
                items.push(GroupDragItem {
                    clip_id: clip.id,
                    kind: GroupDragKind::Timeline(TrackType::V1),
                    original_start: clip.start,
                });
            }
        }
        for (idx, track) in gs.audio_tracks.iter().enumerate() {
            for clip in &track.clips {
                if gs.selected_clip_ids.contains(&clip.id) {
                    items.push(GroupDragItem {
                        clip_id: clip.id,
                        kind: GroupDragKind::Timeline(TrackType::Audio(idx)),
                        original_start: clip.start,
                    });
                }
            }
        }
        for (idx, track) in gs.video_tracks.iter().enumerate() {
            for clip in &track.clips {
                if gs.selected_clip_ids.contains(&clip.id) {
                    items.push(GroupDragItem {
                        clip_id: clip.id,
                        kind: GroupDragKind::Timeline(TrackType::VideoOverlay(idx)),
                        original_start: clip.start,
                    });
                }
            }
        }
        for (idx, track) in gs.subtitle_tracks.iter().enumerate() {
            for clip in &track.clips {
                if gs.selected_subtitle_ids.contains(&clip.id) {
                    items.push(GroupDragItem {
                        clip_id: clip.id,
                        kind: GroupDragKind::Timeline(TrackType::Subtitle(idx)),
                        original_start: clip.start,
                    });
                }
            }
        }

        for layer in gs.layer_effect_clips() {
            if include_range(layer.start, layer.duration) {
                items.push(GroupDragItem {
                    clip_id: layer.id,
                    kind: GroupDragKind::LayerEffect,
                    original_start: layer.start,
                });
            }
        }

        for clip in gs.semantic_clips() {
            if include_range(clip.start, clip.duration) {
                items.push(GroupDragItem {
                    clip_id: clip.id,
                    kind: GroupDragKind::Semantic,
                    original_start: clip.start,
                });
            }
        }

        if items.is_empty() { None } else { Some(items) }
    }

    fn copy_selected_timeline_item(&mut self, cx: &mut Context<Self>) {
        let copied = {
            let gs = self.global.read(cx);
            if let Some(id) = gs.selected_layer_effect_clip_id() {
                Some(TimelineClipboardItem::LayerEffect { clip_id: id })
            } else if let Some(id) = gs.selected_semantic_clip_id() {
                Some(TimelineClipboardItem::Semantic { clip_id: id })
            } else if let Some(id) = gs.selected_subtitle_id {
                Self::subtitle_track_and_start(gs, id).map(|(track_index, _)| {
                    TimelineClipboardItem::Subtitle {
                        clip_id: id,
                        track_index,
                    }
                })
            } else if let Some(id) = gs.selected_clip_id {
                Self::clip_track_and_start(gs, id).map(|(track_type, _)| {
                    TimelineClipboardItem::Clip {
                        clip_id: id,
                        track_type,
                    }
                })
            } else {
                None
            }
        };

        self.copied_timeline_item = copied;
        self.global.update(cx, |gs, cx| {
            gs.ui_notice = Some(match copied {
                Some(TimelineClipboardItem::Clip { .. }) => "Clip copied.".to_string(),
                Some(TimelineClipboardItem::Subtitle { .. }) => "Subtitle copied.".to_string(),
                Some(TimelineClipboardItem::Semantic { .. }) => "Semantic copied.".to_string(),
                Some(TimelineClipboardItem::LayerEffect { .. }) => "Layer copied.".to_string(),
                None => "Nothing selected to copy.".to_string(),
            });
            cx.notify();
        });
    }

    fn paste_copied_timeline_item(&mut self, cx: &mut Context<Self>) {
        let Some(copied) = self.copied_timeline_item else {
            self.global.update(cx, |gs, cx| {
                gs.ui_notice = Some("Clipboard is empty.".to_string());
                cx.notify();
            });
            return;
        };

        self.global.update(cx, |gs, cx| {
            let paste_at = gs.playhead;
            let ok = match copied {
                TimelineClipboardItem::Clip {
                    clip_id,
                    track_type,
                } => gs.duplicate_timeline_clip_at(track_type, clip_id, paste_at),
                TimelineClipboardItem::Subtitle {
                    clip_id,
                    track_index,
                } => gs.duplicate_subtitle_clip_at(track_index, clip_id, paste_at),
                TimelineClipboardItem::Semantic { clip_id } => {
                    gs.duplicate_semantic_clip_at(clip_id, paste_at)
                }
                TimelineClipboardItem::LayerEffect { clip_id } => {
                    gs.duplicate_layer_effect_clip_at(clip_id, paste_at)
                }
            };
            if ok {
                gs.ui_notice = Some("Pasted copied item.".to_string());
            } else {
                gs.ui_notice = Some(
                    "Paste failed. Source item may have been removed or moved to another track."
                        .to_string(),
                );
            }
            cx.notify();
        });
    }

    fn arm_track_sweep_pending_drag(
        &mut self,
        mouse_x: Pixels,
        pivot: Duration,
        forward: bool,
        cx: &mut Context<Self>,
    ) {
        let gs = self.global.read(cx);
        let Some(items) = self.build_track_sweep_group_items(gs, pivot, forward) else {
            self.drag_state = DragState::None;
            self.is_scrubbing = false;
            return;
        };

        let (anchor_clip_id, anchor_track_type) = if let Some(anchor_id) = gs.selected_clip_id {
            if let Some((track_type, _)) = Self::clip_track_and_start(gs, anchor_id) {
                (anchor_id, track_type)
            } else {
                (0, TrackType::V1)
            }
        } else if let Some(anchor_id) = gs.selected_subtitle_id {
            if let Some((track_index, _)) = Self::subtitle_track_and_start(gs, anchor_id) {
                (anchor_id, TrackType::Subtitle(track_index))
            } else {
                (0, TrackType::V1)
            }
        } else {
            (0, TrackType::V1)
        };

        self.drag_state = DragState::Pending {
            start_x: mouse_x,
            clip_id: anchor_clip_id,
            track_type: anchor_track_type,
            original_start: pivot,
            group_items: Some(items),
        };
        self.is_scrubbing = false;
    }

    fn build_ruler(
        total: Duration,
        px_per_sec: f32,
        visible_start_sec: f32,
        visible_end_sec: f32,
    ) -> gpui::Div {
        let total_ms = (total.as_millis() as u64).max(1);
        let (minor_raw, major_raw, label_raw) = get_dynamic_ruler_steps(px_per_sec.max(0.01));
        let minor_ms = minor_raw.max(1);
        let major_ms = major_raw.max(1);
        let label_ms = label_raw.max(1);

        let view_start_ms = (visible_start_sec.max(0.0) * 1000.0).floor() as u64;
        let view_end_ms = (visible_end_sec.max(visible_start_sec) * 1000.0).ceil() as u64;
        let capped_end_ms = view_end_ms.min(total_ms);
        let align_down = |value: u64, step: u64| (value / step) * step;

        let mut ruler = div()
            .h(px(RULER_H))
            .w_full()
            .flex_shrink_0()
            .relative()
            .bg(black().opacity(0.25))
            .border_b_1()
            .border_color(white().opacity(0.08));

        let mut label_tick_ms = align_down(view_start_ms, label_ms);
        while label_tick_ms <= capped_end_ms {
            let x = (label_tick_ms as f32 / 1000.0) * px_per_sec;
            ruler = ruler.child(
                div()
                    .absolute()
                    .left(px(x))
                    .top(px(2.0))
                    .text_xs()
                    .text_color(white().opacity(0.55))
                    .child(fmt_mmss_millis(Duration::from_millis(label_tick_ms))),
            );
            label_tick_ms = match label_tick_ms.checked_add(label_ms) {
                Some(next) => next,
                None => break,
            };
        }

        let mut tick_ms = align_down(view_start_ms, minor_ms);
        while tick_ms <= capped_end_ms {
            let is_label = tick_ms % label_ms == 0;
            let is_major = tick_ms % major_ms == 0;
            if is_label || is_major || (px_per_sec * (minor_ms as f32 / 1000.0) > 5.0) {
                let tick_h = if is_label {
                    14.0
                } else if is_major {
                    9.0
                } else {
                    5.0
                };
                let alpha = if is_label {
                    0.4
                } else if is_major {
                    0.25
                } else {
                    0.1
                };
                let x = (tick_ms as f32 / 1000.0) * px_per_sec;
                ruler = ruler.child(
                    div()
                        .absolute()
                        .left(px(x))
                        .bottom(px(0.0))
                        .w(px(1.0))
                        .h(px(tick_h))
                        .bg(white().opacity(alpha)),
                );
            }
            tick_ms = match tick_ms.checked_add(minor_ms) {
                Some(next) => next,
                None => break,
            };
        }
        ruler
    }

    fn clip_waveform_peak_at_playhead(
        gs: &GlobalState,
        clip: &Clip,
        playhead: Duration,
    ) -> Option<f32> {
        if clip.duration <= Duration::ZERO || playhead < clip.start || playhead >= clip.end() {
            return None;
        }
        let media_secs = clip.media_duration.as_secs_f32();
        if media_secs <= 0.001 {
            return None;
        }

        let local = playhead.saturating_sub(clip.start);
        let source_secs = (clip.source_in + local)
            .as_secs_f32()
            .clamp(0.0, media_secs);
        let bucket_count = waveform_bucket_count_for_media_duration(clip.media_duration);
        let key = waveform_key(Path::new(&clip.file_path), bucket_count);
        let peaks = gs.waveform_entries.get(&key)?.peaks.as_ref()?;
        if peaks.is_empty() {
            return None;
        }

        let ratio = (source_secs / media_secs).clamp(0.0, 1.0);
        let idx = ((ratio * (peaks.len().saturating_sub(1) as f32)).round() as usize)
            .min(peaks.len().saturating_sub(1));
        Some(peaks[idx].clamp(0.0, 1.0))
    }

    fn audio_track_db_label(gs: &GlobalState, track_index: usize, playhead: Duration) -> String {
        let is_muted = gs
            .track_mute
            .get(&format!("audio:{track_index}"))
            .copied()
            .unwrap_or(false);
        if is_muted {
            return "-inf dB".to_string();
        }

        let Some(track) = gs.audio_tracks.get(track_index) else {
            return "--.- dB".to_string();
        };
        let gain_db = track
            .gain_db
            .clamp(AUDIO_TRACK_GAIN_MIN_DB, AUDIO_TRACK_GAIN_MAX_DB);

        let mut active_clip_count = 0usize;
        let mut waveform_ready = false;
        let mut peak = 0.0_f32;
        for clip in &track.clips {
            if playhead >= clip.start && playhead < clip.end() {
                active_clip_count += 1;
                if let Some(amp) = Self::clip_waveform_peak_at_playhead(gs, clip, playhead) {
                    waveform_ready = true;
                    peak = peak.max(amp);
                }
            }
        }

        if active_clip_count == 0 {
            return "-inf dB".to_string();
        }
        if !waveform_ready {
            return "--.- dB".to_string();
        }

        let db = if peak <= 0.000_01 {
            -60.0
        } else {
            (20.0 * peak.log10()) + gain_db
        };
        format!("{:.1} dB", db.clamp(-96.0, 12.0))
    }

    fn all_tracks_db_summary(gs: &GlobalState, playhead: Duration) -> String {
        if gs.audio_tracks.is_empty() {
            return "Audio dB  (no audio tracks)".to_string();
        }
        let mut parts = Vec::new();
        for (idx, track) in gs.audio_tracks.iter().enumerate() {
            let db = Self::audio_track_db_label(gs, idx, playhead);
            parts.push(format!("{}: {db}", track.name));
        }

        format!("Audio dB  {}", parts.join("  |  "))
    }

    fn track_header_row(
        track: &TrackUi,
        cx: &mut Context<TimelinePanel>,
        global: &Entity<GlobalState>,
    ) -> gpui::Div {
        let mute_key = match track.kind {
            TrackHeaderKind::V1 => Some("v1".to_string()),
            TrackHeaderKind::Audio(idx) => Some(format!("audio:{idx}")),
            TrackHeaderKind::VideoOverlay(idx) => Some(format!("video:{idx}")),
            TrackHeaderKind::Subtitle(idx) => Some(format!("subtitle:{idx}")),
            TrackHeaderKind::Semantic => None,
        };
        let is_muted = mute_key
            .as_ref()
            .and_then(|key| global.read(cx).track_mute.get(key).copied())
            .unwrap_or(false);

        let mut row = div()
            .h(px(LANE_H))
            .px_2()
            .flex()
            .items_center()
            .gap_2()
            .bg(white().opacity(0.02))
            .border_b_1()
            .border_color(white().opacity(0.06))
            .child(
                div()
                    .w(px(16.0))
                    .text_xs()
                    .text_color(white().opacity(0.45))
                    .child("🔒"),
            )
            .child(
                div()
                    .w(px(34.0))
                    .h(px(18.0))
                    .rounded_md()
                    .border_1()
                    .border_color(white().opacity(0.08))
                    .bg(white().opacity(0.02))
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_xs()
                    .text_color(white().opacity(0.8))
                    .child(track.name.clone()),
            );

        let mut right = div().ml_auto().flex().items_center().gap_2();

        if !matches!(track.kind, TrackHeaderKind::V1 | TrackHeaderKind::Semantic) {
            let kind = track.kind;
            let global_for_delete = global.clone();
            right = right.child(
                div()
                    .w(px(18.0))
                    .h(px(18.0))
                    .rounded_sm()
                    .bg(white().opacity(0.08))
                    .text_color(white().opacity(0.75))
                    .text_xs()
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .child("✕")
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |_this, _, _, cx| {
                            global_for_delete.update(cx, |gs, cx| {
                                match kind {
                                    TrackHeaderKind::Audio(idx) => {
                                        gs.delete_audio_track(idx);
                                    }
                                    TrackHeaderKind::VideoOverlay(idx) => {
                                        gs.delete_video_track(idx);
                                    }
                                    TrackHeaderKind::Subtitle(idx) => {
                                        gs.delete_subtitle_track(idx);
                                    }
                                    TrackHeaderKind::Semantic => {}
                                    TrackHeaderKind::V1 => {}
                                }
                                cx.notify();
                            });
                        }),
                    ),
            );
        }

        if let Some(mute_key) = mute_key {
            let global_for_toggle_mute = global.clone();
            let track_name = track.name.clone();
            right = right.child(
                div()
                    .w(px(18.0))
                    .h(px(18.0))
                    .rounded_sm()
                    .bg(white().opacity(if is_muted { 0.18 } else { 0.08 }))
                    .text_color(white().opacity(if is_muted { 0.95 } else { 0.75 }))
                    .text_xs()
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .child(if is_muted { "⊘" } else { "👁" })
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |_this, _, _, cx| {
                            let key = mute_key.clone();
                            global_for_toggle_mute.update(cx, |gs, cx| {
                                let current = gs.track_mute.get(&key).copied().unwrap_or(false);
                                let next = !current;
                                if next {
                                    gs.track_mute.insert(key.clone(), true);
                                    gs.ui_notice = Some(format!("{track_name} muted."));
                                } else {
                                    gs.track_mute.remove(&key);
                                    gs.ui_notice = Some(format!("{track_name} unmuted."));
                                }
                                cx.notify();
                            });
                        }),
                    ),
            );
        } else {
            right = right.child(div().text_xs().text_color(white().opacity(0.35)).child("👁"));
        }
        row = row.child(right);
        row
    }
    fn start_scrubbing(
        &mut self,
        evt: &MouseDownEvent,
        win: &mut Window, // <--- This must be mutable
        cx: &mut Context<Self>,
        global_entity: &Entity<GlobalState>,
    ) {
        self.is_scrubbing = true;
        self.clip_link_menu = None;
        self.timeline_clip_menu = None;
        self.layer_clip_menu = None;
        self.subtitle_clip_menu = None;
        self.semantic_clip_menu = None;
        self.drag_state = DragState::Scrubbing;
        cx.focus_self(win); // Now this works because win is mutable

        // 1. Calculate Time
        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
        let raw_x = evt.position.x;
        let window_x_px = raw_x - px(offset_w);
        // Calculate relative X
        let local_x_f32 = (window_x_px / px(1.0) + self.scroll_offset_x).max(0.0);
        let t = Duration::from_secs_f32((local_x_f32 / self.px_per_sec).max(0.0));

        // 2. Update Global State
        global_entity.update(cx, |gs, cx| {
            gs.is_playing = false;
            gs.set_playhead(t);
            gs.clear_layer_effect_clip_selection();
            gs.clear_semantic_clip_selection();

            // Razor Logic (Optional)
            if gs.active_tool == ActiveTool::Razor {
                let _ = gs.razor_v1_at_playhead();
            }

            cx.notify();
        });
    }

    fn start_marquee(&mut self, evt: &MouseDownEvent, win: &mut Window, cx: &mut Context<Self>) {
        self.is_scrubbing = false;
        self.clip_link_menu = None;
        self.timeline_clip_menu = None;
        self.layer_clip_menu = None;
        self.subtitle_clip_menu = None;
        self.semantic_clip_menu = None;
        self.drag_state = DragState::Marquee {
            start: evt.position,
            current: evt.position,
        };
        cx.focus_self(win);
        cx.notify();
    }

    fn window_to_panel_point(&self, position: gpui::Point<Pixels>, window: &Window) -> (f32, f32) {
        // Translate window-space cursor coordinates into timeline panel-space coordinates.
        let win_h = window.viewport_size().height / px(1.0);
        let panel_top = (win_h - TIMELINE_PANEL_H).max(0.0);
        let x = (position.x / px(1.0) - APP_NAV_W).max(0.0);
        let y = (position.y / px(1.0) - panel_top).max(0.0);
        (x, y)
    }

    fn marquee_bounds_in_panel(&self, window: &Window) -> Option<gpui::Bounds<Pixels>> {
        let DragState::Marquee { start, current } = &self.drag_state else {
            return None;
        };
        let win_h = window.viewport_size().height / px(1.0);
        let panel_top = (win_h - TIMELINE_PANEL_H).max(0.0);

        let start_x = start.x / px(1.0) - APP_NAV_W;
        let start_y = start.y / px(1.0) - panel_top;
        let current_x = current.x / px(1.0) - APP_NAV_W;
        let current_y = current.y / px(1.0) - panel_top;

        let left = start_x.min(current_x);
        let right = start_x.max(current_x);
        let top = start_y.min(current_y);
        let bottom = start_y.max(current_y);

        Some(gpui::bounds(
            gpui::point(px(left), px(top)),
            gpui::size(px((right - left).max(0.0)), px((bottom - top).max(0.0))),
        ))
    }

    fn collect_marquee_selection(
        &self,
        rect: gpui::Bounds<Pixels>,
        gs: &GlobalState,
    ) -> (Vec<u64>, Vec<u64>) {
        let rect_left = rect.origin.x / px(1.0);
        let rect_top = rect.origin.y / px(1.0);
        let rect_right = rect_left + rect.size.width / px(1.0);
        let rect_bottom = rect_top + rect.size.height / px(1.0);

        let content_left = LEFT_TOOL_W + TRACK_LIST_W;
        let tracks_top = TIMELINE_HEADER_H + RULER_H - self.scroll_offset_y;

        let mut selected_clip_ids = Vec::new();
        let mut selected_subtitle_ids = Vec::new();

        let mut lane_index = 0usize;

        for track in gs.subtitle_tracks.iter().rev() {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            for clip in &track.clips {
                let clip_left =
                    content_left + dur_to_px(clip.start, self.px_per_sec) - self.scroll_offset_x;
                let clip_width = dur_to_px(clip.duration, self.px_per_sec).max(24.0);
                let clip_right = clip_left + clip_width;
                let clip_top = lane_top;
                let clip_bottom = lane_top + LANE_H;
                if rect_left <= clip_right
                    && rect_right >= clip_left
                    && rect_top <= clip_bottom
                    && rect_bottom >= clip_top
                {
                    selected_subtitle_ids.push(clip.id);
                }
            }
            lane_index += 1;
        }

        for track in gs.video_tracks.iter().rev() {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            for clip in &track.clips {
                let clip_left =
                    content_left + dur_to_px(clip.start, self.px_per_sec) - self.scroll_offset_x;
                let clip_width = dur_to_px(clip.duration, self.px_per_sec).max(2.0);
                let clip_right = clip_left + clip_width;
                let clip_top = lane_top;
                let clip_bottom = lane_top + LANE_H;
                if rect_left <= clip_right
                    && rect_right >= clip_left
                    && rect_top <= clip_bottom
                    && rect_bottom >= clip_top
                {
                    selected_clip_ids.push(clip.id);
                }
            }
            lane_index += 1;
        }

        {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            for clip in &gs.v1_clips {
                let clip_left =
                    content_left + dur_to_px(clip.start, self.px_per_sec) - self.scroll_offset_x;
                let clip_width = dur_to_px(clip.duration, self.px_per_sec).max(2.0);
                let clip_right = clip_left + clip_width;
                let clip_top = lane_top;
                let clip_bottom = lane_top + LANE_H;
                if rect_left <= clip_right
                    && rect_right >= clip_left
                    && rect_top <= clip_bottom
                    && rect_bottom >= clip_top
                {
                    selected_clip_ids.push(clip.id);
                }
            }
            lane_index += 1;
        }

        for track in &gs.audio_tracks {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            for clip in &track.clips {
                let clip_left =
                    content_left + dur_to_px(clip.start, self.px_per_sec) - self.scroll_offset_x;
                let clip_width = dur_to_px(clip.duration, self.px_per_sec).max(2.0);
                let clip_right = clip_left + clip_width;
                let clip_top = lane_top;
                let clip_bottom = lane_top + LANE_H;
                if rect_left <= clip_right
                    && rect_right >= clip_left
                    && rect_top <= clip_bottom
                    && rect_bottom >= clip_top
                {
                    selected_clip_ids.push(clip.id);
                }
            }
            lane_index += 1;
        }

        (selected_clip_ids, selected_subtitle_ids)
    }

    fn clip_at_point(
        &self,
        gs: &GlobalState,
        window: &Window,
        pos: gpui::Point<Pixels>,
    ) -> Option<(u64, TrackType)> {
        let win_h = window.viewport_size().height / px(1.0);
        let panel_top = (win_h - TIMELINE_PANEL_H).max(0.0);

        let local_x = pos.x / px(1.0) - APP_NAV_W;
        let local_y = pos.y / px(1.0) - panel_top;

        let content_left = LEFT_TOOL_W + TRACK_LIST_W;
        let timeline_x = local_x - content_left + self.scroll_offset_x;
        if timeline_x < 0.0 {
            return None;
        }

        let t = Duration::from_secs_f32((timeline_x / self.px_per_sec).max(0.0));
        let tracks_top = TIMELINE_HEADER_H + RULER_H - self.scroll_offset_y;

        let mut lane_index = 0usize;

        for _idx in (0..gs.subtitle_tracks.len()).rev() {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            if local_y >= lane_top && local_y <= lane_top + LANE_H {
                return None;
            }
            lane_index += 1;
        }

        if self.show_semantic_lane {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            if local_y >= lane_top && local_y <= lane_top + LANE_H {
                return None;
            }
            lane_index += 1;
        }

        for idx in (0..gs.video_tracks.len()).rev() {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            if local_y >= lane_top && local_y <= lane_top + LANE_H {
                let track = &gs.video_tracks[idx];
                if let Some(clip) = track
                    .clips
                    .iter()
                    .find(|c| t >= c.start && t < (c.start + c.duration))
                {
                    return Some((clip.id, TrackType::VideoOverlay(idx)));
                }
                return None;
            }
            lane_index += 1;
        }

        {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            if local_y >= lane_top && local_y <= lane_top + LANE_H {
                if let Some(clip) = gs
                    .v1_clips
                    .iter()
                    .find(|c| t >= c.start && t < (c.start + c.duration))
                {
                    return Some((clip.id, TrackType::V1));
                }
                return None;
            }
            lane_index += 1;
        }

        for _idx in 0..gs.audio_tracks.len() {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            if local_y >= lane_top && local_y <= lane_top + LANE_H {
                return None;
            }
            lane_index += 1;
        }

        None
    }

    fn layer_clip_at_point(
        &self,
        gs: &GlobalState,
        window: &Window,
        pos: gpui::Point<Pixels>,
    ) -> Option<u64> {
        let win_h = window.viewport_size().height / px(1.0);
        let panel_top = (win_h - TIMELINE_PANEL_H).max(0.0);

        let local_x = pos.x / px(1.0) - APP_NAV_W;
        let local_y = pos.y / px(1.0) - panel_top;

        let content_left = LEFT_TOOL_W + TRACK_LIST_W;
        let timeline_x = local_x - content_left + self.scroll_offset_x;
        if timeline_x < 0.0 {
            return None;
        }

        let t = Duration::from_secs_f32((timeline_x / self.px_per_sec).max(0.0));
        let tracks_top = TIMELINE_HEADER_H + RULER_H - self.scroll_offset_y;
        let mut lane_index = 0usize;

        for _idx in (0..gs.subtitle_tracks.len()).rev() {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            if local_y >= lane_top && local_y <= lane_top + LANE_H {
                return None;
            }
            lane_index += 1;
        }

        if self.show_semantic_lane {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            if local_y >= lane_top && local_y <= lane_top + LANE_H {
                return None;
            }
            lane_index += 1;
        }

        for idx in (0..gs.video_tracks.len()).rev() {
            let lane_top = tracks_top + lane_index as f32 * LANE_H;
            if local_y >= lane_top && local_y <= lane_top + LANE_H {
                return gs
                    .layer_effect_clips()
                    .iter()
                    .rev()
                    .find(|layer| {
                        layer.track_index == idx
                            && t >= layer.start
                            && t < layer.start.saturating_add(layer.duration)
                    })
                    .map(|layer| layer.id);
            }
            lane_index += 1;
        }

        None
    }

    fn media_drop_target_at_point(
        &self,
        gs: &GlobalState,
        window: &Window,
        pos: gpui::Point<Pixels>,
    ) -> Option<(TrackType, Duration)> {
        // Convert pointer to timeline time and lane using the same visual layout math.
        let win_h = window.viewport_size().height / px(1.0);
        let panel_top = (win_h - TIMELINE_PANEL_H).max(0.0);
        let local_x = pos.x / px(1.0);
        let local_y = pos.y / px(1.0) - panel_top;
        let content_left = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
        let timeline_x = local_x - content_left + self.scroll_offset_x;
        if timeline_x < 0.0 {
            return None;
        }

        let t = Duration::from_secs_f32((timeline_x / self.px_per_sec).max(0.0));
        let tracks_top = TIMELINE_HEADER_H + RULER_H - self.scroll_offset_y;
        let lane_y = local_y - tracks_top;
        if lane_y < 0.0 {
            return None;
        }

        let lane_index = (lane_y / LANE_H).floor() as usize;
        let subtitle_count = gs.subtitle_tracks.len();
        if lane_index < subtitle_count {
            let idx = subtitle_count.saturating_sub(1).saturating_sub(lane_index);
            return Some((TrackType::Subtitle(idx), t));
        }

        let semantic_offset = if self.show_semantic_lane { 1 } else { 0 };
        if self.show_semantic_lane && lane_index == subtitle_count {
            return None;
        }

        let lane_after_subtitle = lane_index.saturating_sub(subtitle_count + semantic_offset);
        let video_count = gs.video_tracks.len();
        if lane_after_subtitle < video_count {
            let idx = video_count
                .saturating_sub(1)
                .saturating_sub(lane_after_subtitle);
            return Some((TrackType::VideoOverlay(idx), t));
        }

        if lane_after_subtitle == video_count {
            return Some((TrackType::V1, t));
        }

        let lane_after_v1 = lane_after_subtitle - video_count - 1;
        if lane_after_v1 < gs.audio_tracks.len() {
            return Some((TrackType::Audio(lane_after_v1), t));
        }

        None
    }

    fn build_group_clip_items(
        &self,
        gs: &GlobalState,
        anchor_id: u64,
    ) -> Option<Vec<GroupDragItem>> {
        if gs.selected_clip_ids.len() <= 1 || !gs.selected_clip_ids.contains(&anchor_id) {
            return None;
        }

        let mut items = Vec::new();
        for clip in &gs.v1_clips {
            if gs.selected_clip_ids.contains(&clip.id) {
                items.push(GroupDragItem {
                    clip_id: clip.id,
                    kind: GroupDragKind::Timeline(TrackType::V1),
                    original_start: clip.start,
                });
            }
        }
        for (idx, track) in gs.audio_tracks.iter().enumerate() {
            for clip in &track.clips {
                if gs.selected_clip_ids.contains(&clip.id) {
                    items.push(GroupDragItem {
                        clip_id: clip.id,
                        kind: GroupDragKind::Timeline(TrackType::Audio(idx)),
                        original_start: clip.start,
                    });
                }
            }
        }
        for (idx, track) in gs.video_tracks.iter().enumerate() {
            for clip in &track.clips {
                if gs.selected_clip_ids.contains(&clip.id) {
                    items.push(GroupDragItem {
                        clip_id: clip.id,
                        kind: GroupDragKind::Timeline(TrackType::VideoOverlay(idx)),
                        original_start: clip.start,
                    });
                }
            }
        }

        if items.len() > 1 { Some(items) } else { None }
    }

    fn build_group_subtitle_items(
        &self,
        gs: &GlobalState,
        anchor_id: u64,
    ) -> Option<Vec<GroupDragItem>> {
        if gs.selected_subtitle_ids.len() <= 1 || !gs.selected_subtitle_ids.contains(&anchor_id) {
            return None;
        }

        let mut items = Vec::new();
        for (idx, track) in gs.subtitle_tracks.iter().enumerate() {
            for clip in &track.clips {
                if gs.selected_subtitle_ids.contains(&clip.id) {
                    items.push(GroupDragItem {
                        clip_id: clip.id,
                        kind: GroupDragKind::Timeline(TrackType::Subtitle(idx)),
                        original_start: clip.start,
                    });
                }
            }
        }

        if items.len() > 1 { Some(items) } else { None }
    }

    fn clip_bar_ui(
        clip: &Clip,
        selected: bool,
        px_per_sec: f32,
        is_audio: bool,
        show_clip_text: bool,
        draw_inline_waveform: bool,
        waveform: Option<Arc<Vec<f32>>>,
        waveform_min_samples: usize,
        waveform_max_samples: usize,
        waveform_detail_level: AudioWaveformDetailLevel,
    ) -> gpui::Div {
        let w = dur_to_px(clip.duration, px_per_sec).max(2.0);
        let show_content = show_clip_text && w > 20.0;

        let (bg, border) = if is_audio {
            if selected {
                (rgb(0x86efac), rgb(0x4ade80))
            } else {
                (rgba(0x86efaccc), rgba(0xffffff1f))
            }
        } else if selected {
            (rgb(0xfef08a), rgb(0xfacc15))
        } else {
            (rgba(0xffffffcc), rgba(0xffffff1f))
        };

        let mut container = div()
            .h(px(22.0))
            .w(px(w))
            .rounded_md()
            .border_1()
            .border_color(border)
            .bg(bg)
            .relative()
            .px_1()
            .flex()
            .items_center()
            .gap_1()
            .text_xs()
            .text_color(rgb(0x111111))
            .overflow_hidden();

        if is_audio && draw_inline_waveform {
            let inner_w = (w - 4.0).max(4.0);
            let inner_h = 18.0_f32;
            let mid_y = inner_h * 0.5;
            let half_h = (inner_h * 0.5 - 1.0).max(1.0);
            match waveform_detail_level {
                AudioWaveformDetailLevel::Extreme => {
                    // Extreme mode keeps node count near-constant per clip.
                    // It intentionally draws a compact envelope block instead of many bars.
                    let fallback_amp_norm = 0.56_f32;
                    let amp_norm = waveform
                        .as_ref()
                        .and_then(|peaks| summarize_waveform_for_clip(peaks.as_slice(), clip))
                        .unwrap_or(fallback_amp_norm);
                    let amp = (amp_norm.clamp(0.0, 1.0) * half_h).max(1.0);
                    let top = (mid_y - amp).max(0.0);
                    let h = (amp * 2.0).min(inner_h);
                    let waveform_layer = div()
                        .absolute()
                        .left(px(2.0))
                        .top(px(2.0))
                        .h(px(inner_h))
                        .w(px(inner_w))
                        .overflow_hidden()
                        .child(
                            div()
                                .absolute()
                                .left(px(0.0))
                                .top(px(mid_y - 0.5))
                                .w(px(inner_w))
                                .h(px(1.0))
                                .bg(rgba(0x14532d7a)),
                        )
                        .child(
                            div()
                                .absolute()
                                .left(px(0.0))
                                .top(px(top))
                                .w(px(inner_w))
                                .h(px(h))
                                .bg(rgba(0x14532da6)),
                        );
                    container = container.child(waveform_layer);
                }
                AudioWaveformDetailLevel::Normal => {
                    if let Some(peaks) = waveform {
                        let bars = sample_waveform_for_clip(
                            peaks.as_slice(),
                            clip,
                            w,
                            waveform_min_samples,
                            waveform_max_samples,
                        );
                        if !bars.is_empty() {
                            let bar_w = (inner_w / bars.len() as f32).max(0.5);
                            let mut waveform_layer = div()
                                .absolute()
                                .left(px(2.0))
                                .top(px(2.0))
                                .h(px(inner_h))
                                .w(px(inner_w))
                                .overflow_hidden();

                            // Draw a mirrored filled envelope for a DaVinci-like waveform look.
                            waveform_layer = waveform_layer.child(
                                div()
                                    .absolute()
                                    .left(px(0.0))
                                    .top(px(mid_y - 0.5))
                                    .w(px(inner_w))
                                    .h(px(1.0))
                                    .bg(rgba(0x14532d80)),
                            );

                            for (idx, value) in bars.into_iter().enumerate() {
                                let x0 = idx as f32 * bar_w;
                                let x1 = ((idx + 1) as f32 * bar_w).min(inner_w);
                                let draw_left = x0.floor();
                                let draw_w = (x1.ceil() - draw_left).max(1.0);
                                let amp = (value.clamp(0.0, 1.0) * half_h).max(0.5);
                                let top = (mid_y - amp).max(0.0);
                                let h = (amp * 2.0).min(inner_h);
                                waveform_layer = waveform_layer.child(
                                    div()
                                        .absolute()
                                        .left(px(draw_left))
                                        .top(px(top))
                                        .w(px(draw_w))
                                        .h(px(h))
                                        .bg(rgba(0x14532dbd)),
                                );
                            }
                            container = container.child(waveform_layer);
                        }
                    }
                }
            }
        }

        if show_content {
            container = container
                .child(if is_audio { "🎵" } else { "🟨" })
                .child(clip.label.to_string());
        }

        container
    }

    fn audio_cluster_bar_ui(count: usize, width: f32) -> gpui::Div {
        let w = width.max(2.0);
        let mut node = div()
            .h(px(22.0))
            .w(px(w))
            .rounded_md()
            .border_1()
            .border_color(rgba(0xffffff22))
            .bg(rgba(0x86efac6e))
            .overflow_hidden()
            .px_1()
            .flex()
            .items_center()
            .gap_1()
            .text_xs()
            .text_color(rgb(0x0f2a1c));
        if w > 30.0 {
            node = node.child("🎵");
        }
        if w > 54.0 {
            node = node.child(format!("x{count}"));
        }
        node
    }

    fn layer_effect_clip_ui(
        layer_clip: crate::core::global_state::LayerEffectClip,
        selected: bool,
        px_per_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
        cx: &mut Context<TimelinePanel>,
        active_tool: ActiveTool,
        global_entity: Entity<GlobalState>,
    ) -> gpui::Div {
        let t_start = Duration::from_secs_f32(window_start_sec.max(0.0));
        let t_end = Duration::from_secs_f32(window_end_sec.max(window_start_sec + 0.001));
        if layer_clip.duration <= Duration::ZERO {
            return div();
        }
        let clip_end = layer_clip.start + layer_clip.duration;
        if layer_clip.start > t_end || clip_end < t_start {
            return div();
        }

        let left = dur_to_px(layer_clip.start, px_per_sec);
        // Keep visual width close to real duration to avoid UI-vs-runtime mismatch.
        let w = dur_to_px(layer_clip.duration, px_per_sec).max(2.0);
        let label = if w > 96.0 {
            "LAYER FX"
        } else if w > 40.0 {
            "LAYER"
        } else {
            ""
        };

        let (bg, border, text_color) = if selected {
            (rgba(0x22c55ecc), rgb(0x16a34a), rgb(0x052e16))
        } else {
            (rgba(0x22c55e66), rgba(0xffffff2a), rgb(0xf0fdf4))
        };
        let clip_id = layer_clip.id;
        let clip_start = layer_clip.start;
        let clip_duration = layer_clip.duration;
        let layer_track_type = TrackType::VideoOverlay(layer_clip.track_index);
        let global_for_layer_select_left = global_entity.clone();
        let global_for_layer_select_right = global_entity.clone();
        let global_for_layer_resize = global_entity.clone();

        div()
            .absolute()
            .left(px(left))
            .top(px(2.0))
            .h(px(LANE_H - 4.0))
            .w(px(w))
            .rounded_sm()
            .border_1()
            .border_color(border)
            .bg(bg)
            .px_1()
            .cursor_pointer()
            .overflow_hidden()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, evt: &MouseDownEvent, _win, cx| {
                    cx.stop_propagation();
                    let mut consumed_pending_mode = false;
                    global_for_layer_select_left.update(cx, |gs, cx| {
                        if let Some(pending_transition) = gs.pending_transition {
                            let _ = pending_transition;
                            gs.clear_transition_drag();
                            gs.ui_notice = Some(
                                "Layer FX no longer supports Dissolve. Use Layer FX tab Fade In / Fade Out."
                                    .to_string(),
                            );
                            gs.select_layer_effect_clip(clip_id);
                            consumed_pending_mode = true;
                            cx.notify();
                            return;
                        }
                        gs.select_layer_effect_clip(clip_id);
                        cx.notify();
                    });
                    if consumed_pending_mode {
                        this.clip_link_menu = None;
                        this.timeline_clip_menu = None;
                        this.layer_clip_menu = None;
                        this.subtitle_clip_menu = None;
                        this.semantic_clip_menu = None;
                        this.drag_state = DragState::None;
                        cx.notify();
                        return;
                    }
                    if let Some(forward) =
                        Self::track_sweep_direction(active_tool, evt.modifiers.alt)
                    {
                        let pivot = this.timeline_time_from_mouse_x(evt.position.x);
                        this.sweep_select_from_anchor(layer_track_type, pivot, forward, true, cx);
                        this.arm_track_sweep_pending_drag(evt.position.x, pivot, forward, cx);
                        return;
                    }
                    this.clip_link_menu = None;
                    this.timeline_clip_menu = None;
                    this.layer_clip_menu = None;
                    this.subtitle_clip_menu = None;
                    this.semantic_clip_menu = None;
                    this.drag_state = DragState::LayerPending {
                        start_x: evt.position.x,
                        clip_id,
                        original_start: clip_start,
                    };
                    cx.notify();
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                    cx.stop_propagation();
                    global_for_layer_select_right.update(cx, |gs, cx| {
                        gs.select_layer_effect_clip(clip_id);
                        cx.notify();
                    });
                    let (menu_x, menu_y) = this.window_to_panel_point(evt.position, win);
                    this.clip_link_menu = None;
                    this.timeline_clip_menu = None;
                    this.layer_clip_menu = Some(LayerClipContextMenu {
                        x: menu_x,
                        y: menu_y,
                        clip_id,
                    });
                    this.subtitle_clip_menu = None;
                    this.semantic_clip_menu = None;
                    cx.notify();
                }),
            )
            .child(
                div()
                    .h_full()
                    .w_full()
                    .text_xs()
                    .text_color(text_color)
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(label),
            )
            .child(
                div()
                    .absolute()
                    .right(px(0.0))
                    .top(px(0.0))
                    .h_full()
                    .w(px(10.0))
                    .bg(rgba(0xffffff00))
                    .cursor_col_resize()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                            cx.stop_propagation();
                            cx.focus_self(win);
                            global_for_layer_resize.update(cx, |gs, cx| {
                                gs.save_for_undo();
                                gs.select_layer_effect_clip(clip_id);
                                cx.notify();
                            });
                            this.clip_link_menu = None;
                            this.timeline_clip_menu = None;
                            this.layer_clip_menu = None;
                            this.subtitle_clip_menu = None;
                            this.semantic_clip_menu = None;
                            this.drag_state = DragState::LayerResizing {
                                clip_id,
                                start_x: evt.position.x,
                                original_duration: clip_duration,
                            };
                            cx.notify();
                        }),
                    ),
            )
    }

    fn layer_effect_clip_ui_low_load(
        layer_clip: crate::core::global_state::LayerEffectClip,
        selected: bool,
        px_per_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
    ) -> gpui::Div {
        let t_start = Duration::from_secs_f32(window_start_sec.max(0.0));
        let t_end = Duration::from_secs_f32(window_end_sec.max(window_start_sec + 0.001));
        if layer_clip.duration <= Duration::ZERO {
            return div();
        }
        let clip_end = layer_clip.start + layer_clip.duration;
        if layer_clip.start > t_end || clip_end < t_start {
            return div();
        }

        let left = dur_to_px(layer_clip.start, px_per_sec);
        let w = dur_to_px(layer_clip.duration, px_per_sec).max(2.0);
        let label = if w > 54.0 { "FX" } else { "" };
        let (bg, border, text_color) = if selected {
            (rgba(0x22c55ecc), rgb(0x16a34a), rgb(0x052e16))
        } else {
            (rgba(0x22c55e85), rgba(0xffffff22), rgba(0xf0fdf4f2))
        };

        div()
            .absolute()
            .left(px(left))
            .top(px(3.0))
            .h(px((LANE_H - 6.0).max(2.0)))
            .w(px(w))
            .rounded_sm()
            .border_1()
            .border_color(border)
            .bg(bg)
            .overflow_hidden()
            .child(
                div()
                    .h_full()
                    .w_full()
                    .text_xs()
                    .text_color(text_color)
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(label),
            )
    }

    fn semantic_clip_ui(
        clip: &SemanticClip,
        selected: bool,
        px_per_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
        cx: &mut Context<TimelinePanel>,
        active_tool: ActiveTool,
        global_entity: Entity<GlobalState>,
    ) -> gpui::Div {
        let t_start = Duration::from_secs_f32(window_start_sec.max(0.0));
        let t_end = Duration::from_secs_f32(window_end_sec.max(window_start_sec + 0.001));
        if clip.duration <= Duration::ZERO {
            return div();
        }
        if clip.start > t_end || clip.end() < t_start {
            return div();
        }

        let left = dur_to_px(clip.start, px_per_sec);
        let width = dur_to_px(clip.duration, px_per_sec).max(1.0);
        let semantic_type = if clip.semantic_type.trim().is_empty() {
            "content_support"
        } else {
            clip.semantic_type.trim()
        };
        let label = if clip.label.trim().is_empty() {
            "semantic"
        } else {
            clip.label.trim()
        };
        // Render type + label together so planning intent is visible on the semantic lane.
        let semantic_text = if label.is_empty() {
            format!("[{semantic_type}]")
        } else {
            format!("[{semantic_type}] {label}")
        };

        let clip_id = clip.id;
        let clip_start = clip.start;
        let clip_duration = clip.duration;
        let global_for_click = global_entity.clone();
        let global_for_resize = global_entity.clone();
        let global_for_right_click = global_entity.clone();

        let (bg, border, fg) = if selected {
            (rgba(0x0ea5e9c0), rgb(0x7dd3fc), rgb(0xe0f2fe))
        } else {
            (rgba(0x33415588), rgba(0xffffff2a), rgb(0xe2e8f0))
        };

        div()
            .absolute()
            .left(px(left))
            .top(px(3.0))
            .h(px(LANE_H - 6.0))
            .w(px(width))
            .rounded_sm()
            .border_1()
            .border_color(border)
            .bg(bg)
            .px_1()
            .overflow_hidden()
            .flex()
            .items_center()
            .cursor_pointer()
            .text_xs()
            .text_color(fg)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, evt: &MouseDownEvent, _win, cx| {
                    cx.stop_propagation();
                    if let Some(forward) =
                        Self::track_sweep_direction(active_tool, evt.modifiers.alt)
                    {
                        let pivot = this.timeline_time_from_mouse_x(evt.position.x);
                        this.sweep_select_from_anchor(TrackType::V1, pivot, forward, true, cx);
                        this.arm_track_sweep_pending_drag(evt.position.x, pivot, forward, cx);
                        return;
                    }
                    this.clip_link_menu = None;
                    this.timeline_clip_menu = None;
                    this.layer_clip_menu = None;
                    this.subtitle_clip_menu = None;
                    this.semantic_clip_menu = None;
                    this.drag_state = DragState::SemanticPending {
                        start_x: evt.position.x,
                        clip_id,
                        original_start: clip_start,
                    };
                    global_for_click.update(cx, |gs, cx| {
                        gs.select_semantic_clip(clip_id);
                        cx.notify();
                    });
                    cx.notify();
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                    cx.stop_propagation();
                    global_for_right_click.update(cx, |gs, cx| {
                        gs.select_semantic_clip(clip_id);
                        cx.notify();
                    });
                    let (menu_x, menu_y) = this.window_to_panel_point(evt.position, win);
                    this.clip_link_menu = None;
                    this.timeline_clip_menu = None;
                    this.layer_clip_menu = None;
                    this.subtitle_clip_menu = None;
                    this.semantic_clip_menu = Some(SemanticClipContextMenu {
                        x: menu_x,
                        y: menu_y,
                        clip_id,
                    });
                    cx.notify();
                }),
            )
            .children(if width >= 24.0 {
                vec![semantic_text]
            } else {
                Vec::new()
            })
            .child(
                div()
                    .absolute()
                    .right(px(0.0))
                    .top(px(0.0))
                    .h_full()
                    .w(px(10.0))
                    .bg(rgba(0xffffff00))
                    .cursor_col_resize()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, evt: &MouseDownEvent, _win, cx| {
                            cx.stop_propagation();
                            global_for_resize.update(cx, |gs, cx| {
                                gs.save_for_undo();
                                gs.select_semantic_clip(clip_id);
                                cx.notify();
                            });
                            this.clip_link_menu = None;
                            this.timeline_clip_menu = None;
                            this.layer_clip_menu = None;
                            this.subtitle_clip_menu = None;
                            this.semantic_clip_menu = None;
                            this.drag_state = DragState::SemanticResizing {
                                clip_id,
                                start_x: evt.position.x,
                                original_duration: clip_duration,
                            };
                            cx.notify();
                        }),
                    ),
            )
    }

    fn render_semantic_lane(
        clips: &[SemanticClip],
        selected_id: Option<u64>,
        px_per_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
        cx: &mut Context<TimelinePanel>,
        active_tool: ActiveTool,
        global_entity: Entity<GlobalState>,
    ) -> gpui::Div {
        let mut lane = div()
            .h(px(LANE_H))
            .w_full()
            .flex_shrink_0()
            .relative()
            .bg(black().opacity(0.17))
            .border_b_1()
            .border_color(white().opacity(0.06))
            .overflow_hidden();

        let t_start = Duration::from_secs_f32(window_start_sec.max(0.0));
        let t_end = Duration::from_secs_f32(window_end_sec.max(window_start_sec + 0.001));
        let mut i = clips.partition_point(|c| c.start + c.duration < t_start);
        while i < clips.len() {
            let clip = &clips[i];
            if clip.start > t_end {
                break;
            }
            lane = lane.child(Self::semantic_clip_ui(
                clip,
                selected_id == Some(clip.id),
                px_per_sec,
                window_start_sec,
                window_end_sec,
                cx,
                active_tool,
                global_entity.clone(),
            ));
            i += 1;
        }

        lane
    }

    fn compact_low_load_segments(mut segments: Vec<LowLoadSegmentRect>) -> Vec<LowLoadSegmentRect> {
        if segments.is_empty() {
            return segments;
        }

        if segments.len() > TIMELINE_LOW_LOAD_MAX_SEGMENTS {
            let chunk_size = segments.len().div_ceil(TIMELINE_LOW_LOAD_MAX_SEGMENTS);
            let mut downsampled: Vec<LowLoadSegmentRect> =
                Vec::with_capacity(segments.len().div_ceil(chunk_size));
            for chunk in segments.chunks(chunk_size) {
                let mut left = chunk[0].left;
                let mut right = chunk[0].left + chunk[0].width;
                for seg in chunk.iter().skip(1) {
                    left = left.min(seg.left);
                    right = right.max(seg.left + seg.width);
                }
                downsampled.push(LowLoadSegmentRect {
                    left,
                    width: (right - left).max(TIMELINE_LOW_LOAD_MIN_SEGMENT_PX),
                });
            }
            segments = downsampled;
        }

        let mut merged: Vec<LowLoadSegmentRect> = Vec::with_capacity(segments.len());
        for seg in segments {
            if let Some(last) = merged.last_mut() {
                let last_right = last.left + last.width;
                if seg.left <= last_right + TIMELINE_LOW_LOAD_MERGE_GAP_PX {
                    let right = (seg.left + seg.width).max(last_right);
                    last.width = (right - last.left).max(TIMELINE_LOW_LOAD_MIN_SEGMENT_PX);
                    continue;
                }
            }
            merged.push(seg);
        }
        merged
    }

    fn render_lane_low_load(
        clips: &[Clip],
        clip_start_index: Option<&[f32]>,
        clip_end_index: Option<&[f32]>,
        px_per_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
        is_audio: bool,
        is_media_drop_hovered: bool,
    ) -> gpui::Div {
        let mut lane = div()
            .h(px(LANE_H))
            .w_full()
            .flex_shrink_0()
            .relative()
            .bg(if is_media_drop_hovered {
                white().opacity(0.14)
            } else {
                black().opacity(if is_audio { 0.12 } else { 0.15 })
            })
            .border_b_1()
            .border_color(white().opacity(0.06))
            .overflow_hidden();

        if clips.is_empty() {
            return lane;
        }

        let t_start = Duration::from_secs_f32(window_start_sec.max(0.0));
        let t_end = Duration::from_secs_f32(window_end_sec.max(window_start_sec + 0.001));
        let t_start_sec = t_start.as_secs_f32();
        let t_end_sec = t_end.as_secs_f32();
        let index_usable = clip_start_index
            .zip(clip_end_index)
            .is_some_and(|(starts, ends)| starts.len() == clips.len() && ends.len() == clips.len());
        let (mut i, end_exclusive) = if index_usable {
            let starts = clip_start_index.expect("validated start index");
            let ends = clip_end_index.expect("validated end index");
            (
                ends.partition_point(|end| *end < t_start_sec),
                starts.partition_point(|start| *start <= t_end_sec),
            )
        } else {
            (
                clips.partition_point(|c| c.start + c.duration < t_start),
                clips.partition_point(|c| c.start <= t_end),
            )
        };
        let visible_count = end_exclusive.saturating_sub(i);
        if visible_count == 0 {
            return lane;
        }
        let chunk_size = visible_count
            .div_ceil(TIMELINE_LOW_LOAD_MAX_SEGMENTS)
            .max(1);
        let mut segments: Vec<LowLoadSegmentRect> =
            Vec::with_capacity(visible_count.div_ceil(chunk_size));
        while i < end_exclusive {
            let chunk_end = (i + chunk_size).min(end_exclusive);
            let mut chunk_left = f32::INFINITY;
            let mut chunk_right = f32::NEG_INFINITY;
            while i < chunk_end {
                let clip = &clips[i];
                if clip.duration > Duration::ZERO {
                    let left = dur_to_px(clip.start, px_per_sec);
                    let right = left + dur_to_px(clip.duration, px_per_sec);
                    chunk_left = chunk_left.min(left);
                    chunk_right = chunk_right.max(right);
                }
                i += 1;
            }
            if chunk_right.is_finite() {
                segments.push(LowLoadSegmentRect {
                    left: chunk_left,
                    width: (chunk_right - chunk_left).max(TIMELINE_LOW_LOAD_MIN_SEGMENT_PX),
                });
            }
        }
        let segments = Self::compact_low_load_segments(segments);
        if segments.is_empty() {
            return lane;
        }

        let segment_color = if is_audio {
            rgba(0x16a34a99)
        } else {
            rgba(0xfacc1570)
        };
        lane = lane.child(
            canvas(
                move |_bounds, _window, _cx| segments,
                move |bounds, rects, window, _cx| {
                    let bar_top = bounds.origin.y + px(4.0);
                    let bar_h = px((LANE_H - 8.0).max(2.0));
                    for rect in rects {
                        let draw_w = rect.width.max(TIMELINE_LOW_LOAD_MIN_SEGMENT_PX);
                        if draw_w <= 0.5 {
                            continue;
                        }
                        window.paint_quad(quad(
                            Bounds {
                                origin: point(bounds.origin.x + px(rect.left), bar_top),
                                size: size(px(draw_w), bar_h),
                            },
                            px(2.0),
                            segment_color,
                            px(0.0),
                            transparent_black(),
                            Default::default(),
                        ));
                    }
                },
            )
            .absolute()
            .left(px(0.0))
            .top(px(0.0))
            .w_full()
            .h_full(),
        );
        lane
    }

    fn render_subtitle_lane_low_load(
        clips: &[SubtitleClip],
        px_per_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
    ) -> gpui::Div {
        let mut lane = div()
            .h(px(LANE_H))
            .w_full()
            .flex_shrink_0()
            .relative()
            .bg(black().opacity(0.12))
            .border_b_1()
            .border_color(white().opacity(0.06))
            .overflow_hidden();

        if clips.is_empty() {
            return lane;
        }

        let t_start = Duration::from_secs_f32(window_start_sec.max(0.0));
        let t_end = Duration::from_secs_f32(window_end_sec.max(window_start_sec + 0.001));
        let mut i = clips.partition_point(|c| c.start + c.duration < t_start);
        let mut segments = Vec::new();
        while i < clips.len() {
            let clip = &clips[i];
            if clip.start > t_end {
                break;
            }
            if clip.duration > Duration::ZERO {
                segments.push(LowLoadSegmentRect {
                    left: dur_to_px(clip.start, px_per_sec),
                    width: dur_to_px(clip.duration, px_per_sec)
                        .max(TIMELINE_LOW_LOAD_MIN_SEGMENT_PX),
                });
            }
            i += 1;
        }
        let segments = Self::compact_low_load_segments(segments);
        if segments.is_empty() {
            return lane;
        }

        lane = lane.child(
            canvas(
                move |_bounds, _window, _cx| segments,
                move |bounds, rects, window, _cx| {
                    let bar_top = bounds.origin.y + px(4.0);
                    let bar_h = px((LANE_H - 8.0).max(2.0));
                    for rect in rects {
                        let draw_w = rect.width.max(TIMELINE_LOW_LOAD_MIN_SEGMENT_PX);
                        if draw_w <= 0.5 {
                            continue;
                        }
                        window.paint_quad(quad(
                            Bounds {
                                origin: point(bounds.origin.x + px(rect.left), bar_top),
                                size: size(px(draw_w), bar_h),
                            },
                            px(2.0),
                            rgba(0x93c5fdb8),
                            px(0.0),
                            transparent_black(),
                            Default::default(),
                        ));
                    }
                },
            )
            .absolute()
            .left(px(0.0))
            .top(px(0.0))
            .w_full()
            .h_full(),
        );
        lane
    }

    fn render_semantic_lane_low_load(
        clips: &[SemanticClip],
        px_per_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
    ) -> gpui::Div {
        let mut lane = div()
            .h(px(LANE_H))
            .w_full()
            .flex_shrink_0()
            .relative()
            .bg(black().opacity(0.17))
            .border_b_1()
            .border_color(white().opacity(0.06))
            .overflow_hidden();

        if clips.is_empty() {
            return lane;
        }

        let t_start = Duration::from_secs_f32(window_start_sec.max(0.0));
        let t_end = Duration::from_secs_f32(window_end_sec.max(window_start_sec + 0.001));
        let mut i = clips.partition_point(|c| c.start + c.duration < t_start);
        let mut segments = Vec::new();
        while i < clips.len() {
            let clip = &clips[i];
            if clip.start > t_end {
                break;
            }
            if clip.duration > Duration::ZERO {
                segments.push(LowLoadSegmentRect {
                    left: dur_to_px(clip.start, px_per_sec),
                    width: dur_to_px(clip.duration, px_per_sec)
                        .max(TIMELINE_LOW_LOAD_MIN_SEGMENT_PX),
                });
            }
            i += 1;
        }
        let segments = Self::compact_low_load_segments(segments);
        if segments.is_empty() {
            return lane;
        }

        lane = lane.child(
            canvas(
                move |_bounds, _window, _cx| segments,
                move |bounds, rects, window, _cx| {
                    let bar_top = bounds.origin.y + px(4.0);
                    let bar_h = px((LANE_H - 8.0).max(2.0));
                    for rect in rects {
                        let draw_w = rect.width.max(TIMELINE_LOW_LOAD_MIN_SEGMENT_PX);
                        if draw_w <= 0.5 {
                            continue;
                        }
                        window.paint_quad(quad(
                            Bounds {
                                origin: point(bounds.origin.x + px(rect.left), bar_top),
                                size: size(px(draw_w), bar_h),
                            },
                            px(2.0),
                            rgba(0x34d399b0),
                            px(0.0),
                            transparent_black(),
                            Default::default(),
                        ));
                    }
                },
            )
            .absolute()
            .left(px(0.0))
            .top(px(0.0))
            .w_full()
            .h_full(),
        );
        lane
    }

    fn subtitle_clip_ui(
        label: &str,
        duration: Duration,
        selected: bool,
        px_per_sec: f32,
    ) -> gpui::Div {
        let w = dur_to_px(duration, px_per_sec).max(2.0);
        let label_single = label.split_whitespace().collect::<Vec<_>>().join(" ");
        let (bg, border) = if selected {
            (rgb(0x93c5fd), rgb(0x3b82f6))
        } else {
            (rgba(0x93c5fdcc), rgba(0xffffff1f))
        };
        let mut node = div()
            .h(px(22.0))
            .w(px(w))
            .rounded_md()
            .border_1()
            .border_color(border)
            .bg(bg)
            .overflow_hidden()
            .px_2()
            .flex()
            .items_center()
            .gap_2()
            .text_xs()
            .text_color(rgb(0x111111))
            .min_w_0();
        if w > 36.0 {
            node = node.child("CC");
        }
        if w > 56.0 {
            node = node.child(div().flex_1().min_w_0().truncate().child(label_single));
        }
        node
    }

    fn render_lane(
        clips: &[Clip],
        clip_start_index: Option<&[f32]>,
        clip_end_index: Option<&[f32]>,
        selected_ids: &[u64],
        px_per_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
        is_audio: bool,
        is_media_drop_hovered: bool,
        track_type: TrackType,
        cx: &mut Context<TimelinePanel>,
        active_tool: ActiveTool,
        global_entity: Entity<GlobalState>,
    ) -> gpui::Div {
        let mut lane = div()
            .h(px(LANE_H))
            .w_full()
            .flex_shrink_0()
            .relative()
            .bg(if is_media_drop_hovered {
                white().opacity(0.14)
            } else {
                black().opacity(if is_audio { 0.12 } else { 0.15 })
            })
            .border_b_1()
            .border_color(white().opacity(0.06))
            .overflow_hidden();

        let t_start = Duration::from_secs_f32(window_start_sec.max(0.0));
        let t_end = Duration::from_secs_f32(window_end_sec.max(window_start_sec + 0.001));
        let t_start_sec = t_start.as_secs_f32();
        let t_end_sec = t_end.as_secs_f32();

        let index_usable = clip_start_index
            .zip(clip_end_index)
            .is_some_and(|(starts, ends)| starts.len() == clips.len() && ends.len() == clips.len());
        let (mut i, end_exclusive) = if index_usable {
            let starts = clip_start_index.expect("validated start index");
            let ends = clip_end_index.expect("validated end index");
            (
                ends.partition_point(|end| *end < t_start_sec),
                starts.partition_point(|start| *start <= t_end_sec),
            )
        } else {
            (
                clips.partition_point(|c| c.start + c.duration < t_start),
                clips.partition_point(|c| c.start <= t_end),
            )
        };
        let mut visible_clip_count = 0usize;
        let mut probe = i;
        while probe < end_exclusive {
            let c = &clips[probe];
            if c.duration > Duration::ZERO && c.duration.as_secs_f32() >= 0.05 {
                visible_clip_count = visible_clip_count.saturating_add(1);
            }
            probe += 1;
        }
        let (waveform_min_samples, waveform_max_samples, waveform_detail_level) = if is_audio {
            waveform_render_profile_for_lane(visible_clip_count)
        } else {
            (0, 0, AudioWaveformDetailLevel::Normal)
        };
        let show_clip_text = !is_audio || visible_clip_count < 80;
        let draw_inline_waveform = !is_audio;
        let mut waveform_peaks_cache: HashMap<&str, Option<Arc<Vec<f32>>>> = HashMap::new();
        let mut waveform_overlay_rects: Vec<AudioWaveformOverlayRect> = Vec::new();
        let should_virtualize_audio_clip_nodes =
            is_audio && visible_clip_count >= AUDIO_CLIP_NODE_VIRTUALIZE_THRESHOLD;
        let mut pending_audio_cluster: Option<(f32, f32, usize)> = None;

        let mut rendered = 0usize;
        while i < end_exclusive {
            let c = &clips[i];

            if c.duration.as_secs_f32() < 0.05 {
                i += 1;
                continue;
            }

            if c.duration > Duration::ZERO {
                let left = dur_to_px(c.start, px_per_sec);
                let selected = selected_ids.contains(&c.id);
                let clip_width_px = dur_to_px(c.duration, px_per_sec).max(2.0);
                let collapse_to_cluster = should_virtualize_audio_clip_nodes
                    && !selected
                    && clip_width_px <= AUDIO_CLIP_NODE_MIN_PIXEL_WIDTH;

                if collapse_to_cluster {
                    let clip_right = left + clip_width_px;
                    if let Some((_, right, count)) = pending_audio_cluster.as_mut() {
                        if left - *right <= AUDIO_CLIP_CLUSTER_GAP_PX {
                            *right = clip_right.max(*right);
                            *count = count.saturating_add(1);
                        } else {
                            let (cluster_left, cluster_right, cluster_count) =
                                pending_audio_cluster.take().expect("cluster exists");
                            let global_for_cluster_click = global_entity.clone();
                            let cluster_element = Self::audio_cluster_bar_ui(
                                cluster_count,
                                cluster_right - cluster_left,
                            )
                            .absolute()
                            .left(px(cluster_left))
                            .top(px(3.0))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, evt: &MouseDownEvent, _win, cx| {
                                    cx.stop_propagation();
                                    if let TrackType::Audio(track_idx) = track_type {
                                        let t = this.timeline_time_from_mouse_x(evt.position.x);
                                        global_for_cluster_click.update(cx, |gs, cx| {
                                            gs.set_playhead(t);
                                            if active_tool == ActiveTool::Razor {
                                                let _ = gs.razor_audio_at_playhead(track_idx);
                                            } else {
                                                gs.select_audio_clip_at(track_idx, t);
                                            }
                                            cx.notify();
                                        });
                                    }
                                }),
                            );
                            lane = lane.child(cluster_element);
                            rendered = rendered.saturating_add(1);
                            pending_audio_cluster = Some((left, clip_right, 1));
                        }
                    } else {
                        pending_audio_cluster = Some((left, clip_right, 1));
                    }
                    i += 1;
                    continue;
                } else if let Some((cluster_left, cluster_right, cluster_count)) =
                    pending_audio_cluster.take()
                {
                    let global_for_cluster_click = global_entity.clone();
                    let cluster_element =
                        Self::audio_cluster_bar_ui(cluster_count, cluster_right - cluster_left)
                            .absolute()
                            .left(px(cluster_left))
                            .top(px(3.0))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, evt: &MouseDownEvent, _win, cx| {
                                    cx.stop_propagation();
                                    if let TrackType::Audio(track_idx) = track_type {
                                        let t = this.timeline_time_from_mouse_x(evt.position.x);
                                        global_for_cluster_click.update(cx, |gs, cx| {
                                            gs.set_playhead(t);
                                            if active_tool == ActiveTool::Razor {
                                                let _ = gs.razor_audio_at_playhead(track_idx);
                                            } else {
                                                gs.select_audio_clip_at(track_idx, t);
                                            }
                                            cx.notify();
                                        });
                                    }
                                }),
                            );
                    lane = lane.child(cluster_element);
                    rendered = rendered.saturating_add(1);
                }
                let waveform_bucket_count =
                    if is_audio && waveform_detail_level == AudioWaveformDetailLevel::Extreme {
                        AUDIO_WAVEFORM_BUCKETS_MIN
                    } else {
                        waveform_bucket_count_for_media_duration(c.media_duration)
                    };
                let waveform_peaks = if is_audio {
                    if let Some(cached) = waveform_peaks_cache.get(c.file_path.as_str()) {
                        cached.clone()
                    } else {
                        // First try a read-only lookup to avoid marking GlobalState
                        // dirty and triggering an observer re-render loop.
                        let lookup = global_entity
                            .read(cx)
                            .lookup_waveform_for_path(&c.file_path, waveform_bucket_count);
                        let peaks = if lookup.peaks.is_some()
                            || lookup.status != WaveformStatus::Missing
                        {
                            lookup.peaks
                        } else {
                            // Waveform not yet known — need mutable access to queue generation.
                            global_entity
                                .update(cx, |gs, _| {
                                    gs.ensure_waveform_for_path(&c.file_path, waveform_bucket_count)
                                })
                                .peaks
                        };
                        waveform_peaks_cache.insert(c.file_path.as_str(), peaks.clone());
                        peaks
                    }
                } else {
                    None
                };
                if is_audio {
                    let clip_width = dur_to_px(c.duration, px_per_sec).max(2.0);
                    let inner_left = left + 2.0;
                    let inner_width = (clip_width - 4.0).max(1.0);
                    // Canvas overlays can afford near pixel-density sampling
                    // without creating one UI node per waveform bar.
                    let (overlay_min, overlay_max) =
                        if waveform_detail_level == AudioWaveformDetailLevel::Extreme {
                            (
                                AUDIO_WAVEFORM_MIN_SAMPLES_EXTREME,
                                AUDIO_WAVEFORM_MAX_SAMPLES_EXTREME,
                            )
                        } else {
                            (waveform_min_samples, waveform_max_samples)
                        };
                    if let Some(peaks) = waveform_peaks.as_ref() {
                        let bars = sample_waveform_for_clip(
                            peaks.as_slice(),
                            c,
                            inner_width,
                            overlay_min,
                            overlay_max,
                        );
                        if bars.is_empty() {
                            waveform_overlay_rects.push(AudioWaveformOverlayRect {
                                left: inner_left,
                                width: inner_width,
                                amp_norm: 0.56_f32,
                            });
                        } else {
                            let bar_w = (inner_width / bars.len() as f32).max(0.5);
                            for (idx, value) in bars.into_iter().enumerate() {
                                let x0 = inner_left + idx as f32 * bar_w;
                                let x1 = (inner_left + (idx + 1) as f32 * bar_w)
                                    .min(inner_left + inner_width);
                                waveform_overlay_rects.push(AudioWaveformOverlayRect {
                                    left: x0,
                                    width: (x1 - x0).max(0.5),
                                    amp_norm: value,
                                });
                            }
                        }
                    } else {
                        waveform_overlay_rects.push(AudioWaveformOverlayRect {
                            left: inner_left,
                            width: inner_width,
                            amp_norm: 0.56_f32,
                        });
                    }
                }

                let clip_id = c.id;
                let clip_start = c.start;
                let current_duration = c.duration;

                let global_for_click = global_entity.clone();
                let global_for_resize = global_entity.clone();
                let global_for_link_menu = global_entity.clone();
                let global_for_clip_context = global_entity.clone();

                let clip_element = Self::clip_bar_ui(
                    c,
                    selected,
                    px_per_sec,
                    is_audio,
                    show_clip_text,
                    draw_inline_waveform,
                    waveform_peaks,
                    waveform_min_samples,
                    waveform_max_samples,
                    waveform_detail_level,
                )
                .absolute()
                .left(px(left))
                .top(px(3.0))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, evt: &MouseDownEvent, _win, cx| {
                        cx.stop_propagation();
                        if let Some(pending) = global_for_click.read(cx).pending_transition {
                            let allow =
                                matches!(track_type, TrackType::V1 | TrackType::VideoOverlay(_));
                            if allow {
                                global_for_click.update(cx, |gs, cx| {
                                    if gs.apply_transition_to_clip(clip_id, pending) {
                                        gs.clear_layer_effect_clip_selection();
                                        gs.selected_clip_id = Some(clip_id);
                                        gs.selected_clip_ids = vec![clip_id];
                                        gs.selected_subtitle_id = None;
                                        gs.selected_subtitle_ids.clear();
                                    }
                                    gs.clear_transition_drag();
                                    cx.notify();
                                });
                                return;
                            }
                        }

                        if Self::track_sweep_direction(active_tool, evt.modifiers.alt).is_some() {
                            if let Some(forward) =
                                Self::track_sweep_direction(active_tool, evt.modifiers.alt)
                            {
                                let pivot = this.timeline_time_from_mouse_x(evt.position.x);
                                this.sweep_select_from_anchor(track_type, pivot, forward, true, cx);
                                this.arm_track_sweep_pending_drag(
                                    evt.position.x,
                                    pivot,
                                    forward,
                                    cx,
                                );
                            }
                            return;
                        }

                        if active_tool == ActiveTool::Razor {
                            // --- Razor tool logic ---
                            let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                            let raw_x = evt.position.x;
                            let window_x_px = raw_x - px(offset_w);
                            let window_x_f32 = window_x_px / px(1.0);
                            let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);
                            let cut_time =
                                Duration::from_secs_f32((local_x_f32 / this.px_per_sec).max(0.0));

                            global_for_click.update(cx, |gs, cx| {
                                gs.set_playhead(cut_time);
                                // Dispatch to the matching razor helper for the current track type.
                                match track_type {
                                    TrackType::V1 => {
                                        let _ = gs.razor_v1_at_playhead();
                                    }
                                    TrackType::Audio(idx) => {
                                        let _ = gs.razor_audio_at_playhead(idx);
                                    }
                                    TrackType::VideoOverlay(idx) => {
                                        let _ = gs.razor_video_at_playhead(idx);
                                    }
                                    TrackType::Subtitle(idx) => {
                                        let _ = gs.razor_subtitle_at_playhead(idx);
                                    }
                                }
                                cx.notify();
                            });
                        } else {
                            // --- Selection / drag logic ---
                            let start_x_px = evt.position.x;
                            let group_items = {
                                let gs = global_for_click.read(cx);
                                this.build_group_clip_items(gs, clip_id)
                            };
                            let group_active = group_items.is_some();
                            this.drag_state = DragState::Pending {
                                start_x: start_x_px,
                                clip_id,
                                track_type, // Preserve the full track enum through drag state.
                                original_start: clip_start,
                                group_items,
                            };

                            // Select immediately so the user gets visual feedback.
                            global_for_click.update(cx, |gs, cx| {
                                if group_active {
                                    gs.clear_layer_effect_clip_selection();
                                    gs.selected_clip_id = Some(clip_id);
                                } else {
                                    match track_type {
                                        TrackType::V1 => gs.select_v1_clip_at(gs.playhead),
                                        TrackType::Audio(idx) => {
                                            gs.select_audio_clip_at(idx, gs.playhead)
                                        }
                                        TrackType::VideoOverlay(idx) => {
                                            gs.select_video_clip_at(idx, gs.playhead)
                                        }
                                        TrackType::Subtitle(idx) => {
                                            gs.select_subtitle_clip_at(idx, gs.playhead)
                                        }
                                    }
                                    gs.selected_clip_id = Some(clip_id);
                                    gs.selected_clip_ids = vec![clip_id];
                                    gs.selected_subtitle_id = None;
                                    gs.selected_subtitle_ids.clear();
                                }
                                cx.notify();
                            });
                        }
                    }),
                )
                .on_mouse_down(
                    MouseButton::Right,
                    cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                        cx.stop_propagation();

                        let keep_multi_selection = {
                            let gs = global_for_link_menu.read(cx);
                            gs.selected_clip_ids.len() >= 2
                                && gs.selected_clip_ids.contains(&clip_id)
                        };
                        if !keep_multi_selection {
                            global_for_clip_context.update(cx, |gs, cx| {
                                gs.clear_layer_effect_clip_selection();
                                gs.clear_semantic_clip_selection();
                                gs.selected_clip_id = Some(clip_id);
                                gs.selected_clip_ids = vec![clip_id];
                                gs.selected_subtitle_id = None;
                                gs.selected_subtitle_ids.clear();
                                cx.notify();
                            });
                        }

                        let (menu_x, menu_y) = this.window_to_panel_point(evt.position, win);
                        let selected_count = global_for_link_menu.read(cx).selected_clip_ids.len();

                        this.timeline_clip_menu = None;
                        this.layer_clip_menu = None;
                        this.subtitle_clip_menu = None;
                        this.semantic_clip_menu = None;
                        if selected_count >= 2 {
                            this.clip_link_menu = Some(ClipLinkContextMenu {
                                x: menu_x,
                                y: menu_y,
                            });
                        } else {
                            this.clip_link_menu = None;
                            this.timeline_clip_menu = Some(TimelineClipContextMenu {
                                x: menu_x,
                                y: menu_y,
                                clip_id,
                                track_type,
                            });
                        }
                        cx.notify();
                    }),
                )
                // 2. Resize handle.
                // --- Resize Handle ---
                .child(
                    div()
                        .absolute()
                        .right(px(0.0))
                        .top(px(0.0))
                        .h_full()
                        .w(px(10.0))
                        .bg(rgba(0xffffff00))
                        .cursor_col_resize()
                        // .hover(|s| s.bg(white().opacity(0.3)))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                                cx.stop_propagation();
                                cx.focus_self(win);

                                global_for_resize.update(cx, |gs, cx| {
                                    gs.save_for_undo();
                                    gs.clear_layer_effect_clip_selection();
                                    gs.selected_clip_id = Some(clip_id);
                                    gs.selected_clip_ids = vec![clip_id];
                                    gs.selected_subtitle_id = None;
                                    gs.selected_subtitle_ids.clear();
                                    cx.notify();
                                });

                                this.drag_state = DragState::Resizing {
                                    clip_id,
                                    track_type,
                                    start_x: evt.position.x,
                                    // Use the captured duration instead of reading c.duration inside the closure.
                                    original_duration: current_duration,
                                };
                            }),
                        ),
                );

                lane = lane.child(clip_element);
                rendered += 1;
                if rendered >= V1_HARD_CAP {
                    break;
                }
            }
            i += 1;
        }
        if let Some((cluster_left, cluster_right, cluster_count)) = pending_audio_cluster.take() {
            let global_for_cluster_click = global_entity.clone();
            let cluster_element =
                Self::audio_cluster_bar_ui(cluster_count, cluster_right - cluster_left)
                    .absolute()
                    .left(px(cluster_left))
                    .top(px(3.0))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, evt: &MouseDownEvent, _win, cx| {
                            cx.stop_propagation();
                            if let TrackType::Audio(track_idx) = track_type {
                                let t = this.timeline_time_from_mouse_x(evt.position.x);
                                global_for_cluster_click.update(cx, |gs, cx| {
                                    gs.set_playhead(t);
                                    if active_tool == ActiveTool::Razor {
                                        let _ = gs.razor_audio_at_playhead(track_idx);
                                    } else {
                                        gs.select_audio_clip_at(track_idx, t);
                                    }
                                    cx.notify();
                                });
                            }
                        }),
                    );
            lane = lane.child(cluster_element);
        }
        if is_audio && !waveform_overlay_rects.is_empty() {
            let overlay_rects = waveform_overlay_rects;
            lane = lane.child(
                canvas(
                    move |_bounds, _window, _cx| overlay_rects,
                    move |bounds, rects, window, _cx| {
                        let center_y = bounds.origin.y + px(14.0);
                        // Single centerline for the entire lane instead of per-rect
                        window.paint_quad(quad(
                            Bounds {
                                origin: point(bounds.origin.x, center_y - px(0.5)),
                                size: size(bounds.size.width, px(1.0)),
                            },
                            px(0.0),
                            rgba(0x14532d66),
                            px(0.0),
                            transparent_black(),
                            Default::default(),
                        ));
                        for rect in rects {
                            let draw_w = rect.width.max(0.5);
                            if draw_w <= 0.5 {
                                continue;
                            }
                            let draw_left = rect.left;
                            let amp_px = (rect.amp_norm.clamp(0.0, 1.0) * 8.0).max(0.8);
                            let top = center_y - px(amp_px);
                            let h = px((amp_px * 2.0).min(18.0));
                            let fill_bounds = Bounds {
                                origin: point(bounds.origin.x + px(draw_left), top),
                                size: size(px(draw_w), h),
                            };
                            window.paint_quad(quad(
                                fill_bounds,
                                px(0.0),
                                rgba(0x14532d8a),
                                px(0.0),
                                transparent_black(),
                                Default::default(),
                            ));
                        }
                    },
                )
                .absolute()
                .left(px(0.0))
                .top(px(0.0))
                .w_full()
                .h_full(),
            );
        }
        lane
    }

    fn render_subtitle_lane(
        clips: &[SubtitleClip],
        selected_ids: &[u64],
        px_per_sec: f32,
        window_start_sec: f32,
        window_end_sec: f32,
        track_index: usize,
        cx: &mut Context<TimelinePanel>,
        active_tool: ActiveTool,
        global_entity: Entity<GlobalState>,
    ) -> gpui::Div {
        let mut lane = div()
            .h(px(LANE_H))
            .w_full()
            .flex_shrink_0()
            .relative()
            .bg(black().opacity(0.12))
            .border_b_1()
            .border_color(white().opacity(0.06))
            .overflow_hidden();

        let t_start = Duration::from_secs_f32(window_start_sec.max(0.0));
        let t_end = Duration::from_secs_f32(window_end_sec.max(window_start_sec + 0.001));

        let mut i = clips.partition_point(|c| c.start + c.duration < t_start);
        let mut rendered = 0usize;
        while i < clips.len() {
            let c = &clips[i];
            if c.start > t_end {
                break;
            }

            if c.duration > Duration::ZERO {
                let left = dur_to_px(c.start, px_per_sec);
                let selected = selected_ids.contains(&c.id);

                let clip_id = c.id;
                let clip_start = c.start;
                let current_duration = c.duration;
                let global_for_click = global_entity.clone();
                let global_for_resize = global_entity.clone();
                let global_for_subtitle_context = global_entity.clone();

                let clip_element =
                    Self::subtitle_clip_ui(&c.text, c.duration, selected, px_per_sec)
                        .absolute()
                        .left(px(left))
                        .top(px(3.0))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, evt: &MouseDownEvent, _win, cx| {
                                cx.stop_propagation();

                                if Self::track_sweep_direction(active_tool, evt.modifiers.alt)
                                    .is_some()
                                {
                                    if let Some(forward) =
                                        Self::track_sweep_direction(active_tool, evt.modifiers.alt)
                                    {
                                        let pivot = this.timeline_time_from_mouse_x(evt.position.x);
                                        this.sweep_select_from_anchor(
                                            TrackType::Subtitle(track_index),
                                            pivot,
                                            forward,
                                            true,
                                            cx,
                                        );
                                        this.arm_track_sweep_pending_drag(
                                            evt.position.x,
                                            pivot,
                                            forward,
                                            cx,
                                        );
                                    }
                                    return;
                                }

                                if active_tool == ActiveTool::Razor {
                                    let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                                    let raw_x = evt.position.x;
                                    let window_x_px = raw_x - px(offset_w);
                                    let window_x_f32 = window_x_px / px(1.0);
                                    let local_x_f32 =
                                        (window_x_f32 + this.scroll_offset_x).max(0.0);
                                    let cut_time = Duration::from_secs_f32(
                                        (local_x_f32 / this.px_per_sec).max(0.0),
                                    );

                                    global_for_click.update(cx, |gs, cx| {
                                        gs.set_playhead(cut_time);
                                        let _ = gs.razor_subtitle_at_playhead(track_index);
                                        cx.notify();
                                    });
                                } else {
                                    this.clip_link_menu = None;
                                    this.timeline_clip_menu = None;
                                    this.layer_clip_menu = None;
                                    this.subtitle_clip_menu = None;
                                    this.semantic_clip_menu = None;
                                    let start_x_px = evt.position.x;
                                    let group_items = {
                                        let gs = global_for_click.read(cx);
                                        this.build_group_subtitle_items(gs, clip_id)
                                    };
                                    let group_active = group_items.is_some();
                                    this.drag_state = DragState::Pending {
                                        start_x: start_x_px,
                                        clip_id,
                                        track_type: TrackType::Subtitle(track_index),
                                        original_start: clip_start,
                                        group_items,
                                    };

                                    global_for_click.update(cx, |gs, cx| {
                                        if group_active {
                                            gs.clear_layer_effect_clip_selection();
                                            gs.selected_subtitle_id = Some(clip_id);
                                        } else {
                                            gs.select_subtitle_clip_at(track_index, gs.playhead);
                                            gs.selected_subtitle_id = Some(clip_id);
                                            gs.selected_subtitle_ids = vec![clip_id];
                                            gs.selected_clip_id = None;
                                            gs.selected_clip_ids.clear();
                                        }
                                        cx.notify();
                                    });
                                }
                            }),
                        )
                        .on_mouse_down(
                            MouseButton::Right,
                            cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                                cx.stop_propagation();
                                global_for_subtitle_context.update(cx, |gs, cx| {
                                    gs.clear_layer_effect_clip_selection();
                                    gs.clear_semantic_clip_selection();
                                    gs.selected_subtitle_id = Some(clip_id);
                                    gs.selected_subtitle_ids = vec![clip_id];
                                    gs.selected_clip_id = None;
                                    gs.selected_clip_ids.clear();
                                    cx.notify();
                                });
                                let (menu_x, menu_y) =
                                    this.window_to_panel_point(evt.position, win);
                                this.clip_link_menu = None;
                                this.timeline_clip_menu = None;
                                this.layer_clip_menu = None;
                                this.subtitle_clip_menu = Some(SubtitleClipContextMenu {
                                    x: menu_x,
                                    y: menu_y,
                                    clip_id,
                                    track_index,
                                });
                                this.semantic_clip_menu = None;
                                cx.notify();
                            }),
                        )
                        .child(
                            div()
                                .absolute()
                                .right(px(0.0))
                                .top(px(0.0))
                                .h_full()
                                .w(px(10.0))
                                .bg(rgba(0xffffff00))
                                .cursor_col_resize()
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                                        cx.stop_propagation();
                                        cx.focus_self(win);

                                        global_for_resize.update(cx, |gs, cx| {
                                            gs.save_for_undo();
                                            gs.clear_layer_effect_clip_selection();
                                            gs.selected_subtitle_id = Some(clip_id);
                                            gs.selected_subtitle_ids = vec![clip_id];
                                            gs.selected_clip_id = None;
                                            gs.selected_clip_ids.clear();
                                            cx.notify();
                                        });

                                        this.clip_link_menu = None;
                                        this.timeline_clip_menu = None;
                                        this.layer_clip_menu = None;
                                        this.subtitle_clip_menu = None;
                                        this.semantic_clip_menu = None;
                                        this.drag_state = DragState::Resizing {
                                            clip_id,
                                            track_type: TrackType::Subtitle(track_index),
                                            start_x: evt.position.x,
                                            original_duration: current_duration,
                                        };
                                    }),
                                ),
                        );

                lane = lane.child(clip_element);
                rendered += 1;
                if rendered >= V1_HARD_CAP {
                    break;
                }
            }
            i += 1;
        }
        lane
    }

    fn fmt_srt_timestamp(d: Duration) -> String {
        let total_ms = d.as_millis() as u64;
        let ms = total_ms % 1000;
        let total_sec = total_ms / 1000;
        let sec = total_sec % 60;
        let total_min = total_sec / 60;
        let min = total_min % 60;
        let hour = total_min / 60;
        format!("{hour:02}:{min:02}:{sec:02},{ms:03}")
    }

    fn build_full_timeline_srt(gs: &GlobalState) -> Result<String, String> {
        let mut clips: Vec<(Duration, Duration, u64, String)> = Vec::new();
        for track in &gs.subtitle_tracks {
            for clip in &track.clips {
                let start = clip.start;
                let end = clip.end();
                if end <= start {
                    continue;
                }
                let text = clip.text.trim().to_string();
                if text.is_empty() {
                    continue;
                }
                clips.push((start, end, clip.id, text));
            }
        }
        if clips.is_empty() {
            return Err("No subtitle clips found on timeline.".to_string());
        }
        clips.sort_by_key(|(start, _, id, _)| (*start, *id));

        let mut out = String::new();
        for (idx, (start, end, _, text)) in clips.into_iter().enumerate() {
            out.push_str(&(idx + 1).to_string());
            out.push('\n');
            out.push_str(&Self::fmt_srt_timestamp(start));
            out.push_str(" --> ");
            out.push_str(&Self::fmt_srt_timestamp(end));
            out.push('\n');
            out.push_str(&text);
            out.push_str("\n\n");
        }
        Ok(out)
    }

    pub(crate) fn default_export_dir() -> PathBuf {
        if let Ok(home) = std::env::var("HOME") {
            let movies_dir = PathBuf::from(&home).join("Movies");
            if movies_dir.exists() {
                return movies_dir;
            }
            let desktop_dir = PathBuf::from(home).join("Desktop");
            if desktop_dir.exists() {
                return desktop_dir;
            }
        }
        PathBuf::from(".")
    }

    // ==========
}

impl Render for TimelinePanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let now = Instant::now();
        if let Some(prev) = self.ui_fps_last_instant {
            let dt = now.saturating_duration_since(prev).as_secs_f32();
            if (1.0 / 240.0..=0.5).contains(&dt) {
                let sample_fps = 1.0 / dt;
                self.ui_fps_ema = if self.ui_fps_ema <= 0.0 {
                    sample_fps
                } else {
                    self.ui_fps_ema * 0.90 + sample_fps * 0.10
                };
            }
        }
        self.ui_fps_last_instant = Some(now);

        let (
            playing,
            active_tool,
            timeline_edit_token,
            playhead,
            selected_clip_ids,
            selected_subtitle_ids,
            selected_clips_have_any_link_group,
            export_in_progress,
            export_progress_ratio,
            export_eta,
            export_last_error,
            export_last_out,
            media_tools_ready_for_export,
            preview_fps,
            preview_video_input_fps,
            preview_present_fps,
            preview_present_dropped_frames,
            preview_quality,
            v1_move_mode,
            ui_notice,
            pending_trim_to_fit,
            timeline_total,
            active_source_name,
            canvas_w,
            canvas_h,
            layer_effect_clips,
            selected_layer_effect_clip_id,
            selected_semantic_clip_id,
            v1_clips,
            audio_tracks_data,
            video_tracks_data,
            subtitle_tracks_data,
            semantic_clips,
        ) = {
            let gs = self.global.read(cx);
            // println!("[Render] V1: {}, Audio: {}, VideoOverlay: {}",
            //     gs.v1_clips.len(),
            //     gs.audio_tracks.len(),
            //     gs.video_tracks.len()
            // );
            (
                gs.is_playing,
                gs.active_tool,
                gs.timeline_edit_token(),
                gs.playhead,
                gs.selected_clip_ids.clone(),
                gs.selected_subtitle_ids.clone(),
                gs.selected_clips_have_any_link_group(),
                gs.export_in_progress,
                gs.export_progress_ratio,
                gs.export_eta,
                gs.export_last_error.clone(),
                gs.export_last_out_path.clone(),
                gs.media_tools_ready_for_export(),
                gs.preview_fps,
                gs.preview_video_input_fps,
                gs.preview_present_fps,
                gs.preview_present_dropped_frames,
                gs.preview_quality,
                gs.v1_move_mode,
                gs.ui_notice.clone(),
                gs.pending_trim_to_fit.clone(),
                gs.timeline_total(),
                gs.active_source_name.clone(),
                gs.canvas_w,
                gs.canvas_h,
                gs.layer_effect_clips().to_vec(),
                gs.selected_layer_effect_clip_id(),
                gs.selected_semantic_clip_id(),
                gs.v1_clips.clone(),
                gs.audio_tracks.clone(),
                gs.video_tracks.clone(),
                gs.subtitle_tracks.clone(),
                gs.semantic_clips().to_vec(),
            )
        };

        // Keep semantic lane visibility fully data-driven.
        self.show_semantic_lane = !semantic_clips.is_empty();
        self.ensure_audio_lane_time_index(&audio_tracks_data, timeline_edit_token);

        let play_label: &'static str = if playing { "Pause" } else { "Play" };
        let timeline_low_load_mode = self.timeline_load_mode.use_low_load(playing);
        let current_px_per_sec = self.px_per_sec;
        let ruler_w = dur_to_px(timeline_total, current_px_per_sec).max(1000.0);
        let playhead_x = dur_to_px(playhead, current_px_per_sec);
        let display_settings_label = format!(
            "Display {} @{}fps",
            display_ratio_label(canvas_w, canvas_h),
            preview_fps.value()
        );
        let timeline_fps_label = if self.ui_fps_ema > 0.0 {
            format!("UI {:.1} fps", self.ui_fps_ema)
        } else {
            "UI --.- fps".to_string()
        };
        let video_fps_label = if preview_video_input_fps > 0.0 {
            format!("Video {:.1} fps", preview_video_input_fps)
        } else {
            "Video --.- fps".to_string()
        };
        let present_fps_label = if preview_present_fps > 0.0 {
            format!("Actual {:.1} fps", preview_present_fps)
        } else {
            "Actual --.- fps".to_string()
        };
        let present_drop_label = format!("Drop {}", preview_present_dropped_frames);

        // let total_tracks = 1 + audio_tracks_data.len();
        // let total_tracks_height = LANE_H * total_tracks as f32;
        // V1 + Video Overlays + Subtitle + Audio
        let semantic_lane_count = if self.show_semantic_lane { 1 } else { 0 };
        let total_tracks = 1
            + audio_tracks_data.len()
            + video_tracks_data.len()
            + subtitle_tracks_data.len()
            + semantic_lane_count;
        let total_tracks_height = LANE_H * total_tracks as f32;
        let view_height = TIMELINE_BODY_H - RULER_H;

        let marquee_overlay = if let Some(bounds) = self.marquee_bounds_in_panel(_window) {
            div()
                .absolute()
                .left(bounds.origin.x)
                .top(bounds.origin.y)
                .w(bounds.size.width)
                .h(bounds.size.height)
                .border_1()
                .border_color(rgb(0x60a5fa))
                .bg(rgba(0x60a5fa33))
        } else {
            div()
        };

        let proxy_modal_overlay = if self.proxy_modal_open {
            let (
                proxy_state_high,
                proxy_state_medium,
                proxy_state_low,
                proxy_mode_high,
                proxy_mode_medium,
                proxy_mode_low,
                mac_preview_mode,
            ) = {
                let gs = self.global.read(cx);
                (
                    Self::proxy_state_for_quality(gs, PreviewQuality::High),
                    Self::proxy_state_for_quality(gs, PreviewQuality::Medium),
                    Self::proxy_state_for_quality(gs, PreviewQuality::Low),
                    gs.proxy_render_mode_for_quality(PreviewQuality::High),
                    gs.proxy_render_mode_for_quality(PreviewQuality::Medium),
                    gs.proxy_render_mode_for_quality(PreviewQuality::Low),
                    gs.mac_preview_render_mode,
                )
            };
            let has_pending_proxy = matches!(proxy_state_high, ProxyRowState::Pending)
                || matches!(proxy_state_medium, ProxyRowState::Pending)
                || matches!(proxy_state_low, ProxyRowState::Pending);
            if has_pending_proxy {
                self.ensure_proxy_spinner_running(_window, cx);
            }

            let quality_name = |quality: PreviewQuality| -> &'static str {
                match quality {
                    PreviewQuality::High => "Medium 480p",
                    PreviewQuality::Medium => "Low 360p",
                    PreviewQuality::Low => "Super Low 144p",
                    PreviewQuality::Full => "Full",
                }
            };

            let preview_btn =
                |label: &'static str, quality: PreviewQuality, global: Entity<GlobalState>| {
                    TimelinePanel::transport_btn(label).on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |_this, _, _, cx| {
                            global.update(cx, |gs, cx| {
                                gs.preview_quality = quality;
                                gs.ui_notice = Some(format!("Preview quality set to {}.", label));
                                cx.notify();
                            });
                        }),
                    )
                };

            let spinner_frames = ["|", "/", "-", "\\"];
            let spinner_icon = spinner_frames[self.proxy_spinner_phase % spinner_frames.len()];
            let proxy_mode_locked_by_global = cfg!(target_os = "macos")
                && matches!(mac_preview_mode, MacPreviewRenderMode::FullBgra);

            let proxy_icon_btn = |quality: PreviewQuality, state: ProxyRowState| {
                let (icon, bg, border, text_color) = match state {
                    ProxyRowState::Ready => (
                        "OK",
                        rgba(0x14532dff),
                        rgba(0x16a34aff),
                        white().opacity(0.95),
                    ),
                    ProxyRowState::Pending => (
                        spinner_icon,
                        rgba(0x1e293bff),
                        rgba(0x334155ff),
                        white().opacity(0.95),
                    ),
                    ProxyRowState::Missing => (
                        "↻",
                        rgba(0xffffff14),
                        rgba(0xffffff2e),
                        white().opacity(0.9),
                    ),
                    ProxyRowState::NoSelection => (
                        "↻",
                        rgba(0xffffff0a),
                        rgba(0xffffff19),
                        white().opacity(0.35),
                    ),
                };

                let btn = div()
                    .w(px(28.0))
                    .h(px(28.0))
                    .rounded_lg()
                    .border_1()
                    .border_color(border)
                    .bg(bg)
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_xs()
                    .text_color(text_color)
                    .child(icon);

                match state {
                    ProxyRowState::Missing => btn
                        .hover(|s| s.bg(white().opacity(0.14)))
                        .cursor_pointer()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, _, _, cx| {
                                this.proxy_confirm_quality = Some(quality);
                                this.proxy_confirm_delete_quality = None;
                                this.proxy_confirm_delete_all = false;
                                cx.notify();
                            }),
                        ),
                    ProxyRowState::NoSelection => {
                        let global_for_notice = self.global.clone();
                        btn.cursor_pointer().on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |_this, _, _, cx| {
                                global_for_notice.update(cx, |gs, cx| {
                                    gs.ui_notice =
                                        Some("Select a video clip to generate proxy.".to_string());
                                    cx.notify();
                                });
                            }),
                        )
                    }
                    ProxyRowState::Pending | ProxyRowState::Ready => btn,
                }
            };

            let delete_proxy_icon_btn = |quality: PreviewQuality, state: ProxyRowState| {
                let btn = div()
                    .w(px(28.0))
                    .h(px(28.0))
                    .rounded_lg()
                    .border_1()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_xs()
                    .child("x");

                match state {
                    ProxyRowState::Ready | ProxyRowState::Pending => btn
                        .border_color(rgba(0xef444480))
                        .bg(rgba(0xef44441f))
                        .text_color(white().opacity(0.9))
                        .hover(|s| s.bg(rgba(0xef444440)))
                        .cursor_pointer()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, _, _, cx| {
                                this.proxy_confirm_quality = None;
                                this.proxy_confirm_delete_quality = Some(quality);
                                this.proxy_confirm_delete_all = false;
                                cx.notify();
                            }),
                        ),
                    ProxyRowState::NoSelection => {
                        let global_for_notice = self.global.clone();
                        btn.border_color(white().opacity(0.10))
                            .bg(white().opacity(0.03))
                            .text_color(white().opacity(0.35))
                            .cursor_pointer()
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |_this, _, _, cx| {
                                    global_for_notice.update(cx, |gs, cx| {
                                        gs.ui_notice = Some(
                                            "Select a video clip to delete proxy.".to_string(),
                                        );
                                        cx.notify();
                                    });
                                }),
                            )
                    }
                    ProxyRowState::Missing => btn
                        .border_color(white().opacity(0.10))
                        .bg(white().opacity(0.03))
                        .text_color(white().opacity(0.30)),
                }
            };

            let proxy_mode_btn =
                |quality: PreviewQuality, mode: ProxyRenderMode, global: Entity<GlobalState>| {
                    let quality_label = quality_name(quality);
                    let effective_mode = if proxy_mode_locked_by_global {
                        ProxyRenderMode::BgraImage
                    } else {
                        mode
                    };
                    let (label, border, bg, text_color) = if proxy_mode_locked_by_global {
                        (
                            "BG",
                            rgba(0xffffff1f),
                            rgba(0xffffff0d),
                            white().opacity(0.45),
                        )
                    } else {
                        match effective_mode {
                            ProxyRenderMode::Nv12Surface => (
                                "NV",
                                rgba(0x0ea5e980),
                                rgba(0x0ea5e930),
                                white().opacity(0.95),
                            ),
                            ProxyRenderMode::BgraImage => (
                                "BG",
                                rgba(0xf59e0b80),
                                rgba(0xf59e0b30),
                                white().opacity(0.95),
                            ),
                        }
                    };
                    let btn = div()
                        .w(px(28.0))
                        .h(px(28.0))
                        .rounded_lg()
                        .border_1()
                        .border_color(border)
                        .bg(bg)
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_xs()
                        .text_color(text_color)
                        .child(label);
                    if proxy_mode_locked_by_global {
                        btn
                    } else {
                        btn.cursor_pointer()
                            .hover(|s| s.bg(white().opacity(0.18)))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |_this, _, _, cx| {
                                    global.update(cx, |gs, cx| {
                                        if let Some(next_mode) =
                                            gs.toggle_proxy_render_mode_for_quality(quality)
                                        {
                                            gs.ui_notice = Some(format!(
                                                "{} proxy render mode: {}",
                                                quality_label,
                                                next_mode.label()
                                            ));
                                        }
                                        cx.notify();
                                    });
                                }),
                            )
                    }
                };

            let confirm_card = if let Some(confirm_quality) = self.proxy_confirm_quality {
                let confirm_quality_label = quality_name(confirm_quality);
                let confirm_dim = confirm_quality.max_dim().unwrap_or(0);
                let global_for_confirm = self.global.clone();

                div()
                    .rounded_md()
                    .border_1()
                    .border_color(white().opacity(0.12))
                    .bg(black().opacity(0.2))
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.9))
                            .child("Generate proxy now?"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.55))
                            .child(format!(
                                "Generate {} proxy for selected video clips?",
                                confirm_quality_label
                            )),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(TimelinePanel::transport_btn("Generate").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    global_for_confirm.update(cx, |gs, cx| {
                                        gs.preview_quality = confirm_quality;
                                        if TimelinePanel::queue_selected_proxies(gs, confirm_dim) {
                                            gs.ui_notice =
                                                Some(format!("Proxy queued ({}p).", confirm_dim));
                                        } else {
                                            gs.ui_notice = Some(
                                                "Select a video clip to generate proxy."
                                                    .to_string(),
                                            );
                                        }
                                        cx.notify();
                                    });
                                    this.proxy_confirm_quality = None;
                                    this.proxy_confirm_delete_quality = None;
                                    cx.notify();
                                }),
                            ))
                            .child(TimelinePanel::transport_btn("Cancel").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    this.proxy_confirm_quality = None;
                                    cx.notify();
                                }),
                            )),
                    )
            } else {
                div()
            };

            let delete_quality_confirm_card = if let Some(delete_quality) =
                self.proxy_confirm_delete_quality
            {
                let delete_quality_label = quality_name(delete_quality);
                let confirm_dim = delete_quality.max_dim().unwrap_or(0);
                let global_for_delete_confirm = self.global.clone();

                div()
                    .rounded_md()
                    .border_1()
                    .border_color(rgba(0xef444480))
                    .bg(rgba(0x450a0aff))
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(div().text_xs().text_color(white().opacity(0.95)).child("Delete proxy now?"))
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.65))
                            .child(format!("Delete {} proxy for selected video clip(s)?", delete_quality_label))
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                TimelinePanel::transport_btn("Delete")
                                    .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _, cx| {
                                        global_for_delete_confirm.update(cx, |gs, cx| {
                                            let report = gs.delete_selected_proxy_for_quality(confirm_dim);
                                            if report.deleted_files > 0 || report.removed_jobs > 0 {
                                                gs.ui_notice = Some(format!(
                                                    "Deleted {} proxy file(s), removed {} queued job(s).",
                                                    report.deleted_files, report.removed_jobs
                                                ));
                                            } else if report.blocked_active_jobs > 0 {
                                                gs.ui_notice = Some("Cannot delete while this proxy is actively generating.".to_string());
                                            } else {
                                                gs.ui_notice = Some("No proxy found for selected clip(s) at this resolution.".to_string());
                                            }
                                            cx.notify();
                                        });
                                        this.proxy_confirm_delete_quality = None;
                                        cx.notify();
                                    }))
                            )
                            .child(
                                TimelinePanel::transport_btn("Cancel")
                                    .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _, cx| {
                                        this.proxy_confirm_delete_quality = None;
                                        cx.notify();
                                    }))
                            )
                    )
            } else {
                div()
            };

            let delete_all_confirm_card = if self.proxy_confirm_delete_all {
                let global_for_delete_all = self.global.clone();
                div()
                    .rounded_md()
                    .border_1()
                    .border_color(rgba(0xef444480))
                    .bg(rgba(0x450a0aff))
                    .p_2()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.95))
                            .child("Delete all proxy files?"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.65))
                            .child("This will remove all cached proxies for this project."),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(TimelinePanel::transport_btn("Delete All").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    global_for_delete_all.update(cx, |gs, cx| {
                                        let report = gs.delete_all_proxies();
                                        gs.ui_notice = Some(format!(
                                            "Deleted {} proxy file(s), removed {} queued job(s).",
                                            report.deleted_files, report.removed_jobs
                                        ));
                                        cx.notify();
                                    });
                                    this.proxy_confirm_delete_all = false;
                                    cx.notify();
                                }),
                            ))
                            .child(TimelinePanel::transport_btn("Cancel").on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    this.proxy_confirm_delete_all = false;
                                    cx.notify();
                                }),
                            )),
                    )
            } else {
                div()
            };

            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left_0()
                .right_0()
                .bg(black().opacity(0.6))
                .flex()
                .items_center()
                .justify_center()
                .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _, cx| {
                    this.proxy_modal_open = false;
                    this.proxy_confirm_quality = None;
                    this.proxy_confirm_delete_quality = None;
                    this.proxy_confirm_delete_all = false;
                    cx.notify();
                }))
                .child(
                    div()
                        .w(px(360.0))
                        .rounded_md()
                        .bg(rgb(0x1f1f23))
                        .border_1()
                        .border_color(white().opacity(0.12))
                        .p_3()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .on_mouse_down(MouseButton::Left, cx.listener(move |_this, _, _, cx| {
                            cx.stop_propagation();
                        }))
                        .child(div().text_sm().text_color(white().opacity(0.9)).child("Preview & Proxy"))
                        .child(
                            div()
                                .text_xs()
                                .text_color(white().opacity(0.5))
                                .child(if proxy_mode_locked_by_global {
                                    "Choose preview quality. Use ↻ to generate, x to delete. NV/BG is locked to BG in Full BGRA mode."
                                } else {
                                    "Choose preview quality. Use ↻ to generate, x to delete, NV/BG to switch proxy render mode."
                                })
                        )
                        .child(
                            if cfg!(target_os = "macos") {
                                div()
                                    .text_xs()
                                    .text_color(white().opacity(0.65))
                                    .child(format!(
                                        "macOS pipeline mode: {} (Cmd+B to toggle)",
                                        match mac_preview_mode {
                                            MacPreviewRenderMode::HybridNv12 => "Hybrid NV12",
                                            MacPreviewRenderMode::FullBgra => "Full BGRA",
                                        }
                                    ))
                            } else {
                                div()
                            }
                        )
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap_2()
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .gap_2()
                                        .child(preview_btn("Full (Original)", PreviewQuality::Full, self.global.clone()))
                                        .child(
                                            div()
                                                .w(px(28.0))
                                                .h(px(28.0))
                                                .rounded_lg()
                                                .border_1()
                                                .border_color(white().opacity(0.08))
                                                .bg(white().opacity(0.03))
                                                .flex()
                                                .items_center()
                                                .justify_center()
                                                .text_xs()
                                                .text_color(white().opacity(0.35))
                                                .child("-")
                                        )
                                )
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .gap_2()
                                        .child(preview_btn("Medium 480p", PreviewQuality::High, self.global.clone()))
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap_1()
                                                .child(proxy_icon_btn(PreviewQuality::High, proxy_state_high))
                                                .child(delete_proxy_icon_btn(PreviewQuality::High, proxy_state_high))
                                                .child(proxy_mode_btn(
                                                    PreviewQuality::High,
                                                    proxy_mode_high,
                                                    self.global.clone(),
                                                ))
                                        )
                                )
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .gap_2()
                                        .child(preview_btn("Low 360p", PreviewQuality::Medium, self.global.clone()))
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap_1()
                                                .child(proxy_icon_btn(PreviewQuality::Medium, proxy_state_medium))
                                                .child(delete_proxy_icon_btn(PreviewQuality::Medium, proxy_state_medium))
                                                .child(proxy_mode_btn(
                                                    PreviewQuality::Medium,
                                                    proxy_mode_medium,
                                                    self.global.clone(),
                                                ))
                                        )
                                )
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .gap_2()
                                        .child(preview_btn("Super Low 144p", PreviewQuality::Low, self.global.clone()))
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap_1()
                                                .child(proxy_icon_btn(PreviewQuality::Low, proxy_state_low))
                                                .child(delete_proxy_icon_btn(PreviewQuality::Low, proxy_state_low))
                                                .child(proxy_mode_btn(
                                                    PreviewQuality::Low,
                                                    proxy_mode_low,
                                                    self.global.clone(),
                                                ))
                                        )
                                )
                        )
                        .child(confirm_card)
                        .child(delete_quality_confirm_card)
                        .child(
                            TimelinePanel::transport_btn("Delete All Proxies")
                                .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _, cx| {
                                    this.proxy_confirm_quality = None;
                                    this.proxy_confirm_delete_quality = None;
                                    this.proxy_confirm_delete_all = true;
                                    cx.notify();
                                }))
                        )
                        .child(delete_all_confirm_card)
                        .child(
                            TimelinePanel::transport_btn("Cancel")
                                .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _, cx| {
                                    this.proxy_modal_open = false;
                                    this.proxy_confirm_quality = None;
                                    this.proxy_confirm_delete_quality = None;
                                    this.proxy_confirm_delete_all = false;
                                    cx.notify();
                                }))
                        )
                )
        } else {
            div()
        };

        let global_for_link_menu_link = self.global.clone();
        let global_for_link_menu_unlink = self.global.clone();
        let global_for_timeline_menu_duplicate = self.global.clone();
        let global_for_subtitle_menu_duplicate = self.global.clone();
        let global_for_layer_menu_remove = self.global.clone();
        let global_for_layer_menu_duplicate = self.global.clone();
        let global_for_semantic_menu_duplicate = self.global.clone();
        let global_for_semantic_menu_remove = self.global.clone();
        let clip_link_menu_overlay = if let Some(menu) = self.clip_link_menu {
            // Render a lightweight right-click menu for multi-selection link/unlink actions.
            let unlink_enabled = selected_clips_have_any_link_group;
            let unlink_opacity = if unlink_enabled { 1.0 } else { 0.4 };
            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left_0()
                .right_0()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.clip_link_menu = None;
                        cx.notify();
                    }),
                )
                .on_mouse_down(
                    MouseButton::Right,
                    cx.listener(move |this, _, _, cx| {
                        this.clip_link_menu = None;
                        cx.notify();
                    }),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(menu.x))
                        .top(px(menu.y))
                        .w(px(170.0))
                        .rounded_md()
                        .bg(rgb(0x1f1f23))
                        .border_1()
                        .border_color(white().opacity(0.14))
                        .p_1()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |_this, _, _, cx| {
                                cx.stop_propagation();
                            }),
                        )
                        .on_mouse_down(
                            MouseButton::Right,
                            cx.listener(move |_this, _, _, cx| {
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
                                .text_color(white().opacity(0.9))
                                .bg(white().opacity(0.03))
                                .hover(|style| style.bg(white().opacity(0.10)))
                                .cursor_pointer()
                                .child("Link")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        global_for_link_menu_link.update(cx, |gs, cx| {
                                            if gs.link_selected_clips_into_group() {
                                                gs.ui_notice =
                                                    Some("Linked selected clips.".to_string());
                                            } else {
                                                gs.ui_notice = Some(
                                                    "Select two or more clips to link.".to_string(),
                                                );
                                            }
                                            cx.notify();
                                        });
                                        this.clip_link_menu = None;
                                        cx.notify();
                                    }),
                                ),
                        )
                        .child(
                            div()
                                .h(px(28.0))
                                .rounded_sm()
                                .px_2()
                                .flex()
                                .items_center()
                                .text_sm()
                                .text_color(white().opacity(0.9 * unlink_opacity))
                                .bg(white().opacity(0.03))
                                .hover(|style| style.bg(white().opacity(0.10)))
                                .cursor_pointer()
                                .child("Unlink")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        if unlink_enabled {
                                            global_for_link_menu_unlink.update(cx, |gs, cx| {
                                                if gs.unlink_selected_clips_groups() {
                                                    gs.ui_notice =
                                                        Some("A/V link removed.".to_string());
                                                }
                                                cx.notify();
                                            });
                                        }
                                        this.clip_link_menu = None;
                                        cx.notify();
                                    }),
                                ),
                        ),
                )
        } else {
            div()
        };

        let timeline_clip_menu_overlay = if let Some(menu) = self.timeline_clip_menu {
            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left_0()
                .right_0()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.timeline_clip_menu = None;
                        cx.notify();
                    }),
                )
                .on_mouse_down(
                    MouseButton::Right,
                    cx.listener(move |this, _, _, cx| {
                        this.timeline_clip_menu = None;
                        cx.notify();
                    }),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(menu.x))
                        .top(px(menu.y))
                        .w(px(220.0))
                        .rounded_md()
                        .bg(rgb(0x1f1f23))
                        .border_1()
                        .border_color(white().opacity(0.14))
                        .p_1()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |_this, _, _, cx| {
                                cx.stop_propagation();
                            }),
                        )
                        .on_mouse_down(
                            MouseButton::Right,
                            cx.listener(move |_this, _, _, cx| {
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
                                .text_color(white().opacity(0.9))
                                .bg(white().opacity(0.03))
                                .hover(|style| style.bg(white().opacity(0.10)))
                                .cursor_pointer()
                                .child("Duplicate Clip")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        global_for_timeline_menu_duplicate.update(cx, |gs, cx| {
                                            if gs.duplicate_timeline_clip_after(
                                                menu.track_type,
                                                menu.clip_id,
                                            ) {
                                                gs.ui_notice = Some("Clip duplicated.".to_string());
                                            }
                                            cx.notify();
                                        });
                                        this.timeline_clip_menu = None;
                                        cx.notify();
                                    }),
                                ),
                        ),
                )
        } else {
            div()
        };

        let subtitle_clip_menu_overlay = if let Some(menu) = self.subtitle_clip_menu {
            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left_0()
                .right_0()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.subtitle_clip_menu = None;
                        cx.notify();
                    }),
                )
                .on_mouse_down(
                    MouseButton::Right,
                    cx.listener(move |this, _, _, cx| {
                        this.subtitle_clip_menu = None;
                        cx.notify();
                    }),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(menu.x))
                        .top(px(menu.y))
                        .w(px(220.0))
                        .rounded_md()
                        .bg(rgb(0x1f1f23))
                        .border_1()
                        .border_color(white().opacity(0.14))
                        .p_1()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |_this, _, _, cx| {
                                cx.stop_propagation();
                            }),
                        )
                        .on_mouse_down(
                            MouseButton::Right,
                            cx.listener(move |_this, _, _, cx| {
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
                                .text_color(white().opacity(0.9))
                                .bg(white().opacity(0.03))
                                .hover(|style| style.bg(white().opacity(0.10)))
                                .cursor_pointer()
                                .child("Duplicate Subtitle")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        global_for_subtitle_menu_duplicate.update(cx, |gs, cx| {
                                            if gs.duplicate_subtitle_clip_after(
                                                menu.track_index,
                                                menu.clip_id,
                                            ) {
                                                gs.ui_notice =
                                                    Some("Subtitle duplicated.".to_string());
                                            }
                                            cx.notify();
                                        });
                                        this.subtitle_clip_menu = None;
                                        cx.notify();
                                    }),
                                ),
                        ),
                )
        } else {
            div()
        };

        let layer_clip_menu_overlay = if let Some(menu) = self.layer_clip_menu {
            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left_0()
                .right_0()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.layer_clip_menu = None;
                        cx.notify();
                    }),
                )
                .on_mouse_down(
                    MouseButton::Right,
                    cx.listener(move |this, _, _, cx| {
                        this.layer_clip_menu = None;
                        cx.notify();
                    }),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(menu.x))
                        .top(px(menu.y))
                        .w(px(220.0))
                        .rounded_md()
                        .bg(rgb(0x1f1f23))
                        .border_1()
                        .border_color(white().opacity(0.14))
                        .p_1()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |_this, _, _, cx| {
                                cx.stop_propagation();
                            }),
                        )
                        .on_mouse_down(
                            MouseButton::Right,
                            cx.listener(move |_this, _, _, cx| {
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
                                .text_color(white().opacity(0.9))
                                .bg(white().opacity(0.03))
                                .hover(|style| style.bg(white().opacity(0.10)))
                                .cursor_pointer()
                                .child("Duplicate Layer")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        global_for_layer_menu_duplicate.update(cx, |gs, cx| {
                                            gs.select_layer_effect_clip(menu.clip_id);
                                            if gs.duplicate_selected_layer_effect_clip() {
                                                gs.ui_notice =
                                                    Some("Layer duplicated.".to_string());
                                            }
                                            cx.notify();
                                        });
                                        this.layer_clip_menu = None;
                                        cx.notify();
                                    }),
                                ),
                        )
                        .child(
                            div()
                                .h(px(28.0))
                                .rounded_sm()
                                .px_2()
                                .flex()
                                .items_center()
                                .text_sm()
                                .text_color(rgb(0xfca5a5))
                                .bg(white().opacity(0.03))
                                .hover(|style| style.bg(white().opacity(0.10)))
                                .cursor_pointer()
                                .child("Remove Layer")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        global_for_layer_menu_remove.update(cx, |gs, cx| {
                                            gs.select_layer_effect_clip(menu.clip_id);
                                            if gs.remove_selected_layer_effect_clip() {
                                                gs.ui_notice = Some("Layer removed.".to_string());
                                            }
                                            cx.notify();
                                        });
                                        this.layer_clip_menu = None;
                                        cx.notify();
                                    }),
                                ),
                        ),
                )
        } else {
            div()
        };

        let semantic_clip_menu_overlay = if let Some(menu) = self.semantic_clip_menu {
            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left_0()
                .right_0()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.semantic_clip_menu = None;
                        cx.notify();
                    }),
                )
                .on_mouse_down(
                    MouseButton::Right,
                    cx.listener(move |this, _, _, cx| {
                        this.semantic_clip_menu = None;
                        cx.notify();
                    }),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(menu.x))
                        .top(px(menu.y))
                        .w(px(220.0))
                        .rounded_md()
                        .bg(rgb(0x1f1f23))
                        .border_1()
                        .border_color(white().opacity(0.14))
                        .p_1()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |_this, _, _, cx| {
                                cx.stop_propagation();
                            }),
                        )
                        .on_mouse_down(
                            MouseButton::Right,
                            cx.listener(move |_this, _, _, cx| {
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
                                .text_color(white().opacity(0.9))
                                .bg(white().opacity(0.03))
                                .hover(|style| style.bg(white().opacity(0.10)))
                                .cursor_pointer()
                                .child("Duplicate Semantic")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        global_for_semantic_menu_duplicate.update(cx, |gs, cx| {
                                            if gs.duplicate_semantic_clip_after(menu.clip_id) {
                                                gs.ui_notice =
                                                    Some("Semantic duplicated.".to_string());
                                            }
                                            cx.notify();
                                        });
                                        this.semantic_clip_menu = None;
                                        cx.notify();
                                    }),
                                ),
                        )
                        .child(
                            div()
                                .h(px(28.0))
                                .rounded_sm()
                                .px_2()
                                .flex()
                                .items_center()
                                .text_sm()
                                .text_color(rgb(0xfca5a5))
                                .bg(white().opacity(0.03))
                                .hover(|style| style.bg(white().opacity(0.10)))
                                .cursor_pointer()
                                .child("Remove Semantic")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        global_for_semantic_menu_remove.update(cx, |gs, cx| {
                                            gs.select_semantic_clip(menu.clip_id);
                                            if gs.remove_selected_semantic_clip() {
                                                gs.ui_notice =
                                                    Some("Semantic removed.".to_string());
                                            }
                                            cx.notify();
                                        });
                                        this.semantic_clip_menu = None;
                                        cx.notify();
                                    }),
                                ),
                        ),
                )
        } else {
            div()
        };

        let srt_menu_overlay = if self.srt_menu_open {
            let (menu_x, menu_y) = self.srt_menu_anchor.unwrap_or((980.0, 84.0));
            let global_for_srt_import_menu = self.global.clone();
            let global_for_srt_export_menu = self.global.clone();
            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left_0()
                .right_0()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.srt_menu_open = false;
                        this.srt_menu_anchor = None;
                        cx.notify();
                    }),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(menu_x))
                        .top(px(menu_y))
                        .w(px(190.0))
                        .rounded_md()
                        .bg(rgb(0x101318))
                        .border_1()
                        .border_color(white().opacity(0.14))
                        .p_1()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |_this, _, _, cx| {
                                cx.stop_propagation();
                            }),
                        )
                        .child(
                            div()
                                .h(px(28.0))
                                .px_2()
                                .rounded_sm()
                                .text_sm()
                                .text_color(white().opacity(0.9))
                                .hover(|s| s.bg(white().opacity(0.08)))
                                .cursor_pointer()
                                .flex()
                                .items_center()
                                .child("Import SRT")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, win, cx| {
                                        this.srt_menu_open = false;
                                        this.srt_menu_anchor = None;
                                        let rx = cx.prompt_for_paths(PathPromptOptions {
                                            files: true,
                                            directories: false,
                                            multiple: false,
                                            prompt: Some("Import SRT".into()),
                                        });
                                        let global_for_import = global_for_srt_import_menu.clone();
                                        cx.spawn_in(win, async move |view, window| {
                                            let Ok(result) = rx.await else {
                                                return;
                                            };
                                            let paths = match result {
                                                Ok(Some(paths)) => paths,
                                                Ok(None) => return,
                                                Err(err) => {
                                                    eprintln!("[SRT] File picker error: {err}");
                                                    return;
                                                }
                                            };
                                            let Some(path) = paths.into_iter().next() else {
                                                return;
                                            };
                                            let bytes = match fs::read(&path) {
                                                Ok(bytes) => bytes,
                                                Err(err) => {
                                                    eprintln!(
                                                        "[SRT] Failed to read {}: {err}",
                                                        path.display()
                                                    );
                                                    return;
                                                }
                                            };
                                            let srt_text = String::from_utf8_lossy(&bytes).to_string();
                                            let path_label = path.to_string_lossy().to_string();
                                            let _ = view.update_in(window, |_this, _window, cx| {
                                                global_for_import.update(cx, |gs, cx| {
                                                    match gs.import_srt(&srt_text) {
                                                        Ok(count) => {
                                                            gs.ui_notice = Some(format!(
                                                                "Imported {count} cues from {path_label}"
                                                            ));
                                                        }
                                                        Err(err) => {
                                                            gs.ui_notice =
                                                                Some(format!("SRT import failed: {err}"));
                                                        }
                                                    }
                                                    cx.notify();
                                                });
                                            });
                                        })
                                        .detach();
                                        cx.notify();
                                    }),
                                ),
                        )
                        .child(
                            div()
                                .h(px(28.0))
                                .px_2()
                                .rounded_sm()
                                .text_sm()
                                .text_color(white().opacity(0.9))
                                .hover(|s| s.bg(white().opacity(0.08)))
                                .cursor_pointer()
                                .flex()
                                .items_center()
                                .child("Export SRT")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, win, cx| {
                                        this.srt_menu_open = false;
                                        this.srt_menu_anchor = None;
                                        let rx = cx.prompt_for_paths(PathPromptOptions {
                                            files: false,
                                            directories: true,
                                            multiple: false,
                                            prompt: Some("Choose folder for SRT export".into()),
                                        });
                                        let global_for_export = global_for_srt_export_menu.clone();
                                        cx.spawn_in(win, async move |view, window| {
                                            let Ok(result) = rx.await else {
                                                return;
                                            };
                                            let paths = match result {
                                                Ok(Some(paths)) => paths,
                                                Ok(None) => return,
                                                Err(err) => {
                                                    eprintln!("[SRT] Export picker error: {err}");
                                                    return;
                                                }
                                            };
                                            let Some(dir) = paths.into_iter().next() else {
                                                return;
                                            };
                                            let _ = view.update_in(window, |_this, _window, cx| {
                                                global_for_export.update(cx, |gs, cx| {
                                                    match TimelinePanel::build_full_timeline_srt(gs) {
                                                        Ok(srt_text) => {
                                                            let out_path =
                                                                dir.join("timeline_subtitles.srt");
                                                            match fs::write(&out_path, srt_text) {
                                                                Ok(_) => {
                                                                    gs.ui_notice = Some(format!(
                                                                        "SRT exported: {}",
                                                                        out_path.to_string_lossy()
                                                                    ));
                                                                }
                                                                Err(err) => {
                                                                    gs.ui_notice = Some(format!(
                                                                        "SRT export failed: {err}"
                                                                    ));
                                                                }
                                                            }
                                                        }
                                                        Err(err) => {
                                                            gs.ui_notice = Some(err);
                                                        }
                                                    }
                                                    cx.notify();
                                                });
                                            });
                                        })
                                        .detach();
                                        cx.notify();
                                    }),
                                ),
                        ),
                )
        } else {
            div()
        };

        let global_for_play = self.global.clone();
        let global_for_delete = self.global.clone();
        let global_for_unlink = self.global.clone();
        let global_for_tool_select = self.global.clone();
        let global_for_tool_razor = self.global.clone();
        let global_for_tool_sweep = self.global.clone();
        let global_for_keys = self.global.clone();
        let global_for_v1_mode = self.global.clone();

        let global_drop_v1 = self.global.clone();

        // Keep separate clones for move and mouse-up handlers.
        let global_mouse_move = self.global.clone();
        let global_mouse_up = self.global.clone();

        // Restore the dedicated background handle for V1 interactions.
        let global_v1_bg = self.global.clone();
        let global_ruler = self.global.clone();

        let global_add_track = self.global.clone();
        let global_for_lane = self.global.clone();

        let current_scroll_x = self.scroll_offset_x;
        let current_scroll_y = self.scroll_offset_y;
        let timeline_viewport_w = ((_window.viewport_size().width / px(1.0))
            - APP_NAV_W
            - LEFT_TOOL_W
            - TRACK_LIST_W
            - RIGHT_STRIP_W)
            .max(200.0);
        let visible_start_sec = (current_scroll_x / current_px_per_sec.max(0.01)).max(0.0);
        let visible_end_sec = ((current_scroll_x + timeline_viewport_w)
            / current_px_per_sec.max(0.01))
        .max(visible_start_sec + 0.001);
        // Keep one extra viewport-width on both sides so scrolling stays smooth.
        let virtual_pad_sec =
            (timeline_viewport_w / current_px_per_sec.max(0.01)).clamp(10.0, V1_WINDOW_SECS as f32);
        let lane_window_start_sec = (visible_start_sec - virtual_pad_sec).max(0.0);
        let lane_window_end_sec = visible_end_sec + virtual_pad_sec;

        // Show unlink action only when the current selected clip still belongs to a link group.
        let unlink_btn = if selected_clips_have_any_link_group {
            TimelinePanel::transport_btn("Unlink A/V").on_mouse_down(
                MouseButton::Left,
                cx.listener(move |_this, _, _, cx| {
                    global_for_unlink.update(cx, |gs, cx| {
                        if gs.unlink_selected_clips_groups() {
                            gs.ui_notice = Some("A/V link removed.".to_string());
                        }
                        cx.notify();
                    });
                }),
            )
        } else {
            TimelinePanel::transport_btn("Unlink A/V")
                .bg(white().opacity(0.02))
                .text_color(white().opacity(0.35))
        };

        let mut audio_lane_divs = Vec::new();
        for (idx, track) in audio_tracks_data.iter().enumerate() {
            let global_drop = self.global.clone();
            let global_bg_click = self.global.clone();
            let global_for_this_lane = self.global.clone(); // Each lane keeps its own state handle.
            let lane_is_hovered = self.media_pool_hover_track == Some(TrackType::Audio(idx));

            let lane_div = div()
                .w_full()
                .on_drop(cx.listener(move |_this, paths: &ExternalPaths, _, cx| {
                    if let Some(path) = paths.paths().first() {
                        let path_str = path.to_string_lossy().to_string();
                        if !is_supported_media_path(&path_str) {
                            return;
                        }
                        let duration = get_media_duration(&path_str);
                        if duration == Duration::ZERO {
                            return;
                        }
                        global_drop.update(cx, |gs, cx| {
                            gs.load_source_video(path.to_path_buf(), duration);
                            cx.emit(MediaPoolUiEvent::StateChanged);
                            if gs.audio_tracks.get(idx).is_some() {
                                gs.ripple_insert_active_source_audio(idx, duration);
                                cx.notify();
                            }
                        });
                    }
                }))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                        if active_tool == ActiveTool::Select && evt.modifiers.shift {
                            this.start_marquee(evt, win, cx);
                            return;
                        }

                        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                        let raw_x = evt.position.x;
                        let window_x_px = raw_x - px(offset_w);
                        let window_x_f32 = window_x_px / px(1.0);
                        let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);
                        let t = Duration::from_secs_f32((local_x_f32 / this.px_per_sec).max(0.0));

                        // If media pool placement is armed, insert directly into this lane.
                        let mut inserted_from_pool = false;
                        global_bg_click.update(cx, |gs, cx| {
                            gs.is_playing = false;
                            if let Some(path) = gs.pending_media_pool_path.clone()
                                && gs.activate_media_pool_item(&path)
                            {
                                let duration = gs.active_source_duration;
                                gs.set_playhead(t);
                                if gs.audio_tracks.get(idx).is_some() {
                                    gs.ripple_insert_active_source_audio(idx, duration);
                                    gs.clear_media_pool_drag();
                                    cx.emit(MediaPoolUiEvent::StateChanged);
                                    inserted_from_pool = true;
                                    cx.notify();
                                    return;
                                }
                            }
                            gs.set_playhead(t);
                            if gs.active_tool == ActiveTool::Razor {
                                let _ = gs.razor_audio_at_playhead(idx);
                            }
                            cx.notify();
                        });

                        if inserted_from_pool {
                            this.drag_state = DragState::None;
                            this.is_scrubbing = false;
                            cx.notify();
                            return;
                        }

                        if let Some(forward) =
                            Self::track_sweep_direction(active_tool, evt.modifiers.alt)
                        {
                            this.sweep_select_from_anchor(
                                TrackType::Audio(idx),
                                t,
                                forward,
                                true,
                                cx,
                            );
                            this.arm_track_sweep_pending_drag(evt.position.x, t, forward, cx);
                            cx.focus_self(win);
                            return;
                        }

                        this.is_scrubbing = true;
                        this.drag_state = DragState::Scrubbing;
                        cx.focus_self(win);
                    }),
                )
                .child(if timeline_low_load_mode {
                    Self::render_lane_low_load(
                        &track.clips,
                        self.audio_lane_clip_start_index.get(idx).map(Vec::as_slice),
                        self.audio_lane_clip_end_index.get(idx).map(Vec::as_slice),
                        current_px_per_sec,
                        lane_window_start_sec,
                        lane_window_end_sec,
                        true,
                        lane_is_hovered,
                    )
                } else {
                    Self::render_lane(
                        &track.clips,
                        self.audio_lane_clip_start_index.get(idx).map(Vec::as_slice),
                        self.audio_lane_clip_end_index.get(idx).map(Vec::as_slice),
                        &selected_clip_ids,
                        current_px_per_sec,
                        lane_window_start_sec,
                        lane_window_end_sec,
                        true,
                        lane_is_hovered,
                        TrackType::Audio(idx),
                        cx,
                        active_tool,
                        global_for_this_lane,
                    )
                });

            audio_lane_divs.push(lane_div);
        }

        let mut subtitle_lane_divs = Vec::new();
        for (idx, track) in subtitle_tracks_data.iter().enumerate() {
            let global_bg_click = self.global.clone();
            let global_for_this_lane = self.global.clone();

            let lane_div = div()
                .w_full()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                        let raw_x = evt.position.x;
                        let window_x_px = raw_x - px(offset_w);
                        let window_x_f32 = window_x_px / px(1.0);
                        let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);
                        let t = Duration::from_secs_f32((local_x_f32 / this.px_per_sec).max(0.0));

                        if this.pending_subtitle_drop {
                            this.pending_subtitle_drop = false;
                            global_bg_click.update(cx, |gs, cx| {
                                let _ = gs.add_subtitle_clip(
                                    idx,
                                    t,
                                    Duration::from_secs(5),
                                    "Subtitle".to_string(),
                                );
                                gs.set_playhead(t);
                                cx.notify();
                            });
                            cx.notify();
                            return;
                        }

                        if let Some(forward) =
                            Self::track_sweep_direction(active_tool, evt.modifiers.alt)
                        {
                            this.sweep_select_from_anchor(
                                TrackType::Subtitle(idx),
                                t,
                                forward,
                                true,
                                cx,
                            );
                            this.arm_track_sweep_pending_drag(evt.position.x, t, forward, cx);
                            cx.focus_self(win);
                            return;
                        }

                        if active_tool == ActiveTool::Select && evt.modifiers.shift {
                            this.start_marquee(evt, win, cx);
                            return;
                        }

                        this.is_scrubbing = true;
                        this.drag_state = DragState::Scrubbing;
                        cx.focus_self(win);

                        global_bg_click.update(cx, |gs, cx| {
                            gs.is_playing = false;
                            gs.set_playhead(t);
                            if gs.active_tool == ActiveTool::Razor {
                                let _ = gs.razor_subtitle_at_playhead(idx);
                            }
                            cx.notify();
                        });
                    }),
                )
                .child(if timeline_low_load_mode {
                    Self::render_subtitle_lane_low_load(
                        &track.clips,
                        current_px_per_sec,
                        lane_window_start_sec,
                        lane_window_end_sec,
                    )
                } else {
                    Self::render_subtitle_lane(
                        &track.clips,
                        &selected_subtitle_ids,
                        current_px_per_sec,
                        lane_window_start_sec,
                        lane_window_end_sec,
                        idx,
                        cx,
                        active_tool,
                        global_for_this_lane,
                    )
                });

            subtitle_lane_divs.push(lane_div);
        }
        subtitle_lane_divs.reverse();
        self.ensure_audio_gain_sliders(cx);

        let mut subtitle_header_rows = Vec::new();
        for idx in (0..subtitle_tracks_data.len()).rev() {
            let track = &subtitle_tracks_data[idx];
            subtitle_header_rows.push(TimelinePanel::track_header_row(
                &TrackUi {
                    name: track.name.clone(),
                    kind: TrackHeaderKind::Subtitle(idx),
                },
                cx,
                &self.global,
            ));
        }

        let mut video_header_rows = Vec::new();
        for idx in (0..video_tracks_data.len()).rev() {
            let track = &video_tracks_data[idx];
            video_header_rows.push(TimelinePanel::track_header_row(
                &TrackUi {
                    name: track.name.clone(),
                    kind: TrackHeaderKind::VideoOverlay(idx),
                },
                cx,
                &self.global,
            ));
        }

        let mut audio_header_rows = Vec::new();
        for (idx, track) in audio_tracks_data.iter().enumerate() {
            audio_header_rows.push(TimelinePanel::track_header_row(
                &TrackUi {
                    name: track.name.clone(),
                    kind: TrackHeaderKind::Audio(idx),
                },
                cx,
                &self.global,
            ));
        }

        let semantic_header_row = if self.show_semantic_lane {
            Some(TimelinePanel::track_header_row(
                &TrackUi {
                    name: "SL".to_string(),
                    kind: TrackHeaderKind::Semantic,
                },
                cx,
                &self.global,
            ))
        } else {
            None
        };

        let track_db_summary = {
            let gs = self.global.read(cx);
            Self::all_tracks_db_summary(gs, playhead)
        };
        let audio_gain_rows: Vec<(String, String, f32, Entity<SliderState>)> =
            if timeline_low_load_mode {
                Vec::new()
            } else {
                let gs = self.global.read(cx);
                gs.audio_tracks
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, track)| {
                        self.audio_gain_sliders.get(&idx).cloned().map(|slider| {
                            (
                                track.name.clone(),
                                Self::audio_track_db_label(gs, idx, playhead),
                                gs.get_audio_track_gain_db(idx)
                                    .clamp(AUDIO_TRACK_GAIN_MIN_DB, AUDIO_TRACK_GAIN_MAX_DB),
                                slider,
                            )
                        })
                    })
                    .collect()
            };
        if !timeline_low_load_mode {
            for (_, _, gain_db, slider) in &audio_gain_rows {
                if (slider.read(cx).value().start() - *gain_db).abs() > 0.01 {
                    slider.update(cx, |state, cx| state.set_value(*gain_db, _window, cx));
                }
            }
        }
        let audio_gain_controls: Vec<gpui::Div> = if timeline_low_load_mode {
            Vec::new()
        } else {
            audio_gain_rows
                .iter()
                .map(|(track_name, meter_db, gain_db, slider)| {
                    div()
                        .h(px(18.0))
                        .px_2()
                        .rounded_sm()
                        .border_1()
                        .border_color(white().opacity(0.08))
                        .bg(white().opacity(0.02))
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(
                            div()
                                .font_family("Mono")
                                .text_xs()
                                .text_color(white().opacity(0.82))
                                .child(format!("{track_name} {meter_db}")),
                        )
                        .child(Slider::new(slider).horizontal().h(px(14.0)).w(px(96.0)))
                        .child(
                            div()
                                .font_family("Mono")
                                .text_xs()
                                .text_color(rgb(0x93c5fd))
                                .child(format!("{:+.1} dB", gain_db)),
                        )
                })
                .collect()
        };
        let source_hint_text = format!("Source: {}", active_source_name);

        let global_for_trim_to_fit = self.global.clone();
        let status_view = {
            // Keep status text bounded so long paths/messages do not stretch the toolbar row.
            let show_status_text = ui_notice.is_some()
                || export_in_progress
                || export_last_error.is_some()
                || export_last_out.is_some();
            let mut s = div().flex().items_center().gap_2().min_w_0();
            if show_status_text {
                s = s.w(px(320.0)).flex_shrink_0();
                if let Some(note) = ui_notice {
                    let note_single = note.lines().next().unwrap_or_default().to_string();
                    s = s.child(
                        div()
                            .w_full()
                            .truncate()
                            .text_xs()
                            .text_color(rgb(0xfbbf24))
                            .child(note_single),
                    );
                } else if export_in_progress {
                    let pct = (export_progress_ratio * 100.0).round() as u32;
                    let eta_text = export_eta
                        .map(|d| format!(" ETA {}", fmt_mmss(d)))
                        .unwrap_or_default();
                    s = s.child(
                        div()
                            .w_full()
                            .truncate()
                            .text_xs()
                            .text_color(rgb(0xfbbf24))
                            .child(format!("Exporting… {pct}%{eta_text}")),
                    );
                } else if let Some(err) = export_last_error {
                    let err_single = err.lines().next().unwrap_or("Export failed.").to_string();
                    s = s.child(
                        div()
                            .w_full()
                            .truncate()
                            .text_xs()
                            .text_color(rgb(0xf87171))
                            .child(format!("Error: {err_single}")),
                    );
                } else if let Some(p) = export_last_out {
                    s = s.child(
                        div()
                            .w_full()
                            .truncate()
                            .text_xs()
                            .text_color(rgb(0x86efac))
                            .child(format!("Exported: {p}")),
                    );
                }
            }
            if pending_trim_to_fit.is_some() {
                s = s.child(
                    TimelinePanel::transport_btn("Trim to Fit Transition").on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |_this, _, _, cx| {
                            global_for_trim_to_fit.update(cx, |gs, cx| {
                                if gs.apply_pending_trim_to_fit() {
                                    cx.notify();
                                }
                            });
                        }),
                    ),
                );
            }
            s
        };

        div()
            .track_focus(&self.focus_handle)
            .w_full()
            .relative()
            .bg(rgb(0x0b0b0d))
            .border_t_1()
            .border_color(white().opacity(0.10))
            // ===================
            .on_key_down(cx.listener(move |this, evt: &KeyDownEvent, _win, cx| {
                let key = evt.keystroke.key.as_str();
                let m = evt.keystroke.modifiers;

                // Extract all keyboard modifiers used by timeline shortcuts.
                let control  = m.control;
                let _alt      = m.alt;      // Meta / Option
                let shift    = m.shift;
                let platform = m.platform; // Mac: Command, Win: Windows Key
                let _function = m.function; // Fn key

                // Define a cross-platform action key: Cmd on macOS, Ctrl elsewhere.
                // This keeps undo/redo shortcuts consistent across platforms.
                let is_cmd_or_ctrl = platform || control;

                // [Zoom Logic] Zoom the timeline
                match (key, is_cmd_or_ctrl) {
                     ("=", true) => {
                        this.px_per_sec = (this.px_per_sec * 1.25).clamp(0.01, 1000.0);
                        cx.notify();
                    },
                    ("-", true) => {
                        this.px_per_sec = (this.px_per_sec * 0.8).clamp(0.01, 1000.0);
                        cx.notify();
                    },
                    _ => {}
                }

                // Cross-platform copy/paste:
                // macOS => Command+C / Command+V
                // others => Ctrl+C / Ctrl+V
                if is_cmd_or_ctrl && !shift && key.eq_ignore_ascii_case("c") {
                    this.copy_selected_timeline_item(cx);
                    return;
                }
                if is_cmd_or_ctrl && !shift && key.eq_ignore_ascii_case("v") {
                    this.paste_copied_timeline_item(cx);
                    return;
                }
                if !is_cmd_or_ctrl && !shift && key.eq_ignore_ascii_case("j") {
                    global_for_keys.update(cx, |gs, cx| {
                        gs.mark_semantic_start_at_playhead();
                        cx.notify();
                    });
                    return;
                }
                if !is_cmd_or_ctrl && !shift && key.eq_ignore_ascii_case("k") {
                    global_for_keys.update(cx, |gs, cx| {
                        let _ = gs.commit_semantic_segment_at_playhead();
                        cx.notify();
                    });
                    return;
                }

                // 3. Global Actions (Undo, Play, etc.)
                global_for_keys.update(cx, |gs, cx| {

                    // Match on key, action-key state, and shift state together.
                    match (key, is_cmd_or_ctrl, shift) {

                        // --- Undo: Cmd/Ctrl + Z (without Shift) ---
                        ("z", true, false) => {
                            println!("[User Action] Undo");
                            gs.undo();
                            cx.notify();
                        },

                        // --- Redo: Cmd/Ctrl + Shift + Z or Cmd/Ctrl + Y ---
                        ("z", true, true) | ("y", true, _) => {
                            println!("[User Action] Redo");
                            gs.redo();
                            cx.notify();
                        },

                        // --- Play / pause (Space) ---
                        ("space", _, _) => {
                            gs.toggle_playing();
                            cx.notify();
                        },

                        // --- macOS preview render mode toggle (Cmd/Ctrl+B) ---
                        ("b", true, false) => {
                            #[cfg(target_os = "macos")]
                            {
                                let next_mode = gs.toggle_mac_preview_render_mode();
                                gs.ui_notice = Some(format!(
                                    "macOS preview render mode: {}",
                                    next_mode.label()
                                ));
                                cx.notify();
                            }
                        },

                        // --- Delete (Backspace / Delete) ---
                        ("backspace", _, _) | ("delete", _, _) => {
                            if gs.layer_effect_clip_selected() {
                                let _ = gs.remove_selected_layer_effect_clip();
                            } else if gs.selected_semantic_clip_id().is_some() {
                                let _ = gs.remove_selected_semantic_clip();
                            } else {
                                let _ = gs.delete_selected_items();
                            }
                            cx.notify();
                        },

                        // --- Navigation: move the playhead with arrow keys ---

                        // Fine step (1 frame): no Shift
                        ("left", false, false) => {
                            let fps = gs.preview_fps.value().max(1) as f64;
                            let dt = Duration::from_secs_f64(1.0 / fps);
                            gs.set_playhead(gs.playhead.saturating_sub(dt)); cx.notify();
                        },
                        ("right", false, false) => {
                            let fps = gs.preview_fps.value().max(1) as f64;
                            let dt = Duration::from_secs_f64(1.0 / fps);
                            gs.set_playhead(gs.playhead + dt); cx.notify();
                        },

                        // Fast step (1 second): hold Shift
                        ("left", false, true) => {
                            gs.set_playhead(gs.playhead.saturating_sub(Duration::from_secs(1))); cx.notify();
                        },
                        ("right", false, true) => {
                            gs.set_playhead(gs.playhead + Duration::from_secs(1)); cx.notify();
                        },

                        _ => {}
                    }
                });
            }))
            // ===================

            // Mouse Move Logic (Scrubbing / Dragging)
            .on_mouse_move(cx.listener(move |this, evt: &MouseMoveEvent, win, cx| {
                // Track hover lane/time while a media pool drag is active.
                let drag_path = global_mouse_move.read(cx).media_pool_drag.as_ref().map(|d| d.path.clone());
                if drag_path.is_some() {
                    let mut next_hover_track = None;
                    let mut next_hover_time = None;
                    if evt.dragging() {
                        let gs = global_mouse_move.read(cx);
                        if let Some((track, time)) = this.media_drop_target_at_point(gs, win, evt.position)
                            && matches!(track, TrackType::V1 | TrackType::VideoOverlay(_) | TrackType::Audio(_))
                        {
                            next_hover_track = Some(track);
                            next_hover_time = Some(time);
                        }
                    }
                    if this.media_pool_hover_track != next_hover_track
                        || this.media_pool_hover_time != next_hover_time
                    {
                        this.media_pool_hover_track = next_hover_track;
                        this.media_pool_hover_time = next_hover_time;
                        cx.notify();
                    }
                } else if this.media_pool_hover_track.is_some() || this.media_pool_hover_time.is_some() {
                    this.media_pool_hover_track = None;
                    this.media_pool_hover_time = None;
                    cx.notify();
                }

                match &this.drag_state {
                    DragState::Scrubbing => {
                        if evt.dragging() {
                            let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                            let raw_x = evt.position.x;
                            let window_x_px = raw_x - px(offset_w);
                            let window_x_f32 = window_x_px / px(1.0);
                            let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);
                            let t = Duration::from_secs_f32((local_x_f32 / this.px_per_sec).max(0.0));

                            global_mouse_move.update(cx, |gs, cx| {
                                gs.is_playing = false;
                                gs.set_playhead(t);
                                cx.notify();
                            });
                        }
                    },
                    DragState::Pending { start_x, clip_id, track_type, original_start, group_items } => {
                        let current_x = evt.position.x;
                        let start_x_f32 = f32::from(*start_x);
                        let current_x_f32 = f32::from(current_x);

                        if (current_x_f32 - start_x_f32).abs() > 3.0 {

                            // - Save undo state only once per drag action
                            // -------
                            global_mouse_move.update(cx, |gs, _cx| {
                                gs.save_for_undo();
                            });
                            // -------
                            let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                            let raw_x_val = start_x_f32 - offset_w + this.scroll_offset_x;
                            let start_local_x = raw_x_val.max(0.0);

                            let start_time_sec = (start_local_x / this.px_per_sec) as f64;
                            let offset_sec = start_time_sec - original_start.as_secs_f64();

                            if let Some(items) = group_items.clone() {
                                if !items.is_empty() {
                                    this.drag_state = DragState::GroupDragging {
                                        items,
                                        anchor_start: *original_start,
                                        offset_seconds: offset_sec,
                                    };
                                } else {
                                    this.drag_state = DragState::Dragging {
                                        clip_id: *clip_id,
                                        track_type: *track_type,
                                        offset_seconds: offset_sec,
                                    };
                                }
                            } else {
                                this.drag_state = DragState::Dragging {
                                    clip_id: *clip_id,
                                    track_type: *track_type,
                                    offset_seconds: offset_sec,
                                };
                            }
                        }
                    },
                    DragState::Dragging { clip_id, track_type, offset_seconds } => {
                        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                        let raw_x = evt.position.x;
                        let window_x_px = raw_x - px(offset_w);
                        let window_x_f32 = window_x_px / px(1.0);
                        let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);

                        let mouse_time_sec = (local_x_f32 / this.px_per_sec) as f64;

                        let new_start_sec = (mouse_time_sec - offset_seconds).max(0.0);
                        let new_start = Duration::from_secs_f64(new_start_sec);
                        let alt_invert = evt.modifiers.alt;
                        let target_overlay_track = if matches!(track_type, TrackType::VideoOverlay(_))
                        {
                            let gs = global_mouse_move.read(cx);
                            match this.media_drop_target_at_point(gs, win, evt.position) {
                                Some((TrackType::VideoOverlay(idx), _)) => Some(idx),
                                _ => None,
                            }
                        } else {
                            None
                        };

                        global_mouse_move.update(cx, |gs, cx| {

                            match track_type {
                                TrackType::V1 => {
                                    // Use toolbar mode by default and allow Alt/Option to invert behavior for one drag gesture.
                                    if gs.effective_v1_magnetic(alt_invert) {
                                        gs.move_v1_clip_magnetic(*clip_id, new_start);
                                    } else {
                                        gs.move_v1_clip_free(*clip_id, new_start);
                                    }
                                },
                                TrackType::Audio(idx) => {
                                    gs.move_audio_clip_free(*idx, *clip_id, new_start);
                                },
                                TrackType::VideoOverlay(_idx) => {
                                    gs.move_video_clip_free_any_track(
                                        *clip_id,
                                        new_start,
                                        target_overlay_track,
                                    );
                                },
                                TrackType::Subtitle(idx) => {
                                    gs.move_subtitle_clip_free(*idx, *clip_id, new_start);
                                }
                            }
                            cx.notify();
                        });
                    },
                    DragState::GroupDragging { items, anchor_start, offset_seconds } => {
                        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                        let raw_x = evt.position.x;
                        let window_x_px = raw_x - px(offset_w);
                        let window_x_f32 = window_x_px / px(1.0);
                        let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);

                        let mouse_time_sec = (local_x_f32 / this.px_per_sec) as f64;
                        let new_anchor_start_sec = (mouse_time_sec - offset_seconds).max(0.0);
                        let mut delta_sec = new_anchor_start_sec - anchor_start.as_secs_f64();

                        let min_start_sec = items
                            .iter()
                            .map(|item| item.original_start.as_secs_f64())
                            .fold(f64::INFINITY, f64::min);
                        if min_start_sec.is_finite() && min_start_sec + delta_sec < 0.0 {
                            delta_sec = -min_start_sec;
                        }
                        let alt_invert = evt.modifiers.alt;

                        global_mouse_move.update(cx, |gs, cx| {
                            let has_v1 = items.iter().any(|item| {
                                matches!(item.kind, GroupDragKind::Timeline(TrackType::V1))
                            });
                            let use_v1_magnetic = has_v1 && gs.effective_v1_magnetic(alt_invert);

                            if use_v1_magnetic {
                                // Force V1 clips in group drag to use magnetic reorder semantics.
                                for item in items.iter().filter(|item| {
                                    matches!(item.kind, GroupDragKind::Timeline(TrackType::V1))
                                }) {
                                    let new_start_sec =
                                        (item.original_start.as_secs_f64() + delta_sec).max(0.0);
                                    let new_start = Duration::from_secs_f64(new_start_sec);
                                    gs.move_v1_clip_magnetic(item.clip_id, new_start);
                                }

                                // Non-V1 clips still follow the drag delta in magnetic mode.
                                for item in items
                                    .iter()
                                    .filter(|item| !matches!(item.kind, GroupDragKind::Timeline(TrackType::V1)))
                                {
                                    let new_start_sec =
                                        (item.original_start.as_secs_f64() + delta_sec).max(0.0);
                                    let new_start = Duration::from_secs_f64(new_start_sec);
                                    match item.kind {
                                        GroupDragKind::Timeline(TrackType::Audio(idx)) => {
                                            gs.move_audio_clip_free(idx, item.clip_id, new_start)
                                        }
                                        GroupDragKind::Timeline(TrackType::VideoOverlay(idx)) => {
                                            gs.move_video_clip_free(idx, item.clip_id, new_start)
                                        }
                                        GroupDragKind::Timeline(TrackType::Subtitle(idx)) => {
                                            gs.move_subtitle_clip_free(idx, item.clip_id, new_start)
                                        }
                                        GroupDragKind::Timeline(TrackType::V1) => {}
                                        GroupDragKind::LayerEffect => {
                                            let _ =
                                                gs.move_layer_effect_clip(item.clip_id, new_start, None);
                                        }
                                        GroupDragKind::Semantic => {
                                            let _ = gs.move_semantic_clip(item.clip_id, new_start);
                                        }
                                    }
                                }
                            } else {
                                for item in items {
                                    let new_start_sec =
                                        (item.original_start.as_secs_f64() + delta_sec).max(0.0);
                                    let new_start = Duration::from_secs_f64(new_start_sec);
                                    match item.kind {
                                        GroupDragKind::Timeline(TrackType::V1) => {
                                            gs.move_v1_clip_free(item.clip_id, new_start)
                                        }
                                        GroupDragKind::Timeline(TrackType::Audio(idx)) => {
                                            gs.move_audio_clip_free(idx, item.clip_id, new_start)
                                        }
                                        GroupDragKind::Timeline(TrackType::VideoOverlay(idx)) => {
                                            gs.move_video_clip_free(idx, item.clip_id, new_start)
                                        }
                                        GroupDragKind::Timeline(TrackType::Subtitle(idx)) => {
                                            gs.move_subtitle_clip_free(idx, item.clip_id, new_start)
                                        }
                                        GroupDragKind::LayerEffect => {
                                            let _ =
                                                gs.move_layer_effect_clip(item.clip_id, new_start, None);
                                        }
                                        GroupDragKind::Semantic => {
                                            let _ = gs.move_semantic_clip(item.clip_id, new_start);
                                        }
                                    }
                                }
                            }
                            cx.notify();
                        });
                    },
                    DragState::LayerPending {
                        start_x,
                        clip_id,
                        original_start,
                    } => {
                        let current_x = evt.position.x;
                        let start_x_f32 = f32::from(*start_x);
                        let current_x_f32 = f32::from(current_x);
                        if (current_x_f32 - start_x_f32).abs() > 3.0 {
                            global_mouse_move.update(cx, |gs, _cx| {
                                gs.save_for_undo();
                            });
                            let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                            let raw_x_val = start_x_f32 - offset_w + this.scroll_offset_x;
                            let start_local_x = raw_x_val.max(0.0);
                            let start_time_sec = (start_local_x / this.px_per_sec) as f64;
                            let offset_sec = start_time_sec - original_start.as_secs_f64();
                            this.drag_state = DragState::LayerDragging {
                                clip_id: *clip_id,
                                offset_seconds: offset_sec,
                            };
                        }
                    }
                    DragState::LayerDragging {
                        clip_id,
                        offset_seconds,
                    } => {
                        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                        let raw_x = evt.position.x;
                        let window_x_px = raw_x - px(offset_w);
                        let window_x_f32 = window_x_px / px(1.0);
                        let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);
                        let mouse_time_sec = (local_x_f32 / this.px_per_sec) as f64;
                        let new_start_sec = (mouse_time_sec - offset_seconds).max(0.0);
                        let new_start = Duration::from_secs_f64(new_start_sec);
                        let target_overlay_track = {
                            let gs = global_mouse_move.read(cx);
                            match this.media_drop_target_at_point(gs, win, evt.position) {
                                Some((TrackType::VideoOverlay(idx), _)) => Some(idx),
                                _ => None,
                            }
                        };
                        global_mouse_move.update(cx, |gs, cx| {
                            let _ = gs.move_layer_effect_clip(*clip_id, new_start, target_overlay_track);
                            cx.notify();
                        });
                    }
                    DragState::LayerResizing {
                        clip_id,
                        start_x,
                        original_duration,
                    } => {
                        let current_x = evt.position.x;
                        let diff_px = (current_x - *start_x) / px(1.0);
                        let diff_secs = diff_px / this.px_per_sec;
                        let new_dur_secs =
                            (original_duration.as_secs_f64() + diff_secs as f64).max(0.1);
                        let new_duration = Duration::from_secs_f64(new_dur_secs);
                        global_mouse_move.update(cx, |gs, cx| {
                            let _ = gs.resize_layer_effect_clip(*clip_id, new_duration);
                            cx.notify();
                        });
                    }
                    DragState::SemanticPending {
                        start_x,
                        clip_id,
                        original_start,
                    } => {
                        let current_x = evt.position.x;
                        let start_x_f32 = f32::from(*start_x);
                        let current_x_f32 = f32::from(current_x);
                        if (current_x_f32 - start_x_f32).abs() > 3.0 {
                            global_mouse_move.update(cx, |gs, _cx| {
                                gs.save_for_undo();
                            });
                            let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                            let raw_x_val = start_x_f32 - offset_w + this.scroll_offset_x;
                            let start_local_x = raw_x_val.max(0.0);
                            let start_time_sec = (start_local_x / this.px_per_sec) as f64;
                            let offset_sec = start_time_sec - original_start.as_secs_f64();
                            this.drag_state = DragState::SemanticDragging {
                                clip_id: *clip_id,
                                offset_seconds: offset_sec,
                            };
                        }
                    }
                    DragState::SemanticDragging {
                        clip_id,
                        offset_seconds,
                    } => {
                        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                        let raw_x = evt.position.x;
                        let window_x_px = raw_x - px(offset_w);
                        let window_x_f32 = window_x_px / px(1.0);
                        let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);
                        let mouse_time_sec = (local_x_f32 / this.px_per_sec) as f64;
                        let new_start_sec = (mouse_time_sec - offset_seconds).max(0.0);
                        let new_start = Duration::from_secs_f64(new_start_sec);
                        global_mouse_move.update(cx, |gs, cx| {
                            let _ = gs.move_semantic_clip(*clip_id, new_start);
                            cx.notify();
                        });
                    }
                    DragState::SemanticResizing {
                        clip_id,
                        start_x,
                        original_duration,
                    } => {
                        let current_x = evt.position.x;
                        let diff_px = (current_x - *start_x) / px(1.0);
                        let diff_secs = diff_px / this.px_per_sec;
                        let new_dur_secs =
                            (original_duration.as_secs_f64() + diff_secs as f64).max(0.001);
                        let new_duration = Duration::from_secs_f64(new_dur_secs);
                        global_mouse_move.update(cx, |gs, cx| {
                            let _ = gs.resize_semantic_clip(*clip_id, new_duration);
                            cx.notify();
                        });
                    }
                    // 4. Resizing
                    DragState::Resizing { clip_id, track_type, start_x, original_duration } => {
                        let current_x = evt.position.x;

                        // Convert the mouse delta into pixels.
                        let diff_px = (current_x - *start_x) / px(1.0);

                        // Convert the pixel delta into seconds.
                        let diff_secs = diff_px / this.px_per_sec;

                        // Compute the new duration from the original duration plus drag delta.
                        // Clamp to 0.1s so the clip cannot collapse to zero width.
                        let new_dur_secs = (original_duration.as_secs_f64() + diff_secs as f64).max(0.1);
                        let new_dur = Duration::from_secs_f64(new_dur_secs);

                        global_mouse_move.update(cx, |gs, cx| {
                            match track_type {
                                TrackType::Subtitle(idx) => gs.resize_subtitle_clip(*idx, *clip_id, new_dur),
                                _ => gs.resize_clip(*track_type, *clip_id, new_dur),
                            }
                            cx.notify();
                        });
                    },
                    DragState::Marquee { start, .. } => {
                        if evt.dragging() {
                            this.drag_state = DragState::Marquee {
                                start: *start,
                                current: evt.position,
                            };
                            cx.notify();
                        }
                    }
                    DragState::None => {}
                }
            }))
            .on_mouse_up(MouseButton::Left, cx.listener(move |this, evt: &MouseUpEvent, win, cx| {
                let track_sweep_active =
                    matches!(global_mouse_up.read(cx).active_tool, ActiveTool::TrackSweep);
                // Handle internal media-pool drag/drop before other timeline mouse-up behaviors.
                let media_drag_path = global_mouse_up
                    .read(cx)
                    .media_pool_drag
                    .as_ref()
                    .map(|d| d.path.clone());
                if let Some(path) = media_drag_path {
                    let mut drop_applied = false;
                    global_mouse_up.update(cx, |gs, cx| {
                        if let Some((track, t)) = this.media_drop_target_at_point(gs, win, evt.position)
                            && gs.activate_media_pool_item(&path)
                        {
                            let duration = gs.active_source_duration;
                            gs.set_playhead(t);
                            match track {
                                TrackType::V1 => {
                                    // Route V1 insertion through the active move mode (Magnetic/Free).
                                    gs.insert_active_source_v1(duration);
                                    drop_applied = true;
                                }
                                TrackType::VideoOverlay(idx) => {
                                    gs.ripple_insert_active_source_video(idx, duration);
                                    drop_applied = true;
                                }
                                TrackType::Audio(idx) => {
                                    if gs.audio_tracks.get(idx).is_some() {
                                        gs.ripple_insert_active_source_audio(idx, duration);
                                        drop_applied = true;
                                    }
                                }
                                TrackType::Subtitle(_) => {}
                            }
                        }
                        gs.clear_media_pool_drag();
                        cx.emit(MediaPoolUiEvent::StateChanged);
                        cx.notify();
                    });

                    this.media_pool_hover_track = None;
                    this.media_pool_hover_time = None;
                    this.drag_state = DragState::None;
                    this.is_scrubbing = false;
                    return;
                }

                let mut handled_transition = false;

                if let Some(pending) = global_mouse_up.read(cx).pending_transition {
                    global_mouse_up.update(cx, |gs, cx| {
                        if let Some(layer_clip_id) = this.layer_clip_at_point(gs, win, evt.position) {
                            let _ = pending;
                            let _ = layer_clip_id;
                            gs.clear_transition_drag();
                            gs.ui_notice = Some(
                                "Layer FX no longer supports Dissolve. Use Layer FX tab Fade In / Fade Out."
                                    .to_string(),
                            );
                            handled_transition = true;
                        } else {
                            // Resolve a robust transition target: hovered clip first, selected clip fallback.
                            let mut target_clip_id: Option<u64> = None;
                            if let Some((clip_id, track_type)) =
                                this.clip_at_point(gs, win, evt.position)
                            {
                                if matches!(track_type, TrackType::V1 | TrackType::VideoOverlay(_)) {
                                    target_clip_id = Some(clip_id);
                                }
                            } else if let Some(selected_id) = gs.selected_clip_id {
                                let selected_on_video = gs.v1_clips.iter().any(|c| c.id == selected_id)
                                    || gs
                                        .video_tracks
                                        .iter()
                                        .any(|track| track.clips.iter().any(|c| c.id == selected_id));
                                if selected_on_video {
                                    target_clip_id = Some(selected_id);
                                }
                            }

                            if let Some(clip_id) = target_clip_id {
                                if gs.apply_transition_to_clip(clip_id, pending) {
                                    gs.clear_layer_effect_clip_selection();
                                    gs.selected_clip_id = Some(clip_id);
                                    gs.selected_clip_ids = vec![clip_id];
                                    gs.selected_subtitle_id = None;
                                    gs.selected_subtitle_ids.clear();
                                    gs.clear_transition_drag();
                                    handled_transition = true;
                                } else {
                                    gs.ui_notice =
                                        Some("Cannot apply transition on this clip.".to_string());
                                }
                            } else {
                                gs.ui_notice = Some(
                                    "Drop on a video clip (V1/V2+), LAYER FX clip, or select a clip first."
                                        .to_string(),
                                );
                            }
                        }
                        cx.notify();
                    });
                }

                if handled_transition {
                    this.drag_state = DragState::None;
                    this.is_scrubbing = false;
                    return;
                }

                if let DragState::Marquee { .. } = &this.drag_state {
                    if let Some(bounds) = this.marquee_bounds_in_panel(win) {
                        global_mouse_up.update(cx, |gs, cx| {
                            let (selected_clip_ids, selected_subtitle_ids) =
                                this.collect_marquee_selection(bounds, gs);
                            gs.clear_layer_effect_clip_selection();
                            gs.selected_clip_ids = selected_clip_ids.clone();
                            gs.selected_subtitle_ids = selected_subtitle_ids.clone();
                            gs.selected_clip_id = selected_clip_ids.last().copied();
                            gs.selected_subtitle_id = selected_subtitle_ids.last().copied();
                            cx.notify();
                        });
                    }
                } else if let DragState::LayerPending { clip_id, .. } = &this.drag_state {
                    global_mouse_up.update(cx, |gs, cx| {
                        gs.select_layer_effect_clip(*clip_id);
                        cx.notify();
                    });
                } else if let DragState::SemanticPending { clip_id, .. } = &this.drag_state {
                    global_mouse_up.update(cx, |gs, cx| {
                        gs.select_semantic_clip(*clip_id);
                        cx.notify();
                    });
                } else if let DragState::Pending { clip_id, track_type, group_items, .. } = &this.drag_state {
                    global_mouse_up.update(cx, |gs, cx| {
                        // match track_idx {
                        //     None => gs.select_v1_clip_at(gs.playhead),
                        //     Some(idx) => gs.select_audio_clip_at(*idx, gs.playhead),
                        // }
                        if group_items.is_none() {
                            match track_type {
                                TrackType::V1 => gs.select_v1_clip_at(gs.playhead),
                                TrackType::Audio(idx) => gs.select_audio_clip_at(*idx, gs.playhead),
                                TrackType::VideoOverlay(idx) => gs.select_video_clip_at(*idx, gs.playhead),
                                TrackType::Subtitle(idx) => gs.select_subtitle_clip_at(*idx, gs.playhead),
                            }
                        } else {
                            gs.clear_layer_effect_clip_selection();
                        }
                        if matches!(track_type, TrackType::Subtitle(_)) {
                            gs.selected_subtitle_id = Some(*clip_id);
                            if group_items.is_none() {
                                gs.selected_subtitle_ids = vec![*clip_id];
                                gs.selected_clip_id = None;
                                gs.selected_clip_ids.clear();
                            }
                        } else {
                            gs.selected_clip_id = Some(*clip_id);
                            if group_items.is_none() {
                                gs.selected_clip_ids = vec![*clip_id];
                                gs.selected_subtitle_id = None;
                                gs.selected_subtitle_ids.clear();
                            }
                        }
                        cx.notify();
                    });
                }

                if track_sweep_active {
                    global_mouse_up.update(cx, |gs, cx| {
                        gs.selected_clip_id = None;
                        gs.selected_subtitle_id = None;
                        gs.selected_layer_effect_clip_id = None;
                        gs.clear_semantic_clip_selection();
                        gs.selected_clip_ids.clear();
                        gs.selected_subtitle_ids.clear();
                        cx.notify();
                    });
                }

                this.drag_state = DragState::None;
                this.is_scrubbing = false;
            }))

            // --- Header ---
            .child(
                div()
                    .h(px(TIMELINE_HEADER_H))
                    .px_3()
                    .flex()
                    .items_start()
                    .pt_2()
                    .relative()
                    .border_b_1()
                    .border_color(white().opacity(0.08))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .flex_shrink_0()
                            .gap_3()
                            .child(div().text_sm().text_color(white().opacity(0.85)).child("Time"))
                            .child(div().font_family("Mono").text_size(px(18.0)).text_color(rgb(0x3b82f6)).child(fmt_mmss_millis(playhead))),
                    )


                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .pl_3()
                            // Keep timeline actions usable on narrow windows via horizontal scrolling.
                            .overflow_x_scrollbar()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(TimelinePanel::transport_btn(play_label).on_mouse_down(MouseButton::Left, cx.listener(move |_this, _, _, cx| { global_for_play.update(cx, |gs, cx| { gs.toggle_playing(); cx.notify(); }); })))
                                    .child(TimelinePanel::transport_btn("Delete").on_mouse_down(MouseButton::Left, cx.listener(move |_this, _, _, cx| { global_for_delete.update(cx, |gs, cx| { if gs.layer_effect_clip_selected() { let _ = gs.remove_selected_layer_effect_clip(); } else if gs.selected_semantic_clip_id().is_some() { let _ = gs.remove_selected_semantic_clip(); } else { let _ = gs.delete_selected_items(); } cx.notify(); }); })))
                                    .child(unlink_btn)
                                    .child(
                                        div()
                                            .h(px(28.0))
                                            .px_3()
                                            .rounded_lg()
                                            .border_1()
                                            .border_color(white().opacity(0.12))
                                            .bg(white().opacity(0.05))
                                            .text_color(white().opacity(0.85))
                                            .hover(|s| s.bg(white().opacity(0.10)))
                                            .cursor_pointer()
                                            .flex()
                                            .items_center()
                                            .justify_center()
                                            .child(display_settings_label)
                                            .on_mouse_down(MouseButton::Left, cx.listener(move |_this, _, _, cx| {
                                                let (canvas_w, canvas_h, preview_fps) = {
                                                    let gs = _this.global.read(cx);
                                                    (gs.canvas_w, gs.canvas_h, gs.preview_fps.value())
                                                };
                                                _this.display_settings_modal.open_with_current(
                                                    canvas_w,
                                                    canvas_h,
                                                    preview_fps,
                                                );
                                                cx.notify();
                                            }))
                                    )
                                    .child(
                                        TimelinePanel::transport_btn(v1_move_mode.label())
                                            .on_mouse_down(MouseButton::Left, cx.listener(move |_this, _, _, cx| {
                                                global_for_v1_mode.update(cx, |gs, cx| {
                                                    // Toggle default V1 drag mode directly from timeline toolbar.
                                                    gs.cycle_v1_move_mode();
                                                    cx.notify();
                                                });
                                            }))
                                    )
                                    .child(
                                        TimelinePanel::transport_btn(self.timeline_load_mode.label())
                                        .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _, cx| {
                                            this.timeline_load_mode = this.timeline_load_mode.toggled();
                                            cx.notify();
                                        }))
                                    )
                                    .child(
                                        TimelinePanel::transport_btn(preview_quality.label())
                                            .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _, cx| {
                                                this.proxy_modal_open = true;
                                                this.proxy_confirm_quality = None;
                                                cx.notify();
                                            }))
                                    )
                                    .child(
                                        div()
                                            .h(px(28.0))
                                            .px_3()
                                            .rounded_lg()
                                            .border_1()
                                            .border_color(if self.srt_menu_open {
                                                rgb(0x2563eb)
                                            } else {
                                                rgba(0xffffff1f)
                                            })
                                            .bg(if self.srt_menu_open {
                                                rgba(0x2563eb66)
                                            } else {
                                                rgba(0xffffff0d)
                                            })
                                            .text_color(if self.srt_menu_open {
                                                rgb(0xdbeafe)
                                            } else {
                                                rgba(0xffffffd9)
                                            })
                                            .hover(|s| s.bg(white().opacity(0.10)))
                                            .cursor_pointer()
                                            .flex()
                                            .items_center()
                                            .justify_center()
                                            .child("SRT".to_string())
                                            .on_mouse_down(MouseButton::Left, cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                                                if this.srt_menu_open {
                                                    this.srt_menu_open = false;
                                                    this.srt_menu_anchor = None;
                                                } else {
                                                    this.srt_menu_open = true;
                                                    let (x, y) = this.window_to_panel_point(evt.position, win);
                                                    this.srt_menu_anchor = Some((x, y + 30.0));
                                                }
                                                cx.notify();
                                            }))
                                    )

                                    .child(TimelinePanel::transport_btn(if export_in_progress {
                                        "Export…"
                                    } else if media_tools_ready_for_export {
                                        "Export"
                                    } else {
                                        "Export (Needs FFmpeg)"
                                    })
                                        .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _win, cx| {
                                            if export_in_progress {
                                                return;
                                            }
                                            if !media_tools_ready_for_export {
                                                this.global.update(cx, |gs, cx| {
                                                    gs.ui_notice = Some(
                                                        "Export requires FFmpeg and FFprobe. Install tools first."
                                                            .to_string(),
                                                    );
                                                    gs.show_media_dependency_modal();
                                                    cx.notify();
                                                });
                                                return;
                                            }
                                            this.export_modal.open_with_default_name(
                                                TimelinePanel::default_export_dir(),
                                            );
                                            cx.notify();
                                        })))
                                    .child(status_view),
                            )
                    )
                    .child(
                        div()
                            .absolute()
                            .left(px(12.0))
                            .right(px(12.0))
                            .bottom(px(4.0))
                            .h(px(20.0))
                            .rounded_sm()
                            .border_1()
                            .border_color(white().opacity(0.08))
                            .bg(white().opacity(0.03))
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .pl_2()
                                    .pr_1()
                                    .font_family("Mono")
                                    .text_xs()
                                    .text_color(rgb(0x93c5fd))
                                    .flex_shrink_0()
                                    .child(timeline_fps_label),
                            )
                            .child(
                                div()
                                    .pr_1()
                                    .font_family("Mono")
                                    .text_xs()
                                    .text_color(rgb(0x86efac))
                                    .flex_shrink_0()
                                    .child(video_fps_label),
                            )
                            .child(
                                div()
                                    .pr_1()
                                    .font_family("Mono")
                                    .text_xs()
                                    .text_color(rgb(0x67e8f9))
                                    .flex_shrink_0()
                                    .child(present_fps_label),
                            )
                            .child(
                                div()
                                    .pr_1()
                                    .font_family("Mono")
                                    .text_xs()
                                    .text_color(if preview_present_dropped_frames > 0 {
                                        rgb(0xfca5a5)
                                    } else {
                                        rgba(0xffffff73)
                                    })
                                    .flex_shrink_0()
                                    .child(present_drop_label),
                            )
                            .child(
                                div()
                                    .pr_1()
                                    .font_family("Mono")
                                    .text_xs()
                                    .text_color(if timeline_low_load_mode {
                                        rgb(0xfbbf24)
                                    } else {
                                        rgba(0xffffff73)
                                    })
                                    .flex_shrink_0()
                                    .child(if timeline_low_load_mode {
                                        "Timeline Lite"
                                    } else {
                                        "Timeline Full"
                                    }),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .px_1()
                                    .overflow_x_scrollbar()
                                    .child(
                                        if timeline_low_load_mode || audio_gain_controls.is_empty() {
                                            div().w_full().flex().items_center().gap_2()
                                                .child(
                                                    div()
                                                        .flex_1()
                                                        .min_w_0()
                                                        .px_2()
                                                        .truncate()
                                                        .font_family("Mono")
                                                        .text_xs()
                                                        .text_color(white().opacity(0.68))
                                                        .child(track_db_summary),
                                                )
                                                .child(
                                                    div()
                                                        .max_w(px(320.0))
                                                        .truncate()
                                                        .px_2()
                                                        .rounded_sm()
                                                        .border_1()
                                                        .border_color(white().opacity(0.08))
                                                        .bg(white().opacity(0.02))
                                                        .font_family("Mono")
                                                        .text_xs()
                                                        .text_color(white().opacity(0.58))
                                                        .child(source_hint_text.clone()),
                                                )
                                        } else {
                                            div().flex().items_center().gap_2()
                                                .children(audio_gain_controls)
                                                .child(
                                                    div()
                                                        .max_w(px(320.0))
                                                        .truncate()
                                                        .px_2()
                                                        .rounded_sm()
                                                        .border_1()
                                                        .border_color(white().opacity(0.08))
                                                        .bg(white().opacity(0.02))
                                                        .font_family("Mono")
                                                        .text_xs()
                                                        .text_color(white().opacity(0.58))
                                                        .child(source_hint_text.clone()),
                                                )
                                        },
                                    ),
                            ),
                    ),
            )
            // --- Body ---
            .child(
                div()
                    .flex()
                    .w_full()
                    .h(px(TIMELINE_BODY_H)) // Fixed height.
                    // 1. Tool Palette
                    .child(
                        div()
                            .w(px(LEFT_TOOL_W))
                            .h_full()
                            .min_h_0()
                            .border_r_1()
                            .border_color(white().opacity(0.08))
                            .bg(black().opacity(0.20))
                            .py_2()
                            .flex()
                            .flex_col()
                            .items_center()
                            .gap_2()
                            .child(TimelinePanel::tool_btn("✎", active_tool == ActiveTool::Select).on_mouse_down(MouseButton::Left, cx.listener(move |_this, _, _, cx| { global_for_tool_select.update(cx, |gs, cx| { gs.set_tool(ActiveTool::Select); cx.notify(); }); })))
                            .child(TimelinePanel::tool_btn("✂", active_tool == ActiveTool::Razor).on_mouse_down(MouseButton::Left, cx.listener(move |_this, _, _, cx| { global_for_tool_razor.update(cx, |gs, cx| { gs.set_tool(ActiveTool::Razor); cx.notify(); }); })))
                            .child(TimelinePanel::tool_btn("TS", active_tool == ActiveTool::TrackSweep).on_mouse_down(MouseButton::Left, cx.listener(move |_this, _, _, cx| {
                                global_for_tool_sweep.update(cx, |gs, cx| {
                                    gs.set_tool(ActiveTool::TrackSweep);
                                    gs.ui_notice = Some("Track Sweep active. Click timeline: select all tracks on the right. Option/Alt=left side.".to_string());
                                    cx.notify();
                                });
                            })))
                            .child(
                                div()
                                    .w_full()
                                    .flex_1()
                                    .min_h_0()
                                    .overflow_y_scrollbar()
                                    .child(
                                        div()
                                            .w_full()
                                            .pb_2()
                                            .flex()
                                            .flex_col()
                                            .items_center()
                                            .gap_2()
                                            // Place quick track-add actions in a scrollable column.
                                            .child(TimelinePanel::tool_btn("+V", false).on_mouse_down(MouseButton::Left, cx.listener({
                                                let value = global_add_track.clone();
                                                move |_this, _, _, cx| {
                                                    value.update(cx, |gs, cx| {
                                                        gs.add_new_video_track();
                                                        cx.notify();
                                                    });
                                                }
                                            })))
                                            .child(TimelinePanel::tool_btn("+A", false).on_mouse_down(MouseButton::Left, cx.listener({
                                                let value = global_add_track.clone();
                                                move |_this, _, _, cx| {
                                                    value.update(cx, |gs, cx| {
                                                        gs.add_new_audio_track();
                                                        cx.notify();
                                                    });
                                                }
                                            })))
                                            .child(TimelinePanel::tool_btn("+S", false).on_mouse_down(MouseButton::Left, cx.listener({
                                                let value = global_add_track.clone();
                                                move |_this, _, _, cx| {
                                                    value.update(cx, |gs, cx| {
                                                        gs.add_new_subtitle_track();
                                                        cx.notify();
                                                    });
                                                }
                                            })))
                                            .child(TimelinePanel::tool_btn("+T", false).on_mouse_down(MouseButton::Left, cx.listener({
                                                let value = global_add_track.clone();
                                                move |this, _, _, cx| {
                                                    this.pending_subtitle_drop = true;
                                                    value.update(cx, |gs, cx| {
                                                        if gs.subtitle_tracks.is_empty() {
                                                            gs.add_new_subtitle_track();
                                                        }
                                                        cx.notify();
                                                    });
                                                    cx.notify();
                                                }
                                            })))
                                            .child(TimelinePanel::tool_btn("+L", false).on_mouse_down(MouseButton::Left, cx.listener({
                                                let value = global_add_track.clone();
                                                move |_this, _, _, cx| {
                                                    value.update(cx, |gs, cx| {
                                                        gs.add_layer_effect_clip_on_top_video_track();
                                                        cx.notify();
                                                    });
                                                }
                                            })))
                                            .child(TimelinePanel::tool_btn("+SL", false).on_mouse_down(MouseButton::Left, cx.listener({
                                                let value = global_add_track.clone();
                                                move |_this, _, _, cx| {
                                                    value.update(cx, |gs, cx| {
                                                        let _ = gs.add_semantic_clip_at_playhead();
                                                        cx.notify();
                                                    });
                                                    cx.notify();
                                                }
                                            })))
                                    )
                            ),
                    )
                    // 2. Track Headers
                    .child(
                        div()
                            .w(px(TRACK_LIST_W))
                            .h_full()
                            .border_r_1()
                            .border_color(white().opacity(0.08))
                            .bg(black().opacity(0.16))
                            .child(div().h(px(RULER_H)).border_b_1().border_color(white().opacity(0.08)).bg(black().opacity(0.25)).px_2().flex().items_center().text_xs().text_color(white().opacity(0.45)).child("Tracks"))
                            .child(
                                div()
                                    .flex_1()
                                    .overflow_hidden()
                                    .child(
                                        div()
                                            .w_full()
                                            .mt(px(-current_scroll_y))
                                            // Subtitle tracks on top (S2, S1...)
                                            .children(subtitle_header_rows)
                                            .children(semantic_header_row)
                                            // Video overlays next (V3, V2...)
                                            .children(video_header_rows)
                                            .child(TimelinePanel::track_header_row(
                                                &TrackUi {
                                                    name: "V1".to_string(),
                                                    kind: TrackHeaderKind::V1,
                                                },
                                                cx,
                                                &self.global,
                                            ))
                                            .children(audio_header_rows)
                                    )
                            )
                    )
                    // 3. Scrollable Timeline Lanes
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .relative()
                            .bg(black().opacity(0.12))
                            .overflow_hidden()
                            .on_mouse_down(MouseButton::Left, cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                                if active_tool == ActiveTool::Select && evt.modifiers.shift {
                                    cx.stop_propagation();
                                    this.start_marquee(evt, win, cx);
                                }
                            }))
                            .on_scroll_wheel(cx.listener(move |this, evt: &ScrollWheelEvent, _win, cx| {
                                let is_zoom = evt.modifiers.platform || evt.modifiers.control;
                                if is_zoom {
                                    let delta_y_px = evt.delta.pixel_delta(px(10.0)).y;
                                    let delta_y_f32 = delta_y_px / px(1.0);
                                    if delta_y_f32 != 0.0 {
                                        let zoom_factor = if delta_y_f32 > 0.0 { 1.10 } else { 0.90 };
                                        let old_px_per_sec = this.px_per_sec;
                                        // let new_px_per_sec = (old_px_per_sec * zoom_factor).clamp(1.0, 500.0);
                                        let new_px_per_sec = (old_px_per_sec * zoom_factor).clamp(0.01, 1000.0);
                                        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                                        let mouse_x_screen_px = (evt.position.x - px(offset_w)).max(px(0.0));
                                        let mouse_x_screen_f32 = mouse_x_screen_px / px(1.0);
                                        let mouse_time_sec = (mouse_x_screen_f32 + this.scroll_offset_x) / old_px_per_sec;
                                        let new_mouse_x_total = mouse_time_sec * new_px_per_sec;
                                        let new_scroll_x = (new_mouse_x_total - mouse_x_screen_f32).max(0.0);
                                        let max_scroll_x = dur_to_px(timeline_total, new_px_per_sec).max(1000.0);
                                        this.px_per_sec = new_px_per_sec;
                                        this.scroll_offset_x = new_scroll_x.min(max_scroll_x);
                                        cx.notify();
                                    }
                                } else {
                                    let delta_x = evt.delta.pixel_delta(px(20.0)).x / px(1.0);
                                    let delta_y = evt.delta.pixel_delta(px(20.0)).y / px(1.0);
                                    let is_horizontal = delta_x.abs() > delta_y.abs();
                                    if is_horizontal {
                                        this.scroll_offset_x -= delta_x;
                                        this.scroll_offset_x = this.scroll_offset_x.clamp(0.0, ruler_w);
                                    } else {
                                        this.scroll_offset_y -= delta_y;
                                        let max_scroll_y = (total_tracks_height - view_height).max(0.0);
                                        this.scroll_offset_y = this.scroll_offset_y.clamp(0.0, max_scroll_y);
                                    }
                                    cx.notify();
                                }
                            }))
                            .child(
                                div()
                                    .size_full()
                                    .relative()
                                    .child(
                                        div()
                                            .min_w_full()
                                            .w(px(ruler_w))
                                            .h_full()
                                            .absolute()
                                            .left(px(-current_scroll_x))

                                            // Ruler
                                            // .child(TimelinePanel::build_ruler(timeline_total, current_px_per_sec))
                                            // ====
                                            // Ruler
                                           .child(
                                                TimelinePanel::build_ruler(
                                                    timeline_total,
                                                    current_px_per_sec,
                                                    visible_start_sec,
                                                    visible_end_sec,
                                                )
                                                    .on_mouse_down(MouseButton::Left, cx.listener(move |this, evt, win, cx| {
                                                        this.start_scrubbing(evt, win, cx, &global_ruler);
                                                    }))
                                            )
                                            // ====

                                            // Tracks Container
                                            .child(
                                                div()
                                                    .absolute()
                                                    .top(px(RULER_H - current_scroll_y))
                                                    .left(px(0.0))
                                                    .w_full()
                                                    // ===========================================
                                                    // Subtitle Tracks
                                                    .children(subtitle_lane_divs)
                                                    .children(if self.show_semantic_lane {
                                                        let global_semantic_bg = self.global.clone();
                                                        vec![
                                                            div()
                                                                .w_full()
                                                                .on_mouse_down(
                                                                    MouseButton::Left,
                                                                    cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                                                                        let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                                                                        let raw_x = evt.position.x;
                                                                        let window_x_px = raw_x - px(offset_w);
                                                                        let window_x_f32 = window_x_px / px(1.0);
                                                                        let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);
                                                                        let t = Duration::from_secs_f32((local_x_f32 / this.px_per_sec).max(0.0));

                                                                        if let Some(forward) =
                                                                            Self::track_sweep_direction(active_tool, evt.modifiers.alt)
                                                                        {
                                                                            this.sweep_select_from_anchor(
                                                                                TrackType::V1,
                                                                                t,
                                                                                forward,
                                                                                true,
                                                                                cx,
                                                                            );
                                                                            this.arm_track_sweep_pending_drag(
                                                                                evt.position.x,
                                                                                t,
                                                                                forward,
                                                                                cx,
                                                                            );
                                                                            cx.focus_self(win);
                                                                            return;
                                                                        }

                                                                        if active_tool == ActiveTool::Select && evt.modifiers.shift {
                                                                            this.start_marquee(evt, win, cx);
                                                                            return;
                                                                        }

                                                                        this.start_scrubbing(evt, win, cx, &global_semantic_bg);
                                                                    }),
                                                                )
                                                                .child(if timeline_low_load_mode {
                                                                    Self::render_semantic_lane_low_load(
                                                                        &semantic_clips,
                                                                        current_px_per_sec,
                                                                        lane_window_start_sec,
                                                                        lane_window_end_sec,
                                                                    )
                                                                } else {
                                                                    Self::render_semantic_lane(
                                                                        &semantic_clips,
                                                                        selected_semantic_clip_id,
                                                                        current_px_per_sec,
                                                                        lane_window_start_sec,
                                                                        lane_window_end_sec,
                                                                        cx,
                                                                        active_tool,
                                                                        self.global.clone(),
                                                                    )
                                                                }),
                                                        ]
                                                    } else {
                                                        Vec::new()
                                                    })

                                                    // This is the content area for V2, V3, and higher video tracks.
                                                    // ===========================================
                                                    .children(
                                                        video_tracks_data.iter().enumerate().rev().map(|(idx, track)| {
                                                            let global_drop = self.global.clone();
                                                            let global_lane = self.global.clone();
                                                            let global_bg_click = self.global.clone();
                                                            let lane_is_hovered = self.media_pool_hover_track == Some(TrackType::VideoOverlay(idx));

                                                            div()
                                                                .w_full()
                                                                .on_drop(cx.listener(move |_this, paths: &ExternalPaths, _, cx| {
                                                                    if let Some(path) = paths.paths().first() {
                                                                        let path_str = path.to_string_lossy().to_string();
                                                                        if !is_supported_media_path(&path_str) {
                                                                            return;
                                                                        }
                                                                        let duration = get_media_duration(&path_str);
                                                                        if duration == Duration::ZERO {
                                                                            return;
                                                                        }
                                                                        global_drop.update(cx, |gs, cx| {
                                                                            gs.load_source_video(path.to_path_buf(), duration);
                                                                            cx.emit(MediaPoolUiEvent::StateChanged);
                                                                            gs.ripple_insert_active_source_video(idx, duration);
                                                                            cx.notify();
                                                                        });
                                                                    }
                                                                }))
                                                                .on_mouse_down(MouseButton::Left, cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                                                                    if active_tool == ActiveTool::Select && evt.modifiers.shift {
                                                                        this.start_marquee(evt, win, cx);
                                                                        return;
                                                                    }

                                                                    let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                                                                    let raw_x = evt.position.x;
                                                                    let window_x_px = raw_x - px(offset_w);
                                                                    let window_x_f32 = window_x_px / px(1.0);
                                                                    let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);
                                                                    let t = Duration::from_secs_f32((local_x_f32 / this.px_per_sec).max(0.0));

                                                                    // Support media pool placement on overlay tracks.
                                                                    let mut inserted_from_pool = false;
                                                                    global_bg_click.update(cx, |gs, cx| {
                                                                        gs.is_playing = false;
                                                                        if let Some(path) = gs.pending_media_pool_path.clone()
                                                                            && gs.activate_media_pool_item(&path)
                                                                        {
                                                                            let duration = gs.active_source_duration;
                                                                            gs.set_playhead(t);
                                                                            gs.ripple_insert_active_source_video(idx, duration);
                                                                            gs.clear_media_pool_drag();
                                                                            cx.emit(MediaPoolUiEvent::StateChanged);
                                                                            inserted_from_pool = true;
                                                                            cx.notify();
                                                                            return;
                                                                        }

                                                                        gs.set_playhead(t);
                                                                        if gs.active_tool == ActiveTool::Razor {
                                                                            let _ = gs.razor_video_at_playhead(idx);
                                                                        }
                                                                        cx.notify();
                                                                    });

                                                                    if inserted_from_pool {
                                                                        this.drag_state = DragState::None;
                                                                        this.is_scrubbing = false;
                                                                        cx.notify();
                                                                        return;
                                                                    }

                                                                    if let Some(forward) = Self::track_sweep_direction(active_tool, evt.modifiers.alt) {
                                                                        this.sweep_select_from_anchor(
                                                                            TrackType::VideoOverlay(idx),
                                                                            t,
                                                                            forward,
                                                                            true,
                                                                            cx,
                                                                        );
                                                                        this.arm_track_sweep_pending_drag(
                                                                            evt.position.x,
                                                                            t,
                                                                            forward,
                                                                            cx,
                                                                        );
                                                                        cx.focus_self(win);
                                                                        return;
                                                                    }

                                                                    this.start_scrubbing(evt, win, cx, &global_bg_click);
                                                                }))
                                                                .child({
                                                                    if timeline_low_load_mode {
                                                                        let mut lane = Self::render_lane_low_load(
                                                                            &track.clips,
                                                                            None,
                                                                            None,
                                                                            current_px_per_sec,
                                                                            lane_window_start_sec,
                                                                            lane_window_end_sec,
                                                                            false,
                                                                            lane_is_hovered,
                                                                        );
                                                                        for layer_clip in layer_effect_clips
                                                                            .iter().filter(|&layer| layer.track_index == idx).cloned()
                                                                        {
                                                                            let layer_selected =
                                                                                selected_layer_effect_clip_id
                                                                                    == Some(layer_clip.id);
                                                                            lane = lane.child(Self::layer_effect_clip_ui_low_load(
                                                                                layer_clip,
                                                                                layer_selected,
                                                                                current_px_per_sec,
                                                                                lane_window_start_sec,
                                                                                lane_window_end_sec,
                                                                            ));
                                                                        }
                                                                        lane
                                                                    } else {
                                                                        let mut lane = Self::render_lane(
                                                                            &track.clips,
                                                                            None,
                                                                            None,
                                                                            &selected_clip_ids,
                                                                            current_px_per_sec,
                                                                            lane_window_start_sec,
                                                                            lane_window_end_sec,
                                                                            false,
                                                                            lane_is_hovered,
                                                                            TrackType::VideoOverlay(idx), // Important: pass VideoOverlay here.
                                                                            cx, active_tool, global_lane
                                                                        );
                                                                        for layer_clip in layer_effect_clips
                                                                            .iter().filter(|&layer| layer.track_index == idx).cloned()
                                                                        {
                                                                            let layer_selected =
                                                                                selected_layer_effect_clip_id
                                                                                    == Some(layer_clip.id);
                                                                            lane = lane.child(Self::layer_effect_clip_ui(
                                                                                layer_clip,
                                                                                layer_selected,
                                                                                current_px_per_sec,
                                                                                lane_window_start_sec,
                                                                                lane_window_end_sec,
                                                                                cx,
                                                                                active_tool,
                                                                                self.global.clone(),
                                                                            ));
                                                                        }
                                                                        lane
                                                                    }
                                                                })
                                                        })
                                                    )
                                                    // ===========================================


                                                    // V1 Track
                                                    .child(
                                                        div()
                                                            .w_full()
                                                            .on_drop(cx.listener(move |_this, paths: &ExternalPaths, _, cx| {
                                                                if let Some(path) = paths.paths().first() {
                                                                    let path_str = path.to_string_lossy().to_string();
                                                                    if !is_supported_media_path(&path_str) {
                                                                        return;
                                                                    }
                                                                    let duration = get_media_duration(&path_str);
                                                                    if duration == Duration::ZERO {
                                                                        return;
                                                                    }
                                                                        global_drop_v1.update(cx, |gs, cx| {
                                                                            gs.load_source_video(path.to_path_buf(), duration);
                                                                            cx.emit(MediaPoolUiEvent::StateChanged);
                                                                            let track_end = gs.v1_clips.iter().map(|c| c.start + c.duration).max().unwrap_or(Duration::ZERO);
                                                                            gs.set_playhead(track_end);
                                                                            // Route V1 insertion through the active move mode (Magnetic/Free).
                                                                            gs.insert_active_source_v1(duration);
                                                                            cx.notify();
                                                                        });
                                                                    }
                                                            }))
                                                            .on_mouse_down(MouseButton::Left, cx.listener(move |this, evt: &MouseDownEvent, win, cx| {
                                                                if active_tool == ActiveTool::Select && evt.modifiers.shift {
                                                                    this.start_marquee(evt, win, cx);
                                                                } else {
                                                                    let offset_w = APP_NAV_W + LEFT_TOOL_W + TRACK_LIST_W;
                                                                    let raw_x = evt.position.x;
                                                                    let window_x_px = raw_x - px(offset_w);
                                                                    let window_x_f32 = window_x_px / px(1.0);
                                                                    let local_x_f32 = (window_x_f32 + this.scroll_offset_x).max(0.0);
                                                                    let t = Duration::from_secs_f32((local_x_f32 / this.px_per_sec).max(0.0));

                                                                    // Allow placing media pool items directly onto V1.
                                                                    let mut inserted_from_pool = false;
                                                                    global_v1_bg.update(cx, |gs, cx| {
                                                                        gs.is_playing = false;
                                                                        if let Some(path) = gs.pending_media_pool_path.clone()
                                                                            && gs.activate_media_pool_item(&path)
                                                                        {
                                                                            let duration = gs.active_source_duration;
                                                                            gs.set_playhead(t);
                                                                            // Route V1 insertion through the active move mode (Magnetic/Free).
                                                                            gs.insert_active_source_v1(duration);
                                                                            gs.clear_media_pool_drag();
                                                                            cx.emit(MediaPoolUiEvent::StateChanged);
                                                                            inserted_from_pool = true;
                                                                            cx.notify();
                                                                        }
                                                                    });
                                                                    if inserted_from_pool {
                                                                        this.drag_state = DragState::None;
                                                                        this.is_scrubbing = false;
                                                                        cx.notify();
                                                                        return;
                                                                    }

                                                                    if let Some(forward) = Self::track_sweep_direction(active_tool, evt.modifiers.alt) {
                                                                        this.sweep_select_from_anchor(
                                                                            TrackType::V1,
                                                                            t,
                                                                            forward,
                                                                            true,
                                                                            cx,
                                                                        );
                                                                        this.arm_track_sweep_pending_drag(
                                                                            evt.position.x,
                                                                            t,
                                                                            forward,
                                                                            cx,
                                                                        );
                                                                        cx.focus_self(win);
                                                                        return;
                                                                    }

                                                                    this.start_scrubbing(evt, win, cx, &global_v1_bg);
                                                                }
                                                            }))
                                                            .child(if timeline_low_load_mode {
                                                                Self::render_lane_low_load(
                                                                    &v1_clips,
                                                                    None,
                                                                    None,
                                                                    current_px_per_sec,
                                                                    lane_window_start_sec,
                                                                    lane_window_end_sec,
                                                                    false,
                                                                    self.media_pool_hover_track
                                                                        == Some(TrackType::V1),
                                                                )
                                                            } else {
                                                                Self::render_lane(
                                                                    &v1_clips,
                                                                    None,
                                                                    None,
                                                                    &selected_clip_ids,
                                                                    current_px_per_sec,
                                                                    lane_window_start_sec,
                                                                    lane_window_end_sec,
                                                                    false,
                                                                    self.media_pool_hover_track
                                                                        == Some(TrackType::V1),
                                                                    TrackType::V1,
                                                                    cx,
                                                                    active_tool,
                                                                    global_for_lane,
                                                                )
                                                            })
                                                    )

                                                    // Audio Tracks
                                                    .children(audio_lane_divs)
                                            )

                                            // Playhead Overlay
                                            .child(div().absolute().top(px(RULER_H - 1.0)).left(px(0.0)).h(px(2.0)).w_full().bg(rgb(0xd04a4a)))
                                            .child(div().absolute().top_0().bottom_0().left(px(playhead_x)).w(px(2.0)).bg(rgb(0x3b82f6))),
                                    )
                            ),
                    )
            .child(div().w(px(RIGHT_STRIP_W)).bg(black().opacity(0.20))),
            )
            .child(marquee_overlay)
            .child(proxy_modal_overlay)
            .child(srt_menu_overlay)
            .child(clip_link_menu_overlay)
            .child(timeline_clip_menu_overlay)
            .child(layer_clip_menu_overlay)
            .child(subtitle_clip_menu_overlay)
            .child(semantic_clip_menu_overlay)
    }
}
