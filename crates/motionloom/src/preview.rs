// =========================================
// =========================================
// crates/motionloom/src/preview.rs

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    GraphParseError, GraphScript, MotionLoomSceneRenderError, SceneGpuTexture, SceneNode,
    ScenePreviewBackend, ScenePreviewSurface, ScenePreviewSurfaceOptions, SceneRenderProfile,
    SceneRenderer, parse_graph_script,
};

fn collect_sequence_activation_frames(
    nodes: &[SceneNode],
    fps: f32,
    total_frames: u32,
    base_ms: u64,
    sampled_frames: &mut Vec<u32>,
) {
    for node in nodes {
        let (children, child_base_ms) = match node {
            SceneNode::Sequence(sequence) => {
                let activation_ms = base_ms.saturating_add(sequence.from_ms);
                sampled_frames.push(
                    ((activation_ms as f32 / 1000.0) * fps)
                        .ceil()
                        .clamp(0.0, total_frames.saturating_sub(1) as f32)
                        as u32,
                );
                (&sequence.children, activation_ms)
            }
            SceneNode::Timeline(value) => (&value.children, base_ms),
            SceneNode::Track(value) => (&value.children, base_ms),
            SceneNode::Chain(value) => (&value.children, base_ms.saturating_add(value.from_ms)),
            SceneNode::Group(value) => (&value.children, base_ms),
            SceneNode::Puppet(value) => (&value.children, base_ms),
            SceneNode::Part(value) => (&value.children, base_ms),
            SceneNode::Repeat(value) => (&value.children, base_ms),
            SceneNode::Mask(value) => (&value.children, base_ms),
            SceneNode::Precompose(value) => (&value.children, base_ms),
            SceneNode::Layer(value) => (&value.children, base_ms),
            SceneNode::Camera(value) => (&value.children, base_ms),
            SceneNode::Character(value) => (&value.children, base_ms),
            _ => continue,
        };
        collect_sequence_activation_frames(
            children,
            fps,
            total_frames,
            child_base_ms,
            sampled_frames,
        );
    }
}

/// Quality presets shared by native live-preview hosts.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WgpuPreviewQuality {
    Full,
    Balanced,
    Speed,
    HighSpeed,
    UltraSpeed,
}

impl WgpuPreviewQuality {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::Balanced => "Balanced 50%",
            Self::Speed => "Speed 25%",
            Self::HighSpeed => "High Speed 10%",
            Self::UltraSpeed => "Ultra Speed 5%",
        }
    }

    pub const fn scale(self) -> f32 {
        match self {
            Self::Full => 1.0,
            Self::Balanced => 0.5,
            Self::Speed => 0.25,
            Self::HighSpeed => 0.10,
            Self::UltraSpeed => 0.05,
        }
    }

    pub const fn index(self) -> u32 {
        match self {
            Self::Full => 0,
            Self::Balanced => 1,
            Self::Speed => 2,
            Self::HighSpeed => 3,
            Self::UltraSpeed => 4,
        }
    }

    /// Return the next lower-cost preview preset without changing export quality.
    pub const fn faster(self) -> Self {
        match self {
            Self::Full => Self::Balanced,
            Self::Balanced => Self::Speed,
            Self::Speed => Self::HighSpeed,
            Self::HighSpeed | Self::UltraSpeed => Self::UltraSpeed,
        }
    }

    /// Return the next higher-quality preview preset.
    pub const fn sharper(self) -> Self {
        match self {
            Self::Full | Self::Balanced => Self::Full,
            Self::Speed => Self::Balanced,
            Self::HighSpeed => Self::Speed,
            Self::UltraSpeed => Self::HighSpeed,
        }
    }
}

/// Result of warming a graph before interactive playback begins.
///
/// Preloading submits representative frames through the same GPU path used by
/// playback. This compiles pipelines and fills GLB, animation, material, mesh,
/// text, and environment caches while the host can still show a loading state.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct WgpuPreviewPreloadReport {
    pub sampled_frames: Vec<u32>,
    pub asset_declarations: usize,
}

/// Incremental preload state owned by an interactive host.
#[derive(Clone, Debug)]
pub struct WgpuPreviewPreloadSession {
    sampled_frames: Vec<u32>,
    next_frame: usize,
    asset_declarations: usize,
}

/// Progress emitted after one representative frame has been warmed.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct WgpuPreviewPreloadProgress {
    pub completed_frames: usize,
    pub total_frames: usize,
    pub warmed_frame: Option<u32>,
    pub finished: bool,
}

impl WgpuPreviewPreloadSession {
    pub fn is_finished(&self) -> bool {
        self.next_frame >= self.sampled_frames.len()
    }

    pub fn report(&self) -> WgpuPreviewPreloadReport {
        WgpuPreviewPreloadReport {
            sampled_frames: self.sampled_frames.clone(),
            asset_declarations: self.asset_declarations,
        }
    }
}

/// Opt-in frame-budget controller for interactive preview hosts.
///
/// It only recommends an existing preview quality; it never mutates a graph or
/// affects export. A manual quality selection can lock the controller until the
/// host explicitly re-enables it.
#[derive(Clone, Debug)]
pub struct WgpuPreviewAdaptiveController {
    enabled: bool,
    manual_lock: bool,
    target_frame_ms: f32,
    slow_streak: u32,
    fast_streak: u32,
    cooldown_frames: u32,
}

impl WgpuPreviewAdaptiveController {
    pub fn for_fps(fps: f32) -> Self {
        Self {
            enabled: true,
            manual_lock: false,
            target_frame_ms: 1000.0 / fps.clamp(1.0, 60.0),
            slow_streak: 0,
            fast_streak: 0,
            cooldown_frames: 0,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.reset_samples();
    }

    pub fn enabled(&self) -> bool {
        self.enabled && !self.manual_lock
    }

    pub fn set_target_fps(&mut self, fps: f32) {
        self.target_frame_ms = 1000.0 / fps.clamp(1.0, 60.0);
        self.reset_samples();
    }

    pub fn lock_manual(&mut self) {
        self.manual_lock = true;
        self.reset_samples();
    }

    pub fn resume_auto(&mut self) {
        self.manual_lock = false;
        self.reset_samples();
    }

    /// Observe one complete renderer frame and optionally recommend a new preset.
    pub fn observe(
        &mut self,
        quality: WgpuPreviewQuality,
        frame_ms: f32,
    ) -> Option<WgpuPreviewQuality> {
        if !self.enabled() || !frame_ms.is_finite() || frame_ms <= 0.0 {
            return None;
        }
        if self.cooldown_frames > 0 {
            self.cooldown_frames -= 1;
            return None;
        }

        if frame_ms > self.target_frame_ms * 1.15 {
            self.slow_streak += 1;
            self.fast_streak = 0;
        } else if frame_ms < self.target_frame_ms * 0.65 {
            self.fast_streak += 1;
            self.slow_streak = 0;
        } else {
            self.slow_streak = 0;
            self.fast_streak = 0;
        }

        if self.slow_streak >= 8 {
            self.reset_samples();
            let next = quality.faster();
            if next != quality {
                self.cooldown_frames = 30;
                return Some(next);
            }
        }
        if self.fast_streak >= 90 {
            self.reset_samples();
            let next = quality.sharper();
            if next != quality {
                self.cooldown_frames = 90;
                return Some(next);
            }
        }
        None
    }

    fn reset_samples(&mut self) {
        self.slow_streak = 0;
        self.fast_streak = 0;
        self.cooldown_frames = 0;
    }
}

/// Render result returned by the reusable preview engine.
#[derive(Debug)]
pub struct WgpuPreviewFrame {
    pub surface: ScenePreviewSurface,
    pub warning: Option<String>,
}

/// Last parsed graph plus a render-size variant for interactive preview hosts.
#[derive(Default)]
pub struct WgpuPreviewGraphCache {
    script_hash: Option<u64>,
    base_graph: Option<GraphScript>,
    render_graph: Option<(Option<(u32, u32)>, GraphScript)>,
}

/// Errors from the host-independent preview engine.
#[derive(Debug, Error)]
pub enum WgpuPreviewEngineError {
    #[error("preview parse failed at line {line}: {message}")]
    Parse { line: usize, message: String },
    #[error("preview render failed: no GPU or CPU renderer initialized")]
    NoRenderer,
    #[error(
        "preview render failed: GPU preview failed ({gpu_error}); CPU fallback failed: {source}"
    )]
    CpuFallbackFailed {
        gpu_error: String,
        #[source]
        source: MotionLoomSceneRenderError,
    },
    #[error("preview render failed: {0}")]
    Render(#[from] MotionLoomSceneRenderError),
}

impl From<GraphParseError> for WgpuPreviewEngineError {
    fn from(err: GraphParseError) -> Self {
        Self::Parse {
            line: err.line,
            message: err.message,
        }
    }
}

impl WgpuPreviewGraphCache {
    pub fn clear(&mut self) {
        self.script_hash = None;
        self.base_graph = None;
        self.render_graph = None;
    }

    /// Return a cached graph for the script and render-size override.
    pub fn graph_for_script(
        &mut self,
        script: &str,
        script_hash: u64,
        render_size: Option<(u32, u32)>,
    ) -> Result<&GraphScript, GraphParseError> {
        if self.script_hash != Some(script_hash) {
            let graph = parse_graph_script(script)?;
            self.script_hash = Some(script_hash);
            self.base_graph = Some(graph);
            self.render_graph = None;
        }

        let Some(base_graph) = self.base_graph.as_ref() else {
            unreachable!("graph cache must contain a parsed graph after successful parse");
        };
        let final_size = base_graph.render_size.unwrap_or(base_graph.size);
        if render_size.is_none() || render_size == Some(final_size) {
            return Ok(base_graph);
        }

        if self
            .render_graph
            .as_ref()
            .is_some_and(|(cached_size, _)| *cached_size == render_size)
        {
            return Ok(&self
                .render_graph
                .as_ref()
                .expect("render graph checked above")
                .1);
        }

        let mut graph = base_graph.clone();
        if let Some(render_size) = render_size {
            graph.render_size = Some((render_size.0.max(1), render_size.1.max(1)));
        }
        self.render_graph = Some((render_size, graph));
        Ok(&self
            .render_graph
            .as_ref()
            .expect("render graph inserted above")
            .1)
    }
}

/// Shared native preview renderer lifecycle used by CLI examples and host apps.
pub struct WgpuPreviewEngine {
    gpu_renderer: Option<SceneRenderer>,
    cpu_renderer: Option<SceneRenderer>,
}

impl WgpuPreviewEngine {
    /// Build a preview engine that tries GPU first and keeps CPU as fallback.
    pub async fn new_with_cpu_fallback() -> Self {
        let gpu_renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.ok();
        let cpu_renderer = SceneRenderer::new(SceneRenderProfile::Cpu).await.ok();
        Self {
            gpu_renderer,
            cpu_renderer,
        }
    }

    /// Build a CPU-only preview engine for panic recovery and headless paths.
    pub async fn new_cpu_only() -> Self {
        Self {
            gpu_renderer: None,
            cpu_renderer: SceneRenderer::new(SceneRenderProfile::Cpu).await.ok(),
        }
    }

    /// Build a GPU preview engine around a host-owned wgpu device and queue.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn new_with_device(
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
    ) -> Result<Self, MotionLoomSceneRenderError> {
        let gpu_renderer =
            SceneRenderer::new_with_device(device, queue, SceneRenderProfile::Gpu).await?;
        Ok(Self {
            gpu_renderer: Some(gpu_renderer),
            cpu_renderer: None,
        })
    }

    pub fn has_gpu_renderer(&self) -> bool {
        self.gpu_renderer.is_some()
    }

    pub fn has_cpu_renderer(&self) -> bool {
        self.cpu_renderer.is_some()
    }

    pub fn drop_gpu_renderer(&mut self) {
        self.gpu_renderer = None;
    }

    /// Warm representative action boundaries before interactive playback.
    ///
    /// This is additive and optional. Existing hosts can keep rendering frames
    /// exactly as before; latency-sensitive hosts can call it while presenting
    /// their loading UI. No CPU readback is performed.
    pub async fn preload_graph_resources(
        &mut self,
        graph: &GraphScript,
    ) -> Result<WgpuPreviewPreloadReport, WgpuPreviewEngineError> {
        let mut session = Self::begin_preload_graph_resources(graph);
        while !session.is_finished() {
            self.preload_graph_resources_step(graph, &mut session)
                .await?;
        }
        Ok(session.report())
    }

    /// Build a bounded representative-frame list without touching GPU state.
    pub fn begin_preload_graph_resources(graph: &GraphScript) -> WgpuPreviewPreloadSession {
        let fps = graph.fps.max(1.0);
        let total_frames = (((graph.duration_ms as f32 / 1000.0) * fps).ceil() as u32).max(1);
        let mut sampled_frames = vec![0];
        sampled_frames.extend(graph.apply_actions.iter().map(|apply| {
            ((apply.at_ms as f32 / 1000.0) * fps)
                .round()
                .clamp(0.0, total_frames.saturating_sub(1) as f32) as u32
        }));
        for scene in &graph.scenes {
            collect_sequence_activation_frames(
                &scene.children,
                fps,
                total_frames,
                0,
                &mut sampled_frames,
            );
        }
        sampled_frames.sort_unstable();
        sampled_frames.dedup();
        // Keep loading bounded for scripts containing hundreds of tiny cues.
        if sampled_frames.len() > 16 {
            sampled_frames.truncate(16);
        }
        WgpuPreviewPreloadSession {
            sampled_frames,
            next_frame: 0,
            asset_declarations: graph.assets.len(),
        }
    }

    /// Warm at most one representative frame so native hosts can return to
    /// their event loop and paint progress between expensive cold resources.
    pub async fn preload_graph_resources_step(
        &mut self,
        graph: &GraphScript,
        session: &mut WgpuPreviewPreloadSession,
    ) -> Result<WgpuPreviewPreloadProgress, WgpuPreviewEngineError> {
        let Some(renderer) = self.gpu_renderer.as_mut() else {
            return Err(WgpuPreviewEngineError::NoRenderer);
        };
        let warmed_frame = session.sampled_frames.get(session.next_frame).copied();
        if let Some(frame) = warmed_frame {
            renderer
                .render_frame_to_wgpu_texture(graph, frame)
                .await
                .map_err(WgpuPreviewEngineError::Render)?;
            session.next_frame += 1;
        }
        Ok(WgpuPreviewPreloadProgress {
            completed_frames: session.next_frame,
            total_frames: session.sampled_frames.len(),
            warmed_frame,
            finished: session.is_finished(),
        })
    }

    /// Render to the fastest displayable preview surface, falling back to CPU BGRA.
    pub async fn render_preview_surface_with_cpu_fallback(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        options: ScenePreviewSurfaceOptions,
    ) -> Result<WgpuPreviewFrame, WgpuPreviewEngineError> {
        let gpu_error = if let Some(renderer) = self.gpu_renderer.as_mut() {
            match renderer
                .render_frame_to_preview_surface(graph, frame, options)
                .await
            {
                Ok(surface) => {
                    return Ok(WgpuPreviewFrame {
                        surface,
                        warning: None,
                    });
                }
                Err(err) => Some(err.to_string()),
            }
        } else {
            None
        };

        let Some(renderer) = self.cpu_renderer.as_mut() else {
            return Err(WgpuPreviewEngineError::NoRenderer);
        };
        let surface = renderer
            .render_frame_to_preview_surface(
                graph,
                frame,
                ScenePreviewSurfaceOptions {
                    backend: ScenePreviewBackend::CpuBgra,
                    ..options
                },
            )
            .await
            .map_err(|source| match gpu_error.clone() {
                Some(gpu_error) => WgpuPreviewEngineError::CpuFallbackFailed { gpu_error, source },
                None => WgpuPreviewEngineError::Render(source),
            })?;
        Ok(WgpuPreviewFrame {
            surface,
            warning: gpu_error.map(|err| format!("Scene live preview used CPU fallback: {err}")),
        })
    }

    /// Parse/cache a script and render it to a displayable preview surface.
    pub async fn render_script_preview_surface_with_cpu_fallback(
        &mut self,
        graph_cache: &mut WgpuPreviewGraphCache,
        script: &str,
        script_hash: u64,
        frame: u32,
        render_size: Option<(u32, u32)>,
        options: ScenePreviewSurfaceOptions,
    ) -> Result<WgpuPreviewFrame, WgpuPreviewEngineError> {
        let graph = graph_cache.graph_for_script(script, script_hash, render_size)?;
        self.render_preview_surface_with_cpu_fallback(graph, frame, options)
            .await
    }

    /// Render into a host-owned texture; windowing and presentation stay outside the engine.
    pub async fn render_frame_to_wgpu_target_texture(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        target: &wgpu::Texture,
        target_width: u32,
        target_height: u32,
    ) -> Result<(), WgpuPreviewEngineError> {
        let Some(renderer) = self.gpu_renderer.as_mut() else {
            return Err(WgpuPreviewEngineError::NoRenderer);
        };
        renderer
            .render_frame_to_wgpu_target_texture(
                graph,
                frame,
                target,
                target_width.max(1),
                target_height.max(1),
            )
            .await
            .map_err(WgpuPreviewEngineError::Render)
    }

    /// Render directly to the compositor-owned GPU texture.
    ///
    /// Native presenters should prefer this over `render_frame_to_wgpu_target_texture`:
    /// sampling the returned texture into the swapchain avoids the extra
    /// compositor-to-host-target texture copy.
    pub async fn render_frame_to_wgpu_texture(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<SceneGpuTexture, WgpuPreviewEngineError> {
        let Some(renderer) = self.gpu_renderer.as_mut() else {
            return Err(WgpuPreviewEngineError::NoRenderer);
        };
        renderer
            .render_frame_to_wgpu_texture(graph, frame)
            .await
            .map_err(WgpuPreviewEngineError::Render)
    }

    /// Render the compositor frame intended for a native swapchain presenter.
    ///
    /// The host samples this texture directly in its surface render pass. The
    /// shared texture ownership also lets the host cache its texture view and
    /// bind group while the compositor keeps reusing the same backing texture.
    pub async fn render_frame_for_native_present(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<SceneGpuTexture, WgpuPreviewEngineError> {
        self.render_frame_to_wgpu_texture(graph, frame).await
    }

    /// Last completed compositor GPU duration, excluding CPU traversal and present.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn last_gpu_frame_ms(&self) -> Option<f64> {
        self.gpu_renderer
            .as_ref()
            .and_then(SceneRenderer::last_gpu_frame_ms)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn last_cpu_frame_profile(&self) -> Option<crate::SceneCpuFrameProfile> {
        self.gpu_renderer
            .as_ref()
            .map(SceneRenderer::last_cpu_frame_profile)
    }

    /// Breakdown of the most recently rendered retained 3D island.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn last_3d_frame_profile(&self) -> Option<crate::Scene3DFrameProfile> {
        self.gpu_renderer
            .as_ref()
            .map(SceneRenderer::last_3d_frame_profile)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_timestamp_supported(&self) -> bool {
        self.gpu_renderer
            .as_ref()
            .is_some_and(SceneRenderer::gpu_timestamp_supported)
    }

    /// Pick a scene node id from the renderer's hidden GPU ID pass.
    pub async fn pick_id_at_wgpu_position(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        x: u32,
        y: u32,
        pick_ids: &[(String, u32)],
    ) -> Result<Option<u32>, WgpuPreviewEngineError> {
        let Some(renderer) = self.gpu_renderer.as_mut() else {
            return Err(WgpuPreviewEngineError::NoRenderer);
        };
        renderer
            .pick_id_at_wgpu_position(graph, frame, x, y, pick_ids)
            .await
            .map_err(WgpuPreviewEngineError::Render)
    }

    pub fn preview_size_for_quality(
        graph: &GraphScript,
        quality: WgpuPreviewQuality,
    ) -> (u32, u32) {
        let (final_width, final_height) = graph.render_size.unwrap_or(graph.size);
        let scale = quality.scale();
        if scale >= 0.999 {
            return (final_width.max(1), final_height.max(1));
        }
        (
            (final_width.max(1) as f32 * scale).round().max(1.0) as u32,
            (final_height.max(1) as f32 * scale).round().max(1.0) as u32,
        )
    }

    /// Create a quality-scaled graph while preserving the original DSL graph.
    pub fn graph_for_quality(base_graph: &GraphScript, quality: WgpuPreviewQuality) -> GraphScript {
        let mut graph = base_graph.clone();
        let final_size = graph.render_size.unwrap_or(graph.size);
        let preview_size = Self::preview_size_for_quality(&graph, quality);
        if preview_size != final_size {
            graph.render_size = Some(preview_size);
            for texture in &mut graph.textures {
                if texture.size == Some(final_size) {
                    texture.size = Some(preview_size);
                }
            }
            Self::scale_resolution_dependent_pass_params(&mut graph, quality.scale());
        }
        graph
    }

    /// Allocate the target texture that CLI windows and embedded hosts can blit.
    pub fn create_target_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motionloom-preview-target-texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn scale_resolution_dependent_pass_params(graph: &mut GraphScript, scale: f32) {
        for pass in &mut graph.passes {
            if pass.effect != "magnify_lens" {
                continue;
            }
            for param in &mut pass.params {
                if matches!(
                    param.key.as_str(),
                    "x" | "y" | "radius" | "feather" | "width"
                ) {
                    param.value = Self::scale_numeric_or_curve_param(&param.value, scale);
                }
            }
        }
    }

    fn scale_numeric_or_curve_param(value: &str, scale: f32) -> String {
        let trimmed = value.trim();
        let unquoted = trimmed
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(trimmed);
        let normalized = unquoted.replace("\\\"", "\"");

        if let Ok(number) = normalized.parse::<f32>() {
            return Self::format_scaled_number(number, scale);
        }

        if let Some(inner) = normalized
            .strip_prefix("curve(\"")
            .and_then(|v| v.strip_suffix("\")"))
        {
            let scaled_points = inner
                .split(',')
                .map(|point| Self::scale_curve_point_value(point.trim(), scale))
                .collect::<Vec<_>>()
                .join(", ");
            return format!("curve(\"{scaled_points}\")");
        }

        normalized
    }

    fn scale_curve_point_value(point: &str, scale: f32) -> String {
        let mut parts = point.splitn(3, ':');
        let Some(time) = parts.next() else {
            return point.to_string();
        };
        let Some(value) = parts.next() else {
            return point.to_string();
        };
        let Some(ease) = parts.next() else {
            return point.to_string();
        };
        let Ok(number) = value.trim().parse::<f32>() else {
            return point.to_string();
        };
        format!(
            "{}:{}:{}",
            time.trim(),
            Self::format_scaled_number(number, scale),
            ease.trim()
        )
    }

    fn format_scaled_number(value: f32, scale: f32) -> String {
        let scaled = value * scale;
        if (scaled.round() - scaled).abs() < 0.001 {
            return format!("{}", scaled.round() as i32);
        }
        let mut text = format!("{scaled:.3}");
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::{
        WgpuPreviewAdaptiveController, WgpuPreviewEngine, WgpuPreviewGraphCache,
        WgpuPreviewQuality, collect_sequence_activation_frames,
    };

    #[test]
    fn preview_preload_includes_delayed_scene_sequence_activation() {
        let graph = crate::parse_graph_script(
            r##"<Graph fps={30} duration="10s" size={[64,64]}>
  <Scene id="main">
    <Timeline>
      <Track space="3d">
        <Sequence from="5.58s" duration="1s">
          <Layer>
            <Rect x="0" y="0" width="8" height="8" color="#ffffff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="main" />
</Graph>"##,
        )
        .expect("delayed 3D scene");
        let mut sampled_frames = vec![0];
        collect_sequence_activation_frames(
            &graph.scenes[0].children,
            graph.fps,
            300,
            0,
            &mut sampled_frames,
        );
        assert!(sampled_frames.contains(&168));
        let session = WgpuPreviewEngine::begin_preload_graph_resources(&graph);
        assert_eq!(session.report().sampled_frames, vec![0, 168]);
    }

    #[test]
    fn adaptive_preview_downgrades_only_after_sustained_slow_frames() {
        let mut adaptive = WgpuPreviewAdaptiveController::for_fps(30.0);
        for _ in 0..7 {
            assert_eq!(adaptive.observe(WgpuPreviewQuality::Full, 50.0), None);
        }
        assert_eq!(
            adaptive.observe(WgpuPreviewQuality::Full, 50.0),
            Some(WgpuPreviewQuality::Balanced)
        );
    }

    #[test]
    fn manual_preview_quality_disables_adaptation() {
        let mut adaptive = WgpuPreviewAdaptiveController::for_fps(30.0);
        adaptive.lock_manual();
        for _ in 0..20 {
            assert_eq!(adaptive.observe(WgpuPreviewQuality::Full, 80.0), None);
        }
    }

    #[test]
    fn scales_quoted_numeric_lens_param() {
        assert_eq!(
            WgpuPreviewEngine::scale_numeric_or_curve_param("\"520\"", 0.25),
            "130"
        );
    }

    #[test]
    fn scales_quoted_escaped_curve_lens_param() {
        assert_eq!(
            WgpuPreviewEngine::scale_numeric_or_curve_param(
                "\"curve(\\\"0:300:ease_out, 3:650:ease_in_out, 6:560:linear\\\")\"",
                0.25,
            ),
            "curve(\"0:75:ease_out, 3:162.5:ease_in_out, 6:140:linear\")"
        );
    }

    #[test]
    fn graph_cache_reuses_script_and_applies_render_size() {
        let script = r##"
<Graph fps={30} duration="1s" size={[640,360]}>
  <Scene id="main">
    <Timeline>
      <Track id="main">
        <Sequence from="0s" duration="1s">
          <Layer>
            <Rect x="0" y="0" width="640" height="360" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="main" />
</Graph>
"##;
        let mut cache = WgpuPreviewGraphCache::default();
        let graph = cache
            .graph_for_script(script, 7, Some((320, 180)))
            .expect("graph parse");
        assert_eq!(graph.size, (640, 360));
        assert_eq!(graph.render_size, Some((320, 180)));

        let graph = cache
            .graph_for_script(script, 7, None)
            .expect("graph parse");
        assert_eq!(graph.render_size, None);
    }
}
