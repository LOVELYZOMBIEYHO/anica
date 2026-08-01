// =========================================
// =========================================
// crates/motionloom/src/scene/backend/gpu/compositor.rs

use std::borrow::Cow;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use image::RgbaImage;

use crate::common::gpu_async::{
    BufferMapAsyncFuture, DevicePoller, request_adapter_async, request_device_async,
};
use crate::dsl::GraphScript;
use crate::scene::backend::gpu::shaders::{
    WGPU_BATCH_SHAPE_SHADER, WGPU_BLOOM_SHADER, WGPU_DOWNSAMPLE_SHADER, WGPU_LIGHT_SWEEP_SHADER,
    WGPU_MATTE_TEXTURE_SHADER, WGPU_POST_SHADER, WGPU_PUPPET_DEFORM_SHADER, WGPU_SCENE_SHADER,
    WGPU_SPECTRAL_ENERGY_SHADER,
};
use crate::scene::composition::{SceneMagnifyLensParams, SceneTextureOverlayParams};
use crate::scene::drawable::{
    GpuSceneMatteMode, GpuSceneNativeTexture, GpuScenePrimitive, GpuSceneTextureLayer,
    GpuSceneTextureSource, PostLightSweepUniformParams, PostMagnifyLensUniformParams,
    PostTextureOverlayUniformParams, SceneBlendMode, batch_shape_storage_bytes,
    batch_shape_uniform, matte_texture_uniform, post_blur_uniform, post_color_uniform,
    post_edge_treatment_uniform, post_hsla_overlay_uniform, post_light_sweep_uniform,
    post_magnify_lens_uniform, post_material_displacement_uniform, post_opacity_uniform,
    post_texture_overlay_uniform, post_tint_uniform, post_tone_map_uniform, texture_layer_bounds,
    texture_layer_projected_bounds,
};
use crate::scene::render::{MotionLoomSceneRenderError, eval_scene_number};
use crate::scene::resource::{load_rgba_image_source, load_svg_source};
use crate::scene::spatial::{Affine2, EvaluatedDeformGrid, TextureRect, resolve_axis};

struct WgpuImageTexture {
    pub(crate) width: u32,
    pub(crate) height: u32,
    texture: std::sync::Arc<wgpu::Texture>,
}

struct PersistentGpuBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
    last_data: Vec<u8>,
}

#[derive(Default)]
pub(crate) struct WgpuDispatchKeepalive {
    textures: Vec<Arc<wgpu::Texture>>,
    texture_views: Vec<wgpu::TextureView>,
    bind_groups: Vec<wgpu::BindGroup>,
    buffers: Vec<wgpu::Buffer>,
}

impl WgpuDispatchKeepalive {
    #[cfg(not(target_arch = "wasm32"))]
    fn append(&mut self, mut other: Self) {
        self.textures.append(&mut other.textures);
        self.texture_views.append(&mut other.texture_views);
        self.bind_groups.append(&mut other.bind_groups);
        self.buffers.append(&mut other.buffers);
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeInFlightSubmission {
    index: wgpu::SubmissionIndex,
    _keepalive: WgpuDispatchKeepalive,
    timestamp: Option<NativeGpuTimestampFrame>,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeGpuTimestampFrame {
    _query_set: wgpu::QuerySet,
    _resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    map_result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>,
    timestamp_period_ns: f32,
}

pub(crate) struct WgpuSceneCompositor {
    #[cfg(target_arch = "wasm32")]
    instance: Option<wgpu::Instance>,
    #[cfg(target_arch = "wasm32")]
    adapter: Option<wgpu::Adapter>,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    _poller: DevicePoller,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    matte_texture_bind_group_layout: wgpu::BindGroupLayout,
    matte_texture_pipeline: wgpu::ComputePipeline,
    puppet_deform_bind_group_layout: wgpu::BindGroupLayout,
    puppet_deform_pipeline: wgpu::ComputePipeline,
    shape_bind_group_layout: wgpu::BindGroupLayout,
    shape_pipeline: wgpu::ComputePipeline,
    post_bind_group_layout: wgpu::BindGroupLayout,
    post_pipeline: wgpu::ComputePipeline,
    light_sweep_pipeline: wgpu::ComputePipeline,
    spectral_energy_bind_group_layout: wgpu::BindGroupLayout,
    spectral_energy_pipeline: wgpu::ComputePipeline,
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    downsample_pipeline: wgpu::ComputePipeline,
    bloom_bind_group_layout: wgpu::BindGroupLayout,
    bloom_pipeline: wgpu::ComputePipeline,
    sampler: wgpu::Sampler,
    pub(crate) width: u32,
    pub(crate) height: u32,
    tex_a: wgpu::Texture,
    tex_b: wgpu::Texture,
    dummy_post_texture: Arc<wgpu::Texture>,
    readback_buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
    image_textures: HashMap<String, WgpuImageTexture>,
    shape_uniform_buffer: Option<PersistentGpuBuffer>,
    shape_primitive_buffer: Option<PersistentGpuBuffer>,
    shape_transform_buffer: Option<PersistentGpuBuffer>,
    shape_tile_range_buffer: Option<PersistentGpuBuffer>,
    shape_tile_index_buffer: Option<PersistentGpuBuffer>,
    asset_resolver: Arc<dyn crate::asset::AssetResolver>,
    #[cfg(not(target_arch = "wasm32"))]
    frame_recording: bool,
    #[cfg(not(target_arch = "wasm32"))]
    pending_command_encoder: Option<wgpu::CommandEncoder>,
    #[cfg(not(target_arch = "wasm32"))]
    pending_keepalive: WgpuDispatchKeepalive,
    #[cfg(not(target_arch = "wasm32"))]
    pending_timestamp: Option<NativeGpuTimestampFrame>,
    #[cfg(not(target_arch = "wasm32"))]
    frame_shape_dispatches: usize,
    #[cfg(not(target_arch = "wasm32"))]
    in_flight_submissions: VecDeque<NativeInFlightSubmission>,
    #[cfg(not(target_arch = "wasm32"))]
    last_gpu_frame_ms: Option<f64>,
    #[cfg(not(target_arch = "wasm32"))]
    current_cpu_upload: Duration,
    #[cfg(not(target_arch = "wasm32"))]
    current_cpu_encode: Duration,
    #[cfg(not(target_arch = "wasm32"))]
    current_cpu_wait: Duration,
    #[cfg(not(target_arch = "wasm32"))]
    last_cpu_stages: (Duration, Duration, Duration),
    #[cfg(not(target_arch = "wasm32"))]
    timestamp_marker_pipeline: Option<wgpu::ComputePipeline>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct ShapeBenchmarkRender {
    pub(crate) texture: GpuSceneNativeTexture,
    pub(crate) encode: Duration,
    pub(crate) gpu: Duration,
    pub(crate) primitive_count: u32,
    pub(crate) upload_bytes: usize,
}

/// Platform-specific surface handles needed only for WASM canvas presentation.
/// Empty on native targets where zero-copy interop uses external surfaces.
#[derive(Clone)]
struct WgpuPresentationContext {
    #[cfg(target_arch = "wasm32")]
    instance: Option<wgpu::Instance>,
    #[cfg(target_arch = "wasm32")]
    adapter: Option<wgpu::Adapter>,
}

impl WgpuPresentationContext {
    #[cfg(target_arch = "wasm32")]
    fn new(instance: Option<wgpu::Instance>, adapter: Option<wgpu::Adapter>) -> Self {
        Self { instance, adapter }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn new() -> Self {
        Self {}
    }
}

impl WgpuSceneCompositor {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn device_queue(&self) -> (Arc<wgpu::Device>, wgpu::Queue) {
        (self.device.clone(), self.queue.clone())
    }

    fn submit_encoder(&mut self, encoder: wgpu::CommandEncoder) {
        self.submit_encoder_with_keepalive(encoder, WgpuDispatchKeepalive::default());
    }

    fn submit_encoder_with_keepalive(
        &mut self,
        encoder: wgpu::CommandEncoder,
        keepalive: WgpuDispatchKeepalive,
    ) {
        #[cfg(target_arch = "wasm32")]
        {
            self.queue.submit([encoder.finish()]);
            drop(keepalive);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.frame_recording {
                self.pending_command_encoder = Some(encoder);
                self.pending_keepalive.append(keepalive);
            } else {
                self.submit_native_frame(vec![encoder.finish()], keepalive, None, true);
            }
        }
    }

    /// Start collecting all command buffers for one native frame.
    pub(crate) fn begin_native_frame(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.current_cpu_upload = Duration::ZERO;
            self.current_cpu_encode = Duration::ZERO;
            self.current_cpu_wait = Duration::ZERO;
            self.poll_native_submissions();
            self.frame_recording = true;
            self.pending_command_encoder = Some(self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("motionloom-native-frame-encoder"),
                },
            ));
            self.pending_keepalive = WgpuDispatchKeepalive::default();
            self.frame_shape_dispatches = 0;
            self.pending_timestamp = self.begin_gpu_timestamp_frame();
        }
    }

    /// Submit one frame as a single queue operation and retain its GPU resources.
    pub(crate) fn end_native_frame(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.frame_recording = false;
            if self.pending_command_encoder.is_none() {
                return;
            }
            let timestamp = self.end_gpu_timestamp_frame();
            let Some(encoder) = self.pending_command_encoder.take() else {
                return;
            };
            let keepalive = std::mem::take(&mut self.pending_keepalive);
            self.submit_native_frame(vec![encoder.finish()], keepalive, timestamp, false);
            self.last_cpu_stages = (
                self.current_cpu_upload,
                self.current_cpu_encode,
                self.current_cpu_wait,
            );
        }
    }

    fn frame_encoder(&mut self, label: &'static str) -> wgpu::CommandEncoder {
        #[cfg(not(target_arch = "wasm32"))]
        if self.frame_recording {
            return self.pending_command_encoder.take().unwrap_or_else(|| {
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
            });
        }
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_native_submissions(&mut self) {
        let queue_empty = self
            .device
            .poll(wgpu::PollType::Poll)
            .is_ok_and(|status| status.is_queue_empty());
        self.collect_gpu_timestamp_results();
        if queue_empty {
            // Queue completion can be reported before the asynchronous map
            // callback is dispatched. Keep timestamp resources alive until
            // their callback result has been consumed.
            self.in_flight_submissions
                .retain(|submission| submission.timestamp.is_some());
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn submit_native_frame(
        &mut self,
        command_buffers: Vec<wgpu::CommandBuffer>,
        keepalive: WgpuDispatchKeepalive,
        timestamp: Option<NativeGpuTimestampFrame>,
        poll_before_submit: bool,
    ) {
        const MAX_FRAMES_IN_FLIGHT: usize = 3;

        if poll_before_submit {
            self.poll_native_submissions();
        }
        while self.in_flight_submissions.len() >= MAX_FRAMES_IN_FLIGHT {
            let Some(mut oldest) = self.in_flight_submissions.pop_front() else {
                break;
            };
            // Backpressure happens only when three complete frames are queued.
            let wait_started = Instant::now();
            self.device
                .poll(wgpu::PollType::WaitForSubmissionIndex(oldest.index.clone()))
                .ok();
            self.current_cpu_wait += wait_started.elapsed();
            self.collect_gpu_timestamp_result(&mut oldest);
        }
        let index = self.queue.submit(command_buffers);
        if let Some(timestamp) = timestamp.as_ref() {
            let result = timestamp.map_result.clone();
            timestamp
                .readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |value| {
                    *result.lock().expect("GPU timestamp map result lock") = Some(value);
                });
        }
        self.in_flight_submissions
            .push_back(NativeInFlightSubmission {
                index,
                _keepalive: keepalive,
                timestamp,
            });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn begin_gpu_timestamp_frame(&mut self) -> Option<NativeGpuTimestampFrame> {
        let required = wgpu::Features::TIMESTAMP_QUERY;
        if !self.device.features().contains(required) {
            return None;
        }
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("motionloom-frame-timestamp-query"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motionloom-frame-timestamp-resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motionloom-frame-timestamp-readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let marker_pipeline = self.timestamp_marker_pipeline.as_ref()?;
        let encoder = self.pending_command_encoder.as_mut()?;
        {
            let mut marker = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motionloom-frame-timestamp-begin-pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                }),
            });
            marker.set_pipeline(marker_pipeline);
            marker.dispatch_workgroups(1, 1, 1);
        }
        Some(NativeGpuTimestampFrame {
            _query_set: query_set,
            _resolve_buffer: resolve_buffer,
            readback_buffer,
            map_result: Arc::new(Mutex::new(None)),
            timestamp_period_ns: self.queue.get_timestamp_period(),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn end_gpu_timestamp_frame(&mut self) -> Option<NativeGpuTimestampFrame> {
        let timestamp = self.pending_timestamp.take()?;
        let marker_pipeline = self.timestamp_marker_pipeline.as_ref()?;
        let encoder = self.pending_command_encoder.as_mut()?;
        {
            let mut marker = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motionloom-frame-timestamp-end-pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &timestamp._query_set,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(1),
                }),
            });
            marker.set_pipeline(marker_pipeline);
            marker.dispatch_workgroups(1, 1, 1);
        }
        encoder.resolve_query_set(&timestamp._query_set, 0..2, &timestamp._resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &timestamp._resolve_buffer,
            0,
            &timestamp.readback_buffer,
            0,
            16,
        );
        Some(timestamp)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn collect_gpu_timestamp_results(&mut self) {
        let mut completed_ms = None;
        for submission in &mut self.in_flight_submissions {
            if let Some(value) = Self::read_gpu_timestamp_result(submission) {
                completed_ms = Some(value);
            }
        }
        if completed_ms.is_some() {
            self.last_gpu_frame_ms = completed_ms;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn collect_gpu_timestamp_result(&mut self, submission: &mut NativeInFlightSubmission) {
        if let Some(value) = Self::read_gpu_timestamp_result(submission) {
            self.last_gpu_frame_ms = Some(value);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_gpu_timestamp_result(submission: &mut NativeInFlightSubmission) -> Option<f64> {
        let timestamp = submission.timestamp.as_mut()?;
        let result = timestamp
            .map_result
            .lock()
            .expect("GPU timestamp map result lock")
            .take()?;
        if result.is_err() {
            submission.timestamp = None;
            return None;
        }
        let mapped = timestamp.readback_buffer.slice(..).get_mapped_range();
        let start = u64::from_ne_bytes(mapped[0..8].try_into().ok()?);
        let end = u64::from_ne_bytes(mapped[8..16].try_into().ok()?);
        if std::env::var_os("MOTIONLOOM_GPU_TIMESTAMP_DEBUG").is_some() {
            eprintln!(
                "[MotionLoom][GPU timestamp] start={start} end={end} period_ns={}",
                timestamp.timestamp_period_ns
            );
        }
        drop(mapped);
        timestamp.readback_buffer.unmap();
        let elapsed_ms =
            end.saturating_sub(start) as f64 * timestamp.timestamp_period_ns as f64 / 1_000_000.0;
        submission.timestamp = None;
        Some(elapsed_ms)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn last_gpu_frame_ms(&self) -> Option<f64> {
        self.last_gpu_frame_ms
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn last_cpu_stages(&self) -> (Duration, Duration, Duration) {
        self.last_cpu_stages
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn gpu_timestamp_supported(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// Profile the same batched shape pipeline used by normal scene rendering.
    ///
    /// This deliberately accepts shapes only: texture layers introduce several
    /// independent submissions and would make the encode/GPU boundary ambiguous.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn render_shape_benchmark_to_texture(
        &mut self,
        primitives: &[GpuScenePrimitive],
        clear: [u8; 4],
    ) -> Result<ShapeBenchmarkRender, MotionLoomSceneRenderError> {
        let encode_started = Instant::now();
        let canvas_len = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        let mut base = vec![0u8; canvas_len];
        for pixel in base.chunks_exact_mut(4) {
            pixel.copy_from_slice(&clear);
        }

        let tex_a = Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));
        let tex_b = Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));
        self.write_texture_rgba(&tex_a, self.width, self.height, &base)?;
        self.write_texture_rgba(&tex_b, self.width, self.height, &base)?;

        let shape_batch = batch_shape_storage_bytes(primitives, self.width, self.height)?;
        let uniform = batch_shape_uniform(
            self.width,
            self.height,
            false,
            shape_batch.primitive_count,
            shape_batch.tile_size,
            shape_batch.tiles_x,
            shape_batch.tiles_y,
        );
        let shape_upload_bytes = self.update_persistent_shape_buffers(
            &uniform,
            &shape_batch.primitive_bytes,
            &shape_batch.transform_bytes,
            &shape_batch.tile_range_bytes,
            &shape_batch.tile_index_bytes,
        );
        let upload_bytes = canvas_len
            .saturating_mul(2)
            .saturating_add(shape_upload_bytes);
        let uniform_buffer = &self
            .shape_uniform_buffer
            .as_ref()
            .expect("shape uniform buffer initialized")
            .buffer;
        let storage_buffer = &self
            .shape_primitive_buffer
            .as_ref()
            .expect("shape primitive buffer initialized")
            .buffer;
        let transform_buffer = &self
            .shape_transform_buffer
            .as_ref()
            .expect("shape transform buffer initialized")
            .buffer;
        let tile_range_buffer = &self
            .shape_tile_range_buffer
            .as_ref()
            .expect("shape tile range buffer initialized")
            .buffer;
        let tile_index_buffer = &self
            .shape_tile_index_buffer
            .as_ref()
            .expect("shape tile index buffer initialized")
            .buffer;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("motionloom-path-benchmark-encoder"),
            });
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_batched_shape_pass(
            &mut encoder,
            &tex_a,
            &tex_b,
            uniform_buffer,
            storage_buffer,
            transform_buffer,
            tile_range_buffer,
            tile_index_buffer,
            &mut keepalive,
        );
        let command_buffer = encoder.finish();
        let encode = encode_started.elapsed();

        let gpu_started = Instant::now();
        let submission = self.queue.submit([command_buffer]);
        self.device
            .poll(wgpu::PollType::WaitForSubmissionIndex(submission))
            .map_err(|error| MotionLoomSceneRenderError::GpuRender {
                message: format!("path benchmark GPU wait failed: {error}"),
            })?;
        let gpu = gpu_started.elapsed();

        drop(keepalive);
        Ok(ShapeBenchmarkRender {
            texture: GpuSceneNativeTexture {
                texture: tex_b.clone(),
                width: self.width,
                height: self.height,
                _keepalive_textures: vec![tex_a, tex_b],
            },
            encode,
            gpu,
            primitive_count: shape_batch.primitive_count,
            upload_bytes,
        })
    }

    fn debug_gpu_matte_enabled() -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::env::var_os("ANICA_DEBUG_GPU_MATTE")
                .map(|value| value != "0")
                .unwrap_or(false)
        }

        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    fn debug_preflight_matte_texture_view(
        &self,
        role: &'static str,
        view: &wgpu::TextureView,
        filterable: bool,
    ) {
        if !Self::debug_gpu_matte_enabled() {
            return;
        }
        eprintln!(
            "[MotionLoom][GPU Matte] preflight {role} begin canvas={}x{} filterable={filterable}",
            self.width, self.height
        );
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-gpu-matte-debug-single-texture-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let _bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-gpu-matte-debug-single-texture-bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        });
        eprintln!("[MotionLoom][GPU Matte] preflight {role} ok");
    }

    pub(crate) async fn new(
        width: u32,
        height: u32,
        asset_resolver: Arc<dyn crate::asset::AssetResolver>,
    ) -> Result<Self, MotionLoomSceneRenderError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = request_scene_gpu_adapter_async(&instance).await?;
        let adapter_limits = adapter.limits();
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
        let max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;
        if width > max_texture_dimension_2d || height > max_texture_dimension_2d {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "requested scene render size {}x{} exceeds GPU max 2D texture dimension {}",
                    width, height, max_texture_dimension_2d
                ),
            });
        }

        let (device, queue) = request_device_async(
            &adapter,
            &wgpu::DeviceDescriptor {
                label: Some("anica-motionloom-scene-gpu-device"),
                required_features,
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            },
        )
        .await
        .map_err(|err| MotionLoomSceneRenderError::GpuRender {
            message: format!("device request failed: {err}"),
        })?;
        let device = Arc::new(device);

        Self::new_with_device_internal(
            device,
            queue,
            #[cfg(target_arch = "wasm32")]
            WgpuPresentationContext::new(Some(instance), Some(adapter)),
            #[cfg(not(target_arch = "wasm32"))]
            WgpuPresentationContext::new(),
            width,
            height,
            asset_resolver,
        )
        .await
    }

    /// Create a compositor from an externally-owned wgpu device and queue.
    ///
    /// This lets downstream applications (e.g. anica) share a GPU context with
    /// motionloom so that rendered textures can be consumed without CPU readback.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) async fn new_with_device(
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
        asset_resolver: Arc<dyn crate::asset::AssetResolver>,
    ) -> Result<Self, MotionLoomSceneRenderError> {
        let limits = device.limits();
        let max_texture_dimension_2d = limits.max_texture_dimension_2d;
        if width > max_texture_dimension_2d || height > max_texture_dimension_2d {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "requested scene render size {}x{} exceeds GPU max 2D texture dimension {}",
                    width, height, max_texture_dimension_2d
                ),
            });
        }

        Self::new_with_device_internal(
            device,
            queue,
            WgpuPresentationContext::new(),
            width,
            height,
            asset_resolver,
        )
        .await
    }

    async fn new_with_device_internal(
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
        #[allow(unused_variables)] presentation: WgpuPresentationContext,
        width: u32,
        height: u32,
        asset_resolver: Arc<dyn crate::asset::AssetResolver>,
    ) -> Result<Self, MotionLoomSceneRenderError> {
        let poller = DevicePoller::start(device.clone());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-scene-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGPU_SCENE_SHADER)),
        });
        let matte_texture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-scene-matte-texture-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGPU_MATTE_TEXTURE_SHADER)),
        });
        let puppet_deform_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-scene-puppet-deform-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGPU_PUPPET_DEFORM_SHADER)),
        });
        let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-scene-shape-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGPU_BATCH_SHAPE_SHADER)),
        });
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-scene-post-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGPU_POST_SHADER)),
        });
        let light_sweep_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-scene-light-sweep-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGPU_LIGHT_SWEEP_SHADER)),
        });
        let spectral_energy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-scene-spectral-energy-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGPU_SPECTRAL_ENERGY_SHADER)),
        });
        let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-scene-downsample-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGPU_DOWNSAMPLE_SHADER)),
        });
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anica-motionloom-scene-bloom-gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGPU_BLOOM_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("anica-motionloom-scene-gpu-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shape_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-scene-shape-gpu-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let matte_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-scene-matte-texture-gpu-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let puppet_deform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-scene-puppet-deform-gpu-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let post_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-scene-post-gpu-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let downsample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-scene-downsample-gpu-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anica-motionloom-scene-gpu-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("anica-motionloom-scene-gpu-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let matte_texture_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-scene-matte-texture-gpu-pipeline-layout"),
                bind_group_layouts: &[&matte_texture_bind_group_layout],
                push_constant_ranges: &[],
            });
        let matte_texture_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("anica-motionloom-scene-matte-texture-gpu-pipeline"),
                layout: Some(&matte_texture_pipeline_layout),
                module: &matte_texture_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let puppet_deform_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-scene-puppet-deform-gpu-pipeline-layout"),
                bind_group_layouts: &[&puppet_deform_bind_group_layout],
                push_constant_ranges: &[],
            });
        let puppet_deform_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("anica-motionloom-scene-puppet-deform-gpu-pipeline"),
                layout: Some(&puppet_deform_pipeline_layout),
                module: &puppet_deform_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let shape_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-scene-shape-gpu-pipeline-layout"),
                bind_group_layouts: &[&shape_bind_group_layout],
                push_constant_ranges: &[],
            });
        let shape_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("anica-motionloom-scene-shape-gpu-pipeline"),
            layout: Some(&shape_pipeline_layout),
            module: &shape_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let post_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anica-motionloom-scene-post-gpu-pipeline-layout"),
            bind_group_layouts: &[&post_bind_group_layout],
            push_constant_ranges: &[],
        });
        let post_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("anica-motionloom-scene-post-gpu-pipeline"),
            layout: Some(&post_pipeline_layout),
            module: &post_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let light_sweep_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("anica-motionloom-scene-light-sweep-gpu-pipeline"),
                layout: Some(&post_pipeline_layout),
                module: &light_sweep_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let spectral_energy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-scene-spectral-energy-gpu-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let spectral_energy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-scene-spectral-energy-gpu-pipeline-layout"),
                bind_group_layouts: &[&spectral_energy_bind_group_layout],
                push_constant_ranges: &[],
            });
        let spectral_energy_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("anica-motionloom-scene-spectral-energy-gpu-pipeline"),
                layout: Some(&spectral_energy_pipeline_layout),
                module: &spectral_energy_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let downsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-scene-downsample-gpu-pipeline-layout"),
                bind_group_layouts: &[&downsample_bind_group_layout],
                push_constant_ranges: &[],
            });
        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("anica-motionloom-scene-downsample-gpu-pipeline"),
                layout: Some(&downsample_pipeline_layout),
                module: &downsample_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bloom_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("anica-motionloom-scene-bloom-gpu-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let bloom_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-scene-bloom-gpu-pipeline-layout"),
                bind_group_layouts: &[&bloom_bind_group_layout],
                push_constant_ranges: &[],
            });
        let bloom_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("anica-motionloom-scene-bloom-gpu-pipeline"),
            layout: Some(&bloom_pipeline_layout),
            module: &bloom_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        #[cfg(not(target_arch = "wasm32"))]
        let timestamp_marker_pipeline = if device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("motionloom-frame-timestamp-marker-shader"),
                source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
            });
            Some(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("motionloom-frame-timestamp-marker-pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }),
            )
        } else {
            None
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("anica-motionloom-scene-gpu-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let dummy_post_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anica-motionloom-scene-post-dummy-texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &dummy_post_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[128, 128, 128, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let tex_a = Self::make_canvas_texture(&device, width, height);
        let tex_b = Self::make_canvas_texture(&device, width, height);
        let padded_bytes_per_row = align_to_256(width.saturating_mul(4));
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anica-motionloom-scene-gpu-readback"),
            size: (padded_bytes_per_row as u64 * height as u64).max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            #[cfg(target_arch = "wasm32")]
            instance: presentation.instance,
            #[cfg(target_arch = "wasm32")]
            adapter: presentation.adapter,
            device,
            queue,
            _poller: poller,
            bind_group_layout,
            pipeline,
            matte_texture_bind_group_layout,
            matte_texture_pipeline,
            puppet_deform_bind_group_layout,
            puppet_deform_pipeline,
            shape_bind_group_layout,
            shape_pipeline,
            post_bind_group_layout,
            post_pipeline,
            light_sweep_pipeline,
            spectral_energy_bind_group_layout,
            spectral_energy_pipeline,
            downsample_bind_group_layout,
            downsample_pipeline,
            bloom_bind_group_layout,
            bloom_pipeline,
            sampler,
            width,
            height,
            tex_a,
            tex_b,
            dummy_post_texture,
            readback_buffer,
            padded_bytes_per_row,
            image_textures: HashMap::new(),
            shape_uniform_buffer: None,
            shape_primitive_buffer: None,
            shape_transform_buffer: None,
            shape_tile_range_buffer: None,
            shape_tile_index_buffer: None,
            asset_resolver,
            #[cfg(not(target_arch = "wasm32"))]
            frame_recording: false,
            #[cfg(not(target_arch = "wasm32"))]
            pending_command_encoder: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_keepalive: WgpuDispatchKeepalive::default(),
            #[cfg(not(target_arch = "wasm32"))]
            pending_timestamp: None,
            #[cfg(not(target_arch = "wasm32"))]
            frame_shape_dispatches: 0,
            #[cfg(not(target_arch = "wasm32"))]
            in_flight_submissions: VecDeque::new(),
            #[cfg(not(target_arch = "wasm32"))]
            last_gpu_frame_ms: None,
            #[cfg(not(target_arch = "wasm32"))]
            current_cpu_upload: Duration::ZERO,
            #[cfg(not(target_arch = "wasm32"))]
            current_cpu_encode: Duration::ZERO,
            #[cfg(not(target_arch = "wasm32"))]
            current_cpu_wait: Duration::ZERO,
            #[cfg(not(target_arch = "wasm32"))]
            last_cpu_stages: (Duration::ZERO, Duration::ZERO, Duration::ZERO),
            #[cfg(not(target_arch = "wasm32"))]
            timestamp_marker_pipeline,
        })
    }

    /// Present a GPU-rendered scene texture directly into a browser canvas surface.
    ///
    /// This path is WASM-only and avoids CPU readback: the compositor samples the
    /// internal RGBA scene texture into the canvas swapchain texture.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn present_texture_to_canvas(
        &self,
        texture: &GpuSceneNativeTexture,
        canvas: &web_sys::HtmlCanvasElement,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let width = texture.width.max(1);
        let height = texture.height.max(1);
        canvas.set_width(width);
        canvas.set_height(height);

        let Some(instance) = self.instance.as_ref() else {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "canvas presentation requires an internally-created wgpu instance; \
                          use the default constructor instead of new_with_device"
                    .to_string(),
            });
        };
        let Some(adapter) = self.adapter.as_ref() else {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "canvas presentation requires an internally-created wgpu adapter; \
                          use the default constructor instead of new_with_device"
                    .to_string(),
            });
        };

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|err| MotionLoomSceneRenderError::GpuRender {
                message: format!("canvas surface creation failed: {err}"),
            })?;
        let caps = surface.get_capabilities(adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|format| {
                matches!(
                    format,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: "canvas surface has no supported texture formats".to_string(),
            })?;
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto)
        };
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            caps.present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::AutoVsync)
        };

        surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width,
                height,
                present_mode,
                desired_maximum_frame_latency: 2,
                alpha_mode,
                view_formats: vec![],
            },
        );
        let frame =
            surface
                .get_current_texture()
                .map_err(|err| MotionLoomSceneRenderError::GpuRender {
                    message: format!("canvas surface frame acquisition failed: {err}"),
                })?;
        let target_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let source_view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("anica-motionloom-scene-canvas-present-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dims = textureDimensions(src_tex);
    let max_px = dims - vec2<u32>(1u, 1u);
    let px = min(vec2<u32>(u32(in.position.x), u32(in.position.y)), max_px);
    let color = textureLoad(src_tex, vec2<i32>(px), 0);
    return vec4<f32>(color.rgb, 1.0);
}
"#,
                )),
            });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("anica-motionloom-scene-canvas-present-bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }],
                });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("anica-motionloom-scene-canvas-present-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("anica-motionloom-scene-canvas-present-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-canvas-present-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&source_view),
            }],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-motionloom-scene-canvas-present-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("anica-motionloom-scene-canvas-present-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        frame.present();
        Ok(())
    }

    /// Present a solid color directly into a browser canvas surface for debugging.
    ///
    /// This does not touch MotionLoom scene textures, so it isolates browser
    /// surface/presentation failures from renderer-output failures.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn debug_present_solid_to_canvas(
        &self,
        canvas: &web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
        color: [f64; 4],
    ) -> Result<(), MotionLoomSceneRenderError> {
        let width = width.max(1);
        let height = height.max(1);
        canvas.set_width(width);
        canvas.set_height(height);

        let surface = self
            .instance
            .as_ref()
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: "debug canvas surface requires an instance".to_string(),
            })?
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|err| MotionLoomSceneRenderError::GpuRender {
                message: format!("debug canvas surface creation failed: {err}"),
            })?;
        let adapter =
            self.adapter
                .as_ref()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "debug canvas surface requires an adapter".to_string(),
                })?;
        let caps = surface.get_capabilities(adapter);
        let format =
            caps.formats
                .first()
                .copied()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "debug canvas surface has no supported texture formats".to_string(),
                })?;
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto)
        };
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            caps.present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::AutoVsync)
        };
        surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width,
                height,
                present_mode,
                desired_maximum_frame_latency: 2,
                alpha_mode,
                view_formats: vec![],
            },
        );
        let frame =
            surface
                .get_current_texture()
                .map_err(|err| MotionLoomSceneRenderError::GpuRender {
                    message: format!("debug canvas surface frame acquisition failed: {err}"),
                })?;
        let target_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-motionloom-scene-debug-solid-encoder"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("anica-motionloom-scene-debug-solid-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: color[0],
                            g: color[1],
                            b: color[2],
                            a: color[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        self.queue.submit([encoder.finish()]);
        frame.present();
        Ok(())
    }

    /// Upload a solid RGBA texture, then present that texture to the canvas.
    ///
    /// This isolates texture upload and texture presentation from scene command
    /// collection and compute-shape passes.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn debug_present_uploaded_texture_to_canvas(
        &self,
        canvas: &web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
        color: [u8; 4],
    ) -> Result<(), MotionLoomSceneRenderError> {
        let width = width.max(1);
        let height = height.max(1);
        let texture = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut rgba = vec![0_u8; width as usize * height as usize * 4];
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.copy_from_slice(&color);
        }
        self.write_texture_rgba(&texture, width, height, &rgba)?;
        let native_texture = GpuSceneNativeTexture {
            texture,
            width,
            height,
            _keepalive_textures: Vec::new(),
        };
        self.present_texture_to_canvas(&native_texture, canvas)
    }

    pub(crate) fn make_canvas_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anica-motionloom-scene-gpu-canvas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    pub(crate) fn make_source_texture(&self, width: u32, height: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anica-motionloom-scene-gpu-source"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Render the scene graph to a freshly-allocated GPU texture.
    ///
    /// The returned texture is owned by the caller and can be presented by an
    /// external renderer without a CPU readback round-trip.
    pub(crate) async fn render_to_texture(
        &mut self,
        graph: &GraphScript,
        solid: [u8; 4],
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let tex_a = Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));
        let tex_b = Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));

        let canvas_len = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        let mut base = vec![0u8; canvas_len];
        for pixel in base.chunks_exact_mut(4) {
            pixel.copy_from_slice(&solid);
        }
        self.write_texture_rgba(&tex_a, self.width, self.height, &base)?;

        let mut current_is_a = true;
        let mut encoder = self.frame_encoder("anica-motionloom-scene-gpu-encoder");
        let mut uniform_buffers = Vec::with_capacity(graph.images.len() + graph.svgs.len());
        let mut keepalive = WgpuDispatchKeepalive::default();

        for image_node in &graph.images {
            let opacity =
                eval_scene_number(&image_node.opacity, time_norm, time_sec)?.clamp(0.0, 1.0);
            if opacity <= 0.0001 {
                continue;
            }

            let (source_w, source_h, source_texture) = self.load_image_texture(&image_node.src)?;
            let scale =
                eval_scene_number(&image_node.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
            let target_w = ((source_w as f32) * scale).round().max(1.0);
            let target_h = ((source_h as f32) * scale).round().max(1.0);
            let x_base = resolve_axis(
                &image_node.x,
                self.width as f32,
                target_w,
                time_norm,
                time_sec,
            )?;
            let y_base = resolve_axis(
                &image_node.y,
                self.height as f32,
                target_h,
                time_norm,
                time_sec,
            )?;

            let mut uniform = [0u8; 48];
            let values = [
                self.width as f32,
                self.height as f32,
                x_base,
                y_base,
                target_w,
                target_h,
                source_w as f32,
                source_h as f32,
                opacity,
                0.0,
                0.0,
                0.0,
            ];
            for (ix, value) in values.iter().enumerate() {
                uniform[ix * 4..ix * 4 + 4].copy_from_slice(&value.to_ne_bytes());
            }
            let uniform_buffer = self.make_uniform_buffer(&uniform);

            let (src_canvas, dst_canvas) = if current_is_a {
                (&tex_a, &tex_b)
            } else {
                (&tex_b, &tex_a)
            };
            self.dispatch_image_pass(
                &mut encoder,
                src_canvas,
                &source_texture,
                dst_canvas,
                &uniform_buffer,
                &mut keepalive,
            );
            uniform_buffers.push(uniform_buffer);
            current_is_a = !current_is_a;
        }

        for svg_node in &graph.svgs {
            let opacity =
                eval_scene_number(&svg_node.opacity, time_norm, time_sec)?.clamp(0.0, 1.0);
            if opacity <= 0.0001 {
                continue;
            }

            let (source_w, source_h, source_texture) = self.load_svg_texture(&svg_node.src)?;
            let scale = eval_scene_number(&svg_node.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
            let target_w = ((source_w as f32) * scale).round().max(1.0);
            let target_h = ((source_h as f32) * scale).round().max(1.0);
            let x_base = resolve_axis(
                &svg_node.x,
                self.width as f32,
                target_w,
                time_norm,
                time_sec,
            )?;
            let y_base = resolve_axis(
                &svg_node.y,
                self.height as f32,
                target_h,
                time_norm,
                time_sec,
            )?;

            let mut uniform = [0u8; 48];
            let values = [
                self.width as f32,
                self.height as f32,
                x_base,
                y_base,
                target_w,
                target_h,
                source_w as f32,
                source_h as f32,
                opacity,
                0.0,
                0.0,
                0.0,
            ];
            for (ix, value) in values.iter().enumerate() {
                uniform[ix * 4..ix * 4 + 4].copy_from_slice(&value.to_ne_bytes());
            }
            let uniform_buffer = self.make_uniform_buffer(&uniform);

            let (src_canvas, dst_canvas) = if current_is_a {
                (&tex_a, &tex_b)
            } else {
                (&tex_b, &tex_a)
            };
            self.dispatch_image_pass(
                &mut encoder,
                src_canvas,
                &source_texture,
                dst_canvas,
                &uniform_buffer,
                &mut keepalive,
            );
            uniform_buffers.push(uniform_buffer);
            current_is_a = !current_is_a;
        }

        let final_texture = if current_is_a {
            tex_a.clone()
        } else {
            tex_b.clone()
        };
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffers);

        Ok(GpuSceneNativeTexture {
            texture: final_texture,
            width: self.width,
            height: self.height,
            _keepalive_textures: Vec::new(),
        })
    }

    pub(crate) async fn render(
        &mut self,
        graph: &GraphScript,
        solid: [u8; 4],
        time_norm: f32,
        time_sec: f32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let final_texture = self
            .render_to_texture(graph, solid, time_norm, time_sec)
            .await?;
        self.readback_texture_rgba_async(&final_texture.texture)
            .await
    }

    pub(crate) fn render_scene_content_to_texture(
        &mut self,
        primitives: &[GpuScenePrimitive],
        texture_layers: &[GpuSceneTextureLayer],
        clear: [u8; 4],
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        self.render_scene_content_to_texture_with_mode(primitives, texture_layers, clear, false)
    }

    pub(crate) fn render_scene_pick_ids_to_texture(
        &mut self,
        primitives: &[GpuScenePrimitive],
        texture_layers: &[GpuSceneTextureLayer],
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        self.render_scene_content_to_texture_with_mode(
            primitives,
            texture_layers,
            [0, 0, 0, 0],
            true,
        )
    }

    fn render_scene_content_to_texture_with_mode(
        &mut self,
        primitives: &[GpuScenePrimitive],
        texture_layers: &[GpuSceneTextureLayer],
        clear: [u8; 4],
        pick_mode: bool,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        // Native shapes are normally batched before textures. Split the shape
        // batch when a rasterized Puppet layer belongs between vector nodes.
        if !pick_mode
            && texture_layers
                .iter()
                .any(|layer| layer.primitive_index < primitives.len())
        {
            return self.render_ordered_scene_content_to_texture(primitives, texture_layers, clear);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let encode_started = Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        let mut local_upload = Duration::ZERO;
        let tex_a = std::sync::Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));
        let tex_b = std::sync::Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));
        if clear == [0, 0, 0, 0] {
            // Offscreen layers start transparent. Clear them on the GPU rather
            // than allocating and uploading two full-size CPU RGBA images.
            let mut encoder = self.frame_encoder("anica-motionloom-scene-transparent-clear");
            let view_a = tex_a.create_view(&wgpu::TextureViewDescriptor::default());
            let view_b = tex_b.create_view(&wgpu::TextureViewDescriptor::default());
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("anica-motionloom-scene-transparent-clear-pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &view_a,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &view_b,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            self.submit_encoder(encoder);
        } else {
            let canvas_len = (self.width as usize)
                .saturating_mul(self.height as usize)
                .saturating_mul(4);
            let mut base = vec![0u8; canvas_len];
            for pixel in base.chunks_exact_mut(4) {
                pixel.copy_from_slice(&clear);
            }
            #[cfg(not(target_arch = "wasm32"))]
            let clear_upload_started = Instant::now();
            self.write_texture_rgba(&tex_a, self.width, self.height, &base)?;
            self.write_texture_rgba(&tex_b, self.width, self.height, &base)?;
            #[cfg(not(target_arch = "wasm32"))]
            {
                local_upload += clear_upload_started.elapsed();
            }
        }

        let mut current_is_a = true;
        let mut dirty_a: Option<TextureRect> = None;
        let mut dirty_b: Option<TextureRect> = None;
        let mut texture_sources =
            Vec::<std::sync::Arc<wgpu::Texture>>::with_capacity(texture_layers.len());
        let mut texture_alias_copies =
            Vec::<std::sync::Arc<wgpu::Texture>>::with_capacity(texture_layers.len());
        let mut gpu_layer_keepalive_textures = Vec::<std::sync::Arc<wgpu::Texture>>::new();
        let mut submitted_buffers = Vec::<wgpu::Buffer>::new();

        let shape_batch = batch_shape_storage_bytes(primitives, self.width, self.height)?;
        if shape_batch.primitive_count > 0 {
            let uniform = batch_shape_uniform(
                self.width,
                self.height,
                pick_mode,
                shape_batch.primitive_count,
                shape_batch.tile_size,
                shape_batch.tiles_x,
                shape_batch.tiles_y,
            );
            #[cfg(not(target_arch = "wasm32"))]
            let use_transient_shape_buffers =
                self.frame_recording && self.frame_shape_dispatches > 0;
            #[cfg(target_arch = "wasm32")]
            let use_transient_shape_buffers = false;
            #[cfg(not(target_arch = "wasm32"))]
            if self.frame_recording {
                self.frame_shape_dispatches = self.frame_shape_dispatches.saturating_add(1);
            }
            #[cfg(not(target_arch = "wasm32"))]
            let shape_upload_started = Instant::now();
            if use_transient_shape_buffers {
                let uniform_buffer = self.make_initialized_buffer(
                    "anica-motionloom-scene-batch-shape-gpu-uniform-transient",
                    &uniform,
                    wgpu::BufferUsages::UNIFORM,
                );
                let storage_buffer = self.make_initialized_buffer(
                    "anica-motionloom-scene-shape-gpu-storage-transient",
                    &shape_batch.primitive_bytes,
                    wgpu::BufferUsages::STORAGE,
                );
                let transform_buffer = self.make_initialized_buffer(
                    "anica-motionloom-scene-shape-gpu-transforms-transient",
                    &shape_batch.transform_bytes,
                    wgpu::BufferUsages::STORAGE,
                );
                let tile_range_buffer = self.make_initialized_buffer(
                    "anica-motionloom-scene-shape-gpu-tile-ranges-transient",
                    &shape_batch.tile_range_bytes,
                    wgpu::BufferUsages::STORAGE,
                );
                let tile_index_buffer = self.make_initialized_buffer(
                    "anica-motionloom-scene-shape-gpu-tile-indices-transient",
                    &shape_batch.tile_index_bytes,
                    wgpu::BufferUsages::STORAGE,
                );
                let mut encoder = self.frame_encoder("anica-motionloom-scene-shape-gpu-encoder");
                let mut keepalive = WgpuDispatchKeepalive::default();
                self.dispatch_batched_shape_pass(
                    &mut encoder,
                    &tex_a,
                    &tex_b,
                    &uniform_buffer,
                    &storage_buffer,
                    &transform_buffer,
                    &tile_range_buffer,
                    &tile_index_buffer,
                    &mut keepalive,
                );
                keepalive.buffers.extend([
                    uniform_buffer,
                    storage_buffer,
                    transform_buffer,
                    tile_range_buffer,
                    tile_index_buffer,
                ]);
                self.submit_encoder_with_keepalive(encoder, keepalive);
            } else {
                self.update_persistent_shape_buffers(
                    &uniform,
                    &shape_batch.primitive_bytes,
                    &shape_batch.transform_bytes,
                    &shape_batch.tile_range_bytes,
                    &shape_batch.tile_index_bytes,
                );
                let mut encoder = self.frame_encoder("anica-motionloom-scene-shape-gpu-encoder");
                let uniform_buffer = &self
                    .shape_uniform_buffer
                    .as_ref()
                    .expect("shape uniform buffer initialized")
                    .buffer;
                let storage_buffer = &self
                    .shape_primitive_buffer
                    .as_ref()
                    .expect("shape primitive buffer initialized")
                    .buffer;
                let transform_buffer = &self
                    .shape_transform_buffer
                    .as_ref()
                    .expect("shape transform buffer initialized")
                    .buffer;
                let tile_range_buffer = &self
                    .shape_tile_range_buffer
                    .as_ref()
                    .expect("shape tile range buffer initialized")
                    .buffer;
                let tile_index_buffer = &self
                    .shape_tile_index_buffer
                    .as_ref()
                    .expect("shape tile index buffer initialized")
                    .buffer;
                let mut keepalive = WgpuDispatchKeepalive::default();
                self.dispatch_batched_shape_pass(
                    &mut encoder,
                    &tex_a,
                    &tex_b,
                    uniform_buffer,
                    storage_buffer,
                    transform_buffer,
                    tile_range_buffer,
                    tile_index_buffer,
                    &mut keepalive,
                );
                self.submit_encoder_with_keepalive(encoder, keepalive);
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                local_upload += shape_upload_started.elapsed();
            }
            current_is_a = false;
            dirty_a = Some(TextureRect {
                x: 0,
                y: 0,
                width: self.width,
                height: self.height,
            });
            dirty_b = None;
        }

        for layer in texture_layers {
            let layer_w = layer.source.width();
            let layer_h = layer.source.height();
            if layer.opacity <= 0.0001 || layer_w == 0 || layer_h == 0 {
                continue;
            }
            let Some((bounds_x, bounds_y, bounds_w, bounds_h)) =
                (if let Some(quad) = layer.projected_quad {
                    texture_layer_projected_bounds(quad, self.width, self.height)
                } else {
                    texture_layer_bounds(layer.transform, layer_w, layer_h, self.width, self.height)
                })
            else {
                continue;
            };
            if bounds_w == 0 || bounds_h == 0 {
                continue;
            }

            let source_texture = match &layer.source {
                GpuSceneTextureSource::Cpu(image) => {
                    let texture = std::sync::Arc::new(
                        self.make_source_texture(image.width().max(1), image.height().max(1)),
                    );
                    #[cfg(not(target_arch = "wasm32"))]
                    let texture_upload_started = Instant::now();
                    self.write_texture_rgba(
                        &texture,
                        image.width().max(1),
                        image.height().max(1),
                        image.as_raw(),
                    )?;
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        local_upload += texture_upload_started.elapsed();
                    }
                    texture_sources.push(texture.clone());
                    texture
                }
                GpuSceneTextureSource::Gpu(texture) => {
                    gpu_layer_keepalive_textures.push(texture.texture.clone());
                    gpu_layer_keepalive_textures
                        .extend(texture._keepalive_textures.iter().cloned());
                    texture.texture.clone()
                }
            };
            let (matte_texture, matte_w, matte_h, matte_mode, invert_matte) =
                if let Some(matte) = layer.matte.as_ref() {
                    gpu_layer_keepalive_textures.push(matte.texture.texture.clone());
                    gpu_layer_keepalive_textures
                        .extend(matte.texture._keepalive_textures.iter().cloned());
                    (
                        matte.texture.texture.clone(),
                        matte.texture.width,
                        matte.texture.height,
                        matte.mode,
                        matte.invert,
                    )
                } else {
                    (
                        source_texture.clone(),
                        layer_w,
                        layer_h,
                        GpuSceneMatteMode::None,
                        false,
                    )
                };
            let uniform = matte_texture_uniform(
                layer,
                self.width,
                self.height,
                bounds_x,
                bounds_y,
                bounds_w,
                bounds_h,
                layer_w,
                layer_h,
                matte_w,
                matte_h,
                matte_mode,
                invert_matte,
                pick_mode,
            )?;
            let uniform_buffer = self.make_matte_texture_uniform_buffer(&uniform);
            let (src_canvas, dst_canvas) = if current_is_a {
                (&tex_a, &tex_b)
            } else {
                (&tex_b, &tex_a)
            };
            let source_texture = if std::sync::Arc::ptr_eq(&source_texture, dst_canvas) {
                let texture = std::sync::Arc::new(Self::make_canvas_texture(
                    &self.device,
                    layer_w.max(1),
                    layer_h.max(1),
                ));
                let mut encoder =
                    self.frame_encoder("anica-motionloom-scene-texture-alias-copy-gpu-encoder");
                self.copy_texture_rect(
                    &mut encoder,
                    &source_texture,
                    &texture,
                    TextureRect {
                        x: 0,
                        y: 0,
                        width: layer_w.max(1),
                        height: layer_h.max(1),
                    },
                );
                self.submit_encoder(encoder);
                texture_alias_copies.push(texture.clone());
                texture
            } else {
                source_texture
            };
            let matte_texture = if std::sync::Arc::ptr_eq(&matte_texture, dst_canvas) {
                let texture = std::sync::Arc::new(Self::make_canvas_texture(
                    &self.device,
                    matte_w.max(1),
                    matte_h.max(1),
                ));
                let mut encoder =
                    self.frame_encoder("anica-motionloom-scene-matte-alias-copy-gpu-encoder");
                self.copy_texture_rect(
                    &mut encoder,
                    &matte_texture,
                    &texture,
                    TextureRect {
                        x: 0,
                        y: 0,
                        width: matte_w.max(1),
                        height: matte_h.max(1),
                    },
                );
                self.submit_encoder(encoder);
                texture_alias_copies.push(texture.clone());
                texture
            } else {
                matte_texture
            };
            let dst_dirty = if current_is_a { dirty_b } else { dirty_a };
            if let Some(rect) = dst_dirty {
                let mut encoder =
                    self.frame_encoder("anica-motionloom-scene-dirty-copy-gpu-encoder");
                self.copy_texture_rect(&mut encoder, src_canvas, dst_canvas, rect);
                self.submit_encoder(encoder);
            }
            let mut encoder =
                self.frame_encoder("anica-motionloom-scene-matte-texture-gpu-encoder");
            let mut keepalive = WgpuDispatchKeepalive::default();
            self.dispatch_matte_texture_pass(
                &mut encoder,
                src_canvas,
                &source_texture,
                &matte_texture,
                dst_canvas,
                &uniform_buffer,
                bounds_w,
                bounds_h,
                &mut keepalive,
            );
            self.submit_encoder_with_keepalive(encoder, keepalive);
            submitted_buffers.push(uniform_buffer);
            let changed = TextureRect {
                x: bounds_x,
                y: bounds_y,
                width: bounds_w,
                height: bounds_h,
            };
            if current_is_a {
                dirty_b = None;
                dirty_a = union_texture_rect(dirty_a, changed);
            } else {
                dirty_a = None;
                dirty_b = union_texture_rect(dirty_b, changed);
            }
            current_is_a = !current_is_a;
        }

        let final_texture = if current_is_a {
            tex_a.clone()
        } else {
            tex_b.clone()
        };
        drop(submitted_buffers);
        let keepalive_textures = texture_sources
            .into_iter()
            .chain(texture_alias_copies)
            .chain(gpu_layer_keepalive_textures)
            .chain([tex_a.clone(), tex_b.clone()])
            .collect();
        let result = GpuSceneNativeTexture {
            texture: final_texture,
            width: self.width,
            height: self.height,
            _keepalive_textures: keepalive_textures,
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.current_cpu_upload += local_upload;
            self.current_cpu_encode += encode_started.elapsed().saturating_sub(local_upload);
        }
        Ok(result)
    }

    fn render_ordered_scene_content_to_texture(
        &mut self,
        primitives: &[GpuScenePrimitive],
        texture_layers: &[GpuSceneTextureLayer],
        clear: [u8; 4],
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let mut ordered_layers = Vec::with_capacity(texture_layers.len() * 2 + 1);
        let mut primitive_cursor = 0usize;

        for layer in texture_layers {
            let boundary = layer
                .primitive_index
                .min(primitives.len())
                .max(primitive_cursor);
            if boundary > primitive_cursor {
                let source = self.render_scene_content_to_texture_with_mode(
                    &primitives[primitive_cursor..boundary],
                    &[],
                    [0, 0, 0, 0],
                    false,
                )?;
                ordered_layers.push(GpuSceneTextureLayer {
                    source: GpuSceneTextureSource::Gpu(source),
                    transform: Affine2::identity(),
                    projected_quad: None,
                    opacity: 1.0,
                    blend: SceneBlendMode::Normal,
                    pick_id: 0,
                    matte: None,
                    primitive_index: usize::MAX,
                });
            }
            let mut ordered_layer = layer.clone();
            ordered_layer.primitive_index = usize::MAX;
            ordered_layers.push(ordered_layer);
            primitive_cursor = boundary;
        }

        if primitive_cursor < primitives.len() {
            let source = self.render_scene_content_to_texture_with_mode(
                &primitives[primitive_cursor..],
                &[],
                [0, 0, 0, 0],
                false,
            )?;
            ordered_layers.push(GpuSceneTextureLayer {
                source: GpuSceneTextureSource::Gpu(source),
                transform: Affine2::identity(),
                projected_quad: None,
                opacity: 1.0,
                blend: SceneBlendMode::Normal,
                pick_id: 0,
                matte: None,
                primitive_index: usize::MAX,
            });
        }

        self.render_scene_content_to_texture_with_mode(&[], &ordered_layers, clear, false)
    }

    pub(crate) fn copy_gpu_native_texture_owned(
        &mut self,
        input: &GpuSceneNativeTexture,
        label: &'static str,
    ) -> GpuSceneNativeTexture {
        let width = input.width.max(1);
        let height = input.height.max(1);
        let texture = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut encoder = self.frame_encoder(label);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &input.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.submit_encoder(encoder);
        let mut keepalive_textures = Vec::with_capacity(input._keepalive_textures.len() + 1);
        keepalive_textures.push(input.texture.clone());
        keepalive_textures.extend(input._keepalive_textures.iter().cloned());
        GpuSceneNativeTexture {
            texture,
            width: input.width,
            height: input.height,
            _keepalive_textures: keepalive_textures,
        }
    }

    pub(crate) async fn apply_gpu_blur_passes(
        &mut self,
        input: &RgbaImage,
        passes: &[(bool, f32)],
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        if passes.is_empty() {
            return Ok(input.clone());
        }
        if input.width() != self.width || input.height() != self.height {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "post-pass input size {}x{} does not match GPU compositor {}x{}",
                    input.width(),
                    input.height(),
                    self.width,
                    self.height
                ),
            });
        }

        self.write_texture_rgba(&self.tex_a, self.width, self.height, input.as_raw())?;

        let mut current_is_a = true;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-motionloom-scene-post-gpu-encoder"),
            });
        let mut uniform_buffers = Vec::with_capacity(passes.len());
        let mut keepalive = WgpuDispatchKeepalive::default();

        for (horizontal, sigma) in passes {
            let uniform = post_blur_uniform(self.width, self.height, *horizontal, *sigma);
            let uniform_buffer = self.make_post_uniform_buffer(&uniform);
            let (src_canvas, dst_canvas) = if current_is_a {
                (&self.tex_a, &self.tex_b)
            } else {
                (&self.tex_b, &self.tex_a)
            };
            self.dispatch_post_pass(
                &mut encoder,
                src_canvas,
                dst_canvas,
                &uniform_buffer,
                &mut keepalive,
            );
            uniform_buffers.push(uniform_buffer);
            current_is_a = !current_is_a;
        }

        let final_texture = if current_is_a {
            &self.tex_a
        } else {
            &self.tex_b
        };
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: final_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let rendered = self.readback_rgba_async().await;
        drop(uniform_buffers);
        drop(keepalive);
        rendered
    }

    pub(crate) async fn apply_gpu_opacity_pass(
        &mut self,
        input: &RgbaImage,
        opacity: f32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        if input.width() != self.width || input.height() != self.height {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "opacity input size {}x{} does not match GPU compositor {}x{}",
                    input.width(),
                    input.height(),
                    self.width,
                    self.height
                ),
            });
        }

        self.write_texture_rgba(&self.tex_a, self.width, self.height, input.as_raw())?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-motionloom-scene-opacity-gpu-encoder"),
            });
        let uniform = post_opacity_uniform(self.width, self.height, opacity);
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass(
            &mut encoder,
            &self.tex_a,
            &self.tex_b,
            &uniform_buffer,
            &mut keepalive,
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.tex_b,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let rendered = self.readback_rgba_async().await;
        drop(uniform_buffer);
        drop(keepalive);
        rendered
    }

    pub(crate) fn apply_gpu_blur_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        passes: &[(bool, f32)],
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        #[cfg(not(target_arch = "wasm32"))]
        let encode_started = Instant::now();
        if passes.is_empty() {
            return Ok(input.clone());
        }
        let width = input.width.max(1);
        let height = input.height.max(1);

        let mut temp_textures = Vec::<std::sync::Arc<wgpu::Texture>>::with_capacity(passes.len());
        let mut current = input.texture.clone();

        for (horizontal, sigma) in passes {
            // Pass boundaries on the retained frame encoder provide the
            // storage-write to sampled-texture usage transition.
            let mut encoder = self.frame_encoder("anica-motionloom-scene-post-texture-gpu-encoder");
            let uniform = post_blur_uniform(width, height, *horizontal, *sigma);
            let uniform_buffer = self.make_post_uniform_buffer(&uniform);
            let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
            let mut keepalive = WgpuDispatchKeepalive::default();
            self.dispatch_post_pass_sized(
                &mut encoder,
                &current,
                &dst,
                &uniform_buffer,
                width,
                height,
                &mut keepalive,
            );
            self.submit_encoder_with_keepalive(encoder, keepalive);
            drop(uniform_buffer);
            current = dst.clone();
            temp_textures.push(dst);
        }

        let result = GpuSceneNativeTexture {
            texture: current,
            width,
            height,
            _keepalive_textures: temp_textures,
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.current_cpu_encode += encode_started.elapsed();
        }
        Ok(result)
    }

    pub(crate) fn apply_gpu_deform_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        grid: &EvaluatedDeformGrid,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let width = input.width.max(1);
        let height = input.height.max(1);
        let triangles = deform_grid_triangle_bytes(grid);
        let triangle_count = triangles.len() / (24 * 4);
        if triangle_count == 0 {
            return Ok(input.clone());
        }

        let uniform =
            puppet_deform_uniform(width, height, triangle_count as u32, grid.preserve_outside);
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let triangle_buffer =
            self.make_storage_buffer("anica-motionloom-scene-puppet-deform-triangles", &triangles);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));

        let mut encoder = self.frame_encoder("anica-motionloom-scene-puppet-deform-gpu-encoder");
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_puppet_deform_pass(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            &triangle_buffer,
            width,
            height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        drop(triangle_buffer);

        let mut keepalive_textures = Vec::with_capacity(input._keepalive_textures.len() + 1);
        keepalive_textures.push(input.texture.clone());
        keepalive_textures.extend(input._keepalive_textures.iter().cloned());
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: keepalive_textures,
        })
    }

    pub(crate) fn apply_gpu_downsample_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        dst_width: u32,
        dst_height: u32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        #[cfg(not(target_arch = "wasm32"))]
        let encode_started = Instant::now();
        let dst_width = dst_width.max(1);
        let dst_height = dst_height.max(1);
        let mut encoder = self.frame_encoder("anica-motionloom-scene-downsample-gpu-encoder");
        let uniform = downsample_uniform(input.width, input.height, dst_width, dst_height);
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(
            &self.device,
            dst_width,
            dst_height,
        ));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_downsample_pass(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            dst_width,
            dst_height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        let result = GpuSceneNativeTexture {
            texture: dst,
            width: dst_width,
            height: dst_height,
            _keepalive_textures: vec![input.texture.clone()],
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.current_cpu_encode += encode_started.elapsed();
        }
        Ok(result)
    }

    pub(crate) fn apply_gpu_bloom_texture_low_res(
        &mut self,
        original: &GpuSceneNativeTexture,
        threshold: f32,
        intensity: f32,
        sigma: f32,
        scale: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let scale = scale.clamp(0.05, 1.0);
        let bloom_width = ((original.width.max(1) as f32) * scale).round().max(1.0) as u32;
        let bloom_height = ((original.height.max(1) as f32) * scale).round().max(1.0) as u32;
        let half = self.apply_gpu_downsample_texture(original, bloom_width, bloom_height)?;
        // A one-level half-resolution pyramid keeps the expensive separable
        // blur off the full-size target. Composite performs the only upscale,
        // avoiding extra intermediate bloom/blur submissions.
        let half_sigma = (sigma * scale).max(1.0);
        let half_blurred =
            self.apply_gpu_blur_texture(&half, &[(true, half_sigma), (false, half_sigma)])?;
        self.apply_gpu_bloom_texture(original, &half_blurred, threshold, intensity)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_gpu_glow_stack_texture(
        &mut self,
        original: &GpuSceneNativeTexture,
        threshold: f32,
        intensity: f32,
        radius_small: f32,
        radius_medium: f32,
        radius_large: f32,
        tint: [u8; 4],
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let small = self.apply_gpu_bloom_texture_low_res_tinted(
            original,
            threshold,
            intensity * 0.45,
            radius_small,
            0.5,
            tint,
        )?;
        let medium = self.apply_gpu_bloom_texture_low_res_tinted(
            &small,
            threshold * 0.85,
            intensity * 0.35,
            radius_medium,
            0.25,
            tint,
        )?;
        self.apply_gpu_bloom_texture_low_res_tinted(
            &medium,
            threshold * 0.65,
            intensity * 0.20,
            radius_large,
            0.125,
            tint,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_gpu_bloom_texture_low_res_tinted(
        &mut self,
        original: &GpuSceneNativeTexture,
        threshold: f32,
        intensity: f32,
        sigma: f32,
        scale: f32,
        tint: [u8; 4],
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let scale = scale.clamp(0.05, 1.0);
        let bloom_width = ((original.width.max(1) as f32) * scale).round().max(1.0) as u32;
        let bloom_height = ((original.height.max(1) as f32) * scale).round().max(1.0) as u32;
        let downsampled = self.apply_gpu_downsample_texture(original, bloom_width, bloom_height)?;
        let scaled_sigma = (sigma * scale).max(1.0);
        let blurred = self
            .apply_gpu_blur_texture(&downsampled, &[(true, scaled_sigma), (false, scaled_sigma)])?;
        self.apply_gpu_bloom_texture_tinted(original, &blurred, threshold, intensity, tint)
    }

    pub(crate) fn apply_gpu_tint_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        color: [u8; 4],
        intensity: f32,
    ) -> GpuSceneNativeTexture {
        let width = input.width.max(1);
        let height = input.height.max(1);
        let mut encoder =
            self.frame_encoder("anica-motionloom-scene-post-tint-texture-gpu-encoder");
        let uniform = post_tint_uniform(width, height, color, intensity);
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass_sized(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            width,
            height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: vec![input.texture.clone()],
        }
    }

    /// Apply a bloom composite pass to a GPU texture.
    ///
    /// The prefilter is applied by the shader (threshold-based luminance mask),
    /// then the blurred texture is composited. This avoids a CPU readback.
    pub(crate) fn apply_gpu_bloom_texture(
        &mut self,
        original: &GpuSceneNativeTexture,
        blurred: &GpuSceneNativeTexture,
        threshold: f32,
        intensity: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        self.apply_gpu_bloom_texture_tinted(
            original,
            blurred,
            threshold,
            intensity,
            [255, 255, 255, 255],
        )
    }

    pub(crate) fn apply_gpu_bloom_texture_tinted(
        &mut self,
        original: &GpuSceneNativeTexture,
        blurred: &GpuSceneNativeTexture,
        threshold: f32,
        intensity: f32,
        tint: [u8; 4],
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        #[cfg(not(target_arch = "wasm32"))]
        let encode_started = Instant::now();
        let width = original.width.max(1);
        let height = original.height.max(1);
        let mut encoder = self.frame_encoder("anica-motionloom-scene-bloom-gpu-encoder");
        let uniform =
            crate::scene::drawable::bloom_tint_uniform(width, height, threshold, intensity, tint);
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_bloom_pass(
            &mut encoder,
            &original.texture,
            &blurred.texture,
            &dst,
            &uniform_buffer,
            width,
            height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        let result = GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: vec![original.texture.clone(), blurred.texture.clone()],
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.current_cpu_encode += encode_started.elapsed();
        }
        Ok(result)
    }

    pub(crate) fn apply_gpu_tone_map_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        exposure: f32,
        contrast: f32,
        shoulder: f32,
        gamma: f32,
        saturation: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let width = input.width.max(1);
        let height = input.height.max(1);
        let mut encoder = self.frame_encoder("anica-motionloom-scene-tone-map-gpu-encoder");
        let uniform = post_tone_map_uniform(
            width, height, exposure, contrast, shoulder, gamma, saturation,
        );
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass_sized(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            width,
            height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_gpu_light_sweep_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        position: f32,
        angle: f32,
        width_param: f32,
        softness: f32,
        intensity: f32,
        color: [u8; 4],
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let width = input.width.max(1);
        let height = input.height.max(1);
        let mut encoder = self.frame_encoder("anica-motionloom-scene-light-sweep-gpu-encoder");
        let uniform = post_light_sweep_uniform(PostLightSweepUniformParams {
            canvas_w: width,
            canvas_h: height,
            position,
            angle,
            width: width_param,
            softness,
            intensity,
            color,
        });
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_light_sweep_pass(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            width,
            height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_gpu_spectral_energy_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        mode: f32,
        active: f32,
        intensity: f32,
        density: f32,
        drift: f32,
        seed: f32,
        time_sec: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let width = input.width.max(1);
        let height = input.height.max(1);
        let values = [
            width as f32,
            height as f32,
            active.clamp(0.0, 1.0),
            intensity.clamp(0.0, 8.0),
            density.clamp(0.05, 8.0),
            drift,
            seed,
            time_sec,
            mode,
            0.0,
            0.0,
            0.0,
        ];
        let mut uniform = [0u8; 48];
        for (index, value) in values.into_iter().enumerate() {
            let offset = index * 4;
            uniform[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        }
        let mut encoder = self.frame_encoder("anica-motionloom-scene-spectral-energy-gpu-encoder");
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_spectral_energy_pass(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            width,
            height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    pub(crate) fn apply_gpu_texture_overlay_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        params: SceneTextureOverlayParams,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let width = input.width.max(1);
        let height = input.height.max(1);
        let mut encoder = self.frame_encoder("anica-motionloom-scene-texture-overlay-gpu-encoder");
        let uniform = post_texture_overlay_uniform(PostTextureOverlayUniformParams {
            canvas_w: width,
            canvas_h: height,
            kind: params.kind.id(),
            scale: params.scale,
            strength: params.strength,
            contrast: params.contrast,
            seed: params.seed,
            brush_angle: params.brush_angle,
            bump_strength: params.bump_strength,
            relief: params.relief,
            asset_flags: 0.0,
        });
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass_sized(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            width,
            height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_gpu_material_displacement_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        kind: f32,
        scale: f32,
        amount: f32,
        seed: f32,
        roughness: f32,
        specular: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let (width, height) = (input.width.max(1), input.height.max(1));
        let uniform = post_material_displacement_uniform(
            width, height, kind, scale, amount, seed, roughness, specular,
        );
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut encoder = self.frame_encoder("motionloom-material-displacement");
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass_sized(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            width,
            height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    pub(crate) fn apply_gpu_image_texture_overlay_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        params: SceneTextureOverlayParams,
        texture_image: Option<&RgbaImage>,
        height_image: Option<&RgbaImage>,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let width = input.width.max(1);
        let height = input.height.max(1);
        let mut encoder =
            self.frame_encoder("anica-motionloom-scene-image-texture-overlay-gpu-encoder");
        let texture_source = if let Some(image) = texture_image {
            Some(self.upload_gpu_rgba_texture(image)?)
        } else {
            None
        };
        let height_source = if let Some(image) = height_image {
            Some(self.upload_gpu_rgba_texture(image)?)
        } else {
            None
        };
        let asset_flags = if height_source.is_some() {
            2.0
        } else if texture_source.is_some() {
            1.0
        } else {
            0.0
        };
        let uniform = post_texture_overlay_uniform(PostTextureOverlayUniformParams {
            canvas_w: width,
            canvas_h: height,
            kind: params.kind.id(),
            scale: params.scale,
            strength: params.strength,
            contrast: params.contrast,
            seed: params.seed,
            brush_angle: params.brush_angle,
            bump_strength: params.bump_strength,
            relief: params.relief,
            asset_flags,
        });
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut keepalive = WgpuDispatchKeepalive::default();
        let overlay_texture = texture_source
            .as_ref()
            .map(|texture| texture.texture.as_ref())
            .unwrap_or(self.dummy_post_texture.as_ref());
        let height_texture = height_source
            .as_ref()
            .map(|texture| texture.texture.as_ref())
            .or_else(|| {
                texture_source
                    .as_ref()
                    .map(|texture| texture.texture.as_ref())
            })
            .unwrap_or(self.dummy_post_texture.as_ref());
        self.dispatch_post_pass_sized_with_aux(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            width,
            height,
            overlay_texture,
            height_texture,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    pub(crate) fn apply_gpu_magnify_lens_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        params: SceneMagnifyLensParams,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let width = input.width.max(1);
        let height = input.height.max(1);
        let mut encoder = self.frame_encoder("anica-motionloom-scene-magnify-lens-gpu-encoder");
        let uniform = post_magnify_lens_uniform(PostMagnifyLensUniformParams {
            canvas_w: width,
            canvas_h: height,
            x: params.x,
            y: params.y,
            radius: params.radius,
            zoom: params.zoom,
            distortion: params.distortion,
            feather: params.feather,
            glass: params.glass,
        });
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(&self.device, width, height));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass_sized(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            width,
            height,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width,
            height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    pub(crate) fn upload_gpu_rgba_texture(
        &mut self,
        image: &RgbaImage,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let width = image.width().max(1);
        let height = image.height().max(1);
        let texture = std::sync::Arc::new(self.make_source_texture(width, height));
        self.write_texture_rgba(&texture, width, height, image.as_raw())?;
        Ok(GpuSceneNativeTexture {
            texture,
            width,
            height,
            _keepalive_textures: Vec::new(),
        })
    }

    pub(crate) fn apply_gpu_color_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        brightness: f32,
        contrast: f32,
        saturation: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        if input.width != self.width || input.height != self.height {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "texture color input size {}x{} does not match GPU compositor {}x{}",
                    input.width, input.height, self.width, self.height
                ),
            });
        }

        let mut encoder =
            self.frame_encoder("anica-motionloom-scene-post-color-texture-gpu-encoder");
        let uniform = post_color_uniform(self.width, self.height, brightness, contrast, saturation);
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width: self.width,
            height: self.height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    pub(crate) fn apply_gpu_opacity_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        opacity: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        if input.width != self.width || input.height != self.height {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "texture opacity input size {}x{} does not match GPU compositor {}x{}",
                    input.width, input.height, self.width, self.height
                ),
            });
        }
        let mut encoder =
            self.frame_encoder("anica-motionloom-scene-post-opacity-texture-gpu-encoder");
        let uniform = post_opacity_uniform(self.width, self.height, opacity);
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width: self.width,
            height: self.height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    pub(crate) fn apply_gpu_edge_treatment_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        mode: f32,
        radius: f32,
        amount: f32,
        scale: f32,
        seed_or_preserve: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        if input.width != self.width || input.height != self.height {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "edge treatment input size {}x{} does not match GPU compositor {}x{}",
                    input.width, input.height, self.width, self.height
                ),
            });
        }
        let mut encoder = self.frame_encoder("anica-motionloom-scene-edge-treatment-gpu-encoder");
        let uniform = post_edge_treatment_uniform(
            self.width,
            self.height,
            mode,
            radius,
            amount,
            scale,
            seed_or_preserve,
        );
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        Ok(GpuSceneNativeTexture {
            texture: dst,
            width: self.width,
            height: self.height,
            _keepalive_textures: vec![input.texture.clone()],
        })
    }

    pub(crate) fn apply_gpu_hsla_overlay_texture(
        &mut self,
        input: &GpuSceneNativeTexture,
        hue: f32,
        saturation: f32,
        lightness: f32,
        alpha: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        #[cfg(not(target_arch = "wasm32"))]
        let encode_started = Instant::now();
        if input.width != self.width || input.height != self.height {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "texture HSLA input size {}x{} does not match GPU compositor {}x{}",
                    input.width, input.height, self.width, self.height
                ),
            });
        }

        let mut encoder =
            self.frame_encoder("anica-motionloom-scene-post-hsla-texture-gpu-encoder");
        let uniform =
            post_hsla_overlay_uniform(self.width, self.height, hue, saturation, lightness, alpha);
        let uniform_buffer = self.make_post_uniform_buffer(&uniform);
        let dst = std::sync::Arc::new(Self::make_canvas_texture(
            &self.device,
            self.width,
            self.height,
        ));
        let mut keepalive = WgpuDispatchKeepalive::default();
        self.dispatch_post_pass(
            &mut encoder,
            &input.texture,
            &dst,
            &uniform_buffer,
            &mut keepalive,
        );
        self.submit_encoder_with_keepalive(encoder, keepalive);
        drop(uniform_buffer);
        let result = GpuSceneNativeTexture {
            texture: dst,
            width: self.width,
            height: self.height,
            _keepalive_textures: vec![input.texture.clone()],
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.current_cpu_encode += encode_started.elapsed();
        }
        Ok(result)
    }

    pub(crate) fn make_uniform_buffer(&self, uniform: &[u8; 48]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anica-motionloom-scene-gpu-uniform"),
            size: uniform.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(uniform);
        buffer.unmap();
        buffer
    }

    pub(crate) fn make_matte_texture_uniform_buffer(&self, uniform: &[u8]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anica-motionloom-scene-matte-texture-gpu-uniform"),
            size: uniform.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(uniform);
        buffer.unmap();
        buffer
    }

    pub(crate) fn make_post_uniform_buffer(&self, uniform: &[u8]) -> wgpu::Buffer {
        self.make_initialized_buffer(
            "anica-motionloom-scene-post-gpu-uniform",
            uniform,
            wgpu::BufferUsages::UNIFORM,
        )
    }

    fn make_initialized_buffer(
        &self,
        label: &'static str,
        data: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len().max(4) as u64,
            usage,
            mapped_at_creation: true,
        });
        if !data.is_empty() {
            buffer
                .slice(..data.len() as u64)
                .get_mapped_range_mut()
                .copy_from_slice(data);
        }
        buffer.unmap();
        buffer
    }

    fn update_persistent_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: &mut Option<PersistentGpuBuffer>,
        label: &'static str,
        data: &[u8],
        usage: wgpu::BufferUsages,
    ) -> usize {
        if slot
            .as_ref()
            .is_some_and(|entry| entry.last_data.as_slice() == data)
        {
            return 0;
        }
        let required = data.len().max(4);
        let needs_grow = slot.as_ref().is_none_or(|entry| entry.capacity < required);
        if needs_grow {
            let capacity = required.checked_next_power_of_two().unwrap_or(required);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: capacity as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            if !data.is_empty() {
                buffer
                    .slice(..data.len() as u64)
                    .get_mapped_range_mut()
                    .copy_from_slice(data);
            }
            buffer.unmap();
            *slot = Some(PersistentGpuBuffer {
                buffer,
                capacity,
                last_data: data.to_vec(),
            });
        } else if let Some(entry) = slot.as_mut() {
            if !data.is_empty() {
                queue.write_buffer(&entry.buffer, 0, data);
            }
            entry.last_data.clear();
            entry.last_data.extend_from_slice(data);
        }
        data.len()
    }

    fn update_persistent_shape_buffers(
        &mut self,
        uniform: &[u8; 32],
        primitive_bytes: &[u8],
        transform_bytes: &[u8],
        tile_range_bytes: &[u8],
        tile_index_bytes: &[u8],
    ) -> usize {
        let mut uploaded = 0usize;
        uploaded += Self::update_persistent_buffer(
            &self.device,
            &self.queue,
            &mut self.shape_uniform_buffer,
            "anica-motionloom-scene-batch-shape-gpu-uniform-persistent",
            uniform,
            wgpu::BufferUsages::UNIFORM,
        );
        uploaded += Self::update_persistent_buffer(
            &self.device,
            &self.queue,
            &mut self.shape_primitive_buffer,
            "anica-motionloom-scene-shape-gpu-storage-persistent",
            primitive_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        uploaded += Self::update_persistent_buffer(
            &self.device,
            &self.queue,
            &mut self.shape_transform_buffer,
            "anica-motionloom-scene-shape-gpu-transforms-persistent",
            transform_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        uploaded += Self::update_persistent_buffer(
            &self.device,
            &self.queue,
            &mut self.shape_tile_range_buffer,
            "anica-motionloom-scene-shape-gpu-tile-ranges-persistent",
            tile_range_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        uploaded += Self::update_persistent_buffer(
            &self.device,
            &self.queue,
            &mut self.shape_tile_index_buffer,
            "anica-motionloom-scene-shape-gpu-tile-indices-persistent",
            tile_index_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        uploaded
    }

    pub(crate) fn make_storage_buffer(&self, label: &'static str, data: &[u8]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        buffer
            .slice(..data.len() as u64)
            .get_mapped_range_mut()
            .copy_from_slice(data);
        buffer.unmap();
        buffer
    }

    pub(crate) fn write_texture_rgba(
        &self,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<(), MotionLoomSceneRenderError> {
        if width == 0 || height == 0 {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "cannot upload zero-sized RGBA texture".to_string(),
            });
        }
        let row_bytes =
            width
                .checked_mul(4)
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: format!("RGBA texture row is too wide: {width} px"),
                })?;
        let expected_len = (row_bytes as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: format!("RGBA texture is too large: {width}x{height}"),
            })?;
        if rgba.len() < expected_len {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "RGBA texture upload has insufficient data: expected {expected_len} bytes for {width}x{height}, got {} bytes",
                    rgba.len()
                ),
            });
        }

        // Text layers are arbitrary-width CPU bitmaps. Padding rows keeps uploads
        // valid across stricter backends while preserving exact texel content.
        const ROW_ALIGNMENT: u32 = 256;
        let padded_row_bytes = row_bytes.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
        let upload: Cow<'_, [u8]> = if padded_row_bytes == row_bytes {
            Cow::Borrowed(&rgba[..expected_len])
        } else {
            let mut padded = vec![0u8; (padded_row_bytes as usize) * (height as usize)];
            for row in 0..height as usize {
                let src_start = row * row_bytes as usize;
                let dst_start = row * padded_row_bytes as usize;
                padded[dst_start..dst_start + row_bytes as usize]
                    .copy_from_slice(&rgba[src_start..src_start + row_bytes as usize]);
            }
            Cow::Owned(padded)
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    pub(crate) fn copy_texture_rect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src_texture: &wgpu::Texture,
        dst_texture: &wgpu::Texture,
        rect: TextureRect,
    ) {
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x,
                    y: rect.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x,
                    y: rect.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: rect.width,
                height: rect.height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub(crate) fn copy_native_texture_to_target(
        &mut self,
        src_texture: &wgpu::Texture,
        dst_texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) {
        if width == 0 || height == 0 {
            return;
        }
        let mut encoder = self.frame_encoder("anica-motionloom-scene-copy-to-target-encoder");
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.submit_encoder(encoder);
    }

    pub(crate) fn dispatch_image_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base_texture: &wgpu::Texture,
        image_texture: &wgpu::Texture,
        out_texture: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        let base_view = base_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let image_view = image_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-gpu-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anica-motionloom-scene-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.width.div_ceil(16).max(1),
            self.height.div_ceil(16).max(1),
            1,
        );
        drop(pass);
        keepalive
            .texture_views
            .extend([base_view, image_view, out_view]);
        keepalive.bind_groups.push(bind_group);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_batched_shape_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base_texture: &wgpu::Texture,
        out_texture: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
        primitive_buffer: &wgpu::Buffer,
        transform_buffer: &wgpu::Buffer,
        tile_range_buffer: &wgpu::Buffer,
        tile_index_buffer: &wgpu::Buffer,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        let base_view = base_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-shape-gpu-bg"),
            layout: &self.shape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: primitive_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tile_range_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tile_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: transform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anica-motionloom-scene-shape-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.shape_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            self.width.div_ceil(16).max(1),
            self.height.div_ceil(16).max(1),
            1,
        );
        drop(pass);
        keepalive.texture_views.extend([base_view, out_view]);
        keepalive.bind_groups.push(bind_group);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_puppet_deform_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_texture: &Arc<wgpu::Texture>,
        out_texture: &Arc<wgpu::Texture>,
        uniform_buffer: &wgpu::Buffer,
        triangle_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        keepalive
            .textures
            .extend([source_texture.clone(), out_texture.clone()]);

        let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-puppet-deform-gpu-bg"),
            layout: &self.puppet_deform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: triangle_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anica-motionloom-scene-puppet-deform-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.puppet_deform_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(16).max(1), height.div_ceil(16).max(1), 1);
        drop(pass);
        keepalive.texture_views.extend([source_view, out_view]);
        keepalive.bind_groups.push(bind_group);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_matte_texture_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base_texture: &Arc<wgpu::Texture>,
        image_texture: &Arc<wgpu::Texture>,
        matte_texture: &Arc<wgpu::Texture>,
        out_texture: &Arc<wgpu::Texture>,
        uniform_buffer: &wgpu::Buffer,
        bounds_w: u32,
        bounds_h: u32,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        keepalive.textures.extend([
            base_texture.clone(),
            image_texture.clone(),
            matte_texture.clone(),
            out_texture.clone(),
        ]);

        let base_view = base_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("anica-motionloom-scene-matte-base-view"),
            ..Default::default()
        });
        let image_view = image_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("anica-motionloom-scene-matte-image-view"),
            ..Default::default()
        });
        let matte_view = matte_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("anica-motionloom-scene-matte-mask-view"),
            ..Default::default()
        });
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("anica-motionloom-scene-matte-out-view"),
            ..Default::default()
        });

        self.debug_preflight_matte_texture_view("base(binding0)", &base_view, false);
        self.debug_preflight_matte_texture_view("image(binding1)", &image_view, true);
        self.debug_preflight_matte_texture_view("matte(binding2)", &matte_view, true);
        self.debug_preflight_matte_texture_view("out(binding4)", &out_view, false);
        if Self::debug_gpu_matte_enabled() {
            eprintln!(
                "[MotionLoom][GPU Matte] actual bind group begin bounds={}x{} canvas={}x{}",
                bounds_w, bounds_h, self.width, self.height
            );
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-matte-texture-gpu-bg"),
            layout: &self.matte_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&matte_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        if Self::debug_gpu_matte_enabled() {
            eprintln!("[MotionLoom][GPU Matte] actual bind group ok");
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anica-motionloom-scene-matte-texture-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.matte_texture_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            bounds_w.div_ceil(16).max(1),
            bounds_h.div_ceil(16).max(1),
            1,
        );
        drop(pass);
        keepalive
            .texture_views
            .extend([base_view, image_view, matte_view, out_view]);
        keepalive.bind_groups.push(bind_group);
    }

    pub(crate) fn dispatch_post_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base_texture: &wgpu::Texture,
        out_texture: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        self.dispatch_post_pass_sized(
            encoder,
            base_texture,
            out_texture,
            uniform_buffer,
            self.width,
            self.height,
            keepalive,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_post_pass_sized(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base_texture: &wgpu::Texture,
        out_texture: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        self.dispatch_post_pass_sized_with_aux(
            encoder,
            base_texture,
            out_texture,
            uniform_buffer,
            width,
            height,
            self.dummy_post_texture.as_ref(),
            self.dummy_post_texture.as_ref(),
            keepalive,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_post_pass_sized_with_aux(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base_texture: &wgpu::Texture,
        out_texture: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        overlay_texture: &wgpu::Texture,
        height_texture: &wgpu::Texture,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        let base_view = base_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let overlay_view = overlay_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let height_view = height_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-post-gpu-bg"),
            layout: &self.post_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&overlay_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anica-motionloom-scene-post-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.post_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(16).max(1), height.div_ceil(16).max(1), 1);
        drop(pass);
        keepalive
            .texture_views
            .extend([base_view, out_view, overlay_view, height_view]);
        keepalive.bind_groups.push(bind_group);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_light_sweep_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base_texture: &wgpu::Texture,
        out_texture: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        let base_view = base_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-light-sweep-gpu-bg"),
            layout: &self.post_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anica-motionloom-scene-light-sweep-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.light_sweep_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(16).max(1), height.div_ceil(16).max(1), 1);
        drop(pass);
        keepalive.texture_views.extend([base_view, out_view]);
        keepalive.bind_groups.push(bind_group);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_spectral_energy_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base_texture: &wgpu::Texture,
        out_texture: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        let base_view = base_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-spectral-energy-gpu-bg"),
            layout: &self.spectral_energy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anica-motionloom-scene-spectral-energy-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.spectral_energy_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(16).max(1), height.div_ceil(16).max(1), 1);
        drop(pass);
        keepalive.texture_views.extend([base_view, out_view]);
        keepalive.bind_groups.push(bind_group);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_downsample_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src_texture: &wgpu::Texture,
        out_texture: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-downsample-gpu-bg"),
            layout: &self.downsample_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anica-motionloom-scene-downsample-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.downsample_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(16).max(1), height.div_ceil(16).max(1), 1);
        drop(pass);
        keepalive.texture_views.extend([src_view, out_view]);
        keepalive.bind_groups.push(bind_group);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_bloom_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        original_texture: &wgpu::Texture,
        blurred_texture: &wgpu::Texture,
        out_texture: &wgpu::Texture,
        uniform_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        keepalive: &mut WgpuDispatchKeepalive,
    ) {
        let original_view = original_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blurred_view = blurred_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anica-motionloom-scene-bloom-gpu-bg"),
            layout: &self.bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&original_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blurred_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anica-motionloom-scene-bloom-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.bloom_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(16).max(1), height.div_ceil(16).max(1), 1);
        drop(pass);
        keepalive
            .texture_views
            .extend([original_view, blurred_view, out_view]);
        keepalive.bind_groups.push(bind_group);
    }

    pub(crate) fn load_image_texture(
        &mut self,
        src: &str,
    ) -> Result<(u32, u32, std::sync::Arc<wgpu::Texture>), MotionLoomSceneRenderError> {
        if !self.image_textures.contains_key(src) {
            let image = load_rgba_image_source(src, self.asset_resolver.as_ref())?;
            let (width, height) = image.dimensions();
            let texture = self.make_source_texture(width.max(1), height.max(1));
            self.write_texture_rgba(&texture, width.max(1), height.max(1), image.as_raw())?;
            self.image_textures.insert(
                src.to_string(),
                WgpuImageTexture {
                    width: width.max(1),
                    height: height.max(1),
                    texture: std::sync::Arc::new(texture),
                },
            );
        }
        let source = self
            .image_textures
            .get(src)
            .expect("GPU image texture inserted before lookup");
        Ok((source.width, source.height, source.texture.clone()))
    }

    pub(crate) fn load_svg_texture(
        &mut self,
        src: &str,
    ) -> Result<(u32, u32, std::sync::Arc<wgpu::Texture>), MotionLoomSceneRenderError> {
        let cache_key = format!("svg:{src}");
        if !self.image_textures.contains_key(&cache_key) {
            let image = load_svg_source(src, self.asset_resolver.as_ref())?;
            let (width, height) = image.dimensions();
            let texture = self.make_source_texture(width.max(1), height.max(1));
            self.write_texture_rgba(&texture, width.max(1), height.max(1), image.as_raw())?;
            self.image_textures.insert(
                cache_key.clone(),
                WgpuImageTexture {
                    width: width.max(1),
                    height: height.max(1),
                    texture: std::sync::Arc::new(texture),
                },
            );
        }
        let source = self
            .image_textures
            .get(&cache_key)
            .expect("GPU SVG texture inserted before lookup");
        Ok((source.width, source.height, source.texture.clone()))
    }

    pub(crate) async fn readback_texture_rgba_async(
        &self,
        texture: &wgpu::Texture,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-motionloom-scene-gpu-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        self.readback_rgba_async().await
    }

    pub(crate) async fn readback_texture_pixel_rgba_async(
        &self,
        texture: &wgpu::Texture,
        x: u32,
        y: u32,
    ) -> Result<[u8; 4], MotionLoomSceneRenderError> {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        let bytes_per_row = 256_u32;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anica-motionloom-scene-gpu-pick-pixel-readback"),
            size: bytes_per_row as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anica-motionloom-scene-gpu-pick-pixel-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        BufferMapAsyncFuture::new(&self._poller, &readback)
            .await
            .map_err(|err| MotionLoomSceneRenderError::GpuRender {
                message: format!("pick pixel readback map failed: {err}"),
            })?;
        let mapped = slice.get_mapped_range();
        let mut pixel = [0_u8; 4];
        pixel.copy_from_slice(&mapped[..4]);
        drop(mapped);
        readback.unmap();
        Ok(pixel)
    }

    pub(crate) async fn readback_rgba_async(
        &self,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let slice = self.readback_buffer.slice(..);
        BufferMapAsyncFuture::new(&self._poller, &self.readback_buffer)
            .await
            .map_err(|err| MotionLoomSceneRenderError::GpuRender {
                message: format!("readback map failed: {err}"),
            })?;

        let mapped = slice.get_mapped_range();
        let row_bytes = self.width as usize * 4;
        let padded_row = self.padded_bytes_per_row as usize;
        let mut out = vec![0u8; row_bytes * self.height as usize];
        for row in 0..self.height as usize {
            let src_off = row * padded_row;
            let dst_off = row * row_bytes;
            out[dst_off..dst_off + row_bytes]
                .copy_from_slice(&mapped[src_off..src_off + row_bytes]);
        }
        drop(mapped);
        self.readback_buffer.unmap();
        RgbaImage::from_raw(self.width, self.height, out).ok_or_else(|| {
            MotionLoomSceneRenderError::GpuRender {
                message: "failed to build RGBA image from GPU readback".to_string(),
            }
        })
    }
}

async fn request_scene_gpu_adapter_async(
    instance: &wgpu::Instance,
) -> Result<wgpu::Adapter, MotionLoomSceneRenderError> {
    let first_preference =
        wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance);
    let preferences = [
        first_preference,
        wgpu::PowerPreference::HighPerformance,
        wgpu::PowerPreference::LowPower,
        wgpu::PowerPreference::None,
    ];

    let mut errors = Vec::<String>::new();
    for (ix, preference) in preferences.iter().copied().enumerate() {
        if preferences[..ix].contains(&preference) {
            continue;
        }
        // Stay GPU-first by falling back between GPU adapter classes, not to CPU rendering.
        match request_adapter_async(
            instance,
            &wgpu::RequestAdapterOptions {
                power_preference: preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        )
        .await
        {
            Ok(adapter) => return Ok(adapter),
            Err(err) => errors.push(format!("{preference:?}: {err}")),
        }
    }

    Err(MotionLoomSceneRenderError::GpuRender {
        message: format!(
            "no compatible GPU adapter was available for scene rendering ({})",
            errors.join("; ")
        ),
    })
}

fn align_to_256(v: u32) -> u32 {
    const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    v.div_ceil(ALIGN) * ALIGN
}

fn downsample_uniform(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0..4].copy_from_slice(&(src_w.max(1) as f32).to_ne_bytes());
    out[4..8].copy_from_slice(&(src_h.max(1) as f32).to_ne_bytes());
    out[8..12].copy_from_slice(&(dst_w.max(1) as f32).to_ne_bytes());
    out[12..16].copy_from_slice(&(dst_h.max(1) as f32).to_ne_bytes());
    out
}

fn puppet_deform_uniform(
    width: u32,
    height: u32,
    triangle_count: u32,
    preserve_outside: bool,
) -> [u8; 16] {
    let values = [
        width.max(1) as f32,
        height.max(1) as f32,
        triangle_count as f32,
        if preserve_outside { 1.0 } else { 0.0 },
    ];
    let mut out = [0u8; 16];
    for (ix, value) in values.iter().enumerate() {
        out[ix * 4..ix * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    out
}

fn deform_grid_triangle_bytes(grid: &EvaluatedDeformGrid) -> Vec<u8> {
    let mut out = Vec::new();
    if !grid.triangles.is_empty() {
        for triangle in &grid.triangles {
            push_deform_triangle_bytes(grid, *triangle, &mut out);
        }
        return out;
    }
    if grid.cols < 2 || grid.rows < 2 {
        return out;
    }
    for row in 0..grid.rows - 1 {
        for col in 0..grid.cols - 1 {
            let i00 = row * grid.cols + col;
            let i10 = i00 + 1;
            let i01 = (row + 1) * grid.cols + col;
            let i11 = i01 + 1;
            push_deform_triangle_bytes(grid, [i00, i10, i11], &mut out);
            push_deform_triangle_bytes(grid, [i00, i11, i01], &mut out);
        }
    }
    out
}

fn push_deform_triangle_bytes(grid: &EvaluatedDeformGrid, triangle: [usize; 3], out: &mut Vec<u8>) {
    if triangle
        .iter()
        .any(|index| *index >= grid.from.len() || *index >= grid.to.len())
    {
        return;
    }
    let src = [
        grid.sample_from[triangle[0]],
        grid.sample_from[triangle[1]],
        grid.sample_from[triangle[2]],
    ];
    let dst = [
        grid.to[triangle[0]],
        grid.to[triangle[1]],
        grid.to[triangle[2]],
    ];
    push_deform_vec4(
        out,
        src[0].x,
        src[0].y,
        grid.from[triangle[0]].x,
        grid.from[triangle[0]].y,
    );
    push_deform_vec4(
        out,
        src[1].x,
        src[1].y,
        grid.from[triangle[1]].x,
        grid.from[triangle[1]].y,
    );
    push_deform_vec4(
        out,
        src[2].x,
        src[2].y,
        grid.from[triangle[2]].x,
        grid.from[triangle[2]].y,
    );
    push_deform_vec4(out, dst[0].x, dst[0].y, 0.0, 0.0);
    push_deform_vec4(out, dst[1].x, dst[1].y, 0.0, 0.0);
    push_deform_vec4(out, dst[2].x, dst[2].y, 0.0, 0.0);
}

fn push_deform_vec4(out: &mut Vec<u8>, x: f32, y: f32, z: f32, w: f32) {
    for value in [x, y, z, w] {
        out.extend_from_slice(&value.to_ne_bytes());
    }
}

fn union_texture_rect(current: Option<TextureRect>, next: TextureRect) -> Option<TextureRect> {
    if next.width == 0 || next.height == 0 {
        return current;
    }
    let Some(current) = current else {
        return Some(next);
    };
    let x0 = current.x.min(next.x);
    let y0 = current.y.min(next.y);
    let x1 = current
        .x
        .saturating_add(current.width)
        .max(next.x.saturating_add(next.width));
    let y1 = current
        .y
        .saturating_add(current.height)
        .max(next.y.saturating_add(next.height));
    Some(TextureRect {
        x: x0,
        y: y0,
        width: x1.saturating_sub(x0),
        height: y1.saturating_sub(y0),
    })
}
