// =========================================
// =========================================
// crates/motionloom/src/scene/drawable/gpu.rs

use std::collections::HashMap;

use image::{Rgba, RgbaImage};

use crate::scene::render::MotionLoomSceneRenderError;

use super::{GradientUnits, PaintBounds, ResolvedGradient, SceneBlendMode};
use crate::scene::model::SceneNode;
use crate::scene::spatial::{Affine2, resolve_axis};
use crate::scene::text::TextNode;

pub(crate) const GPU_SHAPE_RECT_FILL: f32 = 1.0;
pub(crate) const GPU_SHAPE_RECT_STROKE: f32 = 2.0;
pub(crate) const GPU_SHAPE_CIRCLE_FILL: f32 = 3.0;
pub(crate) const GPU_SHAPE_CIRCLE_STROKE: f32 = 4.0;
pub(crate) const GPU_SHAPE_RECT_SHADOW: f32 = 5.0;
pub(crate) const GPU_SHAPE_CIRCLE_SHADOW: f32 = 6.0;
pub(crate) const GPU_SHAPE_SOLID: f32 = 7.0;
pub(crate) const GPU_SHAPE_LINE: f32 = 8.0;
pub(crate) const GPU_SHAPE_TRIANGLE_FILL: f32 = 9.0;
pub(crate) const GPU_SHAPE_ELLIPSE_FILL: f32 = 10.0;
pub(crate) const GPU_SHAPE_ELLIPSE_STROKE: f32 = 11.0;

#[derive(Debug, Clone)]
pub(crate) struct GpuScenePrimitive {
    pub(crate) kind: f32,
    pub(crate) transform: Affine2,
    pub(crate) shape: [f32; 4],
    pub(crate) radius: f32,
    pub(crate) stroke_width: f32,
    pub(crate) blur: f32,
    pub(crate) color: [u8; 4],
    pub(crate) opacity: f32,
    pub(crate) blend: SceneBlendMode,
    pub(crate) pick_id: u32,
    pub(crate) gradient: Option<GpuSceneGradientPaint>,
    pub(crate) line_t0: f32,
    pub(crate) line_t1: f32,
    pub(crate) taper_start: f32,
    pub(crate) taper_end: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct GpuSceneGradientPaint {
    pub(crate) gradient: ResolvedGradient,
    pub(crate) bounds: PaintBounds,
}

#[derive(Debug, Clone)]
pub(crate) struct GpuSceneTextRequest {
    pub(crate) node: TextNode,
    pub(crate) transform: Affine2,
    pub(crate) opacity: f32,
    pub(crate) pick_id: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct GpuSceneNativeTexture {
    pub(crate) texture: std::sync::Arc<wgpu::Texture>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) _keepalive_textures: Vec<std::sync::Arc<wgpu::Texture>>,
}

#[derive(Debug, Clone)]
pub(crate) enum GpuSceneTextureSource {
    Cpu(RgbaImage),
    Gpu(GpuSceneNativeTexture),
}

impl GpuSceneTextureSource {
    pub(crate) fn width(&self) -> u32 {
        match self {
            Self::Cpu(image) => image.width(),
            Self::Gpu(texture) => texture.width,
        }
    }

    pub(crate) fn height(&self) -> u32 {
        match self {
            Self::Cpu(image) => image.height(),
            Self::Gpu(texture) => texture.height,
        }
    }
}

/// Graph-level texture source that can hold either a CPU RGBA image or a
/// GPU-native texture. This is used by the scene renderer resource pipeline
/// so that GPU-rendered scene output can be fed directly into GPU process
/// passes without an intermediate CPU readback.
#[derive(Debug, Clone)]
pub(crate) enum GraphTextureSource {
    Cpu(RgbaImage),
    Gpu(GpuSceneNativeTexture),
}

impl GraphTextureSource {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuSceneMatteMode {
    None,
    Alpha,
    Luma,
}

impl GpuSceneMatteMode {
    pub(crate) fn gpu_code(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Alpha => 1.0,
            Self::Luma => 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GpuSceneTextureMatte {
    pub(crate) texture: GpuSceneNativeTexture,
    pub(crate) mode: GpuSceneMatteMode,
    pub(crate) invert: bool,
    pub(crate) feather: f32,
    pub(crate) expansion: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct GpuSceneTextureLayer {
    pub(crate) source: GpuSceneTextureSource,
    pub(crate) transform: Affine2,
    pub(crate) projected_quad: Option<[(f32, f32, f32); 4]>,
    pub(crate) opacity: f32,
    pub(crate) blend: SceneBlendMode,
    pub(crate) pick_id: u32,
    pub(crate) matte: Option<GpuSceneTextureMatte>,
    /// Number of native primitives that must be composited before this layer.
    pub(crate) primitive_index: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum CpuSceneOverlay {
    Vector { nodes: Vec<SceneNode> },
}

#[derive(Debug, Default)]
pub(crate) struct GpuSceneNativeAssets {
    pub(crate) precomposes: HashMap<String, GpuSceneNativeTexture>,
    pub(crate) masks: HashMap<String, GpuSceneNativeTexture>,
}

pub(crate) fn gpu_solid_primitive(color: [u8; 4]) -> GpuScenePrimitive {
    GpuScenePrimitive {
        kind: GPU_SHAPE_SOLID,
        transform: Affine2::identity(),
        shape: [0.0, 0.0, 0.0, 0.0],
        radius: 0.0,
        stroke_width: 0.0,
        blur: 0.0,
        color,
        opacity: 1.0,
        blend: SceneBlendMode::Normal,
        pick_id: 0,
        gradient: None,
        line_t0: 0.0,
        line_t1: 1.0,
        taper_start: 0.0,
        taper_end: 0.0,
    }
}

pub(crate) fn solid_canvas(size: (u32, u32), color: [u8; 4]) -> RgbaImage {
    RgbaImage::from_pixel(size.0.max(1), size.1.max(1), Rgba(color))
}

pub(crate) fn describe_cpu_scene_overlays(overlays: &[CpuSceneOverlay]) -> String {
    let mut labels: Vec<String> = overlays
        .iter()
        .take(12)
        .map(describe_cpu_scene_overlay)
        .collect();
    if overlays.len() > labels.len() {
        labels.push(format!("+{} more", overlays.len() - labels.len()));
    }
    labels.join(", ")
}

fn describe_cpu_scene_overlay(overlay: &CpuSceneOverlay) -> String {
    match overlay {
        CpuSceneOverlay::Vector { nodes } => nodes
            .first()
            .map(describe_scene_node_for_gpu)
            .unwrap_or_else(|| "Vector".to_string()),
    }
}

fn describe_scene_node_for_gpu(node: &SceneNode) -> String {
    match node {
        SceneNode::Defs(_) => "Defs".to_string(),
        SceneNode::Timeline(timeline) => format!("Timeline{}", id_suffix(timeline.id.as_deref())),
        SceneNode::Track(track) => format!("Track{}", id_suffix(track.id.as_deref())),
        SceneNode::Sequence(sequence) => {
            format!("Sequence{}", id_suffix(sequence.id.as_deref()))
        }
        SceneNode::Chain(chain) => format!("Chain{}", id_suffix(chain.id.as_deref())),
        SceneNode::Palette(palette) => format!("Palette#{}", palette.id),
        SceneNode::PixelGrid(grid) => format!("PixelGrid{}", id_suffix(grid.id.as_deref())),
        SceneNode::Text(text) => format!("Text{}", id_suffix(text.id.as_deref())),
        SceneNode::Image(image) => format!("Image{}", id_suffix(image.id.as_deref())),
        SceneNode::Svg(svg) => format!("Svg{}", id_suffix(svg.id.as_deref())),
        SceneNode::Rect(rect) => format!("Rect{}", id_suffix(rect.id.as_deref())),
        SceneNode::Circle(circle) => format!("Circle{}", id_suffix(circle.id.as_deref())),
        SceneNode::Ellipse(ellipse) => format!("Ellipse{}", id_suffix(ellipse.id.as_deref())),
        SceneNode::Line(line) => format!("Line{}", id_suffix(line.id.as_deref())),
        SceneNode::Polyline(polyline) => {
            format!("Polyline{}", id_suffix(polyline.id.as_deref()))
        }
        SceneNode::Path(path) => format!("Path{}", id_suffix(path.id.as_deref())),
        SceneNode::FaceJaw(face_jaw) => format!("FaceJaw{}", id_suffix(face_jaw.id.as_deref())),
        SceneNode::Shadow(shadow) => format!("Shadow{}", id_suffix(shadow.id.as_deref())),
        SceneNode::Group(group) => format!("Group{}", id_suffix(group.id.as_deref())),
        SceneNode::Puppet(puppet) => format!("Puppet{}", id_suffix(puppet.id.as_deref())),
        SceneNode::Pin(pin) => format!("Pin{}", id_suffix(pin.id.as_deref())),
        SceneNode::LimbEnvelope(envelope) => {
            format!("LimbEnvelope{}", id_suffix(envelope.id.as_deref()))
        }
        SceneNode::LimbRegion(region) => {
            format!("LimbRegion{}", id_suffix(region.id.as_deref()))
        }
        SceneNode::MeshTopology(topology) => {
            format!("MeshTopology{}", id_suffix(topology.id.as_deref()))
        }
        SceneNode::Vertex(vertex) => format!("Vertex#{}", vertex.id),
        SceneNode::Triangle(triangle) => {
            format!("Triangle{}", id_suffix(triangle.id.as_deref()))
        }
        SceneNode::Edge(edge) => format!("Edge{}", id_suffix(edge.id.as_deref())),
        SceneNode::Region(region) => format!("Region#{}", region.id),
        SceneNode::Part(part) => format!("Part{}", id_suffix(part.id.as_deref())),
        SceneNode::Repeat(repeat) => format!("Repeat{}", id_suffix(repeat.id.as_deref())),
        SceneNode::Mask(mask) => format!("Mask{}", id_suffix(mask.id.as_deref())),
        SceneNode::Precompose(precompose) => format!("Precompose#{}", precompose.id),
        SceneNode::Use(use_node) => use_node
            .id
            .as_deref()
            .map(|id| format!("Use#{id}"))
            .unwrap_or_else(|| format!("Use(ref={})", use_node.ref_id)),
        SceneNode::Layer(layer) => layer
            .id
            .as_deref()
            .map(|id| format!("Layer#{id}"))
            .unwrap_or_else(|| {
                layer
                    .source
                    .as_deref()
                    .map(|source| format!("Layer(source={source})"))
                    .unwrap_or_else(|| "Layer".to_string())
            }),
        SceneNode::Camera(camera) => format!("Camera{}", id_suffix(camera.id.as_deref())),
        SceneNode::Character(character) => {
            format!("Character{}", id_suffix(character.id.as_deref()))
        }
        SceneNode::Simulation(_) => "SimulationBinding".to_string(),
    }
}

pub(crate) fn id_suffix(id: Option<&str>) -> String {
    id.map(|id| format!("#{id}")).unwrap_or_default()
}

const GPU_BATCH_TILE_SIZE: u32 = 32;

pub(crate) struct BatchedShapeData {
    pub(crate) primitive_bytes: Vec<u8>,
    pub(crate) transform_bytes: Vec<u8>,
    pub(crate) primitive_count: u32,
    pub(crate) tile_range_bytes: Vec<u8>,
    pub(crate) tile_index_bytes: Vec<u8>,
    pub(crate) tile_size: u32,
    pub(crate) tiles_x: u32,
    pub(crate) tiles_y: u32,
}

pub(crate) fn batch_shape_uniform(
    canvas_w: u32,
    canvas_h: u32,
    pick_mode: bool,
    primitive_count: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
) -> [u8; 32] {
    f32_bytes(&[
        canvas_w as f32,
        canvas_h as f32,
        if pick_mode { 1.0 } else { 0.0 },
        0.0,
        primitive_count as f32,
        tile_size as f32,
        tiles_x as f32,
        tiles_y as f32,
    ])
}

pub(crate) fn batch_shape_storage_bytes(
    primitives: &[GpuScenePrimitive],
    canvas_w: u32,
    canvas_h: u32,
) -> Result<BatchedShapeData, MotionLoomSceneRenderError> {
    let tile_size = GPU_BATCH_TILE_SIZE.max(1);
    let tiles_x = canvas_w.max(1).div_ceil(tile_size);
    let tiles_y = canvas_h.max(1).div_ceil(tile_size);
    let tile_count = tiles_x.saturating_mul(tiles_y) as usize;
    let mut tile_buckets = vec![Vec::<u32>::new(); tile_count];
    let mut primitive_bytes = Vec::with_capacity(primitives.len().saturating_mul(84 * 4));
    let mut transform_bytes = Vec::with_capacity(primitives.len().saturating_mul(12 * 4));
    let mut primitive_count = 0_u32;

    for primitive in primitives {
        let Some((bounds_x, bounds_y, bounds_w, bounds_h)) =
            primitive_bounds(primitive, canvas_w, canvas_h)
        else {
            continue;
        };
        if bounds_w == 0 || bounds_h == 0 {
            continue;
        }
        let values = batch_shape_primitive_values(primitive);
        push_f32_bytes(&mut primitive_bytes, &values);
        let transform_values =
            batch_shape_transform_values(primitive, bounds_x, bounds_y, bounds_w, bounds_h)?;
        push_f32_bytes(&mut transform_bytes, &transform_values);

        let x0 = (bounds_x / tile_size).min(tiles_x.saturating_sub(1));
        let y0 = (bounds_y / tile_size).min(tiles_y.saturating_sub(1));
        let x1 = ((bounds_x.saturating_add(bounds_w).saturating_sub(1)) / tile_size)
            .min(tiles_x.saturating_sub(1));
        let y1 = ((bounds_y.saturating_add(bounds_h).saturating_sub(1)) / tile_size)
            .min(tiles_y.saturating_sub(1));
        for tile_y in y0..=y1 {
            for tile_x in x0..=x1 {
                let tile_ix = tile_y.saturating_mul(tiles_x).saturating_add(tile_x) as usize;
                if let Some(bucket) = tile_buckets.get_mut(tile_ix) {
                    bucket.push(primitive_count);
                }
            }
        }

        primitive_count = primitive_count.saturating_add(1);
    }

    let mut tile_range_bytes = Vec::with_capacity(tile_count.saturating_mul(16));
    let mut tile_index_bytes = Vec::new();
    let mut index_offset = 0_u32;
    for bucket in tile_buckets {
        push_u32_bytes(
            &mut tile_range_bytes,
            &[index_offset, bucket.len() as u32, 0, 0],
        );
        for primitive_ix in bucket {
            push_u32_bytes(&mut tile_index_bytes, &[primitive_ix]);
        }
        index_offset = (tile_index_bytes.len() / 4) as u32;
    }

    if tile_index_bytes.is_empty() {
        push_u32_bytes(&mut tile_index_bytes, &[0]);
    }

    Ok(BatchedShapeData {
        primitive_bytes,
        transform_bytes,
        primitive_count,
        tile_range_bytes,
        tile_index_bytes,
        tile_size,
        tiles_x,
        tiles_y,
    })
}

fn primitive_bounds(
    primitive: &GpuScenePrimitive,
    canvas_w: u32,
    canvas_h: u32,
) -> Option<(u32, u32, u32, u32)> {
    if primitive.kind == GPU_SHAPE_SOLID {
        return Some((0, 0, canvas_w.max(1), canvas_h.max(1)));
    }

    let (min_x, min_y, max_x, max_y) = local_primitive_bbox(primitive)?;
    let points = [
        primitive.transform.transform_point(min_x, min_y),
        primitive.transform.transform_point(max_x, min_y),
        primitive.transform.transform_point(max_x, max_y),
        primitive.transform.transform_point(min_x, max_y),
    ];
    let mut bx0 = f32::INFINITY;
    let mut by0 = f32::INFINITY;
    let mut bx1 = f32::NEG_INFINITY;
    let mut by1 = f32::NEG_INFINITY;
    for (x, y) in points {
        bx0 = bx0.min(x);
        by0 = by0.min(y);
        bx1 = bx1.max(x);
        by1 = by1.max(y);
    }
    let pad = 3.0;
    let x0 = (bx0 - pad).floor().clamp(0.0, canvas_w as f32) as u32;
    let y0 = (by0 - pad).floor().clamp(0.0, canvas_h as f32) as u32;
    let x1 = (bx1 + pad).ceil().clamp(0.0, canvas_w as f32) as u32;
    let y1 = (by1 + pad).ceil().clamp(0.0, canvas_h as f32) as u32;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0, y0, x1 - x0, y1 - y0))
}

fn local_primitive_bbox(primitive: &GpuScenePrimitive) -> Option<(f32, f32, f32, f32)> {
    let s = primitive.shape;
    let spread =
        if primitive.kind == GPU_SHAPE_RECT_SHADOW || primitive.kind == GPU_SHAPE_CIRCLE_SHADOW {
            primitive.blur * 1.8
        } else {
            primitive.stroke_width.max(0.0) + 2.0
        };
    if primitive.kind == GPU_SHAPE_RECT_FILL
        || primitive.kind == GPU_SHAPE_RECT_STROKE
        || primitive.kind == GPU_SHAPE_RECT_SHADOW
    {
        return Some((
            s[0] - spread,
            s[1] - spread,
            s[0] + s[2] + spread,
            s[1] + s[3] + spread,
        ));
    }
    if primitive.kind == GPU_SHAPE_CIRCLE_FILL
        || primitive.kind == GPU_SHAPE_CIRCLE_STROKE
        || primitive.kind == GPU_SHAPE_CIRCLE_SHADOW
    {
        let r = s[2] + spread;
        return Some((s[0] - r, s[1] - r, s[0] + r, s[1] + r));
    }
    if primitive.kind == GPU_SHAPE_LINE {
        let spread = primitive.stroke_width.max(1.0) * 0.5 + 2.0;
        return Some((
            s[0].min(s[2]) - spread,
            s[1].min(s[3]) - spread,
            s[0].max(s[2]) + spread,
            s[1].max(s[3]) + spread,
        ));
    }
    if primitive.kind == GPU_SHAPE_TRIANGLE_FILL {
        let spread = 2.0;
        return Some((
            s[0].min(s[2]).min(primitive.radius) - spread,
            s[1].min(s[3]).min(primitive.stroke_width) - spread,
            s[0].max(s[2]).max(primitive.radius) + spread,
            s[1].max(s[3]).max(primitive.stroke_width) + spread,
        ));
    }
    None
}

fn write_gpu_gradient_uniform(gradient: Option<&GpuSceneGradientPaint>, values: &mut [f32]) {
    debug_assert!(values.len() >= 52);
    let Some(gradient) = gradient else {
        return;
    };

    let (paint_kind, units, params, stops) = match &gradient.gradient {
        ResolvedGradient::Linear {
            x1,
            y1,
            x2,
            y2,
            stops,
            units,
        } => (
            1.0,
            gpu_gradient_units(*units),
            [*x1, *y1, *x2, *y2],
            stops.as_slice(),
        ),
        ResolvedGradient::Radial {
            cx,
            cy,
            r,
            stops,
            units,
        } => (
            2.0,
            gpu_gradient_units(*units),
            [*cx, *cy, *r, 0.0],
            stops.as_slice(),
        ),
    };

    let stop_count = stops.len().min(8);
    values[0..4].copy_from_slice(&[paint_kind, units, stop_count as f32, 0.0]);
    values[4..8].copy_from_slice(&[
        gradient.bounds.min_x,
        gradient.bounds.min_y,
        gradient.bounds.max_x,
        gradient.bounds.max_y,
    ]);
    values[8..12].copy_from_slice(&params);
    for (index, stop) in stops.iter().take(8).enumerate() {
        values[12 + index] = stop.offset;
        let color = rgba_u8_to_unit(stop.color);
        let color_offset = 20 + index * 4;
        values[color_offset..color_offset + 4].copy_from_slice(&color);
    }
}

fn gpu_gradient_units(units: GradientUnits) -> f32 {
    match units {
        GradientUnits::ObjectBoundingBox => 0.0,
        GradientUnits::UserSpace => 1.0,
    }
}

fn batch_shape_primitive_values(primitive: &GpuScenePrimitive) -> [f32; 84] {
    let color = rgba_u8_to_unit(primitive.color);
    let mut values = [0.0_f32; 84];
    values[..28].copy_from_slice(&[
        primitive.kind,
        primitive.blend.gpu_code(),
        primitive.pick_id as f32,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        primitive.shape[0],
        primitive.shape[1],
        primitive.shape[2],
        primitive.shape[3],
        primitive.radius,
        primitive.stroke_width,
        primitive.blur,
        primitive.opacity,
        color[0],
        color[1],
        color[2],
        color[3],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    write_gpu_gradient_uniform(primitive.gradient.as_ref(), &mut values[28..80]);
    values[80..84].copy_from_slice(&[
        primitive.line_t0,
        primitive.line_t1,
        primitive.taper_start,
        primitive.taper_end,
    ]);
    values
}

fn batch_shape_transform_values(
    primitive: &GpuScenePrimitive,
    bounds_x: u32,
    bounds_y: u32,
    bounds_w: u32,
    bounds_h: u32,
) -> Result<[f32; 12], MotionLoomSceneRenderError> {
    let inverse =
        primitive
            .transform
            .inverse()
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: "shape transform is not invertible".to_string(),
            })?;
    Ok([
        bounds_x as f32,
        bounds_y as f32,
        bounds_w as f32,
        bounds_h as f32,
        inverse.m00,
        inverse.m01,
        inverse.m02,
        0.0,
        inverse.m10,
        inverse.m11,
        inverse.m12,
        0.0,
    ])
}

fn push_f32_bytes(out: &mut Vec<u8>, values: &[f32]) {
    out.reserve(values.len().saturating_mul(4));
    for value in values {
        out.extend_from_slice(&value.to_ne_bytes());
    }
}

fn push_u32_bytes(out: &mut Vec<u8>, values: &[u32]) {
    out.reserve(values.len().saturating_mul(4));
    for value in values {
        out.extend_from_slice(&value.to_ne_bytes());
    }
}

fn f32_bytes<const N: usize>(values: &[f32]) -> [u8; N] {
    debug_assert_eq!(N, values.len().saturating_mul(4));
    let mut bytes = [0u8; N];
    for (ix, value) in values.iter().enumerate() {
        bytes[ix * 4..ix * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}

pub(crate) fn texture_layer_bounds(
    transform: Affine2,
    width: u32,
    height: u32,
    canvas_w: u32,
    canvas_h: u32,
) -> Option<(u32, u32, u32, u32)> {
    let w = width as f32;
    let h = height as f32;
    let points = [
        transform.transform_point(0.0, 0.0),
        transform.transform_point(w, 0.0),
        transform.transform_point(w, h),
        transform.transform_point(0.0, h),
    ];
    let mut bx0 = f32::INFINITY;
    let mut by0 = f32::INFINITY;
    let mut bx1 = f32::NEG_INFINITY;
    let mut by1 = f32::NEG_INFINITY;
    for (x, y) in points {
        bx0 = bx0.min(x);
        by0 = by0.min(y);
        bx1 = bx1.max(x);
        by1 = by1.max(y);
    }
    let pad = 2.0;
    let x0 = (bx0 - pad).floor().clamp(0.0, canvas_w as f32) as u32;
    let y0 = (by0 - pad).floor().clamp(0.0, canvas_h as f32) as u32;
    let x1 = (bx1 + pad).ceil().clamp(0.0, canvas_w as f32) as u32;
    let y1 = (by1 + pad).ceil().clamp(0.0, canvas_h as f32) as u32;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0, y0, x1 - x0, y1 - y0))
}

pub(crate) fn texture_layer_projected_bounds(
    quad: [(f32, f32, f32); 4],
    canvas_w: u32,
    canvas_h: u32,
) -> Option<(u32, u32, u32, u32)> {
    let mut bx0 = f32::INFINITY;
    let mut by0 = f32::INFINITY;
    let mut bx1 = f32::NEG_INFINITY;
    let mut by1 = f32::NEG_INFINITY;
    for (x, y, _) in quad {
        bx0 = bx0.min(x);
        by0 = by0.min(y);
        bx1 = bx1.max(x);
        by1 = by1.max(y);
    }
    let pad = 2.0;
    let x0 = (bx0 - pad).floor().clamp(0.0, canvas_w as f32) as u32;
    let y0 = (by0 - pad).floor().clamp(0.0, canvas_h as f32) as u32;
    let x1 = (bx1 + pad).ceil().clamp(0.0, canvas_w as f32) as u32;
    let y1 = (by1 + pad).ceil().clamp(0.0, canvas_h as f32) as u32;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0, y0, x1 - x0, y1 - y0))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn raster_texture_layer(
    texture: std::sync::Arc<wgpu::Texture>,
    source_w: u32,
    source_h: u32,
    x_expr: &str,
    y_expr: &str,
    scale: f32,
    opacity: f32,
    transform: Affine2,
    time_norm: f32,
    time_sec: f32,
    canvas_size: (u32, u32),
) -> Result<Option<GpuSceneTextureLayer>, MotionLoomSceneRenderError> {
    if source_w == 0 || source_h == 0 || opacity <= 0.0001 {
        return Ok(None);
    }
    let target_w = ((source_w as f32) * scale).round().max(1.0);
    let target_h = ((source_h as f32) * scale).round().max(1.0);
    let x_base = resolve_axis(x_expr, canvas_size.0 as f32, target_w, time_norm, time_sec)?;
    let y_base = resolve_axis(y_expr, canvas_size.1 as f32, target_h, time_norm, time_sec)?;
    let local_transform = Affine2::translate(x_base, y_base).mul(Affine2::scale_xy(
        target_w / source_w as f32,
        target_h / source_h as f32,
    ));
    Ok(Some(GpuSceneTextureLayer {
        source: GpuSceneTextureSource::Gpu(GpuSceneNativeTexture {
            texture,
            width: source_w,
            height: source_h,
            _keepalive_textures: Vec::new(),
        }),
        transform: transform.mul(local_transform),
        projected_quad: None,
        opacity,
        blend: SceneBlendMode::Normal,
        pick_id: 0,
        matte: None,
        primitive_index: usize::MAX,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matte_texture_uniform(
    layer: &GpuSceneTextureLayer,
    canvas_w: u32,
    canvas_h: u32,
    bounds_x: u32,
    bounds_y: u32,
    bounds_w: u32,
    bounds_h: u32,
    image_w: u32,
    image_h: u32,
    matte_w: u32,
    matte_h: u32,
    matte_mode: GpuSceneMatteMode,
    invert_matte: bool,
    pick_mode: bool,
) -> Result<[u8; 160], MotionLoomSceneRenderError> {
    let inverse =
        layer
            .transform
            .inverse()
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: "texture transform is not invertible".to_string(),
            })?;
    let point_sample_source =
        layer.projected_quad.is_none() && texture_layer_is_pixel_aligned_1_to_1(layer.transform);
    let point_sample_matte = point_sample_source
        && matte_mode != GpuSceneMatteMode::None
        && matte_w == image_w
        && matte_h == image_h;
    let quad = layer.projected_quad.unwrap_or([
        (0.0, 0.0, 1.0),
        (image_w as f32, 0.0, 1.0),
        (image_w as f32, image_h as f32, 1.0),
        (0.0, image_h as f32, 1.0),
    ]);
    let values = [
        canvas_w as f32,
        canvas_h as f32,
        if point_sample_source { 1.0 } else { 0.0 },
        if point_sample_matte { 1.0 } else { 0.0 },
        bounds_x as f32,
        bounds_y as f32,
        bounds_w as f32,
        bounds_h as f32,
        image_w as f32,
        image_h as f32,
        matte_w as f32,
        matte_h as f32,
        layer.opacity,
        layer.blend.gpu_code(),
        matte_mode.gpu_code(),
        if invert_matte { 1.0 } else { 0.0 },
        inverse.m00,
        inverse.m01,
        inverse.m02,
        if layer.projected_quad.is_some() {
            1.0
        } else {
            0.0
        },
        inverse.m10,
        inverse.m11,
        inverse.m12,
        0.0,
        quad[0].0,
        quad[0].1,
        quad[0].2,
        layer.pick_id as f32,
        quad[1].0,
        quad[1].1,
        quad[1].2,
        if pick_mode { 1.0 } else { 0.0 },
        quad[2].0,
        quad[2].1,
        quad[2].2,
        layer
            .matte
            .as_ref()
            .map(|matte| matte.expansion)
            .unwrap_or(0.0),
        quad[3].0,
        quad[3].1,
        quad[3].2,
        layer
            .matte
            .as_ref()
            .map(|matte| matte.feather)
            .unwrap_or(0.0),
    ];
    let mut uniform = [0u8; 160];
    for (ix, value) in values.iter().enumerate() {
        uniform[ix * 4..ix * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    Ok(uniform)
}

fn texture_layer_is_pixel_aligned_1_to_1(transform: Affine2) -> bool {
    const EPS: f32 = 0.0001;
    (transform.m00 - 1.0).abs() <= EPS
        && transform.m01.abs() <= EPS
        && transform.m10.abs() <= EPS
        && (transform.m11 - 1.0).abs() <= EPS
        && (transform.m02 - transform.m02.round()).abs() <= EPS
        && (transform.m12 - transform.m12.round()).abs() <= EPS
}

pub(crate) fn post_blur_uniform(
    canvas_w: u32,
    canvas_h: u32,
    horizontal: bool,
    sigma: f32,
) -> [u8; 48] {
    let values = [
        canvas_w as f32,
        canvas_h as f32,
        0.0,
        0.0,
        if horizontal { 0.0 } else { 1.0 },
        sigma.ceil().clamp(0.0, 64.0),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let mut uniform = [0u8; 48];
    for (ix, value) in values.iter().enumerate() {
        uniform[ix * 4..ix * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    uniform
}

pub(crate) fn post_color_uniform(
    canvas_w: u32,
    canvas_h: u32,
    brightness: f32,
    contrast: f32,
    saturation: f32,
) -> [u8; 48] {
    let values = [
        canvas_w as f32,
        canvas_h as f32,
        0.0,
        0.0,
        brightness,
        contrast,
        saturation,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let mut uniform = [0u8; 48];
    for (ix, value) in values.iter().enumerate() {
        uniform[ix * 4..ix * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    uniform
}

pub(crate) fn post_tint_uniform(
    canvas_w: u32,
    canvas_h: u32,
    color: [u8; 4],
    intensity: f32,
) -> [u8; 48] {
    let color = rgba_u8_to_unit(color);
    f32_bytes(&[
        canvas_w as f32,
        canvas_h as f32,
        intensity.max(0.0),
        color[3],
        color[0],
        color[1],
        color[2],
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
}

pub(crate) fn post_hsla_overlay_uniform(
    canvas_w: u32,
    canvas_h: u32,
    hue: f32,
    saturation: f32,
    lightness: f32,
    alpha: f32,
) -> [u8; 48] {
    f32_bytes(&[
        canvas_w as f32,
        canvas_h as f32,
        0.0,
        alpha.clamp(0.0, 1.0),
        hue,
        saturation,
        lightness,
        4.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
}

pub(crate) fn post_tone_map_uniform(
    canvas_w: u32,
    canvas_h: u32,
    exposure: f32,
    contrast: f32,
    shoulder: f32,
    gamma: f32,
    saturation: f32,
) -> [u8; 48] {
    f32_bytes(&[
        canvas_w as f32,
        canvas_h as f32,
        shoulder.clamp(0.05, 8.0),
        saturation.clamp(0.0, 4.0),
        exposure.clamp(-8.0, 8.0),
        contrast.clamp(0.0, 4.0),
        gamma.clamp(0.1, 8.0),
        5.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PostLightSweepUniformParams {
    pub(crate) canvas_w: u32,
    pub(crate) canvas_h: u32,
    pub(crate) position: f32,
    pub(crate) angle: f32,
    pub(crate) width: f32,
    pub(crate) softness: f32,
    pub(crate) intensity: f32,
    pub(crate) color: [u8; 4],
}

pub(crate) fn post_light_sweep_uniform(params: PostLightSweepUniformParams) -> [u8; 48] {
    let color = rgba_u8_to_unit(params.color);
    f32_bytes(&[
        params.canvas_w as f32,
        params.canvas_h as f32,
        params.softness.clamp(0.0001, 2.0),
        params.intensity.clamp(0.0, 16.0),
        params.position,
        params.angle,
        params.width.clamp(0.0001, 4.0),
        6.0,
        color[0],
        color[1],
        color[2],
        color[3],
    ])
}

pub(crate) fn post_opacity_uniform(canvas_w: u32, canvas_h: u32, opacity: f32) -> [u8; 48] {
    f32_bytes(&[
        canvas_w as f32,
        canvas_h as f32,
        0.0,
        0.0,
        opacity.clamp(0.0, 1.0),
        0.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
}

pub(crate) fn post_edge_treatment_uniform(
    canvas_w: u32,
    canvas_h: u32,
    mode: f32,
    radius: f32,
    amount: f32,
    scale: f32,
    seed_or_preserve: f32,
) -> [u8; 48] {
    f32_bytes(&[
        canvas_w as f32,
        canvas_h as f32,
        0.0,
        0.0,
        radius.clamp(0.0, 32.0),
        amount.clamp(0.0, 1.0),
        scale.max(0.001),
        mode,
        seed_or_preserve,
        0.0,
        0.0,
        0.0,
    ])
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PostTextureOverlayUniformParams {
    pub(crate) canvas_w: u32,
    pub(crate) canvas_h: u32,
    pub(crate) kind: f32,
    pub(crate) scale: f32,
    pub(crate) strength: f32,
    pub(crate) contrast: f32,
    pub(crate) seed: f32,
    pub(crate) brush_angle: f32,
    pub(crate) bump_strength: f32,
    pub(crate) relief: f32,
    pub(crate) asset_flags: f32,
}

pub(crate) fn post_texture_overlay_uniform(params: PostTextureOverlayUniformParams) -> [u8; 48] {
    f32_bytes(&[
        params.canvas_w as f32,
        params.canvas_h as f32,
        params.seed,
        params.contrast.clamp(0.0, 2.0),
        params.kind,
        params.scale.clamp(0.001, 4096.0),
        params.strength.clamp(0.0, 1.0),
        7.0,
        params.brush_angle,
        params.bump_strength.clamp(0.0, 2.0),
        params.relief.clamp(0.0, 2.0),
        params.asset_flags,
    ])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn post_material_displacement_uniform(
    canvas_w: u32,
    canvas_h: u32,
    kind: f32,
    scale: f32,
    amount: f32,
    seed: f32,
    roughness: f32,
    specular: f32,
) -> [u8; 48] {
    f32_bytes(&[
        canvas_w as f32,
        canvas_h as f32,
        roughness.clamp(0.0, 1.0),
        specular.clamp(0.0, 2.0),
        scale.clamp(0.001, 4096.0),
        amount.clamp(-256.0, 256.0),
        seed,
        9.0,
        kind,
        0.0,
        0.0,
        0.0,
    ])
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PostMagnifyLensUniformParams {
    pub(crate) canvas_w: u32,
    pub(crate) canvas_h: u32,
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) radius: f32,
    pub(crate) zoom: f32,
    pub(crate) distortion: f32,
    pub(crate) feather: f32,
    pub(crate) glass: f32,
}

pub(crate) fn post_magnify_lens_uniform(params: PostMagnifyLensUniformParams) -> [u8; 48] {
    f32_bytes(&[
        params.canvas_w as f32,
        params.canvas_h as f32,
        params.feather.clamp(0.0, 512.0),
        params.glass.clamp(0.0, 1.0),
        params.x,
        params.y,
        params.radius.clamp(0.001, 8192.0),
        8.0,
        params.zoom.clamp(0.001, 16.0),
        params.distortion.clamp(-2.0, 2.0),
        0.0,
        0.0,
    ])
}

pub(crate) fn bloom_tint_uniform(
    canvas_w: u32,
    canvas_h: u32,
    threshold: f32,
    intensity: f32,
    tint: [u8; 4],
) -> [u8; 32] {
    let tint = rgba_u8_to_unit(tint);
    f32_bytes(&[
        canvas_w as f32,
        canvas_h as f32,
        threshold.clamp(0.0, 1.0),
        intensity.clamp(0.0, 8.0),
        tint[0],
        tint[1],
        tint[2],
        tint[3],
    ])
}

fn rgba_u8_to_unit(color: [u8; 4]) -> [f32; 4] {
    [
        color[0] as f32 / 255.0,
        color[1] as f32 / 255.0,
        color[2] as f32 / 255.0,
        color[3] as f32 / 255.0,
    ]
}
