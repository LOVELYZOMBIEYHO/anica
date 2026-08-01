// =========================================
// =========================================
// crates/motionloom/src/scene/render.rs

use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers, math::point,
    path::Path as LyonPath,
};
use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::fs;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use crate::dsl::{GraphAssetKind, GraphScript, ProcessDefinitionNode};
use crate::process::model::{PassNode, PassParam};
use crate::process::runtime::{CompiledTimeExpr, eval_time_expr, parse_curve_ease};

#[cfg(not(target_arch = "wasm32"))]
use crate::scene::backend::encoding::scene_encoder_args;

pub use super::preview_surface::*;
use crate::asset::{AssetResolver, PathAssetResolver};
pub use crate::scene::backend::encoding::{
    SceneRenderProfile, SceneRenderProgress, next_scene_output_path,
    next_scene_output_path_for_profile,
};
use crate::scene::backend::gpu::WgpuSceneCompositor;
use crate::scene::backend::sizing::{
    fit_logical_canvas_to_output, graph_logical_render_size, graph_output_size,
    render_size_root_transform,
};
use crate::scene::compile::{
    graph_has_rich_scene_tree, scene_nodes_contain_image_or_svg, scene_nodes_for_present,
    scene_nodes_require_cpu_scene_compositing,
};
use crate::scene::composition::{
    SceneTextureOverlayParams, TextureOverlayKind, apply_alpha_mask, apply_alpha_mask_with_invert,
    apply_box_blur_pass, apply_color_core_pass, apply_deform_grid, apply_hsla_pass,
    apply_image_texture_overlay_pass, apply_layer_effects, apply_over_pass,
    apply_scene_filter_step, apply_scene_post_pass, blend_pixel, build_scene_bloom_prefilter,
    composite_layer, composite_layer_affine, composite_layer_affine_blend,
    composite_layer_affine_blend_clipped, composite_layer_affine_clipped,
    composite_layer_projected_quad_blend_clipped, composite_scene_bloom,
    composite_transformed_layer, composite_transformed_layer_anchored, draw_rgba_image,
    is_color_key_alpha_effect, pass_param_expr, scene_post_bloom_params, scene_post_blur_passes,
    scene_post_brightness_amount, scene_post_glow_stack_params, scene_post_light_sweep_params,
    scene_post_magnify_lens_params, scene_post_texture_overlay_params, scene_post_tone_map_params,
    shape_alpha_mask,
};
use crate::scene::domain::apply_action_graph_at_time;

use crate::scene::drawable::{
    CpuSceneOverlay, EvaluatedShadow, FillRule, GPU_SHAPE_CIRCLE_FILL, GPU_SHAPE_CIRCLE_SHADOW,
    GPU_SHAPE_CIRCLE_STROKE, GPU_SHAPE_ELLIPSE_FILL, GPU_SHAPE_ELLIPSE_STROKE, GPU_SHAPE_LINE,
    GPU_SHAPE_RECT_FILL, GPU_SHAPE_RECT_SHADOW, GPU_SHAPE_RECT_STROKE, GPU_SHAPE_TRIANGLE_FILL,
    GpuSceneGradientPaint, GpuSceneMatteMode, GpuSceneNativeAssets, GpuSceneNativeTexture,
    GpuScenePrimitive, GpuSceneTextRequest, GpuSceneTextureLayer, GpuSceneTextureMatte,
    GpuSceneTextureSource, GraphTextureSource, PaintBounds, Point2, ResolvedPaint, SceneBlendMode,
    StrokeStyle, StrokeTexture, affine_uniform_scale, describe_cpu_scene_overlays, draw_circle,
    draw_circle_paint, draw_circle_shadow, draw_circle_stroke, draw_ellipse_paint,
    draw_ellipse_stroke, draw_line_segment_styled, draw_rect_shadow, draw_rounded_rect,
    draw_rounded_rect_paint, draw_rounded_rect_stroke, draw_transformed_filled_polylines,
    draw_transformed_filled_polylines_paint_with_rule, draw_transformed_trimmed_polylines_styled,
    eval_line_stroke_style, eval_path_d, eval_path_stroke_style, eval_polyline_stroke_style,
    evaluate_shadow, evaluate_trim, face_jaw_to_path_node, gpu_matte_mode, gpu_solid_primitive,
    gradient_ref_id, id_suffix, is_gpu_native_blend, is_none_paint, normalize_path_subpaths,
    offset_path_subpaths, parse_color, parse_paint, parse_path_subpaths, parse_polyline_points,
    parse_scene_blend, point_distance, raster_texture_layer, resolve_gradient_paint,
    round_path_subpaths, scene_mask_mode_inverts, solid_canvas, stroke_hash_signed,
    stroke_taper_pressure, stroke_texture_copy_count, stroke_texture_seed, stroke_texture_variant,
    trimmed_polyline_segments_with_progress,
};
use crate::scene::dsl::{ImageNode, SvgNode};
use crate::scene::model::{
    CameraNode, CharacterNode, CircleNode, CompositeGroupConfig, DefsNode, EllipseNode,
    FaceJawNode, FilterDef, FontDef, GradientDef, GroupNode, LineNode, MaskNode, MaterialDef,
    NoiseDef, PaletteNode, PartNode, PathNode, PixelGridNode, PolylineNode, PrecomposeNode,
    PuppetNode, RectNode, RepeatNode, Scene3DNode, SceneEffectRef, SceneLayerNode, SceneNode,
    SceneTrackNode, TextureDef, UseNode,
};
pub use crate::scene::resource::{clear_scene_asset_roots, set_scene_asset_roots};
use crate::scene::resource::{
    collect_graph_component_defs, collect_graph_filter_defs, collect_graph_font_defs,
    collect_graph_gradient_defs, collect_graph_mask_defs, collect_graph_material_defs,
    collect_graph_noise_defs, collect_graph_palette_defs, collect_graph_precompose_defs,
    collect_graph_texture_defs, default_world_asset_root, load_extra_fonts, load_rgba_image_source,
    load_svg_source, resolve_local_scene_asset_path,
};

fn scene_track_composite_order(track: &SceneTrackNode) -> i32 {
    track.composite_order.unwrap_or(track.z)
}

fn scene_effect_resource<'a>(
    resources: &'a HashMap<String, GraphTextureSource>,
    id: &str,
) -> Option<&'a GraphTextureSource> {
    resources.get(id).or_else(|| {
        id.strip_prefix("input:")
            .or_else(|| id.strip_prefix("scene:"))
            .and_then(|plain| resources.get(plain))
    })
}

use crate::scene::spatial::{
    Affine2, CameraRect, EvaluatedDeformGrid, active_scene_camera_from_tracks, affine_is_identity,
    camera_transform, camera_viewport, camera_world_bounds, clamp_nonzero_signed_scale,
    eval_group_deform_grid, eval_puppet_deform_grid, find_scene_node_anchor, is_scene_camera_track,
    is_scene_world_track, resolve_axis, scene_character_local_transform,
    scene_circle_local_transform, scene_ellipse_local_transform, scene_group_local_transform,
    scene_layer_local_transform, scene_line_local_transform, scene_path_local_transform,
    scene_polyline_local_transform, scene_puppet_local_transform, scene_rect_local_transform,
    scene_text_local_transform, scene_use_local_transform, transform_and_deform_point,
    transform_and_deform_subpaths, transform_deform_grid,
};
use crate::scene::text::TextNode;

fn apply_animation_targets_at_frame(
    graph: &GraphScript,
    _frame: u32,
) -> Result<Option<GraphScript>, MotionLoomSceneRenderError> {
    if graph.animation_targets.is_empty() {
        return Ok(None);
    }

    let mut graph = graph.clone();
    let fps = graph.fps.max(1.0);
    for target in graph.animation_targets.clone() {
        let value = animation_target_expression(&target.keys, fps, &target.property)?;
        apply_animation_property_to_graph(&mut graph, &target.node, &target.property, value);
    }
    Ok(Some(graph))
}

fn animation_target_expression(
    keys: &[crate::dsl::AnimationKeyNode],
    fps: f32,
    property: &str,
) -> Result<String, MotionLoomSceneRenderError> {
    if keys.is_empty() {
        return Ok("0".to_string());
    }
    if property == "d" {
        return Ok(path_morph_expr_from_animation_keys(keys, fps));
    }
    if keys.len() == 1 {
        keys[0].value.parse::<f32>().map_err(|_| {
            MotionLoomSceneRenderError::InvalidExpression {
                expr: keys[0].value.clone(),
                message: "AnimationTarget key value must be numeric for transform and opacity properties."
                    .to_string(),
            }
        })?;
        return Ok(keys[0].value.clone());
    }

    let mut points = Vec::with_capacity(keys.len());
    for (index, key) in keys.iter().enumerate() {
        key.value.parse::<f32>().map_err(|_| {
            MotionLoomSceneRenderError::InvalidExpression {
                expr: key.value.clone(),
                message: "AnimationTarget key value must be numeric for transform and opacity properties."
                    .to_string(),
            }
        })?;
        // AnimationTarget historically applies the destination key's easing
        // to the segment. curve() stores easing on the segment's first point.
        let ease = keys.get(index + 1).unwrap_or(key).ease.as_str();
        parse_curve_ease(ease).map_err(|message| {
            MotionLoomSceneRenderError::InvalidExpression {
                expr: ease.to_string(),
                message,
            }
        })?;
        points.push(format!(
            "{}:{}:{}",
            format_float_for_animation(key.seconds),
            key.value,
            ease
        ));
    }
    Ok(format!("curve(\"{}\")", points.join(", ")))
}

fn path_morph_expr_from_animation_keys(keys: &[crate::dsl::AnimationKeyNode], _fps: f32) -> String {
    if keys.len() == 1 {
        return keys[0].value.clone();
    }
    let args = keys
        .iter()
        .map(|key| {
            format!(
                "\"{}:{}\"",
                format_float_for_animation(key.seconds),
                escape_path_morph_value(&key.value)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("morph({args})")
}

fn escape_path_morph_value(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn format_float_for_animation(value: f32) -> String {
    let mut text = format!("{value:.4}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

fn apply_animation_property_to_graph(
    graph: &mut GraphScript,
    node_id: &str,
    property: &str,
    value: String,
) {
    for scene in &mut graph.scenes {
        apply_animation_property_to_nodes(&mut scene.children, node_id, property, &value);
    }
    apply_animation_property_to_nodes(&mut graph.scene_nodes, node_id, property, &value);
    for text in &mut graph.texts {
        apply_animation_property_to_text(text, node_id, property, &value);
    }
    for image in &mut graph.images {
        apply_animation_property_to_image(image, node_id, property, &value);
    }
    for svg in &mut graph.svgs {
        apply_animation_property_to_svg(svg, node_id, property, &value);
    }
    for skeleton in &mut graph.skeletons {
        for bone in &mut skeleton.bones {
            if bone.id == node_id && property == "rotation" {
                bone.rotation = value.clone();
            }
        }
    }
}

fn apply_animation_property_to_nodes(
    nodes: &mut [SceneNode],
    node_id: &str,
    property: &str,
    value: &str,
) {
    for node in nodes {
        match node {
            SceneNode::Timeline(timeline) => {
                apply_animation_property_to_nodes(&mut timeline.children, node_id, property, value);
            }
            SceneNode::Track(track) => {
                apply_animation_property_to_nodes(&mut track.children, node_id, property, value);
            }
            SceneNode::Sequence(sequence) => {
                apply_animation_property_to_nodes(&mut sequence.children, node_id, property, value);
            }
            SceneNode::Chain(chain) => {
                apply_animation_property_to_nodes(&mut chain.children, node_id, property, value);
            }
            SceneNode::Text(text) => {
                apply_animation_property_to_text(text, node_id, property, value)
            }
            SceneNode::Image(image) => {
                apply_animation_property_to_image(image, node_id, property, value)
            }
            SceneNode::Svg(svg) => apply_animation_property_to_svg(svg, node_id, property, value),
            SceneNode::Rect(rect) => {
                apply_animation_property_to_transform(
                    rect.id.as_deref(),
                    &mut rect.x,
                    &mut rect.y,
                    Some(&mut rect.rotation),
                    &mut rect.scale,
                    Some(&mut rect.scale_x),
                    Some(&mut rect.scale_y),
                    Some(&mut rect.skew_x),
                    Some(&mut rect.skew_y),
                    Some(&mut rect.transform_origin_x),
                    Some(&mut rect.transform_origin_y),
                    &mut rect.opacity,
                    node_id,
                    property,
                    value,
                );
            }
            SceneNode::Circle(circle) => {
                apply_animation_property_to_transform(
                    circle.id.as_deref(),
                    &mut circle.x,
                    &mut circle.y,
                    Some(&mut circle.rotation),
                    &mut circle.scale,
                    Some(&mut circle.scale_x),
                    Some(&mut circle.scale_y),
                    Some(&mut circle.skew_x),
                    Some(&mut circle.skew_y),
                    Some(&mut circle.transform_origin_x),
                    Some(&mut circle.transform_origin_y),
                    &mut circle.opacity,
                    node_id,
                    property,
                    value,
                );
            }
            SceneNode::Ellipse(ellipse) => {
                apply_animation_property_to_transform(
                    ellipse.id.as_deref(),
                    &mut ellipse.x,
                    &mut ellipse.y,
                    Some(&mut ellipse.rotation),
                    &mut ellipse.scale,
                    Some(&mut ellipse.scale_x),
                    Some(&mut ellipse.scale_y),
                    Some(&mut ellipse.skew_x),
                    Some(&mut ellipse.skew_y),
                    Some(&mut ellipse.transform_origin_x),
                    Some(&mut ellipse.transform_origin_y),
                    &mut ellipse.opacity,
                    node_id,
                    property,
                    value,
                );
            }
            SceneNode::Line(line) => {
                apply_animation_property_to_transform(
                    line.id.as_deref(),
                    &mut line.x,
                    &mut line.y,
                    Some(&mut line.rotation),
                    &mut line.scale,
                    Some(&mut line.scale_x),
                    Some(&mut line.scale_y),
                    Some(&mut line.skew_x),
                    Some(&mut line.skew_y),
                    Some(&mut line.transform_origin_x),
                    Some(&mut line.transform_origin_y),
                    &mut line.opacity,
                    node_id,
                    property,
                    value,
                );
            }
            SceneNode::Polyline(polyline) => {
                apply_animation_property_to_transform(
                    polyline.id.as_deref(),
                    &mut polyline.x,
                    &mut polyline.y,
                    Some(&mut polyline.rotation),
                    &mut polyline.scale,
                    Some(&mut polyline.scale_x),
                    Some(&mut polyline.scale_y),
                    Some(&mut polyline.skew_x),
                    Some(&mut polyline.skew_y),
                    Some(&mut polyline.transform_origin_x),
                    Some(&mut polyline.transform_origin_y),
                    &mut polyline.opacity,
                    node_id,
                    property,
                    value,
                );
            }
            SceneNode::Path(path) => {
                if path.id.as_deref() == Some(node_id) && property == "d" {
                    path.d = value.to_string();
                } else {
                    apply_animation_property_to_transform(
                        path.id.as_deref(),
                        &mut path.x,
                        &mut path.y,
                        Some(&mut path.rotation),
                        &mut path.scale,
                        Some(&mut path.scale_x),
                        Some(&mut path.scale_y),
                        Some(&mut path.skew_x),
                        Some(&mut path.skew_y),
                        Some(&mut path.transform_origin_x),
                        Some(&mut path.transform_origin_y),
                        &mut path.opacity,
                        node_id,
                        property,
                        value,
                    );
                }
            }
            SceneNode::Group(group) => {
                apply_animation_property_to_transform(
                    group.id.as_deref(),
                    &mut group.x,
                    &mut group.y,
                    Some(&mut group.rotation),
                    &mut group.scale,
                    Some(&mut group.scale_x),
                    Some(&mut group.scale_y),
                    Some(&mut group.skew_x),
                    Some(&mut group.skew_y),
                    Some(&mut group.transform_origin_x),
                    Some(&mut group.transform_origin_y),
                    &mut group.opacity,
                    node_id,
                    property,
                    value,
                );
                apply_animation_property_to_nodes(&mut group.children, node_id, property, value);
            }
            SceneNode::Part(part) => {
                apply_animation_property_to_nodes(&mut part.children, node_id, property, value);
            }
            SceneNode::Repeat(repeat) => {
                apply_animation_property_to_nodes(&mut repeat.children, node_id, property, value);
            }
            SceneNode::Mask(mask) => {
                apply_animation_property_to_nodes(&mut mask.children, node_id, property, value);
            }
            SceneNode::Precompose(precompose) => {
                apply_animation_property_to_nodes(
                    &mut precompose.children,
                    node_id,
                    property,
                    value,
                );
            }
            SceneNode::Layer(layer) => {
                apply_animation_property_to_transform(
                    layer.id.as_deref(),
                    &mut layer.x,
                    &mut layer.y,
                    Some(&mut layer.rotation),
                    &mut layer.scale,
                    Some(&mut layer.scale_x),
                    Some(&mut layer.scale_y),
                    Some(&mut layer.skew_x),
                    Some(&mut layer.skew_y),
                    Some(&mut layer.transform_origin_x),
                    Some(&mut layer.transform_origin_y),
                    &mut layer.opacity,
                    node_id,
                    property,
                    value,
                );
                apply_animation_property_to_nodes(&mut layer.children, node_id, property, value);
            }
            SceneNode::Character(character) => {
                apply_animation_property_to_transform(
                    character.id.as_deref(),
                    &mut character.x,
                    &mut character.y,
                    Some(&mut character.rotation),
                    &mut character.scale,
                    Some(&mut character.scale_x),
                    Some(&mut character.scale_y),
                    Some(&mut character.skew_x),
                    Some(&mut character.skew_y),
                    Some(&mut character.transform_origin_x),
                    Some(&mut character.transform_origin_y),
                    &mut character.opacity,
                    node_id,
                    property,
                    value,
                );
                apply_animation_property_to_nodes(
                    &mut character.children,
                    node_id,
                    property,
                    value,
                );
            }
            SceneNode::Puppet(puppet) => {
                apply_animation_property_to_transform(
                    puppet.id.as_deref(),
                    &mut puppet.x,
                    &mut puppet.y,
                    Some(&mut puppet.rotation),
                    &mut puppet.scale,
                    Some(&mut puppet.scale_x),
                    Some(&mut puppet.scale_y),
                    Some(&mut puppet.skew_x),
                    Some(&mut puppet.skew_y),
                    Some(&mut puppet.transform_origin_x),
                    Some(&mut puppet.transform_origin_y),
                    &mut puppet.opacity,
                    node_id,
                    property,
                    value,
                );
                apply_animation_property_to_nodes(&mut puppet.children, node_id, property, value);
            }
            SceneNode::Pin(pin) => {
                if pin.id.as_deref() == Some(node_id) {
                    match property {
                        "x" => pin.target_x = Some(value.to_string()),
                        "y" => pin.target_y = Some(value.to_string()),
                        _ => {}
                    }
                }
            }
            SceneNode::FaceJaw(face) => {
                apply_animation_property_to_transform(
                    face.id.as_deref(),
                    &mut face.x,
                    &mut face.y,
                    None,
                    &mut face.scale,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    &mut face.opacity,
                    node_id,
                    property,
                    value,
                );
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_animation_property_to_transform(
    id: Option<&str>,
    x: &mut String,
    y: &mut String,
    rotation: Option<&mut String>,
    scale: &mut String,
    scale_x: Option<&mut String>,
    scale_y: Option<&mut String>,
    skew_x: Option<&mut String>,
    skew_y: Option<&mut String>,
    transform_origin_x: Option<&mut String>,
    transform_origin_y: Option<&mut String>,
    opacity: &mut String,
    node_id: &str,
    property: &str,
    value: &str,
) {
    if id != Some(node_id) {
        return;
    }
    match property {
        "x" => *x = value.to_string(),
        "y" => *y = value.to_string(),
        "rotation" => {
            if let Some(rotation) = rotation {
                *rotation = value.to_string();
            }
        }
        "scale" => *scale = value.to_string(),
        "scaleX" => {
            if let Some(scale_x) = scale_x {
                *scale_x = value.to_string();
            }
        }
        "scaleY" => {
            if let Some(scale_y) = scale_y {
                *scale_y = value.to_string();
            }
        }
        "skewX" => {
            if let Some(skew_x) = skew_x {
                *skew_x = value.to_string();
            }
        }
        "skewY" => {
            if let Some(skew_y) = skew_y {
                *skew_y = value.to_string();
            }
        }
        "transformOriginX" => {
            if let Some(transform_origin_x) = transform_origin_x {
                *transform_origin_x = value.to_string();
            }
        }
        "transformOriginY" => {
            if let Some(transform_origin_y) = transform_origin_y {
                *transform_origin_y = value.to_string();
            }
        }
        "opacity" => *opacity = value.to_string(),
        _ => {}
    }
}

fn apply_animation_property_to_text(
    text: &mut TextNode,
    node_id: &str,
    property: &str,
    value: &str,
) {
    apply_animation_property_to_transform(
        text.id.as_deref(),
        &mut text.x,
        &mut text.y,
        Some(&mut text.rotation),
        &mut text.scale,
        Some(&mut text.scale_x),
        Some(&mut text.scale_y),
        Some(&mut text.skew_x),
        Some(&mut text.skew_y),
        Some(&mut text.transform_origin_x),
        Some(&mut text.transform_origin_y),
        &mut text.opacity,
        node_id,
        property,
        value,
    );
}

fn apply_animation_property_to_image(
    image: &mut ImageNode,
    node_id: &str,
    property: &str,
    value: &str,
) {
    apply_animation_property_to_transform(
        image.id.as_deref(),
        &mut image.x,
        &mut image.y,
        None,
        &mut image.scale,
        None,
        None,
        None,
        None,
        None,
        None,
        &mut image.opacity,
        node_id,
        property,
        value,
    );
}

fn apply_animation_property_to_svg(svg: &mut SvgNode, node_id: &str, property: &str, value: &str) {
    apply_animation_property_to_transform(
        svg.id.as_deref(),
        &mut svg.x,
        &mut svg.y,
        None,
        &mut svg.scale,
        None,
        None,
        None,
        None,
        None,
        None,
        &mut svg.opacity,
        node_id,
        property,
        value,
    );
}
use crate::scene::text::{
    TextAnimatorRasterParams, TextRasterizedLayer, apply_text_layer_effects,
    draw_text_buffer_with_animators, prepare_text_layout_for_value, soften_text_alpha_edges,
    stroke_layer_from_alpha, text_bounds, text_layer_effect_spec,
};
use crate::scene::timeline::{
    eval_repeat_count, scene_layer_source_time, scene_sequence_local_time,
};
use crate::world::{
    WorldActor, WorldBackground, WorldBackgroundFit, WorldCamera, WorldCameraControl,
    WorldCameraProjection, WorldFrameRenderer, WorldGraph, WorldMaterial, WorldMaterialStyle,
    WorldNode, WorldPathStyle, WorldPresent, parse_world_graph_script,
};
use cosmic_text::{Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, SwashCache, Weight};
use image::{Rgba, RgbaImage, imageops::FilterType};

mod depth;

pub use crate::scene::error::{MotionLoomSceneRenderError, SceneRenderError};
use depth::{
    SceneDepthContext, scene_depth_track_sort_key, scene_layer_effective_z_depth,
    scene_z_depth_transform,
};

#[allow(dead_code)]
pub async fn render_scene_graph_to_video(
    ffmpeg_bin: &str,
    graph: &GraphScript,
    output_path: &Path,
) -> Result<(), MotionLoomSceneRenderError> {
    render_scene_graph_to_video_with_progress(
        ffmpeg_bin,
        graph,
        output_path,
        SceneRenderProfile::Cpu,
        0,
        |_progress| {},
    )
    .await
}

#[cfg_attr(target_arch = "wasm32", allow(unused_mut, unused_variables))]
pub async fn render_scene_graph_to_video_with_progress<F>(
    ffmpeg_bin: &str,
    graph: &GraphScript,
    output_path: &Path,
    profile: SceneRenderProfile,
    progress_every_frames: u32,
    progress_callback: F,
) -> Result<(), MotionLoomSceneRenderError>
where
    F: FnMut(SceneRenderProgress),
{
    render_scene_graph_to_video_with_progress_and_cancel(
        ffmpeg_bin,
        graph,
        output_path,
        profile,
        progress_every_frames,
        None,
        progress_callback,
    )
    .await
}

#[cfg_attr(target_arch = "wasm32", allow(unused_mut, unused_variables))]
pub async fn render_scene_graph_to_video_with_progress_and_cancel<F>(
    ffmpeg_bin: &str,
    graph: &GraphScript,
    output_path: &Path,
    profile: SceneRenderProfile,
    progress_every_frames: u32,
    cancel: Option<Arc<AtomicBool>>,
    mut progress_callback: F,
) -> Result<(), MotionLoomSceneRenderError>
where
    F: FnMut(SceneRenderProgress),
{
    validate_scene_graph(graph)?;
    if profile.is_png_sequence() {
        return render_scene_graph_to_png_sequence_internal(
            graph,
            output_path,
            profile,
            progress_every_frames,
            cancel,
            progress_callback,
        )
        .await;
    }

    #[cfg(target_arch = "wasm32")]
    {
        Err(MotionLoomSceneRenderError::VideoExportNotAvailable {
            message: "FFmpeg video export is not available in WASM".to_string(),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::export::{FfmpegVideoEncoder, VideoEncoder};

        let (w, h) = graph_output_size(graph);
        let fps = graph.fps.max(1.0);
        let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
        let total_frames = ((duration_sec * fps).round() as u32).max(1);
        let encoder_args = scene_encoder_args(profile);
        let mut renderer = SceneFrameRenderer::new_for_profile(profile).await;
        progress_callback(SceneRenderProgress {
            rendered_frames: 0,
            total_frames,
        });
        if cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
        {
            return Err(MotionLoomSceneRenderError::Cancelled);
        }

        // Render the first frame before starting the encoder. This catches GPU
        // adapter and scene render errors before a long-lived encoder is opened.
        let first_image = renderer.render_frame(graph, 0).await?;
        if cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
        {
            return Err(MotionLoomSceneRenderError::Cancelled);
        }

        let mut encoder =
            FfmpegVideoEncoder::new(ffmpeg_bin, output_path).with_encoder_args(encoder_args);
        encoder.begin(w, h, fps)?;

        for frame in 0..total_frames {
            if cancel
                .as_ref()
                .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
            {
                encoder.abort();
                return Err(MotionLoomSceneRenderError::Cancelled);
            }
            let rendered_image;
            let image = if frame == 0 {
                &first_image
            } else {
                rendered_image = renderer.render_frame(graph, frame).await?;
                &rendered_image
            };
            encoder.push_frame(frame, image.as_raw())?;
            let rendered_frames = frame + 1;
            if rendered_frames == total_frames
                || (progress_every_frames > 0 && rendered_frames % progress_every_frames == 0)
            {
                progress_callback(SceneRenderProgress {
                    rendered_frames,
                    total_frames,
                });
            }
        }
        encoder.finish()?;
        Ok(())
    }
}

pub async fn render_scene_graph_to_png_sequence_with_progress<F>(
    graph: &GraphScript,
    output_dir: &Path,
    progress_every_frames: u32,
    progress_callback: F,
) -> Result<(), MotionLoomSceneRenderError>
where
    F: FnMut(SceneRenderProgress),
{
    render_scene_graph_to_png_sequence_with_progress_and_cancel(
        graph,
        output_dir,
        progress_every_frames,
        None,
        progress_callback,
    )
    .await
}

pub async fn render_scene_graph_to_png_sequence_with_progress_and_cancel<F>(
    graph: &GraphScript,
    output_dir: &Path,
    progress_every_frames: u32,
    cancel: Option<Arc<AtomicBool>>,
    progress_callback: F,
) -> Result<(), MotionLoomSceneRenderError>
where
    F: FnMut(SceneRenderProgress),
{
    validate_scene_graph(graph)?;
    render_scene_graph_to_png_sequence_internal(
        graph,
        output_dir,
        SceneRenderProfile::GpuPngSequence,
        progress_every_frames,
        cancel,
        progress_callback,
    )
    .await
}

async fn render_scene_graph_to_png_sequence_internal<F>(
    graph: &GraphScript,
    output_dir: &Path,
    profile: SceneRenderProfile,
    progress_every_frames: u32,
    cancel: Option<Arc<AtomicBool>>,
    mut progress_callback: F,
) -> Result<(), MotionLoomSceneRenderError>
where
    F: FnMut(SceneRenderProgress),
{
    fs::create_dir_all(output_dir).map_err(|source| {
        MotionLoomSceneRenderError::CreateOutputDir {
            path: output_dir.to_path_buf(),
            source,
        }
    })?;

    let fps = graph.fps.max(1.0);
    let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
    let total_frames = ((duration_sec * fps).round() as u32).max(1);
    let mut renderer = SceneFrameRenderer::new_for_profile(profile).await;
    progress_callback(SceneRenderProgress {
        rendered_frames: 0,
        total_frames,
    });

    for frame in 0..total_frames {
        if cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
        {
            return Err(MotionLoomSceneRenderError::Cancelled);
        }
        let image = renderer.render_frame(graph, frame).await?;
        let path = output_dir.join(format!("frame_{frame:06}.png"));
        image
            .save(&path)
            .map_err(|source| MotionLoomSceneRenderError::SavePngFrame { path, source })?;
        let rendered_frames = frame + 1;
        if rendered_frames == total_frames
            || (progress_every_frames > 0 && rendered_frames % progress_every_frames == 0)
        {
            progress_callback(SceneRenderProgress {
                rendered_frames,
                total_frames,
            });
        }
    }

    Ok(())
}

pub async fn render_scene_graph_frame(
    graph: &GraphScript,
    frame: u32,
    profile: SceneRenderProfile,
) -> Result<RgbaImage, MotionLoomSceneRenderError> {
    validate_scene_graph(graph)?;
    let mut renderer = SceneFrameRenderer::new_for_profile(profile).await;
    renderer.render_frame(graph, frame).await
}

/// Render a scene frame using a caller-provided asset resolver.
///
/// This allows WASM hosts and tests to supply assets from memory instead of
/// relying on the filesystem.
pub async fn render_scene_graph_frame_with_resolver(
    graph: &GraphScript,
    frame: u32,
    profile: SceneRenderProfile,
    asset_resolver: Arc<dyn AssetResolver>,
) -> Result<RgbaImage, MotionLoomSceneRenderError> {
    validate_scene_graph(graph)?;
    let mut renderer =
        SceneFrameRenderer::new_for_profile_with_resolver(profile, asset_resolver).await;
    renderer.render_frame(graph, frame).await
}

/// Render a scene/process graph using caller-provided CPU input images.
///
/// Hosts can use this to run the same MotionLoom process graph used by exports
/// while supplying decoded preview frames from their own media pipeline.
pub async fn render_scene_graph_frame_with_cpu_inputs(
    graph: &GraphScript,
    frame: u32,
    profile: SceneRenderProfile,
    cpu_inputs: &HashMap<String, RgbaImage>,
) -> Result<RgbaImage, MotionLoomSceneRenderError> {
    validate_scene_graph(graph)?;
    let mut renderer = SceneFrameRenderer::new_for_profile(profile).await;
    renderer
        .render_frame_with_cpu_inputs(graph, frame, Some(cpu_inputs))
        .await
}

pub struct SceneRenderer {
    inner: SceneFrameRenderer,
}

/// Native CPU work for the most recently submitted scene frame.
///
/// Expression covers animation/action evaluation and dependency invalidation.
/// Traversal is the remaining scene compile/traversal work. Upload and encode
/// are recorded by the GPU compositor, while wait is frame-latency backpressure.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SceneCpuFrameProfile {
    pub expression_ms: f64,
    pub traversal_ms: f64,
    pub upload_ms: f64,
    pub encode_ms: f64,
    pub wait_ms: f64,
}

/// Timings and actual GPU workload generated by one pure-vector scene frame.
///
/// `flatten_ms` includes expression/morph evaluation, Path parsing, curve
/// flattening, and fill/stroke tessellation. `encode_ms` includes tile binning,
/// upload packing, GPU buffer creation, and command encoding. `gpu_ms` waits for
/// the submitted GPU work to complete. Headless benchmarks do not own a window
/// surface, so `present_ms` is `None` rather than a fabricated value.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct VectorFrameBenchmark {
    pub flatten_ms: f64,
    pub encode_ms: f64,
    pub gpu_ms: f64,
    pub present_ms: Option<f64>,
    pub primitive_count: u32,
    pub upload_bytes: usize,
    pub path_cache_hits: usize,
    pub path_cache_misses: usize,
}

impl SceneRenderer {
    pub async fn new(profile: SceneRenderProfile) -> Result<Self, SceneRenderError> {
        Ok(Self {
            inner: SceneFrameRenderer::new_for_profile(profile).await,
        })
    }

    pub async fn with_resolver(
        profile: SceneRenderProfile,
        asset_resolver: Arc<dyn AssetResolver>,
    ) -> Result<Self, SceneRenderError> {
        Ok(Self {
            inner: SceneFrameRenderer::new_for_profile_with_resolver(profile, asset_resolver).await,
        })
    }

    /// Create a renderer that reuses an externally-owned wgpu device and queue.
    ///
    /// This is intended for host applications that already manage a wgpu context
    /// and want to consume motionloom output as a GPU texture without a separate
    /// device or CPU readback.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn new_with_device(
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
        profile: SceneRenderProfile,
    ) -> Result<Self, SceneRenderError> {
        Ok(Self {
            inner: SceneFrameRenderer::new_with_resolver_and_device(
                profile,
                Arc::new(PathAssetResolver),
                device,
                queue,
            )
            .await,
        })
    }

    /// Create a renderer that reuses an externally-owned wgpu device and queue
    /// with a custom asset resolver.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn with_resolver_and_device(
        profile: SceneRenderProfile,
        asset_resolver: Arc<dyn AssetResolver>,
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
    ) -> Result<Self, SceneRenderError> {
        Ok(Self {
            inner: SceneFrameRenderer::new_with_resolver_and_device(
                profile,
                asset_resolver,
                device,
                queue,
            )
            .await,
        })
    }

    pub async fn render_frame(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<RgbaImage, SceneRenderError> {
        validate_scene_graph(graph)?;
        self.inner.render_frame(graph, frame).await
    }

    pub async fn render_frame_with_cpu_inputs(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        cpu_inputs: &HashMap<String, RgbaImage>,
    ) -> Result<RgbaImage, SceneRenderError> {
        validate_scene_graph(graph)?;
        self.inner
            .render_frame_with_cpu_inputs(graph, frame, Some(cpu_inputs))
            .await
    }

    pub fn load_font_data(&mut self, bytes: Vec<u8>) {
        self.inner.font_system.db_mut().load_font_data(bytes);
    }

    /// Invalidate retained scene data after an editor mutates parsed Scene nodes.
    ///
    /// Runtime editor APIs intentionally keep the parsed graph and Path geometry
    /// cache alive. Retained scene entries, however, contain compiled ancestor
    /// transforms and must be rebuilt before the next frame.
    pub fn invalidate_runtime_scene_transforms(&mut self) {
        self.inner.retained_gpu_shape_scenes.clear();
        self.inner.retained_gpu_transform_scenes.clear();
        self.inner.gpu_text_raster_cache.clear();
        self.inner.gpu_layer3d_texture_cache.clear();
    }

    /// Render a scene frame to a GPU texture without CPU readback.
    ///
    /// The returned `SceneGpuTexture` wraps an `Arc<wgpu::Texture>` that is safe
    /// to keep alive after the renderer is dropped. The texture's dimensions
    /// match the graph's output size.
    pub async fn render_frame_to_wgpu_texture(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<SceneGpuTexture, SceneRenderError> {
        validate_scene_graph(graph)?;
        self.inner.render_frame_to_wgpu_texture(graph, frame).await
    }

    /// Last completed native GPU frame duration measured by timestamp queries.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn last_gpu_frame_ms(&self) -> Option<f64> {
        self.inner
            .gpu_compositor
            .as_ref()
            .and_then(WgpuSceneCompositor::last_gpu_frame_ms)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn last_cpu_frame_profile(&self) -> SceneCpuFrameProfile {
        self.inner.last_cpu_frame_profile
    }

    /// Whether the active native adapter exposes WebGPU timestamp queries.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_timestamp_supported(&self) -> bool {
        self.inner
            .gpu_compositor
            .as_ref()
            .is_some_and(WgpuSceneCompositor::gpu_timestamp_supported)
    }

    /// Benchmark a pure Scene vector frame through MotionLoom's production
    /// batched WGPU shape pipeline without CPU readback.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn benchmark_vector_frame(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<VectorFrameBenchmark, SceneRenderError> {
        validate_scene_graph(graph)?;
        self.inner.benchmark_vector_frame(graph, frame).await
    }

    /// Render through the strict GPU texture path and read back that exact texture.
    /// Intended for diagnostics, thumbnails, and tests that must not silently use CPU rendering.
    pub async fn render_frame_gpu_readback(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<RgbaImage, SceneRenderError> {
        validate_scene_graph(graph)?;
        let texture = self
            .inner
            .render_frame_to_wgpu_texture(graph, frame)
            .await?;
        self.inner
            .gpu_compositor
            .as_mut()
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: "GPU compositor was not initialized".to_string(),
            })?
            .readback_texture_rgba_async(&texture.texture)
            .await
    }

    /// Render a scene frame into a caller-owned wgpu texture.
    ///
    /// The target texture must belong to the same `wgpu::Device` used to create
    /// this renderer, must be `Rgba8Unorm`, and must include `COPY_DST` usage.
    /// This keeps the frame on GPU and lets hosts reuse their own preview
    /// texture across frames.
    pub async fn render_frame_to_wgpu_target_texture(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        target: &wgpu::Texture,
        target_width: u32,
        target_height: u32,
    ) -> Result<(), SceneRenderError> {
        validate_scene_graph(graph)?;
        self.inner
            .render_frame_to_wgpu_target_texture(graph, frame, target, target_width, target_height)
            .await
    }

    /// Return the renderer-level pick id at a graph-space pixel.
    ///
    /// The caller supplies a stable `(node_id, pick_id)` table; `0` is reserved
    /// for empty pixels. The renderer draws a hidden ID pass through the same
    /// scene GPU command pipeline used for preview rendering, then reads one
    /// pixel back from GPU memory.
    pub async fn pick_id_at_wgpu_position(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        x: u32,
        y: u32,
        pick_ids: &[(String, u32)],
    ) -> Result<Option<u32>, SceneRenderError> {
        validate_scene_graph(graph)?;
        self.inner
            .pick_id_at_wgpu_position(graph, frame, x, y, pick_ids)
            .await
    }

    /// Render a frame to the best preview surface requested by the host.
    ///
    /// `Auto` tries future platform-native surfaces first, then wgpu texture,
    /// then CPU BGRA. This keeps the public API stable while platform bridges
    /// are added behind the same abstraction.
    pub async fn render_frame_to_preview_surface(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        options: ScenePreviewSurfaceOptions,
    ) -> Result<ScenePreviewSurface, SceneRenderError> {
        validate_scene_graph(graph)?;
        self.inner
            .render_frame_to_preview_surface(graph, frame, options)
            .await
    }

    /// Render a GPU-native scene frame directly into a browser canvas.
    ///
    /// This WASM-only path presents the compositor texture to the canvas
    /// surface and avoids CPU readback. It is intentionally strict for now:
    /// unsupported scene graphs return a GPU render error instead of falling
    /// back silently.
    #[cfg(target_arch = "wasm32")]
    pub async fn render_frame_to_canvas(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(), SceneRenderError> {
        validate_scene_graph(graph)?;
        self.inner
            .render_frame_to_canvas(graph, frame, canvas)
            .await
    }

    /// Draw a solid WebGPU color into a browser canvas for surface debugging.
    #[cfg(target_arch = "wasm32")]
    pub async fn debug_solid_to_canvas(
        &mut self,
        canvas: web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
        color: [f64; 4],
    ) -> Result<(), SceneRenderError> {
        self.inner
            .debug_solid_to_canvas(canvas, width, height, color)
            .await
    }

    /// Upload a solid texture and present it to a browser canvas for debugging.
    #[cfg(target_arch = "wasm32")]
    pub async fn debug_uploaded_texture_to_canvas(
        &mut self,
        canvas: web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
        color: [u8; 4],
    ) -> Result<(), SceneRenderError> {
        self.inner
            .debug_uploaded_texture_to_canvas(canvas, width, height, color)
            .await
    }

    /// Render an empty scene texture with a white clear color and present it.
    #[cfg(target_arch = "wasm32")]
    pub async fn debug_empty_scene_texture_to_canvas(
        &mut self,
        canvas: web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
    ) -> Result<(), SceneRenderError> {
        self.inner
            .debug_empty_scene_texture_to_canvas(canvas, width, height)
            .await
    }
}

fn validate_scene_graph(graph: &GraphScript) -> Result<(), MotionLoomSceneRenderError> {
    if !graph.has_scene_nodes() {
        return Err(MotionLoomSceneRenderError::EmptyScene);
    }
    validate_scene_gradient_refs(graph)?;
    Ok(())
}

fn validate_scene_gradient_refs(graph: &GraphScript) -> Result<(), MotionLoomSceneRenderError> {
    let mut gradient_defs = HashMap::new();
    collect_graph_gradient_defs(graph, &mut gradient_defs);

    let mut refs = Vec::new();
    collect_scene_gradient_refs(&graph.scene_nodes, &mut refs);
    for scene in &graph.scenes {
        collect_scene_gradient_refs(&scene.children, &mut refs);
    }

    let mut seen = HashSet::new();
    for (paint, id) in refs {
        if !seen.insert((paint.clone(), id.clone())) {
            continue;
        }
        if !gradient_defs.contains_key(&id) {
            return Err(MotionLoomSceneRenderError::InvalidPaint {
                value: paint,
                message: format!("gradient reference not found: {id}"),
            });
        }
    }
    Ok(())
}

fn collect_scene_gradient_refs(nodes: &[SceneNode], out: &mut Vec<(String, String)>) {
    for node in nodes {
        match node {
            SceneNode::Defs(defs) => collect_defs_gradient_refs(defs, out),
            SceneNode::Timeline(timeline) => collect_scene_gradient_refs(&timeline.children, out),
            SceneNode::Track(track) => collect_scene_gradient_refs(&track.children, out),
            SceneNode::Sequence(sequence) => collect_scene_gradient_refs(&sequence.children, out),
            SceneNode::Chain(chain) => collect_scene_gradient_refs(&chain.children, out),
            SceneNode::Text(text) => collect_text_gradient_refs(text, out),
            SceneNode::Rect(rect) => {
                collect_paint_gradient_ref(&rect.color, out);
                collect_optional_paint_gradient_ref(rect.stroke.as_deref(), out);
            }
            SceneNode::Circle(circle) => {
                collect_paint_gradient_ref(&circle.color, out);
                collect_optional_paint_gradient_ref(circle.stroke.as_deref(), out);
            }
            SceneNode::Ellipse(ellipse) => {
                collect_paint_gradient_ref(&ellipse.color, out);
                collect_optional_paint_gradient_ref(ellipse.stroke.as_deref(), out);
            }
            SceneNode::Line(line) => collect_paint_gradient_ref(&line.color, out),
            SceneNode::Polyline(polyline) => collect_paint_gradient_ref(&polyline.stroke, out),
            SceneNode::Path(path) => {
                collect_paint_gradient_ref(&path.stroke, out);
                collect_optional_paint_gradient_ref(path.fill.as_deref(), out);
            }
            SceneNode::FaceJaw(face_jaw) => {
                collect_paint_gradient_ref(&face_jaw.stroke, out);
                collect_optional_paint_gradient_ref(face_jaw.fill.as_deref(), out);
            }
            SceneNode::Shadow(shadow) => collect_paint_gradient_ref(&shadow.color, out),
            SceneNode::Group(group) => collect_scene_gradient_refs(&group.children, out),
            SceneNode::Puppet(puppet) => collect_scene_gradient_refs(&puppet.children, out),
            SceneNode::Part(part) => collect_scene_gradient_refs(&part.children, out),
            SceneNode::Repeat(repeat) => collect_scene_gradient_refs(&repeat.children, out),
            SceneNode::Mask(mask) => collect_scene_gradient_refs(&mask.children, out),
            SceneNode::Precompose(precompose) => {
                collect_scene_gradient_refs(&precompose.children, out)
            }
            SceneNode::Layer(layer) => collect_scene_gradient_refs(&layer.children, out),
            SceneNode::Camera(camera) => collect_scene_gradient_refs(&camera.children, out),
            SceneNode::Character(character) => {
                collect_scene_gradient_refs(&character.children, out)
            }
            SceneNode::Palette(_)
            | SceneNode::PixelGrid(_)
            | SceneNode::Image(_)
            | SceneNode::Svg(_)
            | SceneNode::Use(_)
            | SceneNode::Pin(_)
            | SceneNode::LimbEnvelope(_)
            | SceneNode::LimbRegion(_)
            | SceneNode::MeshTopology(_)
            | SceneNode::Vertex(_)
            | SceneNode::Triangle(_)
            | SceneNode::Edge(_)
            | SceneNode::Region(_)
            | SceneNode::Simulation(_) => {}
        }
    }
}

fn collect_defs_gradient_refs(defs: &DefsNode, out: &mut Vec<(String, String)>) {
    for brush in &defs.brushes {
        collect_optional_paint_gradient_ref(brush.stroke.as_deref(), out);
        collect_optional_paint_gradient_ref(brush.fill.as_deref(), out);
    }
    for mask in &defs.masks {
        collect_scene_gradient_refs(&mask.children, out);
    }
    for precompose in &defs.precomposes {
        collect_scene_gradient_refs(&precompose.children, out);
    }
    for component in &defs.components {
        collect_scene_gradient_refs(&component.children, out);
    }
}

fn collect_text_gradient_refs(text: &TextNode, out: &mut Vec<(String, String)>) {
    collect_paint_gradient_ref(&text.color, out);
    collect_optional_paint_gradient_ref(text.box_color.as_deref(), out);
    collect_optional_paint_gradient_ref(text.stroke.as_deref(), out);
    for animator in &text.animators {
        if let Some(style) = animator.style.as_ref() {
            collect_optional_paint_gradient_ref(style.color.as_deref(), out);
            collect_optional_paint_gradient_ref(style.stroke.as_deref(), out);
            collect_optional_paint_gradient_ref(style.shadow_color.as_deref(), out);
        }
        for effect in &animator.effects {
            match effect {
                crate::scene::text::TextEffectNode::Glow(glow) => {
                    collect_optional_paint_gradient_ref(glow.color.as_deref(), out);
                }
            }
        }
    }
}

fn collect_optional_paint_gradient_ref(value: Option<&str>, out: &mut Vec<(String, String)>) {
    if let Some(value) = value {
        collect_paint_gradient_ref(value, out);
    }
}

fn collect_paint_gradient_ref(value: &str, out: &mut Vec<(String, String)>) {
    if let Some(id) = gradient_ref_id(value) {
        out.push((value.to_string(), id.to_string()));
    }
}

fn motionloom_font_family(family: &str) -> Option<Family<'_>> {
    let trimmed = family.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "serif" => Some(Family::Serif),
        "sans" | "sans-serif" | "sans serif" | "system" => Some(Family::SansSerif),
        "mono" | "monospace" | "monospace-serif" => Some(Family::Monospace),
        "cursive" => Some(Family::Cursive),
        "fantasy" => Some(Family::Fantasy),
        _ => Some(Family::Name(trimmed)),
    }
}

#[cfg(target_arch = "wasm32")]
fn motionloom_wasm_loaded_font_family(font_family: Option<&str>, text: &str) -> Family<'static> {
    if text.chars().any(|ch| !ch.is_ascii()) {
        return Family::Name("Noto Sans CJK TC");
    }
    let _ = font_family;
    Family::Name("Noto Sans CJK TC")
}

struct SceneFrameRenderer {
    profile: SceneRenderProfile,
    asset_resolver: Arc<dyn AssetResolver>,
    font_system: FontSystem,
    swash_cache: SwashCache,
    image_cache: HashMap<String, RgbaImage>,
    svg_cache: HashMap<String, RgbaImage>,
    path_cache: HashMap<String, Vec<Vec<Point2>>>,
    polyline_cache: HashMap<String, Vec<Point2>>,
    vector_overlay_cache: HashMap<String, CachedVectorOverlay>,
    gpu_path_geometry_cache: HashMap<String, CachedGpuPathGeometry>,
    retained_gpu_shape_scenes: HashMap<RetainedGpuShapeSceneKey, CachedRetainedGpuShapeScene>,
    retained_gpu_transform_scenes:
        HashMap<RetainedGpuShapeSceneKey, CachedRetainedGpuTransformScene>,
    gpu_text_raster_cache: HashMap<u64, CachedGpuTextRaster>,
    gpu_layer3d_texture_cache: HashMap<u64, CachedGpuLayer3dTexture>,
    prepared_graph_signature: u64,
    prepared_graph_fps: f32,
    prepared_graph_duration_ms: u64,
    gpu_path_cache_hits: usize,
    gpu_path_cache_misses: usize,
    gradient_defs: HashMap<String, GradientDef>,
    palette_defs: HashMap<String, PaletteNode>,
    font_defs: HashMap<String, FontDef>,
    texture_defs: HashMap<String, TextureDef>,
    noise_defs: HashMap<String, NoiseDef>,
    material_defs: HashMap<String, MaterialDef>,
    filter_defs: HashMap<String, FilterDef>,
    scene_node_defs: HashMap<String, SceneNode>,
    scene_follow_nodes: Vec<SceneNode>,
    scene_components: HashMap<String, Vec<SceneNode>>,
    scene_precompose_defs: HashMap<String, PrecomposeNode>,
    scene_precomposes: HashMap<String, RgbaImage>,
    scene_masks: HashMap<String, MaskNode>,
    model_asset_sources: HashMap<String, String>,
    image_asset_sources: HashMap<String, String>,
    gpu_pick_ids: HashMap<String, u32>,
    world_renderer: WorldFrameRenderer,
    gpu_compositor: Option<WgpuSceneCompositor>,
    #[cfg(not(target_arch = "wasm32"))]
    last_cpu_frame_profile: SceneCpuFrameProfile,
    /// Optional externally-owned wgpu device/queue for shared-context rendering.
    #[cfg(not(target_arch = "wasm32"))]
    external_device_queue: Option<(Arc<wgpu::Device>, wgpu::Queue)>,
    /// Cached D3D11 device/context used to allocate Windows shared preview surfaces.
    #[cfg(target_os = "windows")]
    windows_d3d11: Option<WindowsD3D11Context>,
}

const VECTOR_OVERLAY_CACHE_LIMIT: usize = 64;
const GPU_PATH_GEOMETRY_CACHE_LIMIT: usize = 100_000;
const RETAINED_GPU_SHAPE_SCENE_CACHE_LIMIT: usize = 8;
const RETAINED_GPU_TRANSFORM_SCENE_CACHE_LIMIT: usize = 64;
const GPU_TEXT_RASTER_CACHE_LIMIT: usize = 256;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RetainedGpuShapeSceneKey {
    graph_signature: u64,
    scope: u64,
    nodes_len: usize,
    canvas_size: (u32, u32),
    transform_bits: [u32; 6],
}

#[derive(Clone)]
struct CachedRetainedGpuShapeScene {
    primitives: Arc<Vec<GpuScenePrimitive>>,
    path_count: usize,
}

#[derive(Clone)]
struct CachedGpuTextRaster {
    texture: GpuSceneNativeTexture,
    local_transform: Affine2,
}

#[derive(Clone)]
struct CachedGpuLayer3dTexture {
    texture: GpuSceneNativeTexture,
    offset_x: u32,
    offset_y: u32,
}

#[derive(Clone)]
enum RetainedTransformAncestor {
    Transform(RetainedCompiledTransform),
    Repeat(RetainedCompiledRepeat, usize),
}

#[derive(Clone)]
struct RetainedCompiledRepeat {
    x: CompiledTimeExpr,
    y: CompiledTimeExpr,
    rotation: CompiledTimeExpr,
    scale: CompiledTimeExpr,
    opacity: CompiledTimeExpr,
    x_step: CompiledTimeExpr,
    y_step: CompiledTimeExpr,
    rotation_step: CompiledTimeExpr,
    scale_step: CompiledTimeExpr,
    opacity_step: CompiledTimeExpr,
}

impl RetainedCompiledRepeat {
    fn compile(node: &RepeatNode) -> Result<Self, MotionLoomSceneRenderError> {
        let compile = |value: &str| {
            CompiledTimeExpr::compile(value).map_err(|message| {
                MotionLoomSceneRenderError::InvalidExpression {
                    expr: value.to_string(),
                    message,
                }
            })
        };
        Ok(Self {
            x: compile(&node.x)?,
            y: compile(&node.y)?,
            rotation: compile(&node.rotation)?,
            scale: compile(&node.scale)?,
            opacity: compile(&node.opacity)?,
            x_step: compile(&node.x_step)?,
            y_step: compile(&node.y_step)?,
            rotation_step: compile(&node.rotation_step)?,
            scale_step: compile(&node.scale_step)?,
            opacity_step: compile(&node.opacity_step)?,
        })
    }

    fn evaluate(
        &self,
        index: usize,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(Affine2, f32), MotionLoomSceneRenderError> {
        let eval = |expr: &CompiledTimeExpr| {
            expr.evaluate(time_norm, time_sec).map_err(|message| {
                MotionLoomSceneRenderError::InvalidExpression {
                    expr: "compiled Repeat".to_string(),
                    message,
                }
            })
        };
        let i = index as f32;
        let scale = (eval(&self.scale)? + eval(&self.scale_step)? * i).clamp(0.001, 64.0);
        Ok((
            Affine2::translate(
                eval(&self.x)? + eval(&self.x_step)? * i,
                eval(&self.y)? + eval(&self.y_step)? * i,
            )
            .mul(Affine2::rotate_deg(
                eval(&self.rotation)? + eval(&self.rotation_step)? * i,
            ))
            .mul(Affine2::scale(scale)),
            (eval(&self.opacity)? + eval(&self.opacity_step)? * i).clamp(0.0, 1.0),
        ))
    }
}

#[derive(Clone)]
struct RetainedCompiledTransform {
    x: CompiledTimeExpr,
    y: CompiledTimeExpr,
    rotation: CompiledTimeExpr,
    scale: CompiledTimeExpr,
    scale_x: CompiledTimeExpr,
    scale_y: CompiledTimeExpr,
    skew_x: CompiledTimeExpr,
    skew_y: CompiledTimeExpr,
    origin_x: CompiledTimeExpr,
    origin_y: CompiledTimeExpr,
    opacity: CompiledTimeExpr,
}

impl RetainedCompiledTransform {
    fn compile(values: [&str; 11]) -> Result<Self, MotionLoomSceneRenderError> {
        let [
            x,
            y,
            rotation,
            scale,
            scale_x,
            scale_y,
            skew_x,
            skew_y,
            origin_x,
            origin_y,
            opacity,
        ] = values;
        let compile = |value: &str| {
            CompiledTimeExpr::compile(value).map_err(|message| {
                MotionLoomSceneRenderError::InvalidExpression {
                    expr: value.to_string(),
                    message,
                }
            })
        };
        Ok(Self {
            x: compile(x)?,
            y: compile(y)?,
            rotation: compile(rotation)?,
            scale: compile(scale)?,
            scale_x: compile(scale_x)?,
            scale_y: compile(scale_y)?,
            skew_x: compile(skew_x)?,
            skew_y: compile(skew_y)?,
            origin_x: compile(origin_x)?,
            origin_y: compile(origin_y)?,
            opacity: compile(opacity)?,
        })
    }

    fn evaluate(
        &self,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(Affine2, f32), MotionLoomSceneRenderError> {
        let eval = |expr: &CompiledTimeExpr| {
            expr.evaluate(time_norm, time_sec).map_err(|message| {
                MotionLoomSceneRenderError::InvalidExpression {
                    expr: "compiled transform".to_string(),
                    message,
                }
            })
        };
        let scale = eval(&self.scale)?.clamp(0.001, 64.0);
        let signed_scale = |value: f32| {
            if value.abs() < 0.001 {
                if value.is_sign_negative() {
                    -0.001
                } else {
                    0.001
                }
            } else {
                value
            }
        };
        let scale_x = signed_scale(scale * eval(&self.scale_x)?);
        let scale_y = signed_scale(scale * eval(&self.scale_y)?);
        let origin_x = eval(&self.origin_x)?;
        let origin_y = eval(&self.origin_y)?;
        let transform = Affine2::translate(eval(&self.x)?, eval(&self.y)?)
            .mul(Affine2::translate(origin_x, origin_y))
            .mul(Affine2::rotate_deg(eval(&self.rotation)?))
            .mul(Affine2::skew_deg(eval(&self.skew_x)?, eval(&self.skew_y)?))
            .mul(Affine2::scale_xy(scale_x, scale_y))
            .mul(Affine2::translate(-origin_x, -origin_y));
        Ok((transform, eval(&self.opacity)?.clamp(0.0, 1.0)))
    }
}

#[derive(Clone)]
struct RetainedTransformPrimitiveEntry {
    primitive_range: std::ops::Range<usize>,
    ancestors: Vec<RetainedTransformAncestor>,
    source: SceneNode,
    dependency: RetainedExpressionDependency,
    last_dependency_token: u64,
}

#[derive(Clone)]
struct RetainedTransformTextEntry {
    node: TextNode,
    ancestors: Vec<RetainedTransformAncestor>,
}

#[derive(Clone, Copy)]
enum RetainedExpressionDependency {
    Static,
    Frame,
    QuantizedHz(f32),
}

impl RetainedExpressionDependency {
    fn for_node(node: &SceneNode) -> Self {
        let debug = format!("{node:?}").to_ascii_lowercase();
        if !debug.contains("$time") && !debug.contains("$frame") && !debug.contains("curve(") {
            return Self::Static;
        }
        if let Some(hz) = quantized_time_hz(&debug) {
            return Self::QuantizedHz(hz);
        }
        Self::Frame
    }

    fn token(self, time_sec: f32) -> u64 {
        match self {
            Self::Static => 0,
            Self::Frame => u64::MAX,
            Self::QuantizedHz(hz) => (time_sec.max(0.0) * hz).floor() as u64,
        }
    }
}

fn quantized_time_hz(value: &str) -> Option<f32> {
    let compact = value
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let marker = "floor($time.sec*";
    let start = compact.find(marker)? + marker.len();
    let tail = &compact[start..];
    let end = tail.find(')')?;
    tail[..end].trim().parse::<f32>().ok().filter(|v| *v > 0.0)
}

#[derive(Clone)]
struct CachedRetainedGpuTransformScene {
    primitives: Arc<Vec<GpuScenePrimitive>>,
    base_transforms: Vec<Affine2>,
    base_opacities: Vec<f32>,
    entries: Vec<RetainedTransformPrimitiveEntry>,
    text_entries: Vec<RetainedTransformTextEntry>,
    path_count: usize,
}

struct RetainedGpuTransformFrame {
    primitives: Arc<Vec<GpuScenePrimitive>>,
    text_requests: Vec<GpuSceneTextRequest>,
}

#[derive(Default)]
struct RetainedCompileOutput {
    primitives: Vec<GpuScenePrimitive>,
    entries: Vec<RetainedTransformPrimitiveEntry>,
    text_entries: Vec<RetainedTransformTextEntry>,
}

impl RetainedGpuShapeSceneKey {
    fn new(
        graph_signature: u64,
        nodes: &[SceneNode],
        transform: Affine2,
        canvas_size: (u32, u32),
    ) -> Self {
        Self::new_scoped(graph_signature, 0, nodes, transform, canvas_size)
    }

    fn new_scoped(
        graph_signature: u64,
        scope: u64,
        nodes: &[SceneNode],
        transform: Affine2,
        canvas_size: (u32, u32),
    ) -> Self {
        Self {
            graph_signature,
            scope,
            nodes_len: nodes.len(),
            canvas_size,
            transform_bits: [
                transform.m00.to_bits(),
                transform.m01.to_bits(),
                transform.m02.to_bits(),
                transform.m10.to_bits(),
                transform.m11.to_bits(),
                transform.m12.to_bits(),
            ],
        }
    }
}

#[derive(Clone)]
struct CachedGpuPathGeometry {
    signature: u64,
    primitives: Arc<Vec<GpuScenePrimitive>>,
}

#[derive(Clone)]
struct CachedVectorOverlay {
    image: RgbaImage,
    offset_x: f32,
    offset_y: f32,
    gpu_texture: Option<GpuSceneNativeTexture>,
}

impl CachedVectorOverlay {
    fn texture_layer(&self, transform: Affine2) -> GpuSceneTextureLayer {
        GpuSceneTextureLayer {
            source: self
                .gpu_texture
                .clone()
                .map(GpuSceneTextureSource::Gpu)
                .unwrap_or_else(|| GpuSceneTextureSource::Cpu(self.image.clone())),
            transform: transform.mul(Affine2::translate(self.offset_x, self.offset_y)),
            projected_quad: None,
            opacity: 1.0,
            blend: SceneBlendMode::Normal,
            pick_id: 0,
            matte: None,
            primitive_index: usize::MAX,
        }
    }
}

#[derive(Clone, Copy)]
struct SceneLayerDrawParams {
    source_size: (u32, u32),
    base_transform: Affine2,
    clip: Option<CameraRect>,
    time_norm: f32,
    time_sec: f32,
    inherited_opacity: f32,
}

fn crop_layer_alpha_bounds(layer: &RgbaImage) -> Option<(RgbaImage, u32, u32)> {
    let mut min_x = layer.width();
    let mut min_y = layer.height();
    let mut max_x = 0_u32;
    let mut max_y = 0_u32;
    let mut found = false;
    for (x, y, pixel) in layer.enumerate_pixels() {
        if pixel[3] == 0 {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    if !found {
        return None;
    }

    let width = max_x - min_x + 1;
    let height = max_y - min_y + 1;
    let mut cropped = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));
    for y in 0..height {
        for x in 0..width {
            cropped.put_pixel(x, y, *layer.get_pixel(min_x + x, min_y + y));
        }
    }
    Some((cropped, min_x, min_y))
}

fn vector_overlay_nodes_are_static(nodes: &[SceneNode]) -> bool {
    let debug = format!("{nodes:?}").to_ascii_lowercase();
    !debug.contains("curve(")
        && !debug.contains("morph(")
        && !debug.contains("$time")
        && !debug.contains("$frame")
        && !debug.contains("random(")
        && !debug.contains("noise(")
}

fn gpu_layer3d_texture_cache_key(
    graph_signature: u64,
    layer: &SceneLayerNode,
    canvas_size: (u32, u32),
) -> Option<u64> {
    if layer.source.is_some()
        || layer.effect.is_some()
        || layer.mask.is_some()
        || layer.mask_from.is_some()
        || layer.matte.is_some()
        || layer.matte_from.is_some()
        || layer.children.is_empty()
        || !vector_overlay_nodes_are_static(&layer.children)
    {
        return None;
    }

    let mut hasher = DefaultHasher::new();
    graph_signature.hash(&mut hasher);
    canvas_size.hash(&mut hasher);
    layer.id.hash(&mut hasher);
    format!("{:?}", layer.children).hash(&mut hasher);
    Some(hasher.finish())
}

fn text_raster_content_is_static(text: &TextNode) -> bool {
    if !text.animators.is_empty() || text.visible_chars.is_some() {
        return false;
    }
    let mut content = text.clone();
    // Opacity is a compositing property. Keeping it out of the raster
    // signature lets a 4x glyph texture survive a fade animation.
    content.opacity = "1".to_string();
    let debug = format!("{content:?}").to_ascii_lowercase();
    !debug.contains("curve(")
        && !debug.contains("morph(")
        && !debug.contains("$time")
        && !debug.contains("$frame")
        && !debug.contains("random(")
        && !debug.contains("noise(")
}

fn retained_path_geometry_and_style_are_static(path: &PathNode) -> bool {
    if !gpu_path_style_is_cacheable(path) || path.material.is_some() {
        return false;
    }
    let debug = format!("{path:?}").to_ascii_lowercase();
    !debug.contains("curve(")
        && !debug.contains("morph(")
        && !debug.contains("$time")
        && !debug.contains("$frame")
        && !debug.contains("random(")
        && !debug.contains("noise(")
}

fn gpu_text_raster_cache_key(
    graph_signature: u64,
    text: &TextNode,
    canvas_size: (u32, u32),
) -> u64 {
    let mut content = text.clone();
    content.opacity = "1".to_string();
    let mut hasher = DefaultHasher::new();
    graph_signature.hash(&mut hasher);
    canvas_size.hash(&mut hasher);
    format!("{content:?}").hash(&mut hasher);
    hasher.finish()
}

fn vector_overlay_cache_key(
    nodes: &[SceneNode],
    raster_transform: Affine2,
    inherited_opacity: f32,
    canvas_size: (u32, u32),
) -> String {
    let mut hasher = DefaultHasher::new();
    canvas_size.hash(&mut hasher);
    for value in [
        raster_transform.m00,
        raster_transform.m01,
        raster_transform.m02,
        raster_transform.m10,
        raster_transform.m11,
        raster_transform.m12,
        inherited_opacity,
    ] {
        value.to_bits().hash(&mut hasher);
    }
    format!("{nodes:?}").hash(&mut hasher);
    format!("vector-overlay:{:016x}", hasher.finish())
}

fn cache_vector_overlay(
    cache: &mut HashMap<String, CachedVectorOverlay>,
    key: String,
    value: CachedVectorOverlay,
) {
    if cache.len() >= VECTOR_OVERLAY_CACHE_LIMIT && !cache.contains_key(&key) {
        cache.clear();
    }
    cache.insert(key, value);
}

fn vector_overlay_raster_transform(transform: Affine2) -> Affine2 {
    Affine2 {
        m02: 0.0,
        m12: 0.0,
        ..transform
    }
}

fn vector_overlay_present_transform(transform: Affine2) -> Affine2 {
    Affine2::translate(transform.m02, transform.m12)
}

fn path_has_default_local_transform(path: &PathNode) -> bool {
    path.x.trim() == "0"
        && path.y.trim() == "0"
        && path.rotation.trim() == "0"
        && path.scale.trim() == "1"
        && path.scale_x.trim() == "1"
        && path.scale_y.trim() == "1"
        && path.skew_x.trim() == "0"
        && path.skew_y.trim() == "0"
        && path.transform_origin_x.trim() == "0"
        && path.transform_origin_y.trim() == "0"
}

fn scene_layer_3d_projected_quad(
    layer: &SceneLayerNode,
    base_transform: Affine2,
    offset: (f32, f32),
    size: (f32, f32),
    time_norm: f32,
    time_sec: f32,
) -> Result<[(f32, f32, f32); 4], MotionLoomSceneRenderError> {
    let x = eval_scene_number(&layer.x, time_norm, time_sec)?;
    let y = eval_scene_number(&layer.y, time_norm, time_sec)?;
    let z = eval_scene_number(&layer.z, time_norm, time_sec)?;
    let rotation_x = eval_scene_number(&layer.rotation_x, time_norm, time_sec)?.to_radians();
    let rotation_y = eval_scene_number(&layer.rotation_y, time_norm, time_sec)?.to_radians();
    let rotation_z = eval_scene_number(&layer.rotation, time_norm, time_sec)?.to_radians();
    let scale = eval_scene_number(&layer.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
    let scale_x =
        clamp_nonzero_signed_scale(scale * eval_scene_number(&layer.scale_x, time_norm, time_sec)?);
    let scale_y =
        clamp_nonzero_signed_scale(scale * eval_scene_number(&layer.scale_y, time_norm, time_sec)?);
    let origin_x = eval_scene_number(&layer.transform_origin_x, time_norm, time_sec)?;
    let origin_y = eval_scene_number(&layer.transform_origin_y, time_norm, time_sec)?;
    let perspective = eval_scene_number(&layer.perspective, time_norm, time_sec)?
        .abs()
        .clamp(1.0, 100_000.0);
    let (sin_x, cos_x) = rotation_x.sin_cos();
    let (sin_y, cos_y) = rotation_y.sin_cos();
    let (sin_z, cos_z) = rotation_z.sin_cos();
    let corners = [
        (offset.0, offset.1),
        (offset.0 + size.0, offset.1),
        (offset.0 + size.0, offset.1 + size.1),
        (offset.0, offset.1 + size.1),
    ];

    Ok(corners.map(|(corner_x, corner_y)| {
        // Project the flat layer plane through a simple perspective camera.
        let mut px = (corner_x - origin_x) * scale_x;
        let mut py = (corner_y - origin_y) * scale_y;
        let mut pz = 0.0_f32;
        let y2 = py * cos_x - pz * sin_x;
        let z2 = py * sin_x + pz * cos_x;
        py = y2;
        pz = z2;
        let x2 = px * cos_y + pz * sin_y;
        let z3 = -px * sin_y + pz * cos_y;
        px = x2;
        pz = z3;
        let x3 = px * cos_z - py * sin_z;
        let y3 = px * sin_z + py * cos_z;
        let depth = (perspective + z + pz).max(1.0);
        let projected_scale = perspective / depth;
        let (screen_x, screen_y) = base_transform.transform_point(
            x + origin_x + x3 * projected_scale,
            y + origin_y + y3 * projected_scale,
        );
        (screen_x, screen_y, 1.0 / depth)
    }))
}

fn collect_graph_scene_node_defs(graph: &GraphScript, out: &mut HashMap<String, SceneNode>) {
    collect_scene_node_defs_from_nodes(&graph.scene_nodes, out);
    for scene in &graph.scenes {
        collect_scene_node_defs_from_nodes(&scene.children, out);
    }
}

fn collect_graph_scene_follow_nodes(graph: &GraphScript) -> Vec<SceneNode> {
    let mut nodes = graph.scene_nodes.clone();
    for scene in &graph.scenes {
        nodes.extend(scene.children.clone());
    }
    nodes
}

fn collect_scene_node_defs_from_nodes(nodes: &[SceneNode], out: &mut HashMap<String, SceneNode>) {
    for node in nodes {
        if let Some(id) = scene_node_id(node) {
            out.insert(id.to_string(), node.clone());
        }
        match node {
            SceneNode::Defs(defs) => {
                for mask in &defs.masks {
                    collect_scene_node_defs_from_nodes(&mask.children, out);
                }
                for precompose in &defs.precomposes {
                    out.insert(
                        precompose.id.clone(),
                        SceneNode::Precompose(precompose.clone()),
                    );
                    collect_scene_node_defs_from_nodes(&precompose.children, out);
                }
                for component in &defs.components {
                    out.insert(
                        component.id.clone(),
                        SceneNode::Group(GroupNode {
                            id: Some(component.id.clone()),
                            brush: None,
                            material: None,
                            x: "0".to_string(),
                            y: "0".to_string(),
                            rotation: "0".to_string(),
                            scale: "1".to_string(),
                            scale_x: "1".to_string(),
                            scale_y: "1".to_string(),
                            skew_x: "0".to_string(),
                            skew_y: "0".to_string(),
                            transform_origin_x: "0".to_string(),
                            transform_origin_y: "0".to_string(),
                            deform_grid: None,
                            grid_from: None,
                            grid_to: None,
                            deform_amount: "0".to_string(),
                            mask: None,
                            mask_from: None,
                            mask_mode: "alpha".to_string(),
                            mask_feather: "0".to_string(),
                            mask_expansion: "0".to_string(),
                            effects: Vec::new(),
                            process_effects: Vec::new(),
                            composite: None,
                            opacity: "1".to_string(),
                            children: component.children.clone(),
                        }),
                    );
                    collect_scene_node_defs_from_nodes(&component.children, out);
                }
            }
            SceneNode::Timeline(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Track(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Sequence(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Chain(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Group(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Part(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Repeat(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Mask(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Precompose(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Camera(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            SceneNode::Character(node) => collect_scene_node_defs_from_nodes(&node.children, out),
            _ => {}
        }
    }
}

fn scene_node_id(node: &SceneNode) -> Option<&str> {
    match node {
        SceneNode::Defs(node) => node.id.as_deref(),
        SceneNode::Timeline(node) => node.id.as_deref(),
        SceneNode::Track(node) => node.id.as_deref(),
        SceneNode::Sequence(node) => node.id.as_deref(),
        SceneNode::Chain(node) => node.id.as_deref(),
        SceneNode::Palette(node) => Some(node.id.as_str()),
        SceneNode::PixelGrid(node) => node.id.as_deref(),
        SceneNode::Text(node) => node.id.as_deref(),
        SceneNode::Image(node) => node.id.as_deref(),
        SceneNode::Svg(node) => node.id.as_deref(),
        SceneNode::Rect(node) => node.id.as_deref(),
        SceneNode::Circle(node) => node.id.as_deref(),
        SceneNode::Ellipse(node) => node.id.as_deref(),
        SceneNode::Line(node) => node.id.as_deref(),
        SceneNode::Polyline(node) => node.id.as_deref(),
        SceneNode::Path(node) => node.id.as_deref(),
        SceneNode::FaceJaw(node) => node.id.as_deref(),
        SceneNode::Shadow(_) => None,
        SceneNode::Group(node) => node.id.as_deref(),
        SceneNode::Puppet(node) => node.id.as_deref(),
        SceneNode::Pin(node) => node.id.as_deref(),
        SceneNode::LimbEnvelope(node) => node.id.as_deref(),
        SceneNode::LimbRegion(node) => node.id.as_deref(),
        SceneNode::MeshTopology(node) => node.id.as_deref(),
        SceneNode::Vertex(node) => Some(node.id.as_str()),
        SceneNode::Triangle(node) => node.id.as_deref(),
        SceneNode::Edge(node) => node.id.as_deref(),
        SceneNode::Region(node) => Some(node.id.as_str()),
        SceneNode::Part(node) => node.id.as_deref(),
        SceneNode::Repeat(node) => node.id.as_deref(),
        SceneNode::Mask(node) => node.id.as_deref(),
        SceneNode::Precompose(node) => Some(node.id.as_str()),
        SceneNode::Use(node) => node.id.as_deref(),
        SceneNode::Layer(node) => node.id.as_deref(),
        SceneNode::Camera(node) => node.id.as_deref(),
        SceneNode::Character(node) => node.id.as_deref(),
        SceneNode::Simulation(_) => None,
    }
}

fn apply_gpu_pick_id_to_empty_commands(
    primitives: &mut [GpuScenePrimitive],
    texture_layers: &mut [GpuSceneTextureLayer],
    text_requests: &mut [GpuSceneTextRequest],
    pick_id: u32,
) {
    if pick_id == 0 {
        return;
    }
    for primitive in primitives {
        if primitive.pick_id == 0 {
            primitive.pick_id = pick_id;
        }
    }
    for layer in texture_layers {
        if layer.pick_id == 0 {
            layer.pick_id = pick_id;
        }
    }
    for request in text_requests {
        if request.pick_id == 0 {
            request.pick_id = pick_id;
        }
    }
}

impl SceneFrameRenderer {
    #[allow(dead_code)]
    async fn new() -> Self {
        Self::new_for_profile(SceneRenderProfile::Cpu).await
    }

    async fn new_for_profile(profile: SceneRenderProfile) -> Self {
        Self::new_for_profile_with_resolver(profile, Arc::new(PathAssetResolver)).await
    }

    async fn new_for_profile_with_resolver(
        profile: SceneRenderProfile,
        asset_resolver: Arc<dyn AssetResolver>,
    ) -> Self {
        let mut font_system = FontSystem::new();
        load_extra_fonts(&mut font_system);
        let world_asset_resolver = asset_resolver.clone();
        Self {
            profile,
            asset_resolver,
            font_system,
            swash_cache: SwashCache::new(),
            image_cache: HashMap::new(),
            svg_cache: HashMap::new(),
            path_cache: HashMap::new(),
            polyline_cache: HashMap::new(),
            vector_overlay_cache: HashMap::new(),
            gpu_path_geometry_cache: HashMap::new(),
            retained_gpu_shape_scenes: HashMap::new(),
            retained_gpu_transform_scenes: HashMap::new(),
            gpu_text_raster_cache: HashMap::new(),
            gpu_layer3d_texture_cache: HashMap::new(),
            prepared_graph_signature: 0,
            prepared_graph_fps: 30.0,
            prepared_graph_duration_ms: 1_000,
            gpu_path_cache_hits: 0,
            gpu_path_cache_misses: 0,
            gradient_defs: HashMap::new(),
            palette_defs: HashMap::new(),
            font_defs: HashMap::new(),
            texture_defs: HashMap::new(),
            noise_defs: HashMap::new(),
            material_defs: HashMap::new(),
            filter_defs: HashMap::new(),
            scene_node_defs: HashMap::new(),
            scene_follow_nodes: Vec::new(),
            scene_components: HashMap::new(),
            scene_precompose_defs: HashMap::new(),
            scene_precomposes: HashMap::new(),
            scene_masks: HashMap::new(),
            model_asset_sources: HashMap::new(),
            image_asset_sources: HashMap::new(),
            gpu_pick_ids: HashMap::new(),
            world_renderer: WorldFrameRenderer::with_resolver(world_asset_resolver),
            gpu_compositor: None,
            #[cfg(not(target_arch = "wasm32"))]
            last_cpu_frame_profile: SceneCpuFrameProfile::default(),
            #[cfg(not(target_arch = "wasm32"))]
            external_device_queue: None,
            #[cfg(target_os = "windows")]
            windows_d3d11: None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn new_with_resolver_and_device(
        profile: SceneRenderProfile,
        asset_resolver: Arc<dyn AssetResolver>,
        device: Arc<wgpu::Device>,
        queue: wgpu::Queue,
    ) -> Self {
        let mut font_system = FontSystem::new();
        load_extra_fonts(&mut font_system);
        let world_asset_resolver = asset_resolver.clone();
        Self {
            profile,
            asset_resolver,
            font_system,
            swash_cache: SwashCache::new(),
            image_cache: HashMap::new(),
            svg_cache: HashMap::new(),
            path_cache: HashMap::new(),
            polyline_cache: HashMap::new(),
            vector_overlay_cache: HashMap::new(),
            gpu_path_geometry_cache: HashMap::new(),
            retained_gpu_shape_scenes: HashMap::new(),
            retained_gpu_transform_scenes: HashMap::new(),
            gpu_text_raster_cache: HashMap::new(),
            gpu_layer3d_texture_cache: HashMap::new(),
            prepared_graph_signature: 0,
            prepared_graph_fps: 30.0,
            prepared_graph_duration_ms: 1_000,
            gpu_path_cache_hits: 0,
            gpu_path_cache_misses: 0,
            gradient_defs: HashMap::new(),
            palette_defs: HashMap::new(),
            font_defs: HashMap::new(),
            texture_defs: HashMap::new(),
            noise_defs: HashMap::new(),
            material_defs: HashMap::new(),
            filter_defs: HashMap::new(),
            scene_node_defs: HashMap::new(),
            scene_follow_nodes: Vec::new(),
            scene_components: HashMap::new(),
            scene_precompose_defs: HashMap::new(),
            scene_precomposes: HashMap::new(),
            scene_masks: HashMap::new(),
            model_asset_sources: HashMap::new(),
            image_asset_sources: HashMap::new(),
            gpu_pick_ids: HashMap::new(),
            world_renderer: WorldFrameRenderer::with_resolver(world_asset_resolver),
            gpu_compositor: None,
            last_cpu_frame_profile: SceneCpuFrameProfile::default(),
            external_device_queue: Some((device, queue)),
            #[cfg(target_os = "windows")]
            windows_d3d11: None,
        }
    }

    async fn render_frame(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        self.render_frame_with_cpu_inputs(graph, frame, None).await
    }

    async fn render_frame_with_cpu_inputs(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        cpu_inputs: Option<&HashMap<String, RgbaImage>>,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let fps = graph.fps.max(1.0);
        let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
        let time_sec = frame as f32 / fps;
        let time_norm = (time_sec / duration_sec).clamp(0.0, 1.0);
        let animated_graph = apply_animation_targets_at_frame(graph, frame)?;
        let graph = animated_graph.as_ref().unwrap_or(graph);
        let simulated_graph =
            crate::simulation::bridge::scene::apply_scene_simulation_at_frame(graph, frame)?;
        let graph = simulated_graph.as_ref().unwrap_or(graph);
        let applied_graph = apply_action_graph_at_time(graph, time_norm, time_sec)?;
        let graph = applied_graph.as_ref().unwrap_or(graph);
        self.prepare_frame_caches(graph);

        let has_process_pipeline = !graph.inputs.is_empty()
            || !graph.textures.is_empty()
            || !graph.passes.is_empty()
            || !graph.outputs.is_empty();
        if graph_has_rich_scene_tree(graph) || has_process_pipeline || cpu_inputs.is_some() {
            return self
                .render_scene_tree_frame_with_cpu_inputs(graph, time_norm, time_sec, cpu_inputs)
                .await;
        }

        let mut canvas = if self.profile.uses_gpu_compositor() {
            self.render_gpu_base_frame(graph, time_norm, time_sec)
                .await?
        } else {
            self.render_cpu_base_frame(graph, time_norm, time_sec)?
        };

        for text in &graph.texts {
            self.draw_text(&mut canvas, text, time_norm, time_sec)?;
        }

        if let Some(output_size) = graph.render_size {
            Ok(fit_logical_canvas_to_output(&canvas, output_size))
        } else {
            Ok(canvas)
        }
    }

    /// Render a frame to a GPU texture without CPU readback.
    ///
    /// This path requires a GPU profile. CPU-only graphs and graphs that need
    /// process-pass composition are rejected so that callers can rely on a true
    /// GPU texture output.
    async fn render_frame_to_wgpu_texture(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<SceneGpuTexture, MotionLoomSceneRenderError> {
        #[cfg(not(target_arch = "wasm32"))]
        let frame_cpu_started = Instant::now();
        if !self.profile.uses_gpu_compositor() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "render_frame_to_wgpu_texture requires a GPU profile".to_string(),
            });
        }

        let fps = graph.fps.max(1.0);
        let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
        let time_sec = frame as f32 / fps;
        let time_norm = (time_sec / duration_sec).clamp(0.0, 1.0);
        #[cfg(not(target_arch = "wasm32"))]
        let expression_started = Instant::now();
        let animated_graph = apply_animation_targets_at_frame(graph, frame)?;
        let graph = animated_graph.as_ref().unwrap_or(graph);
        let simulated_graph =
            crate::simulation::bridge::scene::apply_scene_simulation_at_frame(graph, frame)?;
        let graph = simulated_graph.as_ref().unwrap_or(graph);
        let applied_graph = apply_action_graph_at_time(graph, time_norm, time_sec)?;
        let graph = applied_graph.as_ref().unwrap_or(graph);
        self.prepare_frame_caches(graph);
        #[cfg(not(target_arch = "wasm32"))]
        let expression_elapsed = expression_started.elapsed();

        let output_size = graph_output_size(graph);
        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        if let Some(compositor) = self.gpu_compositor.as_mut() {
            compositor.begin_native_frame();
        }

        let has_composition = !graph.textures.is_empty()
            || !graph.passes.is_empty()
            || !graph.outputs.is_empty()
            || !graph.layers.is_empty()
            || !graph.world_sources.is_empty();

        let native_texture_result = async {
            let native_texture = if graph_has_rich_scene_tree(graph) && !has_composition {
                // Pure scene graph: use the direct GPU scene path.
                self.render_scene_tree_frame_to_wgpu_texture(graph, time_norm, time_sec)
                    .await?
            } else if graph_has_rich_scene_tree(graph) && has_composition {
                // Mixed graph with composition: run the full resource pipeline
                // keeping textures on GPU whenever possible.
                let source = self
                    .render_scene_tree_frame_gpu(graph, time_norm, time_sec, None)
                    .await?;
                match source {
                    GraphTextureSource::Gpu(texture) => texture,
                    GraphTextureSource::Cpu(image) => {
                        self.ensure_gpu_compositor_size(image.width().max(1), image.height().max(1))
                            .await?;
                        let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                            MotionLoomSceneRenderError::GpuRender {
                                message: "GPU compositor was not initialized".to_string(),
                            }
                        })?;
                        compositor.upload_gpu_rgba_texture(&image)?
                    }
                }
            } else {
                if !graph.texts.is_empty() {
                    return Err(MotionLoomSceneRenderError::GpuRender {
                        message: "render_frame_to_wgpu_texture does not support top-level <Text> nodes in the simple scene path yet"
                            .to_string(),
                    });
                }
                self.render_gpu_base_frame_to_texture(graph, time_norm, time_sec)
                    .await?
            };
            Ok::<GpuSceneNativeTexture, MotionLoomSceneRenderError>(native_texture)
        }
        .await;
        if let Some(compositor) = self.gpu_compositor.as_mut() {
            compositor.end_native_frame();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (upload, encode, wait) = self
                .gpu_compositor
                .as_ref()
                .map(WgpuSceneCompositor::last_cpu_stages)
                .unwrap_or((Duration::ZERO, Duration::ZERO, Duration::ZERO));
            let traversal = frame_cpu_started
                .elapsed()
                .saturating_sub(expression_elapsed)
                .saturating_sub(upload)
                .saturating_sub(encode)
                .saturating_sub(wait);
            self.last_cpu_frame_profile = SceneCpuFrameProfile {
                expression_ms: expression_elapsed.as_secs_f64() * 1_000.0,
                traversal_ms: traversal.as_secs_f64() * 1_000.0,
                upload_ms: upload.as_secs_f64() * 1_000.0,
                encode_ms: encode.as_secs_f64() * 1_000.0,
                wait_ms: wait.as_secs_f64() * 1_000.0,
            };
        }
        let native_texture = native_texture_result?;

        Ok(SceneGpuTexture {
            texture: native_texture.texture,
            width: native_texture.width,
            height: native_texture.height,
            format: wgpu::TextureFormat::Rgba8Unorm,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn benchmark_vector_frame(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<VectorFrameBenchmark, MotionLoomSceneRenderError> {
        self.gpu_path_cache_hits = 0;
        self.gpu_path_cache_misses = 0;
        if !self.profile.uses_gpu_compositor() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "vector benchmark requires a GPU profile".to_string(),
            });
        }
        let fps = graph.fps.max(1.0);
        let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
        let time_sec = frame as f32 / fps;
        let time_norm = (time_sec / duration_sec).clamp(0.0, 1.0);
        let animated_graph = apply_animation_targets_at_frame(graph, frame)?;
        let graph = animated_graph.as_ref().unwrap_or(graph);
        let simulated_graph =
            crate::simulation::bridge::scene::apply_scene_simulation_at_frame(graph, frame)?;
        let graph = simulated_graph.as_ref().unwrap_or(graph);
        let applied_graph = apply_action_graph_at_time(graph, time_norm, time_sec)?;
        let graph = applied_graph.as_ref().unwrap_or(graph);
        self.prepare_frame_caches(graph);

        let nodes = scene_nodes_for_present(graph).ok_or_else(|| {
            MotionLoomSceneRenderError::GpuRender {
                message: "vector benchmark needs a presentable <Scene>".to_string(),
            }
        })?;
        let output_size = graph_output_size(graph);
        let logical_size = graph_logical_render_size(graph);
        let root_transform = render_size_root_transform(output_size, logical_size);
        let scaled_scene = output_size != logical_size || !affine_is_identity(root_transform);
        let scene_transform = if scaled_scene {
            root_transform
        } else {
            Affine2::identity()
        };
        let scene_canvas_size = if scaled_scene {
            logical_size
        } else {
            output_size
        };
        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;

        let flatten_started = Instant::now();
        let retained_static = self
            .try_collect_retained_gpu_shape_scene(
                nodes,
                scene_transform,
                1.0,
                time_norm,
                time_sec,
                scene_canvas_size,
            )
            .await?;
        let retained_transform = if retained_static.is_some() {
            None
        } else {
            self.try_collect_retained_gpu_transform_scene(
                nodes,
                scene_transform,
                time_norm,
                time_sec,
                scene_canvas_size,
            )?
        };
        let mut assets = GpuSceneNativeAssets::default();
        let mut collected_primitives = Vec::<GpuScenePrimitive>::new();
        let mut texture_layers = Vec::<GpuSceneTextureLayer>::new();
        let mut text_requests = retained_transform
            .as_ref()
            .map(|frame| frame.text_requests.clone())
            .unwrap_or_default();
        let mut unsupported = false;
        if retained_static.is_none() && retained_transform.is_none() {
            self.collect_gpu_scene_native_commands(
                nodes,
                scene_transform,
                None,
                1.0,
                time_norm,
                time_sec,
                scene_canvas_size,
                &mut assets,
                &mut collected_primitives,
                &mut texture_layers,
                &mut text_requests,
                &mut unsupported,
            )
            .await?;
        }
        let primitives = retained_static
            .as_ref()
            .map(|value| value.as_slice())
            .or_else(|| {
                retained_transform
                    .as_ref()
                    .map(|frame| frame.primitives.as_slice())
            })
            .unwrap_or(&collected_primitives);
        let flatten_ms = flatten_started.elapsed().as_secs_f64() * 1000.0;
        if unsupported || !texture_layers.is_empty() || !text_requests.is_empty() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "vector benchmark accepts only GPU-native shape nodes; images, SVG assets, and text are excluded"
                    .to_string(),
            });
        }

        let background = graph
            .backgrounds
            .last()
            .map(|background| parse_color(&background.color))
            .transpose()?
            .unwrap_or([0, 0, 0, 0]);
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let profiled = compositor.render_shape_benchmark_to_texture(primitives, background)?;
        let _texture = profiled.texture;
        Ok(VectorFrameBenchmark {
            flatten_ms,
            encode_ms: profiled.encode.as_secs_f64() * 1000.0,
            gpu_ms: profiled.gpu.as_secs_f64() * 1000.0,
            present_ms: None,
            primitive_count: profiled.primitive_count,
            upload_bytes: profiled.upload_bytes,
            path_cache_hits: self.gpu_path_cache_hits,
            path_cache_misses: self.gpu_path_cache_misses,
        })
    }

    async fn pick_id_at_wgpu_position(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        x: u32,
        y: u32,
        pick_ids: &[(String, u32)],
    ) -> Result<Option<u32>, MotionLoomSceneRenderError> {
        if !self.profile.uses_gpu_compositor() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "pick_id_at_wgpu_position requires a GPU profile".to_string(),
            });
        }

        let fps = graph.fps.max(1.0);
        let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
        let time_sec = frame as f32 / fps;
        let time_norm = (time_sec / duration_sec).clamp(0.0, 1.0);
        let animated_graph = apply_animation_targets_at_frame(graph, frame)?;
        let graph = animated_graph.as_ref().unwrap_or(graph);
        let simulated_graph =
            crate::simulation::bridge::scene::apply_scene_simulation_at_frame(graph, frame)?;
        let graph = simulated_graph.as_ref().unwrap_or(graph);
        let applied_graph = apply_action_graph_at_time(graph, time_norm, time_sec)?;
        let graph = applied_graph.as_ref().unwrap_or(graph);
        self.prepare_frame_caches(graph);

        let previous_pick_ids = std::mem::take(&mut self.gpu_pick_ids);
        self.gpu_pick_ids = pick_ids
            .iter()
            .filter_map(|(node, id)| (*id > 0).then_some((node.clone(), *id)))
            .collect();
        let result = self
            .render_scene_tree_pick_id_at(graph, time_norm, time_sec, x, y)
            .await;
        self.gpu_pick_ids = previous_pick_ids;
        result
    }

    async fn render_frame_to_wgpu_target_texture(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        target: &wgpu::Texture,
        target_width: u32,
        target_height: u32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let source = self.render_frame_to_wgpu_texture(graph, frame).await?;
        if source.width != target_width || source.height != target_height {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "wgpu target size {}x{} does not match rendered frame {}x{}",
                    target_width, target_height, source.width, source.height
                ),
            });
        }
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        compositor.copy_native_texture_to_target(
            &source.texture,
            target,
            source.width,
            source.height,
        );
        Ok(())
    }

    async fn render_frame_to_preview_surface(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        options: ScenePreviewSurfaceOptions,
    ) -> Result<ScenePreviewSurface, MotionLoomSceneRenderError> {
        match options.backend {
            ScenePreviewBackend::Auto => {
                if let Ok(surface) = self
                    .render_frame_to_platform_preview_surface(graph, frame, options)
                    .await
                {
                    return Ok(surface);
                }
                if let Ok(texture) = self.render_frame_to_wgpu_texture(graph, frame).await {
                    return Ok(ScenePreviewSurface::WgpuTexture(texture));
                }
                self.render_frame_to_cpu_bgra_preview_surface(graph, frame)
                    .await
            }
            ScenePreviewBackend::WgpuTexture => self
                .render_frame_to_wgpu_texture(graph, frame)
                .await
                .map(ScenePreviewSurface::WgpuTexture),
            ScenePreviewBackend::PlatformSurface => {
                self.render_frame_to_platform_preview_surface(graph, frame, options)
                    .await
            }
            ScenePreviewBackend::CpuBgra => {
                self.render_frame_to_cpu_bgra_preview_surface(graph, frame)
                    .await
            }
        }
    }

    async fn render_frame_to_cpu_bgra_preview_surface(
        &mut self,
        graph: &GraphScript,
        frame: u32,
    ) -> Result<ScenePreviewSurface, MotionLoomSceneRenderError> {
        let image = self.render_frame(graph, frame).await?;
        let width = image.width();
        let height = image.height();
        let mut data = image.into_raw();
        for pixel in data.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        Ok(ScenePreviewSurface::CpuBgra {
            width,
            height,
            data: Arc::new(data),
            format: ScenePreviewPixelFormat::Bgra8Unorm,
        })
    }

    async fn render_frame_to_platform_preview_surface(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        options: ScenePreviewSurfaceOptions,
    ) -> Result<ScenePreviewSurface, MotionLoomSceneRenderError> {
        #[cfg(target_os = "macos")]
        {
            self.render_frame_to_macos_preview_surface(graph, frame, options)
                .await
        }

        #[cfg(target_os = "windows")]
        {
            self.render_frame_to_windows_preview_surface(graph, frame, options)
                .await
        }

        #[cfg(all(unix, not(target_os = "macos"), not(target_arch = "wasm32")))]
        {
            let _ = (graph, frame, options);
            Err(MotionLoomSceneRenderError::GpuRender {
                message: "Linux DMA-BUF platform preview surfaces are not implemented yet"
                    .to_string(),
            })
        }

        #[cfg(not(any(
            target_os = "macos",
            target_os = "windows",
            all(unix, not(target_os = "macos"), not(target_arch = "wasm32"))
        )))]
        {
            let _ = (graph, frame, options);
            Err(MotionLoomSceneRenderError::GpuRender {
                message: "platform preview surfaces are not implemented for this target yet"
                    .to_string(),
            })
        }
    }

    #[cfg(target_os = "macos")]
    async fn render_frame_to_macos_preview_surface(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        _options: ScenePreviewSurfaceOptions,
    ) -> Result<ScenePreviewSurface, MotionLoomSceneRenderError> {
        let image = self.render_frame(graph, frame).await?;
        let width = image.width();
        let height = image.height();
        let mut bgra = image.into_raw();
        for pixel in bgra.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        let surface = create_macos_bgra_surface(width, height).ok_or_else(|| {
            MotionLoomSceneRenderError::GpuRender {
                message: format!("failed to allocate macOS BGRA preview surface {width}x{height}"),
            }
        })?;
        if !copy_bgra_into_macos_surface(&surface, width, height, &bgra) {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "failed to copy BGRA preview data into macOS surface {width}x{height}"
                ),
            });
        }
        Ok(ScenePreviewSurface::PlatformSurface(
            ScenePlatformPreviewSurface::MacOs {
                surface,
                width,
                height,
                format: ScenePreviewPixelFormat::Bgra8Unorm,
            },
        ))
    }

    #[cfg(target_os = "windows")]
    async fn render_frame_to_windows_preview_surface(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        _options: ScenePreviewSurfaceOptions,
    ) -> Result<ScenePreviewSurface, MotionLoomSceneRenderError> {
        // First render to CPU and convert to BGRA. Future iterations can keep
        // the entire pipeline on GPU once wgpu shared-handle export is stable.
        let image = self.render_frame(graph, frame).await?;
        let width = image.width();
        let height = image.height();
        let mut bgra = image.into_raw();
        for pixel in bgra.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }

        if self.windows_d3d11.is_none() {
            self.windows_d3d11 = WindowsD3D11Context::new();
        }
        let ctx =
            self.windows_d3d11
                .as_ref()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "failed to create D3D11 device for Windows preview surface"
                        .to_string(),
                })?;

        let surface = WindowsD3DSharedSurface::new(&ctx.device, &ctx.context, width, height, &bgra)
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "failed to create D3D11 shared BGRA preview surface {width}x{height}"
                ),
            })?;

        Ok(ScenePreviewSurface::PlatformSurface(
            ScenePlatformPreviewSurface::WindowsD3D(surface),
        ))
    }

    fn prepare_frame_caches(&mut self, graph: &GraphScript) {
        self.prepared_graph_fps = graph.fps.max(1.0);
        self.prepared_graph_duration_ms = graph.duration_ms.max(1);
        // Action evaluation may reuse the same allocation for different
        // frame-local Graph clones, so pointer identity alone is not a valid
        // retained-scene revision. Always verify the content signature.
        let mut hasher = DefaultHasher::new();
        if let Some(raw_script) = graph.raw_script.as_deref() {
            raw_script.hash(&mut hasher);
        } else {
            format!("{graph:?}").hash(&mut hasher);
        }
        let graph_signature = hasher.finish();
        if self.prepared_graph_signature != graph_signature {
            self.retained_gpu_shape_scenes.clear();
            self.retained_gpu_transform_scenes.clear();
            self.gpu_text_raster_cache.clear();
            self.gpu_layer3d_texture_cache.clear();
            self.prepared_graph_signature = graph_signature;
        }
        self.gradient_defs.clear();
        self.palette_defs.clear();
        self.font_defs.clear();
        self.texture_defs.clear();
        self.noise_defs.clear();
        self.material_defs.clear();
        self.filter_defs.clear();
        self.scene_node_defs.clear();
        self.scene_follow_nodes.clear();
        self.scene_components.clear();
        self.scene_precompose_defs.clear();
        self.scene_precomposes.clear();
        self.scene_masks.clear();
        self.model_asset_sources.clear();
        self.image_asset_sources.clear();
        for asset in &graph.assets {
            match asset.kind {
                GraphAssetKind::Model => {
                    self.model_asset_sources
                        .insert(asset.id.clone(), asset.src.clone());
                }
                GraphAssetKind::Image => {
                    self.image_asset_sources
                        .insert(asset.id.clone(), asset.src.clone());
                }
                GraphAssetKind::Video | GraphAssetKind::Audio => {}
            }
        }
        collect_graph_gradient_defs(graph, &mut self.gradient_defs);
        collect_graph_palette_defs(graph, &mut self.palette_defs);
        collect_graph_font_defs(graph, &mut self.font_defs);
        collect_graph_texture_defs(graph, &mut self.texture_defs);
        collect_graph_noise_defs(graph, &mut self.noise_defs);
        collect_graph_material_defs(graph, &mut self.material_defs);
        collect_graph_filter_defs(graph, &mut self.filter_defs);
        collect_graph_scene_node_defs(graph, &mut self.scene_node_defs);
        self.scene_follow_nodes = collect_graph_scene_follow_nodes(graph);
        collect_graph_component_defs(graph, &mut self.scene_components);
        collect_graph_mask_defs(graph, &mut self.scene_masks);
        for precompose in collect_graph_precompose_defs(graph) {
            self.scene_precompose_defs
                .insert(precompose.id.clone(), precompose);
        }
    }

    /// Present one strict GPU scene-tree frame directly to an HTML canvas.
    #[cfg(target_arch = "wasm32")]
    async fn render_frame_to_canvas(
        &mut self,
        graph: &GraphScript,
        frame: u32,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(), MotionLoomSceneRenderError> {
        if !self.profile.uses_gpu_compositor() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "canvas rendering requires the GPU profile".to_string(),
            });
        }

        // The canvas path should use the same GPU-aware graph pipeline as
        // `render_frame_to_wgpu_texture`, otherwise mixed Scene+Process graphs
        // fall back to CPU even when every pass is GPU-native.
        let has_composition = !graph.textures.is_empty()
            || !graph.passes.is_empty()
            || !graph.outputs.is_empty()
            || !graph.layers.is_empty()
            || !graph.world_sources.is_empty();
        match self.render_frame_to_wgpu_texture(graph, frame).await {
            Ok(texture) => {
                let native_texture = GpuSceneNativeTexture {
                    texture: texture.texture,
                    width: texture.width,
                    height: texture.height,
                    _keepalive_textures: Vec::new(),
                };
                let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                    MotionLoomSceneRenderError::GpuRender {
                        message: "GPU compositor was not initialized".to_string(),
                    }
                })?;
                return compositor.present_texture_to_canvas(&native_texture, &canvas);
            }
            Err(err) if has_composition => {
                return Err(MotionLoomSceneRenderError::GpuRender {
                    message: format!(
                        "mixed Scene+Process canvas GPU render failed before CPU fallback: {err}"
                    ),
                });
            }
            Err(_) => {}
        }

        let fps = graph.fps.max(1.0);
        let duration_sec = (graph.duration_ms as f32 / 1000.0).max(1.0 / fps);
        let time_sec = frame as f32 / fps;
        let time_norm = (time_sec / duration_sec).clamp(0.0, 1.0);
        let animated_graph = apply_animation_targets_at_frame(graph, frame)?;
        let graph = animated_graph.as_ref().unwrap_or(graph);
        let simulated_graph =
            crate::simulation::bridge::scene::apply_scene_simulation_at_frame(graph, frame)?;
        let graph = simulated_graph.as_ref().unwrap_or(graph);
        let applied_graph = apply_action_graph_at_time(graph, time_norm, time_sec)?;
        let graph = applied_graph.as_ref().unwrap_or(graph);

        self.gradient_defs.clear();
        self.palette_defs.clear();
        self.font_defs.clear();
        self.texture_defs.clear();
        self.noise_defs.clear();
        self.material_defs.clear();
        self.filter_defs.clear();
        self.scene_node_defs.clear();
        self.scene_follow_nodes.clear();
        self.scene_components.clear();
        self.scene_precompose_defs.clear();
        self.scene_precomposes.clear();
        self.scene_masks.clear();
        collect_graph_gradient_defs(graph, &mut self.gradient_defs);
        collect_graph_palette_defs(graph, &mut self.palette_defs);
        collect_graph_font_defs(graph, &mut self.font_defs);
        collect_graph_texture_defs(graph, &mut self.texture_defs);
        collect_graph_noise_defs(graph, &mut self.noise_defs);
        collect_graph_material_defs(graph, &mut self.material_defs);
        collect_graph_filter_defs(graph, &mut self.filter_defs);
        collect_graph_scene_node_defs(graph, &mut self.scene_node_defs);
        self.scene_follow_nodes = collect_graph_scene_follow_nodes(graph);
        collect_graph_component_defs(graph, &mut self.scene_components);
        collect_graph_mask_defs(graph, &mut self.scene_masks);
        for precompose in collect_graph_precompose_defs(graph) {
            self.scene_precompose_defs
                .insert(precompose.id.clone(), precompose);
        }

        self.render_scene_tree_frame_to_canvas(graph, time_norm, time_sec, canvas)
            .await
    }

    /// Draw a solid WebGPU color into a browser canvas for surface debugging.
    #[cfg(target_arch = "wasm32")]
    async fn debug_solid_to_canvas(
        &mut self,
        canvas: web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
        color: [f64; 4],
    ) -> Result<(), MotionLoomSceneRenderError> {
        if !self.profile.uses_gpu_compositor() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "debug canvas rendering requires the GPU profile".to_string(),
            });
        }
        self.ensure_gpu_compositor_size(width.max(1), height.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        compositor.debug_present_solid_to_canvas(&canvas, width, height, color)
    }

    /// Upload a solid texture and present it to a browser canvas for debugging.
    #[cfg(target_arch = "wasm32")]
    async fn debug_uploaded_texture_to_canvas(
        &mut self,
        canvas: web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
        color: [u8; 4],
    ) -> Result<(), MotionLoomSceneRenderError> {
        if !self.profile.uses_gpu_compositor() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "debug texture rendering requires the GPU profile".to_string(),
            });
        }
        self.ensure_gpu_compositor_size(width.max(1), height.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_ref()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        compositor.debug_present_uploaded_texture_to_canvas(&canvas, width, height, color)
    }

    /// Render an empty scene texture with a white clear color and present it.
    #[cfg(target_arch = "wasm32")]
    async fn debug_empty_scene_texture_to_canvas(
        &mut self,
        canvas: web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        if !self.profile.uses_gpu_compositor() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "debug empty scene rendering requires the GPU profile".to_string(),
            });
        }
        self.ensure_gpu_compositor_size(width.max(1), height.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let texture = compositor.render_scene_content_to_texture(&[], &[], [255, 255, 255, 255])?;
        compositor.present_texture_to_canvas(&texture, &canvas)
    }

    /// Render a GPU-native scene tree to a browser canvas surface.
    #[cfg(target_arch = "wasm32")]
    async fn render_scene_tree_frame_to_canvas(
        &mut self,
        graph: &GraphScript,
        time_norm: f32,
        time_sec: f32,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let has_composition = !graph.textures.is_empty()
            || !graph.passes.is_empty()
            || !graph.outputs.is_empty()
            || !graph.layers.is_empty()
            || !graph.world_sources.is_empty();
        if has_composition {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message:
                    "direct WASM canvas GPU rendering does not support Tex/Pass/Output composition yet"
                        .to_string(),
            });
        }

        let Some(nodes) = scene_nodes_for_present(graph) else {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "direct WASM canvas GPU rendering needs a presentable scene tree"
                    .to_string(),
            });
        };
        let background = graph
            .backgrounds
            .last()
            .map(|background| parse_color(&background.color))
            .transpose()?
            .unwrap_or([0, 0, 0, 0]);
        self.present_gpu_scene_nodes_with_background_to_canvas(
            nodes,
            graph_output_size(graph),
            graph_logical_render_size(graph),
            render_size_root_transform(graph_output_size(graph), graph_logical_render_size(graph)),
            time_norm,
            time_sec,
            Some(background),
            canvas,
        )
        .await
    }

    /// GPU-aware scene tree frame render that returns a `GraphTextureSource`.
    ///
    /// This is the internal pipeline used by `render_frame_to_wgpu_texture`. When
    /// all scene nodes and process passes are GPU-native, the returned source
    /// holds a `GpuSceneNativeTexture` without any CPU readback.
    async fn render_scene_tree_frame_gpu(
        &mut self,
        graph: &GraphScript,
        time_norm: f32,
        time_sec: f32,
        cpu_inputs: Option<&HashMap<String, RgbaImage>>,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let (w, h) = graph_output_size(graph);
        let output_size = (w, h);
        let logical_size = graph_logical_render_size(graph);
        let root_transform = render_size_root_transform(output_size, logical_size);
        let mut resources = HashMap::<String, GraphTextureSource>::new();
        let background = graph
            .backgrounds
            .last()
            .map(|background| parse_color(&background.color))
            .transpose()?
            .unwrap_or([0, 0, 0, 0]);

        // Background starts the graph resource pipeline. Keep it on GPU when
        // possible so scene-to-process graphs do not force CPU readback before
        // the first post pass.
        let background_source = if self.profile.uses_gpu_compositor() {
            match self
                .render_gpu_background_texture(output_size, background)
                .await
            {
                Ok(texture) => GraphTextureSource::Gpu(texture),
                Err(_) => GraphTextureSource::Cpu(solid_canvas(output_size, background)),
            }
        } else {
            GraphTextureSource::Cpu(solid_canvas(output_size, background))
        };
        resources.insert("scene".to_string(), background_source.clone());
        resources.insert("background".to_string(), background_source.clone());
        for background_node in &graph.backgrounds {
            if let Some(id) = background_node.id.as_deref() {
                resources.insert(id.to_string(), background_source.clone());
            }
        }
        if let Some(cpu_inputs) = cpu_inputs {
            Self::insert_cpu_input_resources(&mut resources, graph, cpu_inputs);
        }

        if !graph.world_sources.is_empty() {
            let raw_script = graph.raw_script.as_deref().ok_or_else(|| {
                MotionLoomSceneRenderError::WorldSource {
                    message: "unified graph is missing raw DSL required by a legacy world source"
                        .to_string(),
                }
            })?;
            let world_frame = (time_sec * graph.fps.max(1.0)).round().max(0.0) as u32;
            let world_asset_root = default_world_asset_root();
            let base_world_graph = parse_world_graph_script(raw_script).map_err(|err| {
                MotionLoomSceneRenderError::WorldSource {
                    message: format!("parse error at line {}: {}", err.line, err.message),
                }
            })?;
            for world_source in &graph.world_sources {
                let mut world_graph = base_world_graph.clone();
                world_graph.present.from = world_source.id.clone();
                world_graph.render_size = Some(output_size);
                let image = self
                    .world_renderer
                    .render_frame_gpu(&world_graph, world_frame, &world_asset_root)
                    .await
                    .map_err(|err| MotionLoomSceneRenderError::WorldSource {
                        message: err.to_string(),
                    })?;
                let source = GraphTextureSource::Cpu(image);
                resources.insert(world_source.id.clone(), source.clone());
                resources.insert(format!("world:{}", world_source.id), source);
            }
        }

        if !graph.scene_nodes.is_empty() {
            let maybe_gpu_source = self
                .try_render_gpu_scene_nodes(
                    &graph.scene_nodes,
                    output_size,
                    logical_size,
                    root_transform,
                    time_norm,
                    time_sec,
                )
                .await?;
            let source = if let Some(source) = maybe_gpu_source {
                source
            } else if self.profile.uses_gpu_compositor() {
                return Err(MotionLoomSceneRenderError::GpuRender {
                    message:
                        "GPU render is strict: root scene nodes are not supported by the GPU scene renderer."
                            .to_string(),
                });
            } else {
                let image = self.render_cpu_scene_nodes_scaled(
                    &graph.scene_nodes,
                    output_size,
                    logical_size,
                    time_norm,
                    time_sec,
                    [0, 0, 0, 0],
                )?;
                GraphTextureSource::Cpu(image)
            };
            resources.insert("scene".to_string(), source);
        }

        for scene in &graph.scenes {
            let scene_size = scene.size.unwrap_or(graph.size);
            let (scene_output_size, scene_logical_size, scene_transform) = if scene.size.is_some() {
                (scene_size, scene_size, Affine2::identity())
            } else {
                (output_size, logical_size, root_transform)
            };
            let maybe_gpu_source = self
                .try_render_gpu_scene_nodes_with_background(
                    &scene.children,
                    scene_output_size,
                    scene_logical_size,
                    scene_transform,
                    time_norm,
                    time_sec,
                    Some(background),
                )
                .await?;
            let scene_source = if let Some(source) = maybe_gpu_source {
                source
            } else if self.profile.uses_gpu_compositor() {
                return Err(MotionLoomSceneRenderError::GpuRender {
                    message: format!(
                        "GPU render is strict: <Scene id=\"{}\"> contains nodes not supported by the GPU scene renderer.",
                        scene.id
                    ),
                });
            } else {
                let image = self.render_cpu_scene_nodes_scaled(
                    &scene.children,
                    scene_output_size,
                    scene_logical_size,
                    time_norm,
                    time_sec,
                    background,
                )?;
                GraphTextureSource::Cpu(image)
            };
            let scene_source = self
                .apply_scene_effect_refs(
                    scene_source,
                    &scene.effects,
                    graph,
                    scene_output_size,
                    time_norm,
                    time_sec,
                )
                .await?;
            let scene_source = self
                .apply_scene_effect_refs(
                    scene_source,
                    &scene.post_effects,
                    graph,
                    scene_output_size,
                    time_norm,
                    time_sec,
                )
                .await?;
            resources.insert(scene.id.clone(), scene_source.clone());
            resources.insert(format!("scene:{}", scene.id), scene_source.clone());
            resources.entry("scene".to_string()).or_insert(scene_source);
        }

        for tex in &graph.textures {
            let source = if let (Some(layer_id), Some(input_id)) =
                (tex.from.as_deref(), tex.input.as_deref())
            {
                if let Some(layer) = graph.layers.iter().find(|layer| layer.id == layer_id) {
                    let base = resources.get(input_id).cloned().unwrap_or_else(|| {
                        GraphTextureSource::Cpu(RgbaImage::from_pixel(
                            w.max(1),
                            h.max(1),
                            Rgba([0, 0, 0, 0]),
                        ))
                    });
                    let base_image = self.graph_source_to_cpu(&base).await?;
                    GraphTextureSource::Cpu(apply_layer_effects(
                        &base_image,
                        layer,
                        time_norm,
                        time_sec,
                    )?)
                } else {
                    resources.get(layer_id).cloned().unwrap_or_else(|| {
                        GraphTextureSource::Cpu(RgbaImage::from_pixel(
                            w.max(1),
                            h.max(1),
                            Rgba([0, 0, 0, 0]),
                        ))
                    })
                }
            } else if let Some(from) = tex.from.as_deref() {
                resources.get(from).cloned().unwrap_or_else(|| {
                    GraphTextureSource::Cpu(RgbaImage::from_pixel(
                        w.max(1),
                        h.max(1),
                        Rgba([0, 0, 0, 0]),
                    ))
                })
            } else {
                let size = tex.size.unwrap_or(graph.size);
                GraphTextureSource::Cpu(RgbaImage::from_pixel(
                    size.0.max(1),
                    size.1.max(1),
                    Rgba([0, 0, 0, 0]),
                ))
            };
            resources.insert(tex.id.clone(), source);
        }

        for pass in &graph.passes {
            let inputs = pass
                .inputs
                .iter()
                .filter_map(|input| resources.get(input.resource_id()).cloned())
                .collect::<Vec<_>>();
            if inputs.is_empty() {
                continue;
            }
            let mut output = self
                .apply_scene_post_pass_multi(&inputs, pass, time_norm, time_sec)
                .await?;
            output = self
                .apply_process_pass_mask(
                    output,
                    inputs.first(),
                    pass,
                    &resources,
                    time_norm,
                    time_sec,
                )
                .await?;
            for output_ref in &pass.outputs {
                resources.insert(output_ref.resource_id().to_string(), output.clone());
            }
        }

        // Layer Tex nodes can depend on Pass outputs declared later in the DSL.
        // Re-resolve them after passes so <Tex from="layer" input="comp" /> uses
        // the final upstream resource instead of a transparent placeholder.
        for tex in graph.textures.iter().filter(|tex| tex.input.is_some()) {
            if let (Some(layer_id), Some(input_id)) = (tex.from.as_deref(), tex.input.as_deref())
                && let Some(layer) = graph.layers.iter().find(|layer| layer.id == layer_id)
                && let Some(base) = resources.get(input_id).cloned()
            {
                let base_image = self.graph_source_to_cpu(&base).await?;
                let image = apply_layer_effects(&base_image, layer, time_norm, time_sec)?;
                resources.insert(tex.id.clone(), GraphTextureSource::Cpu(image));
            }
        }

        for output in &graph.outputs {
            if let Some(from) = output.from.as_deref()
                && let Some(source) = resources.get(from).cloned()
            {
                resources.insert(output.id.clone(), source);
            }
        }

        let source = resources
            .get(&graph.present.from)
            .or_else(|| {
                graph
                    .present
                    .from
                    .strip_prefix("scene:")
                    .and_then(|id| resources.get(id))
            })
            .or_else(|| resources.get("scene"))
            .cloned()
            .unwrap_or_else(|| {
                GraphTextureSource::Cpu(RgbaImage::from_pixel(
                    w.max(1),
                    h.max(1),
                    Rgba([0, 0, 0, 0]),
                ))
            });
        Ok(source)
    }

    async fn apply_scene_effect_refs(
        &mut self,
        mut source: GraphTextureSource,
        effects: &[SceneEffectRef],
        graph: &GraphScript,
        output_size: (u32, u32),
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        for effect in effects {
            let process = graph
                .processes
                .iter()
                .find(|process| process.id == effect.process)
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: format!(
                        "Scene Effect references unknown Process \"{}\".",
                        effect.process
                    ),
                })?;
            source = self
                .apply_process_definition_to_scene_source(
                    source,
                    process,
                    effect,
                    graph,
                    output_size,
                    time_norm,
                    time_sec,
                )
                .await?;
        }
        Ok(source)
    }

    #[allow(clippy::too_many_arguments)]
    async fn apply_process_definition_to_scene_source(
        &mut self,
        source: GraphTextureSource,
        process: &ProcessDefinitionNode,
        effect: &SceneEffectRef,
        graph: &GraphScript,
        output_size: (u32, u32),
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let mut local = HashMap::<String, GraphTextureSource>::new();
        local.insert("scene".to_string(), source.clone());
        for input_id in &process.input_ids {
            local.insert(input_id.clone(), source.clone());
            local.insert(format!("input:{input_id}"), source.clone());
        }

        for tex_id in &process.texture_ids {
            let Some(tex) = graph.textures.iter().find(|tex| tex.id == *tex_id) else {
                continue;
            };
            let resolved = tex
                .from
                .as_deref()
                .and_then(|from| scene_effect_resource(&local, from))
                .cloned()
                .or_else(|| {
                    tex.input
                        .as_deref()
                        .and_then(|input| scene_effect_resource(&local, input))
                        .cloned()
                })
                .unwrap_or_else(|| {
                    let size = tex.size.unwrap_or(output_size);
                    GraphTextureSource::Cpu(RgbaImage::from_pixel(
                        size.0.max(1),
                        size.1.max(1),
                        Rgba([0, 0, 0, 0]),
                    ))
                });
            local.insert(tex.id.clone(), resolved);
        }

        for pass_id in &process.pass_ids {
            let Some(base_pass) = graph.passes.iter().find(|pass| pass.id == *pass_id) else {
                continue;
            };
            let mut pass = base_pass.clone();
            for override_param in &effect.params {
                if let Some(existing) = pass
                    .params
                    .iter_mut()
                    .find(|param| param.key == override_param.name)
                {
                    existing.value = override_param.value.clone();
                } else {
                    pass.params.push(PassParam {
                        key: override_param.name.clone(),
                        value: override_param.value.clone(),
                    });
                }
            }
            let inputs = pass
                .inputs
                .iter()
                .filter_map(|input| scene_effect_resource(&local, input.resource_id()).cloned())
                .collect::<Vec<_>>();
            if inputs.is_empty() {
                return Err(MotionLoomSceneRenderError::GpuRender {
                    message: format!(
                        "Scene Effect Process \"{}\" pass \"{}\" has no resolved input.",
                        process.id, pass.id
                    ),
                });
            }
            let mut output = self
                .apply_scene_post_pass_multi(&inputs, &pass, time_norm, time_sec)
                .await?;
            output = self
                .apply_process_pass_mask(output, inputs.first(), &pass, &local, time_norm, time_sec)
                .await?;
            for output_ref in &pass.outputs {
                local.insert(output_ref.resource_id().to_string(), output.clone());
            }
        }

        scene_effect_resource(&local, &process.output)
            .cloned()
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "Scene Effect Process \"{}\" did not produce output \"{}\".",
                    process.id, process.output
                ),
            })
    }

    fn insert_cpu_input_resources(
        resources: &mut HashMap<String, GraphTextureSource>,
        graph: &GraphScript,
        cpu_inputs: &HashMap<String, RgbaImage>,
    ) {
        for (id, image) in cpu_inputs {
            let source = GraphTextureSource::Cpu(image.clone());
            resources.insert(id.clone(), source.clone());
            resources.insert(format!("input:{id}"), source);
        }

        for input in &graph.inputs {
            let source = resources
                .get(&input.id)
                .cloned()
                .or_else(|| {
                    input
                        .from
                        .as_deref()
                        .and_then(|from| resources.get(from).cloned())
                })
                .or_else(|| {
                    input
                        .from
                        .as_deref()
                        .and_then(|from| from.strip_prefix("input:"))
                        .and_then(|raw| resources.get(raw).cloned())
                });
            if let Some(source) = source {
                resources.insert(input.id.clone(), source.clone());
                resources.insert(format!("input:{}", input.id), source.clone());
                if let Some(from) = input.from.as_deref() {
                    resources.insert(from.to_string(), source);
                }
            }
        }
    }

    async fn render_scene_tree_frame_with_cpu_inputs(
        &mut self,
        graph: &GraphScript,
        time_norm: f32,
        time_sec: f32,
        cpu_inputs: Option<&HashMap<String, RgbaImage>>,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let source = self
            .render_scene_tree_frame_gpu(graph, time_norm, time_sec, cpu_inputs)
            .await?;
        let (w, h) = graph_output_size(graph);
        let image = self.graph_source_to_cpu(&source).await?;
        if image.width() == w && image.height() == h {
            Ok(image)
        } else {
            let mut canvas = RgbaImage::from_pixel(w.max(1), h.max(1), Rgba([0, 0, 0, 0]));
            draw_rgba_image(&mut canvas, &image, 0.0, 0.0, 1.0);
            Ok(canvas)
        }
    }

    #[allow(dead_code)]
    async fn try_render_gpu_scene_tree_frame(
        &mut self,
        graph: &GraphScript,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<Option<GraphTextureSource>, MotionLoomSceneRenderError> {
        if !self.profile.uses_gpu_compositor() {
            return Ok(None);
        }
        self.prepare_frame_caches(graph);

        let Some(nodes) = scene_nodes_for_present(graph) else {
            return Ok(None);
        };
        let background = graph
            .backgrounds
            .last()
            .map(|background| parse_color(&background.color))
            .transpose()?
            .unwrap_or([0, 0, 0, 0]);

        self.try_render_gpu_scene_nodes_with_background(
            nodes,
            graph_output_size(graph),
            graph_logical_render_size(graph),
            render_size_root_transform(graph_output_size(graph), graph_logical_render_size(graph)),
            time_norm,
            time_sec,
            Some(background),
        )
        .await
    }

    async fn try_render_gpu_scene_nodes(
        &mut self,
        nodes: &[SceneNode],
        output_size: (u32, u32),
        logical_size: (u32, u32),
        root_transform: Affine2,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<Option<GraphTextureSource>, MotionLoomSceneRenderError> {
        self.try_render_gpu_scene_nodes_with_background(
            nodes,
            output_size,
            logical_size,
            root_transform,
            time_norm,
            time_sec,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_render_gpu_scene_nodes_with_background(
        &mut self,
        nodes: &[SceneNode],
        output_size: (u32, u32),
        logical_size: (u32, u32),
        root_transform: Affine2,
        time_norm: f32,
        time_sec: f32,
        background: Option<[u8; 4]>,
    ) -> Result<Option<GraphTextureSource>, MotionLoomSceneRenderError> {
        if !self.profile.uses_gpu_compositor() {
            return Ok(None);
        }
        if scene_nodes_require_cpu_scene_compositing(nodes)
            || scene_nodes_contain_image_or_svg(nodes)
        {
            if let Some(source) = self
                .try_render_gpu_scene_nodes_composited(
                    nodes,
                    output_size,
                    logical_size,
                    root_transform,
                    time_norm,
                    time_sec,
                    background,
                )
                .await?
            {
                return Ok(Some(source));
            }
            return Ok(None);
        }

        let mut primitives = Vec::<GpuScenePrimitive>::new();
        let mut scene_overlays = Vec::<CpuSceneOverlay>::new();
        let mut text_requests = Vec::<GpuSceneTextRequest>::new();
        collect_gpu_scene_commands(
            nodes,
            root_transform,
            None,
            1.0,
            time_norm,
            time_sec,
            logical_size,
            &self.gradient_defs,
            &self.palette_defs,
            &self.scene_components,
            &mut primitives,
            &mut text_requests,
            &mut scene_overlays,
        )?;

        if !scene_overlays.is_empty() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "GPU Render is strict for MotionLoom: CPU overlays are disabled. Unsupported GPU-native nodes: {}",
                    describe_cpu_scene_overlays(&scene_overlays)
                ),
            });
        }

        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;

        let mut texture_layers = Vec::with_capacity(text_requests.len());
        for request in text_requests {
            texture_layers.extend(self.rasterize_text_texture_layers_gpu_effects(
                &request.node,
                TextTextureRasterContext {
                    transform: request.transform,
                    inherited_opacity: request.opacity,
                    pick_id: request.pick_id,
                    time_norm,
                    time_sec,
                    canvas_size: logical_size,
                },
            )?);
        }

        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let texture = compositor.render_scene_content_to_texture(
            &primitives,
            &texture_layers,
            background.unwrap_or([0, 0, 0, 0]),
        )?;
        Ok(Some(GraphTextureSource::Gpu(texture)))
    }

    /// Present GPU-native scene nodes directly to a browser canvas surface.
    #[cfg(target_arch = "wasm32")]
    #[allow(clippy::too_many_arguments)]
    async fn present_gpu_scene_nodes_with_background_to_canvas(
        &mut self,
        nodes: &[SceneNode],
        output_size: (u32, u32),
        logical_size: (u32, u32),
        root_transform: Affine2,
        time_norm: f32,
        time_sec: f32,
        background: Option<[u8; 4]>,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let scaled_scene = output_size != logical_size || !affine_is_identity(root_transform);
        let scene_transform = if scaled_scene {
            root_transform
        } else {
            Affine2::identity()
        };
        let scene_canvas_size = if scaled_scene {
            logical_size
        } else {
            output_size
        };

        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        let retained_static = self
            .try_collect_retained_gpu_shape_scene(
                nodes,
                scene_transform,
                1.0,
                time_norm,
                time_sec,
                scene_canvas_size,
            )
            .await?;
        let retained_transform = if retained_static.is_some() {
            None
        } else {
            self.try_collect_retained_gpu_transform_scene(
                nodes,
                scene_transform,
                time_norm,
                time_sec,
                scene_canvas_size,
            )?
        };
        let mut assets = GpuSceneNativeAssets::default();
        let mut collected_primitives = Vec::<GpuScenePrimitive>::new();
        let mut texture_layers = Vec::<GpuSceneTextureLayer>::new();
        let mut text_requests = retained_transform
            .as_ref()
            .map(|frame| frame.text_requests.clone())
            .unwrap_or_default();
        let mut unsupported = false;
        if retained_static.is_none() && retained_transform.is_none() {
            self.collect_gpu_scene_native_commands(
                nodes,
                scene_transform,
                None,
                1.0,
                time_norm,
                time_sec,
                scene_canvas_size,
                &mut assets,
                &mut collected_primitives,
                &mut texture_layers,
                &mut text_requests,
                &mut unsupported,
            )
            .await?;
        }
        let primitives = retained_static
            .as_ref()
            .map(|value| value.as_slice())
            .or_else(|| {
                retained_transform
                    .as_ref()
                    .map(|frame| frame.primitives.as_slice())
            })
            .unwrap_or(&collected_primitives);
        if unsupported {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message:
                    "direct WASM canvas GPU rendering does not support this scene node set yet"
                        .to_string(),
            });
        }

        for request in text_requests {
            texture_layers.extend(self.rasterize_text_texture_layers_gpu_effects(
                &request.node,
                TextTextureRasterContext {
                    transform: request.transform,
                    inherited_opacity: request.opacity,
                    pick_id: request.pick_id,
                    time_norm,
                    time_sec,
                    canvas_size: scene_canvas_size,
                },
            )?);
        }

        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let texture = compositor.render_scene_content_to_texture(
            &primitives,
            &texture_layers,
            background.unwrap_or([0, 0, 0, 0]),
        )?;
        compositor.present_texture_to_canvas(&texture, &canvas)
    }

    /// Render a strict GPU scene-tree frame to a texture.
    async fn render_scene_tree_frame_to_wgpu_texture(
        &mut self,
        graph: &GraphScript,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let Some(nodes) = scene_nodes_for_present(graph) else {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "direct GPU texture rendering needs a presentable scene tree".to_string(),
            });
        };
        let background = graph
            .backgrounds
            .last()
            .map(|background| parse_color(&background.color))
            .transpose()?
            .unwrap_or([0, 0, 0, 0]);
        self.present_gpu_scene_nodes_with_background_to_texture(
            nodes,
            graph_output_size(graph),
            graph_logical_render_size(graph),
            render_size_root_transform(graph_output_size(graph), graph_logical_render_size(graph)),
            time_norm,
            time_sec,
            Some(background),
        )
        .await
    }

    async fn render_scene_tree_pick_id_at(
        &mut self,
        graph: &GraphScript,
        time_norm: f32,
        time_sec: f32,
        x: u32,
        y: u32,
    ) -> Result<Option<u32>, MotionLoomSceneRenderError> {
        let Some(nodes) = scene_nodes_for_present(graph) else {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "renderer ID picking needs a presentable scene tree".to_string(),
            });
        };
        let output_size = graph_output_size(graph);
        let logical_size = graph_logical_render_size(graph);
        let root_transform = render_size_root_transform(output_size, logical_size);
        let texture = self
            .present_gpu_scene_nodes_pick_ids_to_texture(
                nodes,
                output_size,
                logical_size,
                root_transform,
                time_norm,
                time_sec,
            )
            .await?;
        let compositor =
            self.gpu_compositor
                .as_ref()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let rgba = compositor
            .readback_texture_pixel_rgba_async(
                texture.texture.as_ref(),
                x.min(texture.width.saturating_sub(1)),
                y.min(texture.height.saturating_sub(1)),
            )
            .await?;
        let id = u32::from(rgba[0]) | (u32::from(rgba[1]) << 8) | (u32::from(rgba[2]) << 16);
        Ok((id > 0).then_some(id))
    }

    #[allow(clippy::too_many_arguments)]
    async fn present_gpu_scene_nodes_pick_ids_to_texture(
        &mut self,
        nodes: &[SceneNode],
        output_size: (u32, u32),
        logical_size: (u32, u32),
        root_transform: Affine2,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let scaled_scene = output_size != logical_size || !affine_is_identity(root_transform);
        let scene_transform = if scaled_scene {
            root_transform
        } else {
            Affine2::identity()
        };
        let scene_canvas_size = if scaled_scene {
            logical_size
        } else {
            output_size
        };

        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        let mut assets = GpuSceneNativeAssets::default();
        let mut primitives = Vec::<GpuScenePrimitive>::new();
        let mut texture_layers = Vec::<GpuSceneTextureLayer>::new();
        let mut text_requests = Vec::<GpuSceneTextRequest>::new();
        let mut unsupported = false;
        self.collect_gpu_scene_native_commands(
            nodes,
            scene_transform,
            None,
            1.0,
            time_norm,
            time_sec,
            scene_canvas_size,
            &mut assets,
            &mut primitives,
            &mut texture_layers,
            &mut text_requests,
            &mut unsupported,
        )
        .await?;
        if unsupported {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "renderer ID picking does not support this scene node set yet".to_string(),
            });
        }

        for request in text_requests {
            texture_layers.extend(self.rasterize_text_texture_layers_gpu_effects(
                &request.node,
                TextTextureRasterContext {
                    transform: request.transform,
                    inherited_opacity: request.opacity,
                    pick_id: request.pick_id,
                    time_norm,
                    time_sec,
                    canvas_size: scene_canvas_size,
                },
            )?);
        }

        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        compositor.render_scene_pick_ids_to_texture(&primitives, &texture_layers)
    }

    /// Collect and render scene nodes to a GPU texture without presenting to a surface.
    #[allow(clippy::too_many_arguments)]
    async fn present_gpu_scene_nodes_with_background_to_texture(
        &mut self,
        nodes: &[SceneNode],
        output_size: (u32, u32),
        logical_size: (u32, u32),
        root_transform: Affine2,
        time_norm: f32,
        time_sec: f32,
        background: Option<[u8; 4]>,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let scaled_scene = output_size != logical_size || !affine_is_identity(root_transform);
        let scene_transform = if scaled_scene {
            root_transform
        } else {
            Affine2::identity()
        };
        let scene_canvas_size = if scaled_scene {
            logical_size
        } else {
            output_size
        };

        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        let retained_static = self
            .try_collect_retained_gpu_shape_scene(
                nodes,
                scene_transform,
                1.0,
                time_norm,
                time_sec,
                scene_canvas_size,
            )
            .await?;
        let retained_transform = if retained_static.is_some() {
            None
        } else {
            self.try_collect_retained_gpu_transform_scene(
                nodes,
                scene_transform,
                time_norm,
                time_sec,
                scene_canvas_size,
            )?
        };
        let mut assets = GpuSceneNativeAssets::default();
        let mut collected_primitives = Vec::<GpuScenePrimitive>::new();
        let mut texture_layers = Vec::<GpuSceneTextureLayer>::new();
        let mut text_requests = retained_transform
            .as_ref()
            .map(|frame| frame.text_requests.clone())
            .unwrap_or_default();
        let mut unsupported = false;
        if retained_static.is_none() && retained_transform.is_none() {
            self.collect_gpu_scene_native_commands(
                nodes,
                scene_transform,
                None,
                1.0,
                time_norm,
                time_sec,
                scene_canvas_size,
                &mut assets,
                &mut collected_primitives,
                &mut texture_layers,
                &mut text_requests,
                &mut unsupported,
            )
            .await?;
        }
        let primitives = retained_static
            .as_ref()
            .map(|value| value.as_slice())
            .or_else(|| {
                retained_transform
                    .as_ref()
                    .map(|frame| frame.primitives.as_slice())
            })
            .unwrap_or(&collected_primitives);
        if unsupported {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: "direct GPU texture rendering does not support this scene node set yet"
                    .to_string(),
            });
        }

        for request in text_requests {
            texture_layers.extend(self.rasterize_text_texture_layers_gpu_effects(
                &request.node,
                TextTextureRasterContext {
                    transform: request.transform,
                    inherited_opacity: request.opacity,
                    pick_id: request.pick_id,
                    time_norm,
                    time_sec,
                    canvas_size: scene_canvas_size,
                },
            )?);
        }

        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        compositor.render_scene_content_to_texture(
            primitives,
            &texture_layers,
            background.unwrap_or([0, 0, 0, 0]),
        )
    }

    async fn ensure_gpu_compositor_size(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let needs_new_compositor = self
            .gpu_compositor
            .as_ref()
            .map(|compositor| compositor.width != width || compositor.height != height)
            .unwrap_or(true);
        if needs_new_compositor {
            #[cfg(not(target_arch = "wasm32"))]
            {
                if let Some((device, queue)) = self
                    .gpu_compositor
                    .as_ref()
                    .map(|compositor| compositor.device_queue())
                    .or_else(|| self.external_device_queue.clone())
                {
                    self.gpu_compositor = Some(
                        WgpuSceneCompositor::new_with_device(
                            device,
                            queue,
                            width,
                            height,
                            self.asset_resolver.clone(),
                        )
                        .await?,
                    );
                } else {
                    self.gpu_compositor = Some(
                        WgpuSceneCompositor::new(width, height, self.asset_resolver.clone())
                            .await?,
                    );
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.gpu_compositor = Some(
                    WgpuSceneCompositor::new(width, height, self.asset_resolver.clone()).await?,
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_render_gpu_scene_nodes_composited(
        &mut self,
        nodes: &[SceneNode],
        output_size: (u32, u32),
        logical_size: (u32, u32),
        root_transform: Affine2,
        time_norm: f32,
        time_sec: f32,
        background: Option<[u8; 4]>,
    ) -> Result<Option<GraphTextureSource>, MotionLoomSceneRenderError> {
        let scaled_scene = output_size != logical_size || !affine_is_identity(root_transform);
        let scene_transform = if scaled_scene {
            root_transform
        } else {
            Affine2::identity()
        };
        let scene_canvas_size = if scaled_scene {
            logical_size
        } else {
            output_size
        };
        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        let mut assets = GpuSceneNativeAssets::default();
        let mut primitives = Vec::<GpuScenePrimitive>::new();
        let mut texture_layers = Vec::<GpuSceneTextureLayer>::new();
        let mut text_requests = Vec::<GpuSceneTextRequest>::new();
        let mut unsupported = false;
        self.collect_gpu_scene_native_commands(
            nodes,
            scene_transform,
            None,
            1.0,
            time_norm,
            time_sec,
            scene_canvas_size,
            &mut assets,
            &mut primitives,
            &mut texture_layers,
            &mut text_requests,
            &mut unsupported,
        )
        .await?;
        if unsupported {
            return Ok(None);
        }

        for request in text_requests {
            texture_layers.extend(self.rasterize_text_texture_layers_gpu_effects(
                &request.node,
                TextTextureRasterContext {
                    transform: request.transform,
                    inherited_opacity: request.opacity,
                    pick_id: request.pick_id,
                    time_norm,
                    time_sec,
                    canvas_size: scene_canvas_size,
                },
            )?);
        }

        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let texture = compositor.render_scene_content_to_texture(
            &primitives,
            &texture_layers,
            background.unwrap_or([0, 0, 0, 0]),
        )?;
        Ok(Some(GraphTextureSource::Gpu(texture)))
    }

    #[allow(clippy::too_many_arguments)]
    async fn render_gpu_scene_texture_from_nodes(
        &mut self,
        nodes: &[SceneNode],
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
        assets: &mut GpuSceneNativeAssets,
    ) -> Result<Option<GpuSceneNativeTexture>, MotionLoomSceneRenderError> {
        self.ensure_gpu_compositor_size(canvas_size.0.max(1), canvas_size.1.max(1))
            .await?;
        let mut primitives = Vec::<GpuScenePrimitive>::new();
        let mut texture_layers = Vec::<GpuSceneTextureLayer>::new();
        let mut text_requests = Vec::<GpuSceneTextRequest>::new();
        let mut unsupported = false;
        self.collect_gpu_scene_native_commands(
            nodes,
            transform,
            None,
            inherited_opacity,
            time_norm,
            time_sec,
            canvas_size,
            assets,
            &mut primitives,
            &mut texture_layers,
            &mut text_requests,
            &mut unsupported,
        )
        .await?;
        if unsupported {
            return Ok(None);
        }
        for request in text_requests {
            texture_layers.extend(self.rasterize_text_texture_layers_gpu_effects(
                &request.node,
                TextTextureRasterContext {
                    transform: request.transform,
                    inherited_opacity: request.opacity,
                    pick_id: request.pick_id,
                    time_norm,
                    time_sec,
                    canvas_size,
                },
            )?);
        }
        self.ensure_gpu_compositor_size(canvas_size.0.max(1), canvas_size.1.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        compositor
            .render_scene_content_to_texture(&primitives, &texture_layers, [0, 0, 0, 0])
            .map(Some)
    }

    #[allow(clippy::too_many_arguments)]
    async fn render_retained_gpu_screen_layer_texture(
        &mut self,
        nodes: &[SceneNode],
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
        scope: u64,
    ) -> Result<Option<GpuSceneNativeTexture>, MotionLoomSceneRenderError> {
        let Some(retained) = self.try_collect_retained_gpu_transform_scene_scoped(
            nodes,
            Affine2::identity(),
            time_norm,
            time_sec,
            canvas_size,
            scope,
        )?
        else {
            return Ok(None);
        };

        let mut texture_layers = Vec::new();
        for request in retained.text_requests {
            texture_layers.extend(self.rasterize_text_texture_layers_gpu_effects(
                &request.node,
                TextTextureRasterContext {
                    transform: request.transform,
                    inherited_opacity: request.opacity,
                    pick_id: request.pick_id,
                    time_norm,
                    time_sec,
                    canvas_size,
                },
            )?);
        }
        self.ensure_gpu_compositor_size(canvas_size.0.max(1), canvas_size.1.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        compositor
            .render_scene_content_to_texture(
                retained.primitives.as_ref(),
                &texture_layers,
                [0, 0, 0, 0],
            )
            .map(Some)
    }

    async fn render_gpu_node_matte_texture(
        &mut self,
        id: &str,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
        assets: &mut GpuSceneNativeAssets,
    ) -> Result<Option<GpuSceneNativeTexture>, MotionLoomSceneRenderError> {
        if let Some(mask) = assets.masks.get(id).cloned() {
            return Ok(Some(mask));
        }
        if let Some(precompose) = self.scene_precompose_defs.get(id).cloned() {
            let texture = self
                .render_gpu_scene_texture_from_nodes(
                    &precompose.children,
                    Affine2::identity(),
                    1.0,
                    time_norm,
                    time_sec,
                    precompose.size.unwrap_or(canvas_size),
                    assets,
                )
                .await?;
            return self.copy_gpu_native_texture_owned_optional(
                texture,
                "anica-motionloom-scene-mask-from-precompose-owned-copy",
            );
        }
        let Some(node) = self.scene_node_defs.get(id).cloned() else {
            return Ok(None);
        };
        let texture = self
            .render_gpu_scene_texture_from_nodes(
                std::slice::from_ref(&node),
                Affine2::identity(),
                1.0,
                time_norm,
                time_sec,
                canvas_size,
                assets,
            )
            .await?;
        self.copy_gpu_native_texture_owned_optional(
            texture,
            "anica-motionloom-scene-mask-from-node-owned-copy",
        )
    }

    async fn render_gpu_precompose_instance(
        &mut self,
        source_id: &str,
        layer: &SceneLayerNode,
        canvas_size: (u32, u32),
        assets: &mut GpuSceneNativeAssets,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<Option<GpuSceneNativeTexture>, MotionLoomSceneRenderError> {
        let Some(precompose) = self.scene_precompose_defs.get(source_id).cloned() else {
            return Ok(assets.precomposes.get(source_id).cloned());
        };
        let size = precompose.size.unwrap_or(canvas_size);
        if size != canvas_size {
            return Ok(None);
        }
        let Some((source_norm, source_sec)) =
            scene_layer_source_time(layer, &precompose, time_norm, time_sec)?
        else {
            return Ok(None);
        };
        let texture = self
            .render_gpu_scene_texture_from_nodes(
                &precompose.children,
                Affine2::identity(),
                1.0,
                source_norm,
                source_sec,
                canvas_size,
                assets,
            )
            .await?;
        self.copy_gpu_native_texture_owned_optional(
            texture,
            "anica-motionloom-scene-precompose-owned-copy",
        )
    }

    fn copy_gpu_native_texture_owned_optional(
        &mut self,
        texture: Option<GpuSceneNativeTexture>,
        label: &'static str,
    ) -> Result<Option<GpuSceneNativeTexture>, MotionLoomSceneRenderError> {
        let Some(texture) = texture else {
            return Ok(None);
        };
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        Ok(Some(
            compositor.copy_gpu_native_texture_owned(&texture, label),
        ))
    }

    fn apply_gpu_scene_filter_texture(
        &mut self,
        input: GpuSceneNativeTexture,
        filter_id: &str,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let Some(filter) = self.filter_defs.get(filter_id).cloned() else {
            return Ok(input);
        };
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let mut output = input;
        for step in filter.steps {
            let kind = step.kind.trim().to_ascii_lowercase();
            if kind == "blur"
                || kind == "gaussian_blur"
                || kind == "gaussian-blur"
                || kind == "gaussian_5tap_blur"
            {
                let sigma = step
                    .radius
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(2.0)
                    .clamp(0.0, 64.0);
                output =
                    compositor.apply_gpu_blur_texture(&output, &[(true, sigma), (false, sigma)])?;
                continue;
            }
            if kind == "colormatrix" || kind == "color_matrix" || kind == "color-matrix" {
                let brightness = step
                    .brightness
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(1.0)
                    .clamp(0.0, 4.0)
                    - 1.0;
                let contrast = step
                    .contrast
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(1.0)
                    .clamp(0.0, 4.0);
                let saturation = step
                    .saturation
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(1.0)
                    .clamp(0.0, 4.0);
                output = compositor
                    .apply_gpu_color_texture(&output, brightness, contrast, saturation)?;
                continue;
            }
            if kind == "opacity" {
                let opacity = step
                    .opacity
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(1.0)
                    .clamp(0.0, 1.0);
                output = compositor.apply_gpu_opacity_texture(&output, opacity)?;
                continue;
            }
            if kind == "edgesoftness" || kind == "edge_softness" || kind == "edge-softness" {
                let radius = step
                    .radius
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(2.0)
                    .clamp(0.0, 32.0);
                let amount = step
                    .amount
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(0.65)
                    .clamp(0.0, 1.0);
                let preserve = step
                    .preserve_interior
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(0.85)
                    .clamp(0.0, 1.0);
                output = compositor.apply_gpu_edge_treatment_texture(
                    &output, 10.0, radius, amount, 1.0, preserve,
                )?;
                continue;
            }
            if kind == "edgeroughness" || kind == "edge_roughness" || kind == "edge-roughness" {
                let radius = step
                    .radius
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(1.5)
                    .clamp(0.0, 16.0);
                let amount = step
                    .amount
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(0.75)
                    .clamp(0.0, 1.0);
                let scale = step
                    .scale
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(22.0)
                    .max(0.001);
                let seed = step
                    .seed
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(0.0);
                output = compositor
                    .apply_gpu_edge_treatment_texture(&output, 11.0, radius, amount, scale, seed)?;
                continue;
            }
            if kind == "colorbleed" || kind == "color_bleed" || kind == "color-bleed" {
                let radius = step
                    .radius
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(3.0)
                    .clamp(0.0, 32.0);
                let amount = step
                    .amount
                    .as_deref()
                    .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(0.18)
                    .clamp(0.0, 1.0);
                output = compositor
                    .apply_gpu_edge_treatment_texture(&output, 12.0, radius, amount, 1.0, 0.0)?;
                continue;
            }
            if kind == "lightstreak" || kind == "light_streak" || kind == "light-streak" {
                let length = step
                    .radius
                    .as_deref()
                    .map(|v| eval_scene_number(v, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(24.0)
                    .clamp(1.0, 64.0);
                let amount = step
                    .amount
                    .as_deref()
                    .map(|v| eval_scene_number(v, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(0.7)
                    .max(0.0);
                let angle = step
                    .scale
                    .as_deref()
                    .map(|v| eval_scene_number(v, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(-18.0);
                let threshold = step
                    .seed
                    .as_deref()
                    .map(|v| eval_scene_number(v, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(0.72)
                    .clamp(0.0, 0.999);
                output = compositor.apply_gpu_edge_treatment_texture(
                    &output, 13.0, length, amount, angle, threshold,
                )?;
                continue;
            }
            if kind == "chromaticaberration"
                || kind == "chromatic_aberration"
                || kind == "chromatic-aberration"
            {
                let offset = step
                    .radius
                    .as_deref()
                    .map(|v| eval_scene_number(v, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(3.0)
                    .max(0.0);
                let amount = step
                    .amount
                    .as_deref()
                    .map(|v| eval_scene_number(v, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(1.0)
                    .max(0.0);
                output = compositor
                    .apply_gpu_edge_treatment_texture(&output, 14.0, offset, amount, 1.0, 0.0)?;
                continue;
            }
            if kind == "highlightcompression"
                || kind == "highlight_compression"
                || kind == "highlight-compression"
            {
                let threshold = step
                    .scale
                    .as_deref()
                    .map(|v| eval_scene_number(v, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(0.72)
                    .clamp(0.0, 0.999);
                let amount = step
                    .amount
                    .as_deref()
                    .map(|v| eval_scene_number(v, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(0.65)
                    .max(0.0);
                output = compositor
                    .apply_gpu_edge_treatment_texture(&output, 15.0, threshold, amount, 1.0, 0.0)?;
                continue;
            }
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "Filter '{}' step '{}' is not GPU-native yet",
                    filter_id, step.kind
                ),
            });
        }
        Ok(output)
    }

    fn apply_gpu_material_texture(
        &mut self,
        mut input: GpuSceneNativeTexture,
        material_id: &str,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        let material = self
            .material_defs
            .get(material_id)
            .cloned()
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: format!("unknown material '{material_id}'"),
            })?;
        let roughness = eval_scene_number(&material.roughness, time_norm, time_sec)?;
        let specular = eval_scene_number(&material.specular, time_norm, time_sec)?;
        let displacement_noise = material
            .displacement
            .as_ref()
            .and_then(|id| self.noise_defs.get(id))
            .cloned();
        if let Some(noise) = displacement_noise.as_ref() {
            let scale = eval_scene_number(&noise.scale, time_norm, time_sec)?;
            let seed = eval_scene_number(&noise.seed, time_norm, time_sec)?
                + eval_scene_number(&noise.evolution, time_norm, time_sec)?;
            let amount = eval_scene_number(&material.displacement_amount, time_norm, time_sec)?;
            input = self
                .gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?
                .apply_gpu_material_displacement_texture(
                    &input,
                    TextureOverlayKind::parse(&noise.kind).id(),
                    scale,
                    amount,
                    seed,
                    roughness,
                    specular,
                )?;
        }
        let texture_noise = material
            .texture
            .as_ref()
            .and_then(|id| self.noise_defs.get(id))
            .cloned();
        if let Some(noise) = texture_noise.as_ref() {
            let scale = eval_scene_number(&noise.scale, time_norm, time_sec)?;
            let seed = eval_scene_number(&noise.seed, time_norm, time_sec)?
                + eval_scene_number(&noise.evolution, time_norm, time_sec)?;
            let strength = eval_scene_number(&material.texture_amount, time_norm, time_sec)?;
            let contrast = eval_scene_number(&noise.contrast, time_norm, time_sec)?;
            input = self
                .gpu_compositor
                .as_mut()
                .unwrap()
                .apply_gpu_texture_overlay_texture(
                    &input,
                    SceneTextureOverlayParams {
                        kind: TextureOverlayKind::parse(&noise.kind),
                        texture_ref: None,
                        height_ref: None,
                        texture_src: None,
                        height_src: None,
                        scale,
                        strength,
                        contrast,
                        seed,
                        brush_angle: 0.0,
                        bump_strength: specular,
                        relief: 1.0 - roughness,
                    },
                )?;
        }
        let opacity = eval_scene_number(&material.opacity, time_norm, time_sec)?.clamp(0.0, 1.0);
        if opacity < 0.999 {
            input = self
                .gpu_compositor
                .as_mut()
                .unwrap()
                .apply_gpu_opacity_texture(&input, opacity)?;
        }
        let refraction = eval_scene_number(&material.refraction, time_norm, time_sec)?.max(0.0);
        let dispersion = eval_scene_number(&material.dispersion, time_norm, time_sec)?.max(0.0);
        let glass = eval_scene_number(&material.glass, time_norm, time_sec)?.clamp(0.0, 1.0);
        if refraction > 0.001 || dispersion > 0.001 || glass > 0.001 {
            input = self
                .gpu_compositor
                .as_mut()
                .unwrap()
                .apply_gpu_edge_treatment_texture(
                    &input, 16.0, refraction, dispersion, glass, 0.0,
                )?;
        }
        Ok(input)
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_gpu_scene_native_commands<'a>(
        &'a mut self,
        nodes: &'a [SceneNode],
        transform: Affine2,
        deform: Option<&'a EvaluatedDeformGrid>,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
        assets: &'a mut GpuSceneNativeAssets,
        primitives: &'a mut Vec<GpuScenePrimitive>,
        texture_layers: &'a mut Vec<GpuSceneTextureLayer>,
        text_requests: &'a mut Vec<GpuSceneTextRequest>,
        unsupported: &'a mut bool,
    ) -> Pin<Box<dyn Future<Output = Result<(), MotionLoomSceneRenderError>> + 'a>> {
        self.collect_gpu_scene_native_commands_with_depth(
            nodes,
            transform,
            deform,
            inherited_opacity,
            time_norm,
            time_sec,
            canvas_size,
            assets,
            primitives,
            texture_layers,
            text_requests,
            unsupported,
            None,
        )
    }

    /// Return a retained primitive list for a time-invariant, GPU-native shape scene.
    ///
    /// This cache deliberately sits above Path parsing and scene traversal. The
    /// per-Path cache avoids rebuilding geometry, but still requires walking the
    /// complete tree and cloning every primitive. A retained scene hit skips that
    /// work entirely. Mixed scenes remain on the regular collector so text,
    /// textures, masks, materials, and other effect ordering stay unchanged.
    #[allow(clippy::too_many_arguments)]
    async fn try_collect_retained_gpu_shape_scene(
        &mut self,
        nodes: &[SceneNode],
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
    ) -> Result<Option<Arc<Vec<GpuScenePrimitive>>>, MotionLoomSceneRenderError> {
        if (inherited_opacity - 1.0).abs() > 0.0001 {
            return Ok(None);
        }
        let key = RetainedGpuShapeSceneKey::new(
            self.prepared_graph_signature,
            nodes,
            transform,
            canvas_size,
        );
        if let Some(scene) = self.retained_gpu_shape_scenes.get(&key) {
            self.gpu_path_cache_hits = self.gpu_path_cache_hits.saturating_add(scene.path_count);
            return Ok(Some(scene.primitives.clone()));
        }
        if !vector_overlay_nodes_are_static(nodes) {
            return Ok(None);
        }

        let mut assets = GpuSceneNativeAssets::default();
        let mut primitives = Vec::new();
        let mut texture_layers = Vec::new();
        let mut text_requests = Vec::new();
        let mut unsupported = false;
        let paths_before = self
            .gpu_path_cache_hits
            .saturating_add(self.gpu_path_cache_misses);
        self.collect_gpu_scene_native_commands(
            nodes,
            transform,
            None,
            inherited_opacity,
            time_norm,
            time_sec,
            canvas_size,
            &mut assets,
            &mut primitives,
            &mut texture_layers,
            &mut text_requests,
            &mut unsupported,
        )
        .await?;
        if unsupported || !texture_layers.is_empty() || !text_requests.is_empty() {
            return Ok(None);
        }

        let path_count = self
            .gpu_path_cache_hits
            .saturating_add(self.gpu_path_cache_misses)
            .saturating_sub(paths_before);
        let primitives = Arc::new(primitives);
        if self.retained_gpu_shape_scenes.len() >= RETAINED_GPU_SHAPE_SCENE_CACHE_LIMIT {
            self.retained_gpu_shape_scenes.clear();
        }
        self.retained_gpu_shape_scenes.insert(
            key,
            CachedRetainedGpuShapeScene {
                primitives: primitives.clone(),
                path_count,
            },
        );
        Ok(Some(primitives))
    }

    fn compile_retained_transform_nodes(
        &mut self,
        nodes: &[SceneNode],
        ancestors: &mut Vec<RetainedTransformAncestor>,
        time_norm: f32,
        time_sec: f32,
        output: &mut RetainedCompileOutput,
    ) -> Result<bool, MotionLoomSceneRenderError> {
        for node in nodes {
            match node {
                SceneNode::Defs(_) | SceneNode::Palette(_) => {}
                SceneNode::Timeline(timeline) => {
                    if timeline.children.len() != 1
                        || !self.compile_retained_transform_nodes(
                            &timeline.children,
                            ancestors,
                            time_norm,
                            time_sec,
                            output,
                        )?
                    {
                        return Ok(false);
                    }
                }
                SceneNode::Track(track) => {
                    if !(track.space.eq_ignore_ascii_case("screen")
                        || track.space.eq_ignore_ascii_case("world"))
                        || !self.compile_retained_transform_nodes(
                            &track.children,
                            ancestors,
                            time_norm,
                            time_sec,
                            output,
                        )?
                    {
                        return Ok(false);
                    }
                }
                SceneNode::Sequence(sequence) => {
                    if sequence.from_ms != 0
                        || !sequence.out.eq_ignore_ascii_case("hold")
                        || !self.compile_retained_transform_nodes(
                            &sequence.children,
                            ancestors,
                            time_norm,
                            time_sec,
                            output,
                        )?
                    {
                        return Ok(false);
                    }
                }
                SceneNode::Layer(layer) => {
                    if layer.source.is_some()
                        || layer.is_3d
                        || layer.effect.is_some()
                        || layer.mask.is_some()
                        || layer.mask_from.is_some()
                        || layer.matte.is_some()
                        || layer.matte_from.is_some()
                        || !layer.blend.eq_ignore_ascii_case("normal")
                    {
                        return Ok(false);
                    }
                    ancestors.push(RetainedTransformAncestor::Transform(
                        RetainedCompiledTransform::compile([
                            &layer.x,
                            &layer.y,
                            &layer.rotation,
                            &layer.scale,
                            &layer.scale_x,
                            &layer.scale_y,
                            &layer.skew_x,
                            &layer.skew_y,
                            &layer.transform_origin_x,
                            &layer.transform_origin_y,
                            &layer.opacity,
                        ])?,
                    ));
                    let supported = self.compile_retained_transform_nodes(
                        &layer.children,
                        ancestors,
                        time_norm,
                        time_sec,
                        output,
                    )?;
                    ancestors.pop();
                    if !supported {
                        return Ok(false);
                    }
                }
                SceneNode::Group(group) => {
                    if group.deform_grid.is_some()
                        || group.grid_from.is_some()
                        || group.grid_to.is_some()
                        || group.mask.is_some()
                        || group.mask_from.is_some()
                        || !group.effects.is_empty()
                        || group.material.is_some()
                    {
                        return Ok(false);
                    }
                    ancestors.push(RetainedTransformAncestor::Transform(
                        RetainedCompiledTransform::compile([
                            &group.x,
                            &group.y,
                            &group.rotation,
                            &group.scale,
                            &group.scale_x,
                            &group.scale_y,
                            &group.skew_x,
                            &group.skew_y,
                            &group.transform_origin_x,
                            &group.transform_origin_y,
                            &group.opacity,
                        ])?,
                    ));
                    let supported = self.compile_retained_transform_nodes(
                        &group.children,
                        ancestors,
                        time_norm,
                        time_sec,
                        output,
                    )?;
                    ancestors.pop();
                    if !supported {
                        return Ok(false);
                    }
                }
                SceneNode::Path(path) => {
                    if path_requires_cpu_overlay(path) || path.material.is_some() {
                        return Ok(false);
                    }
                    let start = output.primitives.len();
                    self.push_gpu_path_commands_cached(
                        path,
                        Affine2::identity(),
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &mut output.primitives,
                    )?;
                    let dependency = if retained_path_geometry_and_style_are_static(path) {
                        RetainedExpressionDependency::Static
                    } else {
                        RetainedExpressionDependency::for_node(node)
                    };
                    output.entries.push(RetainedTransformPrimitiveEntry {
                        primitive_range: start..output.primitives.len(),
                        ancestors: ancestors.clone(),
                        source: node.clone(),
                        dependency,
                        last_dependency_token: dependency.token(time_sec),
                    });
                }
                SceneNode::Rect(rect) if !rect_requires_cpu_overlay(rect) => {
                    let start = output.primitives.len();
                    push_gpu_rect_commands(
                        rect,
                        Affine2::identity(),
                        None,
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &self.gradient_defs,
                        &mut output.primitives,
                    )?;
                    output.entries.push(RetainedTransformPrimitiveEntry {
                        primitive_range: start..output.primitives.len(),
                        ancestors: ancestors.clone(),
                        source: node.clone(),
                        dependency: RetainedExpressionDependency::for_node(node),
                        last_dependency_token: RetainedExpressionDependency::for_node(node)
                            .token(time_sec),
                    });
                }
                SceneNode::Circle(circle) if !circle_requires_cpu_overlay(circle) => {
                    let start = output.primitives.len();
                    push_gpu_circle_commands(
                        circle,
                        Affine2::identity(),
                        None,
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &self.gradient_defs,
                        &mut output.primitives,
                    )?;
                    let dependency = RetainedExpressionDependency::for_node(node);
                    output.entries.push(RetainedTransformPrimitiveEntry {
                        primitive_range: start..output.primitives.len(),
                        ancestors: ancestors.clone(),
                        source: node.clone(),
                        dependency,
                        last_dependency_token: dependency.token(time_sec),
                    });
                }
                SceneNode::Ellipse(ellipse) => {
                    let start = output.primitives.len();
                    push_gpu_ellipse_commands(
                        ellipse,
                        Affine2::identity(),
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &self.gradient_defs,
                        &mut output.primitives,
                    )?;
                    let dependency = RetainedExpressionDependency::for_node(node);
                    output.entries.push(RetainedTransformPrimitiveEntry {
                        primitive_range: start..output.primitives.len(),
                        ancestors: ancestors.clone(),
                        source: node.clone(),
                        dependency,
                        last_dependency_token: dependency.token(time_sec),
                    });
                }
                SceneNode::Line(line) if !line_requires_cpu_overlay(line) => {
                    let start = output.primitives.len();
                    push_gpu_line_command(
                        line,
                        Affine2::identity(),
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &self.gradient_defs,
                        &mut output.primitives,
                    )?;
                    let dependency = RetainedExpressionDependency::for_node(node);
                    output.entries.push(RetainedTransformPrimitiveEntry {
                        primitive_range: start..output.primitives.len(),
                        ancestors: ancestors.clone(),
                        source: node.clone(),
                        dependency,
                        last_dependency_token: dependency.token(time_sec),
                    });
                }
                SceneNode::Text(text)
                    if text.animators.is_empty() && text.visible_chars.is_none() =>
                {
                    output.text_entries.push(RetainedTransformTextEntry {
                        node: text.as_ref().clone(),
                        ancestors: ancestors.clone(),
                    });
                }
                SceneNode::Repeat(repeat) => {
                    let count_debug = repeat.count.to_ascii_lowercase();
                    if count_debug.contains("$time")
                        || count_debug.contains("$frame")
                        || count_debug.contains("curve(")
                    {
                        return Ok(false);
                    }
                    let count = eval_repeat_count(&repeat.count, time_norm, time_sec)?;
                    let compiled = RetainedCompiledRepeat::compile(repeat)?;
                    for index in 0..count {
                        ancestors.push(RetainedTransformAncestor::Repeat(
                            compiled.clone(),
                            index as usize,
                        ));
                        let supported = self.compile_retained_transform_nodes(
                            &repeat.children,
                            ancestors,
                            time_norm,
                            time_sec,
                            output,
                        )?;
                        ancestors.pop();
                        if !supported {
                            return Ok(false);
                        }
                    }
                }
                _ => return Ok(false),
            }
        }
        Ok(true)
    }

    fn try_collect_retained_gpu_transform_scene(
        &mut self,
        nodes: &[SceneNode],
        root_transform: Affine2,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
    ) -> Result<Option<RetainedGpuTransformFrame>, MotionLoomSceneRenderError> {
        self.try_collect_retained_gpu_transform_scene_scoped(
            nodes,
            root_transform,
            time_norm,
            time_sec,
            canvas_size,
            0,
        )
    }

    fn try_collect_retained_gpu_transform_scene_scoped(
        &mut self,
        nodes: &[SceneNode],
        root_transform: Affine2,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
        scope: u64,
    ) -> Result<Option<RetainedGpuTransformFrame>, MotionLoomSceneRenderError> {
        let key = RetainedGpuShapeSceneKey::new_scoped(
            self.prepared_graph_signature,
            scope,
            nodes,
            root_transform,
            canvas_size,
        );
        let compiled_now = !self.retained_gpu_transform_scenes.contains_key(&key);
        if compiled_now {
            let mut output = RetainedCompileOutput::default();
            if !self.compile_retained_transform_nodes(
                nodes,
                &mut Vec::new(),
                time_norm,
                time_sec,
                &mut output,
            )? || (output.entries.is_empty() && output.text_entries.is_empty())
            {
                return Ok(None);
            }
            let base_transforms = output
                .primitives
                .iter()
                .map(|primitive| primitive.transform)
                .collect();
            let base_opacities = output
                .primitives
                .iter()
                .map(|primitive| primitive.opacity)
                .collect();
            let path_count = output
                .entries
                .iter()
                .filter(|entry| matches!(entry.source, SceneNode::Path(_)))
                .count();
            if self.retained_gpu_transform_scenes.len() >= RETAINED_GPU_TRANSFORM_SCENE_CACHE_LIMIT
            {
                self.retained_gpu_transform_scenes.clear();
            }
            self.retained_gpu_transform_scenes.insert(
                key,
                CachedRetainedGpuTransformScene {
                    primitives: Arc::new(output.primitives),
                    base_transforms,
                    base_opacities,
                    entries: output.entries,
                    text_entries: output.text_entries,
                    path_count,
                },
            );
        }

        let mut scene = self
            .retained_gpu_transform_scenes
            .remove(&key)
            .expect("retained transform scene inserted before update");
        let primitives = Arc::make_mut(&mut scene.primitives);
        let mut entry_ix = 0usize;
        let mut refreshed_path_count = 0usize;
        while entry_ix < scene.entries.len() {
            let dependency = scene.entries[entry_ix].dependency;
            let dependency_token = dependency.token(time_sec);
            let refresh = matches!(dependency, RetainedExpressionDependency::Frame)
                || dependency_token != scene.entries[entry_ix].last_dependency_token;
            if refresh {
                if matches!(scene.entries[entry_ix].source, SceneNode::Path(_)) {
                    refreshed_path_count = refreshed_path_count.saturating_add(1);
                }
                let mut refreshed = Vec::new();
                match &scene.entries[entry_ix].source {
                    SceneNode::Rect(rect) => push_gpu_rect_commands(
                        rect,
                        Affine2::identity(),
                        None,
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &self.gradient_defs,
                        &mut refreshed,
                    )?,
                    SceneNode::Circle(circle) => push_gpu_circle_commands(
                        circle,
                        Affine2::identity(),
                        None,
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &self.gradient_defs,
                        &mut refreshed,
                    )?,
                    SceneNode::Ellipse(ellipse) => push_gpu_ellipse_commands(
                        ellipse,
                        Affine2::identity(),
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &self.gradient_defs,
                        &mut refreshed,
                    )?,
                    SceneNode::Line(line) => push_gpu_line_command(
                        line,
                        Affine2::identity(),
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &self.gradient_defs,
                        &mut refreshed,
                    )?,
                    SceneNode::Path(path) => self.push_gpu_path_commands_cached(
                        path,
                        Affine2::identity(),
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &mut refreshed,
                    )?,
                    _ => {}
                }
                let old_range = scene.entries[entry_ix].primitive_range.clone();
                let old_len = old_range.len();
                let new_len = refreshed.len();
                let refreshed_transforms = refreshed
                    .iter()
                    .map(|primitive| primitive.transform)
                    .collect::<Vec<_>>();
                let refreshed_opacities = refreshed
                    .iter()
                    .map(|primitive| primitive.opacity)
                    .collect::<Vec<_>>();
                primitives.splice(old_range.clone(), refreshed);
                scene
                    .base_transforms
                    .splice(old_range.clone(), refreshed_transforms);
                scene
                    .base_opacities
                    .splice(old_range.clone(), refreshed_opacities);
                scene.entries[entry_ix].primitive_range =
                    old_range.start..old_range.start + new_len;
                if new_len != old_len {
                    let delta = new_len as isize - old_len as isize;
                    for later in &mut scene.entries[entry_ix + 1..] {
                        later.primitive_range = ((later.primitive_range.start as isize + delta)
                            as usize)
                            ..((later.primitive_range.end as isize + delta) as usize);
                    }
                }
                scene.entries[entry_ix].last_dependency_token = dependency_token;
            }
            let entry = &scene.entries[entry_ix];
            let mut transform = root_transform;
            let mut opacity = 1.0_f32;
            for ancestor in &entry.ancestors {
                match ancestor {
                    RetainedTransformAncestor::Transform(compiled) => {
                        let (local_transform, local_opacity) =
                            compiled.evaluate(time_norm, time_sec)?;
                        opacity *= local_opacity;
                        transform = transform.mul(local_transform);
                    }
                    RetainedTransformAncestor::Repeat(compiled, index) => {
                        let (local_transform, local_opacity) =
                            compiled.evaluate(*index, time_norm, time_sec)?;
                        opacity *= local_opacity;
                        transform = transform.mul(local_transform);
                    }
                }
            }
            for primitive_ix in entry.primitive_range.clone() {
                primitives[primitive_ix].transform =
                    transform.mul(scene.base_transforms[primitive_ix]);
                primitives[primitive_ix].opacity =
                    (scene.base_opacities[primitive_ix] * opacity).clamp(0.0, 1.0);
            }
            entry_ix += 1;
        }
        let mut text_requests = Vec::with_capacity(scene.text_entries.len());
        for entry in &scene.text_entries {
            let mut transform = root_transform;
            let mut opacity = 1.0_f32;
            for ancestor in &entry.ancestors {
                let (local_transform, local_opacity) = match ancestor {
                    RetainedTransformAncestor::Transform(compiled) => {
                        compiled.evaluate(time_norm, time_sec)?
                    }
                    RetainedTransformAncestor::Repeat(compiled, index) => {
                        compiled.evaluate(*index, time_norm, time_sec)?
                    }
                };
                transform = transform.mul(local_transform);
                opacity *= local_opacity;
            }
            if opacity > 0.0001 {
                text_requests.push(GpuSceneTextRequest {
                    node: entry.node.clone(),
                    transform,
                    opacity,
                    pick_id: 0,
                });
            }
        }
        if !compiled_now {
            self.gpu_path_cache_hits = self
                .gpu_path_cache_hits
                .saturating_add(scene.path_count.saturating_sub(refreshed_path_count));
        }
        let frame = RetainedGpuTransformFrame {
            primitives: scene.primitives.clone(),
            text_requests,
        };
        self.retained_gpu_transform_scenes.insert(key, scene);
        Ok(Some(frame))
    }

    #[allow(clippy::too_many_arguments)]
    fn push_gpu_path_commands_cached(
        &mut self,
        path: &PathNode,
        parent_transform: Affine2,
        deform: Option<&EvaluatedDeformGrid>,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        primitives: &mut Vec<GpuScenePrimitive>,
    ) -> Result<(), MotionLoomSceneRenderError> {
        if deform.is_some() || !gpu_path_style_is_cacheable(path) {
            return push_gpu_path_commands(
                path,
                parent_transform,
                deform,
                inherited_opacity,
                time_norm,
                time_sec,
                &self.gradient_defs,
                primitives,
            );
        }

        let evaluated_d = eval_path_d(&path.d, time_norm, time_sec)?;
        let signature = gpu_path_geometry_signature(path, evaluated_d.as_ref());
        let slot = path
            .id
            .as_deref()
            .map(|id| format!("id:{id}"))
            .unwrap_or_else(|| format!("shape:{signature:016x}"));

        let cached = self
            .gpu_path_geometry_cache
            .get(&slot)
            .filter(|entry| entry.signature == signature)
            .map(|entry| entry.primitives.clone());
        let local_primitives = if let Some(cached) = cached {
            self.gpu_path_cache_hits = self.gpu_path_cache_hits.saturating_add(1);
            cached
        } else {
            self.gpu_path_cache_misses = self.gpu_path_cache_misses.saturating_add(1);
            let mut local_path = path.clone();
            local_path.id = None;
            local_path.x = "0".to_string();
            local_path.y = "0".to_string();
            local_path.rotation = "0".to_string();
            local_path.scale = "1".to_string();
            local_path.scale_x = "1".to_string();
            local_path.scale_y = "1".to_string();
            local_path.skew_x = "0".to_string();
            local_path.skew_y = "0".to_string();
            local_path.transform_origin_x = "0".to_string();
            local_path.transform_origin_y = "0".to_string();
            local_path.d = evaluated_d.into_owned();
            let mut generated = Vec::new();
            push_gpu_path_commands(
                &local_path,
                Affine2::identity(),
                None,
                1.0,
                time_norm,
                time_sec,
                &self.gradient_defs,
                &mut generated,
            )?;
            let generated = Arc::new(generated);
            if self.gpu_path_geometry_cache.len() >= GPU_PATH_GEOMETRY_CACHE_LIMIT
                && !self.gpu_path_geometry_cache.contains_key(&slot)
            {
                self.gpu_path_geometry_cache.clear();
            }
            self.gpu_path_geometry_cache.insert(
                slot,
                CachedGpuPathGeometry {
                    signature,
                    primitives: generated.clone(),
                },
            );
            generated
        };

        let path_transform =
            parent_transform.mul(scene_path_local_transform(path, time_norm, time_sec)?);
        for primitive in local_primitives.iter() {
            let mut primitive = primitive.clone();
            primitive.transform = path_transform.mul(primitive.transform);
            primitive.opacity = (primitive.opacity * inherited_opacity).clamp(0.0, 1.0);
            primitives.push(primitive);
        }
        Ok(())
    }

    /// Lower a typed Scene `<CompositeGroup space="3d">` into the mature GLB
    /// renderer, then return that transparent 3D island as a regular Scene
    /// texture. The island is composited by the same Scene pass ordering as
    /// every 2D Layer, so this bridge does not reintroduce a public `<World>`
    /// render family.
    async fn render_scene_3d_composite(
        &mut self,
        composite: &CompositeGroupConfig,
        frame: u32,
        fps: f32,
        duration_ms: u64,
        canvas_size: (u32, u32),
        time_norm: f32,
        time_sec: f32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let mut camera = WorldCamera::default();
        let mut actors = Vec::new();

        for node in &composite.nodes_3d {
            match node {
                Scene3DNode::Camera(node) => {
                    let position =
                        eval_scene_vec3(&node.position, time_norm, time_sec, [0.0, 0.0, 6.0])?;
                    let target =
                        eval_scene_vec3(&node.target, time_norm, time_sec, [0.0, 0.0, 0.0])?;
                    let dx = position[0] - target[0];
                    let dy = position[1] - target[1];
                    let dz = position[2] - target[2];
                    let horizontal = (dx * dx + dz * dz).sqrt();
                    let distance = (horizontal * horizontal + dy * dy).sqrt().max(0.05);
                    let yaw = dx.atan2(dz).to_degrees();
                    let pitch = dy.atan2(horizontal.max(0.0001)).to_degrees();
                    camera = WorldCamera {
                        id: node.id.clone(),
                        control: WorldCameraControl::Orbit,
                        projection: WorldCameraProjection::Perspective,
                        target: None,
                        x: "0".to_string(),
                        y: "0".to_string(),
                        z: "0".to_string(),
                        target_x: target[0].to_string(),
                        target_y: target[1].to_string(),
                        target_z: target[2].to_string(),
                        yaw: yaw.to_string(),
                        pitch: pitch.to_string(),
                        roll: "0".to_string(),
                        distance: distance.to_string(),
                        zoom: "1".to_string(),
                        fov: eval_scene_number(&node.fov, time_norm, time_sec)?.to_string(),
                        orthographic_scale: None,
                    };
                }
                Scene3DNode::EnvironmentLight(_) => {
                    // GLB embedded materials already use the established World
                    // lighting shader. The typed light remains in compiler data
                    // for the dedicated IBL pass added by the material bridge.
                }
                Scene3DNode::Model(node) => {
                    let src = self
                        .model_asset_sources
                        .get(&node.asset)
                        .cloned()
                        .unwrap_or_else(|| node.asset.clone());
                    let mut position =
                        eval_scene_vec3(&node.position, time_norm, time_sec, [0.0, 0.0, 0.0])?;
                    if let Some(value) = &node.position_x {
                        position[0] = eval_scene_number(value, time_norm, time_sec)?;
                    }
                    if let Some(value) = &node.position_y {
                        position[1] = eval_scene_number(value, time_norm, time_sec)?;
                    }
                    if let Some(value) = &node.position_z {
                        position[2] = eval_scene_number(value, time_norm, time_sec)?;
                    }
                    let mut rotation =
                        eval_scene_vec3(&node.rotation, time_norm, time_sec, [0.0, 0.0, 0.0])?;
                    if let Some(value) = &node.rotation_x {
                        rotation[0] = eval_scene_number(value, time_norm, time_sec)?;
                    }
                    if let Some(value) = &node.rotation_y {
                        rotation[1] = eval_scene_number(value, time_norm, time_sec)?;
                    }
                    if let Some(value) = &node.rotation_z {
                        rotation[2] = eval_scene_number(value, time_norm, time_sec)?;
                    }
                    actors.push(WorldActor {
                        id: node
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("model_{}", actors.len())),
                        model: src,
                        path_style: WorldPathStyle::Relative,
                        hide_meshes: Vec::new(),
                        hide_materials: Vec::new(),
                        profile: None,
                        rig: None,
                        retarget: None,
                        x: position[0].to_string(),
                        y: position[1].to_string(),
                        z: position[2].to_string(),
                        yaw: rotation[1].to_string(),
                        pitch: rotation[0].to_string(),
                        roll: rotation[2].to_string(),
                        scale: eval_scene_number(&node.scale, time_norm, time_sec)?.to_string(),
                        opacity: "1".to_string(),
                        material: Some(WorldMaterial {
                            style: WorldMaterialStyle::Pbr,
                            outline: false,
                            outline_width: "0".to_string(),
                        }),
                        play: None,
                    });
                }
            }
        }

        let world_id = "__scene_3d_island".to_string();
        let world_graph = WorldGraph {
            id: Some(world_id.clone()),
            version: Some("scene-3d-bridge-v1".to_string()),
            fps,
            duration_ms,
            duration_explicit: true,
            size: canvas_size,
            render_size: Some(canvas_size),
            model_profiles: Vec::new(),
            worlds: vec![WorldNode {
                id: world_id.clone(),
                background: Some(WorldBackground {
                    id: None,
                    src: None,
                    fit: WorldBackgroundFit::Stretch,
                    color: "#00000000".to_string(),
                    opacity: "1".to_string(),
                }),
                camera,
                actors,
                directional_characters: Vec::new(),
            }],
            retargets: Vec::new(),
            actions: Vec::new(),
            apply_actions: Vec::new(),
            present: WorldPresent { from: world_id },
        };
        self.world_renderer
            .render_frame_gpu(
                &world_graph,
                frame,
                crate::scene::resource::default_world_asset_root(),
            )
            .await
            .map_err(|err| MotionLoomSceneRenderError::GpuRender {
                message: format!("Scene 3D island render failed: {err}"),
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_gpu_scene_native_commands_with_depth<'a>(
        &'a mut self,
        nodes: &'a [SceneNode],
        transform: Affine2,
        deform: Option<&'a EvaluatedDeformGrid>,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
        assets: &'a mut GpuSceneNativeAssets,
        primitives: &'a mut Vec<GpuScenePrimitive>,
        texture_layers: &'a mut Vec<GpuSceneTextureLayer>,
        text_requests: &'a mut Vec<GpuSceneTextRequest>,
        unsupported: &'a mut bool,
        depth: Option<SceneDepthContext<'a>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), MotionLoomSceneRenderError>> + 'a>> {
        Box::pin(async move {
            for node in nodes {
                match node {
                    SceneNode::Defs(defs) => {
                        for font in &defs.fonts {
                            self.font_defs.insert(font.id.clone(), font.clone());
                        }
                        for filter in &defs.filters {
                            self.filter_defs.insert(filter.id.clone(), filter.clone());
                        }
                        for component in &defs.components {
                            self.scene_components
                                .insert(component.id.clone(), component.children.clone());
                        }
                        for precompose in &defs.precomposes {
                            self.scene_precompose_defs
                                .insert(precompose.id.clone(), precompose.clone());
                        }
                        for mask in &defs.masks {
                            if let Some(id) = mask.id.as_deref() {
                                let Some(texture) = self
                                    .render_gpu_mask_texture(
                                        mask,
                                        transform,
                                        time_norm,
                                        time_sec,
                                        canvas_size,
                                    )
                                    .await?
                                else {
                                    *unsupported = true;
                                    continue;
                                };
                                assets.masks.insert(id.to_string(), texture);
                            }
                        }
                    }
                    SceneNode::Palette(_) | SceneNode::Shadow(_) => {}
                    SceneNode::Timeline(timeline) => {
                        let mut tracks = timeline
                            .children
                            .iter()
                            .filter_map(|node| match node {
                                SceneNode::Track(track) => Some(track),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        tracks.sort_by_key(|track| scene_track_composite_order(track));
                        let active_camera = active_scene_camera_from_tracks(
                            &tracks,
                            canvas_size.0,
                            canvas_size.1,
                            time_norm,
                            time_sec,
                        )?;
                        tracks.sort_by(|a, b| {
                            let a_world = is_scene_world_track(a);
                            let b_world = is_scene_world_track(b);
                            match (a_world, b_world) {
                                (true, true) => {
                                    let a_depth =
                                        scene_depth_track_sort_key(&a.z_depth, time_norm, time_sec)
                                            .unwrap_or(0.0);
                                    let b_depth =
                                        scene_depth_track_sort_key(&b.z_depth, time_norm, time_sec)
                                            .unwrap_or(0.0);
                                    b_depth.total_cmp(&a_depth).then_with(|| {
                                        scene_track_composite_order(a)
                                            .cmp(&scene_track_composite_order(b))
                                    })
                                }
                                _ => scene_track_composite_order(a)
                                    .cmp(&scene_track_composite_order(b)),
                            }
                        });
                        for track in tracks {
                            if is_scene_camera_track(track) {
                                continue;
                            }
                            let track_depth = if is_scene_world_track(track) {
                                Some(SceneDepthContext {
                                    active_camera,
                                    canvas_size,
                                    track_z_depth: &track.z_depth,
                                })
                            } else {
                                depth
                            };
                            self.collect_gpu_scene_native_commands_with_depth(
                                &track.children,
                                transform,
                                deform,
                                inherited_opacity,
                                time_norm,
                                time_sec,
                                canvas_size,
                                assets,
                                primitives,
                                texture_layers,
                                text_requests,
                                unsupported,
                                track_depth,
                            )
                            .await?;
                        }
                    }
                    SceneNode::Track(track) => {
                        self.collect_gpu_scene_native_commands_with_depth(
                            &track.children,
                            transform,
                            deform,
                            inherited_opacity,
                            time_norm,
                            time_sec,
                            canvas_size,
                            assets,
                            primitives,
                            texture_layers,
                            text_requests,
                            unsupported,
                            depth,
                        )
                        .await?;
                    }
                    SceneNode::Sequence(sequence) => {
                        if let Some((local_norm, local_sec)) =
                            scene_sequence_local_time(sequence, None, time_sec)
                        {
                            if depth.is_some() {
                                self.collect_gpu_scene_native_commands_depth_sorted(
                                    &sequence.children,
                                    transform,
                                    deform,
                                    inherited_opacity,
                                    local_norm,
                                    local_sec,
                                    canvas_size,
                                    assets,
                                    primitives,
                                    texture_layers,
                                    text_requests,
                                    unsupported,
                                    depth,
                                )
                                .await?;
                            } else {
                                self.collect_gpu_scene_native_commands_with_depth(
                                    &sequence.children,
                                    transform,
                                    deform,
                                    inherited_opacity,
                                    local_norm,
                                    local_sec,
                                    canvas_size,
                                    assets,
                                    primitives,
                                    texture_layers,
                                    text_requests,
                                    unsupported,
                                    depth,
                                )
                                .await?;
                            }
                        }
                    }
                    SceneNode::Chain(chain) => {
                        let mut cursor_ms = chain.from_ms as i64;
                        for child in &chain.children {
                            if let SceneNode::Sequence(sequence) = child {
                                if let Some((local_norm, local_sec)) =
                                    scene_sequence_local_time(sequence, Some(cursor_ms), time_sec)
                                {
                                    if depth.is_some() {
                                        self.collect_gpu_scene_native_commands_depth_sorted(
                                            &sequence.children,
                                            transform,
                                            deform,
                                            inherited_opacity,
                                            local_norm,
                                            local_sec,
                                            canvas_size,
                                            assets,
                                            primitives,
                                            texture_layers,
                                            text_requests,
                                            unsupported,
                                            depth,
                                        )
                                        .await?;
                                    } else {
                                        self.collect_gpu_scene_native_commands_with_depth(
                                            &sequence.children,
                                            transform,
                                            deform,
                                            inherited_opacity,
                                            local_norm,
                                            local_sec,
                                            canvas_size,
                                            assets,
                                            primitives,
                                            texture_layers,
                                            text_requests,
                                            unsupported,
                                            depth,
                                        )
                                        .await?;
                                    }
                                }
                                cursor_ms += sequence.duration_ms as i64 + chain.gap_ms;
                            }
                        }
                    }
                    SceneNode::Mask(mask) => {
                        if let Some(id) = mask.id.as_deref() {
                            let Some(texture) = self
                                .render_gpu_mask_texture(
                                    mask,
                                    transform,
                                    time_norm,
                                    time_sec,
                                    canvas_size,
                                )
                                .await?
                            else {
                                *unsupported = true;
                                continue;
                            };
                            assets.masks.insert(id.to_string(), texture.clone());
                        }
                        if !mask.children.is_empty() {
                            let Some(source) = self
                                .render_gpu_scene_texture_from_nodes(
                                    &mask.children,
                                    transform,
                                    inherited_opacity,
                                    time_norm,
                                    time_sec,
                                    canvas_size,
                                    assets,
                                )
                                .await?
                            else {
                                *unsupported = true;
                                continue;
                            };
                            let Some(matte) = self
                                .render_gpu_mask_texture(
                                    mask,
                                    transform,
                                    time_norm,
                                    time_sec,
                                    canvas_size,
                                )
                                .await?
                            else {
                                *unsupported = true;
                                continue;
                            };
                            texture_layers.push(GpuSceneTextureLayer {
                                source: GpuSceneTextureSource::Gpu(source),
                                transform: Affine2::identity(),
                                projected_quad: None,
                                opacity: 1.0,
                                blend: SceneBlendMode::Normal,
                                pick_id: 0,
                                matte: Some(GpuSceneTextureMatte {
                                    texture: matte,
                                    mode: GpuSceneMatteMode::Alpha,
                                    invert: false,
                                    feather: 0.0,
                                    expansion: 0.0,
                                }),
                                primitive_index: primitives.len(),
                            });
                        }
                    }
                    SceneNode::Precompose(precompose) => {
                        self.scene_precompose_defs
                            .insert(precompose.id.clone(), precompose.clone());
                    }
                    SceneNode::Layer(layer) => {
                        let layer_pick_id = self
                            .gpu_pick_ids
                            .get(layer.id.as_deref().unwrap_or(""))
                            .copied()
                            .unwrap_or(0);
                        if layer.is_3d {
                            let opacity = (eval_scene_number(&layer.opacity, time_norm, time_sec)?
                                * inherited_opacity)
                                .clamp(0.0, 1.0);
                            if opacity <= 0.0001 {
                                continue;
                            }
                            let blend = parse_scene_blend(&layer.blend)?;
                            let base_transform = if let Some(depth) = depth {
                                let z_depth = scene_layer_effective_z_depth(
                                    layer, depth, time_norm, time_sec,
                                )?;
                                transform.mul(scene_z_depth_transform(
                                    depth.active_camera,
                                    depth.canvas_size,
                                    z_depth,
                                ))
                            } else {
                                transform
                            };
                            let cache_key = gpu_layer3d_texture_cache_key(
                                self.prepared_graph_signature,
                                layer,
                                canvas_size,
                            );
                            let cached = cache_key
                                .and_then(|key| self.gpu_layer3d_texture_cache.get(&key).cloned());
                            let (source, offset_x, offset_y, source_size) = if let Some(cached) =
                                cached
                            {
                                let source_size =
                                    (cached.texture.width as f32, cached.texture.height as f32);
                                (
                                    GpuSceneTextureSource::Gpu(cached.texture),
                                    cached.offset_x,
                                    cached.offset_y,
                                    source_size,
                                )
                            } else {
                                let Some(source) = self.scene_layer_source_image(
                                    layer,
                                    canvas_size,
                                    time_norm,
                                    time_sec,
                                )?
                                else {
                                    continue;
                                };
                                let Some((cropped, offset_x, offset_y)) =
                                    crop_layer_alpha_bounds(&source)
                                else {
                                    continue;
                                };
                                let source_size = (cropped.width() as f32, cropped.height() as f32);
                                if let Some(cache_key) = cache_key {
                                    let texture = self
                                        .gpu_compositor
                                        .as_mut()
                                        .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                                            message: "Layer3D GPU compositor is unavailable"
                                                .to_string(),
                                        })?
                                        .upload_gpu_rgba_texture(&cropped)?;
                                    self.gpu_layer3d_texture_cache.insert(
                                        cache_key,
                                        CachedGpuLayer3dTexture {
                                            texture: texture.clone(),
                                            offset_x,
                                            offset_y,
                                        },
                                    );
                                    (
                                        GpuSceneTextureSource::Gpu(texture),
                                        offset_x,
                                        offset_y,
                                        source_size,
                                    )
                                } else {
                                    (
                                        GpuSceneTextureSource::Cpu(cropped),
                                        offset_x,
                                        offset_y,
                                        source_size,
                                    )
                                }
                            };
                            let projected_quad = scene_layer_3d_projected_quad(
                                layer,
                                base_transform,
                                (offset_x as f32, offset_y as f32),
                                source_size,
                                time_norm,
                                time_sec,
                            )?;
                            texture_layers.push(GpuSceneTextureLayer {
                                source,
                                transform: Affine2::identity(),
                                projected_quad: Some(projected_quad),
                                opacity,
                                blend,
                                pick_id: layer_pick_id,
                                matte: None,
                                primitive_index: primitives.len(),
                            });
                            continue;
                        }
                        let opacity = (eval_scene_number(&layer.opacity, time_norm, time_sec)?
                            * inherited_opacity)
                            .clamp(0.0, 1.0);
                        if opacity <= 0.0001 {
                            continue;
                        }
                        let blend = parse_scene_blend(&layer.blend)?;
                        let base_transform = if let Some(depth) = depth {
                            let z_depth =
                                scene_layer_effective_z_depth(layer, depth, time_norm, time_sec)?;
                            transform.mul(scene_z_depth_transform(
                                depth.active_camera,
                                depth.canvas_size,
                                z_depth,
                            ))
                        } else {
                            transform
                        };
                        if layer.source.is_none()
                            && layer.mask.is_none()
                            && layer.mask_from.is_none()
                            && layer.matte.is_none()
                            && layer.matte_from.is_none()
                            && layer.effect.is_none()
                            && !layer.children.is_empty()
                            && blend == SceneBlendMode::Normal
                            && (opacity - inherited_opacity).abs() <= 0.0001
                        {
                            let primitive_start = primitives.len();
                            let texture_start = texture_layers.len();
                            let text_start = text_requests.len();
                            let layer_transform = base_transform
                                .mul(scene_layer_local_transform(layer, time_norm, time_sec)?);
                            let mut layer_scope_hasher = DefaultHasher::new();
                            layer.id.hash(&mut layer_scope_hasher);
                            let layer_scope = layer_scope_hasher.finish();
                            if layer.id.is_some()
                                && deform.is_none()
                                && !vector_overlay_nodes_are_static(&layer.children)
                                && let Some(retained) = self
                                    .try_collect_retained_gpu_transform_scene_scoped(
                                        &layer.children,
                                        layer_transform,
                                        time_norm,
                                        time_sec,
                                        canvas_size,
                                        layer_scope,
                                    )?
                            {
                                primitives.extend(retained.primitives.iter().cloned());
                                text_requests.extend(retained.text_requests);
                                apply_gpu_pick_id_to_empty_commands(
                                    &mut primitives[primitive_start..],
                                    &mut texture_layers[texture_start..],
                                    &mut text_requests[text_start..],
                                    layer_pick_id,
                                );
                                continue;
                            }
                            self.collect_gpu_scene_native_commands_with_depth(
                                &layer.children,
                                layer_transform,
                                deform,
                                inherited_opacity,
                                time_norm,
                                time_sec,
                                canvas_size,
                                assets,
                                primitives,
                                texture_layers,
                                text_requests,
                                unsupported,
                                None,
                            )
                            .await?;
                            apply_gpu_pick_id_to_empty_commands(
                                &mut primitives[primitive_start..],
                                &mut texture_layers[texture_start..],
                                &mut text_requests[text_start..],
                                layer_pick_id,
                            );
                            continue;
                        }
                        if layer.source.is_some() && !layer.children.is_empty() {
                            *unsupported = true;
                            continue;
                        }
                        let mut source = if !layer.children.is_empty() {
                            let retained_screen_source = if blend == SceneBlendMode::Screen
                                && layer.id.is_some()
                                && deform.is_none()
                                && layer.source.is_none()
                                && layer.mask.is_none()
                                && layer.mask_from.is_none()
                                && layer.matte.is_none()
                                && layer.matte_from.is_none()
                                && layer.effect.is_none()
                            {
                                let mut scope_hasher = DefaultHasher::new();
                                "retained-screen-layer".hash(&mut scope_hasher);
                                layer.id.hash(&mut scope_hasher);
                                self.render_retained_gpu_screen_layer_texture(
                                    &layer.children,
                                    time_norm,
                                    time_sec,
                                    canvas_size,
                                    scope_hasher.finish(),
                                )
                                .await?
                            } else {
                                None
                            };
                            if let Some(texture) = retained_screen_source {
                                texture
                            } else {
                                let Some(texture) = self
                                    .render_gpu_scene_texture_from_nodes(
                                        &layer.children,
                                        Affine2::identity(),
                                        1.0,
                                        time_norm,
                                        time_sec,
                                        canvas_size,
                                        assets,
                                    )
                                    .await?
                                else {
                                    *unsupported = true;
                                    continue;
                                };
                                texture
                            }
                        } else if let Some(source_id) = layer.source.as_deref() {
                            if let Some(precompose) = self.scene_precompose_defs.get(source_id)
                                && precompose.size.unwrap_or(canvas_size) != canvas_size
                            {
                                *unsupported = true;
                                continue;
                            }
                            let Some(source) = self
                                .render_gpu_precompose_instance(
                                    source_id,
                                    layer,
                                    canvas_size,
                                    assets,
                                    time_norm,
                                    time_sec,
                                )
                                .await?
                            else {
                                continue;
                            };
                            source
                        } else {
                            continue;
                        };
                        if let Some(filter_id) = layer.effect.as_deref() {
                            source = self.apply_gpu_scene_filter_texture(
                                source, filter_id, time_norm, time_sec,
                            )?;
                        }
                        let layer_matte_sources = [
                            layer.mask.as_ref(),
                            layer.mask_from.as_ref(),
                            layer.matte.as_ref(),
                            layer.matte_from.as_ref(),
                        ]
                        .iter()
                        .filter(|value| value.is_some())
                        .count();
                        if layer_matte_sources > 1 {
                            *unsupported = true;
                            continue;
                        }
                        let mask_feather =
                            eval_scene_number(&layer.mask_feather, time_norm, time_sec)?.max(0.0);
                        let mask_expansion =
                            eval_scene_number(&layer.mask_expansion, time_norm, time_sec)?;
                        let matte = if let Some(mask_id) = layer.mask.as_deref() {
                            if let Some(mask) = self.scene_masks.get(mask_id).cloned() {
                                let Some(texture) = self
                                    .render_gpu_mask_texture(
                                        &mask,
                                        Affine2::identity(),
                                        time_norm,
                                        time_sec,
                                        canvas_size,
                                    )
                                    .await?
                                else {
                                    *unsupported = true;
                                    continue;
                                };
                                Some(GpuSceneTextureMatte {
                                    texture,
                                    mode: GpuSceneMatteMode::Alpha,
                                    invert: scene_mask_mode_inverts(&layer.mask_mode),
                                    feather: mask_feather,
                                    expansion: mask_expansion,
                                })
                            } else {
                                assets.masks.get(mask_id).cloned().map(|texture| {
                                    GpuSceneTextureMatte {
                                        texture,
                                        mode: GpuSceneMatteMode::Alpha,
                                        invert: scene_mask_mode_inverts(&layer.mask_mode),
                                        feather: mask_feather,
                                        expansion: mask_expansion,
                                    }
                                })
                            }
                        } else if let Some(mask_from) = layer.mask_from.as_deref() {
                            if let Some(precompose) = self.scene_precompose_defs.get(mask_from)
                                && precompose.size.unwrap_or(canvas_size) != canvas_size
                            {
                                *unsupported = true;
                                continue;
                            }
                            if self.scene_precompose_defs.contains_key(mask_from) {
                                self.render_gpu_precompose_instance(
                                    mask_from,
                                    layer,
                                    canvas_size,
                                    assets,
                                    time_norm,
                                    time_sec,
                                )
                                .await?
                                .map(|texture| {
                                    GpuSceneTextureMatte {
                                        texture,
                                        mode: gpu_matte_mode(&layer.mask_mode),
                                        invert: scene_mask_mode_inverts(&layer.mask_mode),
                                        feather: mask_feather,
                                        expansion: mask_expansion,
                                    }
                                })
                            } else {
                                self.render_gpu_node_matte_texture(
                                    mask_from,
                                    time_norm,
                                    time_sec,
                                    canvas_size,
                                    assets,
                                )
                                .await?
                                .map(|texture| {
                                    GpuSceneTextureMatte {
                                        texture,
                                        mode: gpu_matte_mode(&layer.mask_mode),
                                        invert: scene_mask_mode_inverts(&layer.mask_mode),
                                        feather: mask_feather,
                                        expansion: mask_expansion,
                                    }
                                })
                            }
                        } else if let Some(matte_id) = layer.matte.as_deref() {
                            if let Some(mask) = assets.masks.get(matte_id).cloned() {
                                Some(GpuSceneTextureMatte {
                                    texture: mask,
                                    mode: GpuSceneMatteMode::Alpha,
                                    invert: scene_bool(&layer.invert_matte),
                                    feather: 0.0,
                                    expansion: 0.0,
                                })
                            } else {
                                if let Some(precompose) = self.scene_precompose_defs.get(matte_id)
                                    && precompose.size.unwrap_or(canvas_size) != canvas_size
                                {
                                    *unsupported = true;
                                    continue;
                                }
                                self.render_gpu_precompose_instance(
                                    matte_id,
                                    layer,
                                    canvas_size,
                                    assets,
                                    time_norm,
                                    time_sec,
                                )
                                .await?
                                .map(|texture| {
                                    GpuSceneTextureMatte {
                                        texture,
                                        mode: gpu_matte_mode(&layer.matte_mode),
                                        invert: scene_bool(&layer.invert_matte),
                                        feather: 0.0,
                                        expansion: 0.0,
                                    }
                                })
                            }
                        } else if let Some(matte_from) = layer.matte_from.as_deref() {
                            self.render_gpu_node_matte_texture(
                                matte_from,
                                time_norm,
                                time_sec,
                                canvas_size,
                                assets,
                            )
                            .await?
                            .map(|texture| GpuSceneTextureMatte {
                                texture,
                                mode: gpu_matte_mode(&layer.matte_mode),
                                invert: scene_bool(&layer.invert_matte),
                                feather: 0.0,
                                expansion: 0.0,
                            })
                        } else {
                            None
                        };
                        texture_layers.push(GpuSceneTextureLayer {
                            source: GpuSceneTextureSource::Gpu(source),
                            transform: base_transform
                                .mul(scene_layer_local_transform(layer, time_norm, time_sec)?),
                            projected_quad: None,
                            opacity,
                            blend,
                            pick_id: layer_pick_id,
                            matte,
                            primitive_index: primitives.len(),
                        });
                    }
                    SceneNode::Image(image) => {
                        if deform.is_some() {
                            *unsupported = true;
                            continue;
                        }
                        if let Some(mut layer) = self.gpu_image_texture_layer(
                            image,
                            transform,
                            inherited_opacity,
                            time_norm,
                            time_sec,
                            canvas_size,
                        )? {
                            if let Some(material_id) = image.material.as_deref()
                                && let GpuSceneTextureSource::Gpu(source) = layer.source
                            {
                                layer.source =
                                    GpuSceneTextureSource::Gpu(self.apply_gpu_material_texture(
                                        source,
                                        material_id,
                                        time_norm,
                                        time_sec,
                                    )?);
                            }
                            layer.pick_id = self
                                .gpu_pick_ids
                                .get(image.id.as_deref().unwrap_or(""))
                                .copied()
                                .unwrap_or(0);
                            texture_layers.push(layer);
                        }
                    }
                    SceneNode::Svg(svg) => {
                        if deform.is_some() {
                            *unsupported = true;
                            continue;
                        }
                        if let Some(mut layer) = self.gpu_svg_texture_layer(
                            svg,
                            transform,
                            inherited_opacity,
                            time_norm,
                            time_sec,
                            canvas_size,
                        )? {
                            if let Some(material_id) = svg.material.as_deref()
                                && let GpuSceneTextureSource::Gpu(source) = layer.source
                            {
                                layer.source =
                                    GpuSceneTextureSource::Gpu(self.apply_gpu_material_texture(
                                        source,
                                        material_id,
                                        time_norm,
                                        time_sec,
                                    )?);
                            }
                            layer.pick_id = self
                                .gpu_pick_ids
                                .get(svg.id.as_deref().unwrap_or(""))
                                .copied()
                                .unwrap_or(0);
                            texture_layers.push(layer);
                        }
                    }
                    SceneNode::Path(path) => {
                        let path_pick_id = self
                            .gpu_pick_ids
                            .get(path.id.as_deref().unwrap_or(""))
                            .copied()
                            .unwrap_or(0);
                        if path_requires_cpu_overlay(path) || path.material.is_some() {
                            if deform.is_some() || !is_gpu_native_blend(&path.blend) {
                                *unsupported = true;
                                continue;
                            }
                            let mut overlay_layer = self.gpu_path_overlay_texture_layer(
                                path,
                                transform,
                                inherited_opacity,
                                time_norm,
                                time_sec,
                            )?;
                            if overlay_layer.is_none() {
                                overlay_layer = self.gpu_vector_overlay_texture_layer(
                                    std::slice::from_ref(node),
                                    transform,
                                    inherited_opacity,
                                    time_norm,
                                    time_sec,
                                    canvas_size,
                                )?;
                            }
                            if let Some(mut layer) = overlay_layer {
                                if let Some(material_id) = path.material.as_deref()
                                    && let GpuSceneTextureSource::Gpu(source) = layer.source
                                {
                                    layer.source = GpuSceneTextureSource::Gpu(
                                        self.apply_gpu_material_texture(
                                            source,
                                            material_id,
                                            time_norm,
                                            time_sec,
                                        )?,
                                    );
                                }
                                layer.pick_id = path_pick_id;
                                texture_layers.push(layer);
                            }
                        } else {
                            let primitive_start = primitives.len();
                            self.push_gpu_path_commands_cached(
                                path,
                                transform,
                                deform,
                                inherited_opacity,
                                time_norm,
                                time_sec,
                                primitives,
                            )?;
                            if path_pick_id != 0 {
                                for primitive in &mut primitives[primitive_start..] {
                                    if primitive.pick_id == 0 {
                                        primitive.pick_id = path_pick_id;
                                    }
                                }
                            }
                        }
                    }
                    SceneNode::Group(group) => {
                        let group_pick_id = self
                            .gpu_pick_ids
                            .get(group.id.as_deref().unwrap_or(""))
                            .copied()
                            .unwrap_or(0);
                        let opacity = (eval_scene_number(&group.opacity, time_norm, time_sec)?
                            * inherited_opacity)
                            .clamp(0.0, 1.0);
                        if opacity <= 0.0001 {
                            continue;
                        }
                        let group_local = scene_group_local_transform(group, time_norm, time_sec)?;
                        let group_transform = transform.mul(group_local);
                        let group_deform = eval_group_deform_grid(group, time_norm, time_sec)?;
                        if group.mask.is_some() && group.mask_from.is_some() {
                            *unsupported = true;
                            continue;
                        }
                        if let Some(composite) = group.composite.as_ref()
                            && composite.space.eq_ignore_ascii_case("3d")
                            && !composite.nodes_3d.is_empty()
                        {
                            let fps = self.prepared_graph_fps.max(1.0);
                            let frame = (time_sec * fps).round().max(0.0) as u32;
                            let image = self
                                .render_scene_3d_composite(
                                    composite,
                                    frame,
                                    fps,
                                    self.prepared_graph_duration_ms,
                                    canvas_size,
                                    time_norm,
                                    time_sec,
                                )
                                .await?;
                            texture_layers.push(GpuSceneTextureLayer {
                                source: GpuSceneTextureSource::Cpu(image),
                                transform: group_transform,
                                projected_quad: None,
                                opacity,
                                blend: SceneBlendMode::Normal,
                                pick_id: group_pick_id,
                                matte: None,
                                primitive_index: primitives.len(),
                            });

                            // Camera/Model/Light tags are compiler-only nodes,
                            // while ordinary children remain legal overlays in
                            // the same CompositeGroup and keep their authored
                            // order after the 3D island.
                            if !group.children.is_empty() {
                                self.collect_gpu_scene_native_commands(
                                    &group.children,
                                    group_transform,
                                    deform,
                                    opacity,
                                    time_norm,
                                    time_sec,
                                    canvas_size,
                                    assets,
                                    primitives,
                                    texture_layers,
                                    text_requests,
                                    unsupported,
                                )
                                .await?;
                            }
                            continue;
                        }
                        if group.mask.is_some()
                            || group.mask_from.is_some()
                            || !group.effects.is_empty()
                            || group.material.is_some()
                        {
                            if group_deform.is_some() {
                                *unsupported = true;
                                continue;
                            }
                            let Some(mut source) = self
                                .render_gpu_scene_texture_from_nodes(
                                    &group.children,
                                    Affine2::identity(),
                                    opacity,
                                    time_norm,
                                    time_sec,
                                    canvas_size,
                                    assets,
                                )
                                .await?
                            else {
                                *unsupported = true;
                                continue;
                            };
                            for filter_id in &group.effects {
                                source = self.apply_gpu_scene_filter_texture(
                                    source, filter_id, time_norm, time_sec,
                                )?;
                            }
                            if let Some(material_id) = group.material.as_deref() {
                                source = self.apply_gpu_material_texture(
                                    source,
                                    material_id,
                                    time_norm,
                                    time_sec,
                                )?;
                            }
                            let mask_feather =
                                eval_scene_number(&group.mask_feather, time_norm, time_sec)?
                                    .max(0.0);
                            let mask_expansion =
                                eval_scene_number(&group.mask_expansion, time_norm, time_sec)?;
                            let matte = if let Some(mask_id) = group.mask.as_deref() {
                                let Some(matte) = assets.masks.get(mask_id).cloned() else {
                                    *unsupported = true;
                                    continue;
                                };
                                Some(GpuSceneTextureMatte {
                                    texture: matte,
                                    mode: GpuSceneMatteMode::Alpha,
                                    invert: scene_mask_mode_inverts(&group.mask_mode),
                                    feather: mask_feather,
                                    expansion: mask_expansion,
                                })
                            } else if let Some(mask_from) = group.mask_from.as_deref() {
                                let Some(matte) = self
                                    .render_gpu_node_matte_texture(
                                        mask_from,
                                        time_norm,
                                        time_sec,
                                        canvas_size,
                                        assets,
                                    )
                                    .await?
                                else {
                                    *unsupported = true;
                                    continue;
                                };
                                Some(GpuSceneTextureMatte {
                                    texture: matte,
                                    mode: gpu_matte_mode(&group.mask_mode),
                                    invert: scene_mask_mode_inverts(&group.mask_mode),
                                    feather: mask_feather,
                                    expansion: mask_expansion,
                                })
                            } else {
                                None
                            };
                            texture_layers.push(GpuSceneTextureLayer {
                                source: GpuSceneTextureSource::Gpu(source),
                                transform: group_transform,
                                projected_quad: None,
                                opacity: 1.0,
                                blend: SceneBlendMode::Normal,
                                pick_id: group_pick_id,
                                matte,
                                primitive_index: primitives.len(),
                            });
                        } else {
                            let group_deform = group_deform
                                .as_ref()
                                .map(|grid| transform_deform_grid(grid, group_transform));
                            let child_deform = group_deform.as_ref().or(deform);
                            let primitive_start = primitives.len();
                            let texture_start = texture_layers.len();
                            let text_start = text_requests.len();
                            self.collect_gpu_scene_native_commands(
                                &group.children,
                                group_transform,
                                child_deform,
                                opacity,
                                time_norm,
                                time_sec,
                                canvas_size,
                                assets,
                                primitives,
                                texture_layers,
                                text_requests,
                                unsupported,
                            )
                            .await?;
                            apply_gpu_pick_id_to_empty_commands(
                                &mut primitives[primitive_start..],
                                &mut texture_layers[texture_start..],
                                &mut text_requests[text_start..],
                                group_pick_id,
                            );
                        }
                    }
                    SceneNode::Puppet(puppet) => {
                        let puppet_pick_id = self
                            .gpu_pick_ids
                            .get(puppet.id.as_deref().unwrap_or(""))
                            .copied()
                            .unwrap_or(0);
                        let opacity = (eval_scene_number(&puppet.opacity, time_norm, time_sec)?
                            * inherited_opacity)
                            .clamp(0.0, 1.0);
                        if opacity <= 0.0001 {
                            continue;
                        }
                        let puppet_local =
                            scene_puppet_local_transform(puppet, time_norm, time_sec)?;
                        let puppet_transform = transform.mul(puppet_local);
                        let visual_children = puppet
                            .children
                            .iter()
                            .filter(|child| {
                                !matches!(
                                    child,
                                    SceneNode::Pin(_)
                                        | SceneNode::LimbEnvelope(_)
                                        | SceneNode::LimbRegion(_)
                                        | SceneNode::MeshTopology(_)
                                        | SceneNode::Vertex(_)
                                        | SceneNode::Triangle(_)
                                        | SceneNode::Edge(_)
                                        | SceneNode::Region(_)
                                )
                            })
                            .cloned()
                            .collect::<Vec<_>>();
                        if visual_children.is_empty() {
                            continue;
                        }
                        let Some(mut source) = self
                            .render_gpu_scene_texture_from_nodes(
                                &visual_children,
                                puppet_transform,
                                opacity,
                                time_norm,
                                time_sec,
                                canvas_size,
                                assets,
                            )
                            .await?
                        else {
                            *unsupported = true;
                            continue;
                        };
                        if let Some(deform_grid) =
                            eval_puppet_deform_grid(puppet, time_norm, time_sec)?
                        {
                            let transformed_grid =
                                transform_deform_grid(&deform_grid, puppet_transform);
                            source = self
                                .gpu_compositor
                                .as_mut()
                                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                                    message: "GPU compositor was not initialized".to_string(),
                                })?
                                .apply_gpu_deform_texture(&source, &transformed_grid)?;
                        }
                        texture_layers.push(GpuSceneTextureLayer {
                            source: GpuSceneTextureSource::Gpu(source),
                            transform: Affine2::identity(),
                            projected_quad: None,
                            opacity: 1.0,
                            blend: SceneBlendMode::Normal,
                            pick_id: puppet_pick_id,
                            matte: None,
                            primitive_index: primitives.len(),
                        });
                    }
                    SceneNode::Pin(_)
                    | SceneNode::LimbEnvelope(_)
                    | SceneNode::LimbRegion(_)
                    | SceneNode::MeshTopology(_)
                    | SceneNode::Vertex(_)
                    | SceneNode::Triangle(_)
                    | SceneNode::Edge(_)
                    | SceneNode::Region(_) => {
                        continue;
                    }
                    SceneNode::Camera(camera) => {
                        let opacity = (eval_scene_number(&camera.opacity, time_norm, time_sec)?
                            * inherited_opacity)
                            .clamp(0.0, 1.0);
                        if opacity <= 0.0001 {
                            continue;
                        }
                        let camera_transform = camera_transform(
                            camera,
                            &camera.children,
                            canvas_size.0,
                            canvas_size.1,
                            time_norm,
                            time_sec,
                        )?;
                        self.collect_gpu_scene_native_commands(
                            &camera.children,
                            transform.mul(camera_transform),
                            deform,
                            opacity,
                            time_norm,
                            time_sec,
                            canvas_size,
                            assets,
                            primitives,
                            texture_layers,
                            text_requests,
                            unsupported,
                        )
                        .await?;
                    }
                    SceneNode::Character(character) => {
                        let character_pick_id = self
                            .gpu_pick_ids
                            .get(character.id.as_deref().unwrap_or(""))
                            .copied()
                            .unwrap_or(0);
                        let opacity = (eval_scene_number(&character.opacity, time_norm, time_sec)?
                            * inherited_opacity)
                            .clamp(0.0, 1.0);
                        if opacity <= 0.0001 {
                            continue;
                        }
                        let character_transform = transform.mul(scene_character_local_transform(
                            character, time_norm, time_sec,
                        )?);
                        let primitive_start = primitives.len();
                        let texture_start = texture_layers.len();
                        let text_start = text_requests.len();
                        self.collect_gpu_scene_native_commands(
                            &character.children,
                            character_transform,
                            deform,
                            opacity,
                            time_norm,
                            time_sec,
                            canvas_size,
                            assets,
                            primitives,
                            texture_layers,
                            text_requests,
                            unsupported,
                        )
                        .await?;
                        apply_gpu_pick_id_to_empty_commands(
                            &mut primitives[primitive_start..],
                            &mut texture_layers[texture_start..],
                            &mut text_requests[text_start..],
                            character_pick_id,
                        );
                    }
                    SceneNode::Part(part) => {
                        let part_pick_id = self
                            .gpu_pick_ids
                            .get(part.id.as_deref().unwrap_or(""))
                            .copied()
                            .unwrap_or(0);
                        let opacity = (eval_scene_number(&part.opacity, time_norm, time_sec)?
                            * inherited_opacity)
                            .clamp(0.0, 1.0);
                        if opacity <= 0.0001 {
                            continue;
                        }
                        let x = eval_scene_number(&part.x, time_norm, time_sec)?;
                        let y = eval_scene_number(&part.y, time_norm, time_sec)?;
                        let rotation = eval_scene_number(&part.rotation, time_norm, time_sec)?;
                        let scale =
                            eval_scene_number(&part.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
                        let anchor_x = eval_scene_number(&part.anchor_x, time_norm, time_sec)?;
                        let anchor_y = eval_scene_number(&part.anchor_y, time_norm, time_sec)?;
                        let part_transform = transform
                            .mul(Affine2::translate(x, y))
                            .mul(Affine2::rotate_deg(rotation))
                            .mul(Affine2::scale(scale))
                            .mul(Affine2::translate(-anchor_x, -anchor_y));
                        let primitive_start = primitives.len();
                        let texture_start = texture_layers.len();
                        let text_start = text_requests.len();
                        self.collect_gpu_scene_native_commands(
                            &part.children,
                            part_transform,
                            deform,
                            opacity,
                            time_norm,
                            time_sec,
                            canvas_size,
                            assets,
                            primitives,
                            texture_layers,
                            text_requests,
                            unsupported,
                        )
                        .await?;
                        apply_gpu_pick_id_to_empty_commands(
                            &mut primitives[primitive_start..],
                            &mut texture_layers[texture_start..],
                            &mut text_requests[text_start..],
                            part_pick_id,
                        );
                    }
                    SceneNode::Repeat(repeat) => {
                        let repeat_pick_id = self
                            .gpu_pick_ids
                            .get(repeat.id.as_deref().unwrap_or(""))
                            .copied()
                            .unwrap_or(0);
                        let count = eval_repeat_count(&repeat.count, time_norm, time_sec)?;
                        let x = eval_scene_number(&repeat.x, time_norm, time_sec)?;
                        let y = eval_scene_number(&repeat.y, time_norm, time_sec)?;
                        let rotation = eval_scene_number(&repeat.rotation, time_norm, time_sec)?;
                        let scale = eval_scene_number(&repeat.scale, time_norm, time_sec)?
                            .clamp(0.001, 64.0);
                        let opacity = eval_scene_number(&repeat.opacity, time_norm, time_sec)?;
                        let x_step = eval_scene_number(&repeat.x_step, time_norm, time_sec)?;
                        let y_step = eval_scene_number(&repeat.y_step, time_norm, time_sec)?;
                        let rotation_step =
                            eval_scene_number(&repeat.rotation_step, time_norm, time_sec)?;
                        let scale_step =
                            eval_scene_number(&repeat.scale_step, time_norm, time_sec)?;
                        let opacity_step =
                            eval_scene_number(&repeat.opacity_step, time_norm, time_sec)?;
                        let primitive_start = primitives.len();
                        let texture_start = texture_layers.len();
                        let text_start = text_requests.len();
                        for index in 0..count {
                            let i = index as f32;
                            let copy_opacity =
                                ((opacity + opacity_step * i) * inherited_opacity).clamp(0.0, 1.0);
                            if copy_opacity <= 0.0001 {
                                continue;
                            }
                            let repeat_transform = transform
                                .mul(Affine2::translate(x + x_step * i, y + y_step * i))
                                .mul(Affine2::rotate_deg(rotation + rotation_step * i))
                                .mul(Affine2::scale((scale + scale_step * i).clamp(0.001, 64.0)));
                            self.collect_gpu_scene_native_commands(
                                &repeat.children,
                                repeat_transform,
                                deform,
                                copy_opacity,
                                time_norm,
                                time_sec,
                                canvas_size,
                                assets,
                                primitives,
                                texture_layers,
                                text_requests,
                                unsupported,
                            )
                            .await?;
                        }
                        apply_gpu_pick_id_to_empty_commands(
                            &mut primitives[primitive_start..],
                            &mut texture_layers[texture_start..],
                            &mut text_requests[text_start..],
                            repeat_pick_id,
                        );
                    }
                    SceneNode::Use(use_node) => {
                        let use_pick_id = self
                            .gpu_pick_ids
                            .get(use_node.id.as_deref().unwrap_or(""))
                            .copied()
                            .unwrap_or(0);
                        let opacity = (eval_scene_number(&use_node.opacity, time_norm, time_sec)?
                            * inherited_opacity)
                            .clamp(0.0, 1.0);
                        if opacity <= 0.0001 {
                            continue;
                        }
                        let Some(children) = self.scene_components.get(&use_node.ref_id).cloned()
                        else {
                            continue;
                        };
                        let use_transform = transform
                            .mul(scene_use_local_transform(use_node, time_norm, time_sec)?);
                        let primitive_start = primitives.len();
                        let texture_start = texture_layers.len();
                        let text_start = text_requests.len();
                        self.collect_gpu_scene_native_commands(
                            &children,
                            use_transform,
                            deform,
                            opacity,
                            time_norm,
                            time_sec,
                            canvas_size,
                            assets,
                            primitives,
                            texture_layers,
                            text_requests,
                            unsupported,
                        )
                        .await?;
                        apply_gpu_pick_id_to_empty_commands(
                            &mut primitives[primitive_start..],
                            &mut texture_layers[texture_start..],
                            &mut text_requests[text_start..],
                            use_pick_id,
                        );

                        let use_blend = parse_scene_blend(&use_node.blend)?;
                        if use_blend != SceneBlendMode::Normal {
                            for primitive in &mut primitives[primitive_start..] {
                                primitive.blend = use_blend;
                            }
                            for layer in &mut texture_layers[texture_start..] {
                                layer.blend = use_blend;
                            }
                            if text_requests.len() > text_start {
                                // Text is rasterized as a normal texture layer later; fall back
                                // rather than silently losing a non-normal Use-level blend.
                                *unsupported = true;
                            }
                        }
                    }
                    _ => {
                        let node_pick_id = scene_node_id(node)
                            .and_then(|id| self.gpu_pick_ids.get(id).copied())
                            .unwrap_or(0);
                        let primitive_start = primitives.len();
                        let texture_start = texture_layers.len();
                        let text_start = text_requests.len();
                        let mut overlays = Vec::<CpuSceneOverlay>::new();
                        collect_gpu_scene_commands(
                            std::slice::from_ref(node),
                            transform,
                            deform,
                            inherited_opacity,
                            time_norm,
                            time_sec,
                            canvas_size,
                            &self.gradient_defs,
                            &self.palette_defs,
                            &self.scene_components,
                            primitives,
                            text_requests,
                            &mut overlays,
                        )?;
                        if !overlays.is_empty() {
                            *unsupported = true;
                        }
                        apply_gpu_pick_id_to_empty_commands(
                            &mut primitives[primitive_start..],
                            &mut texture_layers[texture_start..],
                            &mut text_requests[text_start..],
                            node_pick_id,
                        );
                    }
                }
            }
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_gpu_scene_native_commands_depth_sorted<'a>(
        &'a mut self,
        nodes: &'a [SceneNode],
        transform: Affine2,
        deform: Option<&'a EvaluatedDeformGrid>,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
        assets: &'a mut GpuSceneNativeAssets,
        primitives: &'a mut Vec<GpuScenePrimitive>,
        texture_layers: &'a mut Vec<GpuSceneTextureLayer>,
        text_requests: &'a mut Vec<GpuSceneTextRequest>,
        unsupported: &'a mut bool,
        depth: Option<SceneDepthContext<'a>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), MotionLoomSceneRenderError>> + 'a>> {
        Box::pin(async move {
            let Some(depth) = depth else {
                return self
                    .collect_gpu_scene_native_commands_with_depth(
                        nodes,
                        transform,
                        deform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        assets,
                        primitives,
                        texture_layers,
                        text_requests,
                        unsupported,
                        None,
                    )
                    .await;
            };
            let mut layer_items = Vec::<(usize, f32, &SceneNode)>::new();
            for (index, node) in nodes.iter().enumerate() {
                if let SceneNode::Layer(layer) = node {
                    layer_items.push((
                        index,
                        scene_layer_effective_z_depth(layer, depth, time_norm, time_sec)?,
                        node,
                    ));
                } else {
                    self.collect_gpu_scene_native_commands_with_depth(
                        std::slice::from_ref(node),
                        transform,
                        deform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        assets,
                        primitives,
                        texture_layers,
                        text_requests,
                        unsupported,
                        Some(depth),
                    )
                    .await?;
                }
            }
            layer_items.sort_by(|(a_order, a_depth, _), (b_order, b_depth, _)| {
                b_depth
                    .total_cmp(a_depth)
                    .then_with(|| a_order.cmp(b_order))
            });
            for (_, _, node) in layer_items {
                self.collect_gpu_scene_native_commands_with_depth(
                    std::slice::from_ref(node),
                    transform,
                    deform,
                    inherited_opacity,
                    time_norm,
                    time_sec,
                    canvas_size,
                    assets,
                    primitives,
                    texture_layers,
                    text_requests,
                    unsupported,
                    Some(depth),
                )
                .await?;
            }
            Ok(())
        })
    }

    async fn render_gpu_mask_texture(
        &mut self,
        mask: &MaskNode,
        transform: Affine2,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
    ) -> Result<Option<GpuSceneNativeTexture>, MotionLoomSceneRenderError> {
        let opacity = eval_scene_number(&mask.opacity, time_norm, time_sec)?.clamp(0.0, 1.0);
        let transform = self.mask_effective_transform(mask, transform, time_norm, time_sec);
        let mut primitives = Vec::<GpuScenePrimitive>::new();
        if opacity > 0.0001 {
            let color = [255, 255, 255, 255];
            match mask.shape.trim().to_ascii_lowercase().as_str() {
                "circle" => {
                    let x = eval_scene_number(&mask.x, time_norm, time_sec)?;
                    let y = eval_scene_number(&mask.y, time_norm, time_sec)?;
                    let radius = eval_scene_number(&mask.radius, time_norm, time_sec)?.max(0.0);
                    primitives.push(GpuScenePrimitive {
                        kind: GPU_SHAPE_CIRCLE_FILL,
                        transform,
                        shape: [x, y, radius, 0.0],
                        radius: 0.0,
                        stroke_width: 0.0,
                        blur: 0.0,
                        color,
                        opacity,
                        blend: SceneBlendMode::Normal,
                        pick_id: 0,
                        gradient: None,
                        line_t0: 0.0,
                        line_t1: 1.0,
                        taper_start: 0.0,
                        taper_end: 0.0,
                    });
                }
                "ellipse" | "oval" => {
                    let x = eval_scene_number(&mask.x, time_norm, time_sec)?;
                    let y = eval_scene_number(&mask.y, time_norm, time_sec)?;
                    let width = eval_scene_number(&mask.width, time_norm, time_sec)?.max(0.0);
                    let height = eval_scene_number(&mask.height, time_norm, time_sec)?.max(0.0);
                    let subpaths = vec![ellipse_polygon(
                        x + width * 0.5,
                        y + height * 0.5,
                        width * 0.5,
                        height * 0.5,
                    )];
                    push_gpu_filled_path_triangles(
                        &mut primitives,
                        transform,
                        &subpaths,
                        color,
                        opacity,
                        None,
                    );
                }
                "path" => {
                    let Some(d) = mask.d.as_deref() else {
                        return Ok(None);
                    };
                    let path = PathNode {
                        id: mask.id.clone(),
                        brush: None,
                        material: None,
                        d: d.to_string(),
                        fill: Some("#ffffff".to_string()),
                        fill_rule: "nonzero".to_string(),
                        boolean_op: "none".to_string(),
                        offset_path: "0".to_string(),
                        round_corners: "0".to_string(),
                        normalize: "false".to_string(),
                        stroke: "none".to_string(),
                        stroke_width: "0".to_string(),
                        stroke_width_start: "1".to_string(),
                        stroke_width_end: "1".to_string(),
                        stroke_width_profile: String::new(),
                        line_cap: "round".to_string(),
                        line_join: "round".to_string(),
                        trim_start: "0".to_string(),
                        trim_end: "1".to_string(),
                        taper_start: "0".to_string(),
                        taper_end: "0".to_string(),
                        stroke_style: "clean".to_string(),
                        stroke_roughness: "0".to_string(),
                        stroke_copies: "1".to_string(),
                        stroke_texture: "0".to_string(),
                        stroke_bristles: "1".to_string(),
                        stroke_pressure: "1".to_string(),
                        stroke_pressure_min: "1".to_string(),
                        stroke_pressure_curve: "1".to_string(),
                        opacity: mask.opacity.clone(),
                        blend: "normal".to_string(),
                        x: "0".to_string(),
                        y: "0".to_string(),
                        rotation: "0".to_string(),
                        scale: "1".to_string(),
                        scale_x: "1".to_string(),
                        scale_y: "1".to_string(),
                        skew_x: "0".to_string(),
                        skew_y: "0".to_string(),
                        transform_origin_x: "0".to_string(),
                        transform_origin_y: "0".to_string(),
                        texture: None,
                        texture_opacity: "1".to_string(),
                        texture_scale: "1".to_string(),
                        texture_mask: "0".to_string(),
                    };
                    push_gpu_path_commands(
                        &path,
                        transform,
                        None,
                        1.0,
                        time_norm,
                        time_sec,
                        &self.gradient_defs,
                        &mut primitives,
                    )?;
                }
                _ => {
                    let x = eval_scene_number(&mask.x, time_norm, time_sec)?;
                    let y = eval_scene_number(&mask.y, time_norm, time_sec)?;
                    let width = eval_scene_number(&mask.width, time_norm, time_sec)?.max(0.0);
                    let height = eval_scene_number(&mask.height, time_norm, time_sec)?.max(0.0);
                    let radius = eval_scene_number(&mask.radius, time_norm, time_sec)?.max(0.0);
                    primitives.push(GpuScenePrimitive {
                        kind: GPU_SHAPE_RECT_FILL,
                        transform,
                        shape: [x, y, width, height],
                        radius,
                        stroke_width: 0.0,
                        blur: 0.0,
                        color,
                        opacity,
                        blend: SceneBlendMode::Normal,
                        pick_id: 0,
                        gradient: None,
                        line_t0: 0.0,
                        line_t1: 1.0,
                        taper_start: 0.0,
                        taper_end: 0.0,
                    });
                }
            }
        }

        self.ensure_gpu_compositor_size(canvas_size.0.max(1), canvas_size.1.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let mut texture =
            compositor.render_scene_content_to_texture(&primitives, &[], [0, 0, 0, 0])?;
        let feather = eval_scene_number(&mask.feather, time_norm, time_sec)?.max(0.0);
        if feather > 0.01 {
            texture = compositor
                .apply_gpu_blur_texture(&texture, &[(true, feather), (false, feather)])?;
        }
        Ok(Some(texture))
    }

    fn render_cpu_scene_nodes_scaled(
        &mut self,
        nodes: &[SceneNode],
        output_size: (u32, u32),
        logical_size: (u32, u32),
        time_norm: f32,
        time_sec: f32,
        background: [u8; 4],
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let mut logical_canvas = RgbaImage::from_pixel(
            logical_size.0.max(1),
            logical_size.1.max(1),
            Rgba(background),
        );
        self.draw_scene_nodes(&mut logical_canvas, nodes, time_norm, time_sec, 1.0)?;
        Ok(fit_logical_canvas_to_output(&logical_canvas, output_size))
    }

    /// Read a `GraphTextureSource` back to CPU, reusing the cached image if
    /// it is already CPU-backed.
    async fn graph_source_to_cpu(
        &mut self,
        source: &GraphTextureSource,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        match source {
            GraphTextureSource::Cpu(image) => Ok(image.clone()),
            GraphTextureSource::Gpu(texture) => {
                self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                    .await?;
                let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                    MotionLoomSceneRenderError::GpuRender {
                        message: "GPU compositor was not initialized".to_string(),
                    }
                })?;
                compositor
                    .readback_texture_rgba_async(&texture.texture)
                    .await
            }
        }
    }

    async fn apply_process_pass_mask(
        &mut self,
        output: GraphTextureSource,
        original: Option<&GraphTextureSource>,
        pass: &PassNode,
        resources: &HashMap<String, GraphTextureSource>,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let Some(mask_id) = pass.mask.as_deref().map(normalize_process_mask_ref) else {
            return Ok(output);
        };
        let Some(mask_source) = resources.get(mask_id.as_str()) else {
            return Ok(output);
        };
        let mut image = self.graph_source_to_cpu(&output).await?;
        let mask = self.graph_source_to_cpu(mask_source).await?;
        let invert = scene_mask_mode_inverts(&pass.mask_mode)
            || eval_scene_bool_like(&pass.mask_invert, time_norm, time_sec)?;
        let mode = pass.mask_mode.trim().to_ascii_lowercase().replace('_', "-");
        if let Some(original) = original {
            let original = self.graph_source_to_cpu(original).await?;
            if mode == "luma" || mode == "luminance" || mode == "brightness" {
                apply_luma_process_mask_mix(&mut image, &original, &mask, invert);
            } else {
                apply_alpha_process_mask_mix(&mut image, &original, &mask, invert);
            }
        } else if mode == "luma" || mode == "luminance" || mode == "brightness" {
            apply_luma_process_mask_alpha(&mut image, &mask, invert);
        } else {
            apply_alpha_mask_with_invert(&mut image, &mask, invert);
        }
        Ok(GraphTextureSource::Cpu(image))
    }

    async fn apply_scene_post_pass_multi(
        &mut self,
        inputs: &[GraphTextureSource],
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let effect = pass.effect.to_ascii_lowercase();
        if effect == "over" || effect == "composite.over" {
            let mut cpu_images = Vec::with_capacity(inputs.len());
            for input in inputs {
                cpu_images.push(self.graph_source_to_cpu(input).await?);
            }
            return Ok(GraphTextureSource::Cpu(apply_over_pass(&cpu_images)));
        }
        if effect == "hsla" || effect == "hsla_overlay" || effect == "color.hsla" {
            let hue = pass_param_expr(pass, "hue")
                .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                .transpose()?
                .unwrap_or(0.0);
            let saturation = pass_param_expr(pass, "saturation")
                .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                .transpose()?
                .unwrap_or(0.0);
            let lightness = pass_param_expr(pass, "lightness")
                .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                .transpose()?
                .unwrap_or(0.5);
            let alpha = pass_param_expr(pass, "alpha")
                .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                .transpose()?
                .unwrap_or(1.0);
            if self.profile.uses_gpu_compositor()
                && let GraphTextureSource::Gpu(texture) = &inputs[0]
            {
                self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                    .await?;
                let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                    MotionLoomSceneRenderError::GpuRender {
                        message: "GPU compositor was not initialized".to_string(),
                    }
                })?;
                return Ok(GraphTextureSource::Gpu(
                    compositor.apply_gpu_hsla_overlay_texture(
                        texture, hue, saturation, lightness, alpha,
                    )?,
                ));
            }
            let image = self.graph_source_to_cpu(&inputs[0]).await?;
            return apply_hsla_pass(&image, pass, time_norm, time_sec).map(GraphTextureSource::Cpu);
        }
        if effect == "blur" || effect == "gaussian_blur" || effect == "gaussian_5tap_blur" {
            let sigma = pass_param_expr(pass, "sigma")
                .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                .transpose()?
                .unwrap_or(2.0)
                .clamp(0.0, 64.0);
            if sigma <= 0.05 {
                return Ok(inputs[0].clone());
            }
            if self.profile.uses_gpu_compositor() {
                match &inputs[0] {
                    GraphTextureSource::Gpu(texture) => {
                        self.ensure_gpu_compositor_size(
                            texture.width.max(1),
                            texture.height.max(1),
                        )
                        .await?;
                        let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                            MotionLoomSceneRenderError::GpuRender {
                                message: "GPU compositor was not initialized".to_string(),
                            }
                        })?;
                        return Ok(GraphTextureSource::Gpu(compositor.apply_gpu_blur_texture(
                            texture,
                            &[(true, sigma), (false, sigma)],
                        )?));
                    }
                    GraphTextureSource::Cpu(image) => {
                        self.ensure_gpu_compositor_size(
                            image.width().max(1),
                            image.height().max(1),
                        )
                        .await?;
                        let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                            MotionLoomSceneRenderError::GpuRender {
                                message: "GPU compositor was not initialized".to_string(),
                            }
                        })?;
                        return Ok(GraphTextureSource::Cpu(
                            compositor
                                .apply_gpu_blur_passes(image, &[(true, sigma), (false, sigma)])
                                .await?,
                        ));
                    }
                }
            }
            let image = self.graph_source_to_cpu(&inputs[0]).await?;
            let blurred = apply_box_blur_pass(&image, sigma, true);
            return Ok(GraphTextureSource::Cpu(apply_box_blur_pass(
                &blurred, sigma, false,
            )));
        }
        if scene_post_glow_stack_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_glow_stack_pass(&inputs[0], pass, time_norm, time_sec)
                .await;
        }
        if scene_post_tone_map_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_tone_map_pass(&inputs[0], pass, time_norm, time_sec)
                .await;
        }
        if scene_post_light_sweep_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_light_sweep_pass(&inputs[0], pass, time_norm, time_sec)
                .await;
        }
        if scene_post_magnify_lens_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_magnify_lens_pass(&inputs[0], pass, time_norm, time_sec)
                .await;
        }
        if scene_post_texture_overlay_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_texture_overlay_pass(&inputs[0], pass, time_norm, time_sec)
                .await;
        }
        if scene_post_brightness_amount(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_brightness_pass(&inputs[0], pass, time_norm, time_sec)
                .await;
        }
        if scene_post_bloom_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_bloom_pass(&inputs[0], pass, time_norm, time_sec)
                .await;
        }
        self.apply_scene_post_pass(&inputs[0], pass, time_norm, time_sec)
            .await
    }

    async fn apply_scene_post_pass(
        &mut self,
        input: &GraphTextureSource,
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        if let Some(blur_passes) = scene_post_blur_passes(pass, time_norm, time_sec)?
            && self.profile.uses_gpu_compositor()
        {
            if blur_passes.iter().all(|(_, sigma)| *sigma <= 0.05) {
                return Ok(input.clone());
            }
            match input {
                GraphTextureSource::Gpu(texture) => {
                    self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                        .await?;
                    let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                        MotionLoomSceneRenderError::GpuRender {
                            message: "GPU compositor was not initialized".to_string(),
                        }
                    })?;
                    return Ok(GraphTextureSource::Gpu(
                        compositor.apply_gpu_blur_texture(texture, &blur_passes)?,
                    ));
                }
                GraphTextureSource::Cpu(image) => {
                    self.ensure_gpu_compositor_size(image.width().max(1), image.height().max(1))
                        .await?;
                    let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                        MotionLoomSceneRenderError::GpuRender {
                            message: "GPU compositor was not initialized".to_string(),
                        }
                    })?;
                    return Ok(GraphTextureSource::Cpu(
                        compositor
                            .apply_gpu_blur_passes(image, &blur_passes)
                            .await?,
                    ));
                }
            }
        }
        if scene_post_bloom_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_bloom_pass(input, pass, time_norm, time_sec)
                .await;
        }
        if scene_post_glow_stack_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_glow_stack_pass(input, pass, time_norm, time_sec)
                .await;
        }
        if scene_post_tone_map_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_tone_map_pass(input, pass, time_norm, time_sec)
                .await;
        }
        if scene_post_light_sweep_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_light_sweep_pass(input, pass, time_norm, time_sec)
                .await;
        }
        if scene_post_magnify_lens_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_magnify_lens_pass(input, pass, time_norm, time_sec)
                .await;
        }
        if scene_post_texture_overlay_params(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_texture_overlay_pass(input, pass, time_norm, time_sec)
                .await;
        }
        if scene_post_brightness_amount(pass, time_norm, time_sec)?.is_some() {
            return self
                .apply_scene_brightness_pass(input, pass, time_norm, time_sec)
                .await;
        }
        if matches!(
            crate::process::effect_kind::resolve_process_effect(&pass.effect),
            Some(crate::process::effect_kind::ProcessEffect::SpectralEnergy)
        ) {
            return self
                .apply_scene_spectral_energy_pass(input, pass, time_norm, time_sec)
                .await;
        }
        let effect = pass.effect.to_ascii_lowercase();
        if effect == "opacity" || effect == "composite.opacity" {
            let opacity = pass_param_expr(pass, "opacity")
                .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                .transpose()?
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            if self.profile.uses_gpu_compositor() {
                let image = self.graph_source_to_cpu(input).await?;
                self.ensure_gpu_compositor_size(image.width().max(1), image.height().max(1))
                    .await?;
                let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                    MotionLoomSceneRenderError::GpuRender {
                        message: "GPU compositor was not initialized".to_string(),
                    }
                })?;
                return Ok(GraphTextureSource::Cpu(
                    compositor.apply_gpu_opacity_pass(&image, opacity).await?,
                ));
            }
        }
        if is_color_key_alpha_effect(&effect) {
            let image = self.graph_source_to_cpu(input).await?;
            return apply_scene_post_pass(&image, pass, time_norm, time_sec)
                .map(GraphTextureSource::Cpu);
        }
        if self.profile.uses_gpu_compositor() {
            return Err(MotionLoomSceneRenderError::GpuRender {
                message: format!(
                    "GPU Render is strict for MotionLoom: post pass '{}' is not GPU-native yet. Use Compatibility Render (CPU) explicitly, or remove/replace this pass.",
                    pass.effect
                ),
            });
        }
        let image = self.graph_source_to_cpu(input).await?;
        apply_scene_post_pass(&image, pass, time_norm, time_sec).map(GraphTextureSource::Cpu)
    }

    async fn apply_scene_bloom_pass(
        &mut self,
        input: &GraphTextureSource,
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let Some(params) = scene_post_bloom_params(pass, time_norm, time_sec)? else {
            return Ok(input.clone());
        };

        // GPU-native path: avoid CPU readback by keeping the texture on the GPU.
        if self.profile.uses_gpu_compositor() {
            if let GraphTextureSource::Gpu(original) = input {
                self.ensure_gpu_compositor_size(original.width.max(1), original.height.max(1))
                    .await?;
                let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                    MotionLoomSceneRenderError::GpuRender {
                        message: "GPU compositor was not initialized".to_string(),
                    }
                })?;
                let bloom_scale =
                    if original.width >= 1280 && original.height >= 720 && params.sigma >= 10.0 {
                        0.25
                    } else {
                        0.5
                    };
                let result = compositor.apply_gpu_bloom_texture_low_res(
                    original,
                    params.threshold,
                    params.intensity,
                    params.sigma,
                    bloom_scale,
                )?;
                return Ok(GraphTextureSource::Gpu(result));
            }
        }

        // CPU fallback path.
        let image = self.graph_source_to_cpu(input).await?;
        let prefiltered = build_scene_bloom_prefilter(&image, params.threshold);
        let blurred = if self.profile.uses_gpu_compositor() {
            self.ensure_gpu_compositor_size(image.width().max(1), image.height().max(1))
                .await?;
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            compositor
                .apply_gpu_blur_passes(&prefiltered, &[(true, params.sigma), (false, params.sigma)])
                .await?
        } else {
            let blurred_h = apply_box_blur_pass(&prefiltered, params.sigma, true);
            apply_box_blur_pass(&blurred_h, params.sigma, false)
        };
        Ok(GraphTextureSource::Cpu(composite_scene_bloom(
            &image,
            &blurred,
            params.intensity,
        )))
    }

    async fn apply_scene_glow_stack_pass(
        &mut self,
        input: &GraphTextureSource,
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let Some(params) = scene_post_glow_stack_params(pass, time_norm, time_sec)? else {
            return Ok(input.clone());
        };
        if self.profile.uses_gpu_compositor()
            && let GraphTextureSource::Gpu(texture) = input
        {
            self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                .await?;
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            return Ok(GraphTextureSource::Gpu(
                compositor.apply_gpu_glow_stack_texture(
                    texture,
                    params.threshold,
                    params.intensity,
                    params.radius_small,
                    params.radius_medium,
                    params.radius_large,
                    params.tint,
                )?,
            ));
        }
        let image = self.graph_source_to_cpu(input).await?;
        apply_scene_post_pass(&image, pass, time_norm, time_sec).map(GraphTextureSource::Cpu)
    }

    async fn apply_scene_tone_map_pass(
        &mut self,
        input: &GraphTextureSource,
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let Some(params) = scene_post_tone_map_params(pass, time_norm, time_sec)? else {
            return Ok(input.clone());
        };
        if self.profile.uses_gpu_compositor()
            && let GraphTextureSource::Gpu(texture) = input
        {
            self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                .await?;
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            return Ok(GraphTextureSource::Gpu(
                compositor.apply_gpu_tone_map_texture(
                    texture,
                    params.exposure,
                    params.contrast,
                    params.shoulder,
                    params.gamma,
                    params.saturation,
                )?,
            ));
        }
        let image = self.graph_source_to_cpu(input).await?;
        apply_scene_post_pass(&image, pass, time_norm, time_sec).map(GraphTextureSource::Cpu)
    }

    async fn apply_scene_brightness_pass(
        &mut self,
        input: &GraphTextureSource,
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let Some(brightness) = scene_post_brightness_amount(pass, time_norm, time_sec)? else {
            return Ok(input.clone());
        };
        if self.profile.uses_gpu_compositor()
            && let GraphTextureSource::Gpu(texture) = input
        {
            self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                .await?;
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            return Ok(GraphTextureSource::Gpu(
                compositor.apply_gpu_color_texture(texture, brightness, 1.0, 1.0)?,
            ));
        }
        let image = self.graph_source_to_cpu(input).await?;
        Ok(GraphTextureSource::Cpu(apply_color_core_pass(
            &image, brightness, 1.0, 1.0,
        )))
    }

    async fn apply_scene_spectral_energy_pass(
        &mut self,
        input: &GraphTextureSource,
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let mode = pass
            .mode
            .as_deref()
            .or_else(|| pass_param_expr(pass, "mode"))
            .map(|value| {
                value
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_ascii_lowercase()
            })
            .map(|value| match value.as_str() {
                "granular" | "granular_light_field" | "light_field" => 1.0,
                _ => 0.0,
            })
            .unwrap_or(0.0);
        let active = pass_param_expr(pass, "active")
            .or_else(|| pass_param_expr(pass, "mix"))
            .map(|expr| eval_scene_number(expr, time_norm, time_sec))
            .transpose()?
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let intensity = pass_param_expr(pass, "intensity")
            .map(|expr| eval_scene_number(expr, time_norm, time_sec))
            .transpose()?
            .unwrap_or(1.0)
            .max(0.0);
        let density = pass_param_expr(pass, "density")
            .or_else(|| pass_param_expr(pass, "grain"))
            .map(|expr| eval_scene_number(expr, time_norm, time_sec))
            .transpose()?
            .unwrap_or(1.0)
            .max(0.05);
        let drift = pass_param_expr(pass, "drift")
            .or_else(|| pass_param_expr(pass, "motion"))
            .map(|expr| eval_scene_number(expr, time_norm, time_sec))
            .transpose()?
            .unwrap_or(0.0);
        let seed = pass_param_expr(pass, "seed")
            .map(|expr| eval_scene_number(expr, time_norm, time_sec))
            .transpose()?
            .unwrap_or(67.0);

        if self.profile.uses_gpu_compositor()
            && let GraphTextureSource::Gpu(texture) = input
        {
            self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                .await?;
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            return Ok(GraphTextureSource::Gpu(
                compositor.apply_gpu_spectral_energy_texture(
                    texture, mode, active, intensity, density, drift, seed, time_sec,
                )?,
            ));
        }

        // Compatibility Render intentionally keeps the authored Scene intact;
        // this procedural look is a WebGPU effect.
        Ok(input.clone())
    }

    async fn apply_scene_light_sweep_pass(
        &mut self,
        input: &GraphTextureSource,
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let Some(params) = scene_post_light_sweep_params(pass, time_norm, time_sec)? else {
            return Ok(input.clone());
        };
        if self.profile.uses_gpu_compositor()
            && let GraphTextureSource::Gpu(texture) = input
        {
            self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                .await?;
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            return Ok(GraphTextureSource::Gpu(
                compositor.apply_gpu_light_sweep_texture(
                    texture,
                    params.position,
                    params.angle,
                    params.width,
                    params.softness,
                    params.intensity,
                    params.color,
                )?,
            ));
        }
        let image = self.graph_source_to_cpu(input).await?;
        apply_scene_post_pass(&image, pass, time_norm, time_sec).map(GraphTextureSource::Cpu)
    }

    async fn apply_scene_texture_overlay_pass(
        &mut self,
        input: &GraphTextureSource,
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let Some(params) = scene_post_texture_overlay_params(pass, time_norm, time_sec)? else {
            return Ok(input.clone());
        };
        let (params, texture_image, height_image) = self.resolve_texture_overlay_assets(params)?;
        if self.profile.uses_gpu_compositor()
            && let GraphTextureSource::Gpu(texture) = input
        {
            self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                .await?;
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            if texture_image.is_some() || height_image.is_some() {
                return Ok(GraphTextureSource::Gpu(
                    compositor.apply_gpu_image_texture_overlay_texture(
                        texture,
                        params,
                        texture_image.as_ref(),
                        height_image.as_ref(),
                    )?,
                ));
            }
            return Ok(GraphTextureSource::Gpu(
                compositor.apply_gpu_texture_overlay_texture(texture, params)?,
            ));
        }
        if texture_image.is_some() || height_image.is_some() {
            let image = self.graph_source_to_cpu(input).await?;
            return Ok(GraphTextureSource::Cpu(apply_image_texture_overlay_pass(
                &image,
                &params,
                texture_image.as_ref(),
                height_image.as_ref(),
            )));
        }
        let image = self.graph_source_to_cpu(input).await?;
        apply_scene_post_pass(&image, pass, time_norm, time_sec).map(GraphTextureSource::Cpu)
    }

    async fn apply_scene_magnify_lens_pass(
        &mut self,
        input: &GraphTextureSource,
        pass: &PassNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GraphTextureSource, MotionLoomSceneRenderError> {
        let Some(params) = scene_post_magnify_lens_params(pass, time_norm, time_sec)? else {
            return Ok(input.clone());
        };
        if self.profile.uses_gpu_compositor()
            && let GraphTextureSource::Gpu(texture) = input
        {
            self.ensure_gpu_compositor_size(texture.width.max(1), texture.height.max(1))
                .await?;
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            return Ok(GraphTextureSource::Gpu(
                compositor.apply_gpu_magnify_lens_texture(texture, params)?,
            ));
        }
        let image = self.graph_source_to_cpu(input).await?;
        apply_scene_post_pass(&image, pass, time_norm, time_sec).map(GraphTextureSource::Cpu)
    }

    fn resolve_texture_overlay_assets(
        &mut self,
        mut params: crate::scene::composition::SceneTextureOverlayParams,
    ) -> Result<
        (
            crate::scene::composition::SceneTextureOverlayParams,
            Option<RgbaImage>,
            Option<RgbaImage>,
        ),
        MotionLoomSceneRenderError,
    > {
        if let Some(texture_ref) = params.texture_ref.as_deref()
            && let Some(texture_def) = self.texture_defs.get(texture_ref)
            && !texture_def.src.is_empty()
        {
            if texture_def.kind.eq_ignore_ascii_case("height") {
                params
                    .height_src
                    .get_or_insert_with(|| texture_def.src.clone());
            } else {
                params
                    .texture_src
                    .get_or_insert_with(|| texture_def.src.clone());
            }
        }
        if let Some(height_ref) = params.height_ref.as_deref()
            && let Some(texture_def) = self.texture_defs.get(height_ref)
            && !texture_def.src.is_empty()
        {
            params
                .height_src
                .get_or_insert_with(|| texture_def.src.clone());
        }
        let texture_image = match params.texture_src.clone() {
            Some(src) => Some(self.load_image_asset(&src)?.clone()),
            None => None,
        };
        let height_image = match params.height_src.clone() {
            Some(src) => Some(self.load_image_asset(&src)?.clone()),
            None => None,
        };
        Ok((params, texture_image, height_image))
    }

    fn draw_scene_nodes(
        &mut self,
        canvas: &mut RgbaImage,
        nodes: &[SceneNode],
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        self.draw_scene_nodes_with_depth(
            canvas,
            nodes,
            time_norm,
            time_sec,
            inherited_opacity,
            None,
        )
    }

    fn draw_scene_nodes_with_depth(
        &mut self,
        canvas: &mut RgbaImage,
        nodes: &[SceneNode],
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
        depth: Option<SceneDepthContext<'_>>,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let mut pending_shadow: Option<EvaluatedShadow> = None;
        for node in nodes {
            match node {
                SceneNode::Defs(defs) => {
                    self.register_defs_resources(defs)?;
                    pending_shadow = None;
                }
                SceneNode::Timeline(timeline) => {
                    let mut tracks = timeline
                        .children
                        .iter()
                        .filter_map(|node| match node {
                            SceneNode::Track(track) => Some(track),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    tracks.sort_by_key(|track| scene_track_composite_order(track));
                    let active_camera = active_scene_camera_from_tracks(
                        &tracks,
                        canvas.width(),
                        canvas.height(),
                        time_norm,
                        time_sec,
                    )?;
                    tracks.sort_by(|a, b| {
                        let a_world = is_scene_world_track(a);
                        let b_world = is_scene_world_track(b);
                        match (a_world, b_world) {
                            (true, true) => {
                                let a_depth =
                                    scene_depth_track_sort_key(&a.z_depth, time_norm, time_sec)
                                        .unwrap_or(0.0);
                                let b_depth =
                                    scene_depth_track_sort_key(&b.z_depth, time_norm, time_sec)
                                        .unwrap_or(0.0);
                                b_depth.total_cmp(&a_depth).then_with(|| {
                                    scene_track_composite_order(a)
                                        .cmp(&scene_track_composite_order(b))
                                })
                            }
                            _ => {
                                scene_track_composite_order(a).cmp(&scene_track_composite_order(b))
                            }
                        }
                    });
                    for track in tracks {
                        if is_scene_camera_track(track) {
                            continue;
                        }
                        if is_scene_world_track(track) {
                            self.draw_scene_nodes_with_depth(
                                canvas,
                                &track.children,
                                time_norm,
                                time_sec,
                                inherited_opacity,
                                Some(SceneDepthContext {
                                    active_camera,
                                    canvas_size: canvas.dimensions(),
                                    track_z_depth: &track.z_depth,
                                }),
                            )?;
                        } else {
                            self.draw_scene_nodes_with_depth(
                                canvas,
                                &track.children,
                                time_norm,
                                time_sec,
                                inherited_opacity,
                                depth,
                            )?;
                        }
                    }
                    pending_shadow = None;
                }
                SceneNode::Track(track) => {
                    self.draw_scene_nodes_with_depth(
                        canvas,
                        &track.children,
                        time_norm,
                        time_sec,
                        inherited_opacity,
                        depth,
                    )?;
                    pending_shadow = None;
                }
                SceneNode::Sequence(sequence) => {
                    if let Some((local_norm, local_sec)) =
                        scene_sequence_local_time(sequence, None, time_sec)
                    {
                        if depth.is_some() {
                            self.draw_depth_sorted_scene_nodes(
                                canvas,
                                &sequence.children,
                                local_norm,
                                local_sec,
                                inherited_opacity,
                                depth,
                            )?;
                        } else {
                            self.draw_scene_nodes_with_depth(
                                canvas,
                                &sequence.children,
                                local_norm,
                                local_sec,
                                inherited_opacity,
                                depth,
                            )?;
                        }
                    }
                    pending_shadow = None;
                }
                SceneNode::Chain(chain) => {
                    let mut cursor_ms = chain.from_ms as i64;
                    for child in &chain.children {
                        if let SceneNode::Sequence(sequence) = child {
                            if let Some((local_norm, local_sec)) =
                                scene_sequence_local_time(sequence, Some(cursor_ms), time_sec)
                            {
                                if depth.is_some() {
                                    self.draw_depth_sorted_scene_nodes(
                                        canvas,
                                        &sequence.children,
                                        local_norm,
                                        local_sec,
                                        inherited_opacity,
                                        depth,
                                    )?;
                                } else {
                                    self.draw_scene_nodes_with_depth(
                                        canvas,
                                        &sequence.children,
                                        local_norm,
                                        local_sec,
                                        inherited_opacity,
                                        depth,
                                    )?;
                                }
                            }
                            cursor_ms += sequence.duration_ms as i64 + chain.gap_ms;
                        }
                    }
                    pending_shadow = None;
                }
                SceneNode::Palette(_) => {
                    pending_shadow = None;
                }
                SceneNode::PixelGrid(grid) => {
                    self.draw_pixel_grid(
                        canvas,
                        grid,
                        Affine2::identity(),
                        inherited_opacity,
                        time_norm,
                        time_sec,
                    )?;
                    pending_shadow = None;
                }
                SceneNode::Text(text) => {
                    self.draw_text_with_opacity(
                        canvas,
                        text,
                        time_norm,
                        time_sec,
                        inherited_opacity,
                    )?;
                    pending_shadow = None;
                }
                SceneNode::Image(image) => {
                    self.draw_image_with_opacity(
                        canvas,
                        image,
                        time_norm,
                        time_sec,
                        inherited_opacity,
                    )?;
                    pending_shadow = None;
                }
                SceneNode::Svg(svg) => {
                    self.draw_svg_with_opacity(
                        canvas,
                        svg,
                        time_norm,
                        time_sec,
                        inherited_opacity,
                    )?;
                    pending_shadow = None;
                }
                SceneNode::Rect(rect) => {
                    self.draw_rect(
                        canvas,
                        rect,
                        pending_shadow.take(),
                        time_norm,
                        time_sec,
                        inherited_opacity,
                    )?;
                }
                SceneNode::Circle(circle) => {
                    self.draw_circle(
                        canvas,
                        circle,
                        pending_shadow.take(),
                        time_norm,
                        time_sec,
                        inherited_opacity,
                    )?;
                }
                SceneNode::Ellipse(ellipse) => {
                    self.draw_ellipse(canvas, ellipse, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::Line(line) => {
                    self.draw_line(canvas, line, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::Polyline(polyline) => {
                    self.draw_polyline(canvas, polyline, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::Path(path) => {
                    self.draw_path(canvas, path, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::FaceJaw(face_jaw) => {
                    self.draw_face_jaw(
                        canvas,
                        face_jaw,
                        Affine2::identity(),
                        time_norm,
                        time_sec,
                        inherited_opacity,
                    )?;
                    pending_shadow = None;
                }
                SceneNode::Shadow(shadow) => {
                    pending_shadow = Some(evaluate_shadow(
                        shadow,
                        time_norm,
                        time_sec,
                        inherited_opacity,
                    )?);
                }
                SceneNode::Group(group) => {
                    self.draw_group(canvas, group, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::Puppet(puppet) => {
                    self.draw_puppet(canvas, puppet, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::Pin(_)
                | SceneNode::LimbEnvelope(_)
                | SceneNode::LimbRegion(_)
                | SceneNode::MeshTopology(_)
                | SceneNode::Vertex(_)
                | SceneNode::Triangle(_)
                | SceneNode::Edge(_)
                | SceneNode::Region(_) => {
                    pending_shadow = None;
                }
                SceneNode::Part(part) => {
                    self.draw_part(canvas, part, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::Repeat(repeat) => {
                    self.draw_repeat(canvas, repeat, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::Mask(mask) => {
                    if let Some(id) = mask.id.as_deref() {
                        self.scene_masks.insert(id.to_string(), mask.clone());
                    }
                    if !mask.children.is_empty() {
                        self.draw_mask(canvas, mask, time_norm, time_sec, inherited_opacity)?;
                    }
                    pending_shadow = None;
                }
                SceneNode::Precompose(precompose) => {
                    self.scene_precompose_defs
                        .insert(precompose.id.clone(), precompose.clone());
                    pending_shadow = None;
                }
                SceneNode::Use(use_node) => {
                    self.draw_use(canvas, use_node, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::Layer(layer) => {
                    if let Some(depth) = depth {
                        let z_depth =
                            scene_layer_effective_z_depth(layer, depth, time_norm, time_sec)?;
                        let depth_transform = scene_z_depth_transform(
                            depth.active_camera,
                            depth.canvas_size,
                            z_depth,
                        );
                        let source_size = depth
                            .active_camera
                            .map(|camera| (camera.layer_width, camera.layer_height))
                            .unwrap_or_else(|| canvas.dimensions());
                        let clip = depth.active_camera.map(|camera| camera.viewport);
                        self.draw_scene_layer_with_transform(
                            canvas,
                            layer,
                            SceneLayerDrawParams {
                                source_size,
                                base_transform: depth_transform,
                                clip,
                                time_norm,
                                time_sec,
                                inherited_opacity,
                            },
                        )?;
                    } else {
                        self.draw_scene_layer(
                            canvas,
                            layer,
                            time_norm,
                            time_sec,
                            inherited_opacity,
                        )?;
                    }
                    pending_shadow = None;
                }
                SceneNode::Camera(camera) => {
                    self.draw_camera(canvas, camera, time_norm, time_sec, inherited_opacity)?;
                    pending_shadow = None;
                }
                SceneNode::Character(character) => {
                    self.draw_character(
                        canvas,
                        character,
                        Affine2::identity(),
                        inherited_opacity,
                        time_norm,
                        time_sec,
                    )?;
                    pending_shadow = None;
                }
                SceneNode::Simulation(_) => {}
            }
        }
        Ok(())
    }

    fn draw_depth_sorted_scene_nodes(
        &mut self,
        canvas: &mut RgbaImage,
        nodes: &[SceneNode],
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
        depth: Option<SceneDepthContext<'_>>,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let Some(depth) = depth else {
            return self.draw_scene_nodes_with_depth(
                canvas,
                nodes,
                time_norm,
                time_sec,
                inherited_opacity,
                None,
            );
        };
        let mut layer_items = Vec::<(usize, f32, &SceneNode)>::new();
        for (index, node) in nodes.iter().enumerate() {
            if let SceneNode::Layer(layer) = node {
                layer_items.push((
                    index,
                    scene_layer_effective_z_depth(layer, depth, time_norm, time_sec)?,
                    node,
                ));
            } else {
                self.draw_scene_nodes_with_depth(
                    canvas,
                    std::slice::from_ref(node),
                    time_norm,
                    time_sec,
                    inherited_opacity,
                    Some(depth),
                )?;
            }
        }
        layer_items.sort_by(|(a_order, a_depth, _), (b_order, b_depth, _)| {
            b_depth
                .total_cmp(a_depth)
                .then_with(|| a_order.cmp(b_order))
        });
        for (_, _, node) in layer_items {
            self.draw_scene_nodes_with_depth(
                canvas,
                std::slice::from_ref(node),
                time_norm,
                time_sec,
                inherited_opacity,
                Some(depth),
            )?;
        }
        Ok(())
    }

    fn register_defs_resources(
        &mut self,
        defs: &DefsNode,
    ) -> Result<(), MotionLoomSceneRenderError> {
        for font in &defs.fonts {
            self.font_defs.insert(font.id.clone(), font.clone());
        }
        for filter in &defs.filters {
            self.filter_defs.insert(filter.id.clone(), filter.clone());
        }
        for mask in &defs.masks {
            if let Some(id) = mask.id.as_deref() {
                self.scene_masks.insert(id.to_string(), mask.clone());
            }
        }
        for component in &defs.components {
            self.scene_components
                .insert(component.id.clone(), component.children.clone());
        }
        for precompose in &defs.precomposes {
            self.scene_precompose_defs
                .insert(precompose.id.clone(), precompose.clone());
        }
        Ok(())
    }

    fn render_cpu_base_frame(
        &mut self,
        graph: &GraphScript,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let (w, h) = graph.size;
        let mut canvas = RgbaImage::from_pixel(w.max(1), h.max(1), Rgba([0, 0, 0, 0]));

        for background in &graph.backgrounds {
            let color = parse_color(&background.color)?;
            for pixel in canvas.pixels_mut() {
                *pixel = Rgba(color);
            }
        }

        for image in &graph.images {
            self.draw_image(&mut canvas, image, time_norm, time_sec)?;
        }
        for svg in &graph.svgs {
            self.draw_svg(&mut canvas, svg, time_norm, time_sec)?;
        }
        Ok(canvas)
    }

    async fn render_gpu_base_frame(
        &mut self,
        graph: &GraphScript,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        self.ensure_gpu_compositor(graph.size.0.max(1), graph.size.1.max(1))
            .await?;

        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let background = graph
            .backgrounds
            .last()
            .map(|background| parse_color(&background.color))
            .transpose()?
            .unwrap_or([0, 0, 0, 0]);
        compositor
            .render(graph, background, time_norm, time_sec)
            .await
    }

    async fn render_gpu_base_frame_to_texture(
        &mut self,
        graph: &GraphScript,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        self.ensure_gpu_compositor(graph.size.0.max(1), graph.size.1.max(1))
            .await?;

        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        let background = graph
            .backgrounds
            .last()
            .map(|background| parse_color(&background.color))
            .transpose()?
            .unwrap_or([0, 0, 0, 0]);
        compositor
            .render_to_texture(graph, background, time_norm, time_sec)
            .await
    }

    async fn ensure_gpu_compositor(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        if self.gpu_compositor.is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                if let Some((device, queue)) = self.external_device_queue.clone() {
                    self.gpu_compositor = Some(
                        WgpuSceneCompositor::new_with_device(
                            device,
                            queue,
                            width,
                            height,
                            self.asset_resolver.clone(),
                        )
                        .await?,
                    );
                } else {
                    self.gpu_compositor = Some(
                        WgpuSceneCompositor::new(width, height, self.asset_resolver.clone())
                            .await?,
                    );
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.gpu_compositor = Some(
                    WgpuSceneCompositor::new(width, height, self.asset_resolver.clone()).await?,
                );
            }
        }
        Ok(())
    }

    async fn render_gpu_background_texture(
        &mut self,
        output_size: (u32, u32),
        color: [u8; 4],
    ) -> Result<GpuSceneNativeTexture, MotionLoomSceneRenderError> {
        self.ensure_gpu_compositor_size(output_size.0.max(1), output_size.1.max(1))
            .await?;
        let compositor =
            self.gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?;
        compositor.render_scene_content_to_texture(&[gpu_solid_primitive(color)], &[], [0, 0, 0, 0])
    }

    fn draw_text(
        &mut self,
        canvas: &mut RgbaImage,
        text: &TextNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        self.draw_text_with_opacity(canvas, text, time_norm, time_sec, 1.0)
    }

    fn draw_text_with_opacity(
        &mut self,
        canvas: &mut RgbaImage,
        text: &TextNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        self.draw_text_transformed(
            canvas,
            text,
            Affine2::identity(),
            inherited_opacity,
            time_norm,
            time_sec,
        )
    }

    fn draw_image(
        &mut self,
        canvas: &mut RgbaImage,
        image_node: &ImageNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        self.draw_image_with_opacity(canvas, image_node, time_norm, time_sec, 1.0)
    }

    fn draw_image_with_opacity(
        &mut self,
        canvas: &mut RgbaImage,
        image_node: &ImageNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&image_node.opacity, time_norm, time_sec)?
            * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }

        let scale = eval_scene_number(&image_node.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
        let source = self.load_image_asset(&image_node.src)?;
        let target_w = ((source.width() as f32) * scale).round().max(1.0) as u32;
        let target_h = ((source.height() as f32) * scale).round().max(1.0) as u32;

        let x_base = resolve_axis(
            &image_node.x,
            canvas.width() as f32,
            target_w as f32,
            time_norm,
            time_sec,
        )?;
        let y_base = resolve_axis(
            &image_node.y,
            canvas.height() as f32,
            target_h as f32,
            time_norm,
            time_sec,
        )?;
        if target_w == source.width() && target_h == source.height() {
            draw_rgba_image(canvas, source, x_base, y_base, opacity);
        } else {
            let scaled = image::imageops::resize(source, target_w, target_h, FilterType::Lanczos3);
            draw_rgba_image(canvas, &scaled, x_base, y_base, opacity);
        }
        Ok(())
    }

    fn draw_svg(
        &mut self,
        canvas: &mut RgbaImage,
        svg_node: &SvgNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        self.draw_svg_with_opacity(canvas, svg_node, time_norm, time_sec, 1.0)
    }

    fn draw_svg_with_opacity(
        &mut self,
        canvas: &mut RgbaImage,
        svg_node: &SvgNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&svg_node.opacity, time_norm, time_sec)?
            * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }

        let scale = eval_scene_number(&svg_node.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
        let source = self.load_svg_asset(&svg_node.src)?;
        let target_w = ((source.width() as f32) * scale).round().max(1.0) as u32;
        let target_h = ((source.height() as f32) * scale).round().max(1.0) as u32;

        let x_base = resolve_axis(
            &svg_node.x,
            canvas.width() as f32,
            target_w as f32,
            time_norm,
            time_sec,
        )?;
        let y_base = resolve_axis(
            &svg_node.y,
            canvas.height() as f32,
            target_h as f32,
            time_norm,
            time_sec,
        )?;
        if target_w == source.width() && target_h == source.height() {
            draw_rgba_image(canvas, source, x_base, y_base, opacity);
        } else {
            let scaled = image::imageops::resize(source, target_w, target_h, FilterType::Lanczos3);
            draw_rgba_image(canvas, &scaled, x_base, y_base, opacity);
        }
        Ok(())
    }

    fn draw_group(
        &mut self,
        canvas: &mut RgbaImage,
        group: &GroupNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&group.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let transform = scene_group_local_transform(group, time_norm, time_sec)?;
        let deform_grid = eval_group_deform_grid(group, time_norm, time_sec)?;

        if group.mask.is_none()
            && group.mask_from.is_none()
            && deform_grid.is_none()
            && group.effects.is_empty()
        {
            self.draw_character_nodes_vector(
                canvas,
                &group.children,
                transform,
                opacity,
                time_norm,
                time_sec,
            )?;
            return Ok(());
        }

        let mut layer = RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
        self.draw_scene_nodes(&mut layer, &group.children, time_norm, time_sec, opacity)?;
        for filter_id in &group.effects {
            layer = self.apply_scene_filter(&layer, filter_id, time_norm, time_sec)?;
        }
        let mask_feather = eval_scene_number(&group.mask_feather, time_norm, time_sec)?.max(0.0);
        let mask_expansion = eval_scene_number(&group.mask_expansion, time_norm, time_sec)?;
        if let Some(mask_id) = group.mask.as_deref()
            && let Some(mask_alpha) = self.scene_mask_alpha(
                mask_id,
                canvas.width(),
                canvas.height(),
                time_norm,
                time_sec,
            )?
        {
            let invert = group.mask_mode.trim().eq_ignore_ascii_case("inverse")
                || group.mask_mode.trim().eq_ignore_ascii_case("invert")
                || group.mask_mode.trim().eq_ignore_ascii_case("inverted");
            let mask_alpha = shape_alpha_mask(&mask_alpha, mask_expansion, mask_feather);
            apply_alpha_mask_with_invert(&mut layer, &mask_alpha, invert);
        } else if let Some(mask_from) = group.mask_from.as_deref()
            && let Some(mask_alpha) = self.scene_matte_alpha(
                mask_from,
                (canvas.width(), canvas.height()),
                &group.mask_mode,
                None,
                time_norm,
                time_sec,
            )?
        {
            let mask_alpha = shape_alpha_mask(&mask_alpha, mask_expansion, mask_feather);
            apply_alpha_mask_with_invert(
                &mut layer,
                &mask_alpha,
                scene_mask_mode_inverts(&group.mask_mode),
            );
        }
        if let Some(deform_grid) = deform_grid {
            layer = apply_deform_grid(&layer, &deform_grid);
        }
        composite_layer_affine(canvas, &layer, transform);
        Ok(())
    }

    fn draw_puppet(
        &mut self,
        canvas: &mut RgbaImage,
        puppet: &PuppetNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&puppet.opacity, time_norm, time_sec)?
            * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let transform = scene_puppet_local_transform(puppet, time_norm, time_sec)?;
        let deform_grid = eval_puppet_deform_grid(puppet, time_norm, time_sec)?;

        let mut layer = RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
        self.draw_scene_nodes(&mut layer, &puppet.children, time_norm, time_sec, opacity)?;
        if let Some(deform_grid) = deform_grid {
            layer = apply_deform_grid(&layer, &deform_grid);
        }
        composite_layer_affine(canvas, &layer, transform);
        Ok(())
    }

    fn render_precompose_image(
        &mut self,
        precompose: &PrecomposeNode,
        fallback_size: (u32, u32),
        time_norm: f32,
        time_sec: f32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let size = precompose.size.unwrap_or(fallback_size);
        let mut layer = RgbaImage::from_pixel(size.0.max(1), size.1.max(1), Rgba([0, 0, 0, 0]));
        self.draw_scene_nodes(&mut layer, &precompose.children, time_norm, time_sec, 1.0)?;
        Ok(layer)
    }

    fn render_precompose_instance(
        &mut self,
        source_id: &str,
        fallback_size: (u32, u32),
        layer: &SceneLayerNode,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<Option<RgbaImage>, MotionLoomSceneRenderError> {
        let Some(precompose) = self.scene_precompose_defs.get(source_id).cloned() else {
            return Ok(self.scene_precomposes.get(source_id).cloned());
        };
        let Some((source_norm, source_sec)) =
            scene_layer_source_time(layer, &precompose, time_norm, time_sec)?
        else {
            return Ok(None);
        };
        self.render_precompose_image(&precompose, fallback_size, source_norm, source_sec)
            .map(Some)
    }

    fn draw_use(
        &mut self,
        canvas: &mut RgbaImage,
        use_node: &UseNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        self.draw_use_transformed(
            canvas,
            use_node,
            Affine2::identity(),
            time_norm,
            time_sec,
            inherited_opacity,
        )
    }

    fn draw_use_transformed(
        &mut self,
        canvas: &mut RgbaImage,
        use_node: &UseNode,
        base_transform: Affine2,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&use_node.opacity, time_norm, time_sec)?
            * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let Some(children) = self.scene_components.get(&use_node.ref_id).cloned() else {
            return Ok(());
        };

        let mut layer = RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
        self.draw_scene_nodes(&mut layer, &children, time_norm, time_sec, opacity)?;
        let transform =
            base_transform.mul(scene_use_local_transform(use_node, time_norm, time_sec)?);
        let blend = parse_scene_blend(&use_node.blend)?;
        composite_layer_affine_blend(canvas, &layer, transform, 1.0, blend);
        Ok(())
    }

    fn draw_scene_layer(
        &mut self,
        canvas: &mut RgbaImage,
        layer: &SceneLayerNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        self.draw_scene_layer_with_transform(
            canvas,
            layer,
            SceneLayerDrawParams {
                source_size: canvas.dimensions(),
                base_transform: Affine2::identity(),
                clip: None,
                time_norm,
                time_sec,
                inherited_opacity,
            },
        )
    }

    fn draw_scene_layer_with_transform(
        &mut self,
        canvas: &mut RgbaImage,
        layer: &SceneLayerNode,
        params: SceneLayerDrawParams,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&layer.opacity, params.time_norm, params.time_sec)?
            * params.inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let Some(source) = self.scene_layer_source_image(
            layer,
            params.source_size,
            params.time_norm,
            params.time_sec,
        )?
        else {
            return Ok(());
        };

        let blend = parse_scene_blend(&layer.blend)?;
        if layer.is_3d {
            let Some((cropped, offset_x, offset_y)) = crop_layer_alpha_bounds(&source) else {
                return Ok(());
            };
            let quad = scene_layer_3d_projected_quad(
                layer,
                params.base_transform,
                (offset_x as f32, offset_y as f32),
                (cropped.width() as f32, cropped.height() as f32),
                params.time_norm,
                params.time_sec,
            )?;
            composite_layer_projected_quad_blend_clipped(
                canvas,
                &cropped,
                quad,
                opacity,
                blend,
                params.clip,
            );
            return Ok(());
        }

        let transform = params.base_transform.mul(scene_layer_local_transform(
            layer,
            params.time_norm,
            params.time_sec,
        )?);
        composite_layer_affine_blend_clipped(
            canvas,
            &source,
            transform,
            opacity,
            blend,
            params.clip,
        );
        Ok(())
    }

    fn scene_layer_source_image(
        &mut self,
        layer: &SceneLayerNode,
        source_size: (u32, u32),
        time_norm: f32,
        time_sec: f32,
    ) -> Result<Option<RgbaImage>, MotionLoomSceneRenderError> {
        let mut source = if let Some(source_id) = layer.source.as_deref() {
            let Some(source) = self.render_precompose_instance(
                source_id,
                source_size,
                layer,
                time_norm,
                time_sec,
            )?
            else {
                return Ok(None);
            };
            source
        } else {
            if layer.children.is_empty() {
                return Ok(None);
            }
            RgbaImage::from_pixel(source_size.0, source_size.1, Rgba([0, 0, 0, 0]))
        };
        if !layer.children.is_empty() {
            self.draw_scene_nodes(&mut source, &layer.children, time_norm, time_sec, 1.0)?;
        }
        if let Some(filter_id) = layer.effect.as_deref() {
            source = self.apply_scene_filter(&source, filter_id, time_norm, time_sec)?;
        }
        let mask_feather = eval_scene_number(&layer.mask_feather, time_norm, time_sec)?.max(0.0);
        let mask_expansion = eval_scene_number(&layer.mask_expansion, time_norm, time_sec)?;
        if let Some(mask_id) = layer.mask.as_deref()
            && let Some(mask_alpha) = self.scene_mask_alpha(
                mask_id,
                source.width(),
                source.height(),
                time_norm,
                time_sec,
            )?
        {
            let mask_alpha = shape_alpha_mask(&mask_alpha, mask_expansion, mask_feather);
            apply_alpha_mask_with_invert(
                &mut source,
                &mask_alpha,
                scene_mask_mode_inverts(&layer.mask_mode),
            );
        }
        if let Some(mask_from) = layer.mask_from.as_deref()
            && let Some(mask_alpha) = self.scene_matte_alpha(
                mask_from,
                (source.width(), source.height()),
                &layer.mask_mode,
                Some(layer),
                time_norm,
                time_sec,
            )?
        {
            let mask_alpha = shape_alpha_mask(&mask_alpha, mask_expansion, mask_feather);
            apply_alpha_mask_with_invert(
                &mut source,
                &mask_alpha,
                scene_mask_mode_inverts(&layer.mask_mode),
            );
        }
        if let Some(matte_id) = layer.matte.as_deref()
            && let Some(matte_alpha) = self.scene_matte_alpha(
                matte_id,
                (source.width(), source.height()),
                &layer.matte_mode,
                Some(layer),
                time_norm,
                time_sec,
            )?
        {
            apply_alpha_mask_with_invert(
                &mut source,
                &matte_alpha,
                scene_bool(&layer.invert_matte),
            );
        }
        if let Some(matte_from) = layer.matte_from.as_deref()
            && let Some(matte_alpha) = self.scene_matte_alpha(
                matte_from,
                (source.width(), source.height()),
                &layer.matte_mode,
                Some(layer),
                time_norm,
                time_sec,
            )?
        {
            apply_alpha_mask_with_invert(
                &mut source,
                &matte_alpha,
                scene_bool(&layer.invert_matte),
            );
        }
        Ok(Some(source))
    }

    fn apply_scene_filter(
        &self,
        input: &RgbaImage,
        filter_id: &str,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let Some(filter) = self.filter_defs.get(filter_id) else {
            return Ok(input.clone());
        };
        let mut output = input.clone();
        for step in &filter.steps {
            output = apply_scene_filter_step(&output, step, time_norm, time_sec)?;
        }
        Ok(output)
    }

    fn scene_mask_alpha(
        &mut self,
        id: &str,
        width: u32,
        height: u32,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<Option<RgbaImage>, MotionLoomSceneRenderError> {
        let Some(mask) = self.scene_masks.get(id).cloned() else {
            return Ok(None);
        };
        self.render_mask_alpha(
            width,
            height,
            &mask,
            Affine2::identity(),
            time_norm,
            time_sec,
        )
        .map(Some)
    }

    fn scene_matte_alpha(
        &mut self,
        id: &str,
        size: (u32, u32),
        mode: &str,
        layer: Option<&SceneLayerNode>,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<Option<RgbaImage>, MotionLoomSceneRenderError> {
        let normalized = mode.trim().to_ascii_lowercase().replace('_', "-");
        let (width, height) = size;
        if let Some(mask) = self.scene_mask_alpha(id, width, height, time_norm, time_sec)? {
            return Ok(Some(mask));
        };
        let matte = if let Some(precompose_layer) = layer {
            self.render_precompose_instance(id, size, precompose_layer, time_norm, time_sec)?
        } else if let Some(precompose) = self.scene_precompose_defs.get(id).cloned() {
            Some(self.render_precompose_image(&precompose, size, time_norm, time_sec)?)
        } else {
            self.scene_precomposes.get(id).cloned()
        };
        let matte = if let Some(matte) = matte {
            Some(matte)
        } else if let Some(node) = self.scene_node_defs.get(id).cloned() {
            Some(self.render_scene_node_def_matte_image(&node, size, time_norm, time_sec)?)
        } else {
            None
        };
        let Some(matte) = matte else {
            return Ok(None);
        };
        let mut alpha =
            RgbaImage::from_pixel(width.max(1), height.max(1), Rgba([255, 255, 255, 0]));
        let w = alpha.width().min(matte.width());
        let h = alpha.height().min(matte.height());
        for y in 0..h {
            for x in 0..w {
                let px = matte.get_pixel(x, y).0;
                let src_alpha = px[3] as f32 / 255.0;
                let amount = if normalized == "luma" || normalized == "luminance" {
                    ((0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32)
                        / 255.0)
                        * src_alpha
                } else {
                    src_alpha
                };
                alpha.put_pixel(
                    x,
                    y,
                    Rgba([
                        255,
                        255,
                        255,
                        (amount * 255.0).round().clamp(0.0, 255.0) as u8,
                    ]),
                );
            }
        }
        Ok(Some(alpha))
    }

    fn render_scene_node_def_matte_image(
        &mut self,
        node: &SceneNode,
        size: (u32, u32),
        time_norm: f32,
        time_sec: f32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let mut image = RgbaImage::from_pixel(size.0.max(1), size.1.max(1), Rgba([0, 0, 0, 0]));
        match node {
            SceneNode::Precompose(precompose) => {
                self.draw_scene_nodes(&mut image, &precompose.children, time_norm, time_sec, 1.0)?;
            }
            _ => {
                self.draw_scene_nodes(
                    &mut image,
                    std::slice::from_ref(node),
                    time_norm,
                    time_sec,
                    1.0,
                )?;
            }
        }
        Ok(image)
    }

    fn draw_part(
        &mut self,
        canvas: &mut RgbaImage,
        part: &PartNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&part.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let x = eval_scene_number(&part.x, time_norm, time_sec)?;
        let y = eval_scene_number(&part.y, time_norm, time_sec)?;
        let rotation = eval_scene_number(&part.rotation, time_norm, time_sec)?;
        let scale = eval_scene_number(&part.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
        let anchor_x = eval_scene_number(&part.anchor_x, time_norm, time_sec)?;
        let anchor_y = eval_scene_number(&part.anchor_y, time_norm, time_sec)?;

        let mut layer = RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
        self.draw_scene_nodes(&mut layer, &part.children, time_norm, time_sec, opacity)?;
        composite_transformed_layer_anchored(
            canvas, &layer, x, y, rotation, scale, anchor_x, anchor_y,
        );
        Ok(())
    }

    fn draw_repeat(
        &mut self,
        canvas: &mut RgbaImage,
        repeat: &RepeatNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let count = eval_repeat_count(&repeat.count, time_norm, time_sec)?;
        if count == 0 {
            return Ok(());
        }

        let x = eval_scene_number(&repeat.x, time_norm, time_sec)?;
        let y = eval_scene_number(&repeat.y, time_norm, time_sec)?;
        let rotation = eval_scene_number(&repeat.rotation, time_norm, time_sec)?;
        let scale = eval_scene_number(&repeat.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
        let opacity = eval_scene_number(&repeat.opacity, time_norm, time_sec)?;
        let x_step = eval_scene_number(&repeat.x_step, time_norm, time_sec)?;
        let y_step = eval_scene_number(&repeat.y_step, time_norm, time_sec)?;
        let rotation_step = eval_scene_number(&repeat.rotation_step, time_norm, time_sec)?;
        let scale_step = eval_scene_number(&repeat.scale_step, time_norm, time_sec)?;
        let opacity_step = eval_scene_number(&repeat.opacity_step, time_norm, time_sec)?;

        for index in 0..count {
            let i = index as f32;
            let copy_opacity = ((opacity + opacity_step * i) * inherited_opacity).clamp(0.0, 1.0);
            if copy_opacity <= 0.0001 {
                continue;
            }
            let copy_scale = (scale + scale_step * i).clamp(0.001, 64.0);
            let mut layer =
                RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
            self.draw_scene_nodes(
                &mut layer,
                &repeat.children,
                time_norm,
                time_sec,
                copy_opacity,
            )?;
            composite_transformed_layer(
                canvas,
                &layer,
                x + x_step * i,
                y + y_step * i,
                rotation + rotation_step * i,
                copy_scale,
            );
        }
        Ok(())
    }

    fn draw_mask(
        &mut self,
        canvas: &mut RgbaImage,
        mask: &MaskNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&mask.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let mut layer = RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
        self.draw_scene_nodes(&mut layer, &mask.children, time_norm, time_sec, opacity)?;
        let mask_alpha = self.render_mask_alpha(
            canvas.width(),
            canvas.height(),
            mask,
            Affine2::identity(),
            time_norm,
            time_sec,
        )?;
        apply_alpha_mask(&mut layer, &mask_alpha);
        composite_layer(canvas, &layer);
        Ok(())
    }

    fn draw_camera(
        &mut self,
        canvas: &mut RgbaImage,
        camera: &CameraNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&camera.opacity, time_norm, time_sec)?
            * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let canvas_w = canvas.width();
        let canvas_h = canvas.height();
        let transform = camera_transform(
            camera,
            &camera.children,
            canvas_w,
            canvas_h,
            time_norm,
            time_sec,
        )?;
        let world_bounds = camera_world_bounds(camera, canvas_w, canvas_h, time_norm, time_sec)?;
        let layer_w = world_bounds
            .map(|rect| canvas_w.max((rect.x + rect.width).ceil().max(canvas_w as f32) as u32 + 2))
            .unwrap_or_else(|| canvas_w.saturating_add(2));
        let layer_h = world_bounds
            .map(|rect| canvas_h.max((rect.y + rect.height).ceil().max(canvas_h as f32) as u32 + 2))
            .unwrap_or_else(|| canvas_h.saturating_add(2));
        let mut layer = RgbaImage::from_pixel(layer_w, layer_h, Rgba([0, 0, 0, 0]));
        self.draw_scene_nodes(&mut layer, &camera.children, time_norm, time_sec, opacity)?;
        let viewport = camera_viewport(camera, canvas_w, canvas_h, time_norm, time_sec)?;
        composite_layer_affine_clipped(canvas, &layer, transform, Some(viewport));
        Ok(())
    }

    fn cached_path_subpaths(
        &mut self,
        data: &str,
    ) -> Result<Vec<Vec<Point2>>, MotionLoomSceneRenderError> {
        if let Some(cached) = self.path_cache.get(data) {
            return Ok(cached.clone());
        }
        let parsed = parse_path_subpaths(data)?;
        self.path_cache.insert(data.to_string(), parsed.clone());
        Ok(parsed)
    }

    fn cached_polyline_points(
        &mut self,
        points: &str,
    ) -> Result<Vec<Point2>, MotionLoomSceneRenderError> {
        if let Some(cached) = self.polyline_cache.get(points) {
            return Ok(cached.clone());
        }
        let parsed = parse_polyline_points(points)?;
        self.polyline_cache
            .insert(points.to_string(), parsed.clone());
        Ok(parsed)
    }

    fn mask_effective_transform(
        &self,
        mask: &MaskNode,
        transform: Affine2,
        time_norm: f32,
        time_sec: f32,
    ) -> Affine2 {
        let Some(follow) = mask.follow.as_deref() else {
            return transform;
        };
        let follow_id = normalize_scene_follow_ref(follow);
        if follow_id.is_empty() {
            return transform;
        }
        let Some((x, y)) = find_scene_node_anchor(
            &self.scene_follow_nodes,
            follow_id.as_str(),
            Affine2::identity(),
            time_norm,
            time_sec,
        ) else {
            return transform;
        };
        transform.mul(Affine2::translate(x, y))
    }

    fn render_mask_alpha(
        &mut self,
        width: u32,
        height: u32,
        mask: &MaskNode,
        transform: Affine2,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<RgbaImage, MotionLoomSceneRenderError> {
        let mut alpha = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));
        let opacity = eval_scene_number(&mask.opacity, time_norm, time_sec)?.clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(alpha);
        }
        let transform = self.mask_effective_transform(mask, transform, time_norm, time_sec);
        let mask_color = [255, 255, 255, (opacity * 255.0).round() as u8];
        match mask.shape.trim().to_ascii_lowercase().as_str() {
            "circle" => {
                let x = eval_scene_number(&mask.x, time_norm, time_sec)?;
                let y = eval_scene_number(&mask.y, time_norm, time_sec)?;
                let radius = eval_scene_number(&mask.radius, time_norm, time_sec)?.max(0.0)
                    * affine_uniform_scale(transform);
                let (x, y) = transform.transform_point(x, y);
                draw_circle(&mut alpha, x, y, radius, mask_color);
            }
            "ellipse" | "oval" => {
                let x = eval_scene_number(&mask.x, time_norm, time_sec)?;
                let y = eval_scene_number(&mask.y, time_norm, time_sec)?;
                let width = eval_scene_number(&mask.width, time_norm, time_sec)?.max(0.0);
                let height = eval_scene_number(&mask.height, time_norm, time_sec)?.max(0.0);
                let subpaths = vec![ellipse_polygon(
                    x + width * 0.5,
                    y + height * 0.5,
                    width * 0.5,
                    height * 0.5,
                )];
                draw_transformed_filled_polylines(&mut alpha, &subpaths, mask_color, transform);
            }
            "path" => {
                let Some(d) = mask.d.as_deref() else {
                    return Ok(alpha);
                };
                let path_d = eval_path_d(d, time_norm, time_sec)?;
                let subpaths = self.cached_path_subpaths(path_d.as_ref())?;
                draw_transformed_filled_polylines(&mut alpha, &subpaths, mask_color, transform);
            }
            _ => {
                let scale = affine_uniform_scale(transform);
                let x = eval_scene_number(&mask.x, time_norm, time_sec)?;
                let y = eval_scene_number(&mask.y, time_norm, time_sec)?;
                let w = eval_scene_number(&mask.width, time_norm, time_sec)?.max(0.0) * scale;
                let h = eval_scene_number(&mask.height, time_norm, time_sec)?.max(0.0) * scale;
                let radius = eval_scene_number(&mask.radius, time_norm, time_sec)?.max(0.0) * scale;
                let (x, y) = transform.transform_point(x, y);
                draw_rounded_rect(&mut alpha, x, y, w, h, radius, mask_color);
            }
        }
        let feather = eval_scene_number(&mask.feather, time_norm, time_sec)?.max(0.0);
        if feather > 0.01 {
            let blurred = apply_box_blur_pass(&alpha, feather, true);
            alpha = apply_box_blur_pass(&blurred, feather, false);
        }
        Ok(alpha)
    }

    fn draw_character(
        &mut self,
        canvas: &mut RgbaImage,
        character: &CharacterNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&character.opacity, time_norm, time_sec)?
            * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let character_transform = transform.mul(scene_character_local_transform(
            character, time_norm, time_sec,
        )?);
        if let Some(src) = character.src.as_deref() {
            // Character-level image sources reuse the same raster loader as <Image>.
            let image = ImageNode {
                id: character.id.clone(),
                material: None,
                src: src.to_string(),
                x: "0".to_string(),
                y: "0".to_string(),
                scale: "1.0".to_string(),
                opacity: "1.0".to_string(),
            };
            self.draw_image_transformed(
                canvas,
                &image,
                character_transform,
                opacity,
                time_norm,
                time_sec,
            )?;
        }
        self.draw_character_nodes_vector(
            canvas,
            &character.children,
            character_transform,
            opacity,
            time_norm,
            time_sec,
        )
    }

    fn draw_character_nodes_vector(
        &mut self,
        canvas: &mut RgbaImage,
        nodes: &[SceneNode],
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        for node in nodes {
            match node {
                SceneNode::Defs(_) => {}
                SceneNode::Timeline(timeline) => {
                    let mut tracks = timeline
                        .children
                        .iter()
                        .filter_map(|node| match node {
                            SceneNode::Track(track) => Some(track),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    tracks.sort_by_key(|track| scene_track_composite_order(track));
                    let active_camera = active_scene_camera_from_tracks(
                        &tracks,
                        canvas.width(),
                        canvas.height(),
                        time_norm,
                        time_sec,
                    )?;
                    for track in tracks {
                        if is_scene_camera_track(track) {
                            continue;
                        }
                        let track_transform = if let Some(active_camera) = active_camera {
                            if is_scene_world_track(track) {
                                transform.mul(active_camera.transform)
                            } else {
                                transform
                            }
                        } else {
                            transform
                        };
                        self.draw_character_nodes_vector(
                            canvas,
                            &track.children,
                            track_transform,
                            inherited_opacity,
                            time_norm,
                            time_sec,
                        )?;
                    }
                }
                SceneNode::Track(track) => {
                    self.draw_character_nodes_vector(
                        canvas,
                        &track.children,
                        transform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                    )?;
                }
                SceneNode::Sequence(sequence) => {
                    if let Some((local_norm, local_sec)) =
                        scene_sequence_local_time(sequence, None, time_sec)
                    {
                        self.draw_character_nodes_vector(
                            canvas,
                            &sequence.children,
                            transform,
                            inherited_opacity,
                            local_norm,
                            local_sec,
                        )?;
                    }
                }
                SceneNode::Chain(chain) => {
                    let mut cursor_ms = chain.from_ms as i64;
                    for child in &chain.children {
                        if let SceneNode::Sequence(sequence) = child {
                            if let Some((local_norm, local_sec)) =
                                scene_sequence_local_time(sequence, Some(cursor_ms), time_sec)
                            {
                                self.draw_character_nodes_vector(
                                    canvas,
                                    &sequence.children,
                                    transform,
                                    inherited_opacity,
                                    local_norm,
                                    local_sec,
                                )?;
                            }
                            cursor_ms += sequence.duration_ms as i64 + chain.gap_ms;
                        }
                    }
                }
                SceneNode::Palette(_) => {}
                SceneNode::PixelGrid(grid) => {
                    self.draw_pixel_grid(
                        canvas,
                        grid,
                        transform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                    )?;
                }
                SceneNode::Group(group) => {
                    let opacity = (eval_scene_number(&group.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let group_transform =
                        transform.mul(scene_group_local_transform(group, time_norm, time_sec)?);
                    self.draw_character_nodes_vector(
                        canvas,
                        &group.children,
                        group_transform,
                        opacity,
                        time_norm,
                        time_sec,
                    )?;
                }
                SceneNode::Puppet(puppet) => {
                    let opacity = (eval_scene_number(&puppet.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let puppet_transform =
                        transform.mul(scene_puppet_local_transform(puppet, time_norm, time_sec)?);
                    let mut layer =
                        RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
                    self.draw_character_nodes_vector(
                        &mut layer,
                        &puppet.children,
                        puppet_transform,
                        opacity,
                        time_norm,
                        time_sec,
                    )?;
                    if let Some(deform_grid) = eval_puppet_deform_grid(puppet, time_norm, time_sec)?
                    {
                        layer = apply_deform_grid(
                            &layer,
                            &transform_deform_grid(&deform_grid, puppet_transform),
                        );
                    }
                    composite_layer(canvas, &layer);
                }
                SceneNode::Pin(_)
                | SceneNode::LimbEnvelope(_)
                | SceneNode::LimbRegion(_)
                | SceneNode::MeshTopology(_)
                | SceneNode::Vertex(_)
                | SceneNode::Triangle(_)
                | SceneNode::Edge(_)
                | SceneNode::Region(_) => {}
                SceneNode::Part(part) => {
                    let opacity = (eval_scene_number(&part.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let x = eval_scene_number(&part.x, time_norm, time_sec)?;
                    let y = eval_scene_number(&part.y, time_norm, time_sec)?;
                    let rotation = eval_scene_number(&part.rotation, time_norm, time_sec)?;
                    let scale =
                        eval_scene_number(&part.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
                    let anchor_x = eval_scene_number(&part.anchor_x, time_norm, time_sec)?;
                    let anchor_y = eval_scene_number(&part.anchor_y, time_norm, time_sec)?;
                    let part_transform = transform
                        .mul(Affine2::translate(x, y))
                        .mul(Affine2::rotate_deg(rotation))
                        .mul(Affine2::scale(scale))
                        .mul(Affine2::translate(-anchor_x, -anchor_y));
                    self.draw_character_nodes_vector(
                        canvas,
                        &part.children,
                        part_transform,
                        opacity,
                        time_norm,
                        time_sec,
                    )?;
                }
                SceneNode::Repeat(repeat) => {
                    let count = eval_repeat_count(&repeat.count, time_norm, time_sec)?;
                    if count == 0 {
                        continue;
                    }
                    let x = eval_scene_number(&repeat.x, time_norm, time_sec)?;
                    let y = eval_scene_number(&repeat.y, time_norm, time_sec)?;
                    let rotation = eval_scene_number(&repeat.rotation, time_norm, time_sec)?;
                    let scale =
                        eval_scene_number(&repeat.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
                    let opacity = eval_scene_number(&repeat.opacity, time_norm, time_sec)?;
                    let x_step = eval_scene_number(&repeat.x_step, time_norm, time_sec)?;
                    let y_step = eval_scene_number(&repeat.y_step, time_norm, time_sec)?;
                    let rotation_step =
                        eval_scene_number(&repeat.rotation_step, time_norm, time_sec)?;
                    let scale_step = eval_scene_number(&repeat.scale_step, time_norm, time_sec)?;
                    let opacity_step =
                        eval_scene_number(&repeat.opacity_step, time_norm, time_sec)?;
                    for index in 0..count {
                        let i = index as f32;
                        let copy_opacity =
                            ((opacity + opacity_step * i) * inherited_opacity).clamp(0.0, 1.0);
                        if copy_opacity <= 0.0001 {
                            continue;
                        }
                        let copy_scale = (scale + scale_step * i).clamp(0.001, 64.0);
                        let repeat_transform = transform
                            .mul(Affine2::translate(x + x_step * i, y + y_step * i))
                            .mul(Affine2::rotate_deg(rotation + rotation_step * i))
                            .mul(Affine2::scale(copy_scale));
                        self.draw_character_nodes_vector(
                            canvas,
                            &repeat.children,
                            repeat_transform,
                            copy_opacity,
                            time_norm,
                            time_sec,
                        )?;
                    }
                }
                SceneNode::Mask(mask) => {
                    let opacity = (eval_scene_number(&mask.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let mut layer =
                        RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
                    self.draw_character_nodes_vector(
                        &mut layer,
                        &mask.children,
                        transform,
                        opacity,
                        time_norm,
                        time_sec,
                    )?;
                    let mask_alpha = self.render_mask_alpha(
                        canvas.width(),
                        canvas.height(),
                        mask,
                        transform,
                        time_norm,
                        time_sec,
                    )?;
                    apply_alpha_mask(&mut layer, &mask_alpha);
                    composite_layer(canvas, &layer);
                }
                SceneNode::Character(character) => {
                    self.draw_character(
                        canvas,
                        character,
                        transform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                    )?;
                }
                SceneNode::Text(text) => {
                    self.draw_text_transformed(
                        canvas,
                        text,
                        transform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                    )?;
                }
                SceneNode::Line(line) => {
                    let node_transform =
                        transform.mul(scene_line_local_transform(line, time_norm, time_sec)?);
                    let opacity = (eval_scene_number(&line.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let style = eval_line_stroke_style(line, time_norm, time_sec)?;
                    let width = eval_scene_number(&line.width, time_norm, time_sec)?.max(0.0)
                        * affine_uniform_scale(node_transform);
                    if width <= 0.0001 {
                        continue;
                    }
                    let x1 = eval_scene_number(&line.x1, time_norm, time_sec)?;
                    let y1 = eval_scene_number(&line.y1, time_norm, time_sec)?;
                    let x2 = eval_scene_number(&line.x2, time_norm, time_sec)?;
                    let y2 = eval_scene_number(&line.y2, time_norm, time_sec)?;
                    let mut color = parse_color(&line.color)?;
                    color[3] = ((color[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
                    let (x1, y1) = node_transform.transform_point(x1, y1);
                    let (x2, y2) = node_transform.transform_point(x2, y2);
                    draw_line_segment_styled(
                        canvas,
                        Point2::new(x1, y1),
                        Point2::new(x2, y2),
                        width,
                        color,
                        style,
                        0.0,
                        1.0,
                    );
                }
                SceneNode::Polyline(polyline) => {
                    let node_transform = transform.mul(scene_polyline_local_transform(
                        polyline, time_norm, time_sec,
                    )?);
                    let opacity = (eval_scene_number(&polyline.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let style = eval_polyline_stroke_style(polyline, time_norm, time_sec)?;
                    let width = eval_scene_number(&polyline.stroke_width, time_norm, time_sec)?
                        .max(0.0)
                        * affine_uniform_scale(node_transform);
                    if width <= 0.0001 {
                        continue;
                    }
                    let points = self.cached_polyline_points(&polyline.points)?;
                    let trim = evaluate_trim(
                        &polyline.trim_start,
                        &polyline.trim_end,
                        time_norm,
                        time_sec,
                    )?;
                    if let Some(mut color) = parse_paint(&polyline.stroke)? {
                        color[3] = ((color[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
                        draw_transformed_trimmed_polylines_styled(
                            canvas,
                            &[points],
                            width,
                            color,
                            trim,
                            node_transform,
                            style,
                        );
                    }
                }
                SceneNode::Path(path) => {
                    let node_transform =
                        transform.mul(scene_path_local_transform(path, time_norm, time_sec)?);
                    let opacity = (eval_scene_number(&path.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let path_d = eval_path_d(&path.d, time_norm, time_sec)?;
                    let subpaths = self.cached_path_subpaths(path_d.as_ref())?;
                    if let Some(fill) = path.fill.as_deref() {
                        let paint = self.resolve_paint(fill)?;
                        let blend = parse_scene_blend(&path.blend)?;
                        draw_transformed_filled_polylines_paint_with_rule(
                            canvas,
                            &subpaths,
                            &paint,
                            opacity,
                            blend,
                            node_transform,
                            FillRule::parse(&path.fill_rule),
                        );
                    }
                    let width = eval_scene_number(&path.stroke_width, time_norm, time_sec)?
                        .max(0.0)
                        * affine_uniform_scale(node_transform);
                    if width <= 0.0001 {
                        continue;
                    }
                    let trim =
                        evaluate_trim(&path.trim_start, &path.trim_end, time_norm, time_sec)?;
                    let style = eval_path_stroke_style(path, time_norm, time_sec)?;
                    if let Some(mut color) = parse_paint(&path.stroke)? {
                        color[3] = ((color[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
                        draw_transformed_trimmed_polylines_styled(
                            canvas,
                            &subpaths,
                            width,
                            color,
                            trim,
                            node_transform,
                            style,
                        );
                    }
                }
                SceneNode::FaceJaw(face_jaw) => {
                    self.draw_face_jaw(
                        canvas,
                        face_jaw,
                        transform,
                        time_norm,
                        time_sec,
                        inherited_opacity,
                    )?;
                }
                SceneNode::Circle(circle) => {
                    let node_transform =
                        transform.mul(scene_circle_local_transform(circle, time_norm, time_sec)?);
                    let opacity = (eval_scene_number(&circle.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let x = eval_scene_number(&circle.x, time_norm, time_sec)?;
                    let y = eval_scene_number(&circle.y, time_norm, time_sec)?;
                    let radius = eval_scene_number(&circle.radius, time_norm, time_sec)?.max(0.0);
                    if radius <= 0.0001 {
                        continue;
                    }
                    let paint = self.resolve_paint(&circle.color)?;
                    let blend = parse_scene_blend(&circle.blend)?;
                    let stroke = circle.stroke.as_deref().map(parse_color).transpose()?;
                    let stroke_width =
                        eval_scene_number(&circle.stroke_width, time_norm, time_sec)?.max(0.0);
                    self.draw_circle_affine(
                        canvas,
                        node_transform,
                        x,
                        y,
                        radius,
                        &paint,
                        opacity,
                        blend,
                        stroke,
                        stroke_width,
                    );
                }
                SceneNode::Ellipse(ellipse) => {
                    let node_transform =
                        transform.mul(scene_ellipse_local_transform(ellipse, time_norm, time_sec)?);
                    let opacity = (eval_scene_number(&ellipse.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let x = eval_scene_number(&ellipse.x, time_norm, time_sec)?;
                    let y = eval_scene_number(&ellipse.y, time_norm, time_sec)?;
                    let radius_x =
                        eval_scene_number(&ellipse.radius_x, time_norm, time_sec)?.max(0.0);
                    let radius_y =
                        eval_scene_number(&ellipse.radius_y, time_norm, time_sec)?.max(0.0);
                    if radius_x <= 0.0001 || radius_y <= 0.0001 {
                        continue;
                    }
                    let paint = self.resolve_paint(&ellipse.color)?;
                    let blend = parse_scene_blend(&ellipse.blend)?;
                    let stroke = ellipse
                        .stroke
                        .as_deref()
                        .filter(|stroke| !is_none_paint(stroke))
                        .map(parse_color)
                        .transpose()?;
                    let stroke_width =
                        eval_scene_number(&ellipse.stroke_width, time_norm, time_sec)?.max(0.0);
                    if affine_is_identity(node_transform) {
                        draw_ellipse_paint(
                            canvas, x, y, radius_x, radius_y, &paint, opacity, blend,
                        );
                        if let Some(mut stroke) = stroke {
                            stroke[3] =
                                ((stroke[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
                            draw_ellipse_stroke(
                                canvas,
                                x,
                                y,
                                radius_x,
                                radius_y,
                                stroke_width,
                                stroke,
                            );
                        }
                    } else {
                        let mut layer = RgbaImage::from_pixel(
                            canvas.width(),
                            canvas.height(),
                            Rgba([0, 0, 0, 0]),
                        );
                        draw_ellipse_paint(
                            &mut layer, x, y, radius_x, radius_y, &paint, opacity, blend,
                        );
                        if let Some(mut stroke) = stroke {
                            stroke[3] =
                                ((stroke[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
                            draw_ellipse_stroke(
                                &mut layer,
                                x,
                                y,
                                radius_x,
                                radius_y,
                                stroke_width,
                                stroke,
                            );
                        }
                        composite_layer_affine(canvas, &layer, node_transform);
                    }
                }
                SceneNode::Rect(rect) => {
                    let node_transform =
                        transform.mul(scene_rect_local_transform(rect, time_norm, time_sec)?);
                    let opacity = (eval_scene_number(&rect.opacity, time_norm, time_sec)?
                        * inherited_opacity)
                        .clamp(0.0, 1.0);
                    if opacity <= 0.0001 {
                        continue;
                    }
                    let x = eval_scene_number(&rect.x, time_norm, time_sec)?;
                    let y = eval_scene_number(&rect.y, time_norm, time_sec)?;
                    let width = eval_scene_number(&rect.width, time_norm, time_sec)?.max(0.0);
                    let height = eval_scene_number(&rect.height, time_norm, time_sec)?.max(0.0);
                    if width <= 0.0001 || height <= 0.0001 {
                        continue;
                    }
                    let radius = eval_scene_number(&rect.radius, time_norm, time_sec)?.max(0.0);
                    let paint = self.resolve_paint(&rect.color)?;
                    let blend = parse_scene_blend(&rect.blend)?;
                    let stroke = rect.stroke.as_deref().map(parse_color).transpose()?;
                    let stroke_width =
                        eval_scene_number(&rect.stroke_width, time_norm, time_sec)?.max(0.0);
                    self.draw_rect_affine(
                        canvas,
                        node_transform,
                        x,
                        y,
                        width,
                        height,
                        radius,
                        &paint,
                        opacity,
                        blend,
                        stroke,
                        stroke_width,
                    );
                }
                SceneNode::Use(use_node) => {
                    self.draw_use_transformed(
                        canvas,
                        use_node,
                        transform,
                        time_norm,
                        time_sec,
                        inherited_opacity,
                    )?;
                }
                SceneNode::Precompose(_) | SceneNode::Layer(_) | SceneNode::Shadow(_) => {}
                SceneNode::Image(image) => {
                    self.draw_image_transformed(
                        canvas,
                        image,
                        transform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                    )?;
                }
                SceneNode::Svg(svg) => {
                    self.draw_svg_transformed(
                        canvas,
                        svg,
                        transform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                    )?;
                }
                SceneNode::Camera(_) => {}
                SceneNode::Simulation(_) => {}
            }
        }
        Ok(())
    }

    fn resolve_paint(&self, value: &str) -> Result<ResolvedPaint, MotionLoomSceneRenderError> {
        if is_none_paint(value) {
            return Ok(ResolvedPaint::None);
        }
        if let Some(id) = gradient_ref_id(value) {
            let Some(gradient) = self.gradient_defs.get(id) else {
                return Err(MotionLoomSceneRenderError::InvalidPaint {
                    value: value.to_string(),
                    message: format!("gradient reference not found: {id}"),
                });
            };
            return resolve_gradient_paint(value, gradient).map(ResolvedPaint::Gradient);
        }
        parse_color(value).map(ResolvedPaint::Solid)
    }

    fn draw_pixel_grid(
        &mut self,
        canvas: &mut RgbaImage,
        grid: &PixelGridNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&grid.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let pixel_size = eval_scene_number(&grid.pixel_size, time_norm, time_sec)?.max(0.0);
        if pixel_size <= 0.0001 {
            return Ok(());
        }
        let x = eval_scene_number(&grid.x, time_norm, time_sec)?;
        let y = eval_scene_number(&grid.y, time_norm, time_sec)?;
        let scale = affine_uniform_scale(transform).max(0.001);
        let draw_size = pixel_size * scale;
        let palette = self.palette_defs.get(&grid.palette).ok_or_else(|| {
            MotionLoomSceneRenderError::InvalidPaint {
                value: grid.palette.clone(),
                message: format!("PixelGrid palette not found: {}", grid.palette),
            }
        })?;
        let blend = parse_scene_blend(&grid.blend)?;

        for (row, line) in grid.data.lines().enumerate() {
            for (col, ch) in line.chars().enumerate() {
                if ch.is_whitespace() {
                    continue;
                }
                let key = ch.to_string();
                let Some(color_def) = palette.colors.iter().find(|color| color.key == key) else {
                    return Err(MotionLoomSceneRenderError::InvalidPaint {
                        value: key,
                        message: format!(
                            "PixelGrid{} references color key not found in palette '{}'",
                            id_suffix(grid.id.as_deref()),
                            grid.palette
                        ),
                    });
                };
                let color = parse_color(&color_def.value)?;
                if color[3] == 0 {
                    continue;
                }
                let local_x = x + col as f32 * pixel_size;
                let local_y = y + row as f32 * pixel_size;
                let (draw_x, draw_y) = transform.transform_point(local_x, local_y);
                draw_rounded_rect_paint(
                    canvas,
                    draw_x,
                    draw_y,
                    draw_size,
                    draw_size,
                    0.0,
                    &ResolvedPaint::Solid(color),
                    opacity,
                    blend,
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_rect_affine(
        &self,
        canvas: &mut RgbaImage,
        transform: Affine2,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        radius: f32,
        paint: &ResolvedPaint,
        opacity: f32,
        blend: SceneBlendMode,
        stroke: Option<[u8; 4]>,
        stroke_width: f32,
    ) {
        // Parent/group transforms can rotate or skew the full shape, not just its origin.
        let pad = (stroke_width * 0.5).ceil() + 2.0;
        let min_x = x - pad;
        let min_y = y - pad;
        let layer_w = (width + pad * 2.0).ceil().max(1.0) as u32;
        let layer_h = (height + pad * 2.0).ceil().max(1.0) as u32;
        let mut layer = RgbaImage::from_pixel(layer_w, layer_h, Rgba([0, 0, 0, 0]));
        let local_x = x - min_x;
        let local_y = y - min_y;
        draw_rounded_rect_paint(
            &mut layer,
            local_x,
            local_y,
            width,
            height,
            radius,
            paint,
            opacity,
            SceneBlendMode::Normal,
        );
        if let Some(mut stroke) = stroke {
            stroke[3] = ((stroke[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            draw_rounded_rect_stroke(
                &mut layer,
                local_x,
                local_y,
                width,
                height,
                radius,
                stroke_width,
                stroke,
            );
        }
        composite_layer_affine_blend(
            canvas,
            &layer,
            transform.mul(Affine2::translate(min_x, min_y)),
            1.0,
            blend,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_circle_affine(
        &self,
        canvas: &mut RgbaImage,
        transform: Affine2,
        x: f32,
        y: f32,
        radius: f32,
        paint: &ResolvedPaint,
        opacity: f32,
        blend: SceneBlendMode,
        stroke: Option<[u8; 4]>,
        stroke_width: f32,
    ) {
        // Parent/group transforms can rotate or skew the full shape, not just its center.
        let pad = (stroke_width * 0.5).ceil() + 2.0;
        let min_x = x - radius - pad;
        let min_y = y - radius - pad;
        let diameter = radius * 2.0;
        let layer_w = (diameter + pad * 2.0).ceil().max(1.0) as u32;
        let layer_h = (diameter + pad * 2.0).ceil().max(1.0) as u32;
        let mut layer = RgbaImage::from_pixel(layer_w, layer_h, Rgba([0, 0, 0, 0]));
        let local_x = x - min_x;
        let local_y = y - min_y;
        draw_circle_paint(
            &mut layer,
            local_x,
            local_y,
            radius,
            paint,
            opacity,
            SceneBlendMode::Normal,
        );
        if let Some(mut stroke) = stroke {
            stroke[3] = ((stroke[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            draw_circle_stroke(&mut layer, local_x, local_y, radius, stroke_width, stroke);
        }
        composite_layer_affine_blend(
            canvas,
            &layer,
            transform.mul(Affine2::translate(min_x, min_y)),
            1.0,
            blend,
        );
    }

    fn draw_rect(
        &mut self,
        canvas: &mut RgbaImage,
        rect: &RectNode,
        shadow: Option<EvaluatedShadow>,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&rect.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let x = eval_scene_number(&rect.x, time_norm, time_sec)?;
        let y = eval_scene_number(&rect.y, time_norm, time_sec)?;
        let width = eval_scene_number(&rect.width, time_norm, time_sec)?.max(0.0);
        let height = eval_scene_number(&rect.height, time_norm, time_sec)?.max(0.0);
        let radius = eval_scene_number(&rect.radius, time_norm, time_sec)?.max(0.0);
        let paint = self.resolve_paint(&rect.color)?;
        let blend = parse_scene_blend(&rect.blend)?;
        let stroke = rect.stroke.as_deref().map(parse_color).transpose()?;
        let stroke_width = eval_scene_number(&rect.stroke_width, time_norm, time_sec)?.max(0.0);
        let transform = scene_rect_local_transform(rect, time_norm, time_sec)?;

        if !affine_is_identity(transform) {
            let mut layer =
                RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
            if let Some(shadow) = shadow {
                draw_rect_shadow(&mut layer, x, y, width, height, radius, &shadow);
            }
            draw_rounded_rect_paint(
                &mut layer, x, y, width, height, radius, &paint, opacity, blend,
            );
            if let Some(stroke) = stroke {
                draw_rounded_rect_stroke(
                    &mut layer,
                    x,
                    y,
                    width,
                    height,
                    radius,
                    stroke_width,
                    stroke,
                );
            }
            composite_layer_affine(canvas, &layer, transform);
            return Ok(());
        }

        if let Some(shadow) = shadow {
            draw_rect_shadow(canvas, x, y, width, height, radius, &shadow);
        }
        draw_rounded_rect_paint(canvas, x, y, width, height, radius, &paint, opacity, blend);
        if let Some(stroke) = stroke {
            draw_rounded_rect_stroke(canvas, x, y, width, height, radius, stroke_width, stroke);
        }
        Ok(())
    }

    fn draw_circle(
        &mut self,
        canvas: &mut RgbaImage,
        circle: &CircleNode,
        shadow: Option<EvaluatedShadow>,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&circle.opacity, time_norm, time_sec)?
            * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let x = eval_scene_number(&circle.x, time_norm, time_sec)?;
        let y = eval_scene_number(&circle.y, time_norm, time_sec)?;
        let radius = eval_scene_number(&circle.radius, time_norm, time_sec)?.max(0.0);
        let paint = self.resolve_paint(&circle.color)?;
        let blend = parse_scene_blend(&circle.blend)?;
        let stroke = circle.stroke.as_deref().map(parse_color).transpose()?;
        let stroke_width = eval_scene_number(&circle.stroke_width, time_norm, time_sec)?.max(0.0);
        let transform = scene_circle_local_transform(circle, time_norm, time_sec)?;

        if !affine_is_identity(transform) {
            let mut layer =
                RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
            if let Some(shadow) = shadow {
                draw_circle_shadow(&mut layer, x, y, radius, &shadow);
            }
            draw_circle_paint(&mut layer, x, y, radius, &paint, opacity, blend);
            if let Some(stroke) = stroke {
                draw_circle_stroke(&mut layer, x, y, radius, stroke_width, stroke);
            }
            composite_layer_affine(canvas, &layer, transform);
            return Ok(());
        }

        if let Some(shadow) = shadow {
            draw_circle_shadow(canvas, x, y, radius, &shadow);
        }
        draw_circle_paint(canvas, x, y, radius, &paint, opacity, blend);
        if let Some(stroke) = stroke {
            draw_circle_stroke(canvas, x, y, radius, stroke_width, stroke);
        }
        Ok(())
    }

    fn draw_ellipse(
        &mut self,
        canvas: &mut RgbaImage,
        ellipse: &EllipseNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&ellipse.opacity, time_norm, time_sec)?
            * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let x = eval_scene_number(&ellipse.x, time_norm, time_sec)?;
        let y = eval_scene_number(&ellipse.y, time_norm, time_sec)?;
        let radius_x = eval_scene_number(&ellipse.radius_x, time_norm, time_sec)?.max(0.0);
        let radius_y = eval_scene_number(&ellipse.radius_y, time_norm, time_sec)?.max(0.0);
        if radius_x <= 0.0001 || radius_y <= 0.0001 {
            return Ok(());
        }
        let paint = self.resolve_paint(&ellipse.color)?;
        let blend = parse_scene_blend(&ellipse.blend)?;
        let stroke = ellipse
            .stroke
            .as_deref()
            .filter(|stroke| !is_none_paint(stroke))
            .map(parse_color)
            .transpose()?;
        let stroke_width = eval_scene_number(&ellipse.stroke_width, time_norm, time_sec)?.max(0.0);
        let transform = scene_ellipse_local_transform(ellipse, time_norm, time_sec)?;

        if !affine_is_identity(transform) {
            let mut layer =
                RgbaImage::from_pixel(canvas.width(), canvas.height(), Rgba([0, 0, 0, 0]));
            draw_ellipse_paint(&mut layer, x, y, radius_x, radius_y, &paint, opacity, blend);
            if let Some(mut stroke) = stroke {
                stroke[3] = ((stroke[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
                draw_ellipse_stroke(&mut layer, x, y, radius_x, radius_y, stroke_width, stroke);
            }
            composite_layer_affine(canvas, &layer, transform);
            return Ok(());
        }

        draw_ellipse_paint(canvas, x, y, radius_x, radius_y, &paint, opacity, blend);
        if let Some(mut stroke) = stroke {
            stroke[3] = ((stroke[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            draw_ellipse_stroke(canvas, x, y, radius_x, radius_y, stroke_width, stroke);
        }
        Ok(())
    }

    fn draw_line(
        &mut self,
        canvas: &mut RgbaImage,
        line: &LineNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&line.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let x1 = eval_scene_number(&line.x1, time_norm, time_sec)?;
        let y1 = eval_scene_number(&line.y1, time_norm, time_sec)?;
        let x2 = eval_scene_number(&line.x2, time_norm, time_sec)?;
        let y2 = eval_scene_number(&line.y2, time_norm, time_sec)?;
        let transform = scene_line_local_transform(line, time_norm, time_sec)?;
        let width = eval_scene_number(&line.width, time_norm, time_sec)?.max(0.0)
            * affine_uniform_scale(transform);
        if width <= 0.0001 {
            return Ok(());
        }
        let style = eval_line_stroke_style(line, time_norm, time_sec)?;
        if let Some(mut color) = parse_paint(&line.color)? {
            color[3] = ((color[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            let (x1, y1) = transform.transform_point(x1, y1);
            let (x2, y2) = transform.transform_point(x2, y2);
            draw_line_segment_styled(
                canvas,
                Point2::new(x1, y1),
                Point2::new(x2, y2),
                width,
                color,
                style,
                0.0,
                1.0,
            );
        }
        Ok(())
    }

    fn draw_polyline(
        &mut self,
        canvas: &mut RgbaImage,
        polyline: &PolylineNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&polyline.opacity, time_norm, time_sec)?
            * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let transform = scene_polyline_local_transform(polyline, time_norm, time_sec)?;
        let width = eval_scene_number(&polyline.stroke_width, time_norm, time_sec)?.max(0.0)
            * affine_uniform_scale(transform);
        if width <= 0.0001 {
            return Ok(());
        }
        let points = parse_polyline_points(&polyline.points)?;
        let trim = evaluate_trim(
            &polyline.trim_start,
            &polyline.trim_end,
            time_norm,
            time_sec,
        )?;
        let style = eval_polyline_stroke_style(polyline, time_norm, time_sec)?;
        if let Some(mut color) = parse_paint(&polyline.stroke)? {
            color[3] = ((color[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            draw_transformed_trimmed_polylines_styled(
                canvas,
                &[points],
                width,
                color,
                trim,
                transform,
                style,
            );
        }
        Ok(())
    }

    fn draw_path(
        &mut self,
        canvas: &mut RgbaImage,
        path: &PathNode,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&path.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }
        let transform = scene_path_local_transform(path, time_norm, time_sec)?;
        let subpaths = evaluated_path_subpaths(path, time_norm, time_sec)?;
        if let Some(fill) = path.fill.as_deref() {
            let paint = self.resolve_paint(fill)?;
            let blend = parse_scene_blend(&path.blend)?;
            draw_transformed_filled_polylines_paint_with_rule(
                canvas,
                &subpaths,
                &paint,
                opacity,
                blend,
                transform,
                evaluated_path_fill_rule(path),
            );
        }
        let width = eval_scene_number(&path.stroke_width, time_norm, time_sec)?.max(0.0)
            * affine_uniform_scale(transform);
        if width <= 0.0001 {
            return Ok(());
        }
        let trim = evaluate_trim(&path.trim_start, &path.trim_end, time_norm, time_sec)?;
        let style = eval_path_stroke_style(path, time_norm, time_sec)?;
        if let Some(mut color) = parse_paint(&path.stroke)? {
            color[3] = ((color[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            draw_transformed_trimmed_polylines_styled(
                canvas, &subpaths, width, color, trim, transform, style,
            );
        }
        Ok(())
    }

    fn draw_face_jaw(
        &mut self,
        canvas: &mut RgbaImage,
        face_jaw: &FaceJawNode,
        transform: Affine2,
        time_norm: f32,
        time_sec: f32,
        inherited_opacity: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let path = face_jaw_to_path_node(face_jaw, time_norm, time_sec)?;
        let opacity = (eval_scene_number(&path.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }

        let subpaths = parse_path_subpaths(&path.d)?;
        if let Some(fill) = path.fill.as_deref() {
            let paint = self.resolve_paint(fill)?;
            let blend = parse_scene_blend(&path.blend)?;
            draw_transformed_filled_polylines_paint_with_rule(
                canvas,
                &subpaths,
                &paint,
                opacity,
                blend,
                transform,
                FillRule::parse(&path.fill_rule),
            );
        }

        let width = eval_scene_number(&path.stroke_width, time_norm, time_sec)?.max(0.0)
            * affine_uniform_scale(transform);
        if width <= 0.0001 {
            return Ok(());
        }
        let trim = evaluate_trim(&path.trim_start, &path.trim_end, time_norm, time_sec)?;
        let style = eval_path_stroke_style(&path, time_norm, time_sec)?;
        if let Some(mut color) = parse_paint(&path.stroke)? {
            color[3] = ((color[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            draw_transformed_trimmed_polylines_styled(
                canvas, &subpaths, width, color, trim, transform, style,
            );
        }
        Ok(())
    }

    fn draw_text_transformed(
        &mut self,
        canvas: &mut RgbaImage,
        text: &TextNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        if let Some(layer) = self.rasterize_text_texture_layer(
            text,
            transform,
            inherited_opacity,
            time_norm,
            time_sec,
            (canvas.width(), canvas.height()),
        )? {
            if let GpuSceneTextureSource::Cpu(image) = &layer.source {
                composite_layer_affine(canvas, image, layer.transform);
            }
        }
        Ok(())
    }

    fn draw_image_transformed(
        &mut self,
        canvas: &mut RgbaImage,
        image: &ImageNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&image.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }

        let scale = eval_scene_number(&image.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
        let source = self.load_image_asset(&image.src)?;
        let target_w = ((source.width() as f32) * scale).round().max(1.0) as u32;
        let target_h = ((source.height() as f32) * scale).round().max(1.0) as u32;
        let x_base = resolve_axis(
            &image.x,
            canvas.width() as f32,
            target_w as f32,
            time_norm,
            time_sec,
        )?;
        let y_base = resolve_axis(
            &image.y,
            canvas.height() as f32,
            target_h as f32,
            time_norm,
            time_sec,
        )?;
        let local_transform = transform.mul(Affine2::translate(x_base, y_base));

        if target_w == source.width() && target_h == source.height() {
            composite_layer_affine_blend(
                canvas,
                source,
                local_transform,
                opacity,
                SceneBlendMode::Normal,
            );
        } else {
            let scaled = image::imageops::resize(source, target_w, target_h, FilterType::Lanczos3);
            composite_layer_affine_blend(
                canvas,
                &scaled,
                local_transform,
                opacity,
                SceneBlendMode::Normal,
            );
        }
        Ok(())
    }

    fn draw_svg_transformed(
        &mut self,
        canvas: &mut RgbaImage,
        svg: &SvgNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<(), MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&svg.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(());
        }

        let scale = eval_scene_number(&svg.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
        let source = self.load_svg_asset(&svg.src)?;
        let target_w = ((source.width() as f32) * scale).round().max(1.0) as u32;
        let target_h = ((source.height() as f32) * scale).round().max(1.0) as u32;
        let x_base = resolve_axis(
            &svg.x,
            canvas.width() as f32,
            target_w as f32,
            time_norm,
            time_sec,
        )?;
        let y_base = resolve_axis(
            &svg.y,
            canvas.height() as f32,
            target_h as f32,
            time_norm,
            time_sec,
        )?;
        let local_transform = transform.mul(Affine2::translate(x_base, y_base));

        if target_w == source.width() && target_h == source.height() {
            composite_layer_affine_blend(
                canvas,
                source,
                local_transform,
                opacity,
                SceneBlendMode::Normal,
            );
        } else {
            let scaled = image::imageops::resize(source, target_w, target_h, FilterType::Lanczos3);
            composite_layer_affine_blend(
                canvas,
                &scaled,
                local_transform,
                opacity,
                SceneBlendMode::Normal,
            );
        }
        Ok(())
    }

    fn gpu_image_texture_layer(
        &mut self,
        image: &ImageNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
    ) -> Result<Option<GpuSceneTextureLayer>, MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&image.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(None);
        }
        let scale = eval_scene_number(&image.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
        let (width, height, texture) = self
            .gpu_compositor
            .as_mut()
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: "GPU compositor was not initialized".to_string(),
            })?
            .load_image_texture(&image.src)?;

        raster_texture_layer(
            texture,
            width,
            height,
            &image.x,
            &image.y,
            scale,
            opacity,
            transform,
            time_norm,
            time_sec,
            canvas_size,
        )
    }

    fn gpu_svg_texture_layer(
        &mut self,
        svg: &SvgNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
    ) -> Result<Option<GpuSceneTextureLayer>, MotionLoomSceneRenderError> {
        let opacity = (eval_scene_number(&svg.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(None);
        }
        let scale = eval_scene_number(&svg.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
        let (width, height, texture) = self
            .gpu_compositor
            .as_mut()
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: "GPU compositor was not initialized".to_string(),
            })?
            .load_svg_texture(&svg.src)?;

        raster_texture_layer(
            texture,
            width,
            height,
            &svg.x,
            &svg.y,
            scale,
            opacity,
            transform,
            time_norm,
            time_sec,
            canvas_size,
        )
    }

    fn gpu_vector_overlay_texture_layer(
        &mut self,
        nodes: &[SceneNode],
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
    ) -> Result<Option<GpuSceneTextureLayer>, MotionLoomSceneRenderError> {
        if canvas_size.0 == 0 || canvas_size.1 == 0 || inherited_opacity <= 0.0001 {
            return Ok(None);
        }
        let raster_transform = vector_overlay_raster_transform(transform);
        let present_transform = vector_overlay_present_transform(transform);
        let cache_key = if vector_overlay_nodes_are_static(nodes) {
            Some(vector_overlay_cache_key(
                nodes,
                raster_transform,
                inherited_opacity,
                canvas_size,
            ))
        } else {
            None
        };

        if let Some(key) = cache_key.as_deref() {
            if let Some(cached) = self.vector_overlay_cache.get(key).cloned() {
                if cached.gpu_texture.is_none() {
                    if let Some(compositor) = self.gpu_compositor.as_mut() {
                        let gpu_texture = compositor.upload_gpu_rgba_texture(&cached.image)?;
                        if let Some(entry) = self.vector_overlay_cache.get_mut(key) {
                            entry.gpu_texture = Some(gpu_texture);
                        }
                    }
                }
                return Ok(self
                    .vector_overlay_cache
                    .get(key)
                    .map(|entry| entry.texture_layer(present_transform))
                    .or_else(|| Some(cached.texture_layer(present_transform))));
            }
        }

        let mut canvas = RgbaImage::from_pixel(canvas_size.0, canvas_size.1, Rgba([0, 0, 0, 0]));
        self.draw_character_nodes_vector(
            &mut canvas,
            nodes,
            raster_transform,
            inherited_opacity,
            time_norm,
            time_sec,
        )?;
        let Some((cropped, offset_x, offset_y)) = crop_layer_alpha_bounds(&canvas) else {
            return Ok(None);
        };
        let mut cached = CachedVectorOverlay {
            image: cropped,
            offset_x: offset_x as f32,
            offset_y: offset_y as f32,
            gpu_texture: None,
        };
        if let Some(compositor) = self.gpu_compositor.as_mut() {
            cached.gpu_texture = Some(compositor.upload_gpu_rgba_texture(&cached.image)?);
        }
        let layer = cached.texture_layer(present_transform);
        if let Some(key) = cache_key {
            cache_vector_overlay(&mut self.vector_overlay_cache, key, cached);
        }
        Ok(Some(layer))
    }

    fn gpu_path_overlay_texture_layer(
        &mut self,
        path: &PathNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
    ) -> Result<Option<GpuSceneTextureLayer>, MotionLoomSceneRenderError> {
        if inherited_opacity <= 0.0001 || !path_has_default_local_transform(path) {
            return Ok(None);
        }
        let subpaths = self.cached_path_subpaths(&eval_path_d(&path.d, time_norm, time_sec)?)?;
        let Some(bounds) = subpaths_bounds(&subpaths) else {
            return Ok(None);
        };
        let stroke_pad = eval_scene_number(&path.stroke_width, time_norm, time_sec)
            .unwrap_or(0.0)
            .abs()
            .ceil()
            .max(2.0);
        let min_x = bounds.min_x - stroke_pad;
        let min_y = bounds.min_y - stroke_pad;
        let max_x = bounds.max_x + stroke_pad;
        let max_y = bounds.max_y + stroke_pad;
        let width = (max_x - min_x).ceil().max(1.0) as u32;
        let height = (max_y - min_y).ceil().max(1.0) as u32;
        if width > 8192 || height > 8192 {
            return Ok(None);
        }

        let node = SceneNode::Path(path.clone());
        let nodes = [node];
        let cache_key = if vector_overlay_nodes_are_static(&nodes) {
            let mut hasher = DefaultHasher::new();
            width.hash(&mut hasher);
            height.hash(&mut hasher);
            inherited_opacity.to_bits().hash(&mut hasher);
            format!("{:?}", nodes).hash(&mut hasher);
            Some(format!("path-overlay-local:{:016x}", hasher.finish()))
        } else {
            None
        };

        if let Some(key) = cache_key.as_deref() {
            if let Some(cached) = self.vector_overlay_cache.get(key).cloned() {
                return Ok(Some(cached.texture_layer(transform)));
            }
        }

        let mut canvas = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));
        let local_transform = Affine2::translate(-min_x, -min_y);
        self.draw_character_nodes_vector(
            &mut canvas,
            &nodes,
            local_transform,
            inherited_opacity,
            time_norm,
            time_sec,
        )?;
        let Some((cropped, offset_x, offset_y)) = crop_layer_alpha_bounds(&canvas) else {
            return Ok(None);
        };
        let mut cached = CachedVectorOverlay {
            image: cropped,
            offset_x: min_x + offset_x as f32,
            offset_y: min_y + offset_y as f32,
            gpu_texture: None,
        };
        if let Some(compositor) = self.gpu_compositor.as_mut() {
            cached.gpu_texture = Some(compositor.upload_gpu_rgba_texture(&cached.image)?);
        }
        let layer = cached.texture_layer(transform);
        if let Some(key) = cache_key {
            cache_vector_overlay(&mut self.vector_overlay_cache, key, cached);
        }
        Ok(Some(layer))
    }

    fn rasterize_text_base_layer(
        &mut self,
        text: &TextNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
    ) -> Result<Option<TextRasterizedLayer>, MotionLoomSceneRenderError> {
        if text.value.trim().is_empty() {
            return Ok(None);
        }
        let font_def = text
            .font
            .as_deref()
            .and_then(|font_id| self.font_defs.get(font_id));
        let font_path = text
            .font_path
            .clone()
            .or_else(|| font_def.and_then(|font| font.path.clone()));
        if let Some(path) = font_path.as_deref() {
            let resolved = resolve_local_scene_asset_path(path);
            if resolved.exists() {
                let _ = self.font_system.db_mut().load_font_file(&resolved);
            }
        }

        let font_size = eval_scene_number(&text.font_size, time_norm, time_sec)?.clamp(1.0, 1024.0);
        let render_scale =
            eval_text_render_scale(&text.render_scale, font_size, time_norm, time_sec)?;
        let raster_font_size = font_size * render_scale;
        let opacity = (eval_scene_number(&text.opacity, time_norm, time_sec)? * inherited_opacity)
            .clamp(0.0, 1.0);
        if opacity <= 0.0001 {
            return Ok(None);
        }

        let line_height_raw = text
            .line_height
            .as_deref()
            .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
            .unwrap_or(1.2);
        let line_height = if line_height_raw <= 10.0 {
            (font_size * line_height_raw).max(1.0)
        } else {
            line_height_raw.max(1.0)
        };
        let raster_line_height = line_height * render_scale;
        let width = text
            .layout_width_expr()
            .map(|expr| eval_scene_number(expr, time_norm, time_sec).map(|value| value.max(1.0)))
            .transpose()?;
        let raster_width = width.map(|value| value * render_scale);
        let visible_value = if let Some(expr) = text.visible_chars.as_deref() {
            let count = eval_scene_number(expr, time_norm, time_sec)
                .unwrap_or(text.value.chars().count() as f32)
                .floor()
                .clamp(0.0, text.value.chars().count() as f32) as usize;
            text.value.chars().take(count).collect::<String>()
        } else {
            text.value.clone()
        };
        let prepared_text = if text.animators.is_empty() {
            None
        } else {
            Some(
                prepare_text_layout_for_value(text, &visible_value).map_err(|message| {
                    MotionLoomSceneRenderError::InvalidExpression {
                        expr: "TextAnimator".to_string(),
                        message,
                    }
                })?,
            )
        };
        let metrics = Metrics::new(raster_font_size, raster_line_height);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        let mut attrs = Attrs::new()
            .family(Family::SansSerif)
            .weight(eval_text_font_weight(
                text.font_weight.as_deref(),
                time_norm,
                time_sec,
            )?)
            .letter_spacing(eval_text_tracking_em(
                text.tracking.as_deref(),
                font_size,
                time_norm,
                time_sec,
            )?);
        let font_family = text
            .font_family
            .as_deref()
            .or_else(|| font_def.and_then(|font| font.family.as_deref()));
        #[cfg(target_arch = "wasm32")]
        let font_family = font_def
            .and_then(|font| font.fallback.as_deref())
            .or(font_family);
        #[cfg(target_arch = "wasm32")]
        let wasm_loaded_fallback_family = Some(motionloom_wasm_loaded_font_family(
            font_family,
            &visible_value,
        ));
        #[cfg(not(target_arch = "wasm32"))]
        let wasm_loaded_fallback_family = None;
        if let Some(family) =
            wasm_loaded_fallback_family.or_else(|| font_family.and_then(motionloom_font_family))
        {
            attrs = attrs.family(family);
        }
        // Browser builds use Basic shaping to avoid native font shaping paths.
        #[cfg(target_arch = "wasm32")]
        let shaping = Shaping::Basic;
        #[cfg(not(target_arch = "wasm32"))]
        let shaping = Shaping::Advanced;
        buffer.set_text(&mut self.font_system, &visible_value, &attrs, shaping);
        buffer.set_size(&mut self.font_system, raster_width, None);
        buffer.shape_until_scroll(&mut self.font_system, true);

        let (raster_text_w, raster_text_h) = text_bounds(&buffer, raster_line_height);
        let text_w = raster_text_w / render_scale;
        let text_h = raster_text_h / render_scale;
        let layout_w = width.unwrap_or(text_w);
        let box_style = eval_text_box_style(text, layout_w, text_h, time_norm, time_sec)?;
        let box_pad_x = box_style.map(|style| style.padding_x).unwrap_or(0.0);
        let box_pad_y = box_style.map(|style| style.padding_y).unwrap_or(0.0);
        let box_w = (layout_w + box_pad_x * 2.0).max(1.0);
        let box_h = (text_h + box_pad_y * 2.0).max(1.0);
        let x_base = resolve_axis(&text.x, canvas_size.0 as f32, layout_w, time_norm, time_sec)?;
        let y_base = resolve_axis(&text.y, canvas_size.1 as f32, text_h, time_norm, time_sec)?;
        let layer_effects =
            text_layer_effect_spec(text, time_norm, time_sec)?.scaled_for_raster(render_scale);
        let pad = (3.0_f32 * render_scale).ceil() as i32 + layer_effects.pad_px;
        let text_offset_x = pad + (box_pad_x * render_scale).round() as i32;
        let text_offset_y = pad + (box_pad_y * render_scale).round() as i32;
        let layer_w =
            ((box_w * render_scale).ceil().max(1.0) as u32).saturating_add((pad * 2) as u32);
        let layer_h =
            ((box_h * render_scale).ceil().max(1.0) as u32).saturating_add((pad * 2) as u32);
        let mut layer = RgbaImage::from_pixel(layer_w, layer_h, Rgba([0, 0, 0, 0]));
        if let Some(box_style) = box_style {
            let mut box_color = box_style.color;
            box_color[3] = ((box_color[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            draw_rounded_rect(
                &mut layer,
                pad as f32,
                pad as f32,
                box_w * render_scale,
                box_h * render_scale,
                box_style.radius * render_scale,
                box_color,
            );
        }
        let color = parse_color(&text.color)?;
        let combined_opacity = (color[3] as f32 / 255.0) * opacity;
        let text_color = Color::rgba(color[0], color[1], color[2], 255);
        let max_lines = text
            .max_lines_expr()
            .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
            .map(|value| value.floor().max(0.0) as usize);

        if let Some(prepared_text) = prepared_text.as_ref() {
            draw_text_buffer_with_animators(
                &buffer,
                &mut self.font_system,
                &mut self.swash_cache,
                &mut layer,
                TextAnimatorRasterParams {
                    text,
                    prepared: prepared_text,
                    value: &visible_value,
                    base_color: color,
                    base_opacity: opacity,
                    offset_x: text_offset_x,
                    offset_y: text_offset_y,
                    raster_scale: render_scale,
                    max_lines,
                    global_time_ms: (time_sec.max(0.0) * 1000.0).round() as i64,
                    time_norm,
                    time_sec,
                },
            )?;
        } else {
            buffer.draw(
                &mut self.font_system,
                &mut self.swash_cache,
                text_color,
                |x, y, _w, _h, color| {
                    if let Some(max_lines) = max_lines {
                        let line_ix = ((y as f32) / raster_line_height).floor().max(0.0) as usize;
                        if line_ix >= max_lines {
                            return;
                        }
                    }
                    let px = x + text_offset_x;
                    let py = y + text_offset_y;
                    if px < 0 || py < 0 {
                        return;
                    }
                    let (px, py) = (px as u32, py as u32);
                    if px >= layer.width() || py >= layer.height() {
                        return;
                    }
                    let (sr, sg, sb, sa) = color.as_rgba_tuple();
                    let sa = ((sa as f32) * combined_opacity).round().clamp(0.0, 255.0) as u8;
                    blend_pixel(&mut layer, px, py, [sr, sg, sb, sa]);
                },
            );
        }
        let edge_smoothing = eval_text_edge_smoothing(text, time_norm, time_sec)?.clamp(0.0, 1.0);
        if edge_smoothing > 0.001 {
            layer = soften_text_alpha_edges(&layer, edge_smoothing);
        }

        let text_transform = transform
            .mul(Affine2::translate(
                x_base - box_pad_x - pad as f32 / render_scale,
                y_base - box_pad_y - pad as f32 / render_scale,
            ))
            .mul(scene_text_local_transform(text, time_norm, time_sec)?)
            .mul(Affine2::scale(1.0 / render_scale));
        Ok(Some(TextRasterizedLayer {
            image: layer,
            transform: text_transform,
            effects: layer_effects,
        }))
    }

    fn rasterize_text_texture_layer(
        &mut self,
        text: &TextNode,
        transform: Affine2,
        inherited_opacity: f32,
        time_norm: f32,
        time_sec: f32,
        canvas_size: (u32, u32),
    ) -> Result<Option<GpuSceneTextureLayer>, MotionLoomSceneRenderError> {
        let Some(base) = self.rasterize_text_base_layer(
            text,
            transform,
            inherited_opacity,
            time_norm,
            time_sec,
            canvas_size,
        )?
        else {
            return Ok(None);
        };
        let image = if base.effects.has_effects() {
            apply_text_layer_effects(&base.image, &base.effects)
        } else {
            base.image
        };
        Ok(Some(GpuSceneTextureLayer {
            source: GpuSceneTextureSource::Cpu(image),
            transform: base.transform,
            projected_quad: None,
            opacity: 1.0,
            blend: SceneBlendMode::Normal,
            pick_id: 0,
            matte: None,
            primitive_index: usize::MAX,
        }))
    }

    fn rasterize_text_texture_layers_gpu_effects(
        &mut self,
        text: &TextNode,
        context: TextTextureRasterContext,
    ) -> Result<Vec<GpuSceneTextureLayer>, MotionLoomSceneRenderError> {
        if context.inherited_opacity <= 0.0001 {
            return Ok(Vec::new());
        }

        // Static text content is retained independently from its parent Group
        // transform. This is especially important for renderScale="4x": moving
        // a title should update one texture-layer matrix, not rerun shaping,
        // supersampled rasterization, and RGBA upload every frame.
        if text_raster_content_is_static(text)
            && !text_layer_effect_spec(text, context.time_norm, context.time_sec)?.has_effects()
        {
            let text_opacity =
                eval_scene_number(&text.opacity, context.time_norm, context.time_sec)?
                    .clamp(0.0, 1.0);
            let layer_opacity = (context.inherited_opacity * text_opacity).clamp(0.0, 1.0);
            if layer_opacity <= 0.0001 {
                return Ok(Vec::new());
            }
            let cache_key =
                gpu_text_raster_cache_key(self.prepared_graph_signature, text, context.canvas_size);
            if let Some(cached) = self.gpu_text_raster_cache.get(&cache_key).cloned() {
                return Ok(vec![GpuSceneTextureLayer {
                    source: GpuSceneTextureSource::Gpu(cached.texture),
                    transform: context.transform.mul(cached.local_transform),
                    projected_quad: None,
                    opacity: layer_opacity,
                    blend: SceneBlendMode::Normal,
                    pick_id: context.pick_id,
                    matte: None,
                    primitive_index: usize::MAX,
                }]);
            }

            let mut raster_text = text.clone();
            raster_text.opacity = "1".to_string();
            let Some(base) = self.rasterize_text_base_layer(
                &raster_text,
                context.transform,
                1.0,
                context.time_norm,
                context.time_sec,
                context.canvas_size,
            )?
            else {
                return Ok(Vec::new());
            };
            let local_transform = context
                .transform
                .inverse()
                .map(|inverse| inverse.mul(base.transform))
                .unwrap_or(base.transform);
            let texture = self
                .gpu_compositor
                .as_mut()
                .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                })?
                .upload_gpu_rgba_texture(&base.image)?;
            if self.gpu_text_raster_cache.len() >= GPU_TEXT_RASTER_CACHE_LIMIT {
                self.gpu_text_raster_cache.clear();
            }
            self.gpu_text_raster_cache.insert(
                cache_key,
                CachedGpuTextRaster {
                    texture: texture.clone(),
                    local_transform,
                },
            );
            return Ok(vec![GpuSceneTextureLayer {
                source: GpuSceneTextureSource::Gpu(texture),
                transform: context.transform.mul(local_transform),
                projected_quad: None,
                opacity: layer_opacity,
                blend: SceneBlendMode::Normal,
                pick_id: context.pick_id,
                matte: None,
                primitive_index: usize::MAX,
            }]);
        }

        let Some(base) = self.rasterize_text_base_layer(
            text,
            context.transform,
            context.inherited_opacity,
            context.time_norm,
            context.time_sec,
            context.canvas_size,
        )?
        else {
            return Ok(Vec::new());
        };

        if !base.effects.has_effects() {
            return Ok(vec![GpuSceneTextureLayer {
                source: GpuSceneTextureSource::Cpu(base.image),
                transform: base.transform,
                projected_quad: None,
                opacity: 1.0,
                blend: SceneBlendMode::Normal,
                pick_id: context.pick_id,
                matte: None,
                primitive_index: usize::MAX,
            }]);
        }

        let mut layers = Vec::new();
        let source = self
            .gpu_compositor
            .as_mut()
            .ok_or_else(|| MotionLoomSceneRenderError::GpuRender {
                message: "GPU compositor was not initialized".to_string(),
            })?
            .upload_gpu_rgba_texture(&base.image)?;

        if !base.effects.shadows.is_empty() || !base.effects.glows.is_empty() {
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            for shadow in &base.effects.shadows {
                let mut shadow_texture =
                    compositor.apply_gpu_tint_texture(&source, shadow.color, 1.0);
                if shadow.blur > 0.001 {
                    shadow_texture = compositor.apply_gpu_blur_texture(
                        &shadow_texture,
                        &[(true, shadow.blur), (false, shadow.blur)],
                    )?;
                }
                layers.push(GpuSceneTextureLayer {
                    source: GpuSceneTextureSource::Gpu(shadow_texture),
                    transform: base.transform.mul(Affine2::translate(shadow.x, shadow.y)),
                    projected_quad: None,
                    opacity: 1.0,
                    blend: SceneBlendMode::Normal,
                    pick_id: 0,
                    matte: None,
                    primitive_index: usize::MAX,
                });
            }
            for glow in &base.effects.glows {
                let glow_texture =
                    compositor.apply_gpu_tint_texture(&source, glow.color, glow.intensity);
                let glow_texture = compositor.apply_gpu_blur_texture(
                    &glow_texture,
                    &[(true, glow.radius), (false, glow.radius)],
                )?;
                layers.push(GpuSceneTextureLayer {
                    source: GpuSceneTextureSource::Gpu(glow_texture),
                    transform: base.transform,
                    projected_quad: None,
                    opacity: 1.0,
                    blend: SceneBlendMode::Screen,
                    pick_id: 0,
                    matte: None,
                    primitive_index: usize::MAX,
                });
            }
        }

        if let Some(stroke) = base.effects.stroke {
            let stroke_layer = stroke_layer_from_alpha(&base.image, stroke.width, stroke.color);
            layers.push(GpuSceneTextureLayer {
                source: GpuSceneTextureSource::Cpu(stroke_layer),
                transform: base.transform,
                projected_quad: None,
                opacity: 1.0,
                blend: SceneBlendMode::Normal,
                pick_id: context.pick_id,
                matte: None,
                primitive_index: usize::MAX,
            });
        }

        let fill_source = if base.effects.blur_radius > 0.001 {
            let compositor = self.gpu_compositor.as_mut().ok_or_else(|| {
                MotionLoomSceneRenderError::GpuRender {
                    message: "GPU compositor was not initialized".to_string(),
                }
            })?;
            GpuSceneTextureSource::Gpu(compositor.apply_gpu_blur_texture(
                &source,
                &[
                    (true, base.effects.blur_radius),
                    (false, base.effects.blur_radius),
                ],
            )?)
        } else {
            GpuSceneTextureSource::Gpu(source)
        };
        layers.push(GpuSceneTextureLayer {
            source: fill_source,
            transform: base.transform,
            projected_quad: None,
            opacity: 1.0,
            blend: SceneBlendMode::Normal,
            pick_id: context.pick_id,
            matte: None,
            primitive_index: usize::MAX,
        });

        Ok(layers)
    }

    fn load_image_asset(&mut self, src: &str) -> Result<&RgbaImage, MotionLoomSceneRenderError> {
        if !self.image_cache.contains_key(src) {
            let decoded = load_rgba_image_source(src, self.asset_resolver.as_ref())?;
            self.image_cache.insert(src.to_string(), decoded);
        }

        Ok(self
            .image_cache
            .get(src)
            .expect("image cache entry inserted before lookup"))
    }

    fn load_svg_asset(&mut self, src: &str) -> Result<&RgbaImage, MotionLoomSceneRenderError> {
        if !self.svg_cache.contains_key(src) {
            let decoded = load_svg_source(src, self.asset_resolver.as_ref())?;
            self.svg_cache.insert(src.to_string(), decoded);
        }

        Ok(self
            .svg_cache
            .get(src)
            .expect("SVG cache entry inserted before lookup"))
    }
}

fn eval_scene_vec3(
    expr: &str,
    time_norm: f32,
    time_sec: f32,
    default: [f32; 3],
) -> Result<[f32; 3], MotionLoomSceneRenderError> {
    let trimmed = expr
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    if trimmed.is_empty() {
        return Ok(default);
    }

    let mut values = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    let mut quote = None;
    for (index, ch) in trimmed.char_indices() {
        if let Some(active_quote) = quote {
            if ch == active_quote {
                quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => quote = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                values.push(trimmed[start..index].trim());
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    values.push(trimmed[start..].trim());
    if values.len() != 3 {
        return Err(MotionLoomSceneRenderError::InvalidExpression {
            expr: expr.to_string(),
            message: "expected a three-component vector such as [0,0,6]".to_string(),
        });
    }
    Ok([
        eval_scene_number(values[0], time_norm, time_sec)?,
        eval_scene_number(values[1], time_norm, time_sec)?,
        eval_scene_number(values[2], time_norm, time_sec)?,
    ])
}

pub(crate) fn eval_scene_number(
    expr: &str,
    time_norm: f32,
    time_sec: f32,
) -> Result<f32, MotionLoomSceneRenderError> {
    eval_time_expr(expr, time_norm, time_sec).map_err(|message| {
        MotionLoomSceneRenderError::InvalidExpression {
            expr: expr.to_string(),
            message,
        }
    })
}

fn eval_text_render_scale(
    expr: &str,
    font_size: f32,
    time_norm: f32,
    time_sec: f32,
) -> Result<f32, MotionLoomSceneRenderError> {
    let trimmed = expr.trim();
    let normalized = trimmed.to_ascii_lowercase();
    let scale = if matches!(normalized.as_str(), "" | "auto") {
        if font_size < 24.0 {
            4.0
        } else if font_size < 64.0 {
            3.0
        } else {
            2.0
        }
    } else if let Some(without_suffix) = trimmed
        .strip_suffix('x')
        .or_else(|| trimmed.strip_suffix('X'))
    {
        without_suffix.trim().parse::<f32>().map_err(|err| {
            MotionLoomSceneRenderError::InvalidExpression {
                expr: expr.to_string(),
                message: format!("invalid Text renderScale value: {err}"),
            }
        })?
    } else {
        eval_scene_number(trimmed, time_norm, time_sec)?
    };
    Ok(scale.clamp(1.0, 8.0))
}

fn eval_text_edge_smoothing(
    text: &TextNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<f32, MotionLoomSceneRenderError> {
    if let Some(expr) = text
        .edge_smoothing
        .as_deref()
        .or(text.soft_edge.as_deref())
        .filter(|value| !value.trim().is_empty())
    {
        return Ok(eval_scene_number(expr, time_norm, time_sec)?.clamp(0.0, 1.0));
    }

    let Some(mode) = text.antialias.as_deref().map(str::trim) else {
        return Ok(0.0);
    };
    let normalized = mode.to_ascii_lowercase().replace(['-', '_', ' '], "");
    Ok(match normalized.as_str() {
        "none" | "off" | "false" => 0.0,
        "grayscale" | "gray" | "standard" | "on" | "true" => 0.25,
        "subpixel" | "smooth" | "strong" => 0.45,
        _ => eval_scene_number(mode, time_norm, time_sec)
            .map(|value| value.clamp(0.0, 1.0))
            .unwrap_or(0.25),
    })
}

fn eval_text_font_weight(
    expr: Option<&str>,
    time_norm: f32,
    time_sec: f32,
) -> Result<Weight, MotionLoomSceneRenderError> {
    let Some(expr) = expr.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(Weight::NORMAL);
    };
    let normalized = expr.to_ascii_lowercase().replace(['-', '_', ' '], "");
    let weight = match normalized.as_str() {
        "thin" | "hairline" => Weight::THIN,
        "extralight" | "ultralight" => Weight::EXTRA_LIGHT,
        "light" => Weight::LIGHT,
        "normal" | "regular" | "book" => Weight::NORMAL,
        "medium" => Weight::MEDIUM,
        "semibold" | "demibold" => Weight::SEMIBOLD,
        "bold" | "bolder" => Weight::BOLD,
        "extrabold" | "ultrabold" => Weight::EXTRA_BOLD,
        "black" | "heavy" => Weight::BLACK,
        "lighter" => Weight::LIGHT,
        _ => {
            let value = eval_scene_number(expr, time_norm, time_sec).map_err(|err| {
                MotionLoomSceneRenderError::InvalidExpression {
                    expr: expr.to_string(),
                    message: format!("invalid Text fontWeight value: {err}"),
                }
            })?;
            Weight(value.round().clamp(1.0, 1000.0) as u16)
        }
    };
    Ok(weight)
}

fn eval_text_tracking_em(
    expr: Option<&str>,
    font_size: f32,
    time_norm: f32,
    time_sec: f32,
) -> Result<f32, MotionLoomSceneRenderError> {
    let Some(expr) = expr.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(0.0);
    };
    let tracking_px = eval_scene_number(expr, time_norm, time_sec)?;
    Ok((tracking_px / font_size.max(1.0)).clamp(-1.0, 4.0))
}

#[derive(Clone, Copy, Debug)]
struct TextBoxStyle {
    color: [u8; 4],
    padding_x: f32,
    padding_y: f32,
    radius: f32,
}

#[derive(Clone, Copy, Debug)]
struct TextTextureRasterContext {
    transform: Affine2,
    inherited_opacity: f32,
    pick_id: u32,
    time_norm: f32,
    time_sec: f32,
    canvas_size: (u32, u32),
}

fn eval_text_box_style(
    text: &TextNode,
    text_width: f32,
    text_height: f32,
    time_norm: f32,
    time_sec: f32,
) -> Result<Option<TextBoxStyle>, MotionLoomSceneRenderError> {
    let Some(kind) = text.box_style.as_deref().map(str::trim) else {
        return Ok(None);
    };
    if kind.is_empty() {
        return Ok(None);
    }
    let normalized = kind.to_ascii_lowercase().replace(['-', '_', ' '], "");
    if matches!(normalized.as_str(), "none" | "false" | "off" | "0") {
        return Ok(None);
    }
    if !matches!(normalized.as_str(), "pill" | "rect" | "rectangle") {
        return Err(MotionLoomSceneRenderError::InvalidExpression {
            expr: kind.to_string(),
            message: "invalid Text box value. Expected pill, rect, or none.".to_string(),
        });
    }

    let (default_pad_x, default_pad_y) = text
        .box_padding
        .as_deref()
        .map(|expr| eval_text_box_padding(expr, time_norm, time_sec))
        .transpose()?
        .unwrap_or((0.0, 0.0));
    let padding_x = text
        .box_padding_x
        .as_deref()
        .map(|expr| eval_scene_number(expr, time_norm, time_sec).map(|value| value.max(0.0)))
        .transpose()?
        .unwrap_or(default_pad_x);
    let padding_y = text
        .box_padding_y
        .as_deref()
        .map(|expr| eval_scene_number(expr, time_norm, time_sec).map(|value| value.max(0.0)))
        .transpose()?
        .unwrap_or(default_pad_y);
    let box_height = (text_height + padding_y * 2.0).max(1.0);
    let radius = text
        .box_radius
        .as_deref()
        .map(|expr| eval_scene_number(expr, time_norm, time_sec).map(|value| value.max(0.0)))
        .transpose()?
        .unwrap_or_else(|| {
            if normalized == "pill" {
                box_height * 0.5
            } else {
                0.0
            }
        });
    let color = parse_color(text.box_color.as_deref().unwrap_or("#000000"))?;

    Ok(Some(TextBoxStyle {
        color,
        padding_x,
        padding_y,
        radius: radius
            .min((text_width + padding_x * 2.0) * 0.5)
            .min(box_height * 0.5),
    }))
}

fn eval_text_box_padding(
    expr: &str,
    time_norm: f32,
    time_sec: f32,
) -> Result<(f32, f32), MotionLoomSceneRenderError> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Ok((0.0, 0.0));
    }
    let (x_expr, y_expr) = split_text_box_padding_pair(trimmed).unwrap_or((trimmed, trimmed));
    let x = eval_scene_number(x_expr, time_norm, time_sec)?.max(0.0);
    let y = eval_scene_number(y_expr, time_norm, time_sec)?.max(0.0);
    Ok((x, y))
}

fn split_text_box_padding_pair(expr: &str) -> Option<(&str, &str)> {
    let mut depth = 0usize;
    let mut in_quote = false;
    let mut quote_char = '\0';
    for (ix, ch) in expr.char_indices() {
        if in_quote {
            if ch == quote_char {
                in_quote = false;
            }
            continue;
        }
        match ch {
            '"' | '\'' => {
                in_quote = true;
                quote_char = ch;
            }
            '(' | '[' | '{' => depth = depth.saturating_add(1),
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                let left = expr[..ix].trim();
                let right = expr[ix + ch.len_utf8()..].trim();
                if !left.is_empty() && !right.is_empty() {
                    return Some((left, right));
                }
            }
            _ if depth == 0 && ch.is_whitespace() => {
                let left = expr[..ix].trim();
                let right = expr[ix + ch.len_utf8()..].trim();
                if !left.is_empty() && !right.is_empty() {
                    return Some((left, right));
                }
            }
            _ => {}
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn collect_gpu_scene_commands(
    nodes: &[SceneNode],
    transform: Affine2,
    deform: Option<&EvaluatedDeformGrid>,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    canvas_size: (u32, u32),
    gradient_defs: &HashMap<String, GradientDef>,
    palette_defs: &HashMap<String, PaletteNode>,
    scene_components: &HashMap<String, Vec<SceneNode>>,
    primitives: &mut Vec<GpuScenePrimitive>,
    text_requests: &mut Vec<GpuSceneTextRequest>,
    scene_overlays: &mut Vec<CpuSceneOverlay>,
) -> Result<(), MotionLoomSceneRenderError> {
    collect_gpu_scene_commands_with_depth(
        nodes,
        transform,
        deform,
        inherited_opacity,
        time_norm,
        time_sec,
        canvas_size,
        gradient_defs,
        palette_defs,
        scene_components,
        primitives,
        text_requests,
        scene_overlays,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn collect_gpu_scene_commands_with_depth(
    nodes: &[SceneNode],
    transform: Affine2,
    deform: Option<&EvaluatedDeformGrid>,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    canvas_size: (u32, u32),
    gradient_defs: &HashMap<String, GradientDef>,
    palette_defs: &HashMap<String, PaletteNode>,
    scene_components: &HashMap<String, Vec<SceneNode>>,
    primitives: &mut Vec<GpuScenePrimitive>,
    text_requests: &mut Vec<GpuSceneTextRequest>,
    scene_overlays: &mut Vec<CpuSceneOverlay>,
    depth: Option<SceneDepthContext<'_>>,
) -> Result<(), MotionLoomSceneRenderError> {
    let mut pending_shadow: Option<EvaluatedShadow> = None;
    for node in nodes {
        match node {
            SceneNode::Defs(_) => {
                pending_shadow = None;
            }
            SceneNode::Timeline(timeline) => {
                let mut tracks = timeline
                    .children
                    .iter()
                    .filter_map(|node| match node {
                        SceneNode::Track(track) => Some(track),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                tracks.sort_by_key(|track| scene_track_composite_order(track));
                let active_camera = active_scene_camera_from_tracks(
                    &tracks,
                    canvas_size.0,
                    canvas_size.1,
                    time_norm,
                    time_sec,
                )?;
                tracks.sort_by(|a, b| {
                    let a_world = is_scene_world_track(a);
                    let b_world = is_scene_world_track(b);
                    match (a_world, b_world) {
                        (true, true) => {
                            let a_depth =
                                scene_depth_track_sort_key(&a.z_depth, time_norm, time_sec)
                                    .unwrap_or(0.0);
                            let b_depth =
                                scene_depth_track_sort_key(&b.z_depth, time_norm, time_sec)
                                    .unwrap_or(0.0);
                            b_depth.total_cmp(&a_depth).then_with(|| {
                                scene_track_composite_order(a).cmp(&scene_track_composite_order(b))
                            })
                        }
                        _ => scene_track_composite_order(a).cmp(&scene_track_composite_order(b)),
                    }
                });
                for track in tracks {
                    if is_scene_camera_track(track) {
                        continue;
                    }
                    let track_depth = if is_scene_world_track(track) {
                        Some(SceneDepthContext {
                            active_camera,
                            canvas_size,
                            track_z_depth: &track.z_depth,
                        })
                    } else {
                        depth
                    };
                    collect_gpu_scene_commands_with_depth(
                        &track.children,
                        transform,
                        deform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        gradient_defs,
                        palette_defs,
                        scene_components,
                        primitives,
                        text_requests,
                        scene_overlays,
                        track_depth,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Track(track) => {
                collect_gpu_scene_commands_with_depth(
                    &track.children,
                    transform,
                    deform,
                    inherited_opacity,
                    time_norm,
                    time_sec,
                    canvas_size,
                    gradient_defs,
                    palette_defs,
                    scene_components,
                    primitives,
                    text_requests,
                    scene_overlays,
                    depth,
                )?;
                pending_shadow = None;
            }
            SceneNode::Sequence(sequence) => {
                if let Some((local_norm, local_sec)) =
                    scene_sequence_local_time(sequence, None, time_sec)
                {
                    if depth.is_some() {
                        collect_gpu_scene_commands_depth_sorted(
                            &sequence.children,
                            transform,
                            deform,
                            inherited_opacity,
                            local_norm,
                            local_sec,
                            canvas_size,
                            gradient_defs,
                            palette_defs,
                            scene_components,
                            primitives,
                            text_requests,
                            scene_overlays,
                            depth,
                        )?;
                    } else {
                        collect_gpu_scene_commands_with_depth(
                            &sequence.children,
                            transform,
                            deform,
                            inherited_opacity,
                            local_norm,
                            local_sec,
                            canvas_size,
                            gradient_defs,
                            palette_defs,
                            scene_components,
                            primitives,
                            text_requests,
                            scene_overlays,
                            depth,
                        )?;
                    }
                }
                pending_shadow = None;
            }
            SceneNode::Chain(chain) => {
                let mut cursor_ms = chain.from_ms as i64;
                for child in &chain.children {
                    if let SceneNode::Sequence(sequence) = child {
                        if let Some((local_norm, local_sec)) =
                            scene_sequence_local_time(sequence, Some(cursor_ms), time_sec)
                        {
                            if depth.is_some() {
                                collect_gpu_scene_commands_depth_sorted(
                                    &sequence.children,
                                    transform,
                                    deform,
                                    inherited_opacity,
                                    local_norm,
                                    local_sec,
                                    canvas_size,
                                    gradient_defs,
                                    palette_defs,
                                    scene_components,
                                    primitives,
                                    text_requests,
                                    scene_overlays,
                                    depth,
                                )?;
                            } else {
                                collect_gpu_scene_commands_with_depth(
                                    &sequence.children,
                                    transform,
                                    deform,
                                    inherited_opacity,
                                    local_norm,
                                    local_sec,
                                    canvas_size,
                                    gradient_defs,
                                    palette_defs,
                                    scene_components,
                                    primitives,
                                    text_requests,
                                    scene_overlays,
                                    depth,
                                )?;
                            }
                        }
                        cursor_ms += sequence.duration_ms as i64 + chain.gap_ms;
                    }
                }
                pending_shadow = None;
            }
            SceneNode::Palette(_) => {
                pending_shadow = None;
            }
            SceneNode::PixelGrid(grid) => {
                if deform.is_some() {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![SceneNode::PixelGrid(grid.clone())],
                    });
                } else {
                    push_gpu_pixel_grid_commands(
                        grid,
                        transform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        palette_defs,
                        primitives,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Text(text) => {
                if deform.is_some() {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![SceneNode::Text(text.clone())],
                    });
                } else {
                    text_requests.push(GpuSceneTextRequest {
                        node: text.as_ref().clone(),
                        transform,
                        opacity: inherited_opacity,
                        pick_id: 0,
                    });
                }
                pending_shadow = None;
            }
            SceneNode::Rect(rect) => {
                if rect_requires_cpu_overlay(rect) {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![SceneNode::Rect(rect.clone())],
                    });
                    pending_shadow = None;
                } else {
                    push_gpu_rect_commands(
                        rect,
                        transform,
                        deform,
                        pending_shadow.take(),
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        gradient_defs,
                        primitives,
                    )?;
                }
            }
            SceneNode::Circle(circle) => {
                if circle_requires_cpu_overlay(circle) {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![SceneNode::Circle(circle.clone())],
                    });
                    pending_shadow = None;
                } else {
                    push_gpu_circle_commands(
                        circle,
                        transform,
                        deform,
                        pending_shadow.take(),
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        gradient_defs,
                        primitives,
                    )?;
                }
            }
            SceneNode::Ellipse(ellipse) => {
                push_gpu_ellipse_commands(
                    ellipse,
                    transform,
                    deform,
                    inherited_opacity,
                    time_norm,
                    time_sec,
                    gradient_defs,
                    primitives,
                )?;
                pending_shadow = None;
            }
            SceneNode::Line(line) => {
                if line_requires_cpu_overlay(line) {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![node.clone()],
                    });
                } else {
                    push_gpu_line_command(
                        line,
                        transform,
                        deform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        gradient_defs,
                        primitives,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Polyline(polyline) => {
                if polyline_requires_cpu_overlay(polyline) {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![node.clone()],
                    });
                } else {
                    push_gpu_polyline_commands(
                        polyline,
                        transform,
                        deform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        gradient_defs,
                        primitives,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Path(path) => {
                if path_requires_cpu_overlay(path) {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![node.clone()],
                    });
                } else {
                    push_gpu_path_commands(
                        path,
                        transform,
                        deform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        gradient_defs,
                        primitives,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::FaceJaw(face_jaw) => {
                let path = face_jaw_to_path_node(face_jaw, time_norm, time_sec)?;
                if path_requires_cpu_overlay(&path) {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![SceneNode::FaceJaw(face_jaw.clone())],
                    });
                } else {
                    push_gpu_path_commands(
                        &path,
                        transform,
                        deform,
                        inherited_opacity,
                        time_norm,
                        time_sec,
                        gradient_defs,
                        primitives,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Shadow(shadow) => {
                pending_shadow = Some(evaluate_shadow(
                    shadow,
                    time_norm,
                    time_sec,
                    inherited_opacity,
                )?);
            }
            SceneNode::Group(group) => {
                let opacity = (eval_scene_number(&group.opacity, time_norm, time_sec)?
                    * inherited_opacity)
                    .clamp(0.0, 1.0);
                if opacity > 0.0001 {
                    let group_transform =
                        transform.mul(scene_group_local_transform(group, time_norm, time_sec)?);
                    let group_deform = eval_group_deform_grid(group, time_norm, time_sec)?
                        .map(|grid| transform_deform_grid(&grid, group_transform));
                    let child_deform = group_deform.as_ref().or(deform);
                    collect_gpu_scene_commands(
                        &group.children,
                        group_transform,
                        child_deform,
                        opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        gradient_defs,
                        palette_defs,
                        scene_components,
                        primitives,
                        text_requests,
                        scene_overlays,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Puppet(puppet) => {
                let opacity = (eval_scene_number(&puppet.opacity, time_norm, time_sec)?
                    * inherited_opacity)
                    .clamp(0.0, 1.0);
                if opacity > 0.0001 {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![SceneNode::Puppet(puppet.clone())],
                    });
                }
                pending_shadow = None;
            }
            SceneNode::Pin(_)
            | SceneNode::LimbEnvelope(_)
            | SceneNode::LimbRegion(_)
            | SceneNode::MeshTopology(_)
            | SceneNode::Vertex(_)
            | SceneNode::Triangle(_)
            | SceneNode::Edge(_)
            | SceneNode::Region(_) => {
                pending_shadow = None;
            }
            SceneNode::Part(part) => {
                let opacity = (eval_scene_number(&part.opacity, time_norm, time_sec)?
                    * inherited_opacity)
                    .clamp(0.0, 1.0);
                if opacity > 0.0001 {
                    let x = eval_scene_number(&part.x, time_norm, time_sec)?;
                    let y = eval_scene_number(&part.y, time_norm, time_sec)?;
                    let rotation = eval_scene_number(&part.rotation, time_norm, time_sec)?;
                    let scale =
                        eval_scene_number(&part.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
                    let anchor_x = eval_scene_number(&part.anchor_x, time_norm, time_sec)?;
                    let anchor_y = eval_scene_number(&part.anchor_y, time_norm, time_sec)?;
                    let part_transform = transform
                        .mul(Affine2::translate(x, y))
                        .mul(Affine2::rotate_deg(rotation))
                        .mul(Affine2::scale(scale))
                        .mul(Affine2::translate(-anchor_x, -anchor_y));
                    collect_gpu_scene_commands(
                        &part.children,
                        part_transform,
                        deform,
                        opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        gradient_defs,
                        palette_defs,
                        scene_components,
                        primitives,
                        text_requests,
                        scene_overlays,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Repeat(repeat) => {
                let count = eval_repeat_count(&repeat.count, time_norm, time_sec)?;
                let x = eval_scene_number(&repeat.x, time_norm, time_sec)?;
                let y = eval_scene_number(&repeat.y, time_norm, time_sec)?;
                let rotation = eval_scene_number(&repeat.rotation, time_norm, time_sec)?;
                let scale =
                    eval_scene_number(&repeat.scale, time_norm, time_sec)?.clamp(0.001, 64.0);
                let opacity = eval_scene_number(&repeat.opacity, time_norm, time_sec)?;
                let x_step = eval_scene_number(&repeat.x_step, time_norm, time_sec)?;
                let y_step = eval_scene_number(&repeat.y_step, time_norm, time_sec)?;
                let rotation_step = eval_scene_number(&repeat.rotation_step, time_norm, time_sec)?;
                let scale_step = eval_scene_number(&repeat.scale_step, time_norm, time_sec)?;
                let opacity_step = eval_scene_number(&repeat.opacity_step, time_norm, time_sec)?;
                for index in 0..count {
                    let i = index as f32;
                    let copy_opacity =
                        ((opacity + opacity_step * i) * inherited_opacity).clamp(0.0, 1.0);
                    if copy_opacity <= 0.0001 {
                        continue;
                    }
                    let repeat_transform = transform
                        .mul(Affine2::translate(x + x_step * i, y + y_step * i))
                        .mul(Affine2::rotate_deg(rotation + rotation_step * i))
                        .mul(Affine2::scale((scale + scale_step * i).clamp(0.001, 64.0)));
                    collect_gpu_scene_commands(
                        &repeat.children,
                        repeat_transform,
                        deform,
                        copy_opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        gradient_defs,
                        palette_defs,
                        scene_components,
                        primitives,
                        text_requests,
                        scene_overlays,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Mask(mask) => {
                scene_overlays.push(CpuSceneOverlay::Vector {
                    nodes: vec![SceneNode::Mask(mask.clone())],
                });
                pending_shadow = None;
            }
            SceneNode::Precompose(precompose) => {
                scene_overlays.push(CpuSceneOverlay::Vector {
                    nodes: vec![SceneNode::Precompose(precompose.clone())],
                });
                pending_shadow = None;
            }
            SceneNode::Use(use_node) => {
                // Expand Component references so reusable scene fragments stay GPU-native.
                let opacity = (eval_scene_number(&use_node.opacity, time_norm, time_sec)?
                    * inherited_opacity)
                    .clamp(0.0, 1.0);
                let Some(children) = scene_components.get(&use_node.ref_id) else {
                    pending_shadow = None;
                    continue;
                };
                if opacity > 0.0001 {
                    let use_transform =
                        transform.mul(scene_use_local_transform(use_node, time_norm, time_sec)?);
                    let primitive_start = primitives.len();
                    let text_start = text_requests.len();
                    collect_gpu_scene_commands(
                        children,
                        use_transform,
                        deform,
                        opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        gradient_defs,
                        palette_defs,
                        scene_components,
                        primitives,
                        text_requests,
                        scene_overlays,
                    )?;

                    let use_blend = parse_scene_blend(&use_node.blend)?;
                    if use_blend != SceneBlendMode::Normal {
                        if text_requests.len() > text_start {
                            primitives.truncate(primitive_start);
                            text_requests.truncate(text_start);
                            scene_overlays.push(CpuSceneOverlay::Vector {
                                nodes: vec![SceneNode::Use(use_node.clone())],
                            });
                        } else {
                            for primitive in &mut primitives[primitive_start..] {
                                primitive.blend = use_blend;
                            }
                        }
                    }
                }
                pending_shadow = None;
            }
            SceneNode::Layer(layer) => {
                if layer.is_3d {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![SceneNode::Layer(layer.clone())],
                    });
                    pending_shadow = None;
                    continue;
                }
                if layer.source.is_some()
                    || layer.mask.is_some()
                    || layer.matte.is_some()
                    || layer.effect.is_some()
                {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![SceneNode::Layer(layer.clone())],
                    });
                    pending_shadow = None;
                    continue;
                }
                let opacity = (eval_scene_number(&layer.opacity, time_norm, time_sec)?
                    * inherited_opacity)
                    .clamp(0.0, 1.0);
                if opacity > 0.0001 && parse_scene_blend(&layer.blend)? == SceneBlendMode::Normal {
                    let base_transform = if let Some(depth) = depth {
                        let z_depth =
                            scene_layer_effective_z_depth(layer, depth, time_norm, time_sec)?;
                        transform.mul(scene_z_depth_transform(
                            depth.active_camera,
                            depth.canvas_size,
                            z_depth,
                        ))
                    } else {
                        transform
                    };
                    collect_gpu_scene_commands_with_depth(
                        &layer.children,
                        base_transform
                            .mul(scene_layer_local_transform(layer, time_norm, time_sec)?),
                        deform,
                        opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        gradient_defs,
                        palette_defs,
                        scene_components,
                        primitives,
                        text_requests,
                        scene_overlays,
                        None,
                    )?;
                } else if opacity > 0.0001 {
                    scene_overlays.push(CpuSceneOverlay::Vector {
                        nodes: vec![SceneNode::Layer(layer.clone())],
                    });
                }
                pending_shadow = None;
            }
            SceneNode::Camera(camera) => {
                let opacity = (eval_scene_number(&camera.opacity, time_norm, time_sec)?
                    * inherited_opacity)
                    .clamp(0.0, 1.0);
                if opacity > 0.0001 {
                    let camera_transform = camera_transform(
                        camera,
                        &camera.children,
                        canvas_size.0,
                        canvas_size.1,
                        time_norm,
                        time_sec,
                    )?;
                    collect_gpu_scene_commands(
                        &camera.children,
                        transform.mul(camera_transform),
                        deform,
                        opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        gradient_defs,
                        palette_defs,
                        scene_components,
                        primitives,
                        text_requests,
                        scene_overlays,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Character(character) => {
                let opacity = (eval_scene_number(&character.opacity, time_norm, time_sec)?
                    * inherited_opacity)
                    .clamp(0.0, 1.0);
                if opacity > 0.0001 {
                    let character_transform = transform.mul(scene_character_local_transform(
                        character, time_norm, time_sec,
                    )?);
                    collect_gpu_scene_commands(
                        &character.children,
                        character_transform,
                        deform,
                        opacity,
                        time_norm,
                        time_sec,
                        canvas_size,
                        gradient_defs,
                        palette_defs,
                        scene_components,
                        primitives,
                        text_requests,
                        scene_overlays,
                    )?;
                }
                pending_shadow = None;
            }
            SceneNode::Image(_) | SceneNode::Svg(_) => {
                pending_shadow = None;
            }
            SceneNode::Simulation(_) => {}
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_gpu_scene_commands_depth_sorted(
    nodes: &[SceneNode],
    transform: Affine2,
    deform: Option<&EvaluatedDeformGrid>,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    canvas_size: (u32, u32),
    gradient_defs: &HashMap<String, GradientDef>,
    palette_defs: &HashMap<String, PaletteNode>,
    scene_components: &HashMap<String, Vec<SceneNode>>,
    primitives: &mut Vec<GpuScenePrimitive>,
    text_requests: &mut Vec<GpuSceneTextRequest>,
    scene_overlays: &mut Vec<CpuSceneOverlay>,
    depth: Option<SceneDepthContext<'_>>,
) -> Result<(), MotionLoomSceneRenderError> {
    let Some(depth) = depth else {
        return collect_gpu_scene_commands_with_depth(
            nodes,
            transform,
            deform,
            inherited_opacity,
            time_norm,
            time_sec,
            canvas_size,
            gradient_defs,
            palette_defs,
            scene_components,
            primitives,
            text_requests,
            scene_overlays,
            None,
        );
    };
    let mut layer_items = Vec::<(usize, f32, &SceneNode)>::new();
    for (index, node) in nodes.iter().enumerate() {
        if let SceneNode::Layer(layer) = node {
            layer_items.push((
                index,
                scene_layer_effective_z_depth(layer, depth, time_norm, time_sec)?,
                node,
            ));
        } else {
            collect_gpu_scene_commands_with_depth(
                std::slice::from_ref(node),
                transform,
                deform,
                inherited_opacity,
                time_norm,
                time_sec,
                canvas_size,
                gradient_defs,
                palette_defs,
                scene_components,
                primitives,
                text_requests,
                scene_overlays,
                Some(depth),
            )?;
        }
    }
    layer_items.sort_by(|(a_order, a_depth, _), (b_order, b_depth, _)| {
        b_depth
            .total_cmp(a_depth)
            .then_with(|| a_order.cmp(b_order))
    });
    for (_, _, node) in layer_items {
        collect_gpu_scene_commands_with_depth(
            std::slice::from_ref(node),
            transform,
            deform,
            inherited_opacity,
            time_norm,
            time_sec,
            canvas_size,
            gradient_defs,
            palette_defs,
            scene_components,
            primitives,
            text_requests,
            scene_overlays,
            Some(depth),
        )?;
    }
    Ok(())
}

fn rect_requires_cpu_overlay(rect: &RectNode) -> bool {
    !is_gpu_native_blend(&rect.blend)
}

fn circle_requires_cpu_overlay(circle: &CircleNode) -> bool {
    !is_gpu_native_blend(&circle.blend)
}

fn line_requires_cpu_overlay(line: &LineNode) -> bool {
    !is_gpu_native_blend(&line.blend) || !is_default_line_cap(&line.line_cap)
}

fn polyline_requires_cpu_overlay(polyline: &PolylineNode) -> bool {
    !is_gpu_native_blend(&polyline.blend)
        || !is_default_line_cap(&polyline.line_cap)
        || !is_default_line_join(&polyline.line_join)
}

fn push_gpu_pixel_grid_commands(
    grid: &PixelGridNode,
    transform: Affine2,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    palette_defs: &HashMap<String, PaletteNode>,
    primitives: &mut Vec<GpuScenePrimitive>,
) -> Result<(), MotionLoomSceneRenderError> {
    let opacity = (eval_scene_number(&grid.opacity, time_norm, time_sec)? * inherited_opacity)
        .clamp(0.0, 1.0);
    if opacity <= 0.0001 {
        return Ok(());
    }
    let pixel_size = eval_scene_number(&grid.pixel_size, time_norm, time_sec)?.max(0.0);
    if pixel_size <= 0.0001 {
        return Ok(());
    }
    let x = eval_scene_number(&grid.x, time_norm, time_sec)?;
    let y = eval_scene_number(&grid.y, time_norm, time_sec)?;
    let blend = parse_scene_blend(&grid.blend)?;
    let palette = palette_defs.get(&grid.palette).ok_or_else(|| {
        MotionLoomSceneRenderError::InvalidPaint {
            value: grid.palette.clone(),
            message: format!("PixelGrid palette not found: {}", grid.palette),
        }
    })?;

    for (row, line) in grid.data.lines().enumerate() {
        for (col, ch) in line.chars().enumerate() {
            if ch.is_whitespace() {
                continue;
            }
            let key = ch.to_string();
            let Some(color_def) = palette.colors.iter().find(|color| color.key == key) else {
                return Err(MotionLoomSceneRenderError::InvalidPaint {
                    value: key,
                    message: format!(
                        "PixelGrid{} references color key not found in palette '{}'",
                        id_suffix(grid.id.as_deref()),
                        grid.palette
                    ),
                });
            };
            let color = parse_color(&color_def.value)?;
            if color[3] == 0 {
                continue;
            }
            primitives.push(GpuScenePrimitive {
                kind: GPU_SHAPE_RECT_FILL,
                transform,
                shape: [
                    x + col as f32 * pixel_size,
                    y + row as f32 * pixel_size,
                    pixel_size,
                    pixel_size,
                ],
                radius: 0.0,
                stroke_width: 0.0,
                blur: 0.0,
                color,
                opacity,
                blend,
                pick_id: 0,
                gradient: None,
                line_t0: 0.0,
                line_t1: 1.0,
                taper_start: 0.0,
                taper_end: 0.0,
            });
        }
    }
    Ok(())
}

fn path_requires_cpu_overlay(path: &PathNode) -> bool {
    let has_visible_fill = path
        .fill
        .as_deref()
        .is_some_and(|fill| !is_none_paint(fill));
    !is_gpu_native_blend(&path.blend)
        || !is_default_line_cap(&path.line_cap)
        || !is_default_line_join(&path.line_join)
        || (is_none_paint(&path.stroke) && !has_visible_fill)
}

fn gpu_path_style_is_cacheable(path: &PathNode) -> bool {
    let uses_gradient = path
        .fill
        .as_deref()
        .is_some_and(|paint| gradient_ref_id(paint).is_some())
        || gradient_ref_id(&path.stroke).is_some();
    if uses_gradient {
        return false;
    }
    [
        path.offset_path.as_str(),
        path.round_corners.as_str(),
        path.stroke_width.as_str(),
        path.stroke_width_start.as_str(),
        path.stroke_width_end.as_str(),
        path.opacity.as_str(),
        path.trim_start.as_str(),
        path.trim_end.as_str(),
        path.taper_start.as_str(),
        path.taper_end.as_str(),
        path.stroke_roughness.as_str(),
        path.stroke_copies.as_str(),
        path.stroke_bristles.as_str(),
        path.stroke_pressure_min.as_str(),
        path.texture_opacity.as_str(),
        path.texture_scale.as_str(),
        path.texture_mask.as_str(),
    ]
    .into_iter()
    .all(|value| value.trim().parse::<f32>().is_ok())
        && (path.stroke_pressure.trim().eq_ignore_ascii_case("none")
            || path.stroke_pressure.trim().parse::<f32>().is_ok())
}

fn gpu_path_geometry_signature(path: &PathNode, evaluated_d: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    path.brush.hash(&mut hasher);
    path.material.hash(&mut hasher);
    evaluated_d.hash(&mut hasher);
    path.stroke.hash(&mut hasher);
    path.fill.hash(&mut hasher);
    path.fill_rule.hash(&mut hasher);
    path.boolean_op.hash(&mut hasher);
    path.offset_path.hash(&mut hasher);
    path.round_corners.hash(&mut hasher);
    path.normalize.hash(&mut hasher);
    path.stroke_width.hash(&mut hasher);
    path.stroke_width_start.hash(&mut hasher);
    path.stroke_width_end.hash(&mut hasher);
    path.stroke_width_profile.hash(&mut hasher);
    path.opacity.hash(&mut hasher);
    path.trim_start.hash(&mut hasher);
    path.trim_end.hash(&mut hasher);
    path.line_cap.hash(&mut hasher);
    path.line_join.hash(&mut hasher);
    path.taper_start.hash(&mut hasher);
    path.taper_end.hash(&mut hasher);
    path.stroke_style.hash(&mut hasher);
    path.stroke_roughness.hash(&mut hasher);
    path.stroke_copies.hash(&mut hasher);
    path.stroke_texture.hash(&mut hasher);
    path.stroke_bristles.hash(&mut hasher);
    path.stroke_pressure.hash(&mut hasher);
    path.stroke_pressure_min.hash(&mut hasher);
    path.stroke_pressure_curve.hash(&mut hasher);
    path.blend.hash(&mut hasher);
    path.texture.hash(&mut hasher);
    path.texture_opacity.hash(&mut hasher);
    path.texture_scale.hash(&mut hasher);
    path.texture_mask.hash(&mut hasher);
    hasher.finish()
}

fn evaluated_path_subpaths(
    path: &PathNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<Vec<Vec<Point2>>, MotionLoomSceneRenderError> {
    let path_d = eval_path_d(&path.d, time_norm, time_sec)?;
    let mut subpaths = parse_path_subpaths(path_d.as_ref())?;
    if scene_bool(&path.normalize) {
        normalize_path_subpaths(&mut subpaths);
    }
    let round_corners = eval_scene_number(&path.round_corners, time_norm, time_sec)?.max(0.0);
    if round_corners > 0.0001 {
        subpaths = round_path_subpaths(&subpaths, round_corners);
    }
    let offset = eval_scene_number(&path.offset_path, time_norm, time_sec)?;
    if offset.abs() > 0.0001 {
        subpaths = offset_path_subpaths(&subpaths, offset);
    }
    Ok(subpaths)
}

fn evaluated_path_fill_rule(path: &PathNode) -> FillRule {
    let boolean_op = path.boolean_op.trim();
    if boolean_op.is_empty() || boolean_op.eq_ignore_ascii_case("none") {
        FillRule::parse(&path.fill_rule)
    } else {
        FillRule::parse(boolean_op)
    }
}

fn is_default_line_cap(value: &str) -> bool {
    let value = value.trim().to_ascii_lowercase();
    value.is_empty() || value == "round"
}

fn is_default_line_join(value: &str) -> bool {
    let value = value.trim().to_ascii_lowercase();
    value.is_empty() || value == "round"
}

fn resolve_gpu_scene_paint(
    value: &str,
    gradient_defs: &HashMap<String, GradientDef>,
    bounds: PaintBounds,
) -> Result<([u8; 4], Option<GpuSceneGradientPaint>), MotionLoomSceneRenderError> {
    if let Some(id) = gradient_ref_id(value) {
        let Some(gradient) = gradient_defs.get(id) else {
            return Err(MotionLoomSceneRenderError::InvalidPaint {
                value: value.to_string(),
                message: format!("gradient reference not found: {id}"),
            });
        };
        return Ok((
            [255, 255, 255, 255],
            Some(GpuSceneGradientPaint {
                gradient: resolve_gradient_paint(value, gradient)?,
                bounds,
            }),
        ));
    }
    Ok((parse_color(value)?, None))
}

fn points_bounds(points: &[Point2]) -> Option<PaintBounds> {
    let mut iter = points.iter();
    let first = *iter.next()?;
    let mut min_x = first.x;
    let mut min_y = first.y;
    let mut max_x = first.x;
    let mut max_y = first.y;
    for point in iter {
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }
    Some(PaintBounds::new(min_x, min_y, max_x, max_y))
}

fn subpaths_bounds(subpaths: &[Vec<Point2>]) -> Option<PaintBounds> {
    let mut bounds: Option<PaintBounds> = None;
    for subpath in subpaths {
        let Some(next) = points_bounds(subpath) else {
            continue;
        };
        bounds = Some(match bounds {
            Some(current) => PaintBounds::new(
                current.min_x.min(next.min_x),
                current.min_y.min(next.min_y),
                current.max_x.max(next.max_x),
                current.max_y.max(next.max_y),
            ),
            None => next,
        });
    }
    bounds
}

fn rect_polygon(x: f32, y: f32, width: f32, height: f32) -> Vec<Point2> {
    vec![
        Point2::new(x, y),
        Point2::new(x + width, y),
        Point2::new(x + width, y + height),
        Point2::new(x, y + height),
        Point2::new(x, y),
    ]
}

fn circle_polygon(x: f32, y: f32, radius: f32) -> Vec<Point2> {
    if radius <= 0.0001 {
        return Vec::new();
    }
    ellipse_polygon(x, y, radius, radius)
}

fn ellipse_polygon(x: f32, y: f32, radius_x: f32, radius_y: f32) -> Vec<Point2> {
    if radius_x <= 0.0001 || radius_y <= 0.0001 {
        return Vec::new();
    }
    let steps = 48usize;
    let mut points = Vec::with_capacity(steps + 1);
    for ix in 0..=steps {
        let t = ix as f32 / steps as f32 * std::f32::consts::TAU;
        points.push(Point2::new(x + t.cos() * radius_x, y + t.sin() * radius_y));
    }
    points
}

fn color_with_opacity(mut color: [u8; 4], opacity: f32) -> [u8; 4] {
    color[3] = ((color[3] as f32) * opacity.clamp(0.0, 1.0))
        .round()
        .clamp(0.0, 255.0) as u8;
    color
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_rect_commands(
    rect: &RectNode,
    transform: Affine2,
    deform: Option<&EvaluatedDeformGrid>,
    shadow: Option<EvaluatedShadow>,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    gradient_defs: &HashMap<String, GradientDef>,
    primitives: &mut Vec<GpuScenePrimitive>,
) -> Result<(), MotionLoomSceneRenderError> {
    let opacity = (eval_scene_number(&rect.opacity, time_norm, time_sec)? * inherited_opacity)
        .clamp(0.0, 1.0);
    if opacity <= 0.0001 {
        return Ok(());
    }
    let x = eval_scene_number(&rect.x, time_norm, time_sec)?;
    let y = eval_scene_number(&rect.y, time_norm, time_sec)?;
    let width = eval_scene_number(&rect.width, time_norm, time_sec)?.max(0.0);
    let height = eval_scene_number(&rect.height, time_norm, time_sec)?.max(0.0);
    let radius = eval_scene_number(&rect.radius, time_norm, time_sec)?.max(0.0);
    let shape_transform = transform.mul(scene_rect_local_transform(rect, time_norm, time_sec)?);
    let paint_bounds = PaintBounds::new(x, y, x + width, y + height);
    let fill_blend = parse_scene_blend(&rect.blend)?;

    if let Some(deform) = deform {
        if let Some(shadow) = shadow {
            let shadow_subpaths = vec![rect_polygon(x + shadow.x, y + shadow.y, width, height)];
            let shadow_subpaths =
                transform_and_deform_subpaths(&shadow_subpaths, shape_transform, deform);
            push_gpu_filled_path_triangles(
                primitives,
                Affine2::identity(),
                &shadow_subpaths,
                shadow.color,
                1.0,
                None,
            );
        }

        let subpaths = vec![rect_polygon(x, y, width, height)];
        let subpaths = transform_and_deform_subpaths(&subpaths, shape_transform, deform);
        let warped_bounds =
            subpaths_bounds(&subpaths).unwrap_or_else(|| PaintBounds::new(0.0, 0.0, 1.0, 1.0));
        if !is_none_paint(&rect.color) {
            let (color, gradient) =
                resolve_gpu_scene_paint(&rect.color, gradient_defs, warped_bounds)?;
            push_gpu_filled_path_triangles_with_blend(
                primitives,
                Affine2::identity(),
                &subpaths,
                GpuFilledPathStyle {
                    color,
                    opacity,
                    gradient,
                    blend: fill_blend,
                    pick_id: 0,
                },
            );
        }

        if let Some(stroke_value) = rect
            .stroke
            .as_deref()
            .filter(|stroke| !is_none_paint(stroke))
        {
            let stroke_width = eval_scene_number(&rect.stroke_width, time_norm, time_sec)?.max(0.0)
                * affine_uniform_scale(shape_transform);
            if stroke_width > 0.0 {
                let (stroke, gradient) =
                    resolve_gpu_scene_paint(stroke_value, gradient_defs, warped_bounds)?;
                push_gpu_stroke_segments(
                    primitives,
                    Affine2::identity(),
                    &subpaths,
                    stroke_width,
                    stroke,
                    opacity,
                    gradient,
                    (0.0, 1.0),
                    StrokeStyle::default(),
                    fill_blend,
                );
            }
        }
        let _ = radius;
        return Ok(());
    }

    if let Some(shadow) = shadow {
        primitives.push(GpuScenePrimitive {
            kind: GPU_SHAPE_RECT_SHADOW,
            transform: shape_transform,
            shape: [x + shadow.x, y + shadow.y, width, height],
            radius,
            stroke_width: 0.0,
            blur: shadow.blur,
            color: shadow.color,
            opacity: 1.0,
            blend: SceneBlendMode::Normal,
            pick_id: 0,
            gradient: None,
            line_t0: 0.0,
            line_t1: 1.0,
            taper_start: 0.0,
            taper_end: 0.0,
        });
    }

    if !is_none_paint(&rect.color) {
        let (color, gradient) = resolve_gpu_scene_paint(&rect.color, gradient_defs, paint_bounds)?;
        primitives.push(GpuScenePrimitive {
            kind: GPU_SHAPE_RECT_FILL,
            transform: shape_transform,
            shape: [x, y, width, height],
            radius,
            stroke_width: 0.0,
            blur: 0.0,
            color,
            opacity,
            blend: fill_blend,
            pick_id: 0,
            gradient,
            line_t0: 0.0,
            line_t1: 1.0,
            taper_start: 0.0,
            taper_end: 0.0,
        });
    }

    if let Some(stroke_value) = rect
        .stroke
        .as_deref()
        .filter(|stroke| !is_none_paint(stroke))
    {
        let stroke_width = eval_scene_number(&rect.stroke_width, time_norm, time_sec)?.max(0.0);
        if stroke_width > 0.0 {
            let (stroke, gradient) =
                resolve_gpu_scene_paint(stroke_value, gradient_defs, paint_bounds)?;
            primitives.push(GpuScenePrimitive {
                kind: GPU_SHAPE_RECT_STROKE,
                transform: shape_transform,
                shape: [x, y, width, height],
                radius,
                stroke_width,
                blur: 0.0,
                color: stroke,
                opacity,
                blend: SceneBlendMode::Normal,
                pick_id: 0,
                gradient,
                line_t0: 0.0,
                line_t1: 1.0,
                taper_start: 0.0,
                taper_end: 0.0,
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_circle_commands(
    circle: &CircleNode,
    transform: Affine2,
    deform: Option<&EvaluatedDeformGrid>,
    shadow: Option<EvaluatedShadow>,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    gradient_defs: &HashMap<String, GradientDef>,
    primitives: &mut Vec<GpuScenePrimitive>,
) -> Result<(), MotionLoomSceneRenderError> {
    let opacity = (eval_scene_number(&circle.opacity, time_norm, time_sec)? * inherited_opacity)
        .clamp(0.0, 1.0);
    if opacity <= 0.0001 {
        return Ok(());
    }
    let x = eval_scene_number(&circle.x, time_norm, time_sec)?;
    let y = eval_scene_number(&circle.y, time_norm, time_sec)?;
    let radius = eval_scene_number(&circle.radius, time_norm, time_sec)?.max(0.0);
    let shape_transform = transform.mul(scene_circle_local_transform(circle, time_norm, time_sec)?);
    let paint_bounds = PaintBounds::new(x - radius, y - radius, x + radius, y + radius);
    let blend = parse_scene_blend(&circle.blend)?;

    if let Some(deform) = deform {
        if let Some(shadow) = shadow {
            let shadow_subpaths = vec![circle_polygon(x + shadow.x, y + shadow.y, radius)];
            let shadow_subpaths =
                transform_and_deform_subpaths(&shadow_subpaths, shape_transform, deform);
            push_gpu_filled_path_triangles(
                primitives,
                Affine2::identity(),
                &shadow_subpaths,
                shadow.color,
                1.0,
                None,
            );
        }

        let subpaths = vec![circle_polygon(x, y, radius)];
        let subpaths = transform_and_deform_subpaths(&subpaths, shape_transform, deform);
        let warped_bounds =
            subpaths_bounds(&subpaths).unwrap_or_else(|| PaintBounds::new(0.0, 0.0, 1.0, 1.0));
        if !is_none_paint(&circle.color) {
            let (color, gradient) =
                resolve_gpu_scene_paint(&circle.color, gradient_defs, warped_bounds)?;
            push_gpu_filled_path_triangles_with_blend(
                primitives,
                Affine2::identity(),
                &subpaths,
                GpuFilledPathStyle {
                    color,
                    opacity,
                    gradient,
                    blend,
                    pick_id: 0,
                },
            );
        }

        if let Some(stroke_value) = circle
            .stroke
            .as_deref()
            .filter(|stroke| !is_none_paint(stroke))
        {
            let stroke_width = eval_scene_number(&circle.stroke_width, time_norm, time_sec)?
                .max(0.0)
                * affine_uniform_scale(shape_transform);
            if stroke_width > 0.0 {
                let (stroke, gradient) =
                    resolve_gpu_scene_paint(stroke_value, gradient_defs, warped_bounds)?;
                push_gpu_stroke_segments(
                    primitives,
                    Affine2::identity(),
                    &subpaths,
                    stroke_width,
                    stroke,
                    opacity,
                    gradient,
                    (0.0, 1.0),
                    StrokeStyle::default(),
                    blend,
                );
            }
        }
        return Ok(());
    }

    if let Some(shadow) = shadow {
        primitives.push(GpuScenePrimitive {
            kind: GPU_SHAPE_CIRCLE_SHADOW,
            transform: shape_transform,
            shape: [x + shadow.x, y + shadow.y, radius, 0.0],
            radius: 0.0,
            stroke_width: 0.0,
            blur: shadow.blur,
            color: shadow.color,
            opacity: 1.0,
            blend: SceneBlendMode::Normal,
            pick_id: 0,
            gradient: None,
            line_t0: 0.0,
            line_t1: 1.0,
            taper_start: 0.0,
            taper_end: 0.0,
        });
    }

    if !is_none_paint(&circle.color) {
        let (color, gradient) =
            resolve_gpu_scene_paint(&circle.color, gradient_defs, paint_bounds)?;
        primitives.push(GpuScenePrimitive {
            kind: GPU_SHAPE_CIRCLE_FILL,
            transform: shape_transform,
            shape: [x, y, radius, 0.0],
            radius: 0.0,
            stroke_width: 0.0,
            blur: 0.0,
            color,
            opacity,
            blend,
            pick_id: 0,
            gradient,
            line_t0: 0.0,
            line_t1: 1.0,
            taper_start: 0.0,
            taper_end: 0.0,
        });
    }

    if let Some(stroke_value) = circle
        .stroke
        .as_deref()
        .filter(|stroke| !is_none_paint(stroke))
    {
        let stroke_width = eval_scene_number(&circle.stroke_width, time_norm, time_sec)?.max(0.0);
        if stroke_width > 0.0 {
            let (stroke, gradient) =
                resolve_gpu_scene_paint(stroke_value, gradient_defs, paint_bounds)?;
            let (stroke, stroke_opacity) = if gradient.is_none() {
                (color_with_opacity(stroke, opacity), 1.0)
            } else {
                (stroke, opacity)
            };
            primitives.push(GpuScenePrimitive {
                kind: GPU_SHAPE_CIRCLE_STROKE,
                transform: shape_transform,
                shape: [x, y, radius, 0.0],
                radius: 0.0,
                stroke_width,
                blur: 0.0,
                color: stroke,
                opacity: stroke_opacity,
                blend,
                pick_id: 0,
                gradient,
                line_t0: 0.0,
                line_t1: 1.0,
                taper_start: 0.0,
                taper_end: 0.0,
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_ellipse_commands(
    ellipse: &EllipseNode,
    transform: Affine2,
    deform: Option<&EvaluatedDeformGrid>,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    gradient_defs: &HashMap<String, GradientDef>,
    primitives: &mut Vec<GpuScenePrimitive>,
) -> Result<(), MotionLoomSceneRenderError> {
    let opacity = (eval_scene_number(&ellipse.opacity, time_norm, time_sec)? * inherited_opacity)
        .clamp(0.0, 1.0);
    if opacity <= 0.0001 {
        return Ok(());
    }
    let x = eval_scene_number(&ellipse.x, time_norm, time_sec)?;
    let y = eval_scene_number(&ellipse.y, time_norm, time_sec)?;
    let radius_x = eval_scene_number(&ellipse.radius_x, time_norm, time_sec)?.max(0.0);
    let radius_y = eval_scene_number(&ellipse.radius_y, time_norm, time_sec)?.max(0.0);
    if radius_x <= 0.0001 || radius_y <= 0.0001 {
        return Ok(());
    }
    let shape_transform =
        transform.mul(scene_ellipse_local_transform(ellipse, time_norm, time_sec)?);
    let blend = parse_scene_blend(&ellipse.blend)?;
    let paint_bounds = PaintBounds::new(x - radius_x, y - radius_y, x + radius_x, y + radius_y);

    if deform.is_none() {
        if !is_none_paint(&ellipse.color) {
            let (color, gradient) =
                resolve_gpu_scene_paint(&ellipse.color, gradient_defs, paint_bounds)?;
            primitives.push(GpuScenePrimitive {
                kind: GPU_SHAPE_ELLIPSE_FILL,
                transform: shape_transform,
                shape: [x, y, radius_x, radius_y],
                radius: 0.0,
                stroke_width: 0.0,
                blur: 0.0,
                color,
                opacity,
                blend,
                pick_id: 0,
                gradient,
                line_t0: 0.0,
                line_t1: 1.0,
                taper_start: 0.0,
                taper_end: 0.0,
            });
        }

        if let Some(stroke_value) = ellipse
            .stroke
            .as_deref()
            .filter(|stroke| !is_none_paint(stroke))
        {
            let stroke_width =
                eval_scene_number(&ellipse.stroke_width, time_norm, time_sec)?.max(0.0);
            if stroke_width > 0.0 {
                let (stroke, gradient) =
                    resolve_gpu_scene_paint(stroke_value, gradient_defs, paint_bounds)?;
                let (stroke, stroke_opacity) = if gradient.is_none() {
                    (color_with_opacity(stroke, opacity), 1.0)
                } else {
                    (stroke, opacity)
                };
                primitives.push(GpuScenePrimitive {
                    kind: GPU_SHAPE_ELLIPSE_STROKE,
                    transform: shape_transform,
                    shape: [x, y, radius_x, radius_y],
                    radius: 0.0,
                    stroke_width,
                    blur: 0.0,
                    color: stroke,
                    opacity: stroke_opacity,
                    blend,
                    pick_id: 0,
                    gradient,
                    line_t0: 0.0,
                    line_t1: 1.0,
                    taper_start: 0.0,
                    taper_end: 0.0,
                });
            }
        }
        return Ok(());
    }

    let subpaths = vec![ellipse_polygon(x, y, radius_x, radius_y)];
    let subpaths =
        transform_and_deform_subpaths(&subpaths, shape_transform, deform.expect("checked above"));
    let warped_bounds =
        subpaths_bounds(&subpaths).unwrap_or_else(|| PaintBounds::new(0.0, 0.0, 1.0, 1.0));
    if !is_none_paint(&ellipse.color) {
        let (color, gradient) =
            resolve_gpu_scene_paint(&ellipse.color, gradient_defs, warped_bounds)?;
        push_gpu_filled_path_triangles_with_blend(
            primitives,
            Affine2::identity(),
            &subpaths,
            GpuFilledPathStyle {
                color,
                opacity,
                gradient,
                blend,
                pick_id: 0,
            },
        );
    }

    if let Some(stroke_value) = ellipse
        .stroke
        .as_deref()
        .filter(|stroke| !is_none_paint(stroke))
    {
        let stroke_width = eval_scene_number(&ellipse.stroke_width, time_norm, time_sec)?.max(0.0)
            * affine_uniform_scale(shape_transform);
        if stroke_width > 0.0 {
            let (stroke, gradient) =
                resolve_gpu_scene_paint(stroke_value, gradient_defs, warped_bounds)?;
            push_gpu_stroke_segments(
                primitives,
                Affine2::identity(),
                &subpaths,
                stroke_width,
                stroke,
                opacity,
                gradient,
                (0.0, 1.0),
                StrokeStyle::default(),
                blend,
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_line_command(
    line: &LineNode,
    transform: Affine2,
    deform: Option<&EvaluatedDeformGrid>,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    gradient_defs: &HashMap<String, GradientDef>,
    primitives: &mut Vec<GpuScenePrimitive>,
) -> Result<(), MotionLoomSceneRenderError> {
    let opacity = (eval_scene_number(&line.opacity, time_norm, time_sec)? * inherited_opacity)
        .clamp(0.0, 1.0);
    if opacity <= 0.0001 {
        return Ok(());
    }

    let x1 = eval_scene_number(&line.x1, time_norm, time_sec)?;
    let y1 = eval_scene_number(&line.y1, time_norm, time_sec)?;
    let x2 = eval_scene_number(&line.x2, time_norm, time_sec)?;
    let y2 = eval_scene_number(&line.y2, time_norm, time_sec)?;
    let transform = transform.mul(scene_line_local_transform(line, time_norm, time_sec)?);
    let width = eval_scene_number(&line.width, time_norm, time_sec)?.max(0.0);
    if width <= 0.0001 {
        return Ok(());
    }
    let blend = parse_scene_blend(&line.blend)?;

    let p0 = transform_and_deform_point(transform, Point2::new(x1, y1), deform);
    let p1 = transform_and_deform_point(transform, Point2::new(x2, y2), deform);
    let (paint_bounds, primitive_transform, p0, p1, width) = if deform.is_some() {
        (
            PaintBounds::new(
                p0.x.min(p1.x),
                p0.y.min(p1.y),
                p0.x.max(p1.x),
                p0.y.max(p1.y),
            ),
            Affine2::identity(),
            p0,
            p1,
            width * affine_uniform_scale(transform),
        )
    } else {
        (
            PaintBounds::new(x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2)),
            transform,
            Point2::new(x1, y1),
            Point2::new(x2, y2),
            width,
        )
    };
    let (color, gradient) = resolve_gpu_scene_paint(&line.color, gradient_defs, paint_bounds)?;
    let style = eval_line_stroke_style(line, time_norm, time_sec)?;
    push_gpu_styled_line_primitives(
        primitives,
        primitive_transform,
        p0,
        p1,
        width,
        color,
        opacity,
        gradient,
        0.0,
        1.0,
        style,
        blend,
        0,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_polyline_commands(
    polyline: &PolylineNode,
    transform: Affine2,
    deform: Option<&EvaluatedDeformGrid>,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    gradient_defs: &HashMap<String, GradientDef>,
    primitives: &mut Vec<GpuScenePrimitive>,
) -> Result<(), MotionLoomSceneRenderError> {
    let opacity = (eval_scene_number(&polyline.opacity, time_norm, time_sec)? * inherited_opacity)
        .clamp(0.0, 1.0);
    if opacity <= 0.0001 {
        return Ok(());
    }
    let width = eval_scene_number(&polyline.stroke_width, time_norm, time_sec)?.max(0.0);
    if width <= 0.0001 {
        return Ok(());
    }
    let points = parse_polyline_points(&polyline.points)?;
    let transform = transform.mul(scene_polyline_local_transform(
        polyline, time_norm, time_sec,
    )?);
    let blend = parse_scene_blend(&polyline.blend)?;
    let (points, primitive_transform, width) = if let Some(deform) = deform {
        (
            points
                .iter()
                .map(|point| transform_and_deform_point(transform, *point, Some(deform)))
                .collect::<Vec<_>>(),
            Affine2::identity(),
            width * affine_uniform_scale(transform),
        )
    } else {
        (points, transform, width)
    };
    let trim = evaluate_trim(
        &polyline.trim_start,
        &polyline.trim_end,
        time_norm,
        time_sec,
    )?;
    let paint_bounds =
        points_bounds(&points).unwrap_or_else(|| PaintBounds::new(0.0, 0.0, 1.0, 1.0));
    let (color, gradient) = resolve_gpu_scene_paint(&polyline.stroke, gradient_defs, paint_bounds)?;
    let style = eval_polyline_stroke_style(polyline, time_norm, time_sec)?;
    push_gpu_stroke_segments(
        primitives,
        primitive_transform,
        &[points],
        width,
        color,
        opacity,
        gradient,
        trim,
        style,
        blend,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_path_commands(
    path: &PathNode,
    transform: Affine2,
    deform: Option<&EvaluatedDeformGrid>,
    inherited_opacity: f32,
    time_norm: f32,
    time_sec: f32,
    gradient_defs: &HashMap<String, GradientDef>,
    primitives: &mut Vec<GpuScenePrimitive>,
) -> Result<(), MotionLoomSceneRenderError> {
    let opacity = (eval_scene_number(&path.opacity, time_norm, time_sec)? * inherited_opacity)
        .clamp(0.0, 1.0);
    if opacity <= 0.0001 {
        return Ok(());
    }
    let transform = transform.mul(scene_path_local_transform(path, time_norm, time_sec)?);
    let blend = parse_scene_blend(&path.blend)?;
    let subpaths = evaluated_path_subpaths(path, time_norm, time_sec)?;
    let (subpaths, primitive_transform, stroke_width_scale) = if let Some(deform) = deform {
        (
            transform_and_deform_subpaths(&subpaths, transform, deform),
            Affine2::identity(),
            affine_uniform_scale(transform),
        )
    } else {
        (subpaths, transform, 1.0)
    };
    let paint_bounds =
        subpaths_bounds(&subpaths).unwrap_or_else(|| PaintBounds::new(0.0, 0.0, 1.0, 1.0));
    if let Some(fill) = path.fill.as_deref().filter(|fill| !is_none_paint(fill)) {
        let (color, gradient) = resolve_gpu_scene_paint(fill, gradient_defs, paint_bounds)?;
        push_gpu_tessellated_path_with_blend(
            primitives,
            primitive_transform,
            &subpaths,
            evaluated_path_fill_rule(path),
            GpuFilledPathStyle {
                color,
                opacity,
                gradient,
                blend,
                pick_id: 0,
            },
        );
    }

    let width =
        eval_scene_number(&path.stroke_width, time_norm, time_sec)?.max(0.0) * stroke_width_scale;
    if width > 0.0001 && !is_none_paint(&path.stroke) {
        let trim = evaluate_trim(&path.trim_start, &path.trim_end, time_norm, time_sec)?;
        let (color, gradient) = resolve_gpu_scene_paint(&path.stroke, gradient_defs, paint_bounds)?;
        let style = eval_path_stroke_style(path, time_norm, time_sec)?;
        push_gpu_stroke_segments(
            primitives,
            primitive_transform,
            &subpaths,
            width,
            color,
            opacity,
            gradient,
            trim,
            style,
            blend,
        );
    }
    Ok(())
}

fn push_gpu_tessellated_path_with_blend(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    subpaths: &[Vec<Point2>],
    fill_rule: FillRule,
    style: GpuFilledPathStyle,
) {
    let mut builder = LyonPath::builder();
    for subpath in subpaths {
        if subpath.len() < 3 {
            continue;
        }
        let closed = point_distance(subpath[0], *subpath.last().unwrap_or(&subpath[0])) <= 0.001;
        let count = if closed {
            subpath.len() - 1
        } else {
            subpath.len()
        };
        if count < 3 {
            continue;
        }
        builder.begin(point(subpath[0].x, subpath[0].y));
        for value in &subpath[1..count] {
            builder.line_to(point(value.x, value.y));
        }
        builder.end(closed);
    }
    let path = builder.build();
    let mut geometry: VertexBuffers<Point2, u32> = VertexBuffers::new();
    let lyon_rule = match fill_rule {
        FillRule::EvenOdd | FillRule::Subtract | FillRule::Xor => {
            lyon_tessellation::FillRule::EvenOdd
        }
        _ => lyon_tessellation::FillRule::NonZero,
    };
    let options = FillOptions::default().with_fill_rule(lyon_rule);
    if FillTessellator::new()
        .tessellate_path(
            &path,
            &options,
            &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex| {
                let position = vertex.position();
                Point2::new(position.x, position.y)
            }),
        )
        .is_err()
    {
        push_gpu_filled_path_triangles_with_blend(primitives, transform, subpaths, style);
        return;
    }
    for triangle in geometry.indices.chunks_exact(3) {
        let a = geometry.vertices[triangle[0] as usize];
        let b = geometry.vertices[triangle[1] as usize];
        let c = geometry.vertices[triangle[2] as usize];
        primitives.push(GpuScenePrimitive {
            kind: GPU_SHAPE_TRIANGLE_FILL,
            transform,
            shape: [a.x, a.y, b.x, b.y],
            radius: c.x,
            stroke_width: c.y,
            blur: 0.0,
            color: style.color,
            opacity: style.opacity,
            blend: style.blend,
            pick_id: style.pick_id,
            gradient: style.gradient.clone(),
            line_t0: 0.0,
            line_t1: 1.0,
            taper_start: 0.0,
            taper_end: 0.0,
        });
    }
}

fn push_gpu_filled_path_triangles(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    subpaths: &[Vec<Point2>],
    color: [u8; 4],
    opacity: f32,
    gradient: Option<GpuSceneGradientPaint>,
) {
    push_gpu_filled_path_triangles_with_blend(
        primitives,
        transform,
        subpaths,
        GpuFilledPathStyle {
            color,
            opacity,
            gradient,
            blend: SceneBlendMode::Normal,
            pick_id: 0,
        },
    );
}

struct GpuFilledPathStyle {
    color: [u8; 4],
    opacity: f32,
    gradient: Option<GpuSceneGradientPaint>,
    blend: SceneBlendMode,
    pick_id: u32,
}

fn push_gpu_filled_path_triangles_with_blend(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    subpaths: &[Vec<Point2>],
    style: GpuFilledPathStyle,
) {
    for subpath in subpaths {
        for [a, b, c] in triangulate_polygon(subpath) {
            primitives.push(GpuScenePrimitive {
                kind: GPU_SHAPE_TRIANGLE_FILL,
                transform,
                shape: [a.x, a.y, b.x, b.y],
                radius: c.x,
                stroke_width: c.y,
                blur: 0.0,
                color: style.color,
                opacity: style.opacity,
                blend: style.blend,
                pick_id: style.pick_id,
                gradient: style.gradient.clone(),
                line_t0: 0.0,
                line_t1: 1.0,
                taper_start: 0.0,
                taper_end: 0.0,
            });
        }
    }
}

pub(crate) fn triangulate_polygon(points: &[Point2]) -> Vec<[Point2; 3]> {
    let polygon = sanitize_polygon(points);
    if polygon.len() < 3 {
        return Vec::new();
    }
    if polygon_area(&polygon).abs() <= 0.0001 {
        return Vec::new();
    }

    let ccw = polygon_area(&polygon) > 0.0;
    let mut indices: Vec<usize> = (0..polygon.len()).collect();
    let mut triangles = Vec::with_capacity(polygon.len().saturating_sub(2));
    let mut guard = 0usize;

    while indices.len() > 3 && guard < polygon.len().saturating_mul(polygon.len()).max(16) {
        guard += 1;
        let len = indices.len();
        let mut clipped = false;
        for i in 0..len {
            let prev = indices[(i + len - 1) % len];
            let curr = indices[i];
            let next = indices[(i + 1) % len];
            let a = polygon[prev];
            let b = polygon[curr];
            let c = polygon[next];
            if !is_convex_corner(a, b, c, ccw) {
                continue;
            }
            let contains_other = indices.iter().any(|&candidate| {
                candidate != prev
                    && candidate != curr
                    && candidate != next
                    && point_in_triangle(polygon[candidate], a, b, c)
            });
            if contains_other {
                continue;
            }
            triangles.push([a, b, c]);
            indices.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            return triangulate_polygon_fan(&polygon);
        }
    }

    if indices.len() == 3 {
        triangles.push([
            polygon[indices[0]],
            polygon[indices[1]],
            polygon[indices[2]],
        ]);
    }
    triangles
}

fn sanitize_polygon(points: &[Point2]) -> Vec<Point2> {
    let mut out = Vec::with_capacity(points.len());
    for &point in points {
        if out
            .last()
            .is_some_and(|last: &Point2| points_close(*last, point))
        {
            continue;
        }
        out.push(point);
    }
    if out.len() > 1
        && out
            .last()
            .zip(out.first())
            .is_some_and(|(last, first)| points_close(*last, *first))
    {
        out.pop();
    }
    out
}

fn triangulate_polygon_fan(points: &[Point2]) -> Vec<[Point2; 3]> {
    if points.len() < 3 {
        return Vec::new();
    }
    let mut triangles = Vec::with_capacity(points.len().saturating_sub(2));
    let a = points[0];
    for i in 1..points.len() - 1 {
        triangles.push([a, points[i], points[i + 1]]);
    }
    triangles
}

fn points_close(a: Point2, b: Point2) -> bool {
    (a.x - b.x).abs() <= 0.001 && (a.y - b.y).abs() <= 0.001
}

fn polygon_area(points: &[Point2]) -> f32 {
    let mut area = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        area += a.x * b.y - b.x * a.y;
    }
    area * 0.5
}

fn is_convex_corner(a: Point2, b: Point2, c: Point2, ccw: bool) -> bool {
    let cross = cross_points(a, b, c);
    if ccw { cross > 0.001 } else { cross < -0.001 }
}

fn point_in_triangle(p: Point2, a: Point2, b: Point2, c: Point2) -> bool {
    let c0 = cross_points(a, b, p);
    let c1 = cross_points(b, c, p);
    let c2 = cross_points(c, a, p);
    let has_neg = c0 < -0.001 || c1 < -0.001 || c2 < -0.001;
    let has_pos = c0 > 0.001 || c1 > 0.001 || c2 > 0.001;
    !(has_neg && has_pos)
}

fn cross_points(a: Point2, b: Point2, c: Point2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_styled_line_primitives(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    p0: Point2,
    p1: Point2,
    width: f32,
    color: [u8; 4],
    opacity: f32,
    gradient: Option<GpuSceneGradientPaint>,
    line_t0: f32,
    line_t1: f32,
    style: StrokeStyle,
    blend: SceneBlendMode,
    pick_id: u32,
) {
    let copies = stroke_texture_copy_count(style);
    for copy_ix in 0..copies {
        let (start, end, width_scale, opacity_scale) =
            stroke_texture_variant(p0, p1, style, copy_ix);
        let stroke_width = (width * width_scale).max(0.01);
        let stroke_opacity = (opacity * opacity_scale).clamp(0.0, 1.0);
        if style.pressure_auto
            || style.width_profile_len >= 2
            || (style.width_start - 1.0).abs() > 0.0001
            || (style.width_end - 1.0).abs() > 0.0001
        {
            push_gpu_pressure_line_primitives(
                primitives,
                transform,
                start,
                end,
                stroke_width,
                color,
                stroke_opacity,
                gradient.clone(),
                line_t0,
                line_t1,
                style,
                blend,
                pick_id,
            );
        } else {
            push_gpu_line_primitive(
                primitives,
                transform,
                start,
                end,
                stroke_width,
                color,
                stroke_opacity,
                gradient.clone(),
                line_t0,
                line_t1,
                style.taper_start,
                style.taper_end,
                blend,
                pick_id,
            );
        }
    }
    push_gpu_stroke_overlay_primitives(
        primitives, transform, p0, p1, width, color, opacity, line_t0, line_t1, style, blend,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_line_primitive(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    p0: Point2,
    p1: Point2,
    width: f32,
    color: [u8; 4],
    opacity: f32,
    gradient: Option<GpuSceneGradientPaint>,
    line_t0: f32,
    line_t1: f32,
    taper_start: f32,
    taper_end: f32,
    blend: SceneBlendMode,
    pick_id: u32,
) {
    primitives.push(GpuScenePrimitive {
        kind: GPU_SHAPE_LINE,
        transform,
        shape: [p0.x, p0.y, p1.x, p1.y],
        radius: 0.0,
        stroke_width: width.max(0.01),
        blur: 0.0,
        color,
        opacity: opacity.clamp(0.0, 1.0),
        blend,
        pick_id,
        gradient,
        line_t0,
        line_t1,
        taper_start,
        taper_end,
    });
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_pressure_line_primitives(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    p0: Point2,
    p1: Point2,
    width: f32,
    color: [u8; 4],
    opacity: f32,
    gradient: Option<GpuSceneGradientPaint>,
    line_t0: f32,
    line_t1: f32,
    style: StrokeStyle,
    blend: SceneBlendMode,
    pick_id: u32,
) {
    let len = point_distance(p0, p1);
    if len <= 0.0001 {
        return;
    }
    let steps = (len / (width.max(1.0) * 4.0)).ceil().clamp(1.0, 8.0) as u32;
    for ix in 0..steps {
        let a_t = ix as f32 / steps as f32;
        let b_t = (ix + 1) as f32 / steps as f32;
        let mid_t = (a_t + b_t) * 0.5;
        let global_t = line_t0 + (line_t1 - line_t0) * mid_t;
        let pressure = stroke_taper_pressure(global_t, style).max(0.05);
        push_gpu_line_primitive(
            primitives,
            transform,
            p0.lerp(p1, a_t),
            p0.lerp(p1, b_t),
            width * pressure,
            color,
            opacity,
            gradient.clone(),
            0.0,
            1.0,
            0.0,
            0.0,
            blend,
            pick_id,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_stroke_overlay_primitives(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    p0: Point2,
    p1: Point2,
    width: f32,
    color: [u8; 4],
    opacity: f32,
    line_t0: f32,
    line_t1: f32,
    style: StrokeStyle,
    blend: SceneBlendMode,
) {
    if width <= 0.0 || opacity <= 0.0 {
        return;
    }
    if style.texture_strength > 0.001 {
        push_gpu_stroke_stamp_primitives(
            primitives, transform, p0, p1, width, color, opacity, line_t0, line_t1, style, blend,
        );
    }
    if style.bristles > 0 {
        push_gpu_stroke_bristle_primitives(
            primitives, transform, p0, p1, width, color, opacity, line_t0, line_t1, style, blend, 0,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_stroke_stamp_primitives(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    p0: Point2,
    p1: Point2,
    width: f32,
    color: [u8; 4],
    opacity: f32,
    line_t0: f32,
    line_t1: f32,
    style: StrokeStyle,
    blend: SceneBlendMode,
) {
    let len = point_distance(p0, p1);
    if len <= 0.0001 {
        return;
    }
    let strength = style.texture_strength.clamp(0.0, 1.0);
    let dx = (p1.x - p0.x) / len;
    let dy = (p1.y - p0.y) / len;
    let nx = -dy;
    let ny = dx;
    let spacing = (width * (1.35 - strength * 0.65)).clamp(2.0, 18.0);
    let steps = (len / spacing).ceil().clamp(1.0, 72.0) as u32;
    let texture_size = match style.texture {
        StrokeTexture::Charcoal => 1.65,
        StrokeTexture::Rough => 1.25,
        StrokeTexture::Pencil => 1.0,
        StrokeTexture::Sketch => 0.82,
        StrokeTexture::Marker => 0.72,
        StrokeTexture::Ink => 0.55,
        StrokeTexture::Hairline | StrokeTexture::Solid => 0.46,
    };
    let alpha_scale = match style.texture {
        StrokeTexture::Charcoal => 0.18,
        StrokeTexture::Pencil => 0.16,
        StrokeTexture::Rough => 0.14,
        StrokeTexture::Sketch => 0.12,
        StrokeTexture::Marker => 0.10,
        StrokeTexture::Ink => 0.08,
        StrokeTexture::Hairline | StrokeTexture::Solid => 0.06,
    };
    for step in 0..steps {
        let seed = stroke_texture_seed(p0, p1, step + 271);
        let keep = ((stroke_hash_signed(seed + 13.1) + 1.0) * 0.5).clamp(0.0, 1.0);
        if keep > strength {
            continue;
        }
        let local_t = ((step as f32 + 0.5) / steps as f32).clamp(0.0, 1.0);
        let global_t = line_t0 + (line_t1 - line_t0) * local_t;
        let pressure = stroke_taper_pressure(global_t, style).max(0.05);
        let tangent_noise = stroke_hash_signed(seed + 37.7) * spacing * 0.25;
        let normal_noise = stroke_hash_signed(seed + 91.3) * width * pressure * 0.45;
        let p = p0.lerp(p1, local_t);
        let size_noise = ((stroke_hash_signed(seed + 163.0) + 1.0) * 0.5).clamp(0.0, 1.0);
        let radius = (width * pressure * (0.035 + size_noise * 0.10) * texture_size).max(0.35);
        primitives.push(GpuScenePrimitive {
            kind: GPU_SHAPE_CIRCLE_FILL,
            transform,
            shape: [
                p.x + dx * tangent_noise + nx * normal_noise,
                p.y + dy * tangent_noise + ny * normal_noise,
                radius,
                0.0,
            ],
            radius: 0.0,
            stroke_width: 0.0,
            blur: 0.0,
            color,
            opacity: (opacity * strength * alpha_scale).clamp(0.0, 1.0),
            blend,
            pick_id: 0,
            gradient: None,
            line_t0: 0.0,
            line_t1: 1.0,
            taper_start: 0.0,
            taper_end: 0.0,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_stroke_bristle_primitives(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    p0: Point2,
    p1: Point2,
    width: f32,
    color: [u8; 4],
    opacity: f32,
    line_t0: f32,
    line_t1: f32,
    style: StrokeStyle,
    blend: SceneBlendMode,
    pick_id: u32,
) {
    let len = point_distance(p0, p1);
    if len <= 0.0001 {
        return;
    }
    let dx = (p1.x - p0.x) / len;
    let dy = (p1.y - p0.y) / len;
    let nx = -dy;
    let ny = dx;
    let count = style.bristles.clamp(0, 24);
    let pressure =
        ((stroke_taper_pressure(line_t0, style) + stroke_taper_pressure(line_t1, style)) * 0.5)
            .max(0.05);
    let bristle_width = (width * 0.08 * pressure).clamp(0.25, 2.2);
    let alpha_scale = match style.texture {
        StrokeTexture::Charcoal => 0.20,
        StrokeTexture::Rough => 0.18,
        StrokeTexture::Pencil => 0.15,
        StrokeTexture::Sketch => 0.13,
        _ => 0.11,
    };
    for ix in 0..count {
        let lane = if count <= 1 {
            0.0
        } else {
            ix as f32 / (count - 1) as f32 * 2.0 - 1.0
        };
        let seed = stroke_texture_seed(p0, p1, ix + 997);
        let offset = lane * width * pressure * 0.42
            + stroke_hash_signed(seed + 21.0) * style.roughness * 0.55;
        let start_t = (stroke_hash_signed(seed + 57.0) * 0.04).max(0.0);
        let end_t = 1.0 - (stroke_hash_signed(seed + 83.0) * 0.04).max(0.0);
        let start = p0.lerp(p1, start_t);
        let end = p0.lerp(p1, end_t);
        push_gpu_line_primitive(
            primitives,
            transform,
            Point2::new(start.x + nx * offset, start.y + ny * offset),
            Point2::new(end.x + nx * offset, end.y + ny * offset),
            bristle_width,
            color,
            (opacity * alpha_scale).clamp(0.0, 1.0),
            None,
            0.0,
            1.0,
            0.0,
            0.0,
            blend,
            pick_id,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_gpu_stroke_segments(
    primitives: &mut Vec<GpuScenePrimitive>,
    transform: Affine2,
    subpaths: &[Vec<Point2>],
    width: f32,
    color: [u8; 4],
    opacity: f32,
    gradient: Option<GpuSceneGradientPaint>,
    trim: (f32, f32),
    style: StrokeStyle,
    blend: SceneBlendMode,
) {
    for segment in trimmed_polyline_segments_with_progress(subpaths, trim) {
        push_gpu_styled_line_primitives(
            primitives,
            transform,
            segment.p0,
            segment.p1,
            width,
            color,
            opacity,
            gradient.clone(),
            segment.t0,
            segment.t1,
            style,
            blend,
            0,
        );
    }
}

fn normalize_process_mask_ref(raw: &str) -> String {
    let trimmed = raw.trim().trim_matches('"').trim();
    if let Some(inner) = trimmed
        .strip_prefix("url(#")
        .and_then(|value| value.strip_suffix(')'))
    {
        inner.to_string()
    } else {
        trimmed.to_string()
    }
}

fn normalize_scene_follow_ref(raw: &str) -> String {
    let trimmed = raw.trim().trim_matches('"').trim();
    let trimmed = trimmed
        .strip_prefix("node:")
        .or_else(|| trimmed.strip_prefix("target:"))
        .unwrap_or(trimmed);
    if let Some(inner) = trimmed
        .strip_prefix("url(#")
        .and_then(|value| value.strip_suffix(')'))
    {
        inner.to_string()
    } else {
        trimmed.trim_start_matches('#').to_string()
    }
}

fn eval_scene_bool_like(
    raw: &str,
    time_norm: f32,
    time_sec: f32,
) -> Result<bool, MotionLoomSceneRenderError> {
    let normalized = raw.trim().trim_matches('"').trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "false" | "no" | "off" | "0" => Ok(false),
        "true" | "yes" | "on" | "1" => Ok(true),
        _ => Ok(eval_scene_number(raw, time_norm, time_sec)?.abs() > 0.0001),
    }
}

fn apply_luma_process_mask_alpha(layer: &mut RgbaImage, mask: &RgbaImage, invert: bool) {
    let w = layer.width().min(mask.width());
    let h = layer.height().min(mask.height());
    for y in 0..h {
        for x in 0..w {
            let mask_pixel = mask.get_pixel(x, y);
            let luma = (0.2126 * mask_pixel[0] as f32
                + 0.7152 * mask_pixel[1] as f32
                + 0.0722 * mask_pixel[2] as f32)
                / 255.0;
            let alpha = mask_pixel[3] as f32 / 255.0;
            let mut mask_amount = luma * alpha;
            if invert {
                mask_amount = 1.0 - mask_amount;
            }
            let pixel = layer.get_pixel_mut(x, y);
            pixel[3] = ((pixel[3] as f32) * mask_amount).round().clamp(0.0, 255.0) as u8;
        }
    }
}

fn apply_luma_process_mask_mix(
    effected: &mut RgbaImage,
    original: &RgbaImage,
    mask: &RgbaImage,
    invert: bool,
) {
    let w = effected.width().min(original.width()).min(mask.width());
    let h = effected.height().min(original.height()).min(mask.height());
    for y in 0..h {
        for x in 0..w {
            let mask_pixel = mask.get_pixel(x, y);
            let luma = (0.2126 * mask_pixel[0] as f32
                + 0.7152 * mask_pixel[1] as f32
                + 0.0722 * mask_pixel[2] as f32)
                / 255.0;
            let alpha = mask_pixel[3] as f32 / 255.0;
            let mut mask_amount = luma * alpha;
            if invert {
                mask_amount = 1.0 - mask_amount;
            }
            mix_process_pixel(
                effected.get_pixel_mut(x, y),
                original.get_pixel(x, y),
                mask_amount,
            );
        }
    }
}

fn apply_alpha_process_mask_mix(
    effected: &mut RgbaImage,
    original: &RgbaImage,
    mask: &RgbaImage,
    invert: bool,
) {
    let w = effected.width().min(original.width()).min(mask.width());
    let h = effected.height().min(original.height()).min(mask.height());
    for y in 0..h {
        for x in 0..w {
            let mask_pixel = mask.get_pixel(x, y);
            let mut mask_amount = mask_pixel[3] as f32 / 255.0;
            if invert {
                mask_amount = 1.0 - mask_amount;
            }
            mix_process_pixel(
                effected.get_pixel_mut(x, y),
                original.get_pixel(x, y),
                mask_amount,
            );
        }
    }
}

fn mix_process_pixel(effected: &mut Rgba<u8>, original: &Rgba<u8>, mask_amount: f32) {
    let t = mask_amount.clamp(0.0, 1.0);
    for channel in 0..4 {
        effected[channel] = (original[channel] as f32 * (1.0 - t) + effected[channel] as f32 * t)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

fn scene_bool(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no" | "off"
    )
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use crate::parse_graph_script;
    use crate::scene::drawable::{
        GpuSceneNativeAssets, GpuScenePrimitive, GpuSceneTextureLayer, GpuSceneTextureSource,
    };
    use crate::scene::model::{PathNode, SceneNode};
    use crate::scene::spatial::Affine2;
    use crate::scene::text::TextNode;
    use base64::Engine;
    use cosmic_text::Weight;
    use image::{Rgba, RgbaImage};
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    use super::{
        MotionLoomSceneRenderError, SceneFrameRenderer, ScenePlatformPreviewSurface,
        ScenePreviewBackend, ScenePreviewPixelFormat, ScenePreviewSurface,
        ScenePreviewSurfaceOptions, SceneRenderError, SceneRenderProfile, SceneRenderer,
        apply_animation_targets_at_frame, eval_text_box_padding, eval_text_font_weight,
        eval_text_tracking_em, graph_logical_render_size, graph_output_size,
        render_size_root_transform, scene_nodes_for_present, validate_scene_graph,
    };

    fn max_rgb(image: &image::RgbaImage) -> u8 {
        image
            .pixels()
            .map(|pixel| pixel[0].max(pixel[1]).max(pixel[2]))
            .max()
            .unwrap_or(0)
    }

    fn max_green(image: &image::RgbaImage) -> u8 {
        image.pixels().map(|pixel| pixel[1]).max().unwrap_or(0)
    }

    fn skip_gpu_texture_test(message: impl std::fmt::Display) {
        eprintln!("Skipping GPU texture render test: {message}");
    }

    fn is_gpu_adapter_unavailable(message: &str) -> bool {
        message.contains("No suitable graphics adapter found")
            || message.contains("no compatible GPU adapter")
            || message.contains("No compatible GPU adapter")
    }

    fn basic_text_node(render_scale: &str) -> TextNode {
        TextNode {
            id: None,
            value: "Text".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            rotation: "0".to_string(),
            scale: "1".to_string(),
            scale_x: "1".to_string(),
            scale_y: "1".to_string(),
            skew_x: "0".to_string(),
            skew_y: "0".to_string(),
            transform_origin_x: "0".to_string(),
            transform_origin_y: "0".to_string(),
            width: None,
            max_width: None,
            align: None,
            tracking: None,
            font_size: "32".to_string(),
            render_scale: render_scale.to_string(),
            antialias: None,
            edge_smoothing: None,
            blur: None,
            soft_edge: None,
            line_height: None,
            color: "#111111".to_string(),
            opacity: "1".to_string(),
            box_style: None,
            box_color: None,
            box_padding: None,
            box_padding_x: None,
            box_padding_y: None,
            box_radius: None,
            stroke: None,
            stroke_width: None,
            stroke_join: None,
            stroke_position: None,
            visible_chars: None,
            max_lines: None,
            font: None,
            font_family: None,
            font_weight: None,
            font_path: None,
            layout: None,
            animators: Vec::new(),
        }
    }

    #[test]
    fn scene_validation_rejects_missing_gradient_refs_before_render() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#000000" />
  <Rect x="0" y="0" width="64" height="48" fill="url(#bg_glow)" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let err = validate_scene_graph(&graph).expect_err("missing gradient ref should fail");
        assert!(
            err.to_string()
                .contains("gradient reference not found: bg_glow"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn text_font_weight_accepts_keywords_and_numeric_values() {
        assert_eq!(
            eval_text_font_weight(Some("bold"), 0.0, 0.0).expect("bold weight"),
            Weight::BOLD
        );
        assert_eq!(
            eval_text_font_weight(Some("semi-bold"), 0.0, 0.0).expect("semibold weight"),
            Weight::SEMIBOLD
        );
        assert_eq!(
            eval_text_font_weight(Some("900"), 0.0, 0.0).expect("numeric weight"),
            Weight::BLACK
        );
        assert_eq!(
            eval_text_font_weight(Some("400 + 300*$time.norm"), 1.0, 0.0)
                .expect("expression weight"),
            Weight::BOLD
        );
    }

    #[test]
    fn text_box_padding_accepts_single_and_pair_values() {
        assert_eq!(
            eval_text_box_padding("54", 0.0, 0.0).expect("single padding"),
            (54.0, 54.0)
        );
        assert_eq!(
            eval_text_box_padding("54 28", 0.0, 0.0).expect("space pair padding"),
            (54.0, 28.0)
        );
        assert_eq!(
            eval_text_box_padding("54,28", 0.0, 0.0).expect("comma pair padding"),
            (54.0, 28.0)
        );
    }

    #[test]
    fn text_gap_tracking_converts_pixels_to_em() {
        assert_eq!(
            eval_text_tracking_em(Some("-2"), 40.0, 0.0, 0.0).expect("negative tracking"),
            -0.05
        );
        assert_eq!(
            eval_text_tracking_em(Some("4"), 40.0, 0.0, 0.0).expect("positive tracking"),
            0.1
        );
    }

    #[test]
    fn gpu_text_raster_cache_reuses_4x_texture_across_parent_transform() {
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        if pollster::block_on(renderer.ensure_gpu_compositor_size(800, 450)).is_err() {
            return;
        }
        renderer.prepared_graph_signature = 42;
        let mut text = basic_text_node("retained 4x title");
        text.render_scale = "4x".to_string();
        text.opacity = "curve(\"0:0.5:linear, 1:1:linear\")".to_string();

        let first = renderer
            .rasterize_text_texture_layers_gpu_effects(
                &text,
                super::TextTextureRasterContext {
                    transform: Affine2::translate(10.0, 20.0),
                    inherited_opacity: 0.5,
                    pick_id: 7,
                    time_norm: 0.0,
                    time_sec: 0.0,
                    canvas_size: (800, 450),
                },
            )
            .expect("first cached text raster");
        let second = renderer
            .rasterize_text_texture_layers_gpu_effects(
                &text,
                super::TextTextureRasterContext {
                    transform: Affine2::translate(110.0, 70.0),
                    inherited_opacity: 0.8,
                    pick_id: 7,
                    time_norm: 0.5,
                    time_sec: 1.0,
                    canvas_size: (800, 450),
                },
            )
            .expect("reuse cached text raster");

        assert_eq!(renderer.gpu_text_raster_cache.len(), 1);
        let (first_texture, second_texture) = match (&first[0].source, &second[0].source) {
            (GpuSceneTextureSource::Gpu(first), GpuSceneTextureSource::Gpu(second)) => {
                (&first.texture, &second.texture)
            }
            _ => panic!("retained text should use a persistent GPU texture"),
        };
        assert!(std::sync::Arc::ptr_eq(first_texture, second_texture));
        assert_ne!(first[0].transform.m02, second[0].transform.m02);
        assert_eq!(first[0].opacity, 0.25);
        assert_eq!(second[0].opacity, 0.8);
    }

    #[test]
    fn gpu_path_geometry_cache_reuses_transform_and_invalidates_changed_morph() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="2s" size={[128,128]}>
  <Scene id="cache_test">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence from="0s" duration="2s" out="hold">
          <Layer>
            <Group id="moving" x={curve("0:0:linear, 2:40:linear")}>
              <Path id="static_path" d="M 0 0 L 20 0 L 10 20 Z" fill="#ff0000" />
            </Group>
            <Path id="morph_path"
                  d={morph("0:M 40 0 L 60 0 L 50 20 Z", "2:M 40 0 L 70 0 L 55 30 Z")}
                  fill="#00ff00" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="cache_test" />
</Graph>
"##,
        )
        .expect("cache test graph");
        let nodes = scene_nodes_for_present(&graph).expect("presentable scene");
        assert!(
            !super::vector_overlay_nodes_are_static(nodes),
            "animated curve/morph scene was classified static: {nodes:#?}"
        );
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        renderer.prepare_frame_caches(&graph);

        let collect = |renderer: &mut SceneFrameRenderer,
                       time_norm: f32,
                       time_sec: f32|
         -> Vec<GpuScenePrimitive> {
            let mut assets = GpuSceneNativeAssets::default();
            let mut primitives = Vec::new();
            let mut texture_layers = Vec::new();
            let mut text_requests = Vec::new();
            let mut unsupported = false;
            pollster::block_on(renderer.collect_gpu_scene_native_commands(
                nodes,
                Affine2::identity(),
                None,
                1.0,
                time_norm,
                time_sec,
                (128, 128),
                &mut assets,
                &mut primitives,
                &mut texture_layers,
                &mut text_requests,
                &mut unsupported,
            ))
            .expect("collect cached Path commands");
            assert!(!unsupported);
            assert!(texture_layers.is_empty());
            assert!(text_requests.is_empty());
            primitives
        };

        let first = collect(&mut renderer, 0.0, 0.0);
        assert_eq!(renderer.gpu_path_cache_hits, 0);
        assert_eq!(renderer.gpu_path_cache_misses, 2);
        renderer.gpu_path_cache_hits = 0;
        renderer.gpu_path_cache_misses = 0;
        let second = collect(&mut renderer, 0.5, 1.0);
        assert_eq!(renderer.gpu_path_cache_hits, 1, "static Path should hit");
        assert_eq!(
            renderer.gpu_path_cache_misses, 1,
            "only the changed Morph Path should miss"
        );
        assert_ne!(
            first[0].transform.m02, second[0].transform.m02,
            "cached local geometry must receive the current Group transform"
        );
    }

    #[test]
    fn retained_gpu_shape_scene_skips_static_path_collection_after_first_frame() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="2s" size={[128,128]}>
  <Scene id="retained_test">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence from="0s" duration="2s" out="hold">
          <Layer>
            <Path id="a" d="M 0 0 L 20 0 L 10 20 Z" fill="#ff0000" />
            <Path id="b" x="40" d="M 0 0 L 20 0 L 10 20 Z" fill="#00ff00" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="retained_test" />
</Graph>
"##,
        )
        .expect("retained scene graph");
        let nodes = scene_nodes_for_present(&graph).expect("presentable scene");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        renderer.prepare_frame_caches(&graph);

        let first = pollster::block_on(renderer.try_collect_retained_gpu_shape_scene(
            nodes,
            Affine2::identity(),
            1.0,
            0.0,
            0.0,
            (128, 128),
        ))
        .expect("compile retained scene")
        .expect("static scene is retainable");
        assert_eq!(renderer.gpu_path_cache_misses, 2);

        renderer.gpu_path_cache_hits = 0;
        renderer.gpu_path_cache_misses = 0;
        let second = pollster::block_on(renderer.try_collect_retained_gpu_shape_scene(
            nodes,
            Affine2::identity(),
            1.0,
            0.5,
            1.0,
            (128, 128),
        ))
        .expect("reuse retained scene")
        .expect("static scene remains retainable");

        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert_eq!(renderer.gpu_path_cache_hits, 2);
        assert_eq!(renderer.gpu_path_cache_misses, 0);
    }

    #[test]
    fn retained_gpu_transform_scene_updates_group_matrix_without_reflattening() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="2s" size={[128,128]}>
  <Scene id="retained_transform_test">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence from="0s" duration="2s" out="hold">
          <Layer>
            <Group id="moving" x={curve("0:0:linear, 2:40:linear")}>
              <Path id="a" d="M 0 0 L 20 0 L 10 20 Z" fill="#ff0000" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="retained_transform_test" />
</Graph>
"##,
        )
        .expect("retained transform graph");
        let nodes = scene_nodes_for_present(&graph).expect("presentable scene");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        renderer.prepare_frame_caches(&graph);
        let first = renderer
            .try_collect_retained_gpu_transform_scene(
                nodes,
                Affine2::identity(),
                0.0,
                0.0,
                (128, 128),
            )
            .expect("compile retained transform scene")
            .unwrap_or_else(|| panic!("transform scene rejected: {nodes:#?}"));
        let first_x = first.primitives[0].transform.m02;
        drop(first);
        renderer.gpu_path_cache_hits = 0;
        renderer.gpu_path_cache_misses = 0;
        let second = renderer
            .try_collect_retained_gpu_transform_scene(
                nodes,
                Affine2::identity(),
                0.5,
                1.0,
                (128, 128),
            )
            .expect("update retained transform scene")
            .expect("transform scene remains retained");
        assert_ne!(first_x, second.primitives[0].transform.m02);
        assert_eq!(renderer.gpu_path_cache_hits, 1);
        assert_eq!(renderer.gpu_path_cache_misses, 0);
    }

    #[test]
    fn retained_gpu_transform_scene_updates_animation_target_between_frames() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="2s" size={[128,128]}>
  <Scene id="animation_target_retained_test">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence from="0s" duration="2s" out="hold">
          <Layer>
            <Group id="moving" x="0">
              <Path id="shape" d="M 0 0 L 20 0 L 10 20 Z" fill="#ff0000" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <AnimationTarget node="moving" property="x">
    <Key frame="0" value="0" ease="linear" />
    <Key frame="30" value="40" ease="linear" />
  </AnimationTarget>
  <Present from="animation_target_retained_test" />
</Graph>
"##,
        )
        .expect("AnimationTarget retained graph");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());

        let first_graph = apply_animation_targets_at_frame(&graph, 0)
            .expect("frame 0 AnimationTarget")
            .expect("animated frame 0 graph");
        renderer.prepare_frame_caches(&first_graph);
        let first_nodes = scene_nodes_for_present(&first_graph).expect("frame 0 scene");
        let first = renderer
            .try_collect_retained_gpu_transform_scene(
                first_nodes,
                Affine2::identity(),
                0.0,
                0.0,
                (128, 128),
            )
            .expect("frame 0 retained scene")
            .expect("frame 0 scene should be retained");
        let first_x = first.primitives[0].transform.m02;
        drop(first);

        let later_graph = apply_animation_targets_at_frame(&graph, 30)
            .expect("frame 30 AnimationTarget")
            .expect("animated frame 30 graph");
        renderer.prepare_frame_caches(&later_graph);
        let later_nodes = scene_nodes_for_present(&later_graph).expect("frame 30 scene");
        let later = renderer
            .try_collect_retained_gpu_transform_scene(
                later_nodes,
                Affine2::identity(),
                0.5,
                1.0,
                (128, 128),
            )
            .expect("frame 30 retained scene")
            .expect("frame 30 scene should remain retained");
        let later_x = later.primitives[0].transform.m02;

        assert!(
            (first_x - 0.0).abs() < 0.001,
            "unexpected frame 0 x: {first_x}"
        );
        assert!(
            (later_x - 40.0).abs() < 0.001,
            "unexpected frame 30 x: {later_x}"
        );
    }

    #[test]
    fn retained_mixed_scene_instances_repeat_and_quantizes_random_updates() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="2s" size={[128,128]}>
  <Scene id="mixed_retained_test">
    <Timeline>
      <Track id="main" space="world" z="0">
        <Sequence from="0s" duration="2s" out="hold">
          <Layer>
            <Rect x="0" y="0" width="128" height="128" color="#222222" />
            <Group x={curve("0:0:linear, 2:20:linear")}>
              <Circle x="32" y="32" radius={curve("0:8:linear, 2:12:linear")} color="#ff0000" />
              <Text value="cached" x="10" y="20" renderScale="4x"
                    opacity={curve("0:0.2:linear, 2:1:linear")} />
            </Group>
            <Repeat count="2" xStep="40">
              <Circle x="random(-4,4,floor($time.sec*3)+17)" y="80" radius="3" color="#ffffff" />
            </Repeat>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="mixed_retained_test" />
</Graph>
"##,
        )
        .expect("mixed retained graph");
        let nodes = scene_nodes_for_present(&graph).expect("presentable scene");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        renderer.prepare_frame_caches(&graph);
        let first = renderer
            .try_collect_retained_gpu_transform_scene(
                nodes,
                Affine2::identity(),
                0.05,
                0.1,
                (128, 128),
            )
            .expect("compile mixed retained scene")
            .expect("mixed primitives should remain retained");
        assert_eq!(first.text_requests.len(), 1);
        assert!(first.primitives.len() >= 4);
        let first_repeat_x = first.primitives[first.primitives.len() - 2].shape[0];
        drop(first);

        let same_tick = renderer
            .try_collect_retained_gpu_transform_scene(
                nodes,
                Affine2::identity(),
                0.1,
                0.2,
                (128, 128),
            )
            .expect("same dependency tick")
            .expect("scene remains retained");
        assert_eq!(
            first_repeat_x,
            same_tick.primitives[same_tick.primitives.len() - 2].shape[0]
        );
        drop(same_tick);

        let next_tick = renderer
            .try_collect_retained_gpu_transform_scene(
                nodes,
                Affine2::identity(),
                0.2,
                0.4,
                (128, 128),
            )
            .expect("next dependency tick")
            .expect("scene remains retained after Repeat update");
        assert_ne!(
            first_repeat_x,
            next_tick.primitives[next_tick.primitives.len() - 2].shape[0]
        );
    }

    fn cpu_texture_size(layer: GpuSceneTextureLayer) -> (u32, u32) {
        match layer.source {
            GpuSceneTextureSource::Cpu(image) => (image.width(), image.height()),
            GpuSceneTextureSource::Gpu(_) => panic!("expected CPU text texture layer"),
        }
    }

    #[test]
    fn scene_text_render_scale_supersamples_raster_texture() {
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let base = renderer
            .rasterize_text_texture_layer(
                &basic_text_node("1x"),
                Affine2::identity(),
                1.0,
                0.0,
                0.0,
                (800, 450),
            )
            .expect("base text raster")
            .expect("base text layer");
        let high = renderer
            .rasterize_text_texture_layer(
                &basic_text_node("4x"),
                Affine2::identity(),
                1.0,
                0.0,
                0.0,
                (800, 450),
            )
            .expect("4x text raster")
            .expect("4x text layer");

        let (base_w, base_h) = cpu_texture_size(base);
        let (high_w, high_h) = cpu_texture_size(high);
        assert!(
            high_w >= base_w * 3 && high_h >= base_h * 3,
            "expected renderScale=4x to allocate a much larger raster texture, base={base_w}x{base_h}, high={high_w}x{high_h}"
        );
    }

    #[test]
    fn scene_text_pill_box_expands_raster_texture_and_draws_background() {
        let mut text = basic_text_node("1x");
        text.value = "WE APPLY THEM".to_string();
        text.font_weight = Some("900".to_string());
        text.box_style = Some("pill".to_string());
        text.box_color = Some("#D9251D".to_string());
        text.box_padding = Some("54 28".to_string());
        text.box_radius = Some("999".to_string());

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let layer = renderer
            .rasterize_text_texture_layer(&text, Affine2::identity(), 1.0, 0.0, 0.0, (800, 450))
            .expect("pill text raster")
            .expect("pill text layer");
        let GpuSceneTextureSource::Cpu(image) = layer.source else {
            panic!("expected CPU text texture layer");
        };

        let red_pixels = image
            .pixels()
            .filter(|pixel| pixel[0] > 180 && pixel[1] < 80 && pixel[2] < 70 && pixel[3] > 180)
            .count();
        assert!(
            red_pixels > 1000,
            "expected pill box background to draw many red pixels, got {red_pixels}"
        );
    }

    #[test]
    fn scene_image_asset_resolver_finds_repo_examples_path() {
        let path = super::resolve_local_scene_asset_path(
            "examples/motionloom/sample_assets/README/readme_showcase1.png",
        );
        assert!(
            path.exists(),
            "resolved image path does not exist: {path:?}"
        );
    }

    #[test]
    fn scene_image_asset_resolver_finds_scene_example_path() {
        let path = super::resolve_local_scene_asset_path(
            "examples/motionloom/scene/characters/character9/parts/face_base.png",
        );
        assert!(
            path.exists(),
            "resolved scene example image path does not exist: {path:?}"
        );
    }

    #[test]
    fn scene_image_asset_resolver_finds_motionloom_root_relative_path() {
        let path = super::resolve_local_scene_asset_path(
            "scene/characters/character9/parts/face_base.png",
        );
        assert!(
            path.exists(),
            "resolved motionloom-root scene image path does not exist: {path:?}"
        );
    }

    #[test]
    fn scene_asset_suffixes_accept_motionloom_root_paths() {
        let suffixes = crate::scene::resource::scene_asset_relative_suffixes(Path::new(
            "examples/motionloom/scene/characters/character9/parts/face_base.png",
        ));
        assert!(
            suffixes
                .iter()
                .any(|path| path == Path::new("scene/characters/character9/parts/face_base.png")),
            "expected motionloom-root relative suffix, got {suffixes:?}"
        );
    }

    #[test]
    fn scene_image_asset_resolver_finds_legacy_sample_assets_path() {
        let path =
            super::resolve_local_scene_asset_path("../sample_assets/README/readme_showcase1.png");
        assert!(
            path.exists(),
            "resolved image path does not exist: {path:?}"
        );
    }

    #[test]
    fn scene_color_parser_accepts_bgra_normalized_array() {
        assert_eq!(super::parse_color("[1,0,0,1]").unwrap(), [0, 0, 255, 255]);
        assert_eq!(super::parse_color("[0,0,1,1]").unwrap(), [255, 0, 0, 255]);
        assert_eq!(super::parse_color("[0,1,0,0.5]").unwrap(), [0, 255, 0, 128]);
    }

    #[test]
    fn scene_color_parser_accepts_bgra_byte_array() {
        assert_eq!(
            super::parse_color("[255, 128, 0, 64]").unwrap(),
            [0, 128, 255, 64]
        );
    }

    #[test]
    fn scene_color_parser_accepts_css_rgb_and_rgba() {
        assert_eq!(
            super::parse_color("rgb(255, 255, 255)").unwrap(),
            [255, 255, 255, 255]
        );
        assert_eq!(
            super::parse_color("rgb(100%, 50%, 0%)").unwrap(),
            [255, 128, 0, 255]
        );
        assert_eq!(
            super::parse_color("rgba(255, 0, 128, 0.5)").unwrap(),
            [255, 0, 128, 128]
        );
        assert_eq!(
            super::parse_color("rgba(10, 20, 30, 25%)").unwrap(),
            [10, 20, 30, 64]
        );
    }

    #[test]
    fn scene_path_morph_interpolates_compatible_path_data() {
        let d = r#"morph("0:M 0 0 L 10 0 L 10 10 Z", "2:M 0 0 L 20 10 L 20 20 Z")"#;
        let interpolated = super::eval_path_d(d, 0.5, 1.0).unwrap().into_owned();

        assert_eq!(interpolated, "M 0 0 L 15 5 L 15 15 Z");
    }

    #[test]
    fn scene_parser_preserves_universal_material_resources_and_group_reference() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="material_test">
  <Defs>
    <Noise id="n" type="ridged" scale="8" />
    <Material id="m" texture="n" opacity="0.12" />
  </Defs>
  <Timeline>
  <Track>
  <Sequence from="0s" duration="1s">
  <Layer>
    <Group id="g" material="m">
      <Rect x="4" y="4" width="20" height="20" color="#fff" />
    </Group>
  </Layer>
  </Sequence>
  </Track>
  </Timeline>
  </Scene>
  <Present from="material_test" />
</Graph>
"##,
        )
        .expect("material graph parse");
        let defs = graph.scenes[0]
            .children
            .iter()
            .find_map(|node| match node {
                SceneNode::Defs(value) => Some(value),
                _ => None,
            })
            .expect("defs");
        assert_eq!(defs.noises[0].kind, "ridged");
        assert_eq!(defs.materials[0].opacity, "0.12");
        fn group_material(nodes: &[SceneNode]) -> Option<&str> {
            for node in nodes {
                match node {
                    SceneNode::Group(group) if group.id.as_deref() == Some("g") => {
                        return group.material.as_deref();
                    }
                    SceneNode::Timeline(v) => {
                        if let Some(x) = group_material(&v.children) {
                            return Some(x);
                        }
                    }
                    SceneNode::Track(v) => {
                        if let Some(x) = group_material(&v.children) {
                            return Some(x);
                        }
                    }
                    SceneNode::Sequence(v) => {
                        if let Some(x) = group_material(&v.children) {
                            return Some(x);
                        }
                    }
                    SceneNode::Layer(v) => {
                        if let Some(x) = group_material(&v.children) {
                            return Some(x);
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        assert_eq!(group_material(&graph.scenes[0].children), Some("m"));
    }

    #[test]
    fn scene_path_morph_normalizes_different_commands_and_point_counts() {
        let d = r#"morph("0:M 8 8 H 40 V 40 H 8 Z", "2:M 24 4 C 46 4 52 20 48 36 C 44 52 20 54 8 38 C -4 22 2 4 24 4 Z")"#;
        let interpolated = super::eval_path_d(d, 0.5, 1.0).unwrap().into_owned();
        let subpaths = super::parse_path_subpaths(&interpolated).unwrap();

        assert_eq!(subpaths.len(), 1);
        assert!(subpaths[0].len() >= 24);
        assert!(super::point_distance(subpaths[0][0], *subpaths[0].last().unwrap()) < 0.01);
    }

    #[test]
    fn scene_path_morph_aligns_closed_contour_winding_and_start() {
        let d = r#"morph("0:M 0 0 L 40 0 L 40 40 L 0 40 Z", "1:M 40 40 L 40 0 L 0 0 L 0 40 Z")"#;
        let interpolated = super::eval_path_d(d, 0.5, 0.5).unwrap().into_owned();
        let subpaths = super::parse_path_subpaths(&interpolated).unwrap();
        let (min_x, max_x) = subpaths[0]
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), point| {
                (min.min(point.x), max.max(point.x))
            });

        assert!(
            max_x - min_x > 35.0,
            "morph collapsed after winding alignment"
        );
    }

    #[test]
    fn scene_renderer_draws_path_d_morph() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#000000" />
  <Scene id="morph_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Path id="morph_rect"
                  d={morph("0:M 4 4 L 18 4 L 18 20 L 4 20 Z", "1:M 4 4 L 42 4 L 42 20 L 4 20 Z")}
                  fill="#ff0000"
                  stroke="none"
                  opacity="1" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="morph_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 15)).expect("frame 15");
        let inside = rendered.get_pixel(12, 10);
        assert!(
            inside[0] > 200,
            "expected red morphed path pixel, got {inside:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_text_animator_word_opacity() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[160,72]}>
  <Background color="#000000" />
  <Scene id="text_anim_scene">
    <Timeline>
      <Track id="main" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Text value="AI edits"
                  x="8"
                  y="44"
                  fontSize="32"
                  color="#ffffff">
              <TextAnimator selector="word" duration="0.40s" stagger="0.10s">
                <Style opacity={curve("0:0:linear, 0.30:1:ease_out")} />
              </TextAnimator>
            </Text>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="text_anim_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let start = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        assert!(
            max_rgb(&start) < 16,
            "expected animated text to start transparent"
        );

        let later = pollster::block_on(renderer.render_frame(&graph, 15)).expect("frame 15");
        assert!(
            max_rgb(&later) > 180,
            "expected animated text to become visible"
        );
    }

    #[test]
    fn scene_renderer_applies_text_stroke() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[120,72]}>
  <Background color="#000000" />
  <Scene id="text_stroke_scene">
    <Timeline>
      <Track id="main" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Text value="I"
                  x="32"
                  y="52"
                  fontSize="48"
                  color="#000000"
                  stroke="#ffffff"
                  strokeWidth="8" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="text_stroke_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        assert!(
            max_rgb(&rendered) > 180,
            "expected white text stroke on black background"
        );
    }

    #[test]
    fn scene_renderer_applies_text_glow_effect() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[180,96]}>
  <Background color="#000000" />
  <Scene id="text_glow_scene">
    <Timeline>
      <Track id="main" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Text value="GLOW"
                  x="22"
                  y="58"
                  fontSize="32"
                  color="#000000">
              <TextAnimator selector="word" duration="1s">
                <Effects>
                  <Glow radius="12" intensity="1.6" color="#00ff66" />
                </Effects>
              </TextAnimator>
            </Text>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="text_glow_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 15)).expect("frame 15");
        // The glow blur kernel spreads the green channel over a wider area,
        // so the peak intensity is lower than the raw glow color.
        assert!(
            max_green(&rendered) > 40,
            "expected green glow generated from text alpha, got {}",
            max_green(&rendered)
        );
    }

    #[test]
    fn scene_sequence_uses_local_curve_time() {
        let graph = parse_graph_script(
            r##"
<Graph fps={10} duration="2s" size={[64,48]}>
  <Background color="#000000" />
  <Scene id="timeline_scene">
    <Timeline>
      <Track id="main" z="0">
        <Sequence from="1s" duration="1s" out="hold">
          <Rect x="8"
                y={curve("0:30:linear, 1:10:linear")}
                width="24"
                height="12"
                color="#ffffff" />
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="timeline_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let before = pollster::block_on(renderer.render_frame(&graph, 5)).expect("frame before");
        assert_eq!(
            before.get_pixel(12, 25)[0],
            0,
            "sequence should be hidden before its from time"
        );

        let mid = pollster::block_on(renderer.render_frame(&graph, 15)).expect("frame mid");
        let inside = mid.get_pixel(12, 25);
        assert!(
            inside[0] > 220,
            "expected local-time rect at frame 15, got {inside:?}"
        );
    }

    #[test]
    fn scene_precompose_source_uses_sequence_local_time() {
        let graph = parse_graph_script(
            r##"
<Graph fps={10} duration="12s" size={[32,32]}>
  <Background color="#000000" />
  <Scene id="precompose_time_scene">
    <Defs>
      <Precompose id="fade_plate" duration="1s" size={[32,32]}>
        <Rect x="0"
              y="0"
              width="32"
              height="32"
              color="#ff0000"
              opacity={curve("0:0:linear, 1:1:linear")} />
      </Precompose>
    </Defs>
    <Timeline>
      <Track id="main" z="0">
        <Sequence from="10s" duration="1s" out="hold">
          <Layer source="fade_plate" />
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="precompose_time_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 105)).expect("frame 105");
        let pixel = rendered.get_pixel(4, 4);
        // The precompose rect has opacity=0.5 at sequence-local 0.5s.
        // When composited onto the opaque black background, the final pixel
        // is half-red (128) with full alpha (255).
        assert!(
            pixel[0] > 100 && pixel[0] < 160 && pixel[3] > 240,
            "expected precompose source to evaluate at sequence-local 0.5s (should be ~127), got {pixel:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_group_mask_alpha() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="masked_scene">
    <Defs>
      <Mask id="left_half" shape="rect" x="0" y="0" width="32" height="32" />
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="masked_red" mask="left_half">
              <Rect x="0" y="0" width="64" height="32" color="#ff0000" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="masked_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let inside = rendered.get_pixel(8, 8);
        let outside = rendered.get_pixel(48, 8);
        assert!(inside[0] > 200, "expected masked red pixel, got {inside:?}");
        assert!(
            outside[0] < 30 && outside[1] < 30 && outside[2] < 30,
            "expected outside mask to remain background, got {outside:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_ellipse_group_mask_alpha() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Background color="#000000" />
  <Scene id="masked_scene">
    <Defs>
      <Mask id="ellipse_mask" shape="ellipse" x="16" y="8" width="32" height="48" />
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="masked_red" mask="ellipse_mask">
              <Rect x="0" y="0" width="64" height="64" color="#ff0000" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="masked_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let center = rendered.get_pixel(32, 32);
        let bounding_box_corner = rendered.get_pixel(17, 9);
        assert!(
            center[0] > 200,
            "expected ellipse center to reveal red, got {center:?}"
        );
        assert!(
            bounding_box_corner[0] < 30
                && bounding_box_corner[1] < 30
                && bounding_box_corner[2] < 30,
            "expected ellipse corner outside shape to remain background, got {bounding_box_corner:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_precompose_layer_luma_matte() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="matte_scene">
    <Defs>
      <Precompose id="green_source" size={[64,32]}>
        <Rect x="0" y="0" width="64" height="32" color="#00ff00" />
      </Precompose>
      <Precompose id="luma_cut" size={[64,32]}>
        <Rect x="0" y="0" width="32" height="32" color="#ffffff" />
        <Rect x="32" y="0" width="32" height="32" color="#000000" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Layer id="matted_green" source="green_source" matte="luma_cut" matteMode="luma" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="matte_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let revealed = rendered.get_pixel(8, 8);
        let hidden = rendered.get_pixel(48, 8);
        assert!(
            revealed[1] > 200,
            "expected luma matte to reveal green pixel, got {revealed:?}"
        );
        assert!(
            hidden[0] < 30 && hidden[1] < 30 && hidden[2] < 30,
            "expected black luma matte to hide source, got {hidden:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_layer_matte_from_path_node() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="matte_from_scene">
    <Defs>
      <Precompose id="matte_library" size={[64,32]}>
        <Path id="left_path_matte" d="M 0 0 L 32 0 L 32 32 L 0 32 Z" fill="#ffffff" stroke="none" />
      </Precompose>
      <Precompose id="green_source" size={[64,32]}>
        <Rect x="0" y="0" width="64" height="32" color="#00ff00" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Layer id="matted_green" source="green_source" matteFrom="left_path_matte" matteMode="alpha" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="matte_from_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let revealed = rendered.get_pixel(8, 8);
        let hidden = rendered.get_pixel(48, 8);
        assert!(
            revealed[1] > 200,
            "expected matteFrom path node to reveal green pixel, got {revealed:?}"
        );
        assert!(
            hidden[0] < 30 && hidden[1] < 30 && hidden[2] < 30,
            "expected outside matteFrom path node to remain background, got {hidden:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_group_mask_from_path_node() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="mask_from_scene">
    <Defs>
      <Precompose id="matte_library" size={[64,32]}>
        <Path id="left_group_matte" d="M 0 0 L 32 0 L 32 32 L 0 32 Z" fill="#ffffff" stroke="none" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="masked_red" maskFrom="left_group_matte" maskMode="alpha">
              <Rect x="0" y="0" width="64" height="32" color="#ff0000" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="mask_from_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let revealed = rendered.get_pixel(8, 8);
        let hidden = rendered.get_pixel(48, 8);
        assert!(
            revealed[0] > 200,
            "expected maskFrom path node to reveal red pixel, got {revealed:?}"
        );
        assert!(
            hidden[0] < 30 && hidden[1] < 30 && hidden[2] < 30,
            "expected outside maskFrom path node to remain background, got {hidden:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_group_mask_shape_and_effect_stack() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="mask_shape_effect_scene">
    <Defs>
      <Filter id="half_opacity">
        <Effect type="opacity" opacity="0.5" />
      </Filter>
      <Precompose id="matte_library" size={[64,32]}>
        <Rect id="center_matte" x="20" y="0" width="24" height="32" color="#ffffff" />
      </Precompose>
    </Defs>
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="treated_red"
                   maskFrom="center_matte"
                   maskFeather="3"
                   maskExpansion="4"
                   effects={["half_opacity"]}>
              <Rect x="0" y="0" width="64" height="32" color="#ff0000" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="mask_shape_effect_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let center = rendered.get_pixel(32, 16);
        let expanded_edge = rendered.get_pixel(17, 16);
        let hidden = rendered.get_pixel(5, 16);
        assert!(
            center[0] > 90 && center[0] < 180,
            "expected ordered opacity effect on masked group, got {center:?}"
        );
        assert!(
            expanded_edge[0] > 5,
            "expected mask expansion plus feather to reveal edge, got {expanded_edge:?}"
        );
        assert!(
            hidden[0] < 5,
            "expected distant pixel to remain hidden, got {hidden:?}"
        );
    }

    #[test]
    fn scene_renderer_renders_ellipse_stroke_fill_none_with_opacity_and_group_rotation() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Background color="#000000" />
  <Scene id="ellipse_stroke_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="rotated_ellipse_group" rotation="90" transformOriginX="32" transformOriginY="32">
              <Ellipse id="ellipse_stroke" cx="32" cy="32" rx="20" ry="8" fill="none" stroke="#ffffff" strokeWidth="6" opacity="0.5" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="ellipse_stroke_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let vertical_ring = rendered.get_pixel(32, 12);
        let center = rendered.get_pixel(32, 32);
        assert!(
            vertical_ring[0] > 70 && vertical_ring[0] < 190,
            "expected semi-transparent rotated ellipse stroke, got {vertical_ring:?}"
        );
        assert!(
            center[0] < 30 && center[1] < 30 && center[2] < 30,
            "expected fill=none ellipse center to remain background, got {center:?}"
        );
    }

    #[test]
    fn scene_renderer_renders_circle_stroke_fill_none() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Background color="#000000" />
  <Scene id="circle_stroke_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Circle id="circle_stroke" x="32" y="32" radius="16" fill="none" stroke="#ffffff" strokeWidth="8" opacity="0.5" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="circle_stroke_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let ring = rendered.get_pixel(44, 32);
        let center = rendered.get_pixel(32, 32);
        assert!(ring[0] > 200, "expected circle stroke, got {ring:?}");
        assert!(
            center[0] < 30 && center[1] < 30 && center[2] < 30,
            "expected fill=none circle center to remain background, got {center:?}"
        );
    }

    #[test]
    fn scene_renderer_renders_rect_stroke_fill_none() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Background color="#000000" />
  <Scene id="rect_stroke_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect id="rect_stroke" x="12" y="16" width="40" height="32" radius="4" fill="none" stroke="#ffffff" strokeWidth="6" opacity="0.5" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="rect_stroke_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let edge = rendered.get_pixel(12, 32);
        let center = rendered.get_pixel(32, 32);
        assert!(edge[0] > 70, "expected rectangle stroke, got {edge:?}");
        assert!(
            center[0] < 30 && center[1] < 30 && center[2] < 30,
            "expected fill=none rectangle center to remain background, got {center:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_group_mask_from_ellipse_stroke_node() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Background color="#000000" />
  <Scene id="ellipse_mask_from_scene">
    <Defs>
      <Precompose id="matte_library" size={[64,64]}>
        <Ellipse id="ellipse_stroke_matte" cx="32" cy="32" rx="20" ry="12" fill="none" stroke="#ffffff" strokeWidth="8" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="masked_red" maskFrom="ellipse_stroke_matte" maskMode="alpha">
              <Rect x="0" y="0" width="64" height="64" color="#ff0000" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="ellipse_mask_from_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let revealed_ring = rendered.get_pixel(48, 32);
        let hidden_center = rendered.get_pixel(32, 32);
        assert!(
            revealed_ring[0] > 160,
            "expected ellipse stroke maskFrom ring to reveal red, got {revealed_ring:?}"
        );
        assert!(
            hidden_center[0] < 40 && hidden_center[1] < 40 && hidden_center[2] < 40,
            "expected fill=none ellipse maskFrom center to hide red, got {hidden_center:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_mask_follow_target_node() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="mask_follow_scene">
    <Defs>
      <Mask id="follow_target_mask" follow="node:target_dot" shape="circle" x="0" y="0" radius="9" />
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Circle id="target_dot" x="16" y="16" radius="2" color="#ffffff" opacity="0" />
            <Group id="masked_red" mask="follow_target_mask" maskMode="alpha">
              <Rect x="0" y="0" width="64" height="32" color="#ff0000" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="mask_follow_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let revealed = rendered.get_pixel(16, 16);
        let hidden = rendered.get_pixel(48, 16);
        assert!(
            revealed[0] > 200,
            "expected followed mask to reveal red near target, got {revealed:?}"
        );
        assert!(
            hidden[0] < 30 && hidden[1] < 30 && hidden[2] < 30,
            "expected outside followed mask to remain background, got {hidden:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_hand_drawn_edge_treatment_stack() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="edge_treatment_scene">
    <Defs>
      <Filter id="hand_drawn_edges">
        <EdgeSoftness radius="2" amount="0.7" preserveInterior="0.9" />
        <EdgeRoughness radius="1.5" amount="0.8" scale="12" seed="7" />
        <ColorBleed radius="3" amount="0.35" />
      </Filter>
    </Defs>
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="treated_red" effects={["hand_drawn_edges"]}>
              <Rect x="20" y="8" width="24" height="16" color="#ff0000" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="edge_treatment_scene" />
</Graph>
"##,
        )
        .expect("edge treatment graph parse");
        let filter = graph.scenes[0]
            .children
            .iter()
            .find_map(|node| match node {
                SceneNode::Defs(defs) => defs.filters.first(),
                _ => None,
            })
            .expect("edge treatment filter");
        assert_eq!(filter.steps.len(), 3);
        assert_eq!(filter.steps[0].kind, "edgeSoftness");
        assert_eq!(filter.steps[1].scale.as_deref(), Some("12"));
        assert_eq!(filter.steps[2].kind, "colorBleed");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let interior = rendered.get_pixel(32, 16);
        let softened_edge = rendered.get_pixel(18, 16);
        assert!(
            interior[0] > 220,
            "expected preserved red interior, got {interior:?}"
        );
        assert!(
            softened_edge[0] > 0,
            "expected edge treatment to spread red beyond the original boundary, got {softened_edge:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_defs_filter_to_layer() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="filter_scene">
    <Defs>
      <Filter id="soft_glow">
        <Blur radius="12" />
        <ColorMatrix saturation="1.0" brightness="1.8" />
      </Filter>
      <Precompose id="white_plate" size={[64,32]}>
        <Rect x="24" y="8" width="16" height="16" color="#ffffff" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Layer id="filtered_plate" source="white_plate" effect="soft_glow" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="filter_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let softened_edge = rendered.get_pixel(18, 16);
        assert!(
            softened_edge[0] > 10 || softened_edge[1] > 10 || softened_edge[2] > 10,
            "expected filter blur to spread light outside source rect, got {softened_edge:?}"
        );
    }

    #[test]
    fn scene_gpu_native_path_applies_defs_filter_to_layer() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="filter_scene">
    <Defs>
      <Filter id="soft_glow">
        <Blur radius="12" />
        <ColorMatrix saturation="1.0" brightness="1.8" />
      </Filter>
      <Precompose id="white_plate" size={[64,32]}>
        <Rect x="24" y="8" width="16" height="16" color="#ffffff" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Layer id="filtered_plate" source="white_plate" effect="soft_glow" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="filter_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let source =
            match pollster::block_on(renderer.try_render_gpu_scene_tree_frame(&graph, 0.0, 0.0)) {
                Ok(Some(source)) => source,
                Ok(None) => panic!("expected GPU-native filter path"),
                Err(MotionLoomSceneRenderError::GpuRender { message })
                    if is_gpu_adapter_unavailable(&message) =>
                {
                    skip_gpu_texture_test(message);
                    return;
                }
                Err(err) => panic!("unexpected GPU native filter render error: {err}"),
            };
        let rendered = pollster::block_on(renderer.graph_source_to_cpu(&source))
            .expect("GPU readback after filter render");

        let softened_edge = rendered.get_pixel(18, 16);
        assert!(
            softened_edge[0] > 10 || softened_edge[1] > 10 || softened_edge[2] > 10,
            "expected GPU filter blur to spread light outside source rect, got {softened_edge:?}"
        );
    }

    #[test]
    fn scene_gpu_native_path_draws_face_jaw_without_cpu_overlay() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Background color="#000000" />
  <Scene id="face_jaw_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <FaceJaw x="32"
                     y="8"
                     width="34"
                     height="48"
                     cheekWidth="28"
                     chinWidth="12"
                     chinSharpness="0.75"
                     jawEase="0.65"
                     closed="true"
                     fill="#ff0000"
                     stroke="none"
                     opacity="1" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="face_jaw_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let source =
            match pollster::block_on(renderer.try_render_gpu_scene_tree_frame(&graph, 0.0, 0.0)) {
                Ok(Some(source)) => source,
                Ok(None) => panic!("expected GPU-native FaceJaw path"),
                Err(MotionLoomSceneRenderError::GpuRender { message })
                    if is_gpu_adapter_unavailable(&message) =>
                {
                    skip_gpu_texture_test(message);
                    return;
                }
                Err(err) => panic!("unexpected GPU native FaceJaw render error: {err}"),
            };
        let rendered = pollster::block_on(renderer.graph_source_to_cpu(&source))
            .expect("GPU readback after FaceJaw render");

        let max_red = rendered.pixels().map(|pixel| pixel[0]).max().unwrap_or(0);
        assert!(
            max_red > 180,
            "expected GPU-native FaceJaw to draw red pixels, max red={max_red}"
        );
    }

    #[test]
    fn scene_renderer_draws_layer_container_children() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="layer_container_scene">
    <Defs>
      <Mask id="left_half" shape="rect" x="0" y="0" width="32" height="32" />
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Layer id="screened_layer" blend="screen" mask="left_half">
              <Rect x="0" y="0" width="64" height="32" color="#ff0000" />
            </Layer>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="layer_container_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let revealed = rendered.get_pixel(8, 8);
        let hidden = rendered.get_pixel(48, 8);
        assert!(
            revealed[0] > 200,
            "expected layer child to render through mask, got {revealed:?}"
        );
        assert!(
            hidden[0] < 30 && hidden[1] < 30 && hidden[2] < 30,
            "expected layer mask to hide child content, got {hidden:?}"
        );
    }

    #[test]
    fn scene_renderer_uses_defs_mask_precompose_and_component() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="defs_resource_scene">
    <Defs>
      <Mask id="left_half" shape="rect" x="0" y="0" width="32" height="32" />
      <Precompose id="red_plate" size={[64,32]}>
        <Rect x="0" y="0" width="64" height="32" color="#ff0000" />
      </Precompose>
      <Component id="green_dot">
        <Circle x="0" y="0" radius="6" color="#00ff00" />
      </Component>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Layer source="red_plate" mask="left_half" />
            <Use ref="green_dot" x="48" y="16" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="defs_resource_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");

        let red_revealed = rendered.get_pixel(8, 8);
        let red_hidden = rendered.get_pixel(40, 8);
        let component_pixel = rendered.get_pixel(48, 16);
        assert!(
            red_revealed[0] > 200,
            "expected Defs precompose to render through Defs mask, got {red_revealed:?}"
        );
        assert!(
            red_hidden[0] < 30 && red_hidden[1] < 30 && red_hidden[2] < 30,
            "expected Defs mask to hide precompose on right side, got {red_hidden:?}"
        );
        assert!(
            component_pixel[1] > 200,
            "expected Defs component to render through Use, got {component_pixel:?}"
        );
    }

    #[test]
    fn scene_gpu_native_path_uses_defs_mask_and_precompose() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="defs_resource_gpu_scene">
    <Defs>
      <Mask id="left_half" shape="rect" x="0" y="0" width="32" height="32" />
      <Precompose id="red_plate" size={[64,32]}>
        <Rect x="0" y="0" width="64" height="32" color="#ff0000" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Layer source="red_plate" mask="left_half" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="defs_resource_gpu_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU Defs resource test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };

        let revealed = rendered.get_pixel(8, 8);
        let hidden = rendered.get_pixel(48, 8);
        assert!(
            revealed[0] > 200,
            "expected GPU Defs precompose to render through Defs mask, got {revealed:?}"
        );
        assert!(
            hidden[0] < 30 && hidden[1] < 30 && hidden[2] < 30,
            "expected GPU Defs mask to hide precompose on right side, got {hidden:?}"
        );
    }

    #[test]
    fn scene_gpu_native_path_expands_repeat_use_components() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[80,32]}>
  <Background color="#000000" />
  <Scene id="component_gpu_scene">
    <Defs>
      <Component id="green_dot">
        <Circle x="0" y="0" radius="5" color="#00ff00" />
      </Component>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Repeat count="3" x="16" y="16" xStep="20">
              <Use ref="green_dot" blend="screen" />
            </Repeat>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="component_gpu_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let nodes = super::scene_nodes_for_present(&graph).expect("present scene");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let source = match pollster::block_on(renderer.try_render_gpu_scene_nodes_composited(
            nodes,
            graph.size,
            graph.size,
            super::Affine2::identity(),
            0.0,
            0.0,
            Some([0, 0, 0, 255]),
        )) {
            Ok(Some(source)) => source,
            Ok(None) => panic!("Repeat + Use + Component should stay GPU-native"),
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if message.contains("GPU adapter") =>
            {
                eprintln!("Skipping GPU Use/Component native test: {message}");
                return;
            }
            Err(err) => panic!("unexpected native GPU render error: {err}"),
        };
        let rendered = pollster::block_on(renderer.graph_source_to_cpu(&source))
            .expect("GPU readback after native composited render");

        for x in [16, 36, 56] {
            let pixel = rendered.get_pixel(x, 16);
            assert!(
                pixel[1] > 180,
                "expected GPU-native Use/Component dot at x={x}, got {pixel:?}"
            );
        }
    }

    #[test]
    fn scene_gpu_primitive_path_expands_component_use() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,40]}>
  <Background color="#000000" />
  <Scene id="component_primitive_gpu_scene">
    <Defs>
      <Component id="blue_bar">
        <Rect x="0" y="0" width="18" height="20" color="#0077ff" />
      </Component>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Use ref="blue_bar" x="22" y="10" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="component_primitive_gpu_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if message.contains("GPU adapter") =>
            {
                eprintln!("Skipping GPU Component primitive test: {message}");
                return;
            }
            Err(err) => panic!("unexpected GPU primitive render error: {err}"),
        };

        let pixel = rendered.get_pixel(30, 18);
        assert!(
            pixel[2] > 180 && pixel[0] < 80,
            "expected Component Use to render on primitive GPU path, got {pixel:?}"
        );
    }

    #[test]
    fn scene_gpu_native_path_draws_layer_container_children() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="layer_container_gpu_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Layer id="screened_layer" blend="screen">
              <Rect x="0" y="0" width="32" height="32" color="#ff0000" />
            </Layer>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="layer_container_gpu_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if is_gpu_adapter_unavailable(&message) =>
            {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected GPU layer container render error: {err}"),
        };

        let inside = rendered.get_pixel(8, 8);
        let outside = rendered.get_pixel(48, 8);
        assert!(
            inside[0] > 200,
            "expected GPU layer child to render, got {inside:?}"
        );
        assert!(
            outside[0] < 30 && outside[1] < 30 && outside[2] < 30,
            "expected outside GPU layer child to remain background, got {outside:?}"
        );
    }

    #[test]
    fn scene_gpu_native_path_composites_compound_path_overlay() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[80,80]}>
  <Background color="#000000" />
  <Scene id="compound_path_gpu_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Path d="M 10 10 L 70 10 L 70 70 L 10 70 Z M 30 30 L 30 50 L 50 50 L 50 30 Z"
                  fill="#ff0000"
                  stroke="none" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="compound_path_gpu_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let nodes = scene_nodes_for_present(&graph).expect("presentable scene");
        let source = match pollster::block_on(renderer.try_render_gpu_scene_nodes_composited(
            nodes,
            graph_output_size(&graph),
            graph_logical_render_size(&graph),
            render_size_root_transform(
                graph_output_size(&graph),
                graph_logical_render_size(&graph),
            ),
            0.0,
            0.0,
            Some([0, 0, 0, 255]),
        )) {
            Ok(Some(source)) => source,
            Ok(None) => panic!("expected GPU composited compound path source"),
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if is_gpu_adapter_unavailable(&message) =>
            {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected GPU compound path compositing error: {err}"),
        };
        let rendered = pollster::block_on(renderer.graph_source_to_cpu(&source))
            .expect("GPU compound path readback");

        let ring = rendered.get_pixel(20, 20);
        let hole = rendered.get_pixel(40, 40);
        assert!(
            ring[0] > 180 && ring[1] < 80 && ring[2] < 80,
            "expected GPU compound path ring, got {ring:?}"
        );
        assert!(
            hole[0] < 40 && hole[1] < 40 && hole[2] < 40,
            "expected GPU compound path hole to stay black, got {hole:?}"
        );
    }

    #[test]
    fn scene_vector_overlay_caches_static_compound_path() {
        let path = PathNode {
            id: Some("compound_path_cache_test".to_string()),
            brush: None,
            material: None,
            x: "0".to_string(),
            y: "0".to_string(),
            rotation: "0".to_string(),
            scale: "1".to_string(),
            scale_x: "1".to_string(),
            scale_y: "1".to_string(),
            skew_x: "0".to_string(),
            skew_y: "0".to_string(),
            transform_origin_x: "0".to_string(),
            transform_origin_y: "0".to_string(),
            d: "M 10 10 L 70 10 L 70 70 L 10 70 Z M 30 30 L 30 50 L 50 50 L 50 30 Z".to_string(),
            stroke: "none".to_string(),
            fill: Some("#ff0000".to_string()),
            fill_rule: "nonzero".to_string(),
            boolean_op: "none".to_string(),
            offset_path: "0".to_string(),
            round_corners: "0".to_string(),
            normalize: "false".to_string(),
            stroke_width: "1".to_string(),
            stroke_width_start: "1".to_string(),
            stroke_width_end: "1".to_string(),
            stroke_width_profile: String::new(),
            opacity: "1".to_string(),
            trim_start: "0".to_string(),
            trim_end: "1".to_string(),
            line_cap: "butt".to_string(),
            line_join: "miter".to_string(),
            taper_start: "0".to_string(),
            taper_end: "0".to_string(),
            stroke_style: "solid".to_string(),
            stroke_roughness: "0".to_string(),
            stroke_copies: "1".to_string(),
            stroke_texture: "0".to_string(),
            stroke_bristles: "0".to_string(),
            stroke_pressure: "1".to_string(),
            stroke_pressure_min: "1".to_string(),
            stroke_pressure_curve: "1".to_string(),
            blend: "normal".to_string(),
            texture: None,
            texture_opacity: "1".to_string(),
            texture_scale: "1".to_string(),
            texture_mask: "0".to_string(),
        };
        let nodes = [SceneNode::Path(path)];
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));

        renderer
            .gpu_vector_overlay_texture_layer(&nodes, Affine2::identity(), 1.0, 0.0, 0.0, (80, 80))
            .expect("first overlay render")
            .expect("first overlay layer");
        renderer
            .gpu_vector_overlay_texture_layer(&nodes, Affine2::identity(), 1.0, 0.0, 0.0, (80, 80))
            .expect("second overlay render")
            .expect("second overlay layer");
        renderer
            .gpu_vector_overlay_texture_layer(
                &nodes,
                Affine2::translate(12.0, 7.0),
                1.0,
                0.0,
                0.0,
                (80, 80),
            )
            .expect("translated overlay render")
            .expect("translated overlay layer");
        if let SceneNode::Path(path) = &nodes[0] {
            renderer
                .gpu_path_overlay_texture_layer(path, Affine2::scale(1.5), 1.0, 0.0, 0.0)
                .expect("scaled local path overlay render")
                .expect("scaled local path overlay layer");
            renderer
                .gpu_path_overlay_texture_layer(path, Affine2::rotate_deg(15.0), 1.0, 0.0, 0.0)
                .expect("rotated local path overlay render")
                .expect("rotated local path overlay layer");
        }

        assert_eq!(
            renderer.vector_overlay_cache.len(),
            2,
            "static compound path overlay should reuse generic and local raster cache entries across transform edits"
        );
    }

    #[test]
    fn scene_gpu_native_path_handles_mask_matte_precompose() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="gpu_native_matte">
    <Defs>
      <Mask id="left_mask" shape="rect" x="0" y="0" width="32" height="32" feather="1" />
      <Precompose id="green_source" size={[64,32]}>
        <Rect x="0" y="0" width="64" height="32" color="#00ff00" />
      </Precompose>
      <Precompose id="luma_cut" size={[64,32]}>
        <Rect x="0" y="0" width="32" height="32" color="#ffffff" />
        <Rect x="32" y="0" width="32" height="32" color="#000000" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Layer id="matted_green" source="green_source" matte="luma_cut" matteMode="luma" />
            <Group id="masked_blue" mask="left_mask">
              <Rect x="0" y="0" width="64" height="32" color="#0000ff" opacity="0.35" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="gpu_native_matte" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let nodes = super::scene_nodes_for_present(&graph).expect("present scene");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let source = match pollster::block_on(renderer.try_render_gpu_scene_nodes_composited(
            nodes,
            graph.size,
            graph.size,
            super::Affine2::identity(),
            0.0,
            0.0,
            Some([0, 0, 0, 255]),
        )) {
            Ok(Some(source)) => source,
            Ok(None) => panic!("expected mask/matte/precompose native GPU path"),
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if is_gpu_adapter_unavailable(&message) =>
            {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected native GPU mask/matte render error: {err}"),
        };
        let rendered = pollster::block_on(renderer.graph_source_to_cpu(&source))
            .expect("GPU readback after native composited render");

        let revealed = rendered.get_pixel(8, 8);
        let hidden = rendered.get_pixel(48, 8);
        assert!(
            revealed[1] > 150 || revealed[2] > 40,
            "expected native GPU matte/mask to reveal left-side content, got {revealed:?}"
        );
        assert!(
            hidden[1] < 80,
            "expected luma matte to hide right-side green source, got {hidden:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_animated_layer_mask_from_precompose() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="6s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="animated_layer_mask_from_precompose">
    <Defs>
      <Precompose id="animated_cut" size={[64,32]}>
        <Path id="animated_cut_path"
              d={morph("0:M 4 4 L 60 4 L 60 28 L 4 28 Z",
                       "2.92:M 4 4 L 60 4 L 60 28 L 4 28 Z",
                       "3.12:M 4 15 L 60 15 L 60 17 L 4 17 Z",
                       "3.26:M 4 15 L 60 15 L 60 17 L 4 17 Z",
                       "3.48:M 4 4 L 60 4 L 60 28 L 4 28 Z",
                       "6:M 4 4 L 60 4 L 60 28 L 4 28 Z")}
              fill="#ffffff"
              stroke="none" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="screen" z="0">
        <Sequence from="0s" duration="6s" out="hold">
          <Layer id="masked_plate" maskFrom="animated_cut" maskMode="alpha">
            <Rect id="red_plate" x="0" y="0" width="64" height="32" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="animated_layer_mask_from_precompose" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let open = pollster::block_on(renderer.render_frame(&graph, 25)).expect("open frame");
        let closed = pollster::block_on(renderer.render_frame(&graph, 94)).expect("closed frame");

        let open_pixel = open.get_pixel(32, 8);
        let closed_outside = closed.get_pixel(32, 8);
        let closed_slit = closed.get_pixel(32, 16);
        assert!(
            open_pixel[0] > 200,
            "expected animated maskFrom precompose to reveal open frame, got {open_pixel:?}"
        );
        assert!(
            closed_slit[0] > 200,
            "expected animated maskFrom precompose to keep slit visible, got {closed_slit:?}"
        );
        assert!(
            closed_outside[0] < 30 && closed_outside[1] < 30 && closed_outside[2] < 30,
            "expected animated maskFrom precompose to hide outside closed slit, got {closed_outside:?}"
        );
    }

    #[test]
    fn scene_gpu_native_path_applies_animated_layer_mask_from_precompose() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="6s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="animated_layer_mask_from_precompose_gpu">
    <Defs>
      <Precompose id="animated_cut" size={[64,32]}>
        <Path id="animated_cut_path"
              d={morph("0:M 4 4 L 60 4 L 60 28 L 4 28 Z",
                       "2.92:M 4 4 L 60 4 L 60 28 L 4 28 Z",
                       "3.12:M 4 15 L 60 15 L 60 17 L 4 17 Z",
                       "3.26:M 4 15 L 60 15 L 60 17 L 4 17 Z",
                       "3.48:M 4 4 L 60 4 L 60 28 L 4 28 Z",
                       "6:M 4 4 L 60 4 L 60 28 L 4 28 Z")}
              fill="#ffffff"
              stroke="none" />
      </Precompose>
    </Defs>

    <Timeline>
      <Track id="scene_content" space="screen" z="0">
        <Sequence from="0s" duration="6s" out="hold">
          <Layer id="masked_plate" maskFrom="animated_cut" maskMode="alpha">
            <Rect id="red_plate" x="0" y="0" width="64" height="32" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="animated_layer_mask_from_precompose_gpu" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let time_sec = 94.0 / 30.0;
        let time_norm = time_sec / 6.0;
        let source = match pollster::block_on(
            renderer.try_render_gpu_scene_tree_frame(&graph, time_norm, time_sec),
        ) {
            Ok(Some(source)) => source,
            Ok(None) => panic!("expected animated layer maskFrom precompose native GPU path"),
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if is_gpu_adapter_unavailable(&message) =>
            {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected native GPU animated maskFrom render error: {err}"),
        };
        let rendered = pollster::block_on(renderer.graph_source_to_cpu(&source))
            .expect("GPU readback after animated maskFrom render");

        let closed_outside = rendered.get_pixel(32, 8);
        let closed_slit = rendered.get_pixel(32, 16);
        assert!(
            closed_slit[0] > 200,
            "expected GPU animated maskFrom precompose to keep slit visible, got {closed_slit:?}"
        );
        assert!(
            closed_outside[0] < 30 && closed_outside[1] < 30 && closed_outside[2] < 30,
            "expected GPU animated maskFrom precompose to hide outside closed slit, got {closed_outside:?}"
        );
    }

    #[test]
    fn scene_gpu_native_frame_preserves_nested_material_and_mask_shape_buffers() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,32]}>
  <Background color="#000000" />
  <Scene id="nested_material_mask_gpu">
    <Defs>
      <Noise id="surface_noise" type="fbm" scale="12" octaves="3" seed="7" contrast="1" evolution="0" />
      <Material id="surface_material"
                texture="surface_noise" textureAmount="0.2"
                displacement="surface_noise" displacementAmount="2"
                roughness="0.4" specular="0.2" opacity="1" />
      <Precompose id="center_mask" size={[64,32]}>
        <Rect x="8" y="4" width="48" height="24" color="#ffffff" />
      </Precompose>
    </Defs>
    <Timeline>
      <Track id="scene_content" space="screen" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer id="scene_layer">
            <Group id="masked_content" maskFrom="center_mask" maskMode="alpha">
              <Rect x="8" y="4" width="48" height="24" color="#202020" />
              <Group id="material_content" material="surface_material">
                <Rect x="16" y="8" width="32" height="16" color="#ff4000" />
              </Group>
            </Group>
            <Rect x="0" y="0" width="6" height="32" color="#0000ff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="nested_material_mask_gpu" />
</Graph>
"##,
        )
        .expect("nested material/mask graph parse");
        let mut renderer = match pollster::block_on(SceneRenderer::new(SceneRenderProfile::Gpu)) {
            Ok(renderer) => renderer,
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if is_gpu_adapter_unavailable(&message) =>
            {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected native GPU renderer initialization error: {err}"),
        };
        let rendered = match pollster::block_on(renderer.render_frame_gpu_readback(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if is_gpu_adapter_unavailable(&message) =>
            {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected nested material/mask GPU render error: {err}"),
        };

        let material_pixel = rendered.get_pixel(32, 16);
        let masked_background = rendered.get_pixel(10, 16);
        assert!(
            material_pixel[0] > 120 && material_pixel[0] > material_pixel[1] * 2,
            "expected nested material content to remain visible, got {material_pixel:?}"
        );
        assert!(
            masked_background[0] > 10 && masked_background[1] > 10,
            "expected the masked sibling geometry to remain visible, got {masked_background:?}"
        );
    }

    #[test]
    fn scene_group_deform_grid_warps_layer_content() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#000000" />
  <Scene id="deform_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="deformed"
                   deformGrid="2x2"
                   deformAmount="1"
                   gridFrom="10,10 30,10; 10,30 30,30"
                   gridTo="20,10 40,10; 20,30 40,30">
              <Rect id="source_rect"
                    x="10"
                    y="10"
                    width="20"
                    height="20"
                    color="#ff0000"
                    opacity="1" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="deform_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let moved = rendered.get_pixel(25, 20);
        let original = rendered.get_pixel(12, 20);
        assert!(
            moved[0] > 200,
            "expected deformed red pixel at moved position, got {moved:?}"
        );
        assert!(
            original[0] < 30,
            "expected original source position to be transparent/black, got {original:?}"
        );
    }

    #[test]
    fn scene_gpu_group_deform_grid_warps_vector_primitives() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#000000" />
  <Scene id="deform_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="deformed"
                   deformGrid="2x2"
                   deformAmount="1"
                   gridFrom="10,10 30,10; 10,30 30,30"
                   gridTo="20,10 40,10; 20,30 40,30">
              <Path id="source_path"
                    d="M 10 10 L 30 10 L 30 30 L 10 30 Z"
                    fill="#ff0000"
                    stroke="none"
                    opacity="1" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="deform_scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                if message.contains("CPU overlays") {
                    panic!("DeformGrid vector primitives must stay GPU-native: {message}");
                }
                eprintln!("Skipping GPU DeformGrid vector test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };
        let moved = rendered.get_pixel(25, 20);
        let original = rendered.get_pixel(12, 20);
        assert!(
            moved[0] > 180,
            "expected GPU-deformed red pixel at moved position, got {moved:?}"
        );
        assert!(
            original[0] < 40,
            "expected original source position to be transparent/black, got {original:?}"
        );
    }

    #[test]
    fn scene_renderer_draws_unified_scene_layer_graph() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="4s" size={[128,72]} renderSize={[128,72]}>
  <Background color="#000000" />
  <Scene id="hello_scene">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="4s" out="hold">
          <Layer>
            <Text id="hello_text"
                  value="hello"
                  x="center"
                  y="center"
                  fontSize="18"
                  color="#ffffff"
                  opacity={curve("0:1:linear, 4:1:linear")} />
            <Circle id="accent_orb"
                    x="64"
                    y="36"
                    radius="18"
                    color="#3B82F6"
                    opacity="0.35" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Layer id="blue_mood_layer">
    <Effect id="blue_hsla"
            type="hsla"
            hue="220"
            saturation="0.22"
            lightness="0.08"
            alpha="0.32" />
  </Layer>
  <Tex id="scene_tex" from="hello_scene" fmt="rgba16f" />
  <Tex id="blurred_scene" fmt="rgba16f" size={[128,72]} />
  <Pass id="soft_blur"
        effect="blur"
        in={["scene_tex"]}
        out={["blurred_scene"]}
        params={{ sigma: "1.0" }} />
  <Tex id="final" from="blue_mood_layer" input="blurred_scene" fmt="rgba16f" />
  <Present from="final" />
</Graph>
"##,
        )
        .expect("graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame");
        assert!(
            max_rgb(&rendered) > 20,
            "expected visible unified scene/layer output"
        );
    }

    #[test]
    fn scene_gpu_renderer_uses_background_as_scene_resource_in_unified_graph() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[32,18]} renderSize={[32,18]}>
  <Background color="#123456" />
  <Scene id="marker">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Circle x="16" y="9" radius="0" color="#ffffff" opacity="0" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Tex id="background_tex" from="scene" fmt="rgba16f" />
  <Present from="background_tex" />
</Graph>
"##,
        )
        .expect("graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if is_gpu_adapter_unavailable(&message) =>
            {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected GPU background resource render error: {err}"),
        };
        assert_eq!(*rendered.get_pixel(1, 1), Rgba([0x12, 0x34, 0x56, 0xff]));
    }

    #[test]
    fn scene_gpu_renderer_applies_graph_background_to_direct_present_scene() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[32,18]} renderSize={[32,18]}>
  <Background color="#f7f7f7" />
  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Circle x="16" y="9" radius="4" color="#000000" opacity="1" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if is_gpu_adapter_unavailable(&message) =>
            {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected direct present GPU render error: {err}"),
        };
        assert_eq!(*rendered.get_pixel(1, 1), Rgba([0xf7, 0xf7, 0xf7, 0xff]));
        assert_eq!(*rendered.get_pixel(16, 9), Rgba([0x00, 0x00, 0x00, 0xff]));
    }

    #[test]
    fn scene_renderer_applies_action_to_matching_character_part() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[160,120]}>
  <Background color="#ffffff" />

  <Action id="raise_arm" skeleton="humanoid_front_v1" duration="1s">
    <Pose t="0">
      <Bone id="upper_arm_r" rotation="0" />
    </Pose>
    <Pose t="0.5">
      <Bone id="upper_arm_r" rotation="-80" />
    </Pose>
    <Pose t="1">
      <Bone id="upper_arm_r" rotation="0" />
    </Pose>
  </Action>

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="hero" rig="humanoid_front_v1" x="80" y="60">
              <Part id="upper_arm_r" x="0" y="0">
                <Path d="M 0 0 L 0 42"
                      stroke="#000000"
                      strokeWidth="8"
                      lineCap="round"
                      fill="none" />
              </Part>
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <ApplyAction target="hero" action="raise_arm" at="0s" />
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let at_rest = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let raised = pollster::block_on(renderer.render_frame(&graph, 15)).expect("frame 15");
        assert_ne!(
            at_rest.as_raw(),
            raised.as_raw(),
            "action should change rendered pixels between poses"
        );
    }

    #[test]
    fn scene_renderer_applies_skeleton_parent_child_constraints() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[180,180]}>
  <Background color="#ffffff" />

  <Skeleton id="humanoid_front_v1">
    <Bone id="root" x="0" y="0" />
    <Bone id="upper_arm_r" parent="root" x="0" y="0" />
    <Bone id="forearm_r" parent="upper_arm_r" x="40" y="0" />
  </Skeleton>

  <Action id="bend" skeleton="humanoid_front_v1" duration="1s">
    <Pose t="0">
      <Bone id="upper_arm_r" rotation="0" />
    </Pose>
    <Pose t="0.5">
      <Bone id="upper_arm_r" rotation="90" />
    </Pose>
    <Pose t="1">
      <Bone id="upper_arm_r" rotation="0" />
    </Pose>
  </Action>

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="hero" rig="humanoid_front_v1" x="80" y="80">
              <Part id="forearm_marker" attachTo="forearm_r">
                <Circle x="0" y="0" radius="8" fill="#ff0000" />
              </Part>
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <ApplyAction target="hero" action="bend" at="0s" />
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let at_rest = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let bent = pollster::block_on(renderer.render_frame(&graph, 15)).expect("frame 15");
        let rest_pixel = at_rest.get_pixel(120, 80);
        let bent_pixel = bent.get_pixel(80, 120);
        assert!(
            rest_pixel[0] > 180 && rest_pixel[1] < 90 && rest_pixel[2] < 90,
            "expected red marker at parent-rest endpoint, got {rest_pixel:?}"
        );
        assert!(
            bent_pixel[0] > 180 && bent_pixel[1] < 90 && bent_pixel[2] < 90,
            "expected red marker to follow rotated parent endpoint, got {bent_pixel:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_two_bone_ik_target() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[180,180]}>
  <Background color="#ffffff" />

  <Skeleton id="arm_v1">
    <Bone id="upper" x="0" y="0" />
    <Bone id="lower" parent="upper" x="40" y="0" />
    <Bone id="hand" parent="lower" x="40" y="0" />
  </Skeleton>

  <Action id="reach" skeleton="arm_v1" duration="1s">
    <IK root="upper" mid="lower" end="hand" targetX="40" targetY="40" bend="1" />
  </Action>

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="hero" rig="arm_v1" x="80" y="80">
              <Part id="hand_marker" attachTo="hand">
                <Circle x="0" y="0" radius="8" fill="#ff0000" />
              </Part>
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <ApplyAction target="hero" action="reach" at="0s" />
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let target_pixel = rendered.get_pixel(120, 120);

        assert!(
            target_pixel[0] > 180 && target_pixel[1] < 90 && target_pixel[2] < 90,
            "expected IK end marker at target, got {target_pixel:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_chain_ik_target() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[180,180]}>
  <Background color="#ffffff" />

  <Skeleton id="finger_v1">
    <Bone id="finger_1" x="0" y="0" />
    <Bone id="finger_2" parent="finger_1" x="0" y="-40" />
    <Bone id="finger_3" parent="finger_2" x="0" y="-32" />
    <Bone id="finger_tip" parent="finger_3" x="0" y="-24" />
  </Skeleton>

  <Action id="touch" skeleton="finger_v1" duration="1s">
    <IK chain="finger_1,finger_2,finger_3,finger_tip"
        targetX="32" targetY="-66" iterations="12" weight="1" />
  </Action>

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="hand" rig="finger_v1" x="80" y="120">
              <Part id="tip_marker" attachTo="finger_tip">
                <Circle x="0" y="0" radius="8" fill="#ff0000" />
              </Part>
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <ApplyAction target="hand" action="touch" at="0s" />
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let target_pixel = rendered.get_pixel(112, 54);

        assert!(
            target_pixel[0] > 180 && target_pixel[1] < 90 && target_pixel[2] < 90,
            "expected chain IK end marker at target, got {target_pixel:?}"
        );
    }

    #[test]
    fn scene_renderer_draws_scene_group_shapes_and_text() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[220,140]}>
  <Background color="[1,1,1,1]" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="card" x="20" y="20" opacity="1">
              <Shadow x="0" y="10" blur="18" color="[0,0,0,0.24]" />
              <Rect width="100" height="58" radius="8" color="[0,0,1,1]" />
              <Circle x="22" y="29" radius="9" color="[1,0,0,1]" />
              <Text value="Card" x="40" y="16" width="50" fontSize="18" lineHeight="22" color="[0,0,0,1]" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let rect_pixel = rendered.get_pixel(112, 70);
        let circle_pixel = rendered.get_pixel(42, 49);

        assert!(
            rect_pixel[0] > 200 && rect_pixel[1] < 40 && rect_pixel[2] < 40,
            "expected red rect pixel, got {rect_pixel:?}"
        );
        assert!(
            circle_pixel[2] > 200 && circle_pixel[0] < 40 && circle_pixel[1] < 40,
            "expected blue circle pixel, got {circle_pixel:?}"
        );
    }

    #[test]
    fn scene_renderer_fits_render_size_into_output_size() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]} renderSize={[200,100]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Circle x="50" y="50" radius="10" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let center = rendered.get_pixel(100, 50);
        let left_logical_position = rendered.get_pixel(50, 50);

        assert!(
            center[0] > 200 && center[1] < 40 && center[2] < 40,
            "expected renderSize content to be centered and scaled into output, got {center:?}"
        );
        assert!(
            left_logical_position[0] < 40
                && left_logical_position[1] < 40
                && left_logical_position[2] < 40,
            "expected untransformed logical position to be background, got {left_logical_position:?}"
        );
    }

    #[test]
    fn scene_renderer_draws_trimmed_polyline_and_path() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[160,90]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Polyline points="10,24 110,24"
                      stroke="#2f83ff"
                      strokeWidth="8"
                      trimStart="0"
                      trimEnd="0.5" />
            <Path d="M 10 66 L 110 66"
                  stroke="#ff2f2f"
                  strokeWidth="8"
                  trimStart="0.5"
                  trimEnd="1" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let polyline_drawn = rendered.get_pixel(50, 24);
        let polyline_trimmed = rendered.get_pixel(95, 24);
        let path_trimmed = rendered.get_pixel(25, 66);
        let path_drawn = rendered.get_pixel(90, 66);

        assert!(
            polyline_drawn[2] > 180 && polyline_drawn[0] < 100,
            "expected blue polyline pixel, got {polyline_drawn:?}"
        );
        assert!(
            polyline_trimmed[0] < 20 && polyline_trimmed[1] < 20 && polyline_trimmed[2] < 20,
            "expected trimmed polyline tail to stay black, got {polyline_trimmed:?}"
        );
        assert!(
            path_trimmed[0] < 20 && path_trimmed[1] < 20 && path_trimmed[2] < 20,
            "expected trimmed path head to stay black, got {path_trimmed:?}"
        );
        assert!(
            path_drawn[0] > 180 && path_drawn[2] < 100,
            "expected red path pixel, got {path_drawn:?}"
        );
    }

    #[test]
    fn scene_renderer_draws_brush_part_path() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[120,80]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Defs>
      <Brush id="red_ink" stroke="#ff0000" strokeWidth="6" opacity="1" />
    </Defs>

    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Part id="mark" brush="red_ink" x="20" y="30">
              <Path id="mark_line" d="M 0 0 L 80 0" />
            </Part>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let pixel = rendered.get_pixel(60, 30);

        assert!(
            pixel[0] > 180 && pixel[1] < 80 && pixel[2] < 80,
            "expected red brush path pixel, got {pixel:?}"
        );
    }

    #[test]
    fn scene_renderer_draws_character_vector_nodes() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[120,90]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="face" x="60" y="45" scale="1.5">
              <Path d="M -30 0 C -12 -20 12 -20 30 0"
                    stroke="#ff77aa"
                    strokeWidth="8" />
              <Line x1="-20" y1="12" x2="20" y2="12" width="6" color="#00ff00" />
              <Circle x="0" y="-5" radius="6" color="#ffffff" />
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let eye = rendered.get_pixel(60, 37);
        let line = rendered.get_pixel(60, 63);

        assert!(
            eye[0] > 200 && eye[1] > 200 && eye[2] > 200,
            "expected white character circle, got {eye:?}"
        );
        assert!(
            line[1] > 180 && line[0] < 80 && line[2] < 80,
            "expected green character line, got {line:?}"
        );
    }

    #[test]
    fn scene_gpu_renderer_draws_character_overlay() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,80]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character x="50" y="40">
              <Path d="M -24 0 L 24 0" stroke="#00ff00" strokeWidth="8" />
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU character overlay test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };
        let center = rendered.get_pixel(50, 40);

        assert!(
            center[1] > 180 && center[0] < 80 && center[2] < 80,
            "expected green GPU character path pixel, got {center:?}"
        );
    }

    #[test]
    fn scene_gpu_renderer_draws_filled_path_overlay() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,80]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Path d="M 20 20 L 70 20 L 70 60 L 20 60 Z"
                  fill="#00ff00"
                  stroke="none" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU filled path overlay test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };
        let center = rendered.get_pixel(45, 40);

        assert!(
            center[1] > 180 && center[0] < 80 && center[2] < 80,
            "expected green GPU filled path overlay pixel, got {center:?}"
        );
    }

    #[test]
    fn scene_gpu_renderer_draws_sketch_stroke_style() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,80]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Line x1="12" y1="40" x2="88" y2="40"
                  width="8"
                  color="#00ff00"
                  strokeStyle="sketch"
                  strokeRoughness="1.4"
                  strokeCopies="4"
                  strokeTexture="0.35"
                  strokeBristles="3"
                  strokePressure="auto"
                  strokePressureMin="0.4"
                  strokePressureCurve="1.2" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU sketch stroke style test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };
        let center = rendered.get_pixel(50, 40);

        assert!(
            center[1] > 150 && center[0] < 100 && center[2] < 100,
            "expected green GPU sketch line pixel, got {center:?}"
        );
    }

    #[test]
    fn scene_renderer_draws_filled_path_and_mask() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[120,90]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Path d="M 10 10 L 45 10 L 45 45 L 10 45 Z"
                  fill="#00ff00"
                  stroke="none" />
            <Mask shape="circle" x="88" y="42" radius="18">
              <Rect x="65" y="20" width="46" height="46" color="#ff0000" />
            </Mask>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let filled = rendered.get_pixel(28, 28);
        let masked_inside = rendered.get_pixel(88, 42);
        let masked_outside = rendered.get_pixel(66, 22);

        assert!(
            filled[1] > 180 && filled[0] < 80 && filled[2] < 80,
            "expected green filled path pixel, got {filled:?}"
        );
        assert!(
            masked_inside[0] > 180 && masked_inside[1] < 80 && masked_inside[2] < 80,
            "expected red masked center, got {masked_inside:?}"
        );
        assert!(
            masked_outside[0] < 40 && masked_outside[1] < 40 && masked_outside[2] < 40,
            "expected masked corner to stay black, got {masked_outside:?}"
        );
    }

    #[test]
    fn scene_renderer_uses_nonzero_fill_rule_for_compound_path_holes() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[80,80]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Path d="M 10 10 L 70 10 L 70 70 L 10 70 Z M 30 30 L 30 50 L 50 50 L 50 30 Z"
                  fill="#ff0000"
                  stroke="none" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let ring = rendered.get_pixel(20, 20);
        let hole = rendered.get_pixel(40, 40);

        assert!(
            ring[0] > 180 && ring[1] < 80 && ring[2] < 80,
            "expected red compound path ring, got {ring:?}"
        );
        assert!(
            hole[0] < 40 && hole[1] < 40 && hole[2] < 40,
            "expected compound path hole to stay black, got {hole:?}"
        );
    }

    #[test]
    fn scene_renderer_character_mask_clips_filled_path() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[120,90]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character x="60" y="45">
              <Mask shape="circle" x="0" y="0" radius="18">
                <Path d="M -30 -30 L 30 -30 L 30 30 L -30 30 Z"
                      fill="#ff77aa"
                      stroke="none" />
              </Mask>
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let inside = rendered.get_pixel(60, 45);
        let outside = rendered.get_pixel(35, 45);

        assert!(
            inside[0] > 180 && inside[2] > 100,
            "expected pink character mask fill, got {inside:?}"
        );
        assert!(
            outside[0] < 40 && outside[1] < 40 && outside[2] < 40,
            "expected clipped outside pixel to stay black, got {outside:?}"
        );
    }

    #[test]
    fn scene_renderer_camera_centers_world_coordinate() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,80]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track role="camera" z="1000">
        <Sequence duration="1s" out="hold">
          <Camera x="100" y="40" zoom="1" />
        </Sequence>
      </Track>
      <Track space="world" z="0">
        <Sequence duration="1s" out="hold">
          <Layer>
            <Circle x="100" y="40" radius="8" color="#00ff00" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let centered = rendered.get_pixel(50, 40);

        assert!(
            centered[1] > 180 && centered[0] < 80 && centered[2] < 80,
            "expected camera-centered green circle, got {centered:?}"
        );
    }

    #[test]
    fn scene_renderer_camera_follow_maps_node_to_anchor() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,80]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track role="camera" z="1000">
        <Sequence duration="1s" out="hold">
          <Camera follow="marker" anchorX="25%" anchorY="75%" zoom="1" worldBounds="0,0,200,160" />
        </Sequence>
      </Track>
      <Track space="world" z="0">
        <Sequence duration="1s" out="hold">
          <Layer>
            <Circle id="marker" x="120" y="60" radius="8" color="#00ff00" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let anchored = rendered.get_pixel(25, 60);

        assert!(
            anchored[1] > 180 && anchored[0] < 80 && anchored[2] < 80,
            "expected camera-followed green circle at anchor, got {anchored:?}"
        );
    }

    #[test]
    fn scene_renderer_higher_track_z_paints_later() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[80,80]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="back" space="world" z="0">
        <Sequence duration="1s" out="hold">
          <Layer>
            <Rect x="16" y="16" width="40" height="40" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
      <Track id="front" space="world" z="10">
        <Sequence duration="1s" out="hold">
          <Layer>
            <Rect x="24" y="24" width="40" height="40" color="#00ff00" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let overlap = rendered.get_pixel(30, 30);

        assert!(
            overlap[1] > 180 && overlap[0] < 80 && overlap[2] < 80,
            "expected higher-z green rect to paint over lower-z red rect, got {overlap:?}"
        );
    }

    #[test]
    fn scene_renderer_layer_z_depth_sorts_far_before_near() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track space="world" z="0">
        <Sequence duration="1s" out="hold">
          <Layer zDepth="-1">
            <Rect x="35" y="35" width="30" height="30" color="#00ff00" />
          </Layer>
          <Layer zDepth="2">
            <Rect x="30" y="30" width="40" height="40" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let overlap = rendered.get_pixel(50, 50);

        assert!(
            overlap[1] > 180 && overlap[0] < 80 && overlap[2] < 80,
            "expected near zDepth=-1 green layer to paint over far zDepth=2 red layer, got {overlap:?}"
        );
    }

    #[test]
    fn scene_gpu_renderer_layer_z_depth_sorts_far_before_near() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track space="world" z="0">
        <Sequence duration="1s" out="hold">
          <Layer zDepth="-1">
            <Rect x="35" y="35" width="30" height="30" color="#00ff00" />
          </Layer>
          <Layer zDepth="2">
            <Rect x="30" y="30" width="40" height="40" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU zDepth sort test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };
        let overlap = rendered.get_pixel(50, 50);

        assert!(
            overlap[1] > 180 && overlap[0] < 80 && overlap[2] < 80,
            "expected GPU near zDepth=-1 green layer to paint over far zDepth=2 red layer, got {overlap:?}"
        );
    }

    #[test]
    fn scene_renderer_layer_z_depth_overrides_track_z_depth() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track space="world" z="0" zDepth="2">
        <Sequence duration="1s" out="hold">
          <Layer zDepth="-1">
            <Rect x="35" y="35" width="30" height="30" color="#00ff00" />
          </Layer>
          <Layer>
            <Rect x="30" y="30" width="40" height="40" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let overlap = rendered.get_pixel(50, 50);

        assert!(
            overlap[1] > 180 && overlap[0] < 80 && overlap[2] < 80,
            "expected Layer zDepth=-1 to override Track zDepth=2 and paint green over inherited-depth red, got {overlap:?}"
        );
    }

    #[test]
    fn scene_renderer_track_z_depth_sorts_world_tracks() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="near" space="world" z="0" zDepth="-1">
        <Sequence duration="1s" out="hold">
          <Layer>
            <Rect x="35" y="35" width="30" height="30" color="#00ff00" />
          </Layer>
        </Sequence>
      </Track>
      <Track id="far" space="world" z="10" zDepth="2">
        <Sequence duration="1s" out="hold">
          <Layer>
            <Rect x="30" y="30" width="40" height="40" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let overlap = rendered.get_pixel(50, 50);

        assert!(
            overlap[1] > 180 && overlap[0] < 80 && overlap[2] < 80,
            "expected near zDepth=-1 track to paint over higher-z far zDepth=2 track, got {overlap:?}"
        );
    }

    #[test]
    fn scene_renderer_accepts_scene_tex_pass_present_pipeline() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="0" y="0" width="64" height="48" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Tex id="src" fmt="rgba16f" from="scene:scene0" />
  <Tex id="out" fmt="rgba16f" size={[64,48]} />
  <Pass id="fx_opacity" kind="compute"
        effect="opacity"
        in={["src"]} out={["out"]}
        params={{ opacity: "0.5" }} />
  <Present from="out" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let pixel = rendered.get_pixel(10, 10);

        assert!(
            pixel[0] > 200 && pixel[1] < 40 && pixel[2] < 40,
            "expected red scene pixel, got {pixel:?}"
        );
        assert!(
            (120..=136).contains(&pixel[3]),
            "expected opacity pass to halve alpha, got {pixel:?}"
        );
    }

    #[test]
    fn scene_renderer_limits_process_effect_with_luma_mask() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#00000000" />
  <Scene id="source_scene">
    <Timeline>
      <Track id="main" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="0" y="0" width="64" height="48" color="#202020" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Scene id="mask_scene">
    <Timeline>
      <Track id="mask" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="32" y="0" width="32" height="48" color="#ffffff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Process id="masked_post">
    <Tex id="src" fmt="rgba16f" from="scene:source_scene" />
    <Tex id="mask_tex" fmt="rgba16f" from="scene:mask_scene" />
    <Tex id="out" fmt="rgba16f" size={[64,48]} />
    <Pass id="masked_brightness" kind="compute"
          effect="brightness"
          in={["src"]} out={["out"]}
          mask="mask_tex"
          maskMode="luma"
          params={{ brightness: "0.5" }} />
  </Process>
  <Present from="masked_post" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let pass = &graph.passes[0];
        assert_eq!(pass.mask.as_deref(), Some("mask_tex"));
        assert_eq!(pass.mask_mode, "luma");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let left = rendered.get_pixel(12, 24);
        let right = rendered.get_pixel(48, 24);

        assert!(
            (28..=36).contains(&left[0]) && left[3] > 240,
            "expected left pixel to keep original source, got {left:?}"
        );
        assert!(
            right[0] > left[0] + 90 && right[3] > 240,
            "expected right pixel to receive masked brightness, got left={left:?} right={right:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_referenced_process_effect_and_param_override() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[32,24]}>
  <Background color="#00000000" />
  <Process id="fx_brightness">
    <Input id="effect_input" type="video" />
    <Tex id="effect_src" fmt="rgba16f" from="input:effect_input" />
    <Tex id="effect_out" fmt="rgba16f" size={[32,24]} />
    <Pass id="brightness_pass" kind="compute"
          effect="brightness"
          in={["effect_src"]} out={["effect_out"]}
          params={{ brightness: "0.0" }} />
  </Process>
  <Scene id="graded_scene">
    <Timeline>
      <Track id="main" space="screen">
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="32" height="24" color="#202020" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
    <Effects>
      <Effect process="fx_brightness">
        <Param name="brightness" value="0.5" />
      </Effect>
    </Effects>
  </Scene>
  <Present from="graded_scene" />
</Graph>
"##,
        )
        .expect("scene Process reference should parse");

        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0))
            .expect("scene Process reference should render");
        let pixel = rendered.get_pixel(12, 12);
        assert!(
            pixel[0] > 120 && pixel[1] > 120 && pixel[2] > 120,
            "expected Scene Effect Param to override brightness, got {pixel:?}"
        );
    }

    #[test]
    fn scene_renderer_color_to_alpha_keys_background_after_process_pass() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#FFFFFF" />
  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="20" y="14" width="24" height="20" radius="4" color="#101827" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Process id="keyed">
    <Tex id="src" fmt="rgba16f" from="scene:scene0" />
    <Tex id="out" fmt="rgba16f" size={[64,48]} />
    <Pass id="key_color" kind="compute"
          effect="color_to_alpha"
          in={["src"]} out={["out"]}
          params={{ color: "#FFFFFF", tolerance: "0.02", softness: "0.12" }} />
  </Process>
  <Present from="keyed" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let bg = rendered.get_pixel(4, 4);
        let center = rendered.get_pixel(32, 24);

        assert!(
            bg[3] < 8,
            "expected white background to key out, got {bg:?}"
        );
        assert!(
            center[3] > 240 && center[0] < 40 && center[1] < 50 && center[2] < 70,
            "expected non-white center to remain opaque, got {center:?}"
        );
    }

    #[test]
    fn scene_gpu_renderer_draws_scene_group_shapes_and_text() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[220,140]}>
  <Background color="[1,1,1,1]" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="card" x="20" y="20" opacity="1">
              <Shadow x="0" y="10" blur="18" color="[0,0,0,0.24]" />
              <Rect width="100" height="58" radius="8" color="[0,0,1,1]" />
              <Circle x="22" y="29" radius="9" color="[1,0,0,1]" />
              <Text value="Card" x="40" y="16" width="50" fontSize="18" lineHeight="22" color="[0,0,0,1]" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU scene primitive test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };
        let rect_pixel = rendered.get_pixel(112, 70);
        let circle_pixel = rendered.get_pixel(42, 49);

        assert!(
            rect_pixel[0] > 200 && rect_pixel[1] < 40 && rect_pixel[2] < 40,
            "expected red rect pixel, got {rect_pixel:?}"
        );
        assert!(
            circle_pixel[2] > 200 && circle_pixel[0] < 40 && circle_pixel[1] < 40,
            "expected blue circle pixel, got {circle_pixel:?}"
        );
    }

    #[test]
    fn scene_gpu_renderer_draws_text_inside_process_scene_source() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="3s" size={[800,450]} renderSize={[800,450]}>
  <Background color="#FFFFFF" />

  <Scene id="ProcessBasicSource">
    <Timeline>
      <Track id="main" space="world" z="0">
        <Sequence from="0s" duration="3s" out="hold">
          <Layer>
            <Rect x="0" y="0" width="800" height="450" color="#FFFFFF" />
            <Text x="52" y="64" value="Process effect: opacity" fontSize="32" color="#111827" />
            <Text x="52" y="98" value="Standalone scene source for the basic Process pass" fontSize="20" color="#6B7280" />
            <Rect x="250" y="154" width="300" height="142" radius="32" color="#111827" opacity="1" />
            <Circle x="400" y="225" radius="38" color="#FFFFFF" opacity="1" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <Process id="ProcessBasic">
    <Tex id="src" fmt="rgba16f" from="scene:ProcessBasicSource" />
    <Tex id="out" fmt="rgba16f" size={[800,450]} />
    <Pass id="fx_opacity" kind="compute" effect="opacity"
          in={["src"]} out={["out"]}
          params={{ opacity: "1.0" }} />
  </Process>

  <Present from="ProcessBasic" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if message.contains("GPU adapter")
                    || message.contains("graphics adapter")
                    || message.contains("metal found no adapters") =>
            {
                eprintln!("Skipping GPU process text source test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };
        let mut text_dark_pixels = 0usize;
        for y in 48..118 {
            for x in 48..500 {
                let pixel = rendered.get_pixel(x, y);
                if pixel[0] < 170 && pixel[1] < 180 && pixel[2] < 190 && pixel[3] > 20 {
                    text_dark_pixels += 1;
                }
            }
        }
        assert!(
            text_dark_pixels > 40,
            "expected GPU-rendered text pixels in the scene source, got {text_dark_pixels}"
        );
    }

    #[test]
    fn scene_gpu_renderer_draws_screen_blend_rect() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[96,64]}>
  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="0" y="0" width="96" height="64" color="#0000ff" />
            <Rect x="20" y="12" width="56" height="40" color="#ff0000" opacity="0.5" blend="screen" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU screen blend rect test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };
        let pixel = rendered.get_pixel(48, 32);
        assert!(
            pixel[0] > 90 && pixel[2] > 200,
            "expected screen-blended magenta/blue pixel, got {pixel:?}"
        );
    }

    #[test]
    fn scene_gpu_renderer_batches_many_gradient_paths() {
        let mut script = String::from(
            r##"
<Graph fps={30} duration="1s" size={[240,160]}>
  <Background color="#ffffff" />

  <Scene id="scene0">
    <Defs>
      <LinearGradient id="sclera_soft" x1="0" y1="0" x2="0" y2="1"
                      stops="0:#D3CEE6, 0.30:#F7F7F7, 0.70:#F7F7F7, 1:#ffffff" />
      <LinearGradient id="lid_grey" x1="0" y1="0" x2="0" y2="1"
                      stops="0:#B7C7C7, 0.70:#B7C7C7, 1:#8da0a3" />
    </Defs>
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
"##,
        );
        for ix in 0..72 {
            let x = 18 + (ix % 12) * 17;
            let y = 20 + (ix / 12) * 18;
            script.push_str(&format!(
                r##"    <Path d="M {x} {y} C {cx1} {cy1} {cx2} {cy2} {x2} {y2} C {cx3} {cy3} {cx4} {cy4} {x} {y} Z"
          fill="url(#sclera_soft)" stroke="#3F5877" strokeWidth="1.2" opacity="0.78" />
"##,
                cx1 = x + 8,
                cy1 = y - 8,
                cx2 = x + 28,
                cy2 = y - 8,
                x2 = x + 36,
                y2 = y,
                cx3 = x + 28,
                cy3 = y + 12,
                cx4 = x + 8,
                cy4 = y + 12,
            ));
            script.push_str(&format!(
                r##"    <Path d="M {x} {line_y} C {cx1} {cy1} {cx2} {cy2} {x2} {line_y}"
          fill="none" stroke="#3F5877" strokeWidth="2.4" lineCap="round" />
"##,
                line_y = y + 6,
                cx1 = x + 8,
                cy1 = y - 5,
                cx2 = x + 28,
                cy2 = y - 5,
                x2 = x + 36,
            ));
        }
        script.push_str(
            r##"          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        );

        let graph = parse_graph_script(&script).expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU batched path stress test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };

        assert!(
            rendered
                .pixels()
                .any(|pixel| pixel[0] < 120 && pixel[1] < 140 && pixel[2] < 170),
            "expected batched gradient paths to render visible dark strokes"
        );
    }

    #[test]
    fn scene_gpu_renderer_applies_scene_blur_post_pass() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[80,40]}>
  <Background color="#ffffff" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="35" y="10" width="10" height="20" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Tex id="src" fmt="rgba16f" from="scene0" />
  <Tex id="out" fmt="rgba16f" size={[80,40]} />
  <Pass id="blur_h" kind="compute"
        effect="gaussian_5tap_h"
        in={["src"]} out={["out"]}
        params={{ sigma: "8" }} />
  <Present from="out" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU scene post-pass test: {message}");
                return;
            }
            Err(err) => panic!("unexpected render error: {err}"),
        };
        let softened_edge = rendered.get_pixel(31, 20);

        assert!(
            softened_edge[0] < 245 && softened_edge[1] < 245 && softened_edge[2] < 245,
            "expected GPU blur to soften edge pixel, got {softened_edge:?}"
        );
    }

    #[test]
    fn scene_renderer_applies_process_bloom_to_whole_scene() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[96,48]}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="42" y="18" width="12" height="12" color="#ffffff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Process id="final_glow">
    <Tex id="src" fmt="rgba16f" from="scene0" />
    <Tex id="out" fmt="rgba16f" size={[96,48]} />
    <Pass id="post_bloom" kind="compute"
          effect="bloom"
          in={["src"]} out={["out"]}
          params={{ threshold: "0.2", intensity: "2.0", sigma: "8.0" }} />
  </Process>
  <Present from="final_glow" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("CPU render");
        let halo = rendered.get_pixel(37, 24);

        assert!(
            halo[0] > 0 || halo[1] > 0 || halo[2] > 0,
            "expected bloom to brighten pixels outside the source rect, got {halo:?}"
        );
    }

    fn write_test_svg(name: &str, body: &str) -> PathBuf {
        let svg_path = std::env::temp_dir().join(format!(
            "anica-motionloom-scene-{name}-{}.svg",
            std::process::id()
        ));
        std::fs::write(&svg_path, body).expect("write test svg");
        svg_path
    }

    #[test]
    fn scene_text_opacity_fades_in_over_time() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="3s" size={[640,360]}>
  <Background color="#000000" />
  <Text value="hello world"
        x="center"
        y="center"
        fontSize="72"
        color="#ffffff"
        opacity="min($time.sec / 1.0, 1.0)" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let at_zero = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let at_half = pollster::block_on(renderer.render_frame(&graph, 15)).expect("frame 15");
        let at_full = pollster::block_on(renderer.render_frame(&graph, 30)).expect("frame 30");

        assert_eq!(max_rgb(&at_zero), 0);
        assert!(max_rgb(&at_half) > 40);
        assert!(max_rgb(&at_half) < max_rgb(&at_full));
        assert!(max_rgb(&at_full) > 200);
    }

    #[test]
    fn scene_image_draws_exact_path_with_scale_and_opacity() {
        let image_path = std::env::temp_dir().join(format!(
            "anica-motionloom-scene-image-{}.png",
            std::process::id()
        ));
        RgbaImage::from_pixel(8, 6, Rgba([255, 0, 0, 255]))
            .save(&image_path)
            .expect("write test image");

        let graph = parse_graph_script(&format!(
            r##"
<Graph fps={{30}} duration="1s" size={{[64,48]}}>
  <Background color="#000000" />
  <Image src="{}"
         x="10"
         y="12"
         scale="2.0"
         opacity="0.5" />
  <Present from="scene" />
</Graph>
"##,
            image_path.to_string_lossy()
        ))
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let inside = rendered.get_pixel(12, 14);
        let outside = rendered.get_pixel(2, 2);

        assert!(inside[0] > 100, "expected red image pixel, got {inside:?}");
        assert_eq!(inside[1], 0);
        assert_eq!(inside[2], 0);
        assert_eq!(outside[0], 0);

        let _ = std::fs::remove_file(image_path);
    }

    #[test]
    fn scene_character_part_draws_image_child() {
        let image_path = std::env::temp_dir().join(format!(
            "anica-motionloom-scene-character-image-{}.png",
            std::process::id()
        ));
        RgbaImage::from_pixel(8, 6, Rgba([255, 0, 0, 255]))
            .save(&image_path)
            .expect("write test image");

        let graph = parse_graph_script(&format!(
            r##"
<Graph fps={{30}} duration="1s" size={{[64,48]}}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="test_character" x="20" y="10">
              <Part id="image_part" x="4" y="3">
                <Image src="{}" x="0" y="0" scale="2.0" opacity="1.0" />
              </Part>
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <Present from="scene0" />
</Graph>
"##,
            image_path.to_string_lossy()
        ))
        .expect("scene graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let inside = rendered.get_pixel(25, 14);
        let outside = rendered.get_pixel(2, 2);

        assert!(
            inside[0] > 200 && inside[1] < 40 && inside[2] < 40,
            "expected transformed character image pixel, got {inside:?}"
        );
        assert_eq!(outside[0], 0);

        let _ = std::fs::remove_file(image_path);
    }

    #[test]
    fn scene_character_draws_data_uri_image_source() {
        let mut bytes = Vec::new();
        RgbaImage::from_pixel(8, 6, Rgba([255, 0, 0, 255]))
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode test image");
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);

        let graph = parse_graph_script(&format!(
            r##"
<Graph fps={{30}} duration="1s" size={{[64,48]}}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="test_character"
                       src="data:image/png;base64,{}"
                       x="20"
                       y="10"
                       scale="2.0" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <Present from="scene0" />
</Graph>
"##,
            encoded
        ))
        .expect("scene graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let inside = rendered.get_pixel(25, 14);
        let outside = rendered.get_pixel(2, 2);

        assert!(
            inside[0] > 200 && inside[1] < 40 && inside[2] < 40,
            "expected character data URI image pixel, got {inside:?}"
        );
        assert_eq!(outside[0], 0);
    }

    #[test]
    fn scene_character_part_draws_image_child_gpu_profile() {
        let image_path = std::env::temp_dir().join(format!(
            "anica-motionloom-scene-character-image-gpu-{}.png",
            std::process::id()
        ));
        RgbaImage::from_pixel(8, 6, Rgba([255, 0, 0, 255]))
            .save(&image_path)
            .expect("write test image");

        let graph = parse_graph_script(&format!(
            r##"
<Graph fps={{30}} duration="1s" size={{[64,48]}}>
  <Background color="#000000" />

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="test_character" x="20" y="10">
              <Part id="image_part" x="4" y="3">
                <Image src="{}" x="0" y="0" scale="2.0" opacity="1.0" />
              </Part>
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <Present from="scene0" />
</Graph>
"##,
            image_path.to_string_lossy()
        ))
        .expect("scene graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => {
                let inside = rendered.get_pixel(25, 14);
                let outside = rendered.get_pixel(2, 2);
                assert!(
                    inside[0] > 200 && inside[1] < 40 && inside[2] < 40,
                    "expected transformed character image pixel, got {inside:?}"
                );
                assert_eq!(outside[0], 0);
            }
            Err(MotionLoomSceneRenderError::GpuRender { message })
                if is_gpu_adapter_unavailable(&message) =>
            {
                skip_gpu_texture_test(message);
            }
            Err(err) => panic!("GPU character image render failed: {err}"),
        }

        let _ = std::fs::remove_file(image_path);
    }

    #[test]
    fn scene_svg_draws_exact_path_with_scale_and_opacity() {
        let svg_path = write_test_svg(
            "svg",
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="8" height="6" viewBox="0 0 8 6">
  <rect width="8" height="6" fill="#ff0000"/>
</svg>"##,
        );

        let graph = parse_graph_script(&format!(
            r##"
<Graph fps={{30}} duration="1s" size={{[64,48]}}>
  <Background color="#000000" />
  <Svg src="{}"
       x="10"
       y="12"
       scale="2.0"
       opacity="0.5" />
  <Present from="scene" />
</Graph>
"##,
            svg_path.to_string_lossy()
        ))
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let inside = rendered.get_pixel(12, 14);
        let outside = rendered.get_pixel(2, 2);

        assert!(inside[0] > 100, "expected red SVG pixel, got {inside:?}");
        assert_eq!(inside[1], 0);
        assert_eq!(inside[2], 0);
        assert_eq!(outside[0], 0);

        let _ = std::fs::remove_file(svg_path);
    }

    #[test]
    fn scene_svg_data_uri_utf8_renders() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#000000" />
  <Svg src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='8' height='6' viewBox='0 0 8 6'><rect width='8' height='6' fill='%23ff0000'/></svg>"
       x="10"
       y="12"
       scale="2.0"
       opacity="1.0" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");
        let mut renderer = pollster::block_on(SceneFrameRenderer::new());
        let rendered = pollster::block_on(renderer.render_frame(&graph, 0)).expect("frame 0");
        let inside = rendered.get_pixel(12, 14);
        assert!(
            inside[0] > 200,
            "expected red SVG data URI pixel, got {inside:?}"
        );
    }

    #[test]
    fn scene_gpu_renderer_draws_image_when_available() {
        let image_path = std::env::temp_dir().join(format!(
            "anica-motionloom-scene-gpu-image-{}.png",
            std::process::id()
        ));
        RgbaImage::from_pixel(8, 6, Rgba([0, 255, 0, 255]))
            .save(&image_path)
            .expect("write test image");

        let graph = parse_graph_script(&format!(
            r##"
<Graph fps={{30}} duration="1s" size={{[64,48]}}>
  <Background color="#000000" />
  <Image src="{}"
         x="10"
         y="12"
         scale="2.0"
         opacity="1.0" />
  <Present from="scene" />
</Graph>
"##,
            image_path.to_string_lossy()
        ))
        .expect("scene graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU render test: {message}");
                let _ = std::fs::remove_file(image_path);
                return;
            }
            Err(err) => panic!("unexpected GPU render error: {err}"),
        };
        let inside = rendered.get_pixel(12, 14);
        assert!(
            inside[1] > 150,
            "expected green GPU image pixel, got {inside:?}"
        );

        let _ = std::fs::remove_file(image_path);
    }

    #[test]
    fn scene_gpu_renderer_draws_svg_when_available() {
        let svg_path = write_test_svg(
            "gpu-svg",
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="8" height="6" viewBox="0 0 8 6">
  <rect width="8" height="6" fill="#00ff00"/>
</svg>"##,
        );

        let graph = parse_graph_script(&format!(
            r##"
<Graph fps={{30}} duration="1s" size={{[64,48]}}>
  <Background color="#000000" />
  <Svg src="{}"
       x="10"
       y="12"
       scale="2.0"
       opacity="1.0" />
  <Present from="scene" />
</Graph>
"##,
            svg_path.to_string_lossy()
        ))
        .expect("scene graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU SVG render test: {message}");
                let _ = std::fs::remove_file(svg_path);
                return;
            }
            Err(err) => panic!("unexpected GPU render error: {err}"),
        };
        let inside = rendered.get_pixel(12, 14);
        assert!(
            inside[1] > 150,
            "expected green GPU SVG pixel, got {inside:?}"
        );

        let _ = std::fs::remove_file(svg_path);
    }

    #[test]
    fn scene_gpu_prores_renderer_draws_image_when_available() {
        let image_path = std::env::temp_dir().join(format!(
            "anica-motionloom-scene-gpu-prores-image-{}.png",
            std::process::id()
        ));
        RgbaImage::from_pixel(8, 6, Rgba([0, 0, 255, 255]))
            .save(&image_path)
            .expect("write test image");

        let graph = parse_graph_script(&format!(
            r##"
<Graph fps={{30}} duration="1s" size={{[64,48]}}>
  <Background color="#000000" />
  <Image src="{}"
         x="10"
         y="12"
         scale="2.0"
         opacity="1.0" />
  <Present from="scene" />
</Graph>
"##,
            image_path.to_string_lossy()
        ))
        .expect("scene graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU render test: {message}");
                let _ = std::fs::remove_file(image_path);
                return;
            }
            Err(err) => panic!("unexpected GPU render error: {err}"),
        };
        let inside = rendered.get_pixel(12, 14);
        assert!(
            inside[2] > 150,
            "expected blue GPU image pixel, got {inside:?}"
        );

        let _ = std::fs::remove_file(image_path);
    }

    #[test]
    fn scene_gpu_renderer_composites_multiple_images_when_available() {
        let image1_path = std::env::temp_dir().join(format!(
            "anica-motionloom-scene-gpu-multi-1-{}.png",
            std::process::id()
        ));
        let image2_path = std::env::temp_dir().join(format!(
            "anica-motionloom-scene-gpu-multi-2-{}.png",
            std::process::id()
        ));
        RgbaImage::from_pixel(8, 8, Rgba([255, 0, 0, 255]))
            .save(&image1_path)
            .expect("write test image 1");
        RgbaImage::from_pixel(8, 8, Rgba([0, 255, 0, 255]))
            .save(&image2_path)
            .expect("write test image 2");

        let graph = parse_graph_script(&format!(
            r##"
<Graph fps={{30}} duration="1s" size={{[64,48]}}>
  <Background color="#000000" />
  <Image src="{}" x="4" y="10" scale="1.0" opacity="1.0" />
  <Image src="{}" x="24" y="10" scale="1.0" opacity="1.0" />
  <Present from="scene" />
</Graph>
"##,
            image1_path.to_string_lossy(),
            image2_path.to_string_lossy()
        ))
        .expect("scene graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU render test: {message}");
                let _ = std::fs::remove_file(image1_path);
                let _ = std::fs::remove_file(image2_path);
                return;
            }
            Err(err) => panic!("unexpected GPU render error: {err}"),
        };
        let first = rendered.get_pixel(5, 11);
        let second = rendered.get_pixel(25, 11);

        assert!(
            first[0] > 200 && first[1] < 30,
            "expected first GPU image to remain visible, got {first:?}"
        );
        assert!(
            second[1] > 200 && second[0] < 30,
            "expected second GPU image to remain visible, got {second:?}"
        );

        let _ = std::fs::remove_file(image1_path);
        let _ = std::fs::remove_file(image2_path);
    }

    #[test]
    fn scene_gpu_puppet_preserves_vector_source_order_when_available() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#FFFFFF" />
  <Scene id="puppet_order">
    <Timeline>
      <Track id="content" space="screen" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Group id="hair">
              <Rect x="0" y="0" width="64" height="48" color="#FF0000" />
            </Group>
            <PuppetWarp id="hair_warp" target="hair" width="64" height="48" density="high">
              <PuppetPin id="tip" x="16" y="16" targetX="17" targetY="16" radius="48" />
            </PuppetWarp>
            <Rect x="20" y="10" width="24" height="28" color="#0000FF" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="puppet_order" />
</Graph>
"##,
        )
        .expect("puppet source-order graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Gpu));
        let rendered = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(rendered) => rendered,
            Err(MotionLoomSceneRenderError::GpuRender { message }) => {
                eprintln!("Skipping GPU render test: {message}");
                return;
            }
            Err(err) => panic!("unexpected GPU render error: {err}"),
        };
        let face = rendered.get_pixel(32, 24);
        assert!(
            face[2] > 200 && face[0] < 30,
            "expected the later blue vector to remain above the Puppet texture, got {face:?}"
        );
    }

    #[test]
    fn scene_cpu_universal_puppet_captures_once_and_keeps_later_overlay() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#000000" />
  <Scene id="universal_order">
    <Timeline>
      <Track id="content" space="screen" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="0" y="0" width="32" height="48"
                  color="#FF0000" opacity="0.5" />
            <PuppetWarp id="layer_warp" target="@layer" capture="before"
                        width="64" height="48" density="high">
              <PuppetPin id="anchor" x="16" y="24"
                         targetX="16" targetY="24" fixed="true" radius="48" />
            </PuppetWarp>
            <Rect x="32" y="0" width="32" height="48" color="#0000FF" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="universal_order" />
</Graph>
"##,
        )
        .expect("universal Puppet source-order graph parse");

        let mut renderer =
            pollster::block_on(SceneFrameRenderer::new_for_profile(SceneRenderProfile::Cpu));
        let rendered =
            pollster::block_on(renderer.render_frame(&graph, 0)).expect("CPU Puppet render");
        let captured = rendered.get_pixel(8, 24);
        let overlay = rendered.get_pixel(48, 24);

        assert!(
            (100..=155).contains(&captured[0]) && captured[1] < 20 && captured[2] < 20,
            "expected one half-opacity red capture instead of duplicate artwork, got {captured:?}"
        );
        assert!(
            overlay[2] > 220 && overlay[0] < 20,
            "expected the later blue overlay to remain outside the capture, got {overlay:?}"
        );
    }

    #[test]
    fn render_frame_to_wgpu_texture_outputs_rgba8_texture() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Gpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#FF0000" />
  <Rect x="0" y="0" width="32" height="48" fill="#00FF00" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let gpu_texture = match pollster::block_on(renderer.render_frame_to_wgpu_texture(&graph, 0))
        {
            Ok(texture) => texture,
            Err(SceneRenderError::GpuRender { message }) => {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected GPU texture render error: {err}"),
        };

        assert_eq!(gpu_texture.width, 64);
        assert_eq!(gpu_texture.height, 48);
        assert_eq!(gpu_texture.format, wgpu::TextureFormat::Rgba8Unorm);

        // The texture should be usable to create a view, confirming it is a
        // valid wgpu texture handle.
        let _view = gpu_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
    }

    #[test]
    fn render_frame_to_wgpu_texture_matches_cpu_render_dimensions() {
        let mut cpu_renderer =
            pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).unwrap();
        let mut gpu_renderer =
            pollster::block_on(SceneRenderer::new(SceneRenderProfile::Gpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[128,96]}>
  <Background color="#0000FF" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let cpu_image =
            pollster::block_on(cpu_renderer.render_frame(&graph, 0)).expect("CPU render");
        let gpu_texture =
            match pollster::block_on(gpu_renderer.render_frame_to_wgpu_texture(&graph, 0)) {
                Ok(texture) => texture,
                Err(SceneRenderError::GpuRender { message }) => {
                    skip_gpu_texture_test(message);
                    return;
                }
                Err(err) => panic!("unexpected GPU texture render error: {err}"),
            };

        assert_eq!(cpu_image.width(), gpu_texture.width);
        assert_eq!(cpu_image.height(), gpu_texture.height);
    }

    #[test]
    fn render_frame_to_wgpu_texture_with_external_device() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let Some((device, queue)) = pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = match instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
            {
                Ok(adapter) => adapter,
                Err(err) => {
                    skip_gpu_texture_test(err);
                    return None;
                }
            };
            match adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
            {
                Ok(device_queue) => Some(device_queue),
                Err(err) => {
                    skip_gpu_texture_test(err);
                    None
                }
            }
        }) else {
            return;
        };
        let device = std::sync::Arc::new(device);

        let mut renderer = pollster::block_on(SceneRenderer::new_with_device(
            device,
            queue,
            SceneRenderProfile::Gpu,
        ))
        .expect("renderer with external device");

        let gpu_texture = match pollster::block_on(renderer.render_frame_to_wgpu_texture(&graph, 0))
        {
            Ok(texture) => texture,
            Err(SceneRenderError::GpuRender { message }) => {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected GPU texture render error: {err}"),
        };

        assert_eq!(gpu_texture.width, 64);
        assert_eq!(gpu_texture.height, 48);
    }

    #[test]
    fn render_frame_to_wgpu_texture_mixed_scene_bloom_no_cpu_fallback() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Gpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Scene id="demo">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Circle x="32" y="24" radius="16" color="#FFFFFF" />
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Process id="post">
    <Tex id="scene_src" fmt="rgba16f" from="scene:demo" />
    <Tex id="out" fmt="rgba16f" size={[64,48]} />
    <Pass id="bloom" kind="compute" effect="bloom"
          in={["scene_src"]} out={["out"]}
          params={{ threshold: "0.1", intensity: "1.0", sigma: "2.0" }} />
  </Process>
  <Present from="post" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let gpu_texture = match pollster::block_on(renderer.render_frame_to_wgpu_texture(&graph, 0))
        {
            Ok(texture) => texture,
            Err(SceneRenderError::GpuRender { message }) => {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected GPU texture render error: {err}"),
        };

        assert_eq!(gpu_texture.width, 64);
        assert_eq!(gpu_texture.height, 48);
        assert_eq!(gpu_texture.format, wgpu::TextureFormat::Rgba8Unorm);

        // Verify the texture handle is valid by creating a view.
        let _view = gpu_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
    }

    #[test]
    fn render_frame_to_wgpu_texture_mixed_scene_hsla_bloom_blur_no_alias_conflict() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Gpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Scene id="demo">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Circle x="32" y="24" radius="16" color="#FFFFFF" blend="screen" />
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Process id="post">
    <Tex id="scene_src" fmt="rgba16f" from="scene:demo" />
    <Tex id="hsla_src" fmt="rgba16f" size={[64,48]} />
    <Tex id="bloom_src" fmt="rgba16f" size={[64,48]} />
    <Tex id="blur_src" fmt="rgba16f" size={[64,48]} />
    <Pass id="hsla" kind="compute" effect="hsla_overlay"
          in={["scene_src"]} out={["hsla_src"]}
          params={{ hue: "45", saturation: "0.4", lightness: "0.1", alpha: "0.5" }} />
    <Pass id="bloom" kind="compute" effect="glow_bloom"
          in={["hsla_src"]} out={["bloom_src"]}
          params={{ threshold: "0.1", intensity: "1.0", sigma: "2.0" }} />
    <Pass id="blur" kind="compute" effect="gaussian_5tap_blur"
          in={["bloom_src"]} out={["blur_src"]}
          params={{ sigma: "1.0" }} />
  </Process>
  <Present from="post" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let gpu_texture = match pollster::block_on(renderer.render_frame_to_wgpu_texture(&graph, 0))
        {
            Ok(texture) => texture,
            Err(SceneRenderError::GpuRender { message }) => {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected GPU texture render error: {err}"),
        };

        assert_eq!(gpu_texture.width, 64);
        assert_eq!(gpu_texture.height, 48);
        assert_eq!(gpu_texture.format, wgpu::TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn render_frame_gpu_mixed_scene_hsla_bloom_blur_no_alias_conflict() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Gpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Scene id="demo">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Circle x="32" y="24" radius="16" color="#FFFFFF" blend="screen" />
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Process id="post">
    <Tex id="scene_src" fmt="rgba16f" from="scene:demo" />
    <Tex id="hsla_src" fmt="rgba16f" size={[64,48]} />
    <Tex id="bloom_src" fmt="rgba16f" size={[64,48]} />
    <Tex id="blur_src" fmt="rgba16f" size={[64,48]} />
    <Pass id="hsla" kind="compute" effect="hsla_overlay"
          in={["scene_src"]} out={["hsla_src"]}
          params={{ hue: "45", saturation: "0.4", lightness: "0.1", alpha: "0.5" }} />
    <Pass id="bloom" kind="compute" effect="glow_bloom"
          in={["hsla_src"]} out={["bloom_src"]}
          params={{ threshold: "0.1", intensity: "1.0", sigma: "2.0" }} />
    <Pass id="blur" kind="compute" effect="gaussian_5tap_blur"
          in={["bloom_src"]} out={["blur_src"]}
          params={{ sigma: "1.0" }} />
  </Process>
  <Present from="post" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let image = match pollster::block_on(renderer.render_frame(&graph, 0)) {
            Ok(image) => image,
            Err(SceneRenderError::GpuRender { message }) => {
                skip_gpu_texture_test(message);
                return;
            }
            Err(err) => panic!("unexpected GPU render error: {err}"),
        };

        assert_eq!(image.width(), 64);
        assert_eq!(image.height(), 48);
    }

    #[test]
    fn render_frame_to_preview_surface_can_return_cpu_bgra_without_gpu() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[8,4]}>
  <Background color="#FF0000" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let surface = pollster::block_on(renderer.render_frame_to_preview_surface(
            &graph,
            0,
            ScenePreviewSurfaceOptions {
                backend: ScenePreviewBackend::CpuBgra,
                preferred_format: ScenePreviewPixelFormat::Bgra8Unorm,
            },
        ))
        .expect("CPU BGRA preview surface");

        let ScenePreviewSurface::CpuBgra {
            width,
            height,
            data,
            format,
        } = surface
        else {
            panic!("expected CPU BGRA preview surface");
        };
        assert_eq!(width, 8);
        assert_eq!(height, 4);
        assert_eq!(format, ScenePreviewPixelFormat::Bgra8Unorm);
        assert_eq!(data.len(), 8 * 4 * 4);
        assert_eq!(&data[0..4], &[0, 0, 255, 255]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn render_frame_to_preview_surface_can_return_macos_surface() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[8,4]}>
  <Background color="#00FF00" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let surface = pollster::block_on(renderer.render_frame_to_preview_surface(
            &graph,
            0,
            ScenePreviewSurfaceOptions {
                backend: ScenePreviewBackend::PlatformSurface,
                preferred_format: ScenePreviewPixelFormat::Bgra8Unorm,
            },
        ))
        .expect("macOS preview surface");

        let ScenePreviewSurface::PlatformSurface(ScenePlatformPreviewSurface::MacOs {
            width,
            height,
            format,
            ..
        }) = surface
        else {
            panic!("expected macOS platform preview surface");
        };
        assert_eq!(width, 8);
        assert_eq!(height, 4);
        assert_eq!(format, ScenePreviewPixelFormat::Bgra8Unorm);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn render_frame_to_preview_surface_auto_prefers_macos_platform_surface() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[8,4]}>
  <Background color="#0000FF" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let surface = pollster::block_on(renderer.render_frame_to_preview_surface(
            &graph,
            0,
            ScenePreviewSurfaceOptions::default(),
        ))
        .expect("auto preview surface on macOS");

        assert!(
            matches!(
                surface,
                ScenePreviewSurface::PlatformSurface(ScenePlatformPreviewSurface::MacOs { .. })
            ),
            "expected macOS platform surface from Auto backend"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn render_frame_to_preview_surface_can_return_windows_d3d_surface() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[8,4]}>
  <Background color="#FF0000" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let surface = pollster::block_on(renderer.render_frame_to_preview_surface(
            &graph,
            0,
            ScenePreviewSurfaceOptions {
                backend: ScenePreviewBackend::PlatformSurface,
                preferred_format: ScenePreviewPixelFormat::Bgra8Unorm,
            },
        ))
        .expect("Windows preview surface");

        let ScenePreviewSurface::PlatformSurface(ScenePlatformPreviewSurface::WindowsD3D(surface)) =
            surface
        else {
            panic!("expected Windows D3D platform preview surface");
        };
        assert_eq!(surface.width, 8);
        assert_eq!(surface.height, 4);
        assert_eq!(surface.format, ScenePreviewPixelFormat::Bgra8Unorm);
        assert_ne!(surface.handle.0, 0, "shared handle must be non-null");
        assert_eq!(surface.stride, 8 * 4);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn render_frame_to_preview_surface_auto_prefers_windows_d3d_surface() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[8,4]}>
  <Background color="#00FF00" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let surface = pollster::block_on(renderer.render_frame_to_preview_surface(
            &graph,
            0,
            ScenePreviewSurfaceOptions::default(),
        ))
        .expect("auto preview surface on Windows");

        assert!(
            matches!(
                surface,
                ScenePreviewSurface::PlatformSurface(ScenePlatformPreviewSurface::WindowsD3D(_))
            ),
            "expected Windows D3D platform surface from Auto backend"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_d3d_shared_surface_can_be_opened_on_another_device() {
        use windows::Win32::Graphics::Direct3D::*;
        use windows::Win32::Graphics::Direct3D11::*;

        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[8,4]}>
  <Background color="#0000FF" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let surface = pollster::block_on(renderer.render_frame_to_preview_surface(
            &graph,
            0,
            ScenePreviewSurfaceOptions {
                backend: ScenePreviewBackend::PlatformSurface,
                preferred_format: ScenePreviewPixelFormat::Bgra8Unorm,
            },
        ))
        .expect("Windows preview surface");

        let ScenePreviewSurface::PlatformSurface(ScenePlatformPreviewSurface::WindowsD3D(shared)) =
            surface
        else {
            panic!("expected Windows D3D platform preview surface");
        };

        // Open the shared handle on a separate D3D11 device to verify lifetime.
        unsafe {
            use windows::Win32::Foundation::HMODULE;
            let mut device = None;
            let mut context = None;
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .expect("second D3D11 device");
            let device = device.unwrap();

            let opened: ID3D11Texture2D = device
                .OpenSharedResource(shared.handle.0 as *const _)
                .expect("open shared texture on second device");

            let mut desc = D3D11_TEXTURE2D_DESC::default();
            opened.GetDesc(&mut desc);
            assert_eq!(desc.Width, shared.width);
            assert_eq!(desc.Height, shared.height);
        }
    }

    #[cfg(all(unix, not(target_os = "macos"), not(target_arch = "wasm32")))]
    #[test]
    fn render_frame_to_preview_surface_linux_platform_returns_clear_error() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[8,4]}>
  <Background color="#FF0000" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let result = pollster::block_on(renderer.render_frame_to_preview_surface(
            &graph,
            0,
            ScenePreviewSurfaceOptions {
                backend: ScenePreviewBackend::PlatformSurface,
                preferred_format: ScenePreviewPixelFormat::Bgra8Unorm,
            },
        ));

        let err = result.expect_err("Linux platform surface should be unsupported");
        let msg = err.to_string();
        assert!(
            msg.contains("Linux DMA-BUF"),
            "expected clear DMA-BUF unsupported message, got: {msg}"
        );
    }

    #[cfg(all(unix, not(target_os = "macos"), not(target_arch = "wasm32")))]
    #[test]
    fn render_frame_to_preview_surface_auto_falls_back_to_cpu_bgra_on_linux() {
        let mut renderer = pollster::block_on(SceneRenderer::new(SceneRenderProfile::Cpu)).unwrap();
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[8,4]}>
  <Background color="#0000FF" />
  <Present from="scene" />
</Graph>
"##,
        )
        .expect("scene graph parse");

        let surface = pollster::block_on(renderer.render_frame_to_preview_surface(
            &graph,
            0,
            ScenePreviewSurfaceOptions::default(),
        ))
        .expect("auto preview surface on Linux should fall back to CPU BGRA");

        assert!(
            matches!(surface, ScenePreviewSurface::CpuBgra { .. }),
            "expected CPU BGRA fallback on Linux from Auto backend"
        );
    }
}
