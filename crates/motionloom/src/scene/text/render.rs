use cosmic_text::{Buffer, Color, FontSystem, SwashCache};
use image::{Rgba, RgbaImage};

use crate::scene::composition::{apply_box_blur_pass, blend_pixel, composite_layer_affine_blend};
use crate::scene::drawable::{SceneBlendMode, is_none_paint, parse_color};
use crate::scene::render::{MotionLoomSceneRenderError, eval_scene_number};
use crate::scene::spatial::Affine2;

use super::animator::{TextAnimatorNode, TextAnimatorTargetState};
use super::layout::ResolvedTextLayoutSpec;
use super::model::TextNode;
use super::selector::TextSelectorKind;
use super::selector::{TextSelectionIndex, build_text_selection_index};
use super::style::TextEffectNode;

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedTextLayout {
    pub layout: ResolvedTextLayoutSpec,
    pub selections: TextSelectionIndex,
    pub animator_targets: Vec<PreparedTextAnimatorTargets>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedTextAnimatorTargets {
    pub id: Option<String>,
    pub selector: TextSelectorKind,
    pub targets: Vec<TextAnimatorTargetState>,
}

pub fn prepare_text_layout(text: &TextNode) -> Result<PreparedTextLayout, String> {
    prepare_text_layout_for_value(text, &text.value)
}

pub fn prepare_text_layout_for_value(
    text: &TextNode,
    value: &str,
) -> Result<PreparedTextLayout, String> {
    let selections = build_text_selection_index(value);
    let animator_targets = text
        .animators
        .iter()
        .map(|animator| PreparedTextAnimatorTargets {
            id: animator.id.clone(),
            selector: animator.selector,
            targets: animator.target_states(&selections),
        })
        .collect();

    Ok(PreparedTextLayout {
        layout: ResolvedTextLayoutSpec::from_parts(text.align.as_deref(), text.layout.as_ref())?,
        selections,
        animator_targets,
    })
}

pub(crate) fn text_bounds(buffer: &Buffer, fallback_line_height: f32) -> (f32, f32) {
    let mut width = 0.0_f32;
    let mut top = f32::INFINITY;
    let mut bottom = f32::NEG_INFINITY;
    for run in buffer.layout_runs() {
        width = width.max(run.line_w);
        top = top.min(run.line_top);
        bottom = bottom.max(run.line_top + run.line_height);
    }
    if !top.is_finite() || !bottom.is_finite() {
        return (0.0, fallback_line_height);
    }
    (width.max(0.0), (bottom - top).max(fallback_line_height))
}

#[derive(Clone, Copy)]
struct TextGlyphVisual {
    dx: f32,
    dy: f32,
    rotation: f32,
    scale_x: f32,
    scale_y: f32,
    opacity: f32,
    color: [u8; 4],
}

impl TextGlyphVisual {
    const fn new(color: [u8; 4], opacity: f32) -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            rotation: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity,
            color,
        }
    }
}

pub(crate) struct TextAnimatorRasterParams<'a> {
    pub(crate) text: &'a TextNode,
    pub(crate) prepared: &'a PreparedTextLayout,
    pub(crate) value: &'a str,
    pub(crate) base_color: [u8; 4],
    pub(crate) base_opacity: f32,
    pub(crate) offset_x: i32,
    pub(crate) offset_y: i32,
    pub(crate) raster_scale: f32,
    pub(crate) max_lines: Option<usize>,
    pub(crate) global_time_ms: i64,
    pub(crate) time_norm: f32,
    pub(crate) time_sec: f32,
}

pub(crate) struct TextRasterParams {
    pub(crate) offset_x: i32,
    pub(crate) offset_y: i32,
    pub(crate) raster_scale: f32,
    pub(crate) opacity: f32,
    pub(crate) max_lines: Option<usize>,
}

#[derive(Clone)]
pub(crate) struct TextLayerEffectSpec {
    pub(crate) stroke: Option<TextLayerStroke>,
    pub(crate) shadows: Vec<TextLayerShadow>,
    pub(crate) glows: Vec<TextLayerGlow>,
    pub(crate) blur_radius: f32,
    pub(crate) pad_px: i32,
}

pub(crate) struct TextRasterizedLayer {
    pub(crate) image: RgbaImage,
    pub(crate) transform: Affine2,
    pub(crate) effects: TextLayerEffectSpec,
}

impl TextLayerEffectSpec {
    pub(crate) fn has_effects(&self) -> bool {
        self.stroke.is_some()
            || !self.shadows.is_empty()
            || !self.glows.is_empty()
            || self.blur_radius > 0.001
    }

    pub(crate) fn scaled_for_raster(mut self, scale: f32) -> Self {
        if scale <= 1.0001 {
            return self;
        }
        if let Some(stroke) = self.stroke.as_mut() {
            stroke.width *= scale;
        }
        for shadow in &mut self.shadows {
            shadow.x *= scale;
            shadow.y *= scale;
            shadow.blur *= scale;
        }
        for glow in &mut self.glows {
            glow.radius *= scale;
        }
        self.blur_radius *= scale;
        self.pad_px = ((self.pad_px as f32) * scale).ceil().clamp(0.0, 2048.0) as i32;
        self
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TextLayerStroke {
    pub(crate) color: [u8; 4],
    pub(crate) width: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct TextLayerShadow {
    pub(crate) color: [u8; 4],
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) blur: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct TextLayerGlow {
    pub(crate) color: [u8; 4],
    pub(crate) radius: f32,
    pub(crate) intensity: f32,
}

pub(crate) fn text_layer_effect_spec(
    text: &TextNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<TextLayerEffectSpec, MotionLoomSceneRenderError> {
    let mut stroke = text
        .stroke
        .as_deref()
        .filter(|value| !is_none_paint(value))
        .map(parse_color)
        .transpose()?
        .and_then(|color| {
            let width = text
                .stroke_width
                .as_deref()
                .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
                .unwrap_or(1.0)
                .max(0.0);
            (width > 0.001 && color[3] > 0).then_some(TextLayerStroke { color, width })
        });
    let mut shadows = Vec::new();
    let mut glows = Vec::new();
    let mut blur_radius = text
        .blur
        .as_deref()
        .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
        .unwrap_or(0.0)
        .max(0.0);

    for animator in &text.animators {
        if let Some(style) = animator.style.as_ref() {
            if let Some(color) = style
                .stroke
                .as_deref()
                .filter(|value| !is_none_paint(value))
                .map(parse_color)
                .transpose()?
            {
                let width = style
                    .stroke_width
                    .as_deref()
                    .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
                    .unwrap_or(1.0)
                    .max(0.0);
                if width > 0.001 && color[3] > 0 {
                    stroke = Some(TextLayerStroke { color, width });
                }
            }
            if let Some(blur) = style
                .blur
                .as_deref()
                .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
            {
                blur_radius = blur_radius.max(blur.max(0.0));
            }
            if style.shadow_color.is_some()
                || style.shadow_x.is_some()
                || style.shadow_y.is_some()
                || style.shadow_blur.is_some()
            {
                let color = style
                    .shadow_color
                    .as_deref()
                    .map(parse_color)
                    .transpose()?
                    .unwrap_or([0, 0, 0, 160]);
                let x = style
                    .shadow_x
                    .as_deref()
                    .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
                    .unwrap_or(0.0);
                let y = style
                    .shadow_y
                    .as_deref()
                    .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
                    .unwrap_or(0.0);
                let blur = style
                    .shadow_blur
                    .as_deref()
                    .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
                    .unwrap_or(0.0)
                    .max(0.0);
                if color[3] > 0 {
                    shadows.push(TextLayerShadow { color, x, y, blur });
                }
            }
        }

        for effect in &animator.effects {
            let TextEffectNode::Glow(glow) = effect;
            let color = glow
                .color
                .as_deref()
                .map(parse_color)
                .transpose()?
                .unwrap_or([255, 176, 0, 255]);
            let radius = eval_scene_number(&glow.radius, time_norm, time_sec)?.max(0.0);
            let intensity = eval_scene_number(&glow.intensity, time_norm, time_sec)?.max(0.0);
            if color[3] > 0 && radius > 0.001 && intensity > 0.001 {
                glows.push(TextLayerGlow {
                    color,
                    radius,
                    intensity,
                });
            }
        }
    }

    let mut extra = blur_radius * 3.0;
    if let Some(stroke) = stroke {
        extra = extra.max(stroke.width + 2.0);
    }
    for shadow in &shadows {
        extra = extra.max(shadow.x.abs() + shadow.blur * 3.0 + 2.0);
        extra = extra.max(shadow.y.abs() + shadow.blur * 3.0 + 2.0);
    }
    for glow in &glows {
        extra = extra.max(glow.radius * 3.0 + 2.0);
    }

    Ok(TextLayerEffectSpec {
        stroke,
        shadows,
        glows,
        blur_radius,
        pad_px: extra.ceil().clamp(0.0, 256.0) as i32,
    })
}

pub(crate) fn apply_text_layer_effects(input: &RgbaImage, spec: &TextLayerEffectSpec) -> RgbaImage {
    let fill_layer = if spec.blur_radius > 0.001 {
        blur_rgba_image(input, spec.blur_radius)
    } else {
        input.clone()
    };
    let mut out = RgbaImage::from_pixel(input.width(), input.height(), Rgba([0, 0, 0, 0]));

    for shadow in &spec.shadows {
        let mut shadow_layer = tint_alpha_layer(input, shadow.color, 1.0);
        if shadow.blur > 0.001 {
            shadow_layer = blur_rgba_image(&shadow_layer, shadow.blur);
        }
        composite_layer_affine_blend(
            &mut out,
            &shadow_layer,
            Affine2::translate(shadow.x, shadow.y),
            1.0,
            SceneBlendMode::Normal,
        );
    }

    for glow in &spec.glows {
        let mut glow_layer = tint_alpha_layer(input, glow.color, glow.intensity);
        glow_layer = blur_rgba_image(&glow_layer, glow.radius);
        composite_layer_affine_blend(
            &mut out,
            &glow_layer,
            Affine2::identity(),
            1.0,
            SceneBlendMode::Screen,
        );
    }

    if let Some(stroke) = spec.stroke {
        let stroke_layer = stroke_layer_from_alpha(input, stroke.width, stroke.color);
        composite_layer_affine_blend(
            &mut out,
            &stroke_layer,
            Affine2::identity(),
            1.0,
            SceneBlendMode::Normal,
        );
    }

    composite_layer_affine_blend(
        &mut out,
        &fill_layer,
        Affine2::identity(),
        1.0,
        SceneBlendMode::Normal,
    );
    out
}

fn blur_rgba_image(input: &RgbaImage, sigma: f32) -> RgbaImage {
    let blurred = apply_box_blur_pass(input, sigma, true);
    apply_box_blur_pass(&blurred, sigma, false)
}

pub(crate) fn soften_text_alpha_edges(input: &RgbaImage, amount: f32) -> RgbaImage {
    let amount = amount.clamp(0.0, 1.0);
    if amount <= 0.001 {
        return input.clone();
    }
    let blurred = blur_rgba_image(input, 0.35 + amount * 1.15);
    let mut out = input.clone();
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let base = input.get_pixel(x, y);
        let soft = blurred.get_pixel(x, y);
        let alpha = (base[3] as f32 * (1.0 - amount) + soft[3] as f32 * amount)
            .round()
            .clamp(0.0, 255.0) as u8;
        if alpha == 0 {
            *pixel = Rgba([0, 0, 0, 0]);
        } else {
            *pixel = Rgba([base[0], base[1], base[2], alpha]);
        }
    }
    out
}

fn tint_alpha_layer(input: &RgbaImage, color: [u8; 4], alpha_scale: f32) -> RgbaImage {
    let alpha_scale = alpha_scale.max(0.0);
    let color_alpha = color[3] as f32 / 255.0;
    let mut out = RgbaImage::from_pixel(input.width(), input.height(), Rgba([0, 0, 0, 0]));
    for (x, y, pixel) in input.enumerate_pixels() {
        let alpha = ((pixel[3] as f32) * color_alpha * alpha_scale)
            .round()
            .clamp(0.0, 255.0) as u8;
        if alpha > 0 {
            out.put_pixel(x, y, Rgba([color[0], color[1], color[2], alpha]));
        }
    }
    out
}

pub(crate) fn stroke_layer_from_alpha(input: &RgbaImage, width: f32, color: [u8; 4]) -> RgbaImage {
    let radius = width.ceil().clamp(1.0, 96.0) as i32;
    let alpha = input.pixels().map(|pixel| pixel[3]).collect::<Vec<_>>();
    let dilated = dilate_alpha(&alpha, input.width(), input.height(), radius);
    let mut out = RgbaImage::from_pixel(input.width(), input.height(), Rgba([0, 0, 0, 0]));
    let color_alpha = color[3] as f32 / 255.0;
    for y in 0..input.height() {
        for x in 0..input.width() {
            let ix = (y * input.width() + x) as usize;
            let alpha = ((dilated[ix] as f32) * color_alpha)
                .round()
                .clamp(0.0, 255.0) as u8;
            if alpha > 0 {
                out.put_pixel(x, y, Rgba([color[0], color[1], color[2], alpha]));
            }
        }
    }
    out
}

fn dilate_alpha(alpha: &[u8], width: u32, height: u32, radius: i32) -> Vec<u8> {
    let mut horizontal = vec![0_u8; alpha.len()];
    for y in 0..height {
        for x in 0..width {
            let mut max_alpha = 0_u8;
            let min_x = (x as i32 - radius).max(0) as u32;
            let max_x = (x as i32 + radius).min(width as i32 - 1) as u32;
            for sx in min_x..=max_x {
                let ix = (y * width + sx) as usize;
                max_alpha = max_alpha.max(alpha[ix]);
            }
            horizontal[(y * width + x) as usize] = max_alpha;
        }
    }

    let mut out = vec![0_u8; alpha.len()];
    for y in 0..height {
        for x in 0..width {
            let mut max_alpha = 0_u8;
            let min_y = (y as i32 - radius).max(0) as u32;
            let max_y = (y as i32 + radius).min(height as i32 - 1) as u32;
            for sy in min_y..=max_y {
                let ix = (sy * width + x) as usize;
                max_alpha = max_alpha.max(horizontal[ix]);
            }
            out[(y * width + x) as usize] = max_alpha;
        }
    }
    out
}

pub(crate) fn draw_text_buffer_with_animators(
    buffer: &Buffer,
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    layer: &mut RgbaImage,
    params: TextAnimatorRasterParams<'_>,
) -> Result<(), MotionLoomSceneRenderError> {
    let line_char_starts = text_line_char_starts(params.value);
    let mut visual_line_ix = 0usize;
    for run in buffer.layout_runs() {
        if let Some(max_lines) = params.max_lines
            && visual_line_ix >= max_lines
        {
            visual_line_ix += 1;
            continue;
        }
        visual_line_ix += 1;

        let line_start_char = line_char_starts.get(run.line_i).copied().unwrap_or(0);
        for glyph in run.glyphs.iter() {
            let start_char = line_start_char + byte_to_char_offset(run.text, glyph.start);
            let end_char =
                (line_start_char + byte_to_char_offset(run.text, glyph.end)).max(start_char + 1);
            let visual = text_glyph_visual(&params, start_char, end_char)?;
            if visual.opacity <= 0.0001 {
                continue;
            }

            let physical_glyph = glyph.physical((0.0, 0.0), params.raster_scale);
            let glyph_color = Color::rgba(visual.color[0], visual.color[1], visual.color[2], 255);
            let raster_line_y = run.line_y * params.raster_scale;
            let anchor_x = physical_glyph.x as f32
                + params.offset_x as f32
                + glyph.w.max(1.0) * params.raster_scale * 0.5;
            let anchor_y = raster_line_y + physical_glyph.y as f32 + params.offset_y as f32;
            let (sin_t, cos_t) = visual.rotation.to_radians().sin_cos();

            swash_cache.with_pixels(
                font_system,
                physical_glyph.cache_key,
                glyph_color,
                |x, y, color| {
                    let raw_x = physical_glyph.x + x + params.offset_x;
                    let raw_y =
                        raster_line_y.round() as i32 + physical_glyph.y + y + params.offset_y;
                    let local_x = raw_x as f32 - anchor_x;
                    let local_y = raw_y as f32 - anchor_y;
                    let scaled_x = local_x * visual.scale_x;
                    let scaled_y = local_y * visual.scale_y;
                    let dst_x = anchor_x + visual.dx * params.raster_scale + scaled_x * cos_t
                        - scaled_y * sin_t;
                    let dst_y = anchor_y
                        + visual.dy * params.raster_scale
                        + scaled_x * sin_t
                        + scaled_y * cos_t;
                    let px = dst_x.round() as i32;
                    let py = dst_y.round() as i32;
                    if px < 0 || py < 0 {
                        return;
                    }
                    let (px, py) = (px as u32, py as u32);
                    if px >= layer.width() || py >= layer.height() {
                        return;
                    }
                    let (sr, sg, sb, sa) = color.as_rgba_tuple();
                    let sa = ((sa as f32) * visual.opacity).round().clamp(0.0, 255.0) as u8;
                    blend_pixel(layer, px, py, [sr, sg, sb, sa]);
                },
            );
        }
    }
    Ok(())
}

pub(crate) fn draw_text_buffer_scaled(
    buffer: &Buffer,
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    layer: &mut RgbaImage,
    color: Color,
    params: TextRasterParams,
) {
    for (visual_line_ix, run) in buffer.layout_runs().enumerate() {
        if params
            .max_lines
            .is_some_and(|max_lines| visual_line_ix >= max_lines)
        {
            continue;
        }
        let raster_line_y = (run.line_y * params.raster_scale).round() as i32;
        for glyph in run.glyphs.iter() {
            // Shape and line-break at logical resolution, then request only the
            // glyph bitmap at output resolution. This keeps layout invariant.
            let physical_glyph = glyph.physical((0.0, 0.0), params.raster_scale);
            let glyph_color = glyph.color_opt.unwrap_or(color);
            swash_cache.with_pixels(
                font_system,
                physical_glyph.cache_key,
                glyph_color,
                |x, y, pixel_color| {
                    let px = physical_glyph.x + x + params.offset_x;
                    let py = raster_line_y + physical_glyph.y + y + params.offset_y;
                    if px < 0 || py < 0 {
                        return;
                    }
                    let (px, py) = (px as u32, py as u32);
                    if px >= layer.width() || py >= layer.height() {
                        return;
                    }
                    let (sr, sg, sb, sa) = pixel_color.as_rgba_tuple();
                    let sa = ((sa as f32) * params.opacity).round().clamp(0.0, 255.0) as u8;
                    blend_pixel(layer, px, py, [sr, sg, sb, sa]);
                },
            );
        }
    }
}

fn text_glyph_visual(
    params: &TextAnimatorRasterParams<'_>,
    glyph_start_char: usize,
    glyph_end_char: usize,
) -> Result<TextGlyphVisual, MotionLoomSceneRenderError> {
    let base_alpha = params.base_color[3] as f32 / 255.0;
    let mut visual = TextGlyphVisual::new(params.base_color, params.base_opacity * base_alpha);
    for (animator, prepared_targets) in params
        .text
        .animators
        .iter()
        .zip(&params.prepared.animator_targets)
    {
        let Some(target) = text_animator_target_for_glyph(
            animator,
            &prepared_targets.targets,
            glyph_start_char,
            glyph_end_char,
            params.time_norm,
            params.time_sec,
        ) else {
            continue;
        };
        let (local_norm, local_sec) =
            text_animator_local_time(animator, target, params.global_time_ms);
        if let Some(transform) = animator.transform.as_ref() {
            visual.dx += eval_text_anim_number(transform.x.as_deref(), local_norm, local_sec, 0.0);
            visual.dy += eval_text_anim_number(transform.y.as_deref(), local_norm, local_sec, 0.0);
            visual.rotation +=
                eval_text_anim_number(transform.rotation.as_deref(), local_norm, local_sec, 0.0);
            let scale =
                eval_text_anim_number(transform.scale.as_deref(), local_norm, local_sec, 1.0)
                    .clamp(0.001, 64.0);
            visual.scale_x *= scale
                * eval_text_anim_number(transform.scale_x.as_deref(), local_norm, local_sec, 1.0)
                    .clamp(0.001, 64.0);
            visual.scale_y *= scale
                * eval_text_anim_number(transform.scale_y.as_deref(), local_norm, local_sec, 1.0)
                    .clamp(0.001, 64.0);
        }
        if let Some(style) = animator.style.as_ref() {
            visual.opacity *=
                eval_text_anim_number(style.opacity.as_deref(), local_norm, local_sec, 1.0)
                    .clamp(0.0, 1.0);
            if let Some(color) = style.color.as_deref() {
                visual.color = parse_color(color)?;
                visual.opacity *= visual.color[3] as f32 / 255.0;
            }
        }
    }
    visual.opacity = visual.opacity.clamp(0.0, 1.0);
    Ok(visual)
}

fn text_animator_target_for_glyph<'a>(
    animator: &TextAnimatorNode,
    targets: &'a [TextAnimatorTargetState],
    glyph_start_char: usize,
    glyph_end_char: usize,
    time_norm: f32,
    time_sec: f32,
) -> Option<&'a TextAnimatorTargetState> {
    if animator.is_karaoke() {
        let active_index = animator
            .active_word
            .as_deref()
            .and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
            .unwrap_or(0.0)
            .floor()
            .max(0.0) as usize;
        return targets.iter().find(|target| {
            target.source_index == active_index
                && text_ranges_intersect(
                    glyph_start_char,
                    glyph_end_char,
                    target.start_char,
                    target.end_char,
                )
        });
    }

    targets.iter().find(|target| {
        text_ranges_intersect(
            glyph_start_char,
            glyph_end_char,
            target.start_char,
            target.end_char,
        )
    })
}

fn text_animator_local_time(
    animator: &TextAnimatorNode,
    target: &TextAnimatorTargetState,
    global_time_ms: i64,
) -> (f32, f32) {
    if animator.is_karaoke() {
        let duration_ms = animator
            .duration_ms
            .map(|value| value.max(1) as i64)
            .unwrap_or(360);
        let local_ms = (global_time_ms - animator.from_ms).rem_euclid(duration_ms);
        return (
            (local_ms as f32 / duration_ms as f32).clamp(0.0, 1.0),
            local_ms as f32 / 1000.0,
        );
    }

    let duration_ms = target.duration_ms.map(|value| value.max(1) as i64);
    let local_ms = global_time_ms - target.start_ms;
    let clamped_ms = if let Some(duration_ms) = duration_ms {
        local_ms.clamp(0, duration_ms)
    } else {
        local_ms.max(0)
    };
    let local_norm = duration_ms
        .map(|duration_ms| clamped_ms as f32 / duration_ms as f32)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    (local_norm, clamped_ms as f32 / 1000.0)
}

fn eval_text_anim_number(expr: Option<&str>, time_norm: f32, time_sec: f32, fallback: f32) -> f32 {
    expr.and_then(|expr| eval_scene_number(expr, time_norm, time_sec).ok())
        .unwrap_or(fallback)
}

fn text_ranges_intersect(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

fn text_line_char_starts(value: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    let mut char_ix = 0usize;
    for ch in value.chars() {
        char_ix += 1;
        if ch == '\n' {
            starts.push(char_ix);
        }
    }
    starts
}

fn byte_to_char_offset(value: &str, byte_ix: usize) -> usize {
    let mut byte_ix = byte_ix.min(value.len());
    while byte_ix > 0 && !value.is_char_boundary(byte_ix) {
        byte_ix -= 1;
    }
    value[..byte_ix].chars().count()
}
