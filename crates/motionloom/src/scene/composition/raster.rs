use image::{Rgba, RgbaImage};

use crate::scene::drawable::Point2;
use crate::scene::drawable::SceneBlendMode;
use crate::scene::spatial::{
    Affine2, CameraRect, EvaluatedDeformGrid, triangle_barycentric,
    triangle_barycentric_denominator,
};

pub(crate) fn apply_alpha_mask(layer: &mut RgbaImage, mask: &RgbaImage) {
    apply_alpha_mask_with_invert(layer, mask, false);
}

pub(crate) fn apply_alpha_mask_with_invert(layer: &mut RgbaImage, mask: &RgbaImage, invert: bool) {
    let w = layer.width().min(mask.width());
    let h = layer.height().min(mask.height());
    for y in 0..h {
        for x in 0..w {
            let mut alpha = mask.get_pixel(x, y)[3] as f32 / 255.0;
            if invert {
                alpha = 1.0 - alpha;
            }
            let pixel = layer.get_pixel_mut(x, y);
            pixel[3] = ((pixel[3] as f32) * alpha).round().clamp(0.0, 255.0) as u8;
        }
    }
}

pub(crate) fn shape_alpha_mask(mask: &RgbaImage, expansion: f32, feather: f32) -> RgbaImage {
    let mut alpha = mask.clone();
    let radius = expansion.abs().ceil().clamp(0.0, 64.0) as i32;
    if radius > 0 {
        let source = alpha.clone();
        let expand = expansion > 0.0;
        for y in 0..alpha.height() {
            for x in 0..alpha.width() {
                let mut value = if expand { 0u8 } else { 255u8 };
                for oy in -radius..=radius {
                    for ox in -radius..=radius {
                        if ox * ox + oy * oy > radius * radius {
                            continue;
                        }
                        let sx = x as i32 + ox;
                        let sy = y as i32 + oy;
                        let sample = if sx < 0
                            || sy < 0
                            || sx >= source.width() as i32
                            || sy >= source.height() as i32
                        {
                            0
                        } else {
                            source.get_pixel(sx as u32, sy as u32)[3]
                        };
                        value = if expand {
                            value.max(sample)
                        } else {
                            value.min(sample)
                        };
                    }
                }
                alpha.get_pixel_mut(x, y)[3] = value;
            }
        }
    }

    let feather_radius = feather.ceil().clamp(0.0, 64.0) as i32;
    if feather_radius > 0 {
        for horizontal in [true, false] {
            let source = alpha.clone();
            for y in 0..alpha.height() {
                for x in 0..alpha.width() {
                    let mut sum = 0u32;
                    let mut count = 0u32;
                    for offset in -feather_radius..=feather_radius {
                        let sx = if horizontal {
                            x as i32 + offset
                        } else {
                            x as i32
                        };
                        let sy = if horizontal {
                            y as i32
                        } else {
                            y as i32 + offset
                        };
                        if sx >= 0
                            && sy >= 0
                            && sx < source.width() as i32
                            && sy < source.height() as i32
                        {
                            sum += source.get_pixel(sx as u32, sy as u32)[3] as u32;
                            count += 1;
                        }
                    }
                    alpha.get_pixel_mut(x, y)[3] = (sum / count.max(1)) as u8;
                }
            }
        }
    }
    alpha
}

pub(crate) fn composite_layer(canvas: &mut RgbaImage, layer: &RgbaImage) {
    for (x, y, pixel) in layer.enumerate_pixels() {
        if pixel[3] > 0 && x < canvas.width() && y < canvas.height() {
            blend_pixel(canvas, x, y, pixel.0);
        }
    }
}

pub(crate) fn composite_transformed_layer(
    canvas: &mut RgbaImage,
    layer: &RgbaImage,
    x: f32,
    y: f32,
    rotation_deg: f32,
    scale: f32,
) {
    let theta = rotation_deg.to_radians();
    let (sin_t, cos_t) = theta.sin_cos();
    for (src_x, src_y, pixel) in layer.enumerate_pixels() {
        if pixel[3] == 0 {
            continue;
        }
        let sx = src_x as f32 * scale;
        let sy = src_y as f32 * scale;
        let dx = x + sx * cos_t - sy * sin_t;
        let dy = y + sx * sin_t + sy * cos_t;
        let dst_x = dx.round() as i32;
        let dst_y = dy.round() as i32;
        if dst_x < 0 || dst_y < 0 {
            continue;
        }
        let (dst_x, dst_y) = (dst_x as u32, dst_y as u32);
        if dst_x >= canvas.width() || dst_y >= canvas.height() {
            continue;
        }
        blend_pixel(canvas, dst_x, dst_y, pixel.0);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_transformed_layer_anchored(
    canvas: &mut RgbaImage,
    layer: &RgbaImage,
    x: f32,
    y: f32,
    rotation_deg: f32,
    scale: f32,
    anchor_x: f32,
    anchor_y: f32,
) {
    let theta = rotation_deg.to_radians();
    let (sin_t, cos_t) = theta.sin_cos();
    for (src_x, src_y, pixel) in layer.enumerate_pixels() {
        if pixel[3] == 0 {
            continue;
        }
        let sx = (src_x as f32 - anchor_x) * scale;
        let sy = (src_y as f32 - anchor_y) * scale;
        let dx = x + sx * cos_t - sy * sin_t;
        let dy = y + sx * sin_t + sy * cos_t;
        let dst_x = dx.round() as i32;
        let dst_y = dy.round() as i32;
        if dst_x < 0 || dst_y < 0 {
            continue;
        }
        let (dst_x, dst_y) = (dst_x as u32, dst_y as u32);
        if dst_x >= canvas.width() || dst_y >= canvas.height() {
            continue;
        }
        blend_pixel(canvas, dst_x, dst_y, pixel.0);
    }
}

pub(crate) fn composite_layer_affine(
    canvas: &mut RgbaImage,
    layer: &RgbaImage,
    transform: Affine2,
) {
    composite_layer_affine_clipped(canvas, layer, transform, None);
}

pub(crate) fn composite_layer_affine_blend(
    canvas: &mut RgbaImage,
    layer: &RgbaImage,
    transform: Affine2,
    opacity: f32,
    blend: SceneBlendMode,
) {
    composite_layer_affine_blend_clipped(canvas, layer, transform, opacity, blend, None);
}

pub(crate) fn composite_layer_affine_blend_clipped(
    canvas: &mut RgbaImage,
    layer: &RgbaImage,
    transform: Affine2,
    opacity: f32,
    blend: SceneBlendMode,
    clip: Option<CameraRect>,
) {
    let opacity = opacity.clamp(0.0, 1.0);
    if opacity <= 0.0001 {
        return;
    }
    let Some(inverse) = transform.inverse() else {
        return;
    };
    let w = layer.width() as f32;
    let h = layer.height() as f32;
    if w <= 0.0 || h <= 0.0 {
        return;
    }

    let corners = [
        transform.transform_point(0.0, 0.0),
        transform.transform_point(w - 1.0, 0.0),
        transform.transform_point(w - 1.0, h - 1.0),
        transform.transform_point(0.0, h - 1.0),
    ];
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for (x, y) in corners {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    let mut x0 = (min_x.floor() as i32 - 2).clamp(0, canvas.width() as i32);
    let mut y0 = (min_y.floor() as i32 - 2).clamp(0, canvas.height() as i32);
    let mut x1 = (max_x.ceil() as i32 + 2).clamp(0, canvas.width() as i32);
    let mut y1 = (max_y.ceil() as i32 + 2).clamp(0, canvas.height() as i32);
    if let Some(clip) = clip {
        x0 = x0.max(clip.x.floor() as i32);
        y0 = y0.max(clip.y.floor() as i32);
        x1 = x1.min((clip.x + clip.width).ceil() as i32);
        y1 = y1.min((clip.y + clip.height).ceil() as i32);
    }
    if x1 <= x0 || y1 <= y0 {
        return;
    }

    for dst_y in y0..y1 {
        for dst_x in x0..x1 {
            let (src_x, src_y) = inverse.transform_point(dst_x as f32, dst_y as f32);
            let Some(mut pixel) = sample_layer_bilinear(layer, src_x, src_y) else {
                continue;
            };
            if pixel[3] == 0 {
                continue;
            }
            pixel[3] = ((pixel[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            blend_pixel_with_mode(canvas, dst_x as u32, dst_y as u32, pixel, blend);
        }
    }
}

pub(crate) fn composite_layer_affine_clipped(
    canvas: &mut RgbaImage,
    layer: &RgbaImage,
    transform: Affine2,
    clip: Option<CameraRect>,
) {
    let Some(inverse) = transform.inverse() else {
        return;
    };
    let w = layer.width() as f32;
    let h = layer.height() as f32;
    if w <= 0.0 || h <= 0.0 {
        return;
    }

    let corners = [
        transform.transform_point(0.0, 0.0),
        transform.transform_point(w - 1.0, 0.0),
        transform.transform_point(w - 1.0, h - 1.0),
        transform.transform_point(0.0, h - 1.0),
    ];
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for (x, y) in corners {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    let mut x0 = (min_x.floor() as i32 - 2).clamp(0, canvas.width() as i32);
    let mut y0 = (min_y.floor() as i32 - 2).clamp(0, canvas.height() as i32);
    let mut x1 = (max_x.ceil() as i32 + 2).clamp(0, canvas.width() as i32);
    let mut y1 = (max_y.ceil() as i32 + 2).clamp(0, canvas.height() as i32);
    if let Some(clip) = clip {
        x0 = x0.max(clip.x.floor() as i32);
        y0 = y0.max(clip.y.floor() as i32);
        x1 = x1.min((clip.x + clip.width).ceil() as i32);
        y1 = y1.min((clip.y + clip.height).ceil() as i32);
    }
    if x1 <= x0 || y1 <= y0 {
        return;
    }

    for dst_y in y0..y1 {
        for dst_x in x0..x1 {
            let (src_x, src_y) = inverse.transform_point(dst_x as f32, dst_y as f32);
            let Some(pixel) = sample_layer_bilinear(layer, src_x, src_y) else {
                continue;
            };
            if pixel[3] == 0 {
                continue;
            }
            blend_pixel(canvas, dst_x as u32, dst_y as u32, pixel);
        }
    }
}

pub(crate) fn composite_layer_projected_quad_blend_clipped(
    canvas: &mut RgbaImage,
    layer: &RgbaImage,
    quad: [(f32, f32, f32); 4],
    opacity: f32,
    blend: SceneBlendMode,
    clip: Option<CameraRect>,
) {
    let opacity = opacity.clamp(0.0, 1.0);
    if opacity <= 0.0001 || layer.width() == 0 || layer.height() == 0 {
        return;
    }

    // Split the projected plane into two triangles. The third component is
    // reciprocal depth and keeps sampling perspective-correct across the card.
    let src = [
        Point2::new(0.0, 0.0),
        Point2::new(layer.width().saturating_sub(1) as f32, 0.0),
        Point2::new(
            layer.width().saturating_sub(1) as f32,
            layer.height().saturating_sub(1) as f32,
        ),
        Point2::new(0.0, layer.height().saturating_sub(1) as f32),
    ];
    let dst = [
        Point2::new(quad[0].0, quad[0].1),
        Point2::new(quad[1].0, quad[1].1),
        Point2::new(quad[2].0, quad[2].1),
        Point2::new(quad[3].0, quad[3].1),
    ];
    raster_projected_triangle(
        canvas,
        layer,
        [src[0], src[1], src[2]],
        [dst[0], dst[1], dst[2]],
        [quad[0].2, quad[1].2, quad[2].2],
        opacity,
        blend,
        clip,
    );
    raster_projected_triangle(
        canvas,
        layer,
        [src[0], src[2], src[3]],
        [dst[0], dst[2], dst[3]],
        [quad[0].2, quad[2].2, quad[3].2],
        opacity,
        blend,
        clip,
    );
}

pub(crate) fn sample_layer_bilinear(layer: &RgbaImage, x: f32, y: f32) -> Option<[u8; 4]> {
    if x < -0.5 || y < -0.5 || x > layer.width() as f32 - 0.5 || y > layer.height() as f32 - 0.5 {
        return None;
    }

    let x = x.clamp(0.0, layer.width().saturating_sub(1) as f32);
    let y = y.clamp(0.0, layer.height().saturating_sub(1) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(layer.width().saturating_sub(1));
    let y1 = (y0 + 1).min(layer.height().saturating_sub(1));
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let samples = [
        (layer.get_pixel(x0, y0).0, (1.0 - tx) * (1.0 - ty)),
        (layer.get_pixel(x1, y0).0, tx * (1.0 - ty)),
        (layer.get_pixel(x0, y1).0, (1.0 - tx) * ty),
        (layer.get_pixel(x1, y1).0, tx * ty),
    ];

    let mut premul = [0.0_f32; 3];
    let mut alpha = 0.0_f32;
    for (rgba, weight) in samples {
        let a = rgba[3] as f32 / 255.0;
        alpha += a * weight;
        premul[0] += rgba[0] as f32 * a * weight;
        premul[1] += rgba[1] as f32 * a * weight;
        premul[2] += rgba[2] as f32 * a * weight;
    }
    if alpha <= 0.0001 {
        return None;
    }

    Some([
        (premul[0] / alpha).round().clamp(0.0, 255.0) as u8,
        (premul[1] / alpha).round().clamp(0.0, 255.0) as u8,
        (premul[2] / alpha).round().clamp(0.0, 255.0) as u8,
        (alpha * 255.0).round().clamp(0.0, 255.0) as u8,
    ])
}

#[allow(clippy::too_many_arguments)]
fn raster_projected_triangle(
    canvas: &mut RgbaImage,
    layer: &RgbaImage,
    src: [Point2; 3],
    dst: [Point2; 3],
    reciprocal_depth: [f32; 3],
    opacity: f32,
    blend: SceneBlendMode,
    clip: Option<CameraRect>,
) {
    let min_x = dst
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .floor() as i32
        - 1;
    let min_y = dst
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .floor() as i32
        - 1;
    let max_x = dst
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32
        + 1;
    let max_y = dst
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32
        + 1;

    let mut x0 = min_x.clamp(0, canvas.width() as i32);
    let mut y0 = min_y.clamp(0, canvas.height() as i32);
    let mut x1 = max_x.clamp(0, canvas.width() as i32);
    let mut y1 = max_y.clamp(0, canvas.height() as i32);
    if let Some(clip) = clip {
        x0 = x0.max(clip.x.floor() as i32);
        y0 = y0.max(clip.y.floor() as i32);
        x1 = x1.min((clip.x + clip.width).ceil() as i32);
        y1 = y1.min((clip.y + clip.height).ceil() as i32);
    }
    if x1 <= x0 || y1 <= y0 {
        return;
    }

    let denom = triangle_barycentric_denominator(dst);
    if denom.abs() <= 0.00001 {
        return;
    }

    for y in y0..y1 {
        for x in x0..x1 {
            let point = Point2::new(x as f32, y as f32);
            let Some((w0, w1, w2)) = triangle_barycentric(point, dst, denom) else {
                continue;
            };
            if w0 < -0.001 || w1 < -0.001 || w2 < -0.001 {
                continue;
            }
            let weighted_depth =
                reciprocal_depth[0] * w0 + reciprocal_depth[1] * w1 + reciprocal_depth[2] * w2;
            if weighted_depth.abs() <= 0.00001 {
                continue;
            }
            let src_x = (src[0].x * reciprocal_depth[0] * w0
                + src[1].x * reciprocal_depth[1] * w1
                + src[2].x * reciprocal_depth[2] * w2)
                / weighted_depth;
            let src_y = (src[0].y * reciprocal_depth[0] * w0
                + src[1].y * reciprocal_depth[1] * w1
                + src[2].y * reciprocal_depth[2] * w2)
                / weighted_depth;
            let Some(mut pixel) = sample_layer_bilinear(layer, src_x, src_y) else {
                continue;
            };
            if pixel[3] == 0 {
                continue;
            }
            pixel[3] = ((pixel[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
            blend_pixel_with_mode(canvas, x as u32, y as u32, pixel, blend);
        }
    }
}

pub(crate) fn draw_rgba_image(
    canvas: &mut RgbaImage,
    image: &RgbaImage,
    x: f32,
    y: f32,
    opacity: f32,
) {
    let base_x = x.round() as i32;
    let base_y = y.round() as i32;
    for (src_x, src_y, pixel) in image.enumerate_pixels() {
        let dst_x = base_x + src_x as i32;
        let dst_y = base_y + src_y as i32;
        if dst_x < 0 || dst_y < 0 {
            continue;
        }
        let (dst_x, dst_y) = (dst_x as u32, dst_y as u32);
        if dst_x >= canvas.width() || dst_y >= canvas.height() {
            continue;
        }
        let mut src = pixel.0;
        src[3] = ((src[3] as f32) * opacity).round().clamp(0.0, 255.0) as u8;
        if src[3] == 0 {
            continue;
        }
        blend_pixel(canvas, dst_x, dst_y, src);
    }
}

pub(crate) fn apply_deform_grid(source: &RgbaImage, grid: &EvaluatedDeformGrid) -> RgbaImage {
    let mut out = if grid.preserve_outside {
        source.clone()
    } else {
        RgbaImage::from_pixel(source.width(), source.height(), Rgba([0, 0, 0, 0]))
    };
    if !grid.triangles.is_empty() {
        if grid.preserve_outside {
            // Clear the bind-pose region before compositing the solved mesh so
            // local arm rigs do not leave a duplicate static limb behind.
            for triangle in &grid.triangles {
                if triangle.iter().any(|index| *index >= grid.from.len()) {
                    continue;
                }
                let moved = triangle.iter().any(|index| {
                    let from = grid.from[*index];
                    let to = grid.to[*index];
                    (from.x - to.x).hypot(from.y - to.y) > 0.01
                });
                if !moved {
                    continue;
                }
                clear_deform_triangle(
                    &mut out,
                    [
                        grid.from[triangle[0]],
                        grid.from[triangle[1]],
                        grid.from[triangle[2]],
                    ],
                    true,
                );
            }
        }
        for triangle in &grid.triangles {
            if triangle.iter().any(|index| {
                *index >= grid.from.len()
                    || *index >= grid.sample_from.len()
                    || *index >= grid.to.len()
            }) {
                continue;
            }
            raster_deform_triangle(
                &mut out,
                source,
                [
                    grid.sample_from[triangle[0]],
                    grid.sample_from[triangle[1]],
                    grid.sample_from[triangle[2]],
                ],
                [
                    grid.to[triangle[0]],
                    grid.to[triangle[1]],
                    grid.to[triangle[2]],
                ],
            );
        }
        return out;
    }
    for row in 0..grid.rows - 1 {
        for col in 0..grid.cols - 1 {
            let i00 = row * grid.cols + col;
            let i10 = i00 + 1;
            let i01 = (row + 1) * grid.cols + col;
            let i11 = i01 + 1;
            raster_deform_triangle(
                &mut out,
                source,
                [
                    grid.sample_from[i00],
                    grid.sample_from[i10],
                    grid.sample_from[i11],
                ],
                [grid.to[i00], grid.to[i10], grid.to[i11]],
            );
            raster_deform_triangle(
                &mut out,
                source,
                [
                    grid.sample_from[i00],
                    grid.sample_from[i11],
                    grid.sample_from[i01],
                ],
                [grid.to[i00], grid.to[i11], grid.to[i01]],
            );
        }
    }
    out
}

fn clear_deform_triangle(out: &mut RgbaImage, triangle: [Point2; 3], moved: bool) {
    // Vector artwork is antialiased beyond the mathematical path boundary.
    // Clearing only the exact bind-pose triangles can therefore leave a
    // one-pixel outline of the original limb after a local Puppet move.
    let clear_fringe_px: f32 = if moved { 2.0 } else { 0.0 };
    let min_x = triangle
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .floor() as i32
        - clear_fringe_px.ceil() as i32;
    let min_y = triangle
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .floor() as i32
        - clear_fringe_px.ceil() as i32;
    let max_x = triangle
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32
        + clear_fringe_px.ceil() as i32;
    let max_y = triangle
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32
        + clear_fringe_px.ceil() as i32;
    let denominator = triangle_barycentric_denominator(triangle);
    for y in min_y.max(0)..max_y.min(out.height() as i32) {
        for x in min_x.max(0)..max_x.min(out.width() as i32) {
            let point = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
            let Some((w0, w1, w2)) = triangle_barycentric(point, triangle, denominator) else {
                continue;
            };
            let inside = w0 >= -0.001 && w1 >= -0.001 && w2 >= -0.001;
            let on_antialiased_fringe = triangle
                .iter()
                .zip(triangle.iter().cycle().skip(1))
                .take(3)
                .any(|(start, end)| {
                    clear_fringe_px > 0.0
                        && point_to_segment_distance(point, *start, *end) <= clear_fringe_px
                });
            if inside || on_antialiased_fringe {
                out.put_pixel(x as u32, y as u32, Rgba([0, 0, 0, 0]));
            }
        }
    }
}

fn point_to_segment_distance(point: Point2, start: Point2, end: Point2) -> f32 {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let length_squared = dx * dx + dy * dy;
    if length_squared <= 0.000001 {
        return (point.x - start.x).hypot(point.y - start.y);
    }
    let t =
        (((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared).clamp(0.0, 1.0);
    let nearest_x = start.x + dx * t;
    let nearest_y = start.y + dy * t;
    (point.x - nearest_x).hypot(point.y - nearest_y)
}

fn raster_deform_triangle(
    out: &mut RgbaImage,
    source: &RgbaImage,
    src: [Point2; 3],
    dst: [Point2; 3],
) {
    let min_x = dst
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .floor() as i32
        - 1;
    let min_y = dst
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .floor() as i32
        - 1;
    let max_x = dst
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32
        + 1;
    let max_y = dst
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32
        + 1;

    let x0 = min_x.clamp(0, out.width() as i32);
    let y0 = min_y.clamp(0, out.height() as i32);
    let x1 = max_x.clamp(0, out.width() as i32);
    let y1 = max_y.clamp(0, out.height() as i32);
    if x1 <= x0 || y1 <= y0 {
        return;
    }

    let denom = triangle_barycentric_denominator(dst);
    if denom.abs() <= 0.00001 {
        return;
    }

    for y in y0..y1 {
        for x in x0..x1 {
            let point = Point2::new(x as f32, y as f32);
            let Some((w0, w1, w2)) = triangle_barycentric(point, dst, denom) else {
                continue;
            };
            if w0 < -0.001 || w1 < -0.001 || w2 < -0.001 {
                continue;
            }
            let src_x = src[0].x * w0 + src[1].x * w1 + src[2].x * w2;
            let src_y = src[0].y * w0 + src[1].y * w1 + src[2].y * w2;
            let Some(pixel) = sample_layer_bilinear(source, src_x, src_y) else {
                continue;
            };
            if pixel[3] == 0 {
                continue;
            }
            out.put_pixel(x as u32, y as u32, Rgba(pixel));
        }
    }
}

pub(crate) fn blend_pixel_with_mode(
    canvas: &mut RgbaImage,
    x: u32,
    y: u32,
    src: [u8; 4],
    mode: SceneBlendMode,
) {
    if mode == SceneBlendMode::Normal {
        blend_pixel(canvas, x, y, src);
        return;
    }

    let dst = canvas.get_pixel_mut(x, y);
    let sa = src[3] as f32 / 255.0;
    if sa <= 0.0 {
        return;
    }
    let da = dst[3] as f32 / 255.0;
    let sr = src[0] as f32 / 255.0;
    let sg = src[1] as f32 / 255.0;
    let sb = src[2] as f32 / 255.0;
    let dr = dst[0] as f32 / 255.0;
    let dg = dst[1] as f32 / 255.0;
    let db = dst[2] as f32 / 255.0;
    let blend_channel = |s: f32, d: f32| match mode {
        SceneBlendMode::Normal => s,
        SceneBlendMode::Multiply => s * d,
        SceneBlendMode::Screen => 1.0 - (1.0 - s) * (1.0 - d),
        SceneBlendMode::Add => (s + d).min(1.0),
        SceneBlendMode::Overlay => {
            if d <= 0.5 {
                2.0 * s * d
            } else {
                1.0 - 2.0 * (1.0 - s) * (1.0 - d)
            }
        }
        SceneBlendMode::SoftLight => {
            if s <= 0.5 {
                (1.0 - 2.0 * s) * d * d + 2.0 * s * d
            } else {
                d + (2.0 * s - 1.0) * (d.sqrt() - d)
            }
        }
        SceneBlendMode::ColorDodge => {
            if s >= 0.9999 {
                1.0
            } else {
                (d / (1.0 - s)).min(1.0)
            }
        }
    };
    let br = blend_channel(sr, dr);
    let bg = blend_channel(sg, dg);
    let bb = blend_channel(sb, db);
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        *dst = Rgba([0, 0, 0, 0]);
        return;
    }

    let out_r = (br * sa + dr * da * (1.0 - sa)) / out_a;
    let out_g = (bg * sa + dg * da * (1.0 - sa)) / out_a;
    let out_b = (bb * sa + db * da * (1.0 - sa)) / out_a;
    *dst = Rgba([
        (out_r * 255.0).round().clamp(0.0, 255.0) as u8,
        (out_g * 255.0).round().clamp(0.0, 255.0) as u8,
        (out_b * 255.0).round().clamp(0.0, 255.0) as u8,
        (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
    ]);
}

pub(crate) fn blend_pixel(canvas: &mut RgbaImage, x: u32, y: u32, src: [u8; 4]) {
    let dst = canvas.get_pixel_mut(x, y);
    let (sr, sg, sb, sa) = (src[0] as f32, src[1] as f32, src[2] as f32, src[3] as f32);
    let (dr, dg, db, da) = (dst[0] as f32, dst[1] as f32, dst[2] as f32, dst[3] as f32);

    let sa = sa / 255.0;
    let da = da / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        *dst = Rgba([0, 0, 0, 0]);
        return;
    }

    let out_r = (sr * sa + dr * da * (1.0 - sa)) / out_a;
    let out_g = (sg * sa + dg * da * (1.0 - sa)) / out_a;
    let out_b = (sb * sa + db * da * (1.0 - sa)) / out_a;

    *dst = Rgba([
        out_r.round().clamp(0.0, 255.0) as u8,
        out_g.round().clamp(0.0, 255.0) as u8,
        out_b.round().clamp(0.0, 255.0) as u8,
        (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
    ]);
}
