// =========================================
// =========================================
// crates/motionloom/src/mesh_reference/analysis.rs

use super::{
    AnalyzeImageReferenceRequest, ImageReferenceAnalysis, MESH_REFERENCE_SCHEMA_VERSION, MaskData,
    MeshReferenceDiagnostic, MeshReferenceError, ReferenceContour, ReferenceFeature,
    ReferenceFeatureKind, ReferenceLandmark, ReferenceRegion, SegmentationMode,
};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::io::Cursor;

const MAX_IMAGE_PIXELS: u64 = 40_000_000;

pub fn analyze_image_reference(
    image_bytes: &[u8],
    request: &AnalyzeImageReferenceRequest,
) -> Result<ImageReferenceAnalysis, MeshReferenceError> {
    if request.schema_version != MESH_REFERENCE_SCHEMA_VERSION {
        return Err(MeshReferenceError::Request(format!(
            "unsupported schemaVersion {}; expected {MESH_REFERENCE_SCHEMA_VERSION}",
            request.schema_version
        )));
    }
    if request.image_id.trim().is_empty() || request.view.trim().is_empty() {
        return Err(MeshReferenceError::Request(
            "imageId and view must not be empty".into(),
        ));
    }
    let image = image::load_from_memory(image_bytes)
        .map_err(|error| MeshReferenceError::Image(error.to_string()))?
        .to_rgba8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS {
        return Err(MeshReferenceError::Image(format!(
            "image dimensions {width}x{height} exceed the analysis limit"
        )));
    }
    let mut diagnostics = vec![];
    let mask = segment(&image, request, &mut diagnostics)?;
    let regions = connected_regions(
        &mask,
        width,
        height,
        request.segmentation.min_component_pixels,
    );
    if regions.is_empty() {
        diagnostics.push(diagnostic(
            "error",
            "NO_FOREGROUND",
            "segmentation produced no foreground component",
        ));
    }
    let contours = trace_contours(
        &mask,
        width,
        height,
        request.segmentation.max_contour_points.max(16),
    );
    let landmarks = geometric_landmarks(&contours, &request.requested_landmarks);
    let features = analyze_features(&image, &mask, request, &mut diagnostics)?;
    let internal_edge_mask = encode_mask(&internal_edges(&image, &mask), width, height);
    let coverage = mask.iter().filter(|&&pixel| pixel).count() as f32
        / (width as f32 * height as f32).max(1.0);
    if !(0.01..=0.95).contains(&coverage) {
        diagnostics.push(diagnostic(
            "warning",
            "SUSPICIOUS_COVERAGE",
            &format!("foreground covers {:.1}% of the image", coverage * 100.0),
        ));
    }
    diagnostics.push(diagnostic(
        "info",
        "DEPTH_UNKNOWN",
        "a single RGB image does not provide trustworthy metric depth; add multiview or host depth hints",
    ));
    let confidence = if regions.is_empty() {
        0.0
    } else {
        let mode = match request.segmentation.mode {
            SegmentationMode::Mask | SegmentationMode::Alpha => 0.98,
            SegmentationMode::Guided => 0.85,
            SegmentationMode::BackgroundColor => 0.78,
            SegmentationMode::Auto => 0.62,
        };
        mode * (1.0 - (coverage - 0.45).abs() * 0.25).clamp(0.5, 1.0)
    };
    Ok(ImageReferenceAnalysis {
        schema_version: MESH_REFERENCE_SCHEMA_VERSION.into(),
        image_id: request.image_id.clone(),
        content_hash: format!("{:x}", Sha256::digest(image_bytes)),
        image_size: [width, height],
        view: request.view.clone(),
        foreground_mask: encode_mask(&mask, width, height),
        contours,
        regions,
        landmarks,
        features,
        internal_edge_mask: Some(internal_edge_mask),
        depth_hints: if request.depth_hints.is_empty() {
            vec![super::DepthHint {
                relation: "unknown".into(),
                subject: "foreground".into(),
                object: None,
                evidence: "singleRgbInsufficient".into(),
                value: None,
                confidence: 0.0,
            }]
        } else {
            request.depth_hints.clone()
        },
        confidence,
        diagnostics,
    })
}

fn analyze_features(
    image: &RgbaImage,
    foreground: &[bool],
    request: &AnalyzeImageReferenceRequest,
    diagnostics: &mut Vec<MeshReferenceDiagnostic>,
) -> Result<Vec<ReferenceFeature>, MeshReferenceError> {
    let (width, height) = image.dimensions();
    if request.analysis_profile.as_deref() == Some("human_head_v1")
        && request.feature_hints.is_empty()
    {
        diagnostics.push(diagnostic(
            "warning",
            "HEAD_FEATURE_HINTS_REQUIRED",
            "human_head_v1 requires host- or LLM-supplied semantic feature hints; the core does not invent facial landmarks",
        ));
    }
    request
        .feature_hints
        .iter()
        .map(|hint| {
            if hint.id.trim().is_empty() || hint.points.is_empty() {
                return Err(MeshReferenceError::Request(
                    "feature hints require a non-empty id and at least one point".into(),
                ));
            }
            if hint.kind == ReferenceFeatureKind::Point && hint.points.len() != 1 {
                return Err(MeshReferenceError::Request(format!(
                    "point feature {} must contain exactly one point",
                    hint.id
                )));
            }
            if matches!(
                hint.kind,
                ReferenceFeatureKind::ClosedContour | ReferenceFeatureKind::Region
            ) && hint.points.len() < 3
            {
                return Err(MeshReferenceError::Request(format!(
                    "closed feature {} requires at least three points",
                    hint.id
                )));
            }
            let mut confidence = hint.confidence.clamp(0.0, 1.0);
            let mut points = Vec::with_capacity(hint.points.len());
            for point in &hint.points {
                if !point[0].is_finite()
                    || !point[1].is_finite()
                    || point[0] < 0.0
                    || point[1] < 0.0
                    || point[0] >= width as f32
                    || point[1] >= height as f32
                {
                    return Err(MeshReferenceError::Request(format!(
                        "feature {} contains a point outside the image",
                        hint.id
                    )));
                }
                let snapped = snap_to_edge(image, foreground, *point, hint.snap_radius);
                confidence *= snapped.1;
                points.push(snapped.0);
            }
            Ok(ReferenceFeature {
                id: hint.id.clone(),
                kind: hint.kind,
                points,
                semantic_label: hint.semantic_label.clone(),
                binding: hint.binding.clone(),
                confidence,
            })
        })
        .collect()
}

fn internal_edges(image: &RgbaImage, foreground: &[bool]) -> Vec<bool> {
    let (width, height) = image.dimensions();
    let mut edges = vec![false; foreground.len()];
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let index = (y * width + x) as usize;
            if !foreground[index] {
                continue;
            }
            let neighbors_are_foreground = foreground[(y * width + x - 1) as usize]
                && foreground[(y * width + x + 1) as usize]
                && foreground[((y - 1) * width + x) as usize]
                && foreground[((y + 1) * width + x) as usize];
            edges[index] = neighbors_are_foreground && gradient_strength(image, x, y) >= 28.0;
        }
    }
    edges
}

fn snap_to_edge(
    image: &RgbaImage,
    foreground: &[bool],
    point: [f32; 2],
    radius: u32,
) -> ([f32; 2], f32) {
    let (width, height) = image.dimensions();
    let center = [point[0].round() as i32, point[1].round() as i32];
    let radius = radius.min(64) as i32;
    let mut best = (point, 0.0_f32, f32::INFINITY);
    for y in (center[1] - radius).max(1)..=(center[1] + radius).min(height as i32 - 2) {
        for x in (center[0] - radius).max(1)..=(center[0] + radius).min(width as i32 - 2) {
            if !foreground[(y as u32 * width + x as u32) as usize] {
                continue;
            }
            let strength = gradient_strength(image, x as u32, y as u32);
            let distance = ((x as f32 - point[0]).powi(2) + (y as f32 - point[1]).powi(2)).sqrt();
            let score = strength - distance * 3.0;
            if score > best.1 || (score == best.1 && distance < best.2) {
                best = ([x as f32, y as f32], score, distance);
            }
        }
    }
    if best.1 <= 0.0 {
        (point, 0.55)
    } else {
        (best.0, (0.65 + best.1 / 512.0).clamp(0.65, 1.0))
    }
}

fn gradient_strength(image: &RgbaImage, x: u32, y: u32) -> f32 {
    let luminance = |x: u32, y: u32| {
        let pixel = image[(x, y)];
        pixel[0] as f32 * 0.2126 + pixel[1] as f32 * 0.7152 + pixel[2] as f32 * 0.0722
    };
    let gx = luminance(x + 1, y) - luminance(x - 1, y);
    let gy = luminance(x, y + 1) - luminance(x, y - 1);
    (gx * gx + gy * gy).sqrt()
}

pub fn analyze_image_reference_json(
    image_bytes: &[u8],
    request_json: &str,
) -> Result<String, MeshReferenceError> {
    let request = serde_json::from_str(request_json)?;
    Ok(serde_json::to_string(&analyze_image_reference(
        image_bytes,
        &request,
    )?)?)
}

fn segment(
    image: &RgbaImage,
    request: &AnalyzeImageReferenceRequest,
    diagnostics: &mut Vec<MeshReferenceDiagnostic>,
) -> Result<Vec<bool>, MeshReferenceError> {
    let (width, height) = image.dimensions();
    if let Some(mask) = &request.segmentation.supplied_mask {
        if request.segmentation.mode != SegmentationMode::Mask {
            return Err(MeshReferenceError::Request(
                "suppliedMask requires segmentation.mode=mask".into(),
            ));
        }
        if [mask.width, mask.height] != [width, height] {
            return Err(MeshReferenceError::Request(
                "supplied mask dimensions must match the image".into(),
            ));
        }
        return decode_mask(mask);
    }
    if request.segmentation.mode == SegmentationMode::Mask {
        return Err(MeshReferenceError::Request(
            "segmentation.mode=mask requires suppliedMask".into(),
        ));
    }
    let roi = checked_roi(request.segmentation.analysis_region, width, height)?;
    if request.segmentation.mode == SegmentationMode::Alpha {
        return Ok((0..height)
            .flat_map(|y| (0..width).map(move |x| inside(x, y, roi) && image[(x, y)][3] >= 128))
            .collect());
    }
    let background = request
        .segmentation
        .background_color
        .map(|rgb| [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32])
        .unwrap_or_else(|| sample_background(image, request, roi));
    let threshold = request.segmentation.threshold.clamp(1.0, 441.0);
    let mut mask = vec![false; width as usize * height as usize];
    for y in roi[1]..roi[1] + roi[3] {
        for x in roi[0]..roi[0] + roi[2] {
            let pixel = image[(x, y)];
            let distance = ((pixel[0] as f32 - background[0]).powi(2)
                + (pixel[1] as f32 - background[1]).powi(2)
                + (pixel[2] as f32 - background[2]).powi(2))
            .sqrt();
            mask[(y * width + x) as usize] = pixel[3] >= 8 && distance >= threshold;
        }
    }
    if request.segmentation.mode == SegmentationMode::Guided
        && !request.segmentation.foreground_points.is_empty()
    {
        let components = label_components(&mask, width, height);
        let mut wanted = vec![false; components.len()];
        for &[x, y] in &request.segmentation.foreground_points {
            if x < width && y < height {
                let index = (y * width + x) as usize;
                if let Some(label) = components[index] {
                    wanted[label] = true;
                }
            }
        }
        if wanted.iter().any(|&value| value) {
            for (index, value) in mask.iter_mut().enumerate() {
                *value = components[index].is_some_and(|label| wanted[label]);
            }
        } else {
            diagnostics.push(diagnostic(
                "warning",
                "GUIDE_MISSED",
                "foreground guide points did not land inside the threshold mask",
            ));
        }
    }
    for &[x, y] in &request.segmentation.background_points {
        if x < width && y < height {
            mask[(y * width + x) as usize] = false;
        }
    }
    fill_small_holes(
        &mut mask,
        width,
        height,
        roi,
        request.segmentation.min_component_pixels.saturating_mul(8),
    );
    Ok(mask)
}

fn fill_small_holes(mask: &mut [bool], width: u32, height: u32, roi: [u32; 4], maximum_area: u32) {
    let mut seen = vec![false; mask.len()];
    for y in roi[1]..roi[1] + roi[3] {
        for x in roi[0]..roi[0] + roi[2] {
            let start = (y * width + x) as usize;
            if mask[start] || seen[start] {
                continue;
            }
            let mut queue = VecDeque::from([start]);
            let mut component = vec![];
            let mut touches_edge = false;
            seen[start] = true;
            while let Some(index) = queue.pop_front() {
                component.push(index);
                let px = index as u32 % width;
                let py = index as u32 / width;
                touches_edge |= px == roi[0]
                    || py == roi[1]
                    || px + 1 == roi[0] + roi[2]
                    || py + 1 == roi[1] + roi[3];
                for (nx, ny) in neighbors(px, py, width, height) {
                    if !inside(nx, ny, roi) {
                        continue;
                    }
                    let next = (ny * width + nx) as usize;
                    if !mask[next] && !seen[next] {
                        seen[next] = true;
                        queue.push_back(next);
                    }
                }
            }
            if !touches_edge && component.len() as u32 <= maximum_area {
                for index in component {
                    mask[index] = true;
                }
            }
        }
    }
}

fn checked_roi(
    roi: Option<[u32; 4]>,
    width: u32,
    height: u32,
) -> Result<[u32; 4], MeshReferenceError> {
    let roi = roi.unwrap_or([0, 0, width, height]);
    if roi[2] == 0
        || roi[3] == 0
        || roi[0].saturating_add(roi[2]) > width
        || roi[1].saturating_add(roi[3]) > height
    {
        return Err(MeshReferenceError::Request(
            "analysisRegion must be a non-empty [x,y,width,height] inside the image".into(),
        ));
    }
    Ok(roi)
}

fn inside(x: u32, y: u32, roi: [u32; 4]) -> bool {
    x >= roi[0] && y >= roi[1] && x < roi[0] + roi[2] && y < roi[1] + roi[3]
}

fn sample_background(
    image: &RgbaImage,
    request: &AnalyzeImageReferenceRequest,
    roi: [u32; 4],
) -> [f32; 3] {
    let mut samples = request.segmentation.background_points.clone();
    if samples.is_empty() {
        let x0 = roi[0];
        let y0 = roi[1];
        let x1 = roi[0] + roi[2] - 1;
        let y1 = roi[1] + roi[3] - 1;
        samples = vec![[x0, y0], [x1, y0], [x0, y1], [x1, y1]];
    }
    let mut values = [vec![], vec![], vec![]];
    for [x, y] in samples {
        if x >= image.width() || y >= image.height() {
            continue;
        }
        let pixel = image[(x, y)];
        for channel in 0..3 {
            values[channel].push(pixel[channel] as f32);
        }
    }
    std::array::from_fn(|channel| {
        values[channel].sort_by(|a, b| a.total_cmp(b));
        values[channel]
            .get(values[channel].len() / 2)
            .copied()
            .unwrap_or(0.0)
    })
}

fn label_components(mask: &[bool], width: u32, height: u32) -> Vec<Option<usize>> {
    let mut labels = vec![None; mask.len()];
    let mut label = 0;
    for start in 0..mask.len() {
        if !mask[start] || labels[start].is_some() {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        labels[start] = Some(label);
        while let Some(index) = queue.pop_front() {
            let x = index as u32 % width;
            let y = index as u32 / width;
            for (nx, ny) in neighbors(x, y, width, height) {
                let next = (ny * width + nx) as usize;
                if mask[next] && labels[next].is_none() {
                    labels[next] = Some(label);
                    queue.push_back(next);
                }
            }
        }
        label += 1;
    }
    labels
}

fn connected_regions(mask: &[bool], width: u32, height: u32, minimum: u32) -> Vec<ReferenceRegion> {
    let labels = label_components(mask, width, height);
    let count = labels.iter().flatten().copied().max().map_or(0, |v| v + 1);
    let mut bounds = vec![[width, height, 0, 0]; count];
    let mut areas = vec![0_u32; count];
    for (index, label) in labels.into_iter().enumerate() {
        let Some(label) = label else { continue };
        let x = index as u32 % width;
        let y = index as u32 / width;
        areas[label] += 1;
        bounds[label][0] = bounds[label][0].min(x);
        bounds[label][1] = bounds[label][1].min(y);
        bounds[label][2] = bounds[label][2].max(x);
        bounds[label][3] = bounds[label][3].max(y);
    }
    let mut regions: Vec<_> = areas
        .into_iter()
        .enumerate()
        .filter(|(_, area)| *area >= minimum)
        .map(|(index, area)| ReferenceRegion {
            id: format!("region_{index}"),
            bounds: [
                bounds[index][0],
                bounds[index][1],
                bounds[index][2] - bounds[index][0] + 1,
                bounds[index][3] - bounds[index][1] + 1,
            ],
            area,
            parent: None,
            semantic_label: None,
            confidence: 1.0,
        })
        .collect();
    regions.sort_by_key(|region| std::cmp::Reverse(region.area));
    regions
}

fn neighbors(x: u32, y: u32, width: u32, height: u32) -> impl Iterator<Item = (u32, u32)> {
    let mut result = [(0, 0); 4];
    let mut count = 0;
    if x > 0 {
        result[count] = (x - 1, y);
        count += 1;
    }
    if x + 1 < width {
        result[count] = (x + 1, y);
        count += 1;
    }
    if y > 0 {
        result[count] = (x, y - 1);
        count += 1;
    }
    if y + 1 < height {
        result[count] = (x, y + 1);
        count += 1;
    }
    result.into_iter().take(count)
}

fn trace_contours(mask: &[bool], width: u32, height: u32, maximum: usize) -> Vec<ReferenceContour> {
    type Point = (u32, u32);
    let mut edges = HashMap::<Point, Vec<Point>>::new();
    let foreground = |x: i64, y: i64| {
        x >= 0
            && y >= 0
            && x < width as i64
            && y < height as i64
            && mask[(y as u32 * width + x as u32) as usize]
    };
    for y in 0..height {
        for x in 0..width {
            if !foreground(x as i64, y as i64) {
                continue;
            }
            let candidates = [
                ((x, y), (x + 1, y), !foreground(x as i64, y as i64 - 1)),
                (
                    (x + 1, y),
                    (x + 1, y + 1),
                    !foreground(x as i64 + 1, y as i64),
                ),
                (
                    (x + 1, y + 1),
                    (x, y + 1),
                    !foreground(x as i64, y as i64 + 1),
                ),
                ((x, y + 1), (x, y), !foreground(x as i64 - 1, y as i64)),
            ];
            for (a, b, boundary) in candidates {
                if boundary {
                    edges.entry(a).or_default().push(b);
                }
            }
        }
    }
    let mut contours = vec![];
    while let Some((&start, _)) = edges.iter().find(|(_, values)| !values.is_empty()) {
        let mut points = vec![start];
        let mut current = start;
        let mut closed = false;
        for _ in 0..(width as usize * height as usize * 4).min(4_000_000) {
            let Some(next) = edges.get_mut(&current).and_then(Vec::pop) else {
                break;
            };
            current = next;
            if current == start {
                closed = true;
                break;
            }
            points.push(current);
        }
        if points.len() < 4 {
            continue;
        }
        let stride = points.len().div_ceil(maximum).max(1);
        let sampled: Vec<_> = points
            .into_iter()
            .step_by(stride)
            .map(|(x, y)| [x as f32, y as f32])
            .collect();
        let area = polygon_area(&sampled);
        contours.push(ReferenceContour {
            id: format!("contour_{}", contours.len()),
            closed,
            hole: area < 0.0,
            points: sampled,
            confidence: if closed { 1.0 } else { 0.4 },
        });
    }
    contours.sort_by(|a, b| {
        polygon_area(&b.points)
            .abs()
            .total_cmp(&polygon_area(&a.points).abs())
    });
    contours
}

fn polygon_area(points: &[[f32; 2]]) -> f32 {
    (0..points.len())
        .map(|index| {
            let a = points[index];
            let b = points[(index + 1) % points.len()];
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f32>()
        * 0.5
}

fn geometric_landmarks(
    contours: &[ReferenceContour],
    requested: &[String],
) -> Vec<ReferenceLandmark> {
    let Some(contour) = contours.iter().find(|contour| !contour.hole) else {
        return vec![];
    };
    let wants = |name: &str| requested.is_empty() || requested.iter().any(|value| value == name);
    let candidates = [
        ("topmost", 1, false),
        ("bottommost", 1, true),
        ("leftmost", 0, false),
        ("rightmost", 0, true),
    ];
    let mut result: Vec<_> = candidates
        .into_iter()
        .filter(|(id, _, _)| wants(id))
        .filter_map(|(id, axis, maximum)| {
            contour
                .points
                .iter()
                .copied()
                .min_by(|a, b| {
                    let order = a[axis].total_cmp(&b[axis]);
                    if maximum { order.reverse() } else { order }
                })
                .map(|pixel| ReferenceLandmark {
                    id: id.into(),
                    pixel,
                    semantic_label: None,
                    binding: None,
                    confidence: 0.95,
                })
        })
        .collect();
    if requested.iter().any(|value| value == "corners") && contour.points.len() >= 9 {
        let step = (contour.points.len() / 32).max(2);
        let mut corners = (0..contour.points.len())
            .map(|index| {
                let previous =
                    contour.points[(index + contour.points.len() - step) % contour.points.len()];
                let point = contour.points[index];
                let next = contour.points[(index + step) % contour.points.len()];
                let a = normalize2([previous[0] - point[0], previous[1] - point[1]]);
                let b = normalize2([next[0] - point[0], next[1] - point[1]]);
                (1.0 - a[0] * b[0] - a[1] * b[1], index, point)
            })
            .collect::<Vec<_>>();
        corners.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut accepted = vec![];
        for (score, index, pixel) in corners {
            if accepted.iter().any(|&other: &usize| {
                let distance = index.abs_diff(other);
                distance.min(contour.points.len() - distance) < step * 2
            }) {
                continue;
            }
            let id = format!("corner_{}", accepted.len());
            result.push(ReferenceLandmark {
                id,
                pixel,
                semantic_label: None,
                binding: None,
                confidence: (score / 2.0).clamp(0.2, 0.9),
            });
            accepted.push(index);
            if accepted.len() == 8 {
                break;
            }
        }
    }
    result
}

fn normalize2(value: [f32; 2]) -> [f32; 2] {
    let length = (value[0] * value[0] + value[1] * value[1]).sqrt().max(1e-8);
    [value[0] / length, value[1] / length]
}

pub(crate) fn encode_mask(mask: &[bool], width: u32, height: u32) -> MaskData {
    let starts_foreground = mask.first().copied().unwrap_or(false);
    let mut runs = vec![];
    let mut current = starts_foreground;
    let mut count = 0_u32;
    for &pixel in mask {
        if pixel == current {
            count += 1;
        } else {
            runs.push(count);
            current = pixel;
            count = 1;
        }
    }
    if !mask.is_empty() {
        runs.push(count);
    }
    MaskData {
        width,
        height,
        starts_foreground,
        runs,
    }
}

pub fn decode_mask(mask: &MaskData) -> Result<Vec<bool>, MeshReferenceError> {
    let expected = u64::from(mask.width) * u64::from(mask.height);
    if expected > MAX_IMAGE_PIXELS {
        return Err(MeshReferenceError::Request(
            "mask exceeds pixel limit".into(),
        ));
    }
    if mask.runs.iter().map(|&run| u64::from(run)).sum::<u64>() != expected {
        return Err(MeshReferenceError::Request(
            "mask RLE length does not match width*height".into(),
        ));
    }
    let mut result = Vec::with_capacity(expected as usize);
    let mut value = mask.starts_foreground;
    for &run in &mask.runs {
        result.extend(std::iter::repeat_n(value, run as usize));
        value = !value;
    }
    Ok(result)
}

pub fn analysis_mask_png(analysis: &ImageReferenceAnalysis) -> Result<Vec<u8>, MeshReferenceError> {
    mask_png(&analysis.foreground_mask)
}

pub fn analysis_internal_edge_png(
    analysis: &ImageReferenceAnalysis,
) -> Result<Vec<u8>, MeshReferenceError> {
    let mask = analysis.internal_edge_mask.as_ref().ok_or_else(|| {
        MeshReferenceError::Image("analysis does not contain an internal edge mask".into())
    })?;
    mask_png(mask)
}

fn mask_png(data: &MaskData) -> Result<Vec<u8>, MeshReferenceError> {
    let mask = decode_mask(data)?;
    let mut image = RgbaImage::new(data.width, data.height);
    for (index, pixel) in image.pixels_mut().enumerate() {
        *pixel = if mask[index] {
            Rgba([255, 255, 255, 255])
        } else {
            Rgba([0, 0, 0, 255])
        };
    }
    encode_png(image)
}

pub fn analysis_overlay_png(
    image_bytes: &[u8],
    analysis: &ImageReferenceAnalysis,
) -> Result<Vec<u8>, MeshReferenceError> {
    let mut image = image::load_from_memory(image_bytes)
        .map_err(|error| MeshReferenceError::Image(error.to_string()))?
        .to_rgba8();
    if [image.width(), image.height()] != analysis.image_size {
        return Err(MeshReferenceError::Image(
            "analysis and image dimensions differ".into(),
        ));
    }
    let mask = decode_mask(&analysis.foreground_mask)?;
    for (index, pixel) in image.pixels_mut().enumerate() {
        if mask[index] {
            pixel[0] = ((u16::from(pixel[0]) + 80) / 2) as u8;
            pixel[1] = ((u16::from(pixel[1]) + 255) / 2) as u8;
            pixel[2] = ((u16::from(pixel[2]) + 80) / 2) as u8;
        }
    }
    for contour in &analysis.contours {
        for point in &contour.points {
            let x = point[0].round() as i32;
            let y = point[1].round() as i32;
            if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
                image[(x as u32, y as u32)] = Rgba([255, 40, 40, 255]);
            }
        }
    }
    for feature in &analysis.features {
        for point in &feature.points {
            let x = point[0].round() as i32;
            let y = point[1].round() as i32;
            for offset_y in -2..=2 {
                for offset_x in -2..=2 {
                    let px = x + offset_x;
                    let py = y + offset_y;
                    if px >= 0 && py >= 0 && px < image.width() as i32 && py < image.height() as i32
                    {
                        image[(px as u32, py as u32)] = Rgba([40, 220, 255, 255]);
                    }
                }
            }
        }
    }
    encode_png(image)
}

fn encode_png(image: RgbaImage) -> Result<Vec<u8>, MeshReferenceError> {
    let mut bytes = Cursor::new(vec![]);
    DynamicImage::ImageRgba8(image)
        .write_to(&mut bytes, ImageFormat::Png)
        .map_err(|error| MeshReferenceError::Image(error.to_string()))?;
    Ok(bytes.into_inner())
}

fn diagnostic(severity: &str, code: &str, message: &str) -> MeshReferenceDiagnostic {
    MeshReferenceDiagnostic {
        severity: severity.into(),
        code: code.into(),
        message: message.into(),
    }
}
