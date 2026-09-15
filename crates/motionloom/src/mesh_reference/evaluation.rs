// =========================================
// =========================================
// crates/motionloom/src/mesh_reference/evaluation.rs

use super::{
    FeatureBinding, FeatureResidual, LandmarkBinding, LandmarkResidual, MESH_REFERENCE_MEASUREMENT,
    MESH_REFERENCE_SCHEMA_VERSION, MeshObjectiveComponents, MeshReferenceDiagnostic,
    MeshReferenceError, MeshReferenceEvaluation, MeshReferenceSet, MeshReferenceViewEvaluation,
    ReferenceFeatureKind, VertexResidual, decode_mask, mesh_source_fingerprint,
    mesh_topology_signature,
};
use crate::parse_graph_script;
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::Cursor;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Snapshot {
    model_id: String,
    asset_id: String,
    positions: Vec<[f32; 3]>,
    faces: Vec<Vec<u32>>,
    origin: [f32; 3],
    basis: Vec<[f32; 3]>,
    size: [u32; 2],
    center: [f32; 2],
    focal: f32,
    near: f32,
}

pub async fn evaluate_mesh_asset_reference(
    source: &str,
    request: &MeshReferenceSet,
) -> Result<MeshReferenceEvaluation, MeshReferenceError> {
    validate_request(request)?;
    let graph = parse_graph_script(source)
        .map_err(|error| MeshReferenceError::Candidate(error.to_string()))?;
    let cage = super::proposal::mesh_asset(source, &request.target_asset_id)?;
    let mut views = vec![];
    let mut camera_hash = Sha256::new();
    for reference in &request.references {
        let value = crate::scene::render::mesh_edit_snapshot(
            &graph,
            &request.target_model_id,
            reference.frame,
        )
        .await
        .map_err(|error| MeshReferenceError::Evaluation(error.to_string()))?;
        let snapshot: Snapshot = serde_json::from_value(value)?;
        if snapshot.asset_id != request.target_asset_id
            || snapshot.model_id != request.target_model_id
        {
            return Err(MeshReferenceError::Evaluation(
                "runtime snapshot resolved a different model or asset".into(),
            ));
        }
        camera_hash.update(serde_json::to_vec(&serde_json::json!({
            "frame": reference.frame,
            "origin": snapshot.origin,
            "basis": snapshot.basis,
            "size": snapshot.size,
            "center": snapshot.center,
            "focal": snapshot.focal,
            "near": snapshot.near,
        }))?);
        views.push(evaluate_view(reference, &snapshot, &request.options)?);
    }
    let objective = if views.is_empty() {
        0.0
    } else {
        views.iter().map(|view| view.error).sum::<f32>() / views.len() as f32
    };
    let reference_set_fingerprint = hash_json(&serde_json::json!({
        "targetAssetId": request.target_asset_id,
        "targetModelId": request.target_model_id,
        "references": request.references.iter().map(|reference| serde_json::json!({
            "id": reference.id,
            "frame": reference.frame,
            "imageId": reference.analysis.image_id,
            "contentHash": reference.analysis.content_hash,
            "features": reference.analysis.features,
            "depthHints": reference.analysis.depth_hints,
        })).collect::<Vec<_>>(),
    }))?;
    let metric_profile_fingerprint = hash_json(&serde_json::json!({
        "weights": request.options.metric_weights,
        "qualityGates": request.options.quality_gates,
    }))?;
    let camera_fingerprint = format!("{:x}", camera_hash.finalize());
    let source_fingerprint = mesh_source_fingerprint(source);
    let topology_signature = mesh_topology_signature(&cage);
    let evaluation_fingerprint = evaluation_context_fingerprint(
        &source_fingerprint,
        &topology_signature,
        &camera_fingerprint,
        &reference_set_fingerprint,
        &metric_profile_fingerprint,
    );
    Ok(MeshReferenceEvaluation {
        schema_version: MESH_REFERENCE_SCHEMA_VERSION.into(),
        measurement: MESH_REFERENCE_MEASUREMENT.into(),
        source_fingerprint,
        topology_signature,
        camera_fingerprint,
        reference_set_fingerprint,
        metric_profile_fingerprint,
        evaluation_fingerprint,
        target_asset_id: request.target_asset_id.clone(),
        target_model_id: request.target_model_id.clone(),
        objective,
        passes_quality_gates: views.iter().all(|view| view.passes_quality_gates),
        views,
        diagnostics: vec![],
    })
}

pub async fn evaluate_mesh_asset_reference_json(
    source: &str,
    request_json: &str,
) -> Result<String, MeshReferenceError> {
    let request = serde_json::from_str(request_json)?;
    Ok(serde_json::to_string(
        &evaluate_mesh_asset_reference(source, &request).await?,
    )?)
}

fn validate_request(request: &MeshReferenceSet) -> Result<(), MeshReferenceError> {
    if request.schema_version != MESH_REFERENCE_SCHEMA_VERSION {
        return Err(MeshReferenceError::Request(format!(
            "unsupported schemaVersion {}; expected {MESH_REFERENCE_SCHEMA_VERSION}",
            request.schema_version
        )));
    }
    if request.target_asset_id.trim().is_empty() || request.target_model_id.trim().is_empty() {
        return Err(MeshReferenceError::Request(
            "targetAssetId and targetModelId must not be empty".into(),
        ));
    }
    if request.references.is_empty() || request.references.len() > 8 {
        return Err(MeshReferenceError::Request(
            "one through eight reference views are required".into(),
        ));
    }
    if request.options.camera_policy != "frozen" {
        return Err(MeshReferenceError::Request(
            "cameraPolicy must be frozen while fitting geometry".into(),
        ));
    }
    for reference in &request.references {
        if reference.analysis.schema_version != MESH_REFERENCE_SCHEMA_VERSION {
            return Err(MeshReferenceError::Request(format!(
                "reference {} has an unsupported schema version",
                reference.id
            )));
        }
    }
    Ok(())
}

fn evaluate_view(
    reference: &super::MeshReferenceView,
    snapshot: &Snapshot,
    options: &super::MeshReferenceOptions,
) -> Result<MeshReferenceViewEvaluation, MeshReferenceError> {
    let width = reference.analysis.image_size[0];
    let height = reference.analysis.image_size[1];
    let target = decode_mask(&reference.analysis.foreground_mask)?;
    let projected: Vec<_> = snapshot
        .positions
        .iter()
        .map(|&position| project(snapshot, position, [width, height]))
        .collect();
    let (candidate, depth) = rasterize(snapshot, &projected, width, height);
    let (intersection, union, reference_pixels, candidate_pixels) = target
        .iter()
        .zip(&candidate)
        .fold((0_u32, 0_u32, 0_u32, 0_u32), |mut sums, (&a, &b)| {
            sums.0 += u32::from(a && b);
            sums.1 += u32::from(a || b);
            sums.2 += u32::from(a);
            sums.3 += u32::from(b);
            sums
        });
    let mask_iou = if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    };
    let target_boundary = boundary_points(&target, width, height, 2048);
    let candidate_boundary = boundary_points(&candidate, width, height, 2048);
    let mut distances = bidirectional_distances(&target_boundary, &candidate_boundary);
    distances.sort_by(f32::total_cmp);
    let mean = if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f32>() / distances.len() as f32
    };
    let p95 = percentile(&distances, 0.95);
    let maximum = distances.last().copied().unwrap_or(0.0);
    let landmarks = reference
        .analysis
        .landmarks
        .iter()
        .map(|landmark| {
            let (candidate, method) = match &landmark.binding {
                Some(LandmarkBinding::NearestContour) | None => (
                    nearest(landmark.pixel, &candidate_boundary),
                    "nearestContour",
                ),
                Some(binding) => (binding_point(binding, snapshot, &projected), "binding"),
            };
            let delta =
                candidate.map(|point| [landmark.pixel[0] - point[0], landmark.pixel[1] - point[1]]);
            LandmarkResidual {
                id: landmark.id.clone(),
                reference: landmark.pixel,
                candidate,
                delta,
                distance_px: delta.map(length2),
                confidence: landmark.confidence
                    * if matches!(
                        landmark.binding.as_ref(),
                        Some(LandmarkBinding::Vertex { .. })
                            | Some(LandmarkBinding::Edge { .. })
                            | Some(LandmarkBinding::Face { .. })
                    ) {
                        1.0
                    } else {
                        0.45
                    },
                method: method.into(),
            }
        })
        .collect::<Vec<_>>();
    let features = reference
        .analysis
        .features
        .iter()
        .map(|feature| feature_residual(feature, snapshot, &projected, &candidate_boundary))
        .collect::<Vec<_>>();
    let depth_violations = depth_violations(reference, snapshot, &projected);
    let mut vertex_residuals = projected
        .iter()
        .enumerate()
        .filter_map(|(vertex, point)| {
            let point = *point.as_ref()?;
            let screen_point = [point[0], point[1]];
            let nearest = nearest(screen_point, &target_boundary)?;
            let delta = [nearest[0] - screen_point[0], nearest[1] - screen_point[1]];
            let visibility = vertex_visibility(point, width, height, &depth);
            if visibility <= 0.0 {
                return None;
            }
            Some(VertexResidual {
                vertex,
                screen_point,
                screen_delta: delta,
                distance_px: length2(delta),
                visibility,
                confidence: 0.7 * visibility,
            })
        })
        .collect::<Vec<_>>();
    vertex_residuals.sort_by(|a, b| b.distance_px.total_cmp(&a.distance_px));
    vertex_residuals.truncate(options.max_vertex_residuals.min(2048));
    let landmark_error = weighted_landmark_error(&landmarks);
    let feature_error = weighted_feature_error(&features);
    let diagonal = ((width as f32).powi(2) + (height as f32).powi(2))
        .sqrt()
        .max(1.0);
    let components = MeshObjectiveComponents {
        silhouette: 1.0 - mask_iou,
        boundary: mean / diagonal,
        landmarks: landmark_error / diagonal,
        features: feature_error / diagonal,
        depth: if reference.analysis.depth_hints.is_empty() {
            0.0
        } else {
            depth_violations as f32 / reference.analysis.depth_hints.len() as f32
        },
    };
    let weights = &options.metric_weights;
    let active_feature_weight = if features
        .iter()
        .any(|feature| feature.mean_distance_px.is_some())
    {
        weights.features
    } else {
        0.0
    };
    let active_depth_weight = if reference
        .analysis
        .depth_hints
        .iter()
        .any(|hint| hint.confidence > 0.0 && hint.object.is_some())
    {
        weights.depth
    } else {
        0.0
    };
    let weight_sum = (weights.silhouette
        + weights.boundary
        + weights.landmarks
        + active_feature_weight
        + active_depth_weight)
        .max(1e-6);
    let error = (components.silhouette * weights.silhouette
        + components.boundary * weights.boundary
        + components.landmarks * weights.landmarks
        + components.features * active_feature_weight
        + components.depth * active_depth_weight)
        / weight_sum;
    let gates = &options.quality_gates;
    let landmark_mean = weighted_landmark_error(&landmarks);
    let passes_quality_gates = mask_iou >= gates.minimum_mask_iou
        && p95 <= gates.maximum_p95_edge_distance_px
        && landmark_mean <= gates.maximum_landmark_mean_distance_px
        && (feature_error <= gates.maximum_feature_mean_distance_px
            || !features
                .iter()
                .any(|feature| feature.mean_distance_px.is_some()))
        && depth_violations <= gates.maximum_depth_violations;
    Ok(MeshReferenceViewEvaluation {
        id: reference.id.clone(),
        view: reference.analysis.view.clone(),
        frame: reference.frame,
        mask_iou,
        mean_edge_distance_px: mean,
        p95_edge_distance_px: p95,
        maximum_edge_distance_px: maximum,
        reference_pixels,
        candidate_pixels,
        candidate_mask: super::analysis::encode_mask(&candidate, width, height),
        landmarks,
        features,
        depth_violations,
        objective_components: components,
        passes_quality_gates,
        vertex_residuals,
        error,
    })
}

fn feature_residual(
    feature: &super::ReferenceFeature,
    snapshot: &Snapshot,
    projected: &[Option<[f32; 3]>],
    candidate_boundary: &[[f32; 2]],
) -> FeatureResidual {
    let (candidate_points, method) = match feature.binding.as_ref() {
        Some(FeatureBinding::NearestContour) | None => (
            feature
                .points
                .iter()
                .filter_map(|&point| nearest(point, candidate_boundary))
                .collect::<Vec<_>>(),
            "nearestContour",
        ),
        Some(binding) => (
            feature_binding_points(binding, snapshot, projected)
                .into_iter()
                .map(|point| [point[0], point[1]])
                .collect::<Vec<_>>(),
            "binding",
        ),
    };
    let mut distances = if candidate_points.is_empty() {
        vec![]
    } else if feature.kind == ReferenceFeatureKind::Point {
        vec![length2([
            feature.points[0][0] - candidate_points[0][0],
            feature.points[0][1] - candidate_points[0][1],
        ])]
    } else {
        bidirectional_distances(&feature.points, &candidate_points)
    };
    distances.sort_by(f32::total_cmp);
    FeatureResidual {
        id: feature.id.clone(),
        kind: feature.kind,
        reference_points: feature.points.clone(),
        candidate_points,
        mean_distance_px: (!distances.is_empty())
            .then(|| distances.iter().sum::<f32>() / distances.len() as f32),
        p95_distance_px: (!distances.is_empty()).then(|| percentile(&distances, 0.95)),
        maximum_distance_px: distances.last().copied(),
        confidence: feature.confidence,
        method: method.into(),
    }
}

fn feature_binding_points(
    binding: &FeatureBinding,
    snapshot: &Snapshot,
    projected: &[Option<[f32; 3]>],
) -> Vec<[f32; 3]> {
    let point = |index: usize| projected.get(index).copied().flatten();
    match binding {
        FeatureBinding::Vertex { vertex } => point(*vertex).into_iter().collect(),
        FeatureBinding::Edge { vertices, t } => point(vertices[0])
            .zip(point(vertices[1]))
            .map(|(a, b)| {
                std::array::from_fn(|axis| a[axis] + (b[axis] - a[axis]) * t.clamp(0.0, 1.0))
            })
            .into_iter()
            .collect(),
        FeatureBinding::Face { face, barycentric } => snapshot
            .faces
            .get(*face)
            .filter(|indices| indices.len() >= 3)
            .and_then(|indices| {
                Some([
                    point(indices[0] as usize)?,
                    point(indices[1] as usize)?,
                    point(indices[2] as usize)?,
                ])
            })
            .map(|points| {
                std::array::from_fn(|axis| {
                    points[0][axis] * barycentric[0]
                        + points[1][axis] * barycentric[1]
                        + points[2][axis] * barycentric[2]
                })
            })
            .into_iter()
            .collect(),
        FeatureBinding::VertexChain { vertices, .. } => {
            vertices.iter().filter_map(|&index| point(index)).collect()
        }
        FeatureBinding::NearestContour => vec![],
    }
}

fn depth_violations(
    reference: &super::MeshReferenceView,
    snapshot: &Snapshot,
    projected: &[Option<[f32; 3]>],
) -> usize {
    let depth = |id: &str| {
        reference
            .analysis
            .features
            .iter()
            .find(|feature| feature.id == id)
            .and_then(|feature| feature.binding.as_ref())
            .map(|binding| feature_binding_points(binding, snapshot, projected))
            .filter(|points| !points.is_empty())
            .map(|points| points.iter().map(|point| point[2]).sum::<f32>() / points.len() as f32)
    };
    reference
        .analysis
        .depth_hints
        .iter()
        .filter(|hint| hint.confidence > 0.0)
        .filter_map(|hint| {
            let object = hint.object.as_deref()?;
            Some((hint, depth(&hint.subject)?, depth(object)?))
        })
        .filter(|(hint, subject, object)| match hint.relation.as_str() {
            "inFrontOf" | "closerThan" => subject >= object,
            "behind" | "fartherThan" => subject <= object,
            _ => false,
        })
        .count()
}

fn weighted_feature_error(features: &[FeatureResidual]) -> f32 {
    let (sum, weight) = features.iter().fold((0.0, 0.0), |(sum, weight), feature| {
        match feature.mean_distance_px {
            Some(distance) => (
                sum + distance * feature.confidence,
                weight + feature.confidence,
            ),
            None => (sum, weight),
        }
    });
    if weight <= 1e-6 { 0.0 } else { sum / weight }
}

fn hash_json(value: &serde_json::Value) -> Result<String, MeshReferenceError> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

pub(crate) fn evaluation_context_fingerprint(
    source_fingerprint: &str,
    topology_signature: &str,
    camera_fingerprint: &str,
    reference_set_fingerprint: &str,
    metric_profile_fingerprint: &str,
) -> String {
    let mut hash = Sha256::new();
    for value in [
        source_fingerprint,
        topology_signature,
        camera_fingerprint,
        reference_set_fingerprint,
        metric_profile_fingerprint,
    ] {
        hash.update(value.as_bytes());
        hash.update([0]);
    }
    format!("{:x}", hash.finalize())
}

pub fn mesh_evaluation_overlay_png(
    analysis: &super::ImageReferenceAnalysis,
    evaluation: &MeshReferenceViewEvaluation,
) -> Result<Vec<u8>, MeshReferenceError> {
    evaluation_mask_png(analysis, evaluation, false)
}

pub fn mesh_evaluation_difference_png(
    analysis: &super::ImageReferenceAnalysis,
    evaluation: &MeshReferenceViewEvaluation,
) -> Result<Vec<u8>, MeshReferenceError> {
    evaluation_mask_png(analysis, evaluation, true)
}

fn evaluation_mask_png(
    analysis: &super::ImageReferenceAnalysis,
    evaluation: &MeshReferenceViewEvaluation,
    difference_only: bool,
) -> Result<Vec<u8>, MeshReferenceError> {
    if analysis.image_size
        != [
            evaluation.candidate_mask.width,
            evaluation.candidate_mask.height,
        ]
    {
        return Err(MeshReferenceError::Evaluation(
            "analysis and evaluation mask dimensions differ".into(),
        ));
    }
    let reference = decode_mask(&analysis.foreground_mask)?;
    let candidate = decode_mask(&evaluation.candidate_mask)?;
    let mut image = RgbaImage::new(analysis.image_size[0], analysis.image_size[1]);
    for (index, pixel) in image.pixels_mut().enumerate() {
        *pixel = if difference_only {
            if reference[index] == candidate[index] {
                Rgba([0, 0, 0, 255])
            } else {
                Rgba([255, 255, 255, 255])
            }
        } else {
            match (reference[index], candidate[index]) {
                (true, true) => Rgba([245, 245, 245, 255]),
                (true, false) => Rgba([40, 220, 90, 255]),
                (false, true) => Rgba([230, 40, 180, 255]),
                (false, false) => Rgba([20, 20, 24, 255]),
            }
        };
    }
    let mut bytes = Cursor::new(vec![]);
    DynamicImage::ImageRgba8(image)
        .write_to(&mut bytes, ImageFormat::Png)
        .map_err(|error| MeshReferenceError::Evaluation(error.to_string()))?;
    Ok(bytes.into_inner())
}

fn project(snapshot: &Snapshot, position: [f32; 3], target_size: [u32; 2]) -> Option<[f32; 3]> {
    if snapshot.basis.len() != 3 {
        return None;
    }
    let mut view = snapshot.origin;
    for (axis, value) in position.iter().enumerate() {
        for (component, output) in view.iter_mut().enumerate() {
            *output += snapshot.basis[axis][component] * value;
        }
    }
    if view[2] <= snapshot.near {
        return None;
    }
    let x = snapshot.center[0] + view[0] * snapshot.focal / view[2];
    let y = snapshot.center[1] - view[1] * snapshot.focal / view[2];
    Some([
        x * target_size[0] as f32 / snapshot.size[0].max(1) as f32,
        y * target_size[1] as f32 / snapshot.size[1].max(1) as f32,
        view[2],
    ])
}

fn rasterize(
    snapshot: &Snapshot,
    projected: &[Option<[f32; 3]>],
    width: u32,
    height: u32,
) -> (Vec<bool>, Vec<f32>) {
    let mut mask = vec![false; width as usize * height as usize];
    let mut depth = vec![f32::INFINITY; mask.len()];
    for face in &snapshot.faces {
        if face.len() < 3 {
            continue;
        }
        for index in 1..face.len() - 1 {
            let ids = [
                face[0] as usize,
                face[index] as usize,
                face[index + 1] as usize,
            ];
            let Some(points) = ids
                .map(|id| projected.get(id).copied().flatten())
                .into_iter()
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            raster_triangle(
                points[0], points[1], points[2], width, height, &mut depth, &mut mask,
            );
        }
    }
    (mask, depth)
}

fn vertex_visibility(point: [f32; 3], width: u32, height: u32, depth: &[f32]) -> f32 {
    let x = point[0].round() as i32;
    let y = point[1].round() as i32;
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return 0.0;
    }
    let nearest_depth = depth[y as usize * width as usize + x as usize];
    if !nearest_depth.is_finite() {
        0.0
    } else if point[2] <= nearest_depth + nearest_depth.abs().max(1.0) * 0.002 {
        1.0
    } else {
        0.0
    }
}

fn raster_triangle(
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    width: u32,
    height: u32,
    depth: &mut [f32],
    mask: &mut [bool],
) {
    let minimum_x = a[0].min(b[0]).min(c[0]).floor().max(0.0) as u32;
    let maximum_x = a[0]
        .max(b[0])
        .max(c[0])
        .ceil()
        .min(width.saturating_sub(1) as f32) as u32;
    let minimum_y = a[1].min(b[1]).min(c[1]).floor().max(0.0) as u32;
    let maximum_y = a[1]
        .max(b[1])
        .max(c[1])
        .ceil()
        .min(height.saturating_sub(1) as f32) as u32;
    let area = edge(a, b, [c[0], c[1]]);
    if area.abs() < 1e-8 {
        return;
    }
    for y in minimum_y..=maximum_y {
        for x in minimum_x..=maximum_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let wa = edge(b, c, point) / area;
            let wb = edge(c, a, point) / area;
            let wc = edge(a, b, point) / area;
            if wa >= -1e-5 && wb >= -1e-5 && wc >= -1e-5 {
                let z = wa * a[2] + wb * b[2] + wc * c[2];
                let pixel = (y * width + x) as usize;
                if z < depth[pixel] {
                    depth[pixel] = z;
                    mask[pixel] = true;
                }
            }
        }
    }
}

fn edge(a: [f32; 3], b: [f32; 3], p: [f32; 2]) -> f32 {
    (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
}

fn boundary_points(mask: &[bool], width: u32, height: u32, maximum: usize) -> Vec<[f32; 2]> {
    let mut points = vec![];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if !mask[index] {
                continue;
            }
            let boundary = x == 0
                || y == 0
                || x + 1 == width
                || y + 1 == height
                || !mask[(y * width + x.saturating_sub(1)) as usize]
                || !mask[(y * width + (x + 1).min(width - 1)) as usize]
                || !mask[(y.saturating_sub(1) * width + x) as usize]
                || !mask[((y + 1).min(height - 1) * width + x) as usize];
            if boundary {
                points.push([x as f32 + 0.5, y as f32 + 0.5]);
            }
        }
    }
    if points.len() <= maximum {
        return points;
    }
    let stride = points.len().div_ceil(maximum);
    points.into_iter().step_by(stride).collect()
}

fn bidirectional_distances(a: &[[f32; 2]], b: &[[f32; 2]]) -> Vec<f32> {
    let mut result: Vec<_> = a
        .iter()
        .filter_map(|&point| {
            nearest(point, b).map(|other| length2([point[0] - other[0], point[1] - other[1]]))
        })
        .collect();
    result.extend(b.iter().filter_map(|&point| {
        nearest(point, a).map(|other| length2([point[0] - other[0], point[1] - other[1]]))
    }));
    result
}

fn nearest(point: [f32; 2], candidates: &[[f32; 2]]) -> Option<[f32; 2]> {
    candidates
        .iter()
        .copied()
        .min_by(|a, b| squared_distance(point, *a).total_cmp(&squared_distance(point, *b)))
}

fn squared_distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn length2(value: [f32; 2]) -> f32 {
    (value[0].powi(2) + value[1].powi(2)).sqrt()
}

fn percentile(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values[((values.len() - 1) as f32 * percentile.clamp(0.0, 1.0)).round() as usize]
}

fn binding_point(
    binding: &LandmarkBinding,
    snapshot: &Snapshot,
    projected: &[Option<[f32; 3]>],
) -> Option<[f32; 2]> {
    let point = |index: usize| {
        projected
            .get(index)
            .copied()
            .flatten()
            .map(|p| [p[0], p[1]])
    };
    match binding {
        LandmarkBinding::Vertex { vertex } => point(*vertex),
        LandmarkBinding::Edge { vertices, t } => {
            let a = point(vertices[0])?;
            let b = point(vertices[1])?;
            Some([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
        }
        LandmarkBinding::Face { face, barycentric } => {
            let indices = snapshot.faces.get(*face)?;
            if indices.len() < 3 {
                return None;
            }
            let points = [
                point(indices[0] as usize)?,
                point(indices[1] as usize)?,
                point(indices[2] as usize)?,
            ];
            Some([
                points[0][0] * barycentric[0]
                    + points[1][0] * barycentric[1]
                    + points[2][0] * barycentric[2],
                points[0][1] * barycentric[0]
                    + points[1][1] * barycentric[1]
                    + points[2][1] * barycentric[2],
            ])
        }
        LandmarkBinding::NearestContour => None,
    }
}

fn weighted_landmark_error(landmarks: &[LandmarkResidual]) -> f32 {
    let (sum, weight) = landmarks
        .iter()
        .fold((0.0, 0.0), |(sum, weight), landmark| {
            let next = landmark.distance_px.unwrap_or(0.0) * landmark.confidence;
            (
                sum + next,
                weight
                    + if landmark.distance_px.is_some() {
                        landmark.confidence
                    } else {
                        0.0
                    },
            )
        });
    if weight <= 1e-6 { 0.0 } else { sum / weight }
}

#[allow(dead_code)]
fn diagnostic(severity: &str, code: &str, message: &str) -> MeshReferenceDiagnostic {
    MeshReferenceDiagnostic {
        severity: severity.into(),
        code: code.into(),
        message: message.into(),
    }
}

#[cfg(test)]
pub(crate) fn evaluate_masks_for_test(
    a: &[bool],
    b: &[bool],
    width: u32,
    height: u32,
) -> (f32, f32) {
    let intersection = a.iter().zip(b).filter(|(a, b)| **a && **b).count();
    let union = a.iter().zip(b).filter(|(a, b)| **a || **b).count();
    let iou = if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    };
    let aa = boundary_points(a, width, height, 2048);
    let bb = boundary_points(b, width, height, 2048);
    let distances = bidirectional_distances(&aa, &bb);
    let mean = if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f32>() / distances.len() as f32
    };
    (iou, mean)
}
