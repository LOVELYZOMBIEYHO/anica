// =========================================
// =========================================
// crates/motionloom/src/head_fitting/projection.rs

use super::validation::{active, distance, height, normalize};
use super::*;
use crate::dsl::{FaceLayoutNode, HeadMorphNode, HeadShapeNode};
use crate::world::primitive::generate_primitive_mesh;
const N: usize = 128;
const CELL: f64 = 2.0 / N as f64;

// A tracked cage vertex is fixed for the entire optimization, including HeadMorph.
pub(super) fn anchors(a: &PrimitiveAssetNode) -> Option<usize> {
    let PrimitiveGeometry::HeadSurface {
        segments,
        rings,
        features,
        face_layout,
        morph,
        ..
    } = &a.geometry
    else {
        return None;
    };
    let center = features
        .iter()
        .find(|f| f.kind == "nose_tip")
        .map(|f| f.center)
        .or_else(|| {
            face_layout
                .as_ref()
                .and_then(|f| f.noses.first())
                .map(|nose| [nose.position[0], nose.position[1] * morph.face_height, 0.96])
        })?;
    let mesh = generate_primitive_mesh(a);
    if !a.modifiers.is_empty()
        || mesh.positions.len()
            != ((*segments).max(12) as usize + 1) * ((*rings).max(8) as usize + 1)
    {
        return None;
    }
    let stride = (*segments).max(12) as usize + 1;
    let rings = (*rings).max(8) as usize;
    // Restrict the initial extremum to the declared nasal region; never track another feature.
    mesh.positions
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let phi = (*i / stride) as f32 / rings as f32 * std::f32::consts::PI;
            let theta = (*i % stride) as f32 / (stride - 1) as f32 * std::f32::consts::TAU;
            let unit = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
            unit[0].abs() < 0.12 && (unit[1] - center[1]).abs() < 0.14 && unit[2] > 0.7
        })
        .max_by(|a, b| a.1[2].total_cmp(&b.1[2]))
        .map(|(i, _)| i)
}
fn project(p: [f32; 3], view: View, h: f64, cy: f64) -> [f64; 2] {
    let u = match view {
        View::Front => p[0],
        View::Left => -p[2],
        View::Right => p[2],
        View::Back => -p[0],
    };
    [u as f64 / h, (p[1] as f64 - cy) / h]
}
fn at(x: usize, y: usize) -> [f64; 2] {
    [
        -1.0 + (x as f64 + 0.5) * CELL,
        -1.0 + (y as f64 + 0.5) * CELL,
    ]
}
fn cross(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}
fn raster(vertices: &[[f64; 2]], indices: &[u32]) -> Vec<bool> {
    let mut mask = vec![false; N * N];
    // Rasterize triangle union, including concavity and internal occlusion, not vertex hull.
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [
            vertices[tri[0] as usize],
            vertices[tri[1] as usize],
            vertices[tri[2] as usize],
        ];
        if cross(a, b, c).abs() < 1e-12 {
            continue;
        }
        let ix = |v: f64| ((v + 1.0) / CELL).floor().clamp(0.0, (N - 1) as f64) as usize;
        let minx = ix(a[0].min(b[0]).min(c[0]));
        let maxx = ix(a[0].max(b[0]).max(c[0]));
        let miny = ix(a[1].min(b[1]).min(c[1]));
        let maxy = ix(a[1].max(b[1]).max(c[1]));
        for y in miny..=maxy {
            for x in minx..=maxx {
                let p = at(x, y);
                let e = [cross(a, b, p), cross(b, c, p), cross(c, a, p)];
                if e.iter().all(|v| *v >= 0.0) || e.iter().all(|v| *v <= 0.0) {
                    mask[y * N + x] = true
                }
            }
        }
    }
    mask
}
fn polygon_mask(points: &[[f64; 2]]) -> Vec<bool> {
    (0..N * N)
        .map(|i| {
            let p = at(i % N, i / N);
            let mut inside = false;
            for j in 0..points.len() {
                let (a, b) = (points[j], points[(j + 1) % points.len()]);
                if (a[1] > p[1]) != (b[1] > p[1])
                    && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
                {
                    inside = !inside
                }
            }
            inside
        })
        .collect()
}
fn boundary(mask: &[bool]) -> Vec<[f64; 2]> {
    (0..N * N)
        .filter(|&i| {
            let (x, y) = (i % N, i / N);
            mask[i]
                && (x == 0
                    || y == 0
                    || x == N - 1
                    || y == N - 1
                    || !mask[i - 1]
                    || !mask[i + 1]
                    || !mask[i - N]
                    || !mask[i + N])
        })
        .map(|i| at(i % N, i / N))
        .collect()
}
fn chamfer(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let directed = |a: &[[f64; 2]], b: &[[f64; 2]]| {
        a.iter()
            .map(|p| {
                b.iter()
                    .map(|q| distance(*p, *q))
                    .fold(f64::INFINITY, f64::min)
            })
            .sum::<f64>()
            / a.len() as f64
    };
    (directed(a, b) + directed(b, a)) * 0.5
}

// Projection fitting passes independent semantic and camera inputs explicitly.
#[allow(clippy::too_many_arguments)]
fn semantic_curve(
    id: &str,
    view: View,
    layout: &FaceLayoutNode,
    morph: &HeadMorphNode,
    shape: &HeadShapeNode,
    h: f64,
    cy: f64,
    projected_mesh: &[[f64; 2]],
) -> Option<Vec<[f64; 2]>> {
    let half_x = shape.size[0] as f64 * 0.5 * morph.head_width as f64 * morph.face_width as f64;
    let half_y = shape.size[1] as f64 * 0.5 * morph.head_height as f64;
    if half_x <= 1e-6 || half_y <= 1e-6 {
        return None;
    }
    let side = if id.ends_with("_left") { 1.0 } else { -1.0 };
    let eye = layout
        .eyes
        .iter()
        .max_by(|a, b| (a.position[0] * side as f32).total_cmp(&(b.position[0] * side as f32)));
    let nose = layout.noses.first();
    let mouth = layout.mouths.first();
    let ear = layout
        .ears
        .iter()
        .max_by(|a, b| (a.position[0] * side as f32).total_cmp(&(b.position[0] * side as f32)));
    if (id.starts_with("eye_") || id.starts_with("iris_") || id.contains("lid")) && eye.is_none() {
        return None;
    }
    if (id.starts_with("mouth") || id.contains("lip")) && mouth.is_none() {
        return None;
    }
    if id.starts_with("nose") && nose.is_none() {
        return None;
    }
    if id.starts_with("ear") && ear.is_none() {
        return None;
    }
    let eye_y = eye.map_or(0.0, |e| e.position[1]) as f64 * morph.face_height as f64 * half_y;
    let eye_cx = side
        * eye.map_or(0.0, |e| e.position[0].abs() * 2.0) as f64
        * 0.5
        * morph.face_width as f64
        * half_x;
    let eye_rx = eye.map_or(0.0, |e| e.width).max(0.04) as f64 * 0.5 * half_x;
    let eye_ry = eye.map_or(0.0, |e| e.opening).max(0.02) as f64 * 0.5 * half_y;
    let world = |x: f64, y: f64| [x / h, (y - cy) / h];
    if matches!(id, "ear_outline_left" | "ear_outline_right") && view == View::Front {
        let ear_cx = side * half_x * 0.94;
        let ear_cy = eye_y - half_y * 0.03;
        let rx = ear.map_or(0.0, |e| e.width).max(0.04) as f64 * half_x * 0.55;
        let ry = ear.map_or(0.0, |e| e.height).max(0.08) as f64 * half_y * 0.5;
        return Some(
            (0..=24)
                .map(|i| {
                    let a = i as f64 / 24.0 * std::f64::consts::TAU;
                    world(ear_cx + rx * a.cos(), ear_cy + ry * a.sin())
                })
                .collect(),
        );
    }
    if matches!(
        id,
        "upper_eyelid_left" | "upper_eyelid_right" | "lower_eyelid_left" | "lower_eyelid_right"
    ) && view == View::Front
    {
        let upper = id.starts_with("upper_");
        return Some(
            (0..=24)
                .map(|i| {
                    let t = -1.0 + i as f64 / 12.0;
                    let fold = if upper { 0.03 * half_y } else { 0.0 };
                    let arch = (1.0 - t * t).max(0.0) * (eye_ry + fold);
                    let y = eye_y
                        + if upper { arch } else { -arch }
                        + t * eye.map_or(0.0, |e| e.tilt) as f64 * half_y;
                    world(eye_cx + t * eye_rx, y)
                })
                .collect(),
        );
    }
    if matches!(id, "iris_left" | "iris_right") && view == View::Front {
        let eye = eye?;
        let iris = eye.iris.as_ref()?;
        let (tilt_sin, tilt_cos) = (eye.tilt as f64).to_radians().sin_cos();
        let center_x = eye_cx
            + (iris.position[0] as f64 * tilt_cos - iris.position[1] as f64 * tilt_sin) * half_x;
        let center_y = eye_y
            + (iris.position[0] as f64 * tilt_sin + iris.position[1] as f64 * tilt_cos) * half_y;
        let radius_x = iris.radius.max(0.015) as f64 * iris.scale[0] as f64 * half_x;
        let radius_y = iris.radius.max(0.015) as f64 * iris.scale[1] as f64 * half_y;
        let count = 32;
        return Some(
            (0..=count)
                .map(|i| {
                    let a = i as f64 / count as f64 * std::f64::consts::TAU;
                    let (mut x, mut y) = (a.cos(), a.sin());
                    if iris.shape == "square" {
                        let edge = x.abs().max(y.abs()).max(1e-9);
                        x /= edge;
                        y /= edge;
                    }
                    world(center_x + radius_x * x, center_y + radius_y * y)
                })
                .collect(),
        );
    }
    if matches!(id, "upper_lip" | "lower_lip") && view == View::Front {
        let upper = id == "upper_lip";
        let half_w = mouth.map_or(0.0, |m| m.width).max(0.04) as f64 * 0.5 * half_x;
        let mouth_y =
            mouth.map_or(0.0, |m| m.position[1]) as f64 * morph.face_height as f64 * half_y;
        let lip = if upper {
            mouth.map_or(0.0, |m| m.upper_lip)
        } else {
            mouth.map_or(0.0, |m| m.lower_lip)
        }
        .max(0.002) as f64
            * half_y;
        let opening = mouth.map_or(0.0, |m| m.opening).max(0.0) as f64 * half_y * 0.5;
        return Some(
            (0..=24)
                .map(|i| {
                    let t = -1.0 + i as f64 / 12.0;
                    let arch = (1.0 - t * t).max(0.0) * lip;
                    let y = mouth_y
                        + if upper {
                            opening + arch
                        } else {
                            -opening - arch
                        };
                    world(t * half_w, y)
                })
                .collect(),
        );
    }
    if id == "nose_profile" && matches!(view, View::Left | View::Right) {
        // Side-view nose is derived from the deformed cage, retaining the continuous
        // surface rather than fitting an unrelated line primitive.
        let mut out = Vec::new();
        let lo = (eye_y - half_y * 0.55 - cy) / h;
        let hi = (eye_y + half_y * 0.35 - cy) / h;
        for bin in 0..=24 {
            let y = lo + (hi - lo) * bin as f64 / 24.0;
            let points: Vec<_> = projected_mesh
                .iter()
                .copied()
                .filter(|p| (p[1] - y).abs() < 0.025)
                .collect();
            let p = if view == View::Left {
                points.into_iter().min_by(|a, b| a[0].total_cmp(&b[0]))
            } else {
                points.into_iter().max_by(|a, b| a[0].total_cmp(&b[0]))
            };
            if let Some(p) = p {
                out.push(p);
            }
        }
        return (out.len() >= 2).then_some(out);
    }
    None
}

fn semantic_landmark(
    id: &str,
    view: View,
    layout: &FaceLayoutNode,
    morph: &HeadMorphNode,
    shape: &HeadShapeNode,
    h: f64,
    cy: f64,
) -> Option<([f64; 2], &'static str)> {
    let half_x = shape.size[0] as f64 * 0.5 * morph.head_width as f64 * morph.face_width as f64;
    let half_y = shape.size[1] as f64 * 0.5 * morph.head_height as f64;
    let world = |x: f64, y: f64| [x / h, (y - cy) / h];
    if view != View::Front && id.starts_with("eye_") {
        return None;
    }
    let side = if id.ends_with("_left") { 1.0 } else { -1.0 };
    let eye = layout
        .eyes
        .iter()
        .max_by(|a, b| (a.position[0] * side as f32).total_cmp(&(b.position[0] * side as f32)));
    let nose = layout.noses.first();
    let mouth = layout.mouths.first();
    let ear = layout
        .ears
        .iter()
        .max_by(|a, b| (a.position[0] * side as f32).total_cmp(&(b.position[0] * side as f32)));
    if (id.starts_with("eye_") || id.starts_with("iris_") || id.contains("lid")) && eye.is_none() {
        return None;
    }
    if (id.starts_with("mouth") || id.contains("lip")) && mouth.is_none() {
        return None;
    }
    if id.starts_with("nose") && nose.is_none() {
        return None;
    }
    if id.starts_with("ear") && ear.is_none() {
        return None;
    }
    let eye_y = eye.map_or(0.0, |e| e.position[1]) as f64 * morph.face_height as f64 * half_y;
    let eye_cx = side
        * eye.map_or(0.0, |e| e.position[0].abs() * 2.0) as f64
        * 0.5
        * morph.face_width as f64
        * half_x;
    let eye_rx = eye.map_or(0.0, |e| e.width).max(0.04) as f64 * 0.5 * half_x;
    if matches!(id, "ear_center_left" | "ear_center_right") && view == View::Front {
        return Some((
            world(side * half_x * 0.94, eye_y - half_y * 0.03),
            "face_layout_ear_center",
        ));
    }
    match id {
        "eye_inner_left" | "eye_outer_left" | "eye_inner_right" | "eye_outer_right" => {
            let inner = id.contains("inner");
            let x = if side > 0.0 {
                eye_cx + if inner { -eye_rx } else { eye_rx }
            } else {
                eye_cx + if inner { eye_rx } else { -eye_rx }
            };
            Some((world(x, eye_y), "face_layout_eye_aperture"))
        }
        "iris_center_left" | "iris_center_right" => {
            let eye = eye?;
            let iris = eye.iris.as_ref()?;
            let (sin, cos) = (eye.tilt as f64).to_radians().sin_cos();
            let x =
                eye_cx + (iris.position[0] as f64 * cos - iris.position[1] as f64 * sin) * half_x;
            let y =
                eye_y + (iris.position[0] as f64 * sin + iris.position[1] as f64 * cos) * half_y;
            Some((world(x, y), "face_layout_iris_center"))
        }
        "mouth_corner_left" | "mouth_corner_right" => Some((
            world(
                side * mouth.map_or(0.0, |m| m.width).max(0.04) as f64 * 0.5 * half_x,
                mouth.map_or(0.0, |m| m.position[1]) as f64 * morph.face_height as f64 * half_y,
            ),
            "face_layout_mouth_width",
        )),
        "upper_lip_center" | "lower_lip_center" => Some((
            world(
                0.0,
                mouth.map_or(0.0, |m| m.position[1]) as f64 * morph.face_height as f64 * half_y
                    + if id.starts_with("upper") {
                        (mouth.map_or(0.0, |m| m.opening) + mouth.map_or(0.0, |m| m.upper_lip))
                            as f64
                            * half_y
                            * 0.5
                    } else {
                        -(mouth.map_or(0.0, |m| m.opening) + mouth.map_or(0.0, |m| m.lower_lip))
                            as f64
                            * half_y
                            * 0.5
                    },
            ),
            "face_layout_lip_curve",
        )),
        "nose_base" => Some((
            world(
                0.0,
                (eye.map_or(0.0, |e| e.position[1]) - nose.map_or(0.0, |n| n.length) * 0.92) as f64
                    * morph.face_height as f64
                    * half_y,
            ),
            "face_layout_nose_base",
        )),
        _ => None,
    }
}

pub(super) fn evaluate(
    a: &PrimitiveAssetNode,
    r: &HeadReferenceSet,
    nose: &Option<usize>,
    mut diagnostics: Vec<Diagnostic>,
) -> Result<FitMetrics, HeadFitError> {
    let mesh = generate_primitive_mesh(a);
    if mesh.positions.is_empty() || mesh.positions.iter().flatten().any(|v| !v.is_finite()) {
        return Err(HeadFitError::Candidate("invalid generated mesh".into()));
    }
    let lo = mesh
        .positions
        .iter()
        .fold([f64::INFINITY; 3], |mut acc, p| {
            for i in 0..3 {
                acc[i] = acc[i].min(p[i] as f64)
            }
            acc
        });
    let hi = mesh
        .positions
        .iter()
        .fold([f64::NEG_INFINITY; 3], |mut acc, p| {
            for i in 0..3 {
                acc[i] = acc[i].max(p[i] as f64)
            }
            acc
        });
    let h = hi[1] - lo[1];
    let cy = (hi[1] + lo[1]) * 0.5;
    if h < 1e-5 || (hi[0] - lo[0]) / h < 0.2 || (hi[2] - lo[2]) / h < 0.2 {
        return Err(HeadFitError::Candidate("collapsed or extreme mesh".into()));
    }
    // Nonzero area faces must have finite bounded edge lengths; polar seam degeneracies are expected.
    let max_edge = mesh
        .indices
        .chunks_exact(3)
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .map(|(i, j)| {
            (0..3)
                .map(|k| {
                    ((mesh.positions[i as usize][k] - mesh.positions[j as usize][k]) as f64 / h)
                        .powi(2)
                })
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    let geometry_penalty = (max_edge - 0.3).max(0.0).powi(2);
    let layout_data = match &a.geometry {
        PrimitiveGeometry::HeadSurface {
            head_shape,
            face_layout: Some(layout),
            morph,
            ..
        } => Some((layout, morph, head_shape)),
        _ => None,
    };
    let mut views = vec![];
    let mut total = 0.0;
    let mut weights = 0.0;
    for v in &r.references {
        let vertices: Vec<_> = mesh
            .positions
            .iter()
            .map(|p| project(*p, v.view, h, cy))
            .collect();
        if vertices.iter().flatten().any(|v| v.abs() >= 0.98) {
            return Err(HeadFitError::Candidate(
                "mesh exceeds analysis domain".into(),
            ));
        }
        let mask = raster(&vertices, &mesh.indices);
        let outline = boundary(&mask);
        if outline.is_empty() {
            return Err(HeadFitError::Candidate("empty projected mesh".into()));
        }
        let contour = v
            .contours
            .iter()
            .find(|c| c.id == "head_outline" && active(c.enabled, c.confidence, c.weight));
        let mut view_error = 0.0;
        let mut view_weight = 0.0;
        let mut ref_contour = vec![];
        let mut cd = None;
        let mut iou = None;
        if let Some(c) = contour {
            ref_contour = c.points.iter().map(|p| normalize(v, *p)).collect();
            let rm = polygon_mask(&ref_contour);
            let rb = boundary(&rm);
            if rb.is_empty() {
                return Err(HeadFitError::References(
                    "contour below raster resolution".into(),
                ));
            }
            cd = Some(chamfer(&outline, &rb));
            let intersection = mask.iter().zip(&rm).filter(|(a, b)| **a && **b).count();
            let union = mask.iter().zip(&rm).filter(|(a, b)| **a || **b).count();
            iou = Some(intersection as f64 / union as f64);
            let w = c.weight * c.confidence;
            view_error += (cd.unwrap() + 0.1 * (1.0 - iou.unwrap())) * w;
            view_weight += w;
        }
        let mut curves = vec![];
        for c in &v.contours {
            if c.id == "head_outline" || !active(c.enabled, c.confidence, c.weight) {
                continue;
            }
            let reference = c
                .points
                .iter()
                .map(|p| normalize(v, *p))
                .collect::<Vec<_>>();
            let candidate = layout_data.and_then(|(layout, morph, shape)| {
                semantic_curve(&c.id, v.view, layout, morph, shape, h, cy, &vertices)
            });
            let d = candidate.as_ref().map(|p| chamfer(&reference, p));
            if let Some(d) = d {
                let w = c.weight * c.confidence;
                view_error += d * w;
                view_weight += w;
            }
            curves.push(CurveFit {
                id: c.id.clone(),
                reference,
                candidate,
                pixel_distance: d.map(|d| d * height(v)),
                distance: d,
                method: if d.is_some() {
                    "analytic_face_layout_curve"
                } else {
                    "unsupported_or_occluded"
                }
                .into(),
            });
        }
        let mut landmarks = vec![];
        for l in &v.landmarks {
            if !active(l.enabled, l.confidence, l.weight) {
                continue;
            }
            let reference = normalize(v, l.pixel);
            let semantic = layout_data.and_then(|(layout, morph, shape)| {
                semantic_landmark(&l.id, v.view, layout, morph, shape, h, cy)
            });
            let candidate = match l.id.as_str() {
                "head_top" => mesh
                    .positions
                    .iter()
                    .max_by(|a, b| a[1].total_cmp(&b[1]))
                    .map(|p| project(*p, v.view, h, cy)),
                "chin" => mesh
                    .positions
                    .iter()
                    .min_by(|a, b| a[1].total_cmp(&b[1]))
                    .map(|p| project(*p, v.view, h, cy)),
                "nose_tip" if v.view != View::Back => nose.and_then(|i| vertices.get(i).copied()),
                _ => semantic.map(|(p, _)| p),
            };
            let d = candidate.map(|p| distance(p, reference));
            let method = if candidate.is_none() {
                "unsupported_or_occluded"
            } else if l.id == "nose_tip" {
                "tracked_nasal_surface_anchor_low_confidence"
            } else if let Some((_, method)) = semantic {
                method
            } else {
                "mesh_vertical_extremum"
            };
            if let Some(d) = d {
                let w = l.weight * l.confidence * if l.id == "nose_tip" { 0.35 } else { 1.0 };
                view_error += d * w;
                view_weight += w;
            }
            landmarks.push(LandmarkFit {
                id: l.id.clone(),
                reference,
                candidate,
                distance: d,
                pixel_distance: d.map(|d| d * height(v)),
                method: method.into(),
            });
        }
        if view_weight == 0.0 {
            diagnostics.push(diagnostic(
                "warning",
                "UNCONSTRAINED_VIEW",
                format!(
                    "{}: no measurable observations; error is null, not zero",
                    v.id
                ),
            ));
        }
        total += view_error;
        weights += view_weight;
        let reference_width_over_height = (!ref_contour.is_empty()).then(|| {
            ref_contour
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max)
                - ref_contour
                    .iter()
                    .map(|p| p[0])
                    .fold(f64::INFINITY, f64::min)
        });
        let candidate_width_over_height = vertices
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max)
            - vertices.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        views.push(ViewFit {
            reference_width_over_height,
            candidate_width_over_height,
            id: v.id.clone(),
            view: v.view,
            contour_distance: cd,
            contour_pixel_distance: cd.map(|d| d * height(v)),
            mask_iou: iou,
            reference_contour: ref_contour,
            candidate_contour: outline,
            landmarks,
            curves,
            error: (view_weight > 0.0).then(|| view_error / view_weight),
        });
    }
    if weights == 0.0 {
        return Err(HeadFitError::References(
            "no measurable supported constraints on this mesh".into(),
        ));
    }
    diagnostics.push(diagnostic("info","CPU_QUANTIZATION",format!("Mask cell = {CELL} head height. Surface anchors are low confidence; this CPU analysis is not GPU rendering.")));
    Ok(FitMetrics {
        schema_version: SCHEMA_VERSION.into(),
        measurement: MEASUREMENT.into(),
        objective: total / weights + geometry_penalty,
        width_over_height: (hi[0] - lo[0]) / h,
        depth_over_height: (hi[2] - lo[2]) / h,
        mesh_height: h,
        geometry_penalty,
        views,
        diagnostics,
    })
}

/// Standalone authoring overlay. Reference images remain host-managed, never embedded/fetched.
pub fn comparison_svg(before: &FitMetrics, after: &FitMetrics) -> String {
    let width = after.views.len() * 300;
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"400\" viewBox=\"0 0 {width} 400\"><rect width=\"100%\" height=\"100%\" fill=\"#111827\"/><text x=\"12\" y=\"24\" fill=\"white\">CPU analysis: cyan reference / gold candidate</text>"
    );
    for (i, v) in after.views.iter().enumerate() {
        let x = i as f64 * 300.0 + 150.0;
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"50\" fill=\"white\">{:?}: {} → {}</text>",
            x - 140.0,
            v.view,
            before
                .views
                .get(i)
                .and_then(|v| v.error)
                .map(|e| format!("{e:.4}"))
                .unwrap_or_else(|| "unmeasured".into()),
            v.error
                .map(|e| format!("{e:.4}"))
                .unwrap_or_else(|| "unmeasured".into())
        ));
        for p in &v.candidate_contour {
            svg.push_str(&format!(
                "<circle cx=\"{}\" cy=\"{}\" r=\"1\" fill=\"#fbbf24\"/>",
                x + p[0] * 220.0,
                220.0 - p[1] * 220.0
            ));
        }
        let points = v
            .reference_contour
            .iter()
            .map(|p| format!("{},{}", x + p[0] * 220.0, 220.0 - p[1] * 220.0))
            .collect::<Vec<_>>()
            .join(" ");
        svg.push_str(&format!(
            "<polygon points=\"{points}\" fill=\"none\" stroke=\"#22d3ee\"/>"
        ));
        for curve in &v.curves {
            let ref_points = curve
                .reference
                .iter()
                .map(|p| format!("{},{}", x + p[0] * 220.0, 220.0 - p[1] * 220.0))
                .collect::<Vec<_>>()
                .join(" ");
            svg.push_str(&format!(
                "<polyline points=\"{ref_points}\" fill=\"none\" stroke=\"#67e8f9\" stroke-width=\"1\"/>"
            ));
            if let Some(candidate) = &curve.candidate {
                let candidate_points = candidate
                    .iter()
                    .map(|p| format!("{},{}", x + p[0] * 220.0, 220.0 - p[1] * 220.0))
                    .collect::<Vec<_>>()
                    .join(" ");
                svg.push_str(&format!(
                    "<polyline points=\"{candidate_points}\" fill=\"none\" stroke=\"#fbbf24\" stroke-width=\"1\"/>"
                ));
            }
        }
        for l in &v.landmarks {
            if let Some(p) = l.candidate {
                svg.push_str(&format!(
                    "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#fb7185\"/>",
                    x + p[0] * 220.0,
                    220.0 - p[1] * 220.0,
                    x + l.reference[0] * 220.0,
                    220.0 - l.reference[1] * 220.0
                ));
            }
        }
    }
    svg.push_str("</svg>");
    svg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anatomical_sides_have_opposite_horizontal_axes() {
        let p = [0.2, 0.1, 0.5];
        assert_eq!(project(p, View::Left, 1.0, 0.0)[0], -0.5);
        assert_eq!(project(p, View::Right, 1.0, 0.0)[0], 0.5);
        assert_eq!(
            project(p, View::Front, 1.0, 0.0)[0],
            -project(p, View::Back, 1.0, 0.0)[0]
        );
    }

    #[test]
    fn triangle_coverage_preserves_concavity() {
        // Two rectangles form an L. The empty upper-right corner lies inside its vertex hull.
        let vertices = [
            [-0.8, -0.8],
            [0.8, -0.8],
            [0.8, -0.4],
            [-0.8, -0.4],
            [-0.4, -0.4],
            [-0.4, 0.8],
            [-0.8, 0.8],
        ];
        let mask = raster(&vertices, &[0, 1, 2, 0, 2, 3, 3, 4, 5, 3, 5, 6]);
        assert!(!mask[72 * N + 72]);
        assert!(mask[24 * N + 64]);
        assert!(mask[64 * N + 24]);
    }
}
