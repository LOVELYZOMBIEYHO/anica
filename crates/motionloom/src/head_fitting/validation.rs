// =========================================
// =========================================
// crates/motionloom/src/head_fitting/validation.rs

use super::*;
use std::collections::{BTreeMap, BTreeSet};

// The oriented centerline removes in-plane roll. Its origin fixes cross-view registration.
pub(super) fn normalize(r: &HeadReference, p: [f64; 2]) -> [f64; 2] {
    let a = &r.alignment;
    let d = [
        a.centerline[1][0] - a.centerline[0][0],
        a.centerline[1][1] - a.centerline[0][1],
    ];
    let length = d[0].hypot(d[1]);
    let down = [d[0] / length, d[1] / length];
    let height = (a.bottom[0] - a.top[0]) * down[0] + (a.bottom[1] - a.top[1]) * down[1];
    let center = [
        (a.top[0] + a.bottom[0]) * 0.5,
        (a.top[1] + a.bottom[1]) * 0.5,
    ];
    [
        ((p[0] - a.centerline[0][0]) * down[1] - (p[1] - a.centerline[0][1]) * down[0]) / height,
        -((p[0] - center[0]) * down[0] + (p[1] - center[1]) * down[1]) / height,
    ]
}
pub(super) fn height(r: &HeadReference) -> f64 {
    let a = &r.alignment;
    let d = [
        a.centerline[1][0] - a.centerline[0][0],
        a.centerline[1][1] - a.centerline[0][1],
    ];
    ((a.bottom[0] - a.top[0]) * d[0] + (a.bottom[1] - a.top[1]) * d[1]) / d[0].hypot(d[1])
}
pub(super) fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
pub(super) fn active(enabled: bool, confidence: f64, weight: f64) -> bool {
    enabled && confidence > 0.0 && weight > 0.0
}
fn point(r: &HeadReference, p: [f64; 2]) -> bool {
    p.iter().all(|v| v.is_finite())
        && p[0] >= 0.0
        && p[1] >= 0.0
        && p[0] < r.image_size[0] as f64
        && p[1] < r.image_size[1] as f64
}
fn weight(c: f64, w: f64) -> bool {
    c.is_finite() && (0.0..=1.0).contains(&c) && w.is_finite() && (0.0..=100.0).contains(&w)
}
fn cross(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}
fn polygon(points: &[[f64; 2]]) -> bool {
    let n = points.len();
    if n < 3 || n > 2048 {
        return false;
    }
    let area: f64 = (0..n)
        .map(|i| points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1])
        .sum();
    if area.abs() < 1.0 {
        return false;
    }
    for i in 0..n {
        if distance(points[i], points[(i + 1) % n]) < 1e-6 {
            return false;
        }
        for j in i + 1..n {
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }
            let (a, b, c, d) = (
                points[i],
                points[(i + 1) % n],
                points[j],
                points[(j + 1) % n],
            );
            if cross(a, b, c) * cross(a, b, d) <= 0.0
                && cross(c, d, a) * cross(c, d, b) <= 0.0
                && (0..2)
                    .all(|k| a[k].min(b[k]) <= c[k].max(d[k]) && c[k].min(d[k]) <= a[k].max(b[k]))
            {
                return false;
            }
        }
    }
    true
}

fn semantic_curve(id: &str) -> bool {
    matches!(
        id,
        "upper_eyelid_left"
            | "upper_eyelid_right"
            | "lower_eyelid_left"
            | "lower_eyelid_right"
            | "iris_left"
            | "iris_right"
            | "nose_profile"
            | "upper_lip"
            | "lower_lip"
            | "ear_outline_left"
            | "ear_outline_right"
    )
}

fn supported_landmark(id: &str) -> bool {
    matches!(
        id,
        "head_top"
            | "chin"
            | "nose_tip"
            | "nose_base"
            | "eye_inner_left"
            | "eye_inner_right"
            | "eye_outer_left"
            | "eye_outer_right"
            | "iris_center_left"
            | "iris_center_right"
            | "ear_center_left"
            | "ear_center_right"
            | "mouth_corner_left"
            | "mouth_corner_right"
            | "upper_lip_center"
            | "lower_lip_center"
    )
}

/// Validate annotations before any geometry search; inconsistent heights are blocking.
pub fn validate_head_reference_set(r: &HeadReferenceSet) -> ReferenceValidation {
    let mut diagnostics = vec![diagnostic(
        "info",
        "ORTHOGRAPHIC_ONLY",
        "Requires aligned orthographic views with pitch/yaw corrected externally. Hair and occluded anatomy are not head surface observations. imageId is opaque host data.",
    )];
    let mut error = |code: &str, msg: String| diagnostics.push(diagnostic("error", code, msg));
    if r.schema_version != SCHEMA_VERSION {
        error("SCHEMA_VERSION", "expected 1.0".into())
    }
    if r.target_asset_id.trim().is_empty() {
        error("TARGET", "empty targetAssetId".into())
    }
    if r.references.is_empty() || r.references.len() > 8 {
        error("REFERENCE_COUNT", "expected 1..8 references".into())
    }
    if r.options.max_iterations > 200 {
        error("ITERATION_LIMIT", "maxIterations must be <=200".into())
    }
    if r.options.symmetry != "x" && r.options.symmetry != "none" {
        error("SYMMETRY", "expected x or none".into())
    }
    for p in r
        .options
        .allowed_parameters
        .iter()
        .chain(&r.options.locked_parameters)
    {
        if !PARAMETERS.iter().any(|v| v.0 == p) {
            error("PARAMETER", format!("unsupported parameter {p}"))
        }
    }
    let mut ids = BTreeSet::new();
    let mut heights: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut observed = false;
    for view in &r.references {
        if view.id.is_empty() || !ids.insert(&view.id) {
            error("DUPLICATE_ID", format!("reference {}", view.id))
        }
        if view.image_id.is_empty()
            || view.image_size.contains(&0)
            || view.image_size.iter().any(|v| *v > 100000)
        {
            error("IMAGE_SIZE", format!("invalid image metadata: {}", view.id))
        }
        if view.projection != "orthographic" {
            error(
                "PROJECTION",
                format!("{} requires orthographic projection", view.id),
            )
        }
        let a = &view.alignment;
        let alignment_ok = [a.top, a.bottom, a.centerline[0], a.centerline[1]]
            .iter()
            .all(|p| point(view, *p))
            && height(view).is_finite()
            && height(view) > 1.0;
        if !alignment_ok {
            error(
                "ALIGNMENT",
                format!(
                    "{}: invalid points, reversed centerline or zero head height",
                    view.id
                ),
            )
        }
        if alignment_ok
            && [a.top, a.bottom]
                .iter()
                .any(|p| normalize(view, *p)[0].abs() > 0.2)
        {
            error(
                "CENTERLINE",
                format!("{}: centerline too far from top/chin", view.id),
            )
        }
        let mut marks = BTreeSet::new();
        if view.landmarks.len() > 128 {
            error("LANDMARK_COUNT", "at most 128 landmarks/view".into())
        }
        for l in &view.landmarks {
            if l.id.is_empty() || !marks.insert(&l.id) {
                error("DUPLICATE_ID", format!("{} landmark {}", view.id, l.id))
            }
            if !point(view, l.pixel) || !weight(l.confidence, l.weight) {
                error("LANDMARK", format!("{}: invalid {}", view.id, l.id))
            }
            if alignment_ok && active(l.enabled, l.confidence, l.weight) {
                heights
                    .entry(l.id.clone())
                    .or_default()
                    .push(normalize(view, l.pixel)[1]);
                observed |= supported_landmark(&l.id);
            }
        }
        let mut contours = BTreeSet::new();
        if view.contours.len() > 32 {
            error("CONTOUR_COUNT", "at most 32 contours/view".into())
        }
        for c in &view.contours {
            if c.id.is_empty() || !contours.insert(&c.id) {
                error("DUPLICATE_ID", format!("{} contour {}", view.id, c.id))
            }
            if !weight(c.confidence, c.weight) || c.points.iter().any(|p| !point(view, *p)) {
                error("CONTOUR", format!("{}: invalid points/weights", view.id))
            }
            if active(c.enabled, c.confidence, c.weight) {
                let valid = if c.id == "head_outline" {
                    c.closed && polygon(&c.points)
                } else if semantic_curve(&c.id) {
                    let min_points = if c.id.starts_with("iris_") { 3 } else { 2 };
                    c.points.len() >= min_points
                        && c.points.len() <= 2048
                        && (!c.id.starts_with("ear_") || c.closed)
                } else {
                    false
                };
                if !valid {
                    error(
                        "CONTOUR",
                        format!(
                            "{}: expected head_outline polygon or supported semantic face curve",
                            view.id
                        ),
                    )
                }
                if alignment_ok
                    && c.points
                        .iter()
                        .any(|p| normalize(view, *p).iter().any(|v| v.abs() >= 0.98))
                {
                    error(
                        "ANALYSIS_BOUNDS",
                        format!("{}: contour exceeds normalized analysis domain", view.id),
                    )
                }
                observed |= c.id == "head_outline" || semantic_curve(&c.id);
            }
        }
    }
    for (id, h) in heights {
        let min = h.iter().copied().fold(f64::INFINITY, f64::min);
        let max = h.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max - min > 0.06 {
            error(
                "CONFLICTING_HEIGHT",
                format!(
                    "{id}: cross-view height spread {:.4} exceeds 0.06 head height",
                    max - min
                ),
            )
        }
    }
    // Orthographic front/back silhouettes must agree in width after registration.
    for front in r.references.iter().filter(|v| v.view == View::Front) {
        for back in r.references.iter().filter(|v| v.view == View::Back) {
            let width = |v: &HeadReference| {
                v.contours
                    .iter()
                    .find(|c| c.id == "head_outline" && active(c.enabled, c.confidence, c.weight))
                    .map(|c| {
                        let xs: Vec<_> = c.points.iter().map(|p| normalize(v, *p)[0]).collect();
                        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                            - xs.iter().copied().fold(f64::INFINITY, f64::min)
                    })
            };
            if let (Some(a), Some(b)) = (width(front), width(back)) {
                if (a - b).abs() > 0.12 {
                    error(
                        "CONFLICTING_SILHOUETTE",
                        format!(
                            "front/back width differs by {:.4} head height; check hair, perspective and annotations",
                            (a - b).abs()
                        ),
                    );
                }
            }
        }
    }
    if !observed {
        error("NO_CONSTRAINTS", "no enabled supported observations".into())
    }
    if !r
        .references
        .iter()
        .any(|v| matches!(v.view, View::Left | View::Right))
    {
        diagnostics.push(diagnostic("warning","MISSING_SIDE","Depth is weakly constrained without an explicitly identified anatomical left/right view."))
    }
    for v in &r.references {
        for l in &v.landmarks {
            if !supported_landmark(&l.id) && l.enabled {
                diagnostics.push(diagnostic(
                    "warning",
                    "UNSUPPORTED_LANDMARK",
                    format!("{}: {} has no reliable surface correspondence", v.id, l.id),
                ))
            }
        }
    }
    ReferenceValidation {
        schema_version: SCHEMA_VERSION.into(),
        measurement: MEASUREMENT.into(),
        valid: !diagnostics.iter().any(|d| d.severity == "error"),
        diagnostics,
    }
}
