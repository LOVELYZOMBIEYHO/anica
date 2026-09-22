// =========================================
// =========================================
// crates/motionloom/src/world/primitive/sweep_mesh.rs

use crate::dsl::{CurveAssetNode, CurveInterpolation, CurvePointNode, SweepProfilePointNode};

use super::{MeshBuilder, dot, normalize};

#[derive(Clone, Copy)]
struct CurveSample {
    position: [f32; 3],
    tangent: [f32; 3],
    side: [f32; 3],
    up: [f32; 3],
    scale: f32,
    distance: f32,
}

/// Compile one reusable spatial curve and inline profile to ordinary mesh data.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate(
    mesh: &mut MeshBuilder,
    curve: &CurveAssetNode,
    profile_closed: bool,
    smooth_profile: bool,
    cap_start: bool,
    cap_end: bool,
    frame: &str,
    uv_mode: &str,
    uv_scale: [f32; 2],
    dash: Option<[f32; 2]>,
    profile: &[SweepProfilePointNode],
) {
    let samples = sample_curve(curve, frame);
    if samples.len() < 2 {
        return;
    }
    let profile_distances = profile_distances(profile, profile_closed);
    let profile_normals = profile_normals(profile, profile_closed, smooth_profile);
    let profile_total = *profile_distances.last().unwrap_or(&1.0);
    let curve_total = samples
        .last()
        .map_or(1.0, |sample| sample.distance)
        .max(1.0e-6);
    let profile_segments = if profile_closed {
        profile.len()
    } else {
        profile.len().saturating_sub(1)
    };

    for pair in samples.windows(2) {
        let intervals = active_intervals(pair[0].distance, pair[1].distance, dash);
        for (start_distance, end_distance) in intervals {
            let span = (pair[1].distance - pair[0].distance).max(1.0e-6);
            let start =
                interpolate_sample(pair[0], pair[1], (start_distance - pair[0].distance) / span);
            let end =
                interpolate_sample(pair[0], pair[1], (end_distance - pair[0].distance) / span);
            for segment in 0..profile_segments {
                let next = (segment + 1) % profile.len();
                let a0 = swept_position(start, profile[segment].position);
                let a1 = swept_position(start, profile[next].position);
                let b0 = swept_position(end, profile[segment].position);
                let b1 = swept_position(end, profile[next].position);
                let normals = [
                    swept_normal(start, profile_normals[segment]),
                    swept_normal(end, profile_normals[segment]),
                    swept_normal(end, profile_normals[next]),
                    swept_normal(start, profile_normals[next]),
                ];
                let profile_u0 = profile_coordinate(
                    profile_distances[segment],
                    profile_total,
                    uv_mode,
                    uv_scale[0],
                );
                let profile_u1 = profile_coordinate(
                    if profile_closed && next == 0 {
                        profile_total
                    } else {
                        profile_distances[next]
                    },
                    profile_total,
                    uv_mode,
                    uv_scale[0],
                );
                let curve_v0 = curve_coordinate(start_distance, curve_total, uv_mode, uv_scale[1]);
                let curve_v1 = curve_coordinate(end_distance, curve_total, uv_mode, uv_scale[1]);
                let ids = [
                    mesh.vertex(a0, normals[0], [profile_u0, curve_v0]),
                    mesh.vertex(b0, normals[1], [profile_u0, curve_v1]),
                    mesh.vertex(b1, normals[2], [profile_u1, curve_v1]),
                    mesh.vertex(a1, normals[3], [profile_u1, curve_v0]),
                ];
                let face_normal = cross(subtract(b0, a0), subtract(a1, a0));
                let expected = add4(normals);
                if dot(face_normal, expected) >= 0.0 {
                    mesh.triangle(ids[0], ids[1], ids[2]);
                    mesh.triangle(ids[0], ids[2], ids[3]);
                } else {
                    mesh.triangle(ids[0], ids[2], ids[1]);
                    mesh.triangle(ids[0], ids[3], ids[2]);
                }
            }
        }
    }

    // Caps are meaningful for closed solid profiles. Open road-like profiles
    // intentionally remain surfaces at their ends.
    if profile_closed && dash.is_none() {
        if cap_start {
            cap(mesh, samples[0], profile, false);
        }
        if cap_end {
            cap(
                mesh,
                *samples.last().expect("sweep samples are non-empty"),
                profile,
                true,
            );
        }
    }
}

fn sample_curve(curve: &CurveAssetNode, frame: &str) -> Vec<CurveSample> {
    let mut raw = Vec::<CurvePointNode>::new();
    let segment_count = if curve.closed {
        curve.points.len()
    } else {
        curve.points.len() - 1
    };
    for segment in 0..segment_count {
        let p1 = &curve.points[segment];
        let p2 = &curve.points[(segment + 1) % curve.points.len()];
        let chord = length(subtract(p2.position, p1.position));
        let steps = (chord / curve.max_segment_length).ceil().max(1.0) as usize;
        for step in 0..steps {
            let t = step as f32 / steps as f32;
            raw.push(match curve.interpolation {
                CurveInterpolation::Linear => interpolate_curve_point(p1, p2, t),
                CurveInterpolation::CatmullRom => {
                    let p0 = curve_point(curve, segment as isize - 1);
                    let p3 = curve_point(curve, segment as isize + 2);
                    catmull_rom(p0, p1, p2, p3, t)
                }
            });
        }
    }
    if curve.closed {
        raw.push(raw[0].clone());
    } else {
        raw.push(curve.points[curve.points.len() - 1].clone());
    }
    build_frames(raw, frame)
}

fn curve_point(curve: &CurveAssetNode, index: isize) -> &CurvePointNode {
    if curve.closed {
        let len = curve.points.len() as isize;
        &curve.points[index.rem_euclid(len) as usize]
    } else {
        &curve.points[index.clamp(0, curve.points.len() as isize - 1) as usize]
    }
}

fn interpolate_curve_point(a: &CurvePointNode, b: &CurvePointNode, t: f32) -> CurvePointNode {
    CurvePointNode {
        position: lerp3(a.position, b.position, t),
        tilt: lerp(a.tilt, b.tilt, t),
        scale: lerp(a.scale, b.scale, t),
    }
}

fn catmull_rom(
    p0: &CurvePointNode,
    p1: &CurvePointNode,
    p2: &CurvePointNode,
    p3: &CurvePointNode,
    t: f32,
) -> CurvePointNode {
    let t2 = t * t;
    let t3 = t2 * t;
    let coordinate = |axis: usize| {
        0.5 * ((2.0 * p1.position[axis])
            + (-p0.position[axis] + p2.position[axis]) * t
            + (2.0 * p0.position[axis] - 5.0 * p1.position[axis] + 4.0 * p2.position[axis]
                - p3.position[axis])
                * t2
            + (-p0.position[axis] + 3.0 * p1.position[axis] - 3.0 * p2.position[axis]
                + p3.position[axis])
                * t3)
    };
    CurvePointNode {
        position: [coordinate(0), coordinate(1), coordinate(2)],
        tilt: lerp(p1.tilt, p2.tilt, t),
        scale: lerp(p1.scale, p2.scale, t),
    }
}

fn build_frames(points: Vec<CurvePointNode>, frame: &str) -> Vec<CurveSample> {
    let mut samples = Vec::with_capacity(points.len());
    let mut previous_side = None;
    let mut distance = 0.0;
    for index in 0..points.len() {
        if index > 0 {
            distance += length(subtract(points[index].position, points[index - 1].position));
        }
        let before = points[index.saturating_sub(1)].position;
        let after = points[(index + 1).min(points.len() - 1)].position;
        let tangent = normalize(subtract(after, before));
        let world_side = cross([0.0, 1.0, 0.0], tangent);
        let mut side = if frame == "worldup" && length(world_side) > 1.0e-5 {
            normalize(world_side)
        } else {
            previous_side
                .map(|previous| project_perpendicular(previous, tangent))
                .filter(|projected| length(*projected) > 1.0e-5)
                .map(normalize)
                .unwrap_or_else(|| initial_side(tangent))
        };
        if previous_side.is_some_and(|previous| dot(previous, side) < 0.0) {
            side = side.map(|value| -value);
        }
        let mut up = if frame == "worldup" {
            [0.0, 1.0, 0.0]
        } else {
            normalize(cross(tangent, side))
        };
        if points[index].tilt != 0.0 {
            side = rotate_about_axis(side, tangent, points[index].tilt.to_radians());
            up = normalize(cross(tangent, side));
        }
        previous_side = Some(side);
        samples.push(CurveSample {
            position: points[index].position,
            tangent,
            side,
            up,
            scale: points[index].scale,
            distance,
        });
    }
    samples
}

fn active_intervals(start: f32, end: f32, dash: Option<[f32; 2]>) -> Vec<(f32, f32)> {
    let Some([on, off]) = dash else {
        return vec![(start, end)];
    };
    let period = on + off;
    let mut cursor = start;
    let mut intervals = Vec::new();
    while cursor < end - 1.0e-6 {
        let cycle = (cursor / period).floor();
        let phase = cursor - cycle * period;
        let active = phase < on;
        let boundary = if active {
            cycle * period + on
        } else {
            (cycle + 1.0) * period
        };
        let next = boundary.min(end).max(cursor + 1.0e-6);
        if active {
            intervals.push((cursor, next));
        }
        cursor = next;
    }
    intervals
}

fn interpolate_sample(a: CurveSample, b: CurveSample, t: f32) -> CurveSample {
    let tangent = normalize(lerp3(a.tangent, b.tangent, t));
    let side = normalize(project_perpendicular(lerp3(a.side, b.side, t), tangent));
    CurveSample {
        position: lerp3(a.position, b.position, t),
        tangent,
        side,
        up: normalize(lerp3(a.up, b.up, t)),
        scale: lerp(a.scale, b.scale, t),
        distance: lerp(a.distance, b.distance, t),
    }
}

fn swept_position(sample: CurveSample, point: [f32; 2]) -> [f32; 3] {
    let side = point[0] * sample.scale;
    let up = point[1] * sample.scale;
    [
        sample.position[0] + sample.side[0] * side + sample.up[0] * up,
        sample.position[1] + sample.side[1] * side + sample.up[1] * up,
        sample.position[2] + sample.side[2] * side + sample.up[2] * up,
    ]
}

fn swept_normal(sample: CurveSample, normal: [f32; 2]) -> [f32; 3] {
    normalize([
        sample.side[0] * normal[0] + sample.up[0] * normal[1],
        sample.side[1] * normal[0] + sample.up[1] * normal[1],
        sample.side[2] * normal[0] + sample.up[2] * normal[1],
    ])
}

fn profile_normals(
    profile: &[SweepProfilePointNode],
    closed: bool,
    smooth_profile: bool,
) -> Vec<[f32; 2]> {
    if smooth_profile && !closed {
        let tangent = normalize2(subtract2(
            profile[profile.len() - 1].position,
            profile[0].position,
        ));
        return vec![[-tangent[1], tangent[0]]; profile.len()];
    }
    let signed_area = if closed {
        (0..profile.len())
            .map(|index| {
                let next = (index + 1) % profile.len();
                profile[index].position[0] * profile[next].position[1]
                    - profile[next].position[0] * profile[index].position[1]
            })
            .sum::<f32>()
    } else {
        0.0
    };
    (0..profile.len())
        .map(|index| {
            let before = if index == 0 {
                if closed { profile.len() - 1 } else { 0 }
            } else {
                index - 1
            };
            let after = if index + 1 == profile.len() {
                if closed { 0 } else { profile.len() - 1 }
            } else {
                index + 1
            };
            let tangent = normalize2(subtract2(profile[after].position, profile[before].position));
            if closed && signed_area >= 0.0 {
                normalize2([tangent[1], -tangent[0]])
            } else {
                normalize2([-tangent[1], tangent[0]])
            }
        })
        .collect()
}

fn profile_distances(profile: &[SweepProfilePointNode], closed: bool) -> Vec<f32> {
    let mut distances = vec![0.0];
    for pair in profile.windows(2) {
        distances.push(
            distances.last().copied().unwrap_or_default()
                + length2(subtract2(pair[1].position, pair[0].position)),
        );
    }
    if closed {
        let closing = length2(subtract2(
            profile[0].position,
            profile[profile.len() - 1].position,
        ));
        let total = distances.last().copied().unwrap_or_default() + closing;
        distances.push(total);
    }
    distances
}

fn profile_coordinate(distance: f32, total: f32, mode: &str, scale: f32) -> f32 {
    // Profiles use a stable zero-to-one coordinate in both modes. `distance`
    // refers to the longitudinal axis, matching common curve-to-mesh tools.
    let _ = mode;
    distance / total.max(1.0e-6) * scale
}

fn curve_coordinate(distance: f32, total: f32, mode: &str, scale: f32) -> f32 {
    if mode == "normalized" {
        distance / total.max(1.0e-6) * scale
    } else {
        distance * scale
    }
}

fn cap(mesh: &mut MeshBuilder, sample: CurveSample, profile: &[SweepProfilePointNode], end: bool) {
    let center = profile
        .iter()
        .fold([0.0_f32; 2], |sum, point| {
            [sum[0] + point.position[0], sum[1] + point.position[1]]
        })
        .map(|value| value / profile.len() as f32);
    let center_position = swept_position(sample, center);
    let normal = sample.tangent.map(|value| if end { value } else { -value });
    for index in 0..profile.len() {
        let next = (index + 1) % profile.len();
        let a = mesh.vertex(center_position, normal, [0.5, 0.5]);
        let b = mesh.vertex(
            swept_position(sample, profile[index].position),
            normal,
            [0.0, 0.0],
        );
        let c = mesh.vertex(
            swept_position(sample, profile[next].position),
            normal,
            [1.0, 0.0],
        );
        if end {
            mesh.triangle(a, b, c);
        } else {
            mesh.triangle(a, c, b);
        }
    }
}

fn initial_side(tangent: [f32; 3]) -> [f32; 3] {
    let reference = if tangent[1].abs() < 0.92 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    normalize(cross(reference, tangent))
}

fn project_perpendicular(vector: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let projection = dot(vector, normal);
    [
        vector[0] - normal[0] * projection,
        vector[1] - normal[1] * projection,
        vector[2] - normal[2] * projection,
    ]
}

fn rotate_about_axis(vector: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin, cos) = angle.sin_cos();
    let crossed = cross(axis, vector);
    let projected = dot(axis, vector) * (1.0 - cos);
    normalize([
        vector[0] * cos + crossed[0] * sin + axis[0] * projected,
        vector[1] * cos + crossed[1] * sin + axis[1] * projected,
        vector[2] * cos + crossed[2] * sin + axis[2] * projected,
    ])
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn subtract2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn length(vector: [f32; 3]) -> f32 {
    dot(vector, vector).sqrt()
}

fn length2(vector: [f32; 2]) -> f32 {
    vector[0].hypot(vector[1])
}

fn normalize2(vector: [f32; 2]) -> [f32; 2] {
    let magnitude = length2(vector);
    if magnitude <= 1.0e-8 {
        [0.0, 1.0]
    } else {
        [vector[0] / magnitude, vector[1] / magnitude]
    }
}

fn add4(values: [[f32; 3]; 4]) -> [f32; 3] {
    [
        values.iter().map(|value| value[0]).sum(),
        values.iter().map(|value| value[1]).sum(),
        values.iter().map(|value| value[2]).sum(),
    ]
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}
