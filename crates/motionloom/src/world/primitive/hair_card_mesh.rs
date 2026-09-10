// =========================================
// =========================================
// crates/motionloom/src/world/primitive/hair_card_mesh.rs

use crate::dsl::{HairGuideNode, HairPointNode};

use super::{MeshBuilder, dot, normalize};

/// Compile representation-neutral guides into curved, closed card geometry.
pub(super) fn generate(
    mesh: &mut MeshBuilder,
    length_segments: u32,
    width_segments: u32,
    thickness: f32,
    cross_section: &str,
    tip_shape: &str,
    guides: &[HairGuideNode],
) {
    for guide in guides {
        generate_guide(
            mesh,
            guide,
            length_segments as usize,
            width_segments as usize,
            thickness,
            cross_section,
            tip_shape,
        );
    }
}

fn generate_guide(
    mesh: &mut MeshBuilder,
    guide: &HairGuideNode,
    length_segments: usize,
    width_segments: usize,
    thickness: f32,
    cross_section: &str,
    tip_shape: &str,
) {
    let frames = guide_frames(guide);
    let samples: Vec<_> = (0..=length_segments)
        .map(|step| sample_frame(&frames, step as f32 / length_segments as f32))
        .collect();

    let row_width = width_segments + 1;
    let mut front = Vec::with_capacity((length_segments + 1) * row_width);
    let mut back = Vec::with_capacity((length_segments + 1) * row_width);
    for (row, (point, side, normal)) in samples.iter().enumerate() {
        let u = row as f32 / length_segments as f32;
        let taper = tip_taper(u, tip_shape);
        let width = point.width * taper;
        for column in 0..=width_segments {
            let v = column as f32 / width_segments as f32;
            let across = v * 2.0 - 1.0;
            let arch = profile_height(cross_section, across) * point.camber * width;
            let center = offset(point.position, *side, across * width * 0.5, *normal, arch);
            front.push(mesh.vertex(
                offset(center, *side, 0.0, *normal, thickness * taper * 0.5),
                *normal,
                [v, u],
            ));
            back.push(mesh.vertex(
                offset(center, *side, 0.0, *normal, -thickness * taper * 0.5),
                normal.map(|value| -value),
                [1.0 - v, u],
            ));
        }
    }

    let surface_start = mesh.indices.len();
    for row in 0..length_segments {
        for column in 0..width_segments {
            let a = row * row_width + column;
            let b = (row + 1) * row_width + column;
            let c = b + 1;
            let d = a + 1;
            surface_triangle(mesh, [front[a], front[b], front[c]]);
            surface_triangle(mesh, [front[a], front[c], front[d]]);
            surface_triangle(mesh, [back[a], back[c], back[b]]);
            surface_triangle(mesh, [back[a], back[d], back[c]]);
        }
    }
    // Surface normals follow the arch and longitudinal curvature, not one
    // constant normal per row. End/side walls are split to keep their hard edge.
    let mut normals = vec![[0.0; 3]; mesh.positions.len()];
    for tri in mesh.indices[surface_start..].chunks_exact(3) {
        let n = super::triangle_cross(
            mesh.positions[tri[0] as usize],
            mesh.positions[tri[1] as usize],
            mesh.positions[tri[2] as usize],
        );
        for &i in tri {
            for k in 0..3 {
                normals[i as usize][k] += n[k];
            }
        }
    }
    for &i in front.iter().chain(&back) {
        if dot(normals[i as usize], normals[i as usize]) > 1.0e-20 {
            mesh.normals[i as usize] = Some(normalize(normals[i as usize]));
        }
    }

    // Close both long edges so the card keeps a readable cel silhouette in profile.
    for row in 0..length_segments {
        let next = (row + 1) * row_width;
        let current = row * row_width;
        wall_triangle(mesh, [front[current], back[next], front[next]]);
        wall_triangle(mesh, [front[current], back[current], back[next]]);
        let current_right = current + width_segments;
        let next_right = next + width_segments;
        wall_triangle(
            mesh,
            [front[current_right], front[next_right], back[next_right]],
        );
        wall_triangle(
            mesh,
            [front[current_right], back[next_right], back[current_right]],
        );
    }
    close_end(mesh, &front, &back, row_width, width_segments, 0, false);
    close_end(
        mesh,
        &front,
        &back,
        row_width,
        width_segments,
        length_segments,
        true,
    );
}

fn close_end(
    mesh: &mut MeshBuilder,
    front: &[u32],
    back: &[u32],
    row_width: usize,
    width_segments: usize,
    row: usize,
    end: bool,
) {
    let start = row * row_width;
    for column in 0..width_segments {
        let a = start + column;
        let b = a + 1;
        if end {
            wall_triangle(mesh, [front[a], back[b], back[a]]);
            wall_triangle(mesh, [front[a], front[b], back[b]]);
        } else {
            wall_triangle(mesh, [front[a], back[a], back[b]]);
            wall_triangle(mesh, [front[a], back[b], front[b]]);
        }
    }
}

// A fixed integration grid decouples orientation from render tessellation.
// Distances also provide even arc-length spacing for geometry and longitudinal UVs.
struct GuideFrame {
    point: HairPointNode,
    tangent: [f32; 3],
    side: [f32; 3],
    distance: f32,
}

fn guide_frames(guide: &HairGuideNode) -> Vec<GuideFrame> {
    let count = ((guide.points.len() - 1) * 64).clamp(128, 8192);
    let points: Vec<_> = (0..=count)
        .map(|i| sample_guide(&guide.points, i as f32 / count as f32))
        .collect();
    let mut frames: Vec<GuideFrame> = Vec::with_capacity(points.len());
    let mut distance = 0.0;
    for (i, point) in points.iter().enumerate() {
        let delta = if i == 0 {
            subtract(guide.points[1].position, guide.points[0].position)
        } else if i == count {
            subtract(
                guide.points[guide.points.len() - 1].position,
                guide.points[guide.points.len() - 2].position,
            )
        } else {
            subtract(points[i + 1].position, points[i - 1].position)
        };
        let tangent = if dot(delta, delta) > 1.0e-16 {
            normalize(delta)
        } else {
            frames.last().map(|f| f.tangent).unwrap_or([0.0, -1.0, 0.0])
        };
        let root_side = || {
            guide
                .normal
                .map(|n| cross(normalize(project_perpendicular(n, tangent)), tangent))
                .filter(|s| dot(*s, *s) > 1.0e-8)
                .map(normalize)
                .unwrap_or_else(|| initial_side(tangent))
        };
        let side = frames
            .last()
            .map(|f| project_perpendicular(f.side, tangent))
            .filter(|s| dot(*s, *s) > 1.0e-8)
            .map(normalize)
            .unwrap_or_else(root_side);
        if i > 0 {
            let d = subtract(point.position, points[i - 1].position);
            distance += dot(d, d).sqrt();
        }
        frames.push(GuideFrame {
            point: point.clone(),
            tangent,
            side,
            distance,
        });
    }
    frames
}

fn sample_frame(frames: &[GuideFrame], u: f32) -> (HairPointNode, [f32; 3], [f32; 3]) {
    let distance = u * frames.last().unwrap().distance;
    let next = frames
        .partition_point(|f| f.distance < distance)
        .clamp(1, frames.len() - 1);
    let a = &frames[next - 1];
    let b = &frames[next];
    let t = ((distance - a.distance) / (b.distance - a.distance).max(1.0e-12)).clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| a + (b - a) * t;
    let point = HairPointNode {
        position: std::array::from_fn(|i| lerp(a.point.position[i], b.point.position[i])),
        width: lerp(a.point.width, b.point.width),
        radius: lerp(a.point.radius, b.point.radius),
        camber: lerp(a.point.camber, b.point.camber),
        roll: lerp(a.point.roll, b.point.roll),
        stiffness: lerp(a.point.stiffness, b.point.stiffness),
    };
    let tangent = normalize(std::array::from_fn(|i| lerp(a.tangent[i], b.tangent[i])));
    let side = normalize(project_perpendicular(
        std::array::from_fn(|i| lerp(a.side[i], b.side[i])),
        tangent,
    ));
    // Apply absolute roll after transport; never feed rolled axes back into it.
    let side = rotate_about_axis(side, tangent, point.roll.to_radians());
    let normal = normalize(cross(tangent, side));
    (point, side, normal)
}

fn surface_triangle(mesh: &mut MeshBuilder, ids: [u32; 3]) {
    let p = ids.map(|i| mesh.positions[i as usize]);
    let n = super::triangle_cross(p[0], p[1], p[2]);
    if dot(n, n) > 1.0e-20 {
        mesh.triangle(ids[0], ids[1], ids[2]);
    }
}

fn wall_triangle(mesh: &mut MeshBuilder, ids: [u32; 3]) {
    let p = ids.map(|i| mesh.positions[i as usize]);
    let n = super::triangle_cross(p[0], p[1], p[2]);
    if dot(n, n) <= 1.0e-20 {
        return;
    }
    let n = normalize(n);
    let corners = ids.map(|i| {
        mesh.vertex(
            mesh.positions[i as usize],
            n,
            mesh.texcoords[i as usize].unwrap(),
        )
    });
    mesh.triangle(corners[0], corners[1], corners[2]);
}

fn sample_guide(points: &[HairPointNode], u: f32) -> HairPointNode {
    let scaled = u.clamp(0.0, 1.0) * (points.len() - 1) as f32;
    let segment = (scaled.floor() as usize).min(points.len() - 2);
    let t = scaled - segment as f32;
    let p0 = &points[segment.saturating_sub(1)];
    let p1 = &points[segment];
    let p2 = &points[segment + 1];
    let p3 = &points[(segment + 2).min(points.len() - 1)];
    HairPointNode {
        position: centripetal_position(p0.position, p1.position, p2.position, p3.position, t),
        width: catmull(p0.width, p1.width, p2.width, p3.width, t).max(0.001),
        radius: catmull(p0.radius, p1.radius, p2.radius, p3.radius, t).max(0.0),
        camber: catmull(p0.camber, p1.camber, p2.camber, p3.camber, t).clamp(0.0, 1.0),
        roll: catmull(p0.roll, p1.roll, p2.roll, p3.roll, t),
        stiffness: catmull(p0.stiffness, p1.stiffness, p2.stiffness, p3.stiffness, t)
            .clamp(0.0, 1.0),
    }
}

fn catmull(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * b)
        + (-a + c) * t
        + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
        + (-a + 3.0 * b - 3.0 * c + d) * t3)
}

// Centripetal knots avoid uniform Catmull-Rom loops with uneven guide spacing.
fn centripetal_position(
    mut a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    mut d: [f32; 3],
    t: f32,
) -> [f32; 3] {
    if a == b {
        a = std::array::from_fn(|i| 2.0 * b[i] - c[i]);
    }
    if c == d {
        d = std::array::from_fn(|i| 2.0 * c[i] - b[i]);
    }
    let knot = |a, b| {
        let v = subtract(a, b);
        dot(v, v).sqrt().sqrt().max(1.0e-6)
    };
    let t0 = 0.0;
    let t1 = knot(a, b);
    let t2 = t1 + knot(b, c);
    let t3 = t2 + knot(c, d);
    let u = t1 + (t2 - t1) * t;
    let mix = |a: [f32; 3], b: [f32; 3], lo: f32, hi: f32| {
        std::array::from_fn(|i| ((hi - u) * a[i] + (u - lo) * b[i]) / (hi - lo))
    };
    let a1 = mix(a, b, t0, t1);
    let a2 = mix(b, c, t1, t2);
    let a3 = mix(c, d, t2, t3);
    mix(mix(a1, a2, t0, t2), mix(a2, a3, t1, t3), t1, t2)
}

fn tip_taper(u: f32, tip_shape: &str) -> f32 {
    if u >= 1.0 && matches!(tip_shape, "point" | "round") {
        return 0.0;
    }
    match tip_shape {
        "point" => 1.0 - smoothstep(0.78, 1.0, u),
        "round" => (1.0 - ((u - 0.85) / 0.15).clamp(0.0, 1.0).powi(2))
            .max(0.0)
            .sqrt(),
        _ => 1.0,
    }
}

fn profile_height(profile: &str, across: f32) -> f32 {
    match profile {
        "arched" => 1.0 - across * across,
        "v_shape" => 1.0 - across.abs(),
        _ => 0.0,
    }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn initial_side(tangent: [f32; 3]) -> [f32; 3] {
    let reference = if tangent[1].abs() < 0.92 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    normalize(cross(reference, tangent))
}

fn project_perpendicular(vector: [f32; 3], tangent: [f32; 3]) -> [f32; 3] {
    let along = dot(vector, tangent);
    [
        vector[0] - tangent[0] * along,
        vector[1] - tangent[1] * along,
        vector[2] - tangent[2] * along,
    ]
}

fn offset(
    origin: [f32; 3],
    side: [f32; 3],
    side_scale: f32,
    normal: [f32; 3],
    normal_scale: f32,
) -> [f32; 3] {
    [
        origin[0] + side[0] * side_scale + normal[0] * normal_scale,
        origin[1] + side[1] * side_scale + normal[1] * normal_scale,
        origin[2] + side[2] * side_scale + normal[2] * normal_scale,
    ]
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
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

#[cfg(test)]
mod tests {
    use super::*;

    fn guide() -> HairGuideNode {
        HairGuideNode {
            id: "test".into(),
            group: "g".into(),
            role: "detail".into(),
            normal: Some([0.0, 0.0, 1.0]),
            points: [1.0, 0.95, 0.0]
                .into_iter()
                .map(|y| HairPointNode {
                    position: [0.0, y, 0.0],
                    width: 0.2,
                    radius: 0.003,
                    camber: 0.2,
                    roll: 10.0,
                    stiffness: 1.0,
                })
                .collect(),
        }
    }

    #[test]
    fn absolute_roll_and_arc_spacing_do_not_depend_on_tessellation() {
        let frames = guide_frames(&guide());
        let expected = rotate_about_axis([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], 10.0_f32.to_radians());
        for segments in [8, 24, 64] {
            for i in 0..=segments {
                let u = i as f32 / segments as f32;
                let (p, side, _) = sample_frame(&frames, u);
                assert!((p.position[1] - (1.0 - u)).abs() < 1e-5);
                for k in 0..3 {
                    assert!((side[k] - expected[k]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn curved_surface_normals_and_tips_follow_geometry() {
        for tip in ["point", "round", "blunt"] {
            let mut mesh = MeshBuilder::default();
            generate(&mut mesh, 24, 6, 0.02, "arched", tip, &[guide()]);
            // First row has separate front/back vertices across the width.
            assert!(dot(mesh.normals[0].unwrap(), mesh.normals[12].unwrap()) < 0.99);
            let row = 24 * 7 * 2;
            let separation = subtract(mesh.positions[row], mesh.positions[row + 1]);
            if tip == "blunt" {
                assert!(dot(separation, separation) > 1e-5);
            } else {
                assert!(dot(separation, separation) < 1e-12);
            }
            assert!(
                mesh.normals
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|n| n.is_finite())
            );
        }
        assert!(tip_taper(0.95, "round") > tip_taper(0.95, "point"));
    }
}
