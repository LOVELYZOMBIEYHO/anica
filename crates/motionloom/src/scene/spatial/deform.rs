// =========================================
// =========================================
// crates/motionloom/src/scene/spatial/deform.rs

use crate::scene::drawable::Point2;
use crate::scene::drawable::parse_path_subpaths;
use crate::scene::model::{
    GroupNode, LimbEnvelopeNode, LimbRegionNode, MeshTopologyNode, PinNode, PuppetNode, SceneNode,
};
use crate::scene::render::{MotionLoomSceneRenderError, eval_scene_number, triangulate_polygon};
use std::collections::{HashMap, HashSet};

use super::{Affine2, find_scene_node_anchor};

#[derive(Debug, Clone)]
pub(crate) struct EvaluatedDeformGrid {
    pub(crate) cols: usize,
    pub(crate) rows: usize,
    /// Bind-pose geometry used for hit testing and bone transforms.
    pub(crate) from: Vec<Point2>,
    /// Source texture coordinates. Usually equal to `from`, but joint skin
    /// completion can sample a safer interior material location.
    pub(crate) sample_from: Vec<Point2>,
    pub(crate) to: Vec<Point2>,
    pub(crate) triangles: Vec<[usize; 3]>,
    // Local bone meshes replace only their covered pixels so the remainder of
    // a full-character target stays in the bind pose.
    pub(crate) preserve_outside: bool,
}

pub(crate) fn eval_group_deform_grid(
    group: &GroupNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<Option<EvaluatedDeformGrid>, MotionLoomSceneRenderError> {
    let Some(size_raw) = group.deform_grid.as_deref() else {
        return Ok(None);
    };
    let size_raw = size_raw.trim();
    if size_raw.is_empty() || size_raw.eq_ignore_ascii_case("none") {
        return Ok(None);
    }

    let amount = eval_scene_number(&group.deform_amount, time_norm, time_sec)?.clamp(0.0, 1.0);
    if amount <= 0.0001 {
        return Ok(None);
    }

    let (cols, rows) = parse_deform_grid_size(size_raw)?;
    let expected = cols * rows;
    let grid_from_raw = group.grid_from.as_deref().ok_or_else(|| {
        invalid_deform_grid(size_raw, "deformGrid requires gridFrom=\"x,y ...\".")
    })?;
    let grid_to_raw = group
        .grid_to
        .as_deref()
        .ok_or_else(|| invalid_deform_grid(size_raw, "deformGrid requires gridTo=\"x,y ...\"."))?;
    let from = parse_deform_grid_points(grid_from_raw, cols, rows, "gridFrom")?;
    let target = parse_deform_grid_points(grid_to_raw, cols, rows, "gridTo")?;
    if from.len() != expected || target.len() != expected {
        return Err(invalid_deform_grid(
            size_raw,
            format!("expected {expected} control points."),
        ));
    }

    let to = from
        .iter()
        .zip(target.iter())
        .map(|(from, target)| from.lerp(*target, amount))
        .collect();

    Ok(Some(EvaluatedDeformGrid {
        cols,
        rows,
        sample_from: from.clone(),
        from,
        to,
        triangles: Vec::new(),
        preserve_outside: false,
    }))
}

pub(crate) fn eval_puppet_deform_grid(
    puppet: &PuppetNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<Option<EvaluatedDeformGrid>, MotionLoomSceneRenderError> {
    let mesh = puppet.mesh.trim();
    if mesh.eq_ignore_ascii_case("none") {
        return Ok(None);
    }

    let amount = eval_scene_number(&puppet.amount, time_norm, time_sec)?.clamp(0.0, 1.0);
    if amount <= 0.0001 {
        return Ok(None);
    }

    let (has_limb_regions, has_complete_limb_regions) = limb_region_completion(puppet);
    let has_region_fallback = puppet.children.iter().any(|child| {
        matches!(
            child,
            SceneNode::MeshTopology(_) | SceneNode::LimbEnvelope(_)
        )
    });
    // Exact Bone IK authoring is a three-region transaction. A single saved
    // hand/control region must never become an isolated deform island while
    // the shoulder and elbow regions are still being drawn.
    if has_limb_regions && !has_complete_limb_regions && !has_region_fallback {
        return Ok(None);
    }

    let width = eval_scene_number(&puppet.width, time_norm, time_sec)?.max(1.0);
    let height = eval_scene_number(&puppet.height, time_norm, time_sec)?.max(1.0);
    let topology = puppet_topology_mesh(puppet, time_norm, time_sec)?;
    let (cols, rows, from, triangles) = if topology.triangles.is_empty() {
        let (cols, rows) = puppet_grid_size(&puppet.density);
        (
            cols,
            rows,
            regular_grid_points(width, height, cols, rows),
            Vec::new(),
        )
    } else {
        (
            topology.vertices.len().max(1),
            1,
            topology.vertices.clone(),
            topology.triangles.clone(),
        )
    };
    let pins = puppet_pin_controls(puppet, &topology.vertex_map, amount, time_norm, time_sec)?;
    if pins.is_empty() {
        return Ok(None);
    }

    let solver = puppet.solver.trim();
    let to = if solver.eq_ignore_ascii_case("bones") {
        apply_bone_puppet_to_points(
            puppet,
            &from,
            &topology.vertex_bones,
            &pins,
            time_norm,
            time_sec,
        )?
    } else if solver.eq_ignore_ascii_case("chain") {
        apply_chain_puppet_to_points(puppet, &from, &pins, time_norm, time_sec)?
    } else if solver.eq_ignore_ascii_case("arap")
        || solver.eq_ignore_ascii_case("shape")
        || solver.eq_ignore_ascii_case("shape_preserving")
    {
        apply_arap_puppet_to_points(&from, &topology.triangles, &pins)
    } else if solver.eq_ignore_ascii_case("rigid")
        || solver.eq_ignore_ascii_case("mls")
        || solver.eq_ignore_ascii_case("mls_rigid")
    {
        apply_rigid_mls_puppet_to_points(&from, &pins)
    } else {
        from.iter()
            .map(|point| apply_puppet_pins_to_point(*point, &pins))
            .collect::<Vec<_>>()
    };
    if from
        .iter()
        .zip(to.iter())
        .all(|(a, b)| (a.x - b.x).abs() <= 0.001 && (a.y - b.y).abs() <= 0.001)
    {
        return Ok(None);
    }

    Ok(Some(EvaluatedDeformGrid {
        cols,
        rows,
        sample_from: if topology.sample_vertices.is_empty() {
            from.clone()
        } else {
            topology.sample_vertices
        },
        from,
        to,
        triangles,
        preserve_outside: (puppet.solver.trim().eq_ignore_ascii_case("bones")
            || puppet.solver.trim().eq_ignore_ascii_case("chain"))
            && eval_scene_bool(&puppet.preserve_outside, time_norm, time_sec)?,
    }))
}

pub(crate) fn transform_deform_grid(
    grid: &EvaluatedDeformGrid,
    transform: Affine2,
) -> EvaluatedDeformGrid {
    EvaluatedDeformGrid {
        cols: grid.cols,
        rows: grid.rows,
        from: grid
            .from
            .iter()
            .map(|point| transform_point2(transform, *point))
            .collect(),
        sample_from: grid
            .sample_from
            .iter()
            .map(|point| transform_point2(transform, *point))
            .collect(),
        to: grid
            .to
            .iter()
            .map(|point| transform_point2(transform, *point))
            .collect(),
        triangles: grid.triangles.clone(),
        preserve_outside: grid.preserve_outside,
    }
}

fn transform_point2(transform: Affine2, point: Point2) -> Point2 {
    let (x, y) = transform.transform_point(point.x, point.y);
    Point2::new(x, y)
}

pub(crate) fn transform_and_deform_point(
    transform: Affine2,
    point: Point2,
    deform: Option<&EvaluatedDeformGrid>,
) -> Point2 {
    let transformed = transform_point2(transform, point);
    deform
        .map(|grid| warp_point_with_deform_grid(transformed, grid))
        .unwrap_or(transformed)
}

pub(crate) fn transform_and_deform_subpaths(
    subpaths: &[Vec<Point2>],
    transform: Affine2,
    deform: &EvaluatedDeformGrid,
) -> Vec<Vec<Point2>> {
    subpaths
        .iter()
        .map(|subpath| {
            subpath
                .iter()
                .map(|point| transform_and_deform_point(transform, *point, Some(deform)))
                .collect()
        })
        .collect()
}

fn warp_point_with_deform_grid(point: Point2, grid: &EvaluatedDeformGrid) -> Point2 {
    if !grid.triangles.is_empty() {
        for triangle in &grid.triangles {
            if triangle
                .iter()
                .any(|index| *index >= grid.from.len() || *index >= grid.to.len())
            {
                continue;
            }
            if let Some(warped) = warp_point_with_deform_triangle(
                point,
                [
                    grid.from[triangle[0]],
                    grid.from[triangle[1]],
                    grid.from[triangle[2]],
                ],
                [
                    grid.to[triangle[0]],
                    grid.to[triangle[1]],
                    grid.to[triangle[2]],
                ],
            ) {
                return warped;
            }
        }
        return point;
    }
    for row in 0..grid.rows - 1 {
        for col in 0..grid.cols - 1 {
            let i00 = row * grid.cols + col;
            let i10 = i00 + 1;
            let i01 = (row + 1) * grid.cols + col;
            let i11 = i01 + 1;
            if let Some(warped) = warp_point_with_deform_triangle(
                point,
                [grid.from[i00], grid.from[i10], grid.from[i11]],
                [grid.to[i00], grid.to[i10], grid.to[i11]],
            ) {
                return warped;
            }
            if let Some(warped) = warp_point_with_deform_triangle(
                point,
                [grid.from[i00], grid.from[i11], grid.from[i01]],
                [grid.to[i00], grid.to[i11], grid.to[i01]],
            ) {
                return warped;
            }
        }
    }
    point
}

fn warp_point_with_deform_triangle(
    point: Point2,
    src: [Point2; 3],
    dst: [Point2; 3],
) -> Option<Point2> {
    let denom = triangle_barycentric_denominator(src);
    let (w0, w1, w2) = triangle_barycentric(point, src, denom)?;
    if w0 < -0.001 || w1 < -0.001 || w2 < -0.001 {
        return None;
    }
    Some(Point2::new(
        dst[0].x * w0 + dst[1].x * w1 + dst[2].x * w2,
        dst[0].y * w0 + dst[1].y * w1 + dst[2].y * w2,
    ))
}

fn parse_deform_grid_size(size: &str) -> Result<(usize, usize), MotionLoomSceneRenderError> {
    let normalized = size.trim().to_ascii_lowercase().replace(' ', "");
    let Some((cols_raw, rows_raw)) = normalized.split_once('x') else {
        return Err(invalid_deform_grid(
            size,
            "deformGrid must use the form \"colsxrows\", for example \"3x3\".",
        ));
    };
    let cols = cols_raw
        .parse::<usize>()
        .map_err(|_| invalid_deform_grid(size, format!("invalid column count: {cols_raw}")))?;
    let rows = rows_raw
        .parse::<usize>()
        .map_err(|_| invalid_deform_grid(size, format!("invalid row count: {rows_raw}")))?;
    if cols < 2 || rows < 2 || cols > 16 || rows > 16 {
        return Err(invalid_deform_grid(
            size,
            "deformGrid supports 2..16 columns and 2..16 rows.",
        ));
    }
    Ok((cols, rows))
}

fn puppet_grid_size(density: &str) -> (usize, usize) {
    match density.trim().to_ascii_lowercase().as_str() {
        "low" | "coarse" => (3, 3),
        "high" | "fine" => (7, 7),
        "ultra" | "dense" => (9, 9),
        raw => parse_deform_grid_size(raw).unwrap_or((5, 5)),
    }
}

fn regular_grid_points(width: f32, height: f32, cols: usize, rows: usize) -> Vec<Point2> {
    let mut points = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        let y = if rows <= 1 {
            0.0
        } else {
            height * row as f32 / (rows - 1) as f32
        };
        for col in 0..cols {
            let x = if cols <= 1 {
                0.0
            } else {
                width * col as f32 / (cols - 1) as f32
            };
            points.push(Point2::new(x, y));
        }
    }
    points
}

#[derive(Debug, Clone, Default)]
struct EvaluatedPuppetTopology {
    vertex_map: HashMap<String, Point2>,
    vertex_indices: HashMap<String, usize>,
    vertices: Vec<Point2>,
    sample_vertices: Vec<Point2>,
    vertex_bones: Vec<Option<String>>,
    triangles: Vec<[usize; 3]>,
}

#[derive(Debug, Clone)]
struct EvaluatedPuppetPin {
    id: String,
    role: Option<String>,
    parent: Option<String>,
    source: Point2,
    target: Point2,
    fixed: bool,
    radius: f32,
    strength: f32,
    rotation_radians: f32,
    scale: f32,
    falloff: String,
}

fn puppet_pin_controls(
    puppet: &PuppetNode,
    vertices: &HashMap<String, Point2>,
    amount: f32,
    time_norm: f32,
    time_sec: f32,
) -> Result<Vec<EvaluatedPuppetPin>, MotionLoomSceneRenderError> {
    let mut pins = Vec::new();
    for child in &puppet.children {
        let SceneNode::Pin(pin) = child else {
            continue;
        };
        let source = if let Some(bind_to) = pin.bind_to.as_deref() {
            find_scene_node_anchor(
                &puppet.children,
                bind_to,
                Affine2::identity(),
                time_norm,
                time_sec,
            )
            .map(|(x, y)| Point2::new(x, y))
            .ok_or_else(|| {
                invalid_deform_grid(
                    pin.id.as_deref().unwrap_or("pin"),
                    format!("PuppetPin bindTo target '{bind_to}' was not found."),
                )
            })?
        } else {
            eval_pin_source(pin, vertices, time_norm, time_sec)?
        };
        let fixed = eval_pin_fixed(pin, time_norm, time_sec)?;
        let target_x = if fixed {
            source.x
        } else {
            pin.target_x
                .as_deref()
                .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                .transpose()?
                .unwrap_or(source.x)
        };
        let target_y = if fixed {
            source.y
        } else {
            pin.target_y
                .as_deref()
                .map(|expr| eval_scene_number(expr, time_norm, time_sec))
                .transpose()?
                .unwrap_or(source.y)
        };
        let radius = eval_scene_number(&pin.radius, time_norm, time_sec)?.max(0.001);
        let strength = eval_scene_number(&pin.strength, time_norm, time_sec)?.clamp(0.0, 8.0);
        let rotation_radians =
            eval_scene_number(&pin.rotation, time_norm, time_sec)?.to_radians() * amount;
        let scale = 1.0
            + (eval_scene_number(&pin.scale, time_norm, time_sec)?.clamp(0.01, 100.0) - 1.0)
                * amount;
        let target = Point2::new(
            source.x + (target_x - source.x) * amount,
            source.y + (target_y - source.y) * amount,
        );
        pins.push(EvaluatedPuppetPin {
            id: pin.id.clone().unwrap_or_default(),
            role: pin.role.clone(),
            parent: pin.parent.clone(),
            source,
            target,
            fixed,
            radius,
            strength,
            rotation_radians,
            scale,
            falloff: pin.falloff.clone(),
        });
    }
    Ok(pins)
}

/// Deforms a mesh with a serial rigid chain while preserving each rest segment.
///
/// Explicit parent ids make the result independent of source ordering. The
/// final control is solved with FABRIK, then every mesh point follows its
/// nearest rest segment as a rigid transform.
fn apply_chain_puppet_to_points(
    puppet: &PuppetNode,
    points: &[Point2],
    pins: &[EvaluatedPuppetPin],
    time_norm: f32,
    time_sec: f32,
) -> Result<Vec<Point2>, MotionLoomSceneRenderError> {
    let ordered = ordered_chain_pins(puppet, pins)?;
    if ordered.len() < 2 {
        return Err(invalid_deform_grid(
            puppet.id.as_deref().unwrap_or("PuppetWarp"),
            "solver=\"chain\" requires at least two parent-linked PuppetPins.",
        ));
    }
    if !ordered[0].fixed {
        return Err(invalid_deform_grid(
            ordered[0].id.as_str(),
            "Chain root pin must use fixed=\"true\".",
        ));
    }

    let sources = ordered.iter().map(|pin| pin.source).collect::<Vec<_>>();
    let requested = ordered.iter().map(|pin| pin.target).collect::<Vec<_>>();
    let preserve_length = eval_scene_bool(&puppet.preserve_length, time_norm, time_sec)?;
    let stretch = eval_scene_number(&puppet.stretch, time_norm, time_sec)?.clamp(0.0, 1.0);
    let solved = solve_serial_chain(&sources, &requested, preserve_length, stretch);
    let transforms = sources
        .windows(2)
        .zip(solved.windows(2))
        .map(|(source, target)| {
            RigidTransform2::between(source[0], source[1], target[0], target[1])
        })
        .collect::<Vec<_>>();

    Ok(points
        .iter()
        .map(|point| {
            let segment = sources
                .windows(2)
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    distance_to_segment(*point, left[0], left[1])
                        .total_cmp(&distance_to_segment(*point, right[0], right[1]))
                })
                .map(|(index, _)| index)
                .unwrap_or(0);
            transforms[segment].apply(*point)
        })
        .collect())
}

fn ordered_chain_pins<'a>(
    puppet: &PuppetNode,
    pins: &'a [EvaluatedPuppetPin],
) -> Result<Vec<&'a EvaluatedPuppetPin>, MotionLoomSceneRenderError> {
    let root = pins.iter().find(|pin| {
        pin.role
            .as_deref()
            .is_some_and(|role| matches!(role.to_ascii_lowercase().as_str(), "anchor" | "root"))
            || pin.parent.is_none()
    });
    let Some(root) = root else {
        return Err(invalid_deform_grid(
            puppet.id.as_deref().unwrap_or("PuppetWarp"),
            "solver=\"chain\" requires a root pin with role=\"anchor\".",
        ));
    };
    if root.id.is_empty() {
        return Err(invalid_deform_grid(
            puppet.id.as_deref().unwrap_or("PuppetWarp"),
            "Chain PuppetPins require explicit ids.",
        ));
    }

    let mut ordered = vec![root];
    let mut visited = HashSet::from([root.id.as_str()]);
    loop {
        let parent_id = ordered
            .last()
            .map(|pin| pin.id.as_str())
            .unwrap_or_default();
        let children = pins
            .iter()
            .filter(|pin| pin.parent.as_deref() == Some(parent_id))
            .collect::<Vec<_>>();
        if children.is_empty() {
            break;
        }
        if children.len() != 1 {
            return Err(invalid_deform_grid(
                puppet.id.as_deref().unwrap_or("PuppetWarp"),
                format!("Chain pin '{parent_id}' must have exactly one child."),
            ));
        }
        let child = children[0];
        if child.id.is_empty() || !visited.insert(child.id.as_str()) {
            return Err(invalid_deform_grid(
                puppet.id.as_deref().unwrap_or("PuppetWarp"),
                "Chain parent links contain an empty id or cycle.",
            ));
        }
        ordered.push(child);
    }
    if ordered.len() != pins.len() {
        return Err(invalid_deform_grid(
            puppet.id.as_deref().unwrap_or("PuppetWarp"),
            "Every chain PuppetPin must belong to one serial parent chain.",
        ));
    }
    Ok(ordered)
}

fn solve_serial_chain(
    sources: &[Point2],
    requested: &[Point2],
    preserve_length: bool,
    stretch: f32,
) -> Vec<Point2> {
    if !preserve_length || sources.len() < 2 {
        return requested.to_vec();
    }
    let mut lengths = sources
        .windows(2)
        .map(|pair| point_distance(pair[0], pair[1]).max(0.0001))
        .collect::<Vec<_>>();
    let root = requested[0];
    let requested_tip = *requested.last().unwrap_or(&root);
    let total: f32 = lengths.iter().sum();
    let reach = point_distance(root, requested_tip);
    if reach > total {
        let scale = 1.0 + (reach / total - 1.0) * stretch;
        lengths.iter_mut().for_each(|length| *length *= scale);
    }

    let mut solved = requested.to_vec();
    solved[0] = root;
    for _ in 0..12 {
        let last = solved.len() - 1;
        solved[last] = requested_tip;
        for index in (0..last).rev() {
            solved[index] = point_at_distance(solved[index + 1], solved[index], lengths[index]);
        }
        solved[0] = root;
        for index in 0..last {
            solved[index + 1] = point_at_distance(solved[index], solved[index + 1], lengths[index]);
        }
    }
    solved
}

fn point_at_distance(origin: Point2, toward: Point2, distance: f32) -> Point2 {
    let dx = toward.x - origin.x;
    let dy = toward.y - origin.y;
    let length = dx.hypot(dy).max(0.0001);
    Point2::new(
        origin.x + dx / length * distance,
        origin.y + dy / length * distance,
    )
}

#[derive(Debug, Clone, Copy)]
struct RigidTransform2 {
    angle: f32,
    cos: f32,
    sin: f32,
    tx: f32,
    ty: f32,
}

impl RigidTransform2 {
    fn between(
        source_from: Point2,
        source_to: Point2,
        target_from: Point2,
        target_to: Point2,
    ) -> Self {
        let source_angle = (source_to.y - source_from.y).atan2(source_to.x - source_from.x);
        let target_angle = (target_to.y - target_from.y).atan2(target_to.x - target_from.x);
        let angle = shortest_angle_radians(target_angle - source_angle);
        let cos = angle.cos();
        let sin = angle.sin();
        let rotated_source_x = cos * source_from.x - sin * source_from.y;
        let rotated_source_y = sin * source_from.x + cos * source_from.y;
        Self {
            angle,
            cos,
            sin,
            tx: target_from.x - rotated_source_x,
            ty: target_from.y - rotated_source_y,
        }
    }

    fn apply(self, point: Point2) -> Point2 {
        Point2::new(
            self.cos * point.x - self.sin * point.y + self.tx,
            self.sin * point.x + self.cos * point.y + self.ty,
        )
    }
}

fn apply_bone_puppet_to_points(
    puppet: &PuppetNode,
    points: &[Point2],
    vertex_bones: &[Option<String>],
    pins: &[EvaluatedPuppetPin],
    time_norm: f32,
    time_sec: f32,
) -> Result<Vec<Point2>, MotionLoomSceneRenderError> {
    let anchor = find_bone_role_pin(pins, "anchor").ok_or_else(|| {
        invalid_deform_grid(
            puppet.id.as_deref().unwrap_or("PuppetWarp"),
            "solver=\"bones\" requires one PuppetPin role=\"anchor\".",
        )
    })?;
    let joint = find_bone_role_pin(pins, "joint").ok_or_else(|| {
        invalid_deform_grid(
            puppet.id.as_deref().unwrap_or("PuppetWarp"),
            "solver=\"bones\" requires one PuppetPin role=\"joint\".",
        )
    })?;
    let control = find_bone_role_pin(pins, "control").ok_or_else(|| {
        invalid_deform_grid(
            puppet.id.as_deref().unwrap_or("PuppetWarp"),
            "solver=\"bones\" requires one PuppetPin role=\"control\".",
        )
    })?;

    if !anchor.fixed {
        return Err(invalid_deform_grid(
            anchor.id.as_str(),
            "Bone Puppet anchor pin must use fixed=\"true\".",
        ));
    }

    let first_length = point_distance(anchor.source, joint.source).max(0.0001);
    let second_length = point_distance(joint.source, control.source).max(0.0001);
    let bend = eval_bone_bend(
        &puppet.bend,
        anchor.source,
        joint.source,
        control.source,
        time_norm,
        time_sec,
    )?;
    let stretch = eval_scene_number(&puppet.stretch, time_norm, time_sec)?.clamp(0.0, 1.0);
    let joint_softness = eval_scene_number(&puppet.joint_softness, time_norm, time_sec)?.max(0.001);
    let preserve_volume = eval_scene_bool(&puppet.preserve_volume, time_norm, time_sec)?;
    let (solved_joint, solved_control) = solve_two_bone_points(
        anchor.target,
        control.target,
        first_length,
        second_length,
        bend,
        stretch,
    );

    let upper = RigidTransform2::between(anchor.source, joint.source, anchor.target, solved_joint);
    let forearm =
        RigidTransform2::between(joint.source, control.source, solved_joint, solved_control);

    Ok(points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            if vertex_uses_fixed_bone(vertex_bones.get(index)) {
                return *point;
            }
            let upper_distance = distance_to_segment(*point, anchor.source, joint.source);
            let forearm_distance = distance_to_segment(*point, joint.source, control.source);
            let forearm_weight =
                explicit_bone_weight(vertex_bones.get(index)).unwrap_or_else(|| {
                    smoothstep(
                        0.0,
                        1.0,
                        0.5 + (upper_distance - forearm_distance) / (joint_softness * 2.0),
                    )
                });
            if preserve_volume {
                blend_rigid_transform_point(
                    upper,
                    forearm,
                    forearm_weight,
                    joint.source,
                    solved_joint,
                    *point,
                )
            } else {
                upper
                    .apply(*point)
                    .lerp(forearm.apply(*point), forearm_weight)
            }
        })
        .collect())
}

fn vertex_uses_fixed_bone(binding: Option<&Option<String>>) -> bool {
    binding
        .and_then(|value| value.as_deref())
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "fixed" | "static" | "bind"
            )
        })
}

fn explicit_bone_weight(binding: Option<&Option<String>>) -> Option<f32> {
    let binding = binding?.as_deref()?.trim().to_ascii_lowercase();
    match binding.as_str() {
        "upper" | "upper_arm" | "root" | "shoulder" => Some(0.0),
        "joint" | "elbow" | "blend" => Some(0.5),
        "forearm" | "lower" | "control" | "wrist" | "hand" => Some(1.0),
        _ => None,
    }
}

fn find_bone_role_pin<'a>(
    pins: &'a [EvaluatedPuppetPin],
    expected_role: &str,
) -> Option<&'a EvaluatedPuppetPin> {
    pins.iter().find(|pin| {
        pin.role
            .as_deref()
            .map(|role| role.trim().eq_ignore_ascii_case(expected_role))
            .unwrap_or_else(|| {
                pin.id
                    .trim()
                    .to_ascii_lowercase()
                    .ends_with(&format!("_{expected_role}_pin"))
            })
    })
}

fn solve_two_bone_points(
    root: Point2,
    requested_target: Point2,
    first_length: f32,
    second_length: f32,
    bend: f32,
    stretch: f32,
) -> (Point2, Point2) {
    let dx = requested_target.x - root.x;
    let dy = requested_target.y - root.y;
    let requested_distance = dx.hypot(dy).max(0.0001);
    let max_rigid_reach = (first_length + second_length).max(0.0002);
    let max_reach = max_rigid_reach + (requested_distance - max_rigid_reach).max(0.0) * stretch;
    let min_reach = (first_length - second_length).abs().max(0.0001);
    let solved_distance = requested_distance.clamp(min_reach, max_reach);
    let scale = if solved_distance > max_rigid_reach {
        solved_distance / max_rigid_reach
    } else {
        1.0
    };
    let first_solved_length = first_length * scale;
    let second_solved_length = second_length * scale;
    let direction_x = dx / requested_distance;
    let direction_y = dy / requested_distance;
    let solved_target = Point2::new(
        root.x + direction_x * solved_distance,
        root.y + direction_y * solved_distance,
    );
    let target_angle = direction_y.atan2(direction_x);
    let root_offset = (((first_solved_length * first_solved_length)
        + (solved_distance * solved_distance)
        - (second_solved_length * second_solved_length))
        / (2.0 * first_solved_length * solved_distance))
        .clamp(-1.0, 1.0)
        .acos();
    let root_angle = target_angle - bend.signum() * root_offset;
    let solved_joint = Point2::new(
        root.x + root_angle.cos() * first_solved_length,
        root.y + root_angle.sin() * first_solved_length,
    );
    (solved_joint, solved_target)
}

fn eval_bone_bend(
    value: &str,
    anchor: Point2,
    joint: Point2,
    control: Point2,
    time_norm: f32,
    time_sec: f32,
) -> Result<f32, MotionLoomSceneRenderError> {
    if value.trim().eq_ignore_ascii_case("auto") {
        let first_x = joint.x - anchor.x;
        let first_y = joint.y - anchor.y;
        let target_x = control.x - anchor.x;
        let target_y = control.y - anchor.y;
        let cross = first_x * target_y - first_y * target_x;
        return Ok(if cross < 0.0 { -1.0 } else { 1.0 });
    }
    Ok(if eval_scene_number(value, time_norm, time_sec)? < 0.0 {
        -1.0
    } else {
        1.0
    })
}

fn eval_scene_bool(
    value: &str,
    time_norm: f32,
    time_sec: f32,
) -> Result<bool, MotionLoomSceneRenderError> {
    let raw = value.trim();
    if raw.eq_ignore_ascii_case("true") || raw.eq_ignore_ascii_case("yes") || raw == "1" {
        return Ok(true);
    }
    if raw.eq_ignore_ascii_case("false") || raw.eq_ignore_ascii_case("no") || raw == "0" {
        return Ok(false);
    }
    Ok(eval_scene_number(raw, time_norm, time_sec)? >= 0.5)
}

fn blend_rigid_transform_point(
    first: RigidTransform2,
    second: RigidTransform2,
    weight: f32,
    source_joint: Point2,
    target_joint: Point2,
    point: Point2,
) -> Point2 {
    let angle_delta = shortest_angle_radians(second.angle - first.angle);
    let angle = first.angle + angle_delta * weight;
    let cos = angle.cos();
    let sin = angle.sin();
    let local_x = point.x - source_joint.x;
    let local_y = point.y - source_joint.y;
    Point2::new(
        target_joint.x + cos * local_x - sin * local_y,
        target_joint.y + sin * local_x + cos * local_y,
    )
}

fn shortest_angle_radians(mut value: f32) -> f32 {
    while value > std::f32::consts::PI {
        value -= std::f32::consts::TAU;
    }
    while value < -std::f32::consts::PI {
        value += std::f32::consts::TAU;
    }
    value
}

fn point_distance(a: Point2, b: Point2) -> f32 {
    (a.x - b.x).hypot(a.y - b.y)
}

fn distance_to_segment(point: Point2, start: Point2, end: Point2) -> f32 {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let length_squared = dx * dx + dy * dy;
    if length_squared <= 0.000001 {
        return point_distance(point, start);
    }
    let t =
        (((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared).clamp(0.0, 1.0);
    point_distance(point, Point2::new(start.x + dx * t, start.y + dy * t))
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(0.000001)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn eval_pin_source(
    pin: &PinNode,
    vertices: &HashMap<String, Point2>,
    time_norm: f32,
    time_sec: f32,
) -> Result<Point2, MotionLoomSceneRenderError> {
    if let Some(vertex) = pin.vertex.as_deref()
        && let Some(point) = vertices.get(vertex)
    {
        return Ok(*point);
    }
    let x = pin.x.as_deref().ok_or_else(|| {
        invalid_deform_grid(
            pin.id.as_deref().unwrap_or("pin"),
            "Pin requires x/y or vertex.",
        )
    })?;
    let y = pin.y.as_deref().ok_or_else(|| {
        invalid_deform_grid(
            pin.id.as_deref().unwrap_or("pin"),
            "Pin requires x/y or vertex.",
        )
    })?;
    Ok(Point2::new(
        eval_scene_number(x, time_norm, time_sec)?,
        eval_scene_number(y, time_norm, time_sec)?,
    ))
}

fn eval_pin_fixed(
    pin: &PinNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<bool, MotionLoomSceneRenderError> {
    let raw = pin.fixed.trim();
    if raw.eq_ignore_ascii_case("true") || raw.eq_ignore_ascii_case("yes") || raw == "1" {
        return Ok(true);
    }
    if raw.eq_ignore_ascii_case("false") || raw.eq_ignore_ascii_case("no") || raw == "0" {
        return Ok(false);
    }
    Ok(eval_scene_number(raw, time_norm, time_sec)? >= 0.5)
}

fn puppet_topology_mesh(
    puppet: &PuppetNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<EvaluatedPuppetTopology, MotionLoomSceneRenderError> {
    let mut topology_eval = EvaluatedPuppetTopology::default();
    for topology in puppet.children.iter().filter_map(|child| match child {
        SceneNode::MeshTopology(topology) => Some(topology),
        _ => None,
    }) {
        collect_topology_vertices(topology, &mut topology_eval, time_norm, time_sec)?;
        collect_topology_triangles(topology, &mut topology_eval);
    }
    let (_, has_complete_limb_regions) = limb_region_completion(puppet);
    if topology_eval.triangles.is_empty() && has_complete_limb_regions {
        topology_eval = topology_from_limb_regions(puppet, time_norm, time_sec)?;
    }
    if topology_eval.triangles.is_empty()
        && let Some(envelope) = puppet.children.iter().find_map(|child| match child {
            SceneNode::LimbEnvelope(envelope) => Some(envelope),
            _ => None,
        })
    {
        topology_eval = topology_from_limb_envelope(puppet, envelope, time_norm, time_sec)?;
    }
    Ok(topology_eval)
}

fn limb_region_completion(puppet: &PuppetNode) -> (bool, bool) {
    let mut has_any = false;
    let mut has_anchor = false;
    let mut has_joint = false;
    let mut has_control = false;
    for region in puppet.children.iter().filter_map(|child| match child {
        SceneNode::LimbRegion(region) => Some(region),
        _ => None,
    }) {
        has_any = true;
        let role = region.role.trim();
        if role.eq_ignore_ascii_case("anchor")
            || role.eq_ignore_ascii_case("upper")
            || role.eq_ignore_ascii_case("shoulder")
        {
            has_anchor = true;
        } else if role.eq_ignore_ascii_case("joint") || role.eq_ignore_ascii_case("elbow") {
            has_joint = true;
        } else if role.eq_ignore_ascii_case("control")
            || role.eq_ignore_ascii_case("forearm")
            || role.eq_ignore_ascii_case("hand")
            || role.eq_ignore_ascii_case("wrist")
        {
            has_control = true;
        }
    }
    (has_any, has_anchor && has_joint && has_control)
}

fn topology_from_limb_regions(
    puppet: &PuppetNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<EvaluatedPuppetTopology, MotionLoomSceneRenderError> {
    let mut topology = EvaluatedPuppetTopology::default();
    for (region_index, region) in puppet
        .children
        .iter()
        .filter_map(|child| match child {
            SceneNode::LimbRegion(region) => Some(region),
            _ => None,
        })
        .enumerate()
    {
        append_limb_region(&mut topology, region, region_index, time_norm, time_sec)?;
    }
    if topology.triangles.is_empty() {
        return Err(invalid_deform_grid(
            puppet.id.as_deref().unwrap_or("PuppetWarp"),
            "Bone Puppet LimbRegion paths could not be triangulated.",
        ));
    }
    Ok(topology)
}

fn append_limb_region(
    topology: &mut EvaluatedPuppetTopology,
    region: &LimbRegionNode,
    region_index: usize,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    let _alpha_clip = eval_scene_bool(&region.alpha_clip, time_norm, time_sec)?;
    let subpaths = parse_path_subpaths(&region.d)?;
    if subpaths.len() != 1 {
        return Err(invalid_deform_grid(
            region.id.as_deref().unwrap_or("LimbRegion"),
            "LimbRegion requires exactly one closed path.",
        ));
    }
    let triangles = triangulate_polygon(&subpaths[0]);
    if triangles.is_empty() {
        return Err(invalid_deform_grid(
            region.id.as_deref().unwrap_or("LimbRegion"),
            "LimbRegion path could not be triangulated.",
        ));
    }
    let bone = match region.role.trim().to_ascii_lowercase().as_str() {
        "anchor" | "upper" | "shoulder" => "upper",
        "joint" | "elbow" => "joint",
        "control" | "lower" | "forearm" | "wrist" | "hand" => "forearm",
        _ => {
            return Err(invalid_deform_grid(
                region.id.as_deref().unwrap_or("LimbRegion"),
                "LimbRegion role must be anchor, joint, or control.",
            ));
        }
    };
    for (triangle_index, triangle) in triangles.into_iter().enumerate() {
        let mut indices = [0usize; 3];
        for (corner, point) in triangle.into_iter().enumerate() {
            let index = topology.vertices.len();
            let id = format!(
                "{}_{}_{}_{}",
                region.id.as_deref().unwrap_or("limb_region"),
                region_index,
                triangle_index,
                corner
            );
            topology.vertex_map.insert(id.clone(), point);
            topology.vertex_indices.insert(id, index);
            topology.vertices.push(point);
            topology.sample_vertices.push(point);
            topology.vertex_bones.push(Some(bone.to_string()));
            indices[corner] = index;
        }
        topology.triangles.push(indices);
    }
    Ok(())
}

fn topology_from_limb_envelope(
    puppet: &PuppetNode,
    envelope: &LimbEnvelopeNode,
    time_norm: f32,
    time_sec: f32,
) -> Result<EvaluatedPuppetTopology, MotionLoomSceneRenderError> {
    // Evaluating the flag here keeps malformed animated booleans deterministic.
    // Texture alpha is already retained by both raster backends; the envelope
    // triangles provide the additional geometric clip.
    let _alpha_clip = eval_scene_bool(&envelope.alpha_clip, time_norm, time_sec)?;
    let subpaths = parse_path_subpaths(&envelope.d)?;
    if subpaths.len() != 1 {
        return Err(invalid_deform_grid(
            envelope.id.as_deref().unwrap_or("LimbEnvelope"),
            "LimbEnvelope requires exactly one closed path.",
        ));
    }
    let triangles = triangulate_polygon(&subpaths[0]);
    if triangles.is_empty() {
        return Err(invalid_deform_grid(
            envelope.id.as_deref().unwrap_or("LimbEnvelope"),
            "LimbEnvelope path could not be triangulated.",
        ));
    }

    let hand_axis = envelope
        .hand_from
        .as_deref()
        .map(|hand_from| limb_hand_axis(puppet, hand_from, time_norm, time_sec))
        .transpose()?;
    let mut topology = EvaluatedPuppetTopology::default();
    for (triangle_index, triangle) in triangles.into_iter().enumerate() {
        let mut indices = [0usize; 3];
        for (corner, point) in triangle.into_iter().enumerate() {
            let index = topology.vertices.len();
            let id = format!(
                "{}_{}_{}",
                envelope.id.as_deref().unwrap_or("limb_envelope"),
                triangle_index,
                corner
            );
            topology.vertex_map.insert(id.clone(), point);
            topology.vertex_indices.insert(id, index);
            topology.vertices.push(point);
            topology.sample_vertices.push(point);
            topology.vertex_bones.push(
                hand_axis
                    .filter(|(wrist, axis)| {
                        (point.x - wrist.x) * axis.x + (point.y - wrist.y) * axis.y >= 0.0
                    })
                    .map(|_| "hand".to_string()),
            );
            indices[corner] = index;
        }
        topology.triangles.push(indices);
    }
    Ok(topology)
}

fn limb_hand_axis(
    puppet: &PuppetNode,
    hand_from: &str,
    time_norm: f32,
    time_sec: f32,
) -> Result<(Point2, Point2), MotionLoomSceneRenderError> {
    let hand_pin = puppet
        .children
        .iter()
        .filter_map(|child| match child {
            SceneNode::Pin(pin) => Some(pin),
            _ => None,
        })
        .find(|pin| pin.id.as_deref() == Some(hand_from))
        .ok_or_else(|| {
            invalid_deform_grid(
                hand_from,
                format!("LimbEnvelope handFrom pin '{hand_from}' was not found."),
            )
        })?;
    let joint_pin = puppet
        .children
        .iter()
        .filter_map(|child| match child {
            SceneNode::Pin(pin) => Some(pin),
            _ => None,
        })
        .find(|pin| {
            pin.role
                .as_deref()
                .is_some_and(|role| role.eq_ignore_ascii_case("joint"))
        })
        .ok_or_else(|| {
            invalid_deform_grid(
                hand_from,
                "LimbEnvelope handFrom requires one PuppetPin role=\"joint\".",
            )
        })?;
    let vertices = HashMap::new();
    let wrist = eval_pin_source(hand_pin, &vertices, time_norm, time_sec)?;
    let joint = eval_pin_source(joint_pin, &vertices, time_norm, time_sec)?;
    let dx = wrist.x - joint.x;
    let dy = wrist.y - joint.y;
    let length = dx.hypot(dy).max(0.0001);
    Ok((wrist, Point2::new(dx / length, dy / length)))
}

fn collect_topology_vertices(
    topology: &MeshTopologyNode,
    out: &mut EvaluatedPuppetTopology,
    time_norm: f32,
    time_sec: f32,
) -> Result<(), MotionLoomSceneRenderError> {
    for child in &topology.children {
        if let SceneNode::Vertex(vertex) = child {
            let point = Point2::new(
                eval_scene_number(&vertex.x, time_norm, time_sec)?,
                eval_scene_number(&vertex.y, time_norm, time_sec)?,
            );
            let index = out.vertices.len();
            out.vertex_map.insert(vertex.id.clone(), point);
            out.vertex_indices.insert(vertex.id.clone(), index);
            out.vertices.push(point);
            out.sample_vertices.push(Point2::new(
                vertex
                    .sample_x
                    .as_deref()
                    .map(|value| eval_scene_number(value, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(point.x),
                vertex
                    .sample_y
                    .as_deref()
                    .map(|value| eval_scene_number(value, time_norm, time_sec))
                    .transpose()?
                    .unwrap_or(point.y),
            ));
            out.vertex_bones.push(vertex.bone.clone());
        }
    }
    Ok(())
}

fn collect_topology_triangles(topology: &MeshTopologyNode, out: &mut EvaluatedPuppetTopology) {
    for child in &topology.children {
        if let SceneNode::Triangle(triangle) = child {
            let Some(a) = out.vertex_indices.get(&triangle.a).copied() else {
                continue;
            };
            let Some(b) = out.vertex_indices.get(&triangle.b).copied() else {
                continue;
            };
            let Some(c) = out.vertex_indices.get(&triangle.c).copied() else {
                continue;
            };
            out.triangles.push([a, b, c]);
        }
    }
}

fn apply_puppet_pins_to_point(point: Point2, pins: &[EvaluatedPuppetPin]) -> Point2 {
    let mut dx = 0.0;
    let mut dy = 0.0;
    let mut weight_sum = 0.0;
    for pin in pins {
        let distance = ((point.x - pin.source.x).powi(2) + (point.y - pin.source.y).powi(2)).sqrt();
        let mut weight = puppet_pin_falloff(distance, pin.radius, &pin.falloff) * pin.strength;
        if !weight.is_finite() {
            weight = 0.0;
        }
        let local_x = point.x - pin.source.x;
        let local_y = point.y - pin.source.y;
        let cos = pin.rotation_radians.cos();
        let sin = pin.rotation_radians.sin();
        let transformed_x = pin.target.x + (cos * local_x - sin * local_y) * pin.scale;
        let transformed_y = pin.target.y + (sin * local_x + cos * local_y) * pin.scale;
        dx += (transformed_x - point.x) * weight;
        dy += (transformed_y - point.y) * weight;
        weight_sum += weight;
    }
    let divisor = weight_sum.max(1.0);
    Point2::new(point.x + dx / divisor, point.y + dy / divisor)
}

/// Rigid moving-least-squares deformation for an arbitrary number of handles.
///
/// Unlike the legacy radius falloff solver, this computes one best-fit local
/// rotation for every mesh vertex. The result interpolates pin positions while
/// resisting the stretching and needle-like collapse that sparse character
/// controls otherwise produce.
fn apply_rigid_mls_puppet_to_points(points: &[Point2], pins: &[EvaluatedPuppetPin]) -> Vec<Point2> {
    points
        .iter()
        .map(|point| apply_rigid_mls_puppet_to_point(*point, pins))
        .collect()
}

fn apply_rigid_mls_puppet_to_point(point: Point2, pins: &[EvaluatedPuppetPin]) -> Point2 {
    const HANDLE_EPSILON_SQUARED: f32 = 0.0001;
    const WEIGHT_EPSILON: f32 = 0.01;

    if let Some(pin) = pins.iter().find(|pin| {
        pin.strength > 0.0
            && (point.x - pin.source.x).powi(2) + (point.y - pin.source.y).powi(2)
                <= HANDLE_EPSILON_SQUARED
    }) {
        return pin.target;
    }

    let mut weight_sum = 0.0;
    let mut source_centroid = Point2::new(0.0, 0.0);
    let mut target_centroid = Point2::new(0.0, 0.0);
    let mut weighted_pins = Vec::with_capacity(pins.len());

    for pin in pins {
        if pin.strength <= 0.0 {
            continue;
        }
        let distance_squared = (point.x - pin.source.x).powi(2) + (point.y - pin.source.y).powi(2);
        let weight = pin.strength / (distance_squared + WEIGHT_EPSILON);
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        weight_sum += weight;
        source_centroid.x += pin.source.x * weight;
        source_centroid.y += pin.source.y * weight;
        target_centroid.x += pin.target.x * weight;
        target_centroid.y += pin.target.y * weight;
        weighted_pins.push((pin, weight));
    }

    if weight_sum <= f32::EPSILON {
        return point;
    }
    source_centroid.x /= weight_sum;
    source_centroid.y /= weight_sum;
    target_centroid.x /= weight_sum;
    target_centroid.y /= weight_sum;

    let mut dot = 0.0;
    let mut cross = 0.0;
    for (pin, weight) in weighted_pins {
        let source_x = pin.source.x - source_centroid.x;
        let source_y = pin.source.y - source_centroid.y;
        let target_x = pin.target.x - target_centroid.x;
        let target_y = pin.target.y - target_centroid.y;
        dot += weight * (source_x * target_x + source_y * target_y);
        cross += weight * (source_x * target_y - source_y * target_x);
    }

    let norm = dot.hypot(cross);
    if norm <= f32::EPSILON {
        return Point2::new(
            point.x + target_centroid.x - source_centroid.x,
            point.y + target_centroid.y - source_centroid.y,
        );
    }

    let cos = dot / norm;
    let sin = cross / norm;
    let local_x = point.x - source_centroid.x;
    let local_y = point.y - source_centroid.y;
    Point2::new(
        target_centroid.x + cos * local_x - sin * local_y,
        target_centroid.y + sin * local_x + cos * local_y,
    )
}

/// Shape-preserving deformation for an explicit triangle mesh.
///
/// Position pins become hard vertex constraints. Repeated edge-length
/// projection then keeps every triangle close to its bind-pose shape, which is
/// the essential property missing from a pure radial-falloff warp.
fn apply_arap_puppet_to_points(
    points: &[Point2],
    triangles: &[[usize; 3]],
    pins: &[EvaluatedPuppetPin],
) -> Vec<Point2> {
    if triangles.is_empty() {
        return apply_rigid_mls_puppet_to_points(points, pins);
    }

    let mut result = apply_rigid_mls_puppet_to_points(points, pins);
    let mut constrained = HashMap::<usize, Point2>::new();
    for pin in pins.iter().filter(|pin| pin.strength > 0.0) {
        let Some((index, source_vertex)) = points.iter().enumerate().min_by(|(_, a), (_, b)| {
            point_distance(**a, pin.source).total_cmp(&point_distance(**b, pin.source))
        }) else {
            continue;
        };
        constrained.insert(
            index,
            Point2::new(
                source_vertex.x + pin.target.x - pin.source.x,
                source_vertex.y + pin.target.y - pin.source.y,
            ),
        );
    }

    let mut unique_edges = HashSet::<(usize, usize)>::new();
    for triangle in triangles {
        for (a, b) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            if a < points.len() && b < points.len() && a != b {
                unique_edges.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
    }
    let edges = unique_edges
        .into_iter()
        .map(|(a, b)| (a, b, point_distance(points[a], points[b])))
        .collect::<Vec<_>>();

    for _ in 0..80 {
        for &(a, b, rest_length) in &edges {
            let dx = result[b].x - result[a].x;
            let dy = result[b].y - result[a].y;
            let current_length = dx.hypot(dy);
            if current_length <= 0.0001 || rest_length <= 0.0001 {
                continue;
            }
            let scale = (current_length - rest_length) / current_length;
            let correction = Point2::new(dx * scale, dy * scale);
            match (constrained.contains_key(&a), constrained.contains_key(&b)) {
                (true, true) => {}
                (true, false) => {
                    result[b].x -= correction.x;
                    result[b].y -= correction.y;
                }
                (false, true) => {
                    result[a].x += correction.x;
                    result[a].y += correction.y;
                }
                (false, false) => {
                    result[a].x += correction.x * 0.5;
                    result[a].y += correction.y * 0.5;
                    result[b].x -= correction.x * 0.5;
                    result[b].y -= correction.y * 0.5;
                }
            }
        }
        for (&index, &target) in &constrained {
            result[index] = target;
        }
    }
    result
}

fn puppet_pin_falloff(distance: f32, radius: f32, falloff: &str) -> f32 {
    if distance >= radius {
        return 0.0;
    }
    let t = (1.0 - distance / radius).clamp(0.0, 1.0);
    match falloff.trim().to_ascii_lowercase().as_str() {
        "linear" => t,
        "gaussian" | "gauss" => (-(distance / radius).powi(2) * 4.0).exp(),
        "rigid" | "plateau" => {
            if distance <= radius * 0.6 {
                1.0
            } else {
                let feather = ((radius - distance) / (radius * 0.4)).clamp(0.0, 1.0);
                feather * feather * (3.0 - 2.0 * feather)
            }
        }
        "none" | "constant" => 1.0,
        _ => t * t * (3.0 - 2.0 * t),
    }
}

fn parse_deform_grid_points(
    value: &str,
    cols: usize,
    rows: usize,
    label: &str,
) -> Result<Vec<Point2>, MotionLoomSceneRenderError> {
    let mut points = Vec::new();
    let row_chunks: Vec<&str> = if value.contains(';') {
        value.split(';').collect()
    } else {
        vec![value]
    };
    if row_chunks.len() != 1 && row_chunks.len() != rows {
        return Err(invalid_deform_grid(
            value,
            format!("{label} expected {rows} rows separated by ';'."),
        ));
    }

    for (row_index, row) in row_chunks.iter().enumerate() {
        let row_points = row
            .split_whitespace()
            .map(|raw| parse_deform_grid_point(raw, value))
            .collect::<Result<Vec<_>, _>>()?;
        if row_chunks.len() != 1 && row_points.len() != cols {
            return Err(invalid_deform_grid(
                value,
                format!(
                    "{label} row {} expected {cols} points, got {}.",
                    row_index + 1,
                    row_points.len()
                ),
            ));
        }
        points.extend(row_points);
    }

    let expected = cols * rows;
    if points.len() != expected {
        return Err(invalid_deform_grid(
            value,
            format!("{label} expected {expected} points, got {}.", points.len()),
        ));
    }
    Ok(points)
}

fn parse_deform_grid_point(raw: &str, source: &str) -> Result<Point2, MotionLoomSceneRenderError> {
    let Some((x_raw, y_raw)) = raw.split_once(',') else {
        return Err(invalid_deform_grid(
            source,
            format!("control point must be \"x,y\": {raw}"),
        ));
    };
    let x = x_raw
        .trim()
        .parse::<f32>()
        .map_err(|_| invalid_deform_grid(source, format!("invalid x value: {x_raw}")))?;
    let y = y_raw
        .trim()
        .parse::<f32>()
        .map_err(|_| invalid_deform_grid(source, format!("invalid y value: {y_raw}")))?;
    if !x.is_finite() || !y.is_finite() {
        return Err(invalid_deform_grid(
            source,
            format!("control point must be finite: {raw}"),
        ));
    }
    Ok(Point2::new(x, y))
}

pub(crate) fn triangle_barycentric_denominator(tri: [Point2; 3]) -> f32 {
    (tri[1].y - tri[2].y) * (tri[0].x - tri[2].x) + (tri[2].x - tri[1].x) * (tri[0].y - tri[2].y)
}

pub(crate) fn triangle_barycentric(
    point: Point2,
    tri: [Point2; 3],
    denom: f32,
) -> Option<(f32, f32, f32)> {
    if denom.abs() <= 0.00001 {
        return None;
    }
    let w0 = ((tri[1].y - tri[2].y) * (point.x - tri[2].x)
        + (tri[2].x - tri[1].x) * (point.y - tri[2].y))
        / denom;
    let w1 = ((tri[2].y - tri[0].y) * (point.x - tri[2].x)
        + (tri[0].x - tri[2].x) * (point.y - tri[2].y))
        / denom;
    let w2 = 1.0 - w0 - w1;
    Some((w0, w1, w2))
}

fn invalid_deform_grid(value: &str, message: impl Into<String>) -> MotionLoomSceneRenderError {
    MotionLoomSceneRenderError::InvalidDeformGrid {
        value: value.to_string(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(actual: f32, expected: f32, epsilon: f32) {
        assert!(
            (actual - expected).abs() <= epsilon,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn two_bone_solver_preserves_rigid_segment_lengths() {
        let root = Point2::new(0.0, 0.0);
        let (joint, control) =
            solve_two_bone_points(root, Point2::new(80.0, 45.0), 70.0, 55.0, -1.0, 0.0);
        assert_near(point_distance(root, joint), 70.0, 0.001);
        assert_near(point_distance(joint, control), 55.0, 0.001);
    }

    #[test]
    fn two_bone_solver_clamps_unreachable_targets_without_stretch() {
        let root = Point2::new(10.0, 20.0);
        let (joint, control) =
            solve_two_bone_points(root, Point2::new(1000.0, 20.0), 60.0, 40.0, 1.0, 0.0);
        assert_near(point_distance(root, joint), 60.0, 0.001);
        assert_near(point_distance(joint, control), 40.0, 0.001);
        assert_near(point_distance(root, control), 100.0, 0.001);
    }

    #[test]
    fn serial_chain_solver_preserves_every_rest_length() {
        let sources = [
            Point2::new(0.0, 0.0),
            Point2::new(40.0, 0.0),
            Point2::new(80.0, 0.0),
            Point2::new(120.0, 0.0),
        ];
        let requested = [
            sources[0],
            Point2::new(30.0, 10.0),
            Point2::new(55.0, 40.0),
            Point2::new(70.0, 85.0),
        ];
        let solved = solve_serial_chain(&sources, &requested, true, 0.0);
        for pair in solved.windows(2) {
            assert_near(point_distance(pair[0], pair[1]), 40.0, 0.01);
        }
        assert_near(solved[0].x, 0.0, 0.001);
        assert_near(solved[0].y, 0.0, 0.001);
    }

    #[test]
    fn chain_parent_links_define_order_independent_of_node_order() {
        let puppet = PuppetNode {
            id: Some("tail".to_string()),
            target: Some("tail_art".to_string()),
            capture: None,
            solver: "chain".to_string(),
            mesh: "auto".to_string(),
            density: "medium".to_string(),
            bend: "auto".to_string(),
            stretch: "0".to_string(),
            joint_softness: "48".to_string(),
            preserve_volume: "true".to_string(),
            preserve_outside: "false".to_string(),
            preserve_length: "true".to_string(),
            stiffness: "0.72".to_string(),
            damping: "0.84".to_string(),
            drag: "0.18".to_string(),
            overlap: "0.12".to_string(),
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
            width: "1920".to_string(),
            height: "1080".to_string(),
            amount: "1".to_string(),
            opacity: "1".to_string(),
            children: Vec::new(),
        };
        let pins = [
            EvaluatedPuppetPin {
                id: "tip".to_string(),
                role: Some("control".to_string()),
                parent: Some("middle".to_string()),
                source: Point2::new(80.0, 0.0),
                target: Point2::new(80.0, 20.0),
                fixed: false,
                radius: 1.0,
                strength: 1.0,
                rotation_radians: 0.0,
                scale: 1.0,
                falloff: "constant".to_string(),
            },
            EvaluatedPuppetPin {
                id: "root".to_string(),
                role: Some("anchor".to_string()),
                parent: None,
                source: Point2::new(0.0, 0.0),
                target: Point2::new(0.0, 0.0),
                fixed: true,
                radius: 1.0,
                strength: 1.0,
                rotation_radians: 0.0,
                scale: 1.0,
                falloff: "constant".to_string(),
            },
            EvaluatedPuppetPin {
                id: "middle".to_string(),
                role: Some("chain".to_string()),
                parent: Some("root".to_string()),
                source: Point2::new(40.0, 0.0),
                target: Point2::new(40.0, 10.0),
                fixed: false,
                radius: 1.0,
                strength: 1.0,
                rotation_radians: 0.0,
                scale: 1.0,
                falloff: "constant".to_string(),
            },
        ];
        let ordered = ordered_chain_pins(&puppet, &pins).expect("serial chain");
        assert_eq!(
            ordered
                .iter()
                .map(|pin| pin.id.as_str())
                .collect::<Vec<_>>(),
            vec!["root", "middle", "tip"]
        );
    }

    #[test]
    fn rigid_forearm_transform_preserves_hand_shape() {
        let source_elbow = Point2::new(0.0, 0.0);
        let source_wrist = Point2::new(100.0, 0.0);
        let target_elbow = Point2::new(20.0, 30.0);
        let target_wrist = Point2::new(20.0, 130.0);
        let transform =
            RigidTransform2::between(source_elbow, source_wrist, target_elbow, target_wrist);
        let hand_a = Point2::new(120.0, -20.0);
        let hand_b = Point2::new(150.0, 25.0);
        assert_near(
            point_distance(transform.apply(hand_a), transform.apply(hand_b)),
            point_distance(hand_a, hand_b),
            0.001,
        );
    }

    #[test]
    fn fixed_bone_binding_marks_seam_vertices_as_static() {
        assert!(vertex_uses_fixed_bone(Some(&Some("fixed".to_string()))));
        assert!(vertex_uses_fixed_bone(Some(&Some("STATIC".to_string()))));
        assert!(!vertex_uses_fixed_bone(Some(&Some("upper".to_string()))));
        assert!(!vertex_uses_fixed_bone(None));
    }

    #[test]
    fn bend_pin_rotates_points_around_its_source() {
        let pin = EvaluatedPuppetPin {
            id: "bend".to_string(),
            role: Some("bend".to_string()),
            parent: None,
            source: Point2::new(10.0, 10.0),
            target: Point2::new(10.0, 10.0),
            fixed: false,
            radius: 100.0,
            strength: 1.0,
            rotation_radians: 90.0_f32.to_radians(),
            scale: 1.0,
            falloff: "constant".to_string(),
        };
        let result = apply_puppet_pins_to_point(Point2::new(30.0, 10.0), &[pin]);
        assert_near(result.x, 10.0, 0.001);
        assert_near(result.y, 30.0, 0.001);
    }

    #[test]
    fn rigid_mls_preserves_shape_between_rotated_handles() {
        let pins = [
            EvaluatedPuppetPin {
                id: "a".to_string(),
                role: Some("position".to_string()),
                parent: None,
                source: Point2::new(0.0, 0.0),
                target: Point2::new(10.0, 20.0),
                fixed: false,
                radius: 1.0,
                strength: 1.0,
                rotation_radians: 0.0,
                scale: 1.0,
                falloff: "smooth".to_string(),
            },
            EvaluatedPuppetPin {
                id: "b".to_string(),
                role: Some("position".to_string()),
                parent: None,
                source: Point2::new(100.0, 0.0),
                target: Point2::new(10.0, 120.0),
                fixed: false,
                radius: 1.0,
                strength: 1.0,
                rotation_radians: 0.0,
                scale: 1.0,
                falloff: "smooth".to_string(),
            },
        ];
        let result = apply_rigid_mls_puppet_to_point(Point2::new(50.0, 20.0), &pins);
        assert_near(result.x, -10.0, 0.001);
        assert_near(result.y, 70.0, 0.001);
        assert_near(
            point_distance(
                result,
                apply_rigid_mls_puppet_to_point(Point2::new(50.0, 0.0), &pins),
            ),
            20.0,
            0.001,
        );
    }

    #[test]
    fn rigid_mls_interpolates_control_positions_exactly() {
        let pin = EvaluatedPuppetPin {
            id: "control".to_string(),
            role: Some("position".to_string()),
            parent: None,
            source: Point2::new(25.0, 35.0),
            target: Point2::new(80.0, 90.0),
            fixed: false,
            radius: 1.0,
            strength: 1.0,
            rotation_radians: 0.0,
            scale: 1.0,
            falloff: "smooth".to_string(),
        };
        let result = apply_rigid_mls_puppet_to_point(pin.source, &[pin]);
        assert_near(result.x, 80.0, 0.001);
        assert_near(result.y, 90.0, 0.001);
    }

    #[test]
    fn arap_constraints_preserve_triangle_edge_lengths() {
        let points = [
            Point2::new(0.0, 0.0),
            Point2::new(100.0, 0.0),
            Point2::new(0.0, 100.0),
        ];
        let triangles = [[0, 1, 2]];
        let pins = [
            EvaluatedPuppetPin {
                id: "fixed".to_string(),
                role: Some("position".to_string()),
                parent: None,
                source: points[0],
                target: points[0],
                fixed: false,
                radius: 1.0,
                strength: 1.0,
                rotation_radians: 0.0,
                scale: 1.0,
                falloff: "constant".to_string(),
            },
            EvaluatedPuppetPin {
                id: "moved".to_string(),
                role: Some("position".to_string()),
                parent: None,
                source: points[1],
                target: Point2::new(0.0, 100.0),
                fixed: false,
                radius: 1.0,
                strength: 1.0,
                rotation_radians: 0.0,
                scale: 1.0,
                falloff: "constant".to_string(),
            },
        ];
        let result = apply_arap_puppet_to_points(&points, &triangles, &pins);
        assert_near(point_distance(result[0], result[2]), 100.0, 0.05);
        assert_near(
            point_distance(result[1], result[2]),
            100.0_f32.hypot(100.0),
            0.05,
        );
    }
}
