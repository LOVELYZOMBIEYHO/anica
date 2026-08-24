// =========================================
// =========================================
// crates/motionloom/src/simulation/rigid.rs

use std::collections::BTreeMap;

use crate::simulation::model::{
    RigidBodyAngularVelocity, RigidBodyColliderSize, RigidBodyLinearVelocity, RigidBodyNode,
    RigidBodyShape, RigidBodyType,
};

type Vec3 = [f32; 3];
type Quat = [f32; 4];

const CONTACT_SLOP: f32 = 1.0e-3;
const POSITION_CORRECTION: f32 = 0.72;
const MAX_ANGULAR_SPEED: f32 = 80.0;

trait PhysicsBackend3D {
    fn sample(
        &self,
        inputs: &[RigidBody3DInput<'_>],
        time_sec: f32,
        fixed_step: f32,
        iterations: u32,
        gravity: Vec3,
    ) -> Vec<RigidBody3DOutput>;
}

#[derive(Debug, Default, Clone, Copy)]
struct DeterministicPhysicsBackend3D;

impl PhysicsBackend3D for DeterministicPhysicsBackend3D {
    fn sample(
        &self,
        inputs: &[RigidBody3DInput<'_>],
        time_sec: f32,
        fixed_step: f32,
        iterations: u32,
        gravity: Vec3,
    ) -> Vec<RigidBody3DOutput> {
        let mut states = initial_states(inputs);
        let mut cache = BTreeMap::new();
        advance_states(
            &mut states,
            &mut cache,
            time_sec.max(0.0),
            fixed_step.clamp(1.0 / 1000.0, 1.0 / 15.0),
            iterations,
            gravity,
        );
        outputs_from_states(&states)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RigidBody3DInput<'a> {
    pub node: &'a RigidBodyNode,
    pub position: Vec3,
    pub rotation: Vec3,
    /// Effective render-space bounds used only by shape=auto.
    pub auto_collider_size: Option<Vec3>,
    /// Primitive geometry selects the safest analytic or convex auto shape.
    pub auto_collider_shape: Option<RigidBodyShape>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RigidBody3DOutput {
    pub position: Vec3,
    pub rotation: Vec3,
    pub orientation: Quat,
    pub collider_size: Vec3,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub sleeping: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RigidBody3DContactSummary {
    pub a: usize,
    pub b: usize,
    pub points: usize,
    pub maximum_penetration: f32,
}

#[derive(Debug, Clone)]
struct BodyState<'a> {
    node: &'a RigidBodyNode,
    position: Vec3,
    orientation: Quat,
    velocity: Vec3,
    angular_velocity: Vec3,
    collider: Collider3D,
    inverse_mass: f32,
    local_inverse_inertia: Vec3,
    sleeping: bool,
    sleep_timer: f32,
}

#[derive(Debug, Clone, Copy)]
struct Collider3D {
    shape: RigidBodyShape,
    half_extent: Vec3,
}

#[derive(Debug, Clone, Copy)]
struct Contact {
    a: usize,
    b: usize,
    feature: u8,
    normal: Vec3,
    point: Vec3,
    penetration: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct CachedImpulse {
    normal: Vec3,
    point: Vec3,
    normal_impulse: f32,
    tangent_impulse: Vec3,
}

type PairKey = (usize, usize, u8);

/// Deterministic fixed-step sampling starts from the authored pose so random
/// access never depends on preview history.
pub(crate) fn sample_rigid_bodies_3d(
    inputs: &[RigidBody3DInput<'_>],
    time_sec: f32,
    fixed_step: f32,
    iterations: u32,
    gravity: Vec3,
) -> Vec<RigidBody3DOutput> {
    if inputs.is_empty() {
        return Vec::new();
    }
    DeterministicPhysicsBackend3D.sample(inputs, time_sec, fixed_step, iterations, gravity)
}

/// Sequential preview and export reuse one deterministic simulation timeline.
pub(crate) fn bake_rigid_bodies_3d(
    inputs: &[RigidBody3DInput<'_>],
    frame_count: u32,
    fps: f32,
    fixed_step: f32,
    iterations: u32,
    gravity: Vec3,
) -> Vec<Vec<RigidBody3DOutput>> {
    if inputs.is_empty() || frame_count == 0 {
        return Vec::new();
    }
    let mut states = initial_states(inputs);
    let mut cache = BTreeMap::new();
    let fixed_step = fixed_step.clamp(1.0 / 1000.0, 1.0 / 15.0);
    let frame_step = 1.0 / fps.max(1.0);
    let mut frames = Vec::with_capacity(frame_count as usize);
    frames.push(outputs_from_states(&states));
    for _ in 1..frame_count {
        advance_states(
            &mut states,
            &mut cache,
            frame_step,
            fixed_step,
            iterations,
            gravity,
        );
        frames.push(outputs_from_states(&states));
    }
    frames
}

/// Reconstructs the final collider pose only when authoring diagnostics ask
/// for contact evidence. Normal preview and export never pay this extra
/// narrow-phase cost.
pub(crate) fn rigid_body_3d_contact_summaries(
    inputs: &[RigidBody3DInput<'_>],
    outputs: &[RigidBody3DOutput],
) -> Vec<RigidBody3DContactSummary> {
    let mut states = initial_states(inputs);
    for (state, output) in states.iter_mut().zip(outputs) {
        state.position = output.position;
        state.orientation = output.orientation;
        state.velocity = output.linear_velocity;
        state.angular_velocity = output.angular_velocity;
        state.sleeping = output.sleeping;
    }
    let mut grouped = BTreeMap::<(usize, usize), (usize, f32)>::new();
    for contact in detect_contacts(&states) {
        let entry = grouped.entry((contact.a, contact.b)).or_default();
        entry.0 += 1;
        entry.1 = entry.1.max(contact.penetration);
    }
    grouped
        .into_iter()
        .map(
            |((a, b), (points, maximum_penetration))| RigidBody3DContactSummary {
                a,
                b,
                points,
                maximum_penetration,
            },
        )
        .collect()
}

fn initial_states<'a>(inputs: &'a [RigidBody3DInput<'a>]) -> Vec<BodyState<'a>> {
    inputs
        .iter()
        .map(|input| {
            let inverse_mass = if input.node.body_type == RigidBodyType::Dynamic {
                1.0 / input.node.mass.max(0.0001)
            } else {
                0.0
            };
            let collider = resolve_collider_3d(
                input.node,
                input.auto_collider_size,
                input.auto_collider_shape,
            );
            BodyState {
                node: input.node,
                position: input.position,
                orientation: quat_from_euler_degrees(input.rotation),
                velocity: linear_velocity_3d(input.node),
                angular_velocity: angular_velocity_3d(input.node),
                collider,
                inverse_mass,
                local_inverse_inertia: local_inverse_inertia(input.node, collider),
                sleeping: false,
                sleep_timer: 0.0,
            }
        })
        .collect()
}

fn advance_states(
    states: &mut [BodyState<'_>],
    cache: &mut BTreeMap<PairKey, CachedImpulse>,
    elapsed: f32,
    fixed_step: f32,
    iterations: u32,
    gravity: Vec3,
) {
    let mut remaining = elapsed.max(0.0);
    while remaining > 1.0e-7 {
        let dt = remaining.min(rigid_body_3d_step(states, fixed_step, gravity));
        integrate_states(states, dt, gravity);
        // Dense piles need more than one positional pass: resolving one pair
        // can push the shared body into a neighbour. Rebuilding contacts after
        // every pass avoids carrying that newly-created penetration into the
        // next fixed step. The pass count remains bounded and deterministic.
        let contacts = solve_position_constraints(states, (iterations / 4).clamp(1, 4));
        wake_contact_islands(states, &contacts);
        retain_active_cache(cache, &contacts);
        warm_start_contacts(states, &contacts, cache);
        for _ in 0..iterations.clamp(1, 16) {
            solve_contacts(states, &contacts, cache);
        }
        apply_rolling_friction(states, &contacts, cache, dt);
        update_sleep_islands(states, &contacts, dt, gravity);
        remaining -= dt;
    }
}

fn solve_position_constraints(states: &mut [BodyState<'_>], passes: u32) -> Vec<Contact> {
    for _ in 0..passes {
        let contacts = detect_contacts(states);
        if contacts.is_empty() {
            return contacts;
        }
        correct_positions(states, &contacts);
    }
    detect_contacts(states)
}

fn outputs_from_states(states: &[BodyState<'_>]) -> Vec<RigidBody3DOutput> {
    states
        .iter()
        .map(|state| RigidBody3DOutput {
            position: state.position,
            rotation: quat_to_euler_degrees(state.orientation),
            orientation: state.orientation,
            collider_size: scale(state.collider.half_extent, 2.0),
            linear_velocity: state.velocity,
            angular_velocity: state.angular_velocity,
            sleeping: state.sleeping,
        })
        .collect()
}

fn rigid_body_3d_step(states: &[BodyState<'_>], fixed_step: f32, gravity: Vec3) -> f32 {
    let adaptive = states
        .iter()
        .filter(|state| {
            state.node.body_type == RigidBodyType::Dynamic
                && state.node.continuous_collision
                && !state.sleeping
        })
        .fold(fixed_step, |step, state| {
            let projected_velocity = add(state.velocity, scale(gravity, fixed_step));
            let speed = length(projected_velocity);
            let safe_distance = state
                .collider
                .half_extent
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
                * 0.35;
            if speed > 1.0e-6 {
                step.min((safe_distance / speed).clamp(1.0 / 1000.0, fixed_step))
            } else {
                step
            }
        });
    let swept = swept_scene_time_of_impact(states, fixed_step);
    adaptive.min(swept.unwrap_or(fixed_step)).max(1.0 / 4000.0)
}

/// Translational swept AABB time-of-impact prevents a fast body from crossing
/// a collider between two discrete solver states. Rotation still uses the
/// conservative world extents, so the test errs toward an earlier contact.
fn swept_scene_time_of_impact(states: &[BodyState<'_>], maximum: f32) -> Option<f32> {
    let mut earliest = None::<f32>;
    for a in 0..states.len() {
        if states[a].node.body_type != RigidBodyType::Dynamic
            || !states[a].node.continuous_collision
            || states[a].sleeping
        {
            continue;
        }
        for b in 0..states.len() {
            if a == b || states[b].node.body_type == RigidBodyType::Dynamic && b < a {
                continue;
            }
            if world_aabb_overlap(&states[a], &states[b]) {
                continue;
            }
            let relative_velocity = sub(states[a].velocity, states[b].velocity);
            let extent_a = world_aabb_extent(&states[a]);
            let extent_b = world_aabb_extent(&states[b]);
            let delta = sub(states[a].position, states[b].position);
            let mut entry = 0.0_f32;
            let mut exit = maximum;
            let mut possible = true;
            for axis in 0..3 {
                let radius = extent_a[axis] + extent_b[axis];
                let speed = relative_velocity[axis];
                if speed.abs() < 1.0e-8 {
                    if delta[axis].abs() > radius {
                        possible = false;
                        break;
                    }
                    continue;
                }
                let first = (-radius - delta[axis]) / speed;
                let second = (radius - delta[axis]) / speed;
                entry = entry.max(first.min(second));
                exit = exit.min(first.max(second));
                if entry > exit {
                    possible = false;
                    break;
                }
            }
            if possible && entry >= 0.0 && entry <= maximum {
                let safe = (entry * 0.98).max(1.0 / 4000.0);
                earliest = Some(earliest.map_or(safe, |current| current.min(safe)));
            }
        }
    }
    earliest
}

fn integrate_states(states: &mut [BodyState<'_>], dt: f32, gravity: Vec3) {
    for state in states {
        if state.node.body_type != RigidBodyType::Dynamic || state.sleeping {
            continue;
        }
        let linear_decay = (-state.node.linear_damping * dt).exp();
        let angular_decay = (-state.node.angular_damping * dt).exp();
        state.velocity = scale(add(state.velocity, scale(gravity, dt)), linear_decay);
        state.position = add(state.position, scale(state.velocity, dt));
        state.angular_velocity = scale(state.angular_velocity, angular_decay);
        let angular_speed = length(state.angular_velocity);
        if angular_speed > MAX_ANGULAR_SPEED {
            state.angular_velocity =
                scale(state.angular_velocity, MAX_ANGULAR_SPEED / angular_speed);
        }
        state.orientation = integrate_orientation(state.orientation, state.angular_velocity, dt);
    }
}

/// Broad phase uses rotation-aware AABBs; narrow phase uses OBB SAT for
/// box-like colliders and a direct sphere path for spherical bodies.
fn detect_contacts(states: &[BodyState<'_>]) -> Vec<Contact> {
    let mut contacts = Vec::new();
    for a in 0..states.len() {
        if states[a].node.body_type != RigidBodyType::Dynamic {
            continue;
        }
        for b in 0..states.len() {
            if a == b || states[b].node.body_type == RigidBodyType::Dynamic && b < a {
                continue;
            }
            if !world_aabb_overlap(&states[a], &states[b]) {
                continue;
            }
            contacts.extend(narrow_phase_contacts(states, a, b));
        }
    }
    contacts
}

fn narrow_phase_contacts(states: &[BodyState<'_>], a: usize, b: usize) -> Vec<Contact> {
    let a_sphere = states[a].collider.shape == RigidBodyShape::Sphere;
    let b_sphere = states[b].collider.shape == RigidBodyShape::Sphere;
    match (a_sphere, b_sphere) {
        (true, true) => sphere_sphere_contact(states, a, b).into_iter().collect(),
        _ => obb_contacts(states, a, b),
    }
}

fn sphere_sphere_contact(states: &[BodyState<'_>], a: usize, b: usize) -> Option<Contact> {
    let delta = sub(states[a].position, states[b].position);
    let distance = length(delta);
    let radius = states[a].collider.half_extent[0] + states[b].collider.half_extent[0];
    if distance > radius {
        return None;
    }
    let normal = if distance > 1.0e-6 {
        scale(delta, 1.0 / distance)
    } else {
        [0.0, 1.0, 0.0]
    };
    Some(Contact {
        a,
        b,
        feature: 0,
        normal,
        point: sub(
            states[a].position,
            scale(normal, states[a].collider.half_extent[0]),
        ),
        penetration: radius - distance,
    })
}

fn obb_contacts(states: &[BodyState<'_>], a: usize, b: usize) -> Vec<Contact> {
    if states[b].node.body_type != RigidBodyType::Dynamic && quat_is_identity(states[b].orientation)
    {
        return dynamic_static_aabb_contacts(states, a, b);
    }
    let axes_a = quat_axes(states[a].orientation);
    let axes_b = quat_axes(states[b].orientation);
    let center_delta = sub(states[a].position, states[b].position);
    let mut candidates = Vec::with_capacity(15);
    candidates.extend(axes_a);
    candidates.extend(axes_b);
    for axis_a in axes_a {
        for axis_b in axes_b {
            let axis = cross(axis_a, axis_b);
            if length_squared(axis) > 1.0e-8 {
                candidates.push(normalize(axis));
            }
        }
    }

    let mut best_axis = [0.0, 1.0, 0.0];
    let mut best_penetration = f32::INFINITY;
    for mut axis in candidates {
        if dot(center_delta, axis) < 0.0 {
            axis = scale(axis, -1.0);
        }
        let distance = dot(center_delta, axis).abs();
        let radius_a = obb_projection_radius(&states[a], axes_a, axis);
        let radius_b = obb_projection_radius(&states[b], axes_b, axis);
        let penetration = radius_a + radius_b - distance;
        if penetration < 0.0 {
            return Vec::new();
        }
        if penetration < best_penetration {
            best_penetration = penetration;
            best_axis = axis;
        }
    }

    let mut points = obb_vertices(&states[a])
        .into_iter()
        .filter(|point| point_inside_obb(*point, &states[b], CONTACT_SLOP * 4.0))
        .chain(
            obb_vertices(&states[b])
                .into_iter()
                .filter(|point| point_inside_obb(*point, &states[a], CONTACT_SLOP * 4.0)),
        )
        .collect::<Vec<_>>();
    deduplicate_points(&mut points);
    if points.is_empty() {
        let point_a = support_face_center(&states[a], axes_a, scale(best_axis, -1.0));
        let point_b = support_face_center(&states[b], axes_b, best_axis);
        points.push(scale(add(point_a, point_b), 0.5));
    }
    reduce_manifold_points(&mut points, best_axis, 4);
    points
        .into_iter()
        .enumerate()
        .map(|(feature, point)| Contact {
            a,
            b,
            feature: feature as u8,
            normal: best_axis,
            point,
            penetration: best_penetration,
        })
        .collect()
}

/// Authored floors and walls are normally axis-aligned. Resolving them from
/// rotation-aware world bounds gives a stable supporting plane while dynamic
/// bodies retain quaternion rotation and angular response.
fn dynamic_static_aabb_contacts(states: &[BodyState<'_>], a: usize, b: usize) -> Vec<Contact> {
    let extent_a = world_aabb_extent(&states[a]);
    let extent_b = world_aabb_extent(&states[b]);
    let delta = sub(states[a].position, states[b].position);
    let overlaps: Vec3 =
        std::array::from_fn(|axis| extent_a[axis] + extent_b[axis] - delta[axis].abs());
    if overlaps.iter().any(|overlap| *overlap < 0.0) {
        return Vec::new();
    }
    let axis = (0..3)
        .min_by(|a, b| overlaps[*a].total_cmp(&overlaps[*b]))
        .unwrap_or(1);
    let sign = if delta[axis] >= 0.0 { 1.0 } else { -1.0 };
    let mut normal = [0.0; 3];
    normal[axis] = sign;
    let tangents = (0..3)
        .filter(|candidate| *candidate != axis)
        .collect::<Vec<_>>();
    let mut ranges = [[0.0; 2]; 2];
    for (range, tangent) in ranges.iter_mut().zip(tangents.iter().copied()) {
        range[0] = (states[a].position[tangent] - extent_a[tangent])
            .max(states[b].position[tangent] - extent_b[tangent]);
        range[1] = (states[a].position[tangent] + extent_a[tangent])
            .min(states[b].position[tangent] + extent_b[tangent]);
    }
    let plane = states[b].position[axis] + extent_b[axis] * sign;
    let mut points = Vec::with_capacity(4);
    for u in [ranges[0][0], ranges[0][1]] {
        for v in [ranges[1][0], ranges[1][1]] {
            let mut point = [0.0; 3];
            point[axis] = plane;
            point[tangents[0]] = u;
            point[tangents[1]] = v;
            if !points
                .iter()
                .any(|existing| length(sub(*existing, point)) < 1.0e-4)
            {
                points.push(point);
            }
        }
    }
    points
        .into_iter()
        .enumerate()
        .map(|(feature, point)| Contact {
            a,
            b,
            feature: feature as u8,
            normal,
            point,
            penetration: overlaps[axis],
        })
        .collect()
}

fn obb_vertices(state: &BodyState<'_>) -> [Vec3; 8] {
    let axes = quat_axes(state.orientation);
    let mut index = 0;
    std::array::from_fn(|_| {
        let signs = [
            if index & 1 == 0 { -1.0 } else { 1.0 },
            if index & 2 == 0 { -1.0 } else { 1.0 },
            if index & 4 == 0 { -1.0 } else { 1.0 },
        ];
        index += 1;
        let mut point = state.position;
        for axis in 0..3 {
            point = add(
                point,
                scale(axes[axis], state.collider.half_extent[axis] * signs[axis]),
            );
        }
        point
    })
}

fn point_inside_obb(point: Vec3, state: &BodyState<'_>, tolerance: f32) -> bool {
    let local = sub(point, state.position);
    quat_axes(state.orientation)
        .iter()
        .enumerate()
        .all(|(axis, basis)| {
            dot(local, *basis).abs() <= state.collider.half_extent[axis] + tolerance
        })
}

fn deduplicate_points(points: &mut Vec<Vec3>) {
    let mut unique = Vec::with_capacity(points.len());
    for point in points.drain(..) {
        if !unique
            .iter()
            .any(|existing| length(sub(*existing, point)) < 1.0e-4)
        {
            unique.push(point);
        }
    }
    *points = unique;
}

fn reduce_manifold_points(points: &mut Vec<Vec3>, normal: Vec3, maximum: usize) {
    if points.len() <= maximum {
        return;
    }
    let tangent = normalize(if normal[1].abs() < 0.9 {
        cross(normal, [0.0, 1.0, 0.0])
    } else {
        cross(normal, [1.0, 0.0, 0.0])
    });
    let bitangent = normalize(cross(normal, tangent));
    points.sort_by(|a, b| {
        dot(*a, tangent)
            .total_cmp(&dot(*b, tangent))
            .then(dot(*a, bitangent).total_cmp(&dot(*b, bitangent)))
    });
    let candidates = [0, points.len() - 1, points.len() / 3, points.len() * 2 / 3];
    let reduced = candidates.map(|index| points[index]);
    points.clear();
    points.extend(reduced);
    deduplicate_points(points);
}

fn obb_projection_radius(state: &BodyState<'_>, axes: [Vec3; 3], axis: Vec3) -> f32 {
    (0..3)
        .map(|i| state.collider.half_extent[i] * dot(axes[i], axis).abs())
        .sum()
}

/// Face-center support avoids artificial corner torque for a box resting flat.
fn support_face_center(state: &BodyState<'_>, axes: [Vec3; 3], direction: Vec3) -> Vec3 {
    let (axis_index, projection) = axes
        .iter()
        .enumerate()
        .map(|(index, axis)| (index, dot(*axis, direction)))
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .unwrap_or((1, 1.0));
    add(
        state.position,
        scale(
            axes[axis_index],
            state.collider.half_extent[axis_index] * projection.signum(),
        ),
    )
}

fn correct_positions(states: &mut [BodyState<'_>], contacts: &[Contact]) {
    let mut corrected_pairs = Vec::new();
    for contact in contacts {
        let pair = (contact.a.min(contact.b), contact.a.max(contact.b));
        if corrected_pairs.contains(&pair) {
            continue;
        }
        corrected_pairs.push(pair);
        let inverse_mass_sum = states[contact.a].inverse_mass + states[contact.b].inverse_mass;
        if inverse_mass_sum <= 0.0 {
            continue;
        }
        let magnitude = ((contact.penetration - CONTACT_SLOP).max(0.0) * POSITION_CORRECTION)
            / inverse_mass_sum;
        let correction = scale(contact.normal, magnitude);
        states[contact.a].position = add(
            states[contact.a].position,
            scale(correction, states[contact.a].inverse_mass),
        );
        states[contact.b].position = sub(
            states[contact.b].position,
            scale(correction, states[contact.b].inverse_mass),
        );
    }
}

fn retain_active_cache(cache: &mut BTreeMap<PairKey, CachedImpulse>, contacts: &[Contact]) {
    let active = contacts
        .iter()
        .map(|contact| pair_key(contact.a, contact.b, contact.feature))
        .collect::<Vec<_>>();
    cache.retain(|key, _| active.contains(key));
    for contact in contacts {
        let entry = cache
            .entry(pair_key(contact.a, contact.b, contact.feature))
            .or_default();
        if dot(entry.normal, contact.normal) < 0.85 || length(sub(entry.point, contact.point)) > 0.2
        {
            *entry = CachedImpulse::default();
        }
        entry.normal = contact.normal;
        entry.point = contact.point;
    }
}

fn warm_start_contacts(
    states: &mut [BodyState<'_>],
    contacts: &[Contact],
    cache: &BTreeMap<PairKey, CachedImpulse>,
) {
    for contact in contacts {
        let Some(cached) = cache.get(&pair_key(contact.a, contact.b, contact.feature)) else {
            continue;
        };
        let impulse = add(
            scale(contact.normal, cached.normal_impulse),
            cached.tangent_impulse,
        );
        apply_contact_impulse(states, *contact, impulse);
    }
}

fn solve_contacts(
    states: &mut [BodyState<'_>],
    contacts: &[Contact],
    cache: &mut BTreeMap<PairKey, CachedImpulse>,
) {
    for contact in contacts {
        let ra = sub(contact.point, states[contact.a].position);
        let rb = sub(contact.point, states[contact.b].position);
        let relative_velocity = contact_relative_velocity(states, *contact, ra, rb);
        let normal_speed = dot(relative_velocity, contact.normal);
        let denominator = impulse_denominator(states, *contact, ra, rb, contact.normal);
        if denominator <= 1.0e-8 {
            continue;
        }

        let threshold = states[contact.a]
            .node
            .restitution_threshold
            .max(states[contact.b].node.restitution_threshold);
        let restitution = if normal_speed < -threshold {
            states[contact.a]
                .node
                .restitution
                .min(states[contact.b].node.restitution)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let entry = cache
            .entry(pair_key(contact.a, contact.b, contact.feature))
            .or_default();
        let previous_normal = entry.normal_impulse;
        entry.normal_impulse =
            (previous_normal + (-(1.0 + restitution) * normal_speed / denominator)).max(0.0);
        let normal_delta = entry.normal_impulse - previous_normal;
        if normal_delta.abs() > 1.0e-8 {
            apply_contact_impulse(states, *contact, scale(contact.normal, normal_delta));
        }

        let relative_velocity = contact_relative_velocity(states, *contact, ra, rb);
        let tangent_velocity = sub(
            relative_velocity,
            scale(contact.normal, dot(relative_velocity, contact.normal)),
        );
        let tangent_speed = length(tangent_velocity);
        if tangent_speed <= 1.0e-6 {
            continue;
        }
        let tangent = scale(tangent_velocity, 1.0 / tangent_speed);
        let tangent_denominator = impulse_denominator(states, *contact, ra, rb, tangent);
        if tangent_denominator <= 1.0e-8 {
            continue;
        }
        let friction = (states[contact.a].node.friction * states[contact.b].node.friction)
            .sqrt()
            .clamp(0.0, 1.0);
        let max_friction = friction * entry.normal_impulse;
        let desired = scale(tangent, -tangent_speed / tangent_denominator);
        let previous_tangent = entry.tangent_impulse;
        entry.tangent_impulse = clamp_length(add(previous_tangent, desired), max_friction);
        apply_contact_impulse(
            states,
            *contact,
            sub(entry.tangent_impulse, previous_tangent),
        );
    }
}

fn contact_relative_velocity(
    states: &[BodyState<'_>],
    contact: Contact,
    ra: Vec3,
    rb: Vec3,
) -> Vec3 {
    let velocity_a = add(
        states[contact.a].velocity,
        cross(states[contact.a].angular_velocity, ra),
    );
    let velocity_b = add(
        states[contact.b].velocity,
        cross(states[contact.b].angular_velocity, rb),
    );
    sub(velocity_a, velocity_b)
}

fn impulse_denominator(
    states: &[BodyState<'_>],
    contact: Contact,
    ra: Vec3,
    rb: Vec3,
    direction: Vec3,
) -> f32 {
    let angular_a = cross(
        world_inverse_inertia_mul(&states[contact.a], cross(ra, direction)),
        ra,
    );
    let angular_b = cross(
        world_inverse_inertia_mul(&states[contact.b], cross(rb, direction)),
        rb,
    );
    states[contact.a].inverse_mass
        + states[contact.b].inverse_mass
        + dot(direction, add(angular_a, angular_b))
}

fn apply_contact_impulse(states: &mut [BodyState<'_>], contact: Contact, impulse: Vec3) {
    let ra = sub(contact.point, states[contact.a].position);
    let rb = sub(contact.point, states[contact.b].position);
    if states[contact.a].inverse_mass > 0.0 {
        states[contact.a].velocity = add(
            states[contact.a].velocity,
            scale(impulse, states[contact.a].inverse_mass),
        );
        states[contact.a].angular_velocity = add(
            states[contact.a].angular_velocity,
            world_inverse_inertia_mul(&states[contact.a], cross(ra, impulse)),
        );
    }
    if states[contact.b].inverse_mass > 0.0 {
        states[contact.b].velocity = sub(
            states[contact.b].velocity,
            scale(impulse, states[contact.b].inverse_mass),
        );
        states[contact.b].angular_velocity = sub(
            states[contact.b].angular_velocity,
            world_inverse_inertia_mul(&states[contact.b], cross(rb, impulse)),
        );
    }
}

fn apply_rolling_friction(
    states: &mut [BodyState<'_>],
    contacts: &[Contact],
    cache: &BTreeMap<PairKey, CachedImpulse>,
    dt: f32,
) {
    for contact in contacts {
        let Some(cached) = cache.get(&pair_key(contact.a, contact.b, contact.feature)) else {
            continue;
        };
        let coefficient = (states[contact.a].node.rolling_friction
            * states[contact.b].node.rolling_friction)
            .sqrt()
            .clamp(0.0, 1.0);
        if coefficient <= 0.0 || cached.normal_impulse <= 0.0 {
            continue;
        }
        for index in [contact.a, contact.b] {
            if states[index].inverse_mass <= 0.0 || states[index].sleeping {
                continue;
            }
            let angular_speed = length(states[index].angular_velocity);
            if angular_speed <= 1.0e-7 {
                continue;
            }
            let impulse_decay = coefficient * cached.normal_impulse * dt;
            let speed_decay = (-coefficient * 30.0 * dt).exp();
            let target_speed = (angular_speed * speed_decay - impulse_decay).max(0.0);
            states[index].angular_velocity =
                scale(states[index].angular_velocity, target_speed / angular_speed);
        }
    }
}

fn wake_contact_islands(states: &mut [BodyState<'_>], contacts: &[Contact]) {
    for contact in contacts {
        let wake_a = states[contact.a].sleeping && body_can_wake(&states[contact.b]);
        let wake_b = states[contact.b].sleeping && body_can_wake(&states[contact.a]);
        if wake_a {
            states[contact.a].sleeping = false;
            states[contact.a].sleep_timer = 0.0;
        }
        if wake_b {
            states[contact.b].sleeping = false;
            states[contact.b].sleep_timer = 0.0;
        }
    }
}

fn body_can_wake(state: &BodyState<'_>) -> bool {
    state.node.body_type == RigidBodyType::Kinematic
        || state.node.body_type == RigidBodyType::Dynamic
            && !state.sleeping
            && body_is_moving(state)
}

/// Sleeping requires sustained linear and angular stillness while supported.
fn update_sleep_islands(
    states: &mut [BodyState<'_>],
    contacts: &[Contact],
    dt: f32,
    gravity: Vec3,
) {
    let up = normalize(scale(gravity, -1.0));
    let mut supported = vec![false; states.len()];
    let mut connected_to_active = vec![false; states.len()];
    for contact in contacts {
        if states[contact.a].node.body_type == RigidBodyType::Dynamic
            && dot(contact.normal, up) > 0.45
        {
            supported[contact.a] = true;
        }
        if states[contact.b].node.body_type == RigidBodyType::Dynamic
            && dot(scale(contact.normal, -1.0), up) > 0.45
        {
            supported[contact.b] = true;
        }
        if states[contact.a].node.body_type == RigidBodyType::Dynamic
            && states[contact.b].node.body_type == RigidBodyType::Dynamic
        {
            let active = body_is_moving(&states[contact.a]) || body_is_moving(&states[contact.b]);
            connected_to_active[contact.a] |= active;
            connected_to_active[contact.b] |= active;
        }
    }

    for index in 0..states.len() {
        if states[index].node.body_type != RigidBodyType::Dynamic || !states[index].node.sleep {
            continue;
        }
        let linear_still =
            length(states[index].velocity) < states[index].node.sleep_linear_threshold.max(1.0e-6);
        let angular_still = length(states[index].angular_velocity)
            < states[index].node.sleep_angular_threshold.max(1.0e-6);
        if supported[index] && linear_still && angular_still && !connected_to_active[index] {
            states[index].sleep_timer += dt;
            if states[index].sleep_timer >= states[index].node.sleep_time {
                states[index].sleeping = true;
                states[index].velocity = [0.0; 3];
                states[index].angular_velocity = [0.0; 3];
            }
        } else if !states[index].sleeping {
            states[index].sleep_timer = 0.0;
        }
    }
}

fn body_is_moving(state: &BodyState<'_>) -> bool {
    state.node.body_type == RigidBodyType::Kinematic
        || state.node.body_type == RigidBodyType::Dynamic
            && !state.sleeping
            && (length(state.velocity) > state.node.sleep_linear_threshold
                || length(state.angular_velocity) > state.node.sleep_angular_threshold)
}

fn world_inverse_inertia_mul(state: &BodyState<'_>, vector: Vec3) -> Vec3 {
    if state.inverse_mass <= 0.0 {
        return [0.0; 3];
    }
    let axes = quat_axes(state.orientation);
    let local = [
        dot(vector, axes[0]) * state.local_inverse_inertia[0],
        dot(vector, axes[1]) * state.local_inverse_inertia[1],
        dot(vector, axes[2]) * state.local_inverse_inertia[2],
    ];
    add(
        add(scale(axes[0], local[0]), scale(axes[1], local[1])),
        scale(axes[2], local[2]),
    )
}

fn local_inverse_inertia(node: &RigidBodyNode, collider: Collider3D) -> Vec3 {
    if node.body_type != RigidBodyType::Dynamic {
        return [0.0; 3];
    }
    let mass = node.mass.max(0.0001);
    let inertia = match collider.shape {
        RigidBodyShape::Sphere => {
            let value = 0.4 * mass * node.radius * node.radius;
            [value; 3]
        }
        RigidBodyShape::Cylinder => {
            let axial = 0.5 * mass * node.radius * node.radius;
            let radial =
                mass * (3.0 * node.radius * node.radius + node.height * node.height) / 12.0;
            [radial, axial, radial]
        }
        RigidBodyShape::Capsule => {
            let radial = node.radius;
            let total_height = node.height + radial * 2.0;
            let axial = 0.5 * mass * radial * radial;
            let side = mass * (3.0 * radial * radial + total_height * total_height) / 12.0;
            [side, axial, side]
        }
        _ => {
            let size = collider.half_extent.map(|value| value * 2.0);
            [
                mass * (size[1] * size[1] + size[2] * size[2]) / 12.0,
                mass * (size[0] * size[0] + size[2] * size[2]) / 12.0,
                mass * (size[0] * size[0] + size[1] * size[1]) / 12.0,
            ]
        }
    };
    inertia.map(|value| if value > 1.0e-8 { 1.0 / value } else { 0.0 })
}

fn world_aabb_overlap(a: &BodyState<'_>, b: &BodyState<'_>) -> bool {
    let extent_a = world_aabb_extent(a);
    let extent_b = world_aabb_extent(b);
    (0..3)
        .all(|axis| (a.position[axis] - b.position[axis]).abs() <= extent_a[axis] + extent_b[axis])
}

fn world_aabb_extent(state: &BodyState<'_>) -> Vec3 {
    let axes = quat_axes(state.orientation);
    std::array::from_fn(|world_axis| {
        (0..3)
            .map(|local_axis| {
                axes[local_axis][world_axis].abs() * state.collider.half_extent[local_axis]
            })
            .sum()
    })
}

fn resolve_collider_3d(
    node: &RigidBodyNode,
    auto_size: Option<Vec3>,
    auto_shape: Option<RigidBodyShape>,
) -> Collider3D {
    let shape = if node.shape == RigidBodyShape::Auto {
        auto_shape.unwrap_or(RigidBodyShape::Box)
    } else {
        node.shape
    };
    let half_extent = match shape {
        RigidBodyShape::Sphere if node.shape == RigidBodyShape::Auto => {
            let radius = auto_size.unwrap_or_else(|| size_3d(node))[0] * 0.5;
            [radius; 3]
        }
        RigidBodyShape::Sphere => [node.radius; 3],
        RigidBodyShape::Cylinder if node.shape == RigidBodyShape::Auto => auto_size
            .unwrap_or_else(|| size_3d(node))
            .map(|axis| axis.abs().max(1.0e-4) * 0.5),
        RigidBodyShape::Cylinder => [node.radius, node.height * 0.5, node.radius],
        RigidBodyShape::Capsule => [node.radius, node.height * 0.5 + node.radius, node.radius],
        _ if node.shape == RigidBodyShape::Auto => auto_size
            .unwrap_or_else(|| size_3d(node))
            .map(|axis| axis.abs().max(1.0e-4) * 0.5),
        _ => size_3d(node).map(|axis| axis * 0.5),
    };
    Collider3D { shape, half_extent }
}

fn size_3d(node: &RigidBodyNode) -> Vec3 {
    match node.size {
        RigidBodyColliderSize::D3(size) => size,
        RigidBodyColliderSize::D2(_) => [1.0; 3],
    }
}

fn linear_velocity_3d(node: &RigidBodyNode) -> Vec3 {
    match node.velocity {
        RigidBodyLinearVelocity::D3(value) => value,
        RigidBodyLinearVelocity::D2(_) => [0.0; 3],
    }
}

fn angular_velocity_3d(node: &RigidBodyNode) -> Vec3 {
    match node.angular_velocity {
        RigidBodyAngularVelocity::D3(value) => value,
        RigidBodyAngularVelocity::D2(_) => [0.0; 3],
    }
}

fn pair_key(a: usize, b: usize, feature: u8) -> PairKey {
    (a.min(b), a.max(b), feature)
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(value: Vec3, factor: f32) -> Vec3 {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length_squared(value: Vec3) -> f32 {
    dot(value, value)
}

fn length(value: Vec3) -> f32 {
    length_squared(value).sqrt()
}

fn normalize(value: Vec3) -> Vec3 {
    let magnitude = length(value);
    if magnitude > 1.0e-8 {
        scale(value, 1.0 / magnitude)
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn clamp_length(value: Vec3, maximum: f32) -> Vec3 {
    let magnitude = length(value);
    if magnitude > maximum && magnitude > 1.0e-8 {
        scale(value, maximum / magnitude)
    } else {
        value
    }
}

fn quat_from_euler_degrees(rotation: Vec3) -> Quat {
    let [x, y, z] = rotation.map(f32::to_radians);
    let qx = axis_angle_quat([1.0, 0.0, 0.0], x);
    let qy = axis_angle_quat([0.0, 1.0, 0.0], y);
    let qz = axis_angle_quat([0.0, 0.0, 1.0], z);
    // Actor authoring applies yaw (Y), then pitch (X), then roll (Z).
    quat_normalize(quat_mul(qz, quat_mul(qx, qy)))
}

fn quat_to_euler_degrees(q: Quat) -> Vec3 {
    let axes = quat_axes(q);
    let pitch_x = axes[1][2].clamp(-1.0, 1.0).asin();
    let cos_x = pitch_x.cos();
    let (yaw_y, roll_z) = if cos_x.abs() > 1.0e-6 {
        (
            (-axes[0][2]).atan2(axes[2][2]),
            (-axes[1][0]).atan2(axes[1][1]),
        )
    } else {
        (axes[2][0].atan2(axes[0][0]), 0.0)
    };
    [
        pitch_x.to_degrees(),
        yaw_y.to_degrees(),
        roll_z.to_degrees(),
    ]
}

fn axis_angle_quat(axis: Vec3, angle: f32) -> Quat {
    let (sine, cosine) = (angle * 0.5).sin_cos();
    [axis[0] * sine, axis[1] * sine, axis[2] * sine, cosine]
}

fn integrate_orientation(orientation: Quat, angular_velocity: Vec3, dt: f32) -> Quat {
    let omega = [
        angular_velocity[0],
        angular_velocity[1],
        angular_velocity[2],
        0.0,
    ];
    let derivative = quat_mul(omega, orientation).map(|value| value * 0.5);
    quat_normalize(std::array::from_fn(|i| orientation[i] + derivative[i] * dt))
}

fn quat_mul(a: Quat, b: Quat) -> Quat {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn quat_normalize(q: Quat) -> Quat {
    let magnitude = q.iter().map(|value| value * value).sum::<f32>().sqrt();
    if magnitude > 1.0e-8 {
        q.map(|value| value / magnitude)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

fn quat_is_identity(q: Quat) -> bool {
    q[0].abs() < 1.0e-6
        && q[1].abs() < 1.0e-6
        && q[2].abs() < 1.0e-6
        && (q[3].abs() - 1.0).abs() < 1.0e-6
}

fn quat_axes(q: Quat) -> [Vec3; 3] {
    let [x, y, z, w] = quat_normalize(q);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::model::{RigidBodyDimension, RigidBodyShape};

    fn body(id: &str, target: &str, body_type: RigidBodyType) -> RigidBodyNode {
        RigidBodyNode {
            id: id.to_string(),
            target: target.to_string(),
            dimension: RigidBodyDimension::D3,
            body_type,
            shape: RigidBodyShape::Box,
            size: RigidBodyColliderSize::D3([1.0, 1.0, 1.0]),
            radius: 0.5,
            height: 1.0,
            mass: 1.0,
            velocity: RigidBodyLinearVelocity::D3([0.0; 3]),
            angular_velocity: RigidBodyAngularVelocity::D3([0.0; 3]),
            gravity: RigidBodyLinearVelocity::D3([0.0, -9.81, 0.0]),
            friction: 0.7,
            rolling_friction: 0.08,
            restitution: 0.0,
            restitution_threshold: 0.5,
            linear_damping: 0.02,
            angular_damping: 0.05,
            continuous_collision: true,
            sleep: true,
            sleep_linear_threshold: 0.03,
            sleep_angular_threshold: 0.04,
            sleep_time: 0.35,
        }
    }

    fn inputs<'a>(
        floor: &'a RigidBodyNode,
        body: &'a RigidBodyNode,
        y: f32,
    ) -> [RigidBody3DInput<'a>; 2] {
        [
            RigidBody3DInput {
                node: floor,
                position: [0.0, -0.5, 0.0],
                rotation: [0.0; 3],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
            RigidBody3DInput {
                node: body,
                position: [0.0, y, 0.0],
                rotation: [0.0; 3],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
        ]
    }

    #[test]
    fn dynamic_box_settles_and_sleeps_on_static_box() {
        let floor = body("floor_body", "floor", RigidBodyType::Static);
        let falling = body("falling_body", "falling", RigidBodyType::Dynamic);
        let inputs = inputs(&floor, &falling, 2.0);
        let sampled = sample_rigid_bodies_3d(&inputs, 3.0, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        assert!(
            (sampled[1].position[1] - 0.5).abs() < 0.02,
            "sampled={sampled:?}"
        );
        assert!(sampled[1].sleeping, "sampled={sampled:?}");
    }

    #[test]
    fn contact_friction_removes_angular_motion() {
        let mut floor = body("floor_body", "floor", RigidBodyType::Static);
        floor.size = RigidBodyColliderSize::D3([12.0, 1.0, 12.0]);
        let mut falling = body("falling_body", "falling", RigidBodyType::Dynamic);
        falling.angular_velocity = RigidBodyAngularVelocity::D3([2.0, 1.0, 3.0]);
        let inputs = inputs(&floor, &falling, 1.2);
        let sampled = sample_rigid_bodies_3d(&inputs, 8.0, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        assert!(
            length(sampled[1].angular_velocity) < 0.05,
            "sampled={sampled:?}"
        );
        assert!(sampled[1].sleeping, "sampled={sampled:?}");
    }

    #[test]
    fn sampling_is_independent_of_call_order() {
        let floor = body("floor_body", "floor", RigidBodyType::Static);
        let falling = body("falling_body", "falling", RigidBodyType::Dynamic);
        let inputs = inputs(&floor, &falling, 2.0);
        let first = sample_rigid_bodies_3d(&inputs, 1.0, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        let _later = sample_rigid_bodies_3d(&inputs, 2.0, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        let repeated = sample_rigid_bodies_3d(&inputs, 1.0, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        assert_eq!(first, repeated);
    }

    #[test]
    fn continuous_collision_prevents_fast_body_tunneling() {
        let mut floor = body("floor_body", "floor", RigidBodyType::Static);
        floor.size = RigidBodyColliderSize::D3([4.0, 0.1, 4.0]);
        let mut falling = body("falling_body", "falling", RigidBodyType::Dynamic);
        falling.velocity = RigidBodyLinearVelocity::D3([0.0, -100.0, 0.0]);
        let inputs = [
            RigidBody3DInput {
                node: &floor,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0; 3],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
            RigidBody3DInput {
                node: &falling,
                position: [0.0, 2.0, 0.0],
                rotation: [0.0; 3],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
        ];
        let sampled = sample_rigid_bodies_3d(&inputs, 0.2, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        assert!(sampled[1].position[1] >= 0.54, "sampled={sampled:?}");
    }

    #[test]
    fn baked_frames_match_random_access_sampling() {
        let floor = body("floor_body", "floor", RigidBodyType::Static);
        let falling = body("falling_body", "falling", RigidBodyType::Dynamic);
        let inputs = inputs(&floor, &falling, 2.0);
        let baked = bake_rigid_bodies_3d(&inputs, 61, 30.0, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        let sampled = sample_rigid_bodies_3d(&inputs, 2.0, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        for (baked_body, sampled_body) in baked[60].iter().zip(sampled) {
            for axis in 0..3 {
                assert!((baked_body.position[axis] - sampled_body.position[axis]).abs() < 0.002);
                assert!((baked_body.rotation[axis] - sampled_body.rotation[axis]).abs() < 0.01);
            }
        }
    }

    #[test]
    fn mixed_drop_lab_bodies_reach_angular_sleep() {
        let mut floor = body("floor", "floor", RigidBodyType::Static);
        floor.size = RigidBodyColliderSize::D3([12.0, 0.4, 8.0]);
        let mut pedestal = body("pedestal", "pedestal", RigidBodyType::Kinematic);
        pedestal.size = RigidBodyColliderSize::D3([2.2, 0.45, 2.2]);
        let mut side_wall = body("side", "side", RigidBodyType::Static);
        side_wall.size = RigidBodyColliderSize::D3([0.35, 3.4, 8.0]);
        let mut back_wall = body("back", "back", RigidBodyType::Static);
        back_wall.size = RigidBodyColliderSize::D3([12.0, 3.4, 0.35]);

        let mut coral = body("coral", "coral", RigidBodyType::Dynamic);
        coral.mass = 0.75;
        coral.velocity = RigidBodyLinearVelocity::D3([-0.1, 0.0, -0.1]);
        coral.angular_velocity = RigidBodyAngularVelocity::D3([0.8, 1.4, 0.45]);
        coral.rolling_friction = 0.14;
        coral.restitution = 0.72;

        let mut cyan = body("cyan", "cyan", RigidBodyType::Dynamic);
        cyan.mass = 4.0;
        cyan.size = RigidBodyColliderSize::D3([1.25; 3]);
        cyan.velocity = RigidBodyLinearVelocity::D3([-0.75, 0.0, 0.1]);
        cyan.angular_velocity = RigidBodyAngularVelocity::D3([0.2, -0.45, 0.3]);
        cyan.rolling_friction = 0.18;
        cyan.restitution = 0.16;

        let inputs = [
            RigidBody3DInput {
                node: &floor,
                position: [0.0, -0.2, 0.0],
                rotation: [0.0; 3],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
            RigidBody3DInput {
                node: &pedestal,
                position: [0.0, 0.225, 0.0],
                rotation: [0.0; 3],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
            RigidBody3DInput {
                node: &side_wall,
                position: [-5.82, 1.5, 0.0],
                rotation: [0.0; 3],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
            RigidBody3DInput {
                node: &side_wall,
                position: [5.82, 1.5, 0.0],
                rotation: [0.0; 3],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
            RigidBody3DInput {
                node: &back_wall,
                position: [0.0, 1.5, -3.82],
                rotation: [0.0; 3],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
            RigidBody3DInput {
                node: &coral,
                position: [-3.7, 4.8, -0.6],
                rotation: [14.0, 0.0, 20.0],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
            RigidBody3DInput {
                node: &cyan,
                position: [2.55, 7.1, -0.35],
                rotation: [0.0, 24.0, 8.0],
                auto_collider_size: None,
                auto_collider_shape: None,
            },
        ];
        let sampled = sample_rigid_bodies_3d(&inputs, 8.0, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        for output in &sampled[5..] {
            assert!(output.sleeping, "sampled={sampled:?}");
            assert_eq!(output.angular_velocity, [0.0; 3]);
        }
    }

    #[test]
    fn auto_box_uses_effective_render_bounds() {
        let mut automatic = body("auto", "model", RigidBodyType::Dynamic);
        automatic.shape = RigidBodyShape::Auto;
        let inputs = [RigidBody3DInput {
            node: &automatic,
            position: [0.0; 3],
            rotation: [0.0; 3],
            auto_collider_size: Some([2.0, 3.0, 4.0]),
            auto_collider_shape: None,
        }];
        let states = initial_states(&inputs);
        assert_eq!(states[0].collider.shape, RigidBodyShape::Box);
        assert_eq!(states[0].collider.half_extent, [1.0, 1.5, 2.0]);
    }

    #[test]
    fn auto_primitive_shape_uses_typed_geometry_mapping() {
        let mut automatic = body("auto", "model", RigidBodyType::Dynamic);
        automatic.shape = RigidBodyShape::Auto;
        let sphere = [RigidBody3DInput {
            node: &automatic,
            position: [0.0; 3],
            rotation: [0.0; 3],
            auto_collider_size: Some([2.0; 3]),
            auto_collider_shape: Some(RigidBodyShape::Sphere),
        }];
        let states = initial_states(&sphere);
        assert_eq!(states[0].collider.shape, RigidBodyShape::Sphere);
        assert_eq!(states[0].collider.half_extent, [1.0; 3]);

        let wedge = [RigidBody3DInput {
            node: &automatic,
            position: [0.0; 3],
            rotation: [0.0; 3],
            auto_collider_size: Some([4.0, 1.0, 3.0]),
            auto_collider_shape: Some(RigidBodyShape::ConvexHull),
        }];
        let states = initial_states(&wedge);
        assert_eq!(states[0].collider.shape, RigidBodyShape::ConvexHull);
        assert_eq!(states[0].collider.half_extent, [2.0, 0.5, 1.5]);
    }

    #[test]
    fn output_keeps_direct_yxz_quaternion() {
        let authored = [23.0, -47.0, 11.0];
        let fixed = body("fixed", "model", RigidBodyType::Static);
        let inputs = [RigidBody3DInput {
            node: &fixed,
            position: [0.0; 3],
            rotation: authored,
            auto_collider_size: None,
            auto_collider_shape: None,
        }];
        let output = sample_rigid_bodies_3d(&inputs, 0.0, 1.0 / 120.0, 8, [0.0, -9.81, 0.0]);
        let expected = quat_from_euler_degrees(authored);
        for axis in 0..4 {
            assert!((output[0].orientation[axis] - expected[axis]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn dense_stack_finishes_without_deep_penetration() {
        let mut floor = body("floor", "floor", RigidBodyType::Static);
        floor.size = RigidBodyColliderSize::D3([8.0, 0.5, 8.0]);
        let boxes = (0..5)
            .map(|index| {
                body(
                    &format!("box_{index}"),
                    &format!("box_{index}"),
                    RigidBodyType::Dynamic,
                )
            })
            .collect::<Vec<_>>();
        let mut inputs = Vec::with_capacity(6);
        inputs.push(RigidBody3DInput {
            node: &floor,
            position: [0.0, -0.25, 0.0],
            rotation: [0.0; 3],
            auto_collider_size: None,
            auto_collider_shape: None,
        });
        for (index, node) in boxes.iter().enumerate() {
            inputs.push(RigidBody3DInput {
                node,
                position: [
                    if index % 2 == 0 { -0.03 } else { 0.03 },
                    0.55 + index as f32 * 1.02,
                    0.0,
                ],
                rotation: [0.0, index as f32 * 2.0, 0.0],
                auto_collider_size: None,
                auto_collider_shape: None,
            });
        }
        let mut states = initial_states(&inputs);
        let mut cache = BTreeMap::new();
        advance_states(
            &mut states,
            &mut cache,
            6.0,
            1.0 / 120.0,
            10,
            [0.0, -9.81, 0.0],
        );
        let maximum_penetration = detect_contacts(&states)
            .iter()
            .map(|contact| contact.penetration)
            .fold(0.0_f32, f32::max);
        assert!(
            maximum_penetration < 0.035,
            "maximum penetration={maximum_penetration}, outputs={:?}",
            outputs_from_states(&states)
        );
    }
}
