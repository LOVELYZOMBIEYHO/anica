// =========================================
// =========================================
// crates/motionloom/src/simulation/bridge/scene.rs

use crate::dsl::GraphScript;
use crate::scene::model::{CircleNode, DefsNode, GroupNode, PuppetNode, SceneNode};
use crate::simulation::clock::SimulationClock;
use crate::simulation::error::SimulationError;
use crate::simulation::model::{
    AttractionNode, ClothNode, ColliderNode, HairStrandFieldNode, ParticleEmitterNode,
    SimulationBindingNode, SimulationResourceNode, WindNode,
};

pub fn apply_scene_simulation_at_frame(
    graph: &GraphScript,
    frame: u32,
) -> Result<Option<GraphScript>, SimulationError> {
    let mut bindings = Vec::new();
    for scene in &graph.scenes {
        collect_bindings(&scene.children, &mut bindings);
    }
    if bindings.is_empty() {
        return Ok(None);
    }
    let mut output = graph.clone();
    for scene in &mut output.scenes {
        let clock = SimulationClock {
            fps: graph.fps,
            frame,
            duration_seconds: graph.duration_ms as f32 / 1000.0,
        };
        let mut resources = collect_resources(&scene.children);
        resolve_resource_targets(&mut resources, &scene.children, clock);
        apply_bindings(&mut scene.children, &bindings, &resources, clock)?;
        remove_binding_nodes(&mut scene.children);
    }
    Ok(Some(output))
}

#[derive(Default)]
struct Resources {
    gravity: Vec<crate::simulation::model::GravityNode>,
    wind: Vec<WindNode>,
    attraction: Vec<AttractionNode>,
    colliders: Vec<ColliderNode>,
}

fn collect_resources(nodes: &[SceneNode]) -> Resources {
    let mut resources = Resources::default();
    for node in nodes {
        if let SceneNode::Defs(defs) = node {
            append_defs(defs, &mut resources);
        }
    }
    resources
}

fn append_defs(defs: &DefsNode, out: &mut Resources) {
    for resource in &defs.simulation {
        match resource {
            SimulationResourceNode::Wind(node) => out.wind.push(node.clone()),
            SimulationResourceNode::Attraction(node) => out.attraction.push(node.clone()),
            SimulationResourceNode::Collider(node) => out.colliders.push(node.clone()),
            SimulationResourceNode::Gravity(node) => out.gravity.push(node.clone()),
        }
    }
}

fn resolve_resource_targets(
    resources: &mut Resources,
    nodes: &[SceneNode],
    clock: SimulationClock,
) {
    for attraction in &mut resources.attraction {
        if let Some(position) = attraction
            .target
            .as_deref()
            .and_then(|id| group_position(nodes, id, clock))
        {
            attraction.point[0] += position[0];
            attraction.point[1] += position[1];
        }
    }
    for collider in &mut resources.colliders {
        if let Some(position) = collider
            .target
            .as_deref()
            .and_then(|id| group_position(nodes, id, clock))
        {
            collider.x += position[0];
            collider.y += position[1];
            collider.from[0] += position[0];
            collider.from[1] += position[1];
            collider.to[0] += position[0];
            collider.to[1] += position[1];
        }
    }
}

fn collect_bindings(nodes: &[SceneNode], out: &mut Vec<SimulationBindingNode>) {
    for node in nodes {
        match node {
            SceneNode::Simulation(binding) => out.push(binding.clone()),
            SceneNode::Timeline(node) => collect_bindings(&node.children, out),
            SceneNode::Track(node) => collect_bindings(&node.children, out),
            SceneNode::Sequence(node) => collect_bindings(&node.children, out),
            SceneNode::Layer(node) => collect_bindings(&node.children, out),
            SceneNode::Group(node) => collect_bindings(&node.children, out),
            SceneNode::Part(node) => collect_bindings(&node.children, out),
            _ => {}
        }
    }
}

fn apply_bindings(
    nodes: &mut Vec<SceneNode>,
    bindings: &[SimulationBindingNode],
    resources: &Resources,
    clock: SimulationClock,
) -> Result<(), SimulationError> {
    for binding in bindings {
        match binding {
            SimulationBindingNode::Hinge(binding) => {
                let angle = group_rotation(nodes, &binding.a, clock).unwrap_or(0.0);
                mutate_group(nodes, &binding.a, |group| {
                    set_group_pivot(group, binding.anchor);
                });
                mutate_group(nodes, &binding.b, |group| {
                    set_group_pivot(group, binding.anchor);
                    group.rotation = format!("{:.4}", angle * binding.stiffness);
                });
            }
            SimulationBindingNode::RigidBody2D(binding) => {
                let time = clock.time_seconds();
                mutate_group(nodes, &binding.target, |group| {
                    group.x = format!(
                        "{:.4}",
                        sample_numeric(&group.x, clock) + binding.velocity[0] * time
                    );
                    group.y = format!(
                        "{:.4}",
                        sample_numeric(&group.y, clock)
                            + binding.velocity[1] * time
                            + 90.0 * time * time
                    );
                    group.rotation = format!(
                        "{:.4}",
                        sample_numeric(&group.rotation, clock)
                            + binding.angular_velocity.to_degrees() * time
                    );
                });
            }
            SimulationBindingNode::Cloth(binding) => {
                mutate_group(nodes, &binding.target, |group| {
                    deform_group_curves(group, binding, clock.time_seconds());
                });
            }
            SimulationBindingNode::HairStrandField(binding) => {
                mutate_group(nodes, &binding.target, |group| {
                    deform_hair_curves(group, binding, clock.time_seconds());
                });
            }
            _ => {}
        }
    }

    // Constraints run after body integration so they resolve the current-frame positions.
    for binding in bindings {
        let SimulationBindingNode::DistanceConstraint(binding) = binding else {
            continue;
        };
        let Some(a) = group_position(nodes, &binding.a, clock) else {
            continue;
        };
        let Some(b) = group_position(nodes, &binding.b, clock) else {
            continue;
        };
        let delta = [b[0] - a[0], b[1] - a[1]];
        let length = delta[0].hypot(delta[1]).max(0.0001);
        let target = [
            a[0] + delta[0] / length * binding.distance,
            a[1] + delta[1] / length * binding.distance,
        ];
        mutate_group(nodes, &binding.b, |group| {
            let blend = binding.stiffness.clamp(0.0, 1.0);
            group.x = format!("{:.4}", b[0] + (target[0] - b[0]) * blend);
            group.y = format!("{:.4}", b[1] + (target[1] - b[1]) * blend);
        });
    }

    apply_curve_bindings(nodes, bindings, resources, clock)?;
    Ok(())
}

fn apply_curve_bindings(
    nodes: &mut Vec<SceneNode>,
    bindings: &[SimulationBindingNode],
    resources: &Resources,
    clock: SimulationClock,
) -> Result<(), SimulationError> {
    for node in nodes.iter_mut() {
        match node {
            SceneNode::Puppet(puppet) => {
                let Some(id) = puppet.id.as_deref() else {
                    continue;
                };
                if !puppet.solver.eq_ignore_ascii_case("chain") {
                    continue;
                }
                let Some(binding) = bindings.iter().find_map(|binding| match binding {
                    SimulationBindingNode::SpringChain(binding) if binding.target == id => {
                        Some(binding)
                    }
                    _ => None,
                }) else {
                    continue;
                };
                apply_puppet_chain_binding(puppet, binding, clock);
            }
            SceneNode::Polyline(curve) => {
                let Some(id) = curve.id.as_deref() else {
                    continue;
                };
                let Some(binding) = bindings.iter().find_map(|binding| match binding {
                    SimulationBindingNode::SpringChain(binding) if binding.target == id => {
                        Some(binding)
                    }
                    _ => None,
                }) else {
                    continue;
                };
                let points = parse_points(&curve.points)?;
                let wind = binding
                    .wind
                    .as_deref()
                    .and_then(|id| resources.wind.iter().find(|node| node.id == id));
                let attraction = binding
                    .attraction
                    .as_deref()
                    .and_then(|id| resources.attraction.iter().find(|node| node.id == id));
                let colliders = binding
                    .colliders
                    .iter()
                    .filter_map(|id| resources.colliders.iter().find(|node| node.id == *id))
                    .cloned()
                    .collect::<Vec<_>>();
                let mut resolved_binding = binding.clone();
                if let Some(id) = binding.gravity_ref.as_deref() {
                    resolved_binding.gravity = resources
                        .gravity
                        .iter()
                        .find(|node| node.id == id)
                        .ok_or_else(|| SimulationError::MissingResource { id: id.to_string() })?
                        .vector;
                }
                let effective_clock = cache_clock(bindings, id, clock);
                let state = crate::simulation::runtime::simulate_spring_chain(
                    &points,
                    &resolved_binding,
                    wind,
                    attraction,
                    &colliders,
                    effective_clock,
                );
                curve.points = state
                    .particles
                    .iter()
                    .map(|particle| {
                        format!("{:.4},{:.4}", particle.position[0], particle.position[1])
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
            }
            SceneNode::Timeline(node) => {
                apply_curve_bindings(&mut node.children, bindings, resources, clock)?
            }
            SceneNode::Track(node) => {
                apply_curve_bindings(&mut node.children, bindings, resources, clock)?
            }
            SceneNode::Sequence(node) => {
                apply_curve_bindings(&mut node.children, bindings, resources, clock)?
            }
            SceneNode::Layer(node) => {
                apply_curve_bindings(&mut node.children, bindings, resources, clock)?
            }
            SceneNode::Group(node) => {
                apply_curve_bindings(&mut node.children, bindings, resources, clock)?
            }
            SceneNode::Part(node) => {
                apply_curve_bindings(&mut node.children, bindings, resources, clock)?
            }
            _ => {}
        }
    }
    append_particles(nodes, clock);
    Ok(())
}

/// Resolves a SpringChain directly into chain PuppetPin targets.
///
/// The controller remains the authored goal while the intermediate pins use a
/// deterministic Verlet solve, so random-access frame rendering stays stable.
fn apply_puppet_chain_binding(
    puppet: &mut PuppetNode,
    binding: &crate::simulation::model::SpringChainNode,
    clock: SimulationClock,
) {
    let ordered_indices = ordered_puppet_pin_indices(puppet);
    if ordered_indices.len() < 2 {
        return;
    }
    let source = ordered_indices
        .iter()
        .filter_map(|index| puppet.children.get(*index))
        .filter_map(|node| match node {
            SceneNode::Pin(pin) => Some([
                pin.x
                    .as_deref()
                    .map(|value| sample_numeric(value, clock))
                    .unwrap_or(0.0),
                pin.y
                    .as_deref()
                    .map(|value| sample_numeric(value, clock))
                    .unwrap_or(0.0),
            ]),
            _ => None,
        })
        .collect::<Vec<_>>();
    if source.len() != ordered_indices.len() {
        return;
    }

    let control_index = *ordered_indices.last().unwrap_or(&ordered_indices[0]);
    let control = match puppet.children.get(control_index) {
        Some(SceneNode::Pin(pin)) => pin.clone(),
        _ => return,
    };
    let source_tip = *source.last().unwrap_or(&[0.0, 0.0]);
    let mut state = crate::simulation::bodies::dynamic_curve::build_dynamic_curve(&source, "start");
    let drag = sample_numeric(&puppet.drag, clock).clamp(0.0, 1.0);
    let overlap = sample_numeric(&puppet.overlap, clock).clamp(0.0, 0.95);
    let stiffness =
        (binding.stiffness * sample_numeric(&puppet.stiffness, clock)).clamp(0.001, 1.0);
    let damping =
        ((binding.damping + sample_numeric(&puppet.damping, clock)) * 0.5).clamp(0.0, 0.999);
    let dt = clock.fixed_dt();
    for frame in 0..=clock.frame {
        let frame_clock = SimulationClock {
            fps: clock.fps,
            frame,
            duration_seconds: clock.duration_seconds,
        };
        let desired = [
            control
                .target_x
                .as_deref()
                .map(|value| sample_numeric(value, frame_clock))
                .unwrap_or(source_tip[0]),
            control
                .target_y
                .as_deref()
                .map(|value| sample_numeric(value, frame_clock))
                .unwrap_or(source_tip[1]),
        ];
        if let Some(tip) = state.particles.last_mut() {
            let response = ((1.0 - overlap) * (0.35 + drag * 0.65)).clamp(0.02, 1.0);
            tip.previous = tip.position;
            tip.position[0] += (desired[0] - tip.position[0]) * response;
            tip.position[1] += (desired[1] - tip.position[1]) * response;
        }
        crate::simulation::solvers::verlet::step(
            &mut state,
            |_| binding.gravity,
            dt,
            damping,
            stiffness,
            &[],
            0.0,
        );
    }

    for (chain_index, child_index) in ordered_indices.into_iter().enumerate() {
        let Some(SceneNode::Pin(pin)) = puppet.children.get_mut(child_index) else {
            continue;
        };
        let position = state.particles[chain_index].position;
        pin.target_x = Some(format!("{:.4}", position[0]));
        pin.target_y = Some(format!("{:.4}", position[1]));
    }
}

fn ordered_puppet_pin_indices(puppet: &PuppetNode) -> Vec<usize> {
    let pins = puppet
        .children
        .iter()
        .enumerate()
        .filter_map(|(index, node)| match node {
            SceneNode::Pin(pin) => Some((index, pin)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let Some((root_index, root)) = pins.iter().find(|(_, pin)| {
        pin.role
            .as_deref()
            .is_some_and(|role| matches!(role.to_ascii_lowercase().as_str(), "anchor" | "root"))
            || pin.parent.is_none()
    }) else {
        return Vec::new();
    };
    let Some(mut current_id) = root.id.clone() else {
        return Vec::new();
    };
    let mut ordered = vec![*root_index];
    while let Some((index, pin)) = pins
        .iter()
        .find(|(_, pin)| pin.parent.as_deref() == Some(current_id.as_str()))
    {
        let Some(id) = pin.id.clone() else {
            break;
        };
        if ordered.contains(index) {
            return Vec::new();
        }
        ordered.push(*index);
        current_id = id;
    }
    if ordered.len() == pins.len() {
        ordered
    } else {
        Vec::new()
    }
}

fn cache_clock(
    bindings: &[SimulationBindingNode],
    target: &str,
    clock: SimulationClock,
) -> SimulationClock {
    let Some(cache) = bindings.iter().find_map(|binding| match binding {
        SimulationBindingNode::CacheBake(cache) if cache.target == target => Some(cache),
        _ => None,
    }) else {
        return clock;
    };
    SimulationClock {
        frame: clock
            .frame
            .clamp(cache.from_frame, cache.to_frame.max(cache.from_frame)),
        ..clock
    }
}

fn mutate_group(nodes: &mut [SceneNode], id: &str, mut apply: impl FnMut(&mut GroupNode)) -> bool {
    mutate_group_inner(nodes, id, &mut apply)
}

fn mutate_group_inner(
    nodes: &mut [SceneNode],
    id: &str,
    apply: &mut dyn FnMut(&mut GroupNode),
) -> bool {
    for node in nodes {
        match node {
            SceneNode::Group(group) => {
                if group.id.as_deref() == Some(id) {
                    apply(group);
                    return true;
                }
                if mutate_group_inner(&mut group.children, id, apply) {
                    return true;
                }
            }
            SceneNode::Timeline(node) => {
                if mutate_group_inner(&mut node.children, id, apply) {
                    return true;
                }
            }
            SceneNode::Track(node) => {
                if mutate_group_inner(&mut node.children, id, apply) {
                    return true;
                }
            }
            SceneNode::Sequence(node) => {
                if mutate_group_inner(&mut node.children, id, apply) {
                    return true;
                }
            }
            SceneNode::Layer(node) => {
                if mutate_group_inner(&mut node.children, id, apply) {
                    return true;
                }
            }
            SceneNode::Part(node) => {
                if mutate_group_inner(&mut node.children, id, apply) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn set_group_pivot(group: &mut GroupNode, anchor: [f32; 2]) {
    group.transform_origin_x = format!("{:.4}", anchor[0]);
    group.transform_origin_y = format!("{:.4}", anchor[1]);
}

fn sample_numeric(value: &str, clock: SimulationClock) -> f32 {
    crate::process::runtime::eval_time_expr(value, clock.time_norm(), clock.time_seconds())
        .unwrap_or(0.0)
}

fn group_position(nodes: &[SceneNode], id: &str, clock: SimulationClock) -> Option<[f32; 2]> {
    for node in nodes {
        match node {
            SceneNode::Group(group) if group.id.as_deref() == Some(id) => {
                return Some([
                    sample_numeric(&group.x, clock),
                    sample_numeric(&group.y, clock),
                ]);
            }
            SceneNode::Timeline(node) => {
                if let Some(value) = group_position(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Track(node) => {
                if let Some(value) = group_position(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Sequence(node) => {
                if let Some(value) = group_position(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Layer(node) => {
                if let Some(value) = group_position(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Group(node) => {
                if let Some(value) = group_position(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Part(node) => {
                if let Some(value) = group_position(&node.children, id, clock) {
                    return Some(value);
                }
            }
            _ => {}
        }
    }
    None
}

fn group_rotation(nodes: &[SceneNode], id: &str, clock: SimulationClock) -> Option<f32> {
    for node in nodes {
        match node {
            SceneNode::Group(group) if group.id.as_deref() == Some(id) => {
                return Some(sample_numeric(&group.rotation, clock));
            }
            SceneNode::Timeline(node) => {
                if let Some(value) = group_rotation(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Track(node) => {
                if let Some(value) = group_rotation(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Sequence(node) => {
                if let Some(value) = group_rotation(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Layer(node) => {
                if let Some(value) = group_rotation(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Group(node) => {
                if let Some(value) = group_rotation(&node.children, id, clock) {
                    return Some(value);
                }
            }
            SceneNode::Part(node) => {
                if let Some(value) = group_rotation(&node.children, id, clock) {
                    return Some(value);
                }
            }
            _ => {}
        }
    }
    None
}

fn deform_group_curves(group: &mut GroupNode, binding: &ClothNode, time: f32) {
    deform_curves(&mut group.children, |point, index| {
        let phase = time * binding.frequency + point[0] * 0.025 + index as f32 * 0.18;
        [
            point[0] + phase.sin() * binding.amplitude * 0.18,
            point[1] + phase.cos() * binding.amplitude,
        ]
    });
}

fn deform_hair_curves(group: &mut GroupNode, binding: &HairStrandFieldNode, time: f32) {
    deform_curves(&mut group.children, |point, index| {
        let weight = index as f32 / binding.segments.max(1) as f32;
        [
            point[0] + (time * 2.4 + index as f32 * 0.35).sin() * 26.0 * weight,
            point[1],
        ]
    });
}

fn deform_curves(nodes: &mut [SceneNode], mut map: impl FnMut([f32; 2], usize) -> [f32; 2]) {
    deform_curves_inner(nodes, &mut map);
}

fn deform_curves_inner(nodes: &mut [SceneNode], map: &mut dyn FnMut([f32; 2], usize) -> [f32; 2]) {
    for node in nodes {
        match node {
            SceneNode::Polyline(curve) => {
                if let Ok(points) = parse_points(&curve.points) {
                    curve.points = points
                        .into_iter()
                        .enumerate()
                        .map(|(index, point)| {
                            let point = map(point, index);
                            format!("{:.4},{:.4}", point[0], point[1])
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                }
            }
            SceneNode::Group(group) => deform_curves_inner(&mut group.children, map),
            _ => {}
        }
    }
}

fn append_particles(nodes: &mut Vec<SceneNode>, clock: SimulationClock) {
    let emitters = nodes
        .iter()
        .filter_map(|node| match node {
            SceneNode::Simulation(SimulationBindingNode::ParticleEmitter(emitter)) => {
                Some(emitter.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    for mut emitter in emitters {
        if let Some(position) = emitter
            .target
            .as_deref()
            .and_then(|id| group_position(nodes, id, clock))
        {
            emitter.x += position[0];
            emitter.y += position[1];
        }
        nodes.extend(
            particles_for_frame(&emitter, clock)
                .into_iter()
                .map(SceneNode::Circle),
        );
    }
}

fn particles_for_frame(emitter: &ParticleEmitterNode, clock: SimulationClock) -> Vec<CircleNode> {
    let time = clock.time_seconds();
    let first = ((time - emitter.lifetime).max(0.0) * emitter.rate).floor() as u32;
    let last = (time * emitter.rate).floor() as u32;
    (first..last)
        .map(|index| {
            let birth = index as f32 / emitter.rate.max(0.001);
            let age = (time - birth).max(0.0);
            let jitter = ((index as f32 * 12.9898).sin() * 43_758.547).fract() - 0.5;
            let vx = emitter.velocity[0] + jitter * 90.0;
            let vy = emitter.velocity[1] + jitter.abs() * 24.0;
            CircleNode {
                id: Some(format!("{}_particle_{index}", emitter.id)),
                x: format!(
                    "{:.4}",
                    emitter.x + vx * age + emitter.gravity[0] * age * age * 0.5
                ),
                y: format!(
                    "{:.4}",
                    emitter.y + vy * age + emitter.gravity[1] * age * age * 0.5
                ),
                radius: format!(
                    "{:.4}",
                    emitter.radius * (1.0 - age / emitter.lifetime).max(0.15)
                ),
                color: emitter.color.clone(),
                stroke: None,
                stroke_width: "0".into(),
                opacity: format!("{:.4}", (1.0 - age / emitter.lifetime).clamp(0.0, 1.0)),
                rotation: "0".into(),
                scale: "1".into(),
                scale_x: "1".into(),
                scale_y: "1".into(),
                skew_x: "0".into(),
                skew_y: "0".into(),
                transform_origin_x: "0".into(),
                transform_origin_y: "0".into(),
                blend: "normal".into(),
                texture: None,
                texture_opacity: "1".into(),
                texture_scale: "1".into(),
                texture_mask: "0".into(),
            }
        })
        .collect()
}

fn remove_binding_nodes(nodes: &mut Vec<SceneNode>) {
    nodes.retain(|node| !matches!(node, SceneNode::Simulation(_)));
    for node in nodes {
        match node {
            SceneNode::Timeline(node) => remove_binding_nodes(&mut node.children),
            SceneNode::Track(node) => remove_binding_nodes(&mut node.children),
            SceneNode::Sequence(node) => remove_binding_nodes(&mut node.children),
            SceneNode::Layer(node) => remove_binding_nodes(&mut node.children),
            SceneNode::Group(node) => remove_binding_nodes(&mut node.children),
            SceneNode::Part(node) => remove_binding_nodes(&mut node.children),
            _ => {}
        }
    }
}

fn parse_points(raw: &str) -> Result<Vec<[f32; 2]>, SimulationError> {
    let values = raw
        .replace(',', " ")
        .split_whitespace()
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| SimulationError::InvalidPoints {
            value: raw.to_string(),
        })?;
    if values.len() < 4 || values.len() % 2 != 0 {
        return Err(SimulationError::InvalidPoints {
            value: raw.to_string(),
        });
    }
    Ok(values
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::dsl::parse_graph_script;

    fn assert_simulation_changes(script: &str, later_frame: u32) {
        let graph = parse_graph_script(script).expect("parse simulation graph");
        let first = super::apply_scene_simulation_at_frame(&graph, 0)
            .expect("simulate first frame")
            .expect("simulation graph");
        let later = super::apply_scene_simulation_at_frame(&graph, later_frame)
            .expect("simulate later frame")
            .expect("simulation graph");
        assert_ne!(first.scenes, later.scenes);
    }

    #[test]
    fn deterministic_frame_changes_curve_and_removes_binding() {
        assert_simulation_changes(
            r##"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Scene id="s">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="1s">
          <Layer>
            <Curve id="strand" points="20,20 20,80 20,140" stroke="#fff" />
            <SpringChain target="strand" pin="start" segments="4" gravity={[200,400]} />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="s" />
</Graph>
"##,
            20,
        );
    }

    #[test]
    fn hinge_binding_changes_group_transforms() {
        assert_simulation_changes(
            r##"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Scene id="s">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="1s">
          <Layer>
            <Group id="a" rotation={curve("0:-20:linear, 1:35:ease_in_out")}>
              <Rect x="20" y="80" width="90" height="20" color="#fff" />
            </Group>
            <Group id="b">
              <Rect x="110" y="80" width="90" height="20" color="#fff" />
            </Group>
            <Hinge a="a" b="b" anchor={[110,90]} stiffness="0.9" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="s" />
</Graph>
"##,
            12,
        );
    }

    #[test]
    fn rigid_body_binding_changes_target_transform() {
        assert_simulation_changes(
            r##"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Scene id="s">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="1s">
          <Layer>
            <Group id="body">
              <Circle x="80" y="80" radius="20" color="#fff" />
            </Group>
            <RigidBody2D id="physics" target="body" mass="1" velocity={[80,-20]} angularVelocity="0.5" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="s" />
</Graph>
"##,
            12,
        );
    }

    #[test]
    fn distance_constraint_resolves_after_body_motion() {
        assert_simulation_changes(
            r##"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Scene id="s">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="1s">
          <Layer>
            <Group id="a" x="60" y="120">
              <Circle x="0" y="0" radius="12" color="#fff" />
            </Group>
            <Group id="b" x="180" y="120">
              <Circle x="0" y="0" radius="12" color="#fff" />
            </Group>
            <RigidBody2D id="physics" target="b" mass="1" velocity={[0,-80]} angularVelocity="0" />
            <DistanceConstraint a="a" b="b" distance="120" stiffness="1" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="s" />
</Graph>
"##,
            12,
        );
    }

    #[test]
    fn particle_emitter_appends_live_particles() {
        assert_simulation_changes(
            r##"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Scene id="s">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="1s">
          <Layer>
            <ParticleEmitter id="sparks" x="160" y="180" rate="30" lifetime="1" velocity={[0,-100]} gravity={[0,80]} radius="4" color="#fff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="s" />
</Graph>
"##,
            12,
        );
    }

    #[test]
    fn cloth_binding_deforms_target_curves() {
        assert_simulation_changes(
            r##"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Scene id="s">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="1s">
          <Layer>
            <Group id="cloth">
              <Curve id="row" points="40,60 100,60 160,60 220,60" stroke="#fff" />
            </Group>
            <Cloth id="cape" target="cloth" columns="4" rows="1" stiffness="0.8" damping="0.2" amplitude="24" frequency="2" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="s" />
</Graph>
"##,
            12,
        );
    }
}
