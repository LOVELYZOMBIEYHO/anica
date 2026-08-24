// =========================================
// =========================================
// crates/motionloom/src/simulation/dsl.rs

use crate::dsl::{attr_value, required_attr_value, strip_wrappers};
use crate::error::GraphParseError;
use crate::simulation::model::*;

pub(crate) fn parse_resource(
    tag: &str,
    line: usize,
) -> Result<SimulationResourceNode, GraphParseError> {
    let name = tag
        .trim_start()
        .trim_start_matches('<')
        .split_whitespace()
        .next()
        .unwrap_or("");
    match name {
        "Gravity" => Ok(SimulationResourceNode::Gravity(GravityNode {
            id: required_string(tag, "id", line)?,
            vector: vec2(tag, "vector", [0.0, 980.0], line)?,
        })),
        "Wind" => Ok(SimulationResourceNode::Wind(WindNode {
            id: required_string(tag, "id", line)?,
            direction: vec2(tag, "direction", [1.0, 0.0], line)?,
            strength: number(tag, "strength", 0.0, line)?,
            turbulence: number(tag, "turbulence", 0.0, line)?,
            noise_scale: number(tag, "noiseScale", 1.0, line)?,
        })),
        "Attraction" => Ok(SimulationResourceNode::Attraction(AttractionNode {
            id: required_string(tag, "id", line)?,
            target: optional_string(tag, "target"),
            point: vec2(tag, "point", [0.0, 0.0], line)?,
            strength: number(tag, "strength", 0.0, line)?,
            radius: number(tag, "radius", f32::MAX, line)?,
        })),
        "Collider" => Ok(SimulationResourceNode::Collider(ColliderNode {
            id: required_string(tag, "id", line)?,
            target: optional_string(tag, "target"),
            shape: collider_shape(tag, line)?,
            x: number(tag, "x", 0.0, line)?,
            y: number(tag, "y", 0.0, line)?,
            radius: number(tag, "radius", 0.0, line)?,
            radius_x: number(tag, "radiusX", 0.0, line)?,
            radius_y: number(tag, "radiusY", 0.0, line)?,
            from: vec2(tag, "from", [0.0, 0.0], line)?,
            to: vec2(tag, "to", [0.0, 0.0], line)?,
        })),
        _ => Err(parse_error(
            line,
            format!("unknown simulation resource <{name}>"),
        )),
    }
}

pub(crate) fn parse_binding(
    tag: &str,
    line: usize,
) -> Result<SimulationBindingNode, GraphParseError> {
    let name = tag
        .trim_start()
        .trim_start_matches('<')
        .split_whitespace()
        .next()
        .unwrap_or("");
    match name {
        "SpringChain" => {
            let (gravity, gravity_ref) = vec2_or_ref(tag, "gravity", [0.0, 520.0], line)?;
            Ok(SimulationBindingNode::SpringChain(SpringChainNode {
                id: optional_string(tag, "id"),
                target: required_string(tag, "target", line)?,
                pin: string(tag, "pin", "start"),
                segments: usize_number(tag, "segments", 16, line)?,
                stiffness: number(tag, "stiffness", 0.75, line)?,
                damping: number(tag, "damping", 0.18, line)?,
                gravity,
                gravity_ref,
                wind: optional_string(tag, "wind"),
                attraction: optional_string(tag, "attraction"),
                colliders: string_list(tag, "colliders"),
                collision_radius: number(tag, "collisionRadius", 0.0, line)?,
            }))
        }
        "DynamicCurve" => Ok(SimulationBindingNode::DynamicCurve(DynamicCurveNode {
            id: optional_string(tag, "id"),
            target: required_string(tag, "target", line)?,
            simulation: string(tag, "simulation", "spring"),
        })),
        "DistanceConstraint" => Ok(SimulationBindingNode::DistanceConstraint(
            DistanceConstraintNode {
                id: optional_string(tag, "id"),
                a: required_string(tag, "a", line)?,
                b: required_string(tag, "b", line)?,
                distance: number(tag, "distance", 0.0, line)?,
                stiffness: number(tag, "stiffness", 1.0, line)?,
            },
        )),
        "Hinge" => Ok(SimulationBindingNode::Hinge(HingeNode {
            id: optional_string(tag, "id"),
            a: required_string(tag, "a", line)?,
            b: required_string(tag, "b", line)?,
            anchor: vec2(tag, "anchor", [0.0, 0.0], line)?,
            stiffness: number(tag, "stiffness", 1.0, line)?,
        })),
        "RigidBody" => Ok(SimulationBindingNode::RigidBody(parse_rigid_body(
            tag, line,
        )?)),
        "ParticleEmitter" => Ok(SimulationBindingNode::ParticleEmitter(
            ParticleEmitterNode {
                id: required_string(tag, "id", line)?,
                target: optional_string(tag, "target"),
                x: number(tag, "x", 0.0, line)?,
                y: number(tag, "y", 0.0, line)?,
                rate: number(tag, "rate", 24.0, line)?,
                lifetime: number(tag, "lifetime", 2.0, line)?,
                velocity: vec2(tag, "velocity", [0.0, -120.0], line)?,
                gravity: vec2(tag, "gravity", [0.0, 300.0], line)?,
                radius: number(tag, "radius", 5.0, line)?,
                color: string(tag, "color", "#D8FF2F"),
            },
        )),
        "Cloth" => Ok(SimulationBindingNode::Cloth(ClothNode {
            id: required_string(tag, "id", line)?,
            target: required_string(tag, "target", line)?,
            columns: usize_number(tag, "columns", 12, line)?,
            rows: usize_number(tag, "rows", 8, line)?,
            stiffness: number(tag, "stiffness", 0.75, line)?,
            damping: number(tag, "damping", 0.2, line)?,
            amplitude: number(tag, "amplitude", 28.0, line)?,
            frequency: number(tag, "frequency", 1.8, line)?,
        })),
        "HairStrandField" => Ok(SimulationBindingNode::HairStrandField(
            HairStrandFieldNode {
                id: required_string(tag, "id", line)?,
                target: required_string(tag, "target", line)?,
                strands: usize_number(tag, "strands", 32, line)?,
                segments: usize_number(tag, "segments", 12, line)?,
                stiffness: number(tag, "stiffness", 0.72, line)?,
                damping: number(tag, "damping", 0.2, line)?,
            },
        )),
        "CacheBake" => Ok(SimulationBindingNode::CacheBake(CacheBakeNode {
            id: required_string(tag, "id", line)?,
            target: required_string(tag, "target", line)?,
            from_frame: usize_number(tag, "fromFrame", 0, line)? as u32,
            to_frame: usize_number(tag, "toFrame", 0, line)? as u32,
        })),
        _ => Err(parse_error(
            line,
            format!("unknown simulation binding <{name}>"),
        )),
    }
}

/// Parse the one public rigid-body tag and reject dimensionally ambiguous data.
pub(crate) fn parse_rigid_body(tag: &str, line: usize) -> Result<RigidBodyNode, GraphParseError> {
    let dimension = match required_string(tag, "dimension", line)?.as_str() {
        "2d" => RigidBodyDimension::D2,
        "3d" => RigidBodyDimension::D3,
        value => {
            return Err(parse_error(
                line,
                format!("RigidBody.dimension must be 2d or 3d, got '{value}'"),
            ));
        }
    };
    let body_type = match required_string(tag, "type", line)?.as_str() {
        "static" => RigidBodyType::Static,
        "dynamic" => RigidBodyType::Dynamic,
        "kinematic" => RigidBodyType::Kinematic,
        value => {
            return Err(parse_error(
                line,
                format!("RigidBody.type must be static, dynamic, or kinematic, got '{value}'"),
            ));
        }
    };
    let shape = rigid_body_shape(tag, dimension, body_type, line)?;
    let mass = number(tag, "mass", 1.0, line)?;
    if body_type == RigidBodyType::Dynamic && (!mass.is_finite() || mass <= 0.0) {
        return Err(parse_error(
            line,
            "RigidBody.mass must be greater than zero for a dynamic body".to_string(),
        ));
    }
    let (size, velocity, angular_velocity, gravity) = match dimension {
        RigidBodyDimension::D2 => (
            RigidBodyColliderSize::D2(positive_vec2(tag, "size", [100.0, 100.0], line)?),
            RigidBodyLinearVelocity::D2(vec2(tag, "velocity", [0.0, 0.0], line)?),
            RigidBodyAngularVelocity::D2(number(tag, "angularVelocity", 0.0, line)?),
            RigidBodyLinearVelocity::D2(vec2(tag, "gravity", [0.0, 180.0], line)?),
        ),
        RigidBodyDimension::D3 => {
            if optional_string(tag, "gravity").is_some() {
                return Err(parse_error(
                    line,
                    "RigidBody dimension=\"3d\" uses the containing <Physics gravity={...}>; remove its body-local gravity attribute"
                        .to_string(),
                ));
            }
            (
                RigidBodyColliderSize::D3(positive_vec3(tag, "size", [1.0, 1.0, 1.0], line)?),
                RigidBodyLinearVelocity::D3(vec3(tag, "velocity", [0.0, 0.0, 0.0], line)?),
                RigidBodyAngularVelocity::D3(vec3(tag, "angularVelocity", [0.0, 0.0, 0.0], line)?),
                RigidBodyLinearVelocity::D3([0.0, -9.81, 0.0]),
            )
        }
    };
    if body_type == RigidBodyType::Static
        && (!linear_velocity_is_zero(velocity) || !angular_velocity_is_zero(angular_velocity))
    {
        return Err(parse_error(
            line,
            "A static RigidBody cannot declare non-zero velocity or angularVelocity".to_string(),
        ));
    }
    let friction = unit_interval(tag, "friction", 0.5, line)?;
    let rolling_friction = unit_interval(tag, "rollingFriction", 0.04, line)?;
    let restitution = unit_interval(tag, "restitution", 0.0, line)?;
    let restitution_threshold = non_negative_number(tag, "restitutionThreshold", 0.5, line)?;
    let linear_damping = non_negative_number(tag, "linearDamping", 0.0, line)?;
    let angular_damping = non_negative_number(tag, "angularDamping", 0.0, line)?;
    let sleep_linear_threshold = non_negative_number(tag, "sleepLinearThreshold", 0.015, line)?;
    let sleep_angular_threshold = non_negative_number(tag, "sleepAngularThreshold", 0.025, line)?;
    let sleep_time = non_negative_number(tag, "sleepTime", 0.5, line)?;
    let radius = non_negative_number(tag, "radius", 0.5, line)?;
    let height = non_negative_number(tag, "height", 1.0, line)?;
    Ok(RigidBodyNode {
        id: required_string(tag, "id", line)?,
        target: required_string(tag, "target", line)?,
        dimension,
        body_type,
        shape,
        size,
        radius,
        height,
        mass,
        velocity,
        angular_velocity,
        gravity,
        friction,
        rolling_friction,
        restitution,
        restitution_threshold,
        linear_damping,
        angular_damping,
        continuous_collision: boolean(tag, "continuousCollision", false, line)?,
        sleep: boolean(tag, "sleep", true, line)?,
        sleep_linear_threshold,
        sleep_angular_threshold,
        sleep_time,
    })
}

fn positive_vec2(
    tag: &str,
    key: &'static str,
    default: [f32; 2],
    line: usize,
) -> Result<[f32; 2], GraphParseError> {
    let value = vec2(tag, key, default, line)?;
    if value.iter().any(|axis| !axis.is_finite() || *axis <= 0.0) {
        return Err(parse_error(
            line,
            format!("{key} values must be greater than zero"),
        ));
    }
    Ok(value)
}

fn positive_vec3(
    tag: &str,
    key: &'static str,
    default: [f32; 3],
    line: usize,
) -> Result<[f32; 3], GraphParseError> {
    let value = vec3(tag, key, default, line)?;
    if value.iter().any(|axis| !axis.is_finite() || *axis <= 0.0) {
        return Err(parse_error(
            line,
            format!("{key} values must be greater than zero"),
        ));
    }
    Ok(value)
}

fn required_string(tag: &str, key: &str, line: usize) -> Result<String, GraphParseError> {
    Ok(strip_wrappers(&required_attr_value(tag, key, line)?).to_string())
}
fn optional_string(tag: &str, key: &str) -> Option<String> {
    attr_value(tag, key).map(|value| strip_wrappers(&value).to_string())
}
fn string(tag: &str, key: &str, default: &str) -> String {
    optional_string(tag, key).unwrap_or_else(|| default.to_string())
}
fn number(tag: &str, key: &'static str, default: f32, line: usize) -> Result<f32, GraphParseError> {
    optional_string(tag, key).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|_| parse_error(line, format!("{key} must be numeric")))
    })
}
fn usize_number(
    tag: &str,
    key: &'static str,
    default: usize,
    line: usize,
) -> Result<usize, GraphParseError> {
    optional_string(tag, key).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|_| parse_error(line, format!("{key} must be an unsigned integer")))
    })
}
fn vec2(
    tag: &str,
    key: &'static str,
    default: [f32; 2],
    line: usize,
) -> Result<[f32; 2], GraphParseError> {
    let Some(raw) = optional_string(tag, key) else {
        return Ok(default);
    };
    let values = raw
        .trim_matches(|c| matches!(c, '[' | ']' | '{' | '}'))
        .split(',')
        .map(str::trim)
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| parse_error(line, format!("{key} must contain two numbers")))?;
    if values.len() != 2 {
        return Err(parse_error(line, format!("{key} must contain two numbers")));
    }
    Ok([values[0], values[1]])
}
fn vec3(
    tag: &str,
    key: &'static str,
    default: [f32; 3],
    line: usize,
) -> Result<[f32; 3], GraphParseError> {
    let Some(raw) = optional_string(tag, key) else {
        return Ok(default);
    };
    let values = raw
        .trim_matches(|c| matches!(c, '[' | ']' | '{' | '}'))
        .split(',')
        .map(str::trim)
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| parse_error(line, format!("{key} must contain three numbers")))?;
    if values.len() != 3 {
        return Err(parse_error(
            line,
            format!("{key} must contain three numbers"),
        ));
    }
    Ok([values[0], values[1], values[2]])
}
fn boolean(
    tag: &str,
    key: &'static str,
    default: bool,
    line: usize,
) -> Result<bool, GraphParseError> {
    optional_string(tag, key).map_or(Ok(default), |value| match value.as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(parse_error(
            line,
            format!("RigidBody.{key} must be boolean"),
        )),
    })
}
fn unit_interval(
    tag: &str,
    key: &'static str,
    default: f32,
    line: usize,
) -> Result<f32, GraphParseError> {
    let value = number(tag, key, default, line)?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(parse_error(
            line,
            format!("RigidBody.{key} must be between 0 and 1"),
        ));
    }
    Ok(value)
}
fn non_negative_number(
    tag: &str,
    key: &'static str,
    default: f32,
    line: usize,
) -> Result<f32, GraphParseError> {
    let value = number(tag, key, default, line)?;
    if !value.is_finite() || value < 0.0 {
        return Err(parse_error(
            line,
            format!("RigidBody.{key} must be non-negative"),
        ));
    }
    Ok(value)
}
fn linear_velocity_is_zero(value: RigidBodyLinearVelocity) -> bool {
    match value {
        RigidBodyLinearVelocity::D2(value) => value == [0.0, 0.0],
        RigidBodyLinearVelocity::D3(value) => value == [0.0, 0.0, 0.0],
    }
}
fn angular_velocity_is_zero(value: RigidBodyAngularVelocity) -> bool {
    match value {
        RigidBodyAngularVelocity::D2(value) => value == 0.0,
        RigidBodyAngularVelocity::D3(value) => value == [0.0, 0.0, 0.0],
    }
}
fn rigid_body_shape(
    tag: &str,
    dimension: RigidBodyDimension,
    body_type: RigidBodyType,
    line: usize,
) -> Result<RigidBodyShape, GraphParseError> {
    let raw = string(tag, "shape", "auto");
    let shape = match raw.as_str() {
        "auto" => RigidBodyShape::Auto,
        "box" => RigidBodyShape::Box,
        "circle" => RigidBodyShape::Circle,
        "sphere" => RigidBodyShape::Sphere,
        "capsule" => RigidBodyShape::Capsule,
        "cylinder" => RigidBodyShape::Cylinder,
        "convexHull" | "convex_hull" => RigidBodyShape::ConvexHull,
        "mesh" => RigidBodyShape::Mesh,
        _ => {
            return Err(parse_error(
                line,
                format!("unsupported RigidBody shape '{raw}'"),
            ));
        }
    };
    let valid = match dimension {
        RigidBodyDimension::D2 => matches!(
            shape,
            RigidBodyShape::Auto
                | RigidBodyShape::Box
                | RigidBodyShape::Circle
                | RigidBodyShape::Capsule
                | RigidBodyShape::ConvexHull
        ),
        RigidBodyDimension::D3 => !matches!(shape, RigidBodyShape::Circle),
    };
    if !valid {
        return Err(parse_error(
            line,
            format!("RigidBody shape '{raw}' is invalid for dimension"),
        ));
    }
    if shape == RigidBodyShape::Mesh && body_type == RigidBodyType::Dynamic {
        return Err(parse_error(
            line,
            "A dynamic RigidBody cannot use a concave mesh; use convexHull".to_string(),
        ));
    }
    Ok(shape)
}
fn vec2_or_ref(
    tag: &str,
    key: &'static str,
    default: [f32; 2],
    line: usize,
) -> Result<([f32; 2], Option<String>), GraphParseError> {
    let Some(raw) = optional_string(tag, key) else {
        return Ok((default, None));
    };
    if !raw.contains(',') {
        return Ok((default, Some(raw)));
    }
    Ok((vec2(tag, key, default, line)?, None))
}
fn string_list(tag: &str, key: &str) -> Vec<String> {
    optional_string(tag, key)
        .map(|raw| {
            raw.trim_matches(|c| matches!(c, '[' | ']' | '{' | '}'))
                .split(',')
                .map(|item| item.trim().trim_matches('"').to_string())
                .filter(|item| !item.is_empty())
                .collect()
        })
        .unwrap_or_default()
}
fn collider_shape(tag: &str, line: usize) -> Result<ColliderShape, GraphParseError> {
    match string(tag, "shape", "circle").as_str() {
        "circle" => Ok(ColliderShape::Circle),
        "ellipse" => Ok(ColliderShape::Ellipse),
        "capsule" => Ok(ColliderShape::Capsule),
        "box" => Ok(ColliderShape::Box),
        "convexHull" | "convex_hull" => Ok(ColliderShape::ConvexHull),
        value => Err(parse_error(
            line,
            format!("unsupported collider shape '{value}'"),
        )),
    }
}
fn parse_error(line: usize, message: String) -> GraphParseError {
    GraphParseError { line, message }
}

#[cfg(test)]
mod tests {
    use super::parse_rigid_body;
    use crate::dsl::parse_graph_script;
    use crate::scene::model::SceneNode;
    use crate::simulation::model::{SimulationBindingNode, SimulationResourceNode};

    #[test]
    fn parses_resources_and_spring_chain_without_wrapper() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Scene id="hair">
    <Defs>
      <Wind id="wind" direction={[1,0]} strength="18" />
      <Collider id="head" shape="circle" x="160" y="100" radius="40" />
    </Defs>
    <Timeline>
      <Track>
        <Sequence from="0s" duration="1s">
          <Layer>
      <Curve id="strand" points="100,40 110,90 120,150" stroke="#fff" />
      <SpringChain target="strand" pin="start" segments="8" wind="wind" colliders={["head"]} />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="hair" />
</Graph>
"##,
        )
        .expect("simulation DSL should parse");
        let defs = graph.scenes[0]
            .children
            .iter()
            .find_map(|node| match node {
                SceneNode::Defs(defs) => Some(defs),
                _ => None,
            })
            .expect("defs");
        assert!(matches!(
            defs.simulation[0],
            SimulationResourceNode::Wind(_)
        ));
        fn has_binding(nodes: &[SceneNode]) -> bool {
            nodes.iter().any(|node| match node {
                SceneNode::Simulation(SimulationBindingNode::SpringChain(_)) => true,
                SceneNode::Timeline(node) => has_binding(&node.children),
                SceneNode::Track(node) => has_binding(&node.children),
                SceneNode::Sequence(node) => has_binding(&node.children),
                SceneNode::Layer(node) => has_binding(&node.children),
                _ => false,
            })
        }
        assert!(has_binding(&graph.scenes[0].children));
    }

    #[test]
    fn rigid_body_requires_an_explicit_dimension_and_type() {
        let missing_dimension = parse_rigid_body(
            r#"<RigidBody id="body" target="shape" type="dynamic" />"#,
            8,
        )
        .expect_err("dimension is required");
        assert!(missing_dimension.message.contains("dimension"));

        let missing_type = parse_rigid_body(
            r#"<RigidBody id="body" target="shape" dimension="2d" />"#,
            9,
        )
        .expect_err("type is required");
        assert!(missing_type.message.contains("type"));
    }

    #[test]
    fn rigid_body_rejects_dimensionally_invalid_shapes_and_vectors() {
        let vector_error = parse_rigid_body(
            r#"<RigidBody id="body" target="shape" dimension="3d" type="dynamic" velocity={[1,2]} />"#,
            12,
        )
        .expect_err("3D velocity must have three values");
        assert!(vector_error.message.contains("three numbers"));

        let shape_error = parse_rigid_body(
            r#"<RigidBody id="body" target="shape" dimension="2d" type="dynamic" shape="sphere" />"#,
            13,
        )
        .expect_err("sphere is not a 2D shape");
        assert!(shape_error.message.contains("invalid for dimension"));

        let gravity_error = parse_rigid_body(
            r#"<RigidBody id="body" target="shape" dimension="3d" type="dynamic" gravity={[0,-9.81,0]} />"#,
            14,
        )
        .expect_err("3D gravity belongs to Physics");
        assert!(gravity_error.message.contains("<Physics gravity"));
    }

    #[test]
    fn rigid_body_parses_rotational_contact_and_sleep_controls() {
        let body = parse_rigid_body(
            r#"<RigidBody id="body" target="shape" dimension="3d" type="dynamic"
                 rollingFriction="0.12" restitutionThreshold="0.7"
                 sleepLinearThreshold="0.02" sleepAngularThreshold="0.03"
                 sleepTime="0.65" />"#,
            15,
        )
        .expect("valid rigid body controls");
        assert_eq!(body.rolling_friction, 0.12);
        assert_eq!(body.restitution_threshold, 0.7);
        assert_eq!(body.sleep_linear_threshold, 0.02);
        assert_eq!(body.sleep_angular_threshold, 0.03);
        assert_eq!(body.sleep_time, 0.65);
    }

    #[test]
    fn removed_rigid_body_2d_tag_is_not_accepted() {
        let error = parse_graph_script(
            r#"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Scene id="legacy">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="1s">
          <Layer>
            <Group id="shape">
              <Rect width="20" height="20" />
            </Group>
            <RigidBody2D id="body" target="shape" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="legacy" />
</Graph>
"#,
        )
        .expect_err("the removed tag must fail");
        assert!(
            error.message.contains("RigidBody2D") || error.message.contains("unsupported"),
            "unexpected parser error: {}",
            error.message
        );
    }
}
