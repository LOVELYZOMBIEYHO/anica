// =========================================
// =========================================
// crates/motionloom/src/scene/animation.rs

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::dsl::GraphScript;
use crate::scene::model::{Scene3DNode, SceneNode};

/// Runtime value families accepted by editor-authored animation channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnimationValueType {
    Number,
    Vector3,
    Color,
    Path,
    Discrete,
}

/// Interpolation behavior selected from the target property's type metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnimationInterpolation {
    Linear,
    ComponentLinear,
    ColorLinear,
    PathMorph,
    Step,
}

/// One machine-readable property capability shared by parser, editor, and LLM tooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimationPropertyDescriptor {
    pub path: &'static str,
    pub value_type: AnimationValueType,
    pub interpolation: AnimationInterpolation,
    pub unit: &'static str,
    pub editor_control: &'static str,
    pub node_kinds: &'static [&'static str],
}

const TRANSFORM_NODES: &[&str] = &[
    "Rect",
    "Circle",
    "Ellipse",
    "Line",
    "Polyline",
    "Path",
    "Group",
    "Layer",
    "Use",
    "Text",
    "Image",
    "Svg",
    "Character",
    "Puppet",
    "FaceJaw",
    "Part",
    "Repeat",
    "Camera",
];
const POSITION_NODES: &[&str] = &[
    "Rect",
    "Circle",
    "Ellipse",
    "Line",
    "Polyline",
    "Path",
    "Group",
    "Layer",
    "Use",
    "Text",
    "Image",
    "Svg",
    "Character",
    "Puppet",
    "FaceJaw",
    "Part",
    "Repeat",
    "Camera",
    "Pin",
    "Shadow",
    "SkeletonBone",
    "Simulation",
];
const ROTATION_NODES: &[&str] = &[
    "Rect",
    "Circle",
    "Ellipse",
    "Line",
    "Polyline",
    "Path",
    "Group",
    "Layer",
    "Use",
    "Text",
    "Image",
    "Svg",
    "Character",
    "Puppet",
    "FaceJaw",
    "Part",
    "Repeat",
    "Camera",
    "Pin",
    "SkeletonBone",
];
const SCALE_NODES: &[&str] = &[
    "Rect",
    "Circle",
    "Ellipse",
    "Line",
    "Polyline",
    "Path",
    "Group",
    "Layer",
    "Use",
    "Text",
    "Image",
    "Svg",
    "Character",
    "Puppet",
    "FaceJaw",
    "Part",
    "Repeat",
    "Pin",
    "SkeletonBone",
    "Model",
];
const OPACITY_NODES: &[&str] = &[
    "Rect",
    "Circle",
    "Ellipse",
    "Line",
    "Polyline",
    "Path",
    "Group",
    "Layer",
    "Use",
    "Text",
    "Image",
    "Svg",
    "Character",
    "Puppet",
    "FaceJaw",
    "Part",
    "Repeat",
    "Camera",
    "Mask",
    "Shadow",
];
const SHAPE_NODES: &[&str] = &[
    "Rect", "Circle", "Ellipse", "Line", "Polyline", "Path", "FaceJaw",
];
const PAINT_NODES: &[&str] = &[
    "Rect",
    "Circle",
    "Ellipse",
    "Line",
    "Polyline",
    "Path",
    "FaceJaw",
    "Text",
    "Shadow",
    "Background",
];

macro_rules! number_property {
    ($path:literal, $unit:literal, $control:literal, $nodes:expr) => {
        AnimationPropertyDescriptor {
            path: $path,
            value_type: AnimationValueType::Number,
            interpolation: AnimationInterpolation::Linear,
            unit: $unit,
            editor_control: $control,
            node_kinds: $nodes,
        }
    };
}

/// Canonical registry for properties that may be addressed by `AnimationTarget`.
///
/// Existing transform and path names remain unchanged. New entries are additive,
/// so older scripts and host integrations continue to parse and render as before.
pub static ANIMATION_PROPERTY_DESCRIPTORS: &[AnimationPropertyDescriptor] = &[
    number_property!("x", "px", "number", POSITION_NODES),
    number_property!("y", "px", "number", POSITION_NODES),
    number_property!("rotation", "deg", "angle", ROTATION_NODES),
    number_property!("scale", "ratio", "number", SCALE_NODES),
    number_property!("scaleX", "ratio", "number", TRANSFORM_NODES),
    number_property!("scaleY", "ratio", "number", TRANSFORM_NODES),
    number_property!("skewX", "deg", "angle", TRANSFORM_NODES),
    number_property!("skewY", "deg", "angle", TRANSFORM_NODES),
    number_property!("transformOriginX", "px", "number", TRANSFORM_NODES),
    number_property!("transformOriginY", "px", "number", TRANSFORM_NODES),
    number_property!("opacity", "ratio", "slider", OPACITY_NODES),
    number_property!(
        "width",
        "px",
        "number",
        &[
            "Rect",
            "Line",
            "FaceJaw",
            "Mask",
            "Puppet",
            "Text",
            "RectAreaLight"
        ]
    ),
    number_property!(
        "height",
        "px",
        "number",
        &["Rect", "FaceJaw", "Mask", "Puppet", "RectAreaLight"]
    ),
    number_property!(
        "radius",
        "px",
        "number",
        &[
            "Rect",
            "Circle",
            "Mask",
            "Pin",
            "Simulation",
            "AmbientOcclusion"
        ]
    ),
    number_property!("radiusX", "px", "number", &["Ellipse"]),
    number_property!("radiusY", "px", "number", &["Ellipse"]),
    number_property!("strokeWidth", "px", "number", SHAPE_NODES),
    number_property!(
        "trimStart",
        "ratio",
        "slider",
        &["Polyline", "Path", "FaceJaw"]
    ),
    number_property!(
        "trimEnd",
        "ratio",
        "slider",
        &["Polyline", "Path", "FaceJaw"]
    ),
    number_property!(
        "taperStart",
        "ratio",
        "slider",
        &["Line", "Polyline", "Path", "FaceJaw"]
    ),
    number_property!(
        "taperEnd",
        "ratio",
        "slider",
        &["Line", "Polyline", "Path", "FaceJaw"]
    ),
    number_property!(
        "textureOpacity",
        "ratio",
        "slider",
        &["Rect", "Circle", "Path"]
    ),
    number_property!(
        "textureScale",
        "ratio",
        "number",
        &["Rect", "Circle", "Path"]
    ),
    number_property!(
        "textureMask",
        "ratio",
        "slider",
        &["Rect", "Circle", "Path"]
    ),
    number_property!("deformAmount", "ratio", "slider", &["Group"]),
    number_property!("maskFeather", "px", "number", &["Group", "Layer"]),
    number_property!("maskExpansion", "px", "number", &["Group", "Layer"]),
    number_property!("z", "px", "number", &["Layer"]),
    number_property!("zDepth", "px", "number", &["Layer", "Track"]),
    number_property!("rotationX", "deg", "angle", &["Layer", "Model"]),
    number_property!(
        "rotationY",
        "deg",
        "angle",
        &["Layer", "Model", "EnvironmentLight"]
    ),
    number_property!("rotationZ", "deg", "angle", &["Model"]),
    number_property!("positionX", "world", "number", &["Model"]),
    number_property!("positionY", "world", "number", &["Model"]),
    number_property!("positionZ", "world", "number", &["Model"]),
    number_property!("perspective", "px", "number", &["Layer"]),
    number_property!("playbackRate", "ratio", "number", &["Layer"]),
    number_property!("fontSize", "px", "number", &["Text"]),
    number_property!("tracking", "px", "number", &["Text"]),
    number_property!("lineHeight", "ratio", "number", &["Text"]),
    number_property!("blur", "px", "number", &["Text", "Shadow"]),
    number_property!("softEdge", "px", "number", &["Text"]),
    number_property!("boxPaddingX", "px", "number", &["Text"]),
    number_property!("boxPaddingY", "px", "number", &["Text"]),
    number_property!("boxRadius", "px", "number", &["Text"]),
    number_property!("visibleChars", "count", "number", &["Text"]),
    number_property!("feather", "px", "number", &["Mask"]),
    number_property!("targetX", "px", "number", &["Camera", "Pin"]),
    number_property!("targetY", "px", "number", &["Camera", "Pin"]),
    number_property!("anchorX", "px", "number", &["Camera", "Part"]),
    number_property!("anchorY", "px", "number", &["Camera", "Part"]),
    number_property!("offsetX", "px", "number", &["Camera"]),
    number_property!("offsetY", "px", "number", &["Camera"]),
    number_property!("shakeX", "px", "number", &["Camera"]),
    number_property!("shakeY", "px", "number", &["Camera"]),
    number_property!("zoom", "ratio", "number", &["Camera"]),
    number_property!("fov", "deg", "angle", &["Camera3D"]),
    number_property!(
        "intensity",
        "ratio",
        "number",
        &[
            "EnvironmentLight",
            "DirectionalLight",
            "PointLight",
            "SpotLight",
            "RectAreaLight",
            "AmbientOcclusion",
            "ContactShadow"
        ]
    ),
    number_property!(
        "backgroundIntensity",
        "ratio",
        "number",
        &["EnvironmentLight"]
    ),
    number_property!("backgroundBlur", "ratio", "slider", &["EnvironmentLight"]),
    number_property!("diffuseIntensity", "ratio", "number", &["EnvironmentLight"]),
    number_property!(
        "specularIntensity",
        "ratio",
        "number",
        &["EnvironmentLight"]
    ),
    number_property!("range", "world", "number", &["PointLight", "SpotLight"]),
    number_property!("innerCone", "deg", "angle", &["SpotLight"]),
    number_property!("outerCone", "deg", "angle", &["SpotLight"]),
    number_property!("distance", "world", "number", &["ContactShadow"]),
    number_property!("softness", "ratio", "slider", &["ContactShadow"]),
    number_property!("exposure", "stops", "number", &["Model", "ColorManagement"]),
    number_property!("whiteBalance", "kelvin", "number", &["ColorManagement"]),
    number_property!("contrast", "ratio", "number", &["ColorManagement"]),
    number_property!("amount", "ratio", "slider", &["Puppet", "Simulation"]),
    number_property!("jointSoftness", "ratio", "slider", &["Puppet"]),
    number_property!("stiffness", "ratio", "slider", &["Puppet", "Simulation"]),
    number_property!("damping", "ratio", "slider", &["Puppet", "Simulation"]),
    number_property!("drag", "ratio", "slider", &["Puppet", "Simulation"]),
    number_property!("overlap", "ratio", "slider", &["Puppet"]),
    number_property!("strength", "ratio", "slider", &["Pin"]),
    number_property!("rate", "perSecond", "number", &["Simulation"]),
    number_property!("length", "px", "number", &["SkeletonBone"]),
    number_property!("params.*", "", "effect", &["Pass"]),
    number_property!("bones.*", "", "bone", &["Model"]),
    AnimationPropertyDescriptor {
        path: "position",
        value_type: AnimationValueType::Vector3,
        interpolation: AnimationInterpolation::ComponentLinear,
        unit: "world",
        editor_control: "vector3",
        node_kinds: &["Camera3D", "Model"],
    },
    AnimationPropertyDescriptor {
        path: "target",
        value_type: AnimationValueType::Vector3,
        interpolation: AnimationInterpolation::ComponentLinear,
        unit: "world",
        editor_control: "vector3",
        node_kinds: &["Camera3D"],
    },
    AnimationPropertyDescriptor {
        path: "d",
        value_type: AnimationValueType::Path,
        interpolation: AnimationInterpolation::PathMorph,
        unit: "path",
        editor_control: "path",
        node_kinds: &["Path", "Mask", "LimbEnvelope", "LimbRegion"],
    },
    AnimationPropertyDescriptor {
        path: "color",
        value_type: AnimationValueType::Color,
        interpolation: AnimationInterpolation::ColorLinear,
        unit: "color",
        editor_control: "color",
        node_kinds: PAINT_NODES,
    },
    AnimationPropertyDescriptor {
        path: "fill",
        value_type: AnimationValueType::Color,
        interpolation: AnimationInterpolation::ColorLinear,
        unit: "color",
        editor_control: "color",
        node_kinds: &["Rect", "Circle", "Ellipse", "Path", "FaceJaw"],
    },
    AnimationPropertyDescriptor {
        path: "stroke",
        value_type: AnimationValueType::Color,
        interpolation: AnimationInterpolation::ColorLinear,
        unit: "color",
        editor_control: "color",
        node_kinds: SHAPE_NODES,
    },
    AnimationPropertyDescriptor {
        path: "value",
        value_type: AnimationValueType::Discrete,
        interpolation: AnimationInterpolation::Step,
        unit: "text",
        editor_control: "text",
        node_kinds: &["Text"],
    },
    AnimationPropertyDescriptor {
        path: "activeCamera",
        value_type: AnimationValueType::Discrete,
        interpolation: AnimationInterpolation::Step,
        unit: "cameraId",
        editor_control: "nodeReference",
        node_kinds: &["Scene", "CompositeGroup"],
    },
];

/// Resolve a canonical property path, including dynamic Process parameter paths.
pub fn animation_property_descriptor(path: &str) -> Option<&'static AnimationPropertyDescriptor> {
    ANIMATION_PROPERTY_DESCRIPTORS.iter().find(|descriptor| {
        descriptor.path == path
            || (descriptor.path == "params.*"
                && path
                    .strip_prefix("params.")
                    .is_some_and(|name| !name.trim().is_empty()))
            || (descriptor.path == "bones.*" && parse_bone_property_path(path).is_some())
    })
}

/// Split `bones.<canonical-or-raw-id>.<component>` into its addressable parts.
pub(crate) fn parse_bone_property_path(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_prefix("bones.")?;
    let (bone, component) = rest.rsplit_once('.')?;
    if bone.trim().is_empty()
        || !matches!(
            component,
            "x" | "y" | "z" | "rotationX" | "rotationY" | "rotationZ" | "rotation" | "scale"
        )
    {
        return None;
    }
    Some((bone, component))
}

/// Return the registry subset applicable to one Scene or Process node kind.
pub fn animation_properties_for_node_kind(
    node_kind: &str,
) -> Vec<&'static AnimationPropertyDescriptor> {
    ANIMATION_PROPERTY_DESCRIPTORS
        .iter()
        .filter(|descriptor| descriptor.node_kinds.contains(&node_kind))
        .collect()
}

/// Serialize the runtime registry for browser editors and LLM capability queries.
pub fn animation_property_schema_json() -> String {
    serde_json::to_string_pretty(ANIMATION_PROPERTY_DESCRIPTORS)
        .expect("static AnimationTarget property descriptors are serializable")
}

/// Validate one authored key value without changing the established DSL syntax.
pub fn validate_animation_key_value(
    descriptor: &AnimationPropertyDescriptor,
    value: &str,
) -> Result<(), &'static str> {
    match descriptor.value_type {
        AnimationValueType::Number => match value.trim().parse::<f32>() {
            Ok(value) if value.is_finite() => Ok(()),
            _ => Err("expected a finite number"),
        },
        AnimationValueType::Vector3 => parse_vector3(value)
            .map(|_| ())
            .ok_or("expected three finite numeric components"),
        AnimationValueType::Color => {
            let value = value.trim();
            if value.starts_with('#') || value.starts_with("rgb") || value.starts_with('[') {
                Ok(())
            } else {
                Err("expected a CSS, hex, or byte-array color")
            }
        }
        AnimationValueType::Path => {
            if value.trim().is_empty() {
                Err("path data cannot be empty")
            } else {
                Ok(())
            }
        }
        AnimationValueType::Discrete => Ok(()),
    }
}

pub(crate) fn parse_vector3(value: &str) -> Option<[f32; 3]> {
    let trimmed = value.trim().trim_start_matches('[').trim_end_matches(']');
    let values = trimmed
        .split(',')
        .map(str::trim)
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if values.len() != 3 || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some([values[0], values[1], values[2]])
}

/// Severity used by static AnimationTarget inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnimationDiagnosticSeverity {
    Warning,
    Error,
}

/// One structured authoring diagnostic suitable for UI and LLM repair loops.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimationDiagnostic {
    pub severity: AnimationDiagnosticSeverity,
    pub code: String,
    pub node: String,
    pub property: String,
    pub message: String,
    pub suggestion: Option<String>,
}

/// Static report for duplicate, missing, or incompatible animation targets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimationCapabilityReport {
    pub diagnostics: Vec<AnimationDiagnostic>,
}

impl AnimationCapabilityReport {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == AnimationDiagnosticSeverity::Error)
    }
}

/// Inspect target bindings without changing parser compatibility or rendering.
pub fn inspect_animation_targets(graph: &GraphScript) -> AnimationCapabilityReport {
    let mut node_kinds = HashMap::<String, &'static str>::new();
    for background in &graph.backgrounds {
        if let Some(id) = background.id.as_ref() {
            node_kinds.insert(id.clone(), "Background");
        }
    }
    for text in &graph.texts {
        collect_optional_id(&mut node_kinds, text.id.as_ref(), "Text");
    }
    for image in &graph.images {
        collect_optional_id(&mut node_kinds, image.id.as_ref(), "Image");
    }
    for svg in &graph.svgs {
        collect_optional_id(&mut node_kinds, svg.id.as_ref(), "Svg");
    }
    for scene in &graph.scenes {
        node_kinds.insert(scene.id.clone(), "Scene");
        collect_scene_node_kinds(&scene.children, &mut node_kinds);
    }
    collect_scene_node_kinds(&graph.scene_nodes, &mut node_kinds);
    for skeleton in &graph.skeletons {
        for bone in &skeleton.bones {
            node_kinds.insert(bone.id.clone(), "SkeletonBone");
        }
    }
    for pass in &graph.passes {
        node_kinds.insert(pass.id.clone(), "Pass");
    }

    let mut seen = HashSet::<(&str, &str)>::new();
    let mut diagnostics = Vec::new();
    for target in &graph.animation_targets {
        if !seen.insert((&target.node, &target.property)) {
            diagnostics.push(AnimationDiagnostic {
                severity: AnimationDiagnosticSeverity::Error,
                code: "duplicate_channel".to_string(),
                node: target.node.clone(),
                property: target.property.clone(),
                message: "More than one AnimationTarget addresses this node/property channel."
                    .to_string(),
                suggestion: Some("Keep one channel and merge its Key children.".to_string()),
            });
        }
        let Some(kind) = node_kinds.get(&target.node).copied() else {
            diagnostics.push(AnimationDiagnostic {
                severity: AnimationDiagnosticSeverity::Error,
                code: "missing_target_node".to_string(),
                node: target.node.clone(),
                property: target.property.clone(),
                message: "AnimationTarget references a node id that is not present in the graph."
                    .to_string(),
                suggestion: Some(
                    "Add the id to the intended node or update target.node.".to_string(),
                ),
            });
            continue;
        };
        let Some(descriptor) = animation_property_descriptor(&target.property) else {
            continue;
        };
        if !descriptor.node_kinds.contains(&kind) {
            diagnostics.push(AnimationDiagnostic {
                severity: AnimationDiagnosticSeverity::Warning,
                code: "property_node_mismatch".to_string(),
                node: target.node.clone(),
                property: target.property.clone(),
                message: format!(
                    "Property is registered but is not declared for node kind {kind}."
                ),
                suggestion: Some(
                    "Query the animation property registry for this node kind.".to_string(),
                ),
            });
        }
    }
    AnimationCapabilityReport { diagnostics }
}

fn collect_optional_id(
    node_kinds: &mut HashMap<String, &'static str>,
    id: Option<&String>,
    kind: &'static str,
) {
    if let Some(id) = id {
        node_kinds.insert(id.clone(), kind);
    }
}

fn collect_scene_node_kinds(nodes: &[SceneNode], node_kinds: &mut HashMap<String, &'static str>) {
    for node in nodes {
        let (id, kind, children): (Option<&String>, &'static str, Option<&[SceneNode]>) = match node
        {
            SceneNode::Timeline(node) => (node.id.as_ref(), "Timeline", Some(&node.children)),
            SceneNode::Track(node) => (node.id.as_ref(), "Track", Some(&node.children)),
            SceneNode::Sequence(node) => (node.id.as_ref(), "Sequence", Some(&node.children)),
            SceneNode::Chain(node) => (node.id.as_ref(), "Chain", Some(&node.children)),
            SceneNode::Text(node) => (node.id.as_ref(), "Text", None),
            SceneNode::Image(node) => (node.id.as_ref(), "Image", None),
            SceneNode::Svg(node) => (node.id.as_ref(), "Svg", None),
            SceneNode::Rect(node) => (node.id.as_ref(), "Rect", None),
            SceneNode::Circle(node) => (node.id.as_ref(), "Circle", None),
            SceneNode::Ellipse(node) => (node.id.as_ref(), "Ellipse", None),
            SceneNode::Line(node) => (node.id.as_ref(), "Line", None),
            SceneNode::Polyline(node) => (node.id.as_ref(), "Polyline", None),
            SceneNode::Path(node) => (node.id.as_ref(), "Path", None),
            SceneNode::FaceJaw(node) => (node.id.as_ref(), "FaceJaw", None),
            SceneNode::Shadow(node) => (node.id.as_ref(), "Shadow", None),
            SceneNode::Group(node) => {
                if let Some(composite) = node.composite.as_ref() {
                    for node_3d in &composite.nodes_3d {
                        match node_3d {
                            Scene3DNode::Camera(node) => {
                                collect_optional_id(node_kinds, node.id.as_ref(), "Camera3D")
                            }
                            Scene3DNode::EnvironmentLight(node) => collect_optional_id(
                                node_kinds,
                                node.id.as_ref(),
                                "EnvironmentLight",
                            ),
                            Scene3DNode::DirectionalLight(node) => collect_optional_id(
                                node_kinds,
                                node.id.as_ref(),
                                "DirectionalLight",
                            ),
                            Scene3DNode::PointLight(node) => {
                                collect_optional_id(node_kinds, node.id.as_ref(), "PointLight")
                            }
                            Scene3DNode::SpotLight(node) => {
                                collect_optional_id(node_kinds, node.id.as_ref(), "SpotLight")
                            }
                            Scene3DNode::RectAreaLight(node) => {
                                collect_optional_id(node_kinds, node.id.as_ref(), "RectAreaLight")
                            }
                            Scene3DNode::AmbientOcclusion(node) => collect_optional_id(
                                node_kinds,
                                node.id.as_ref(),
                                "AmbientOcclusion",
                            ),
                            Scene3DNode::ContactShadow(node) => {
                                collect_optional_id(node_kinds, node.id.as_ref(), "ContactShadow")
                            }
                            Scene3DNode::ColorManagement(node) => {
                                collect_optional_id(node_kinds, node.id.as_ref(), "ColorManagement")
                            }
                            Scene3DNode::Model(node) => {
                                collect_optional_id(node_kinds, node.id.as_ref(), "Model")
                            }
                            Scene3DNode::VolumeRepeat(node) => {
                                collect_optional_id(node_kinds, node.id.as_ref(), "Repeat")
                            }
                            Scene3DNode::Anchor(node) => {
                                node_kinds.insert(node.id.clone(), "Anchor3D");
                            }
                            Scene3DNode::RigidBody(node) => {
                                node_kinds.insert(node.id.clone(), "RigidBody");
                            }
                            Scene3DNode::Debug(_) => {}
                        }
                    }
                }
                (node.id.as_ref(), "Group", Some(&node.children))
            }
            SceneNode::Part(node) => (node.id.as_ref(), "Part", Some(&node.children)),
            SceneNode::Repeat(node) => (node.id.as_ref(), "Repeat", Some(&node.children)),
            SceneNode::Mask(node) => (node.id.as_ref(), "Mask", Some(&node.children)),
            SceneNode::Precompose(node) => (Some(&node.id), "Precompose", Some(&node.children)),
            SceneNode::Use(node) => (node.id.as_ref(), "Use", None),
            SceneNode::Layer(node) => (node.id.as_ref(), "Layer", Some(&node.children)),
            SceneNode::Camera(node) => (node.id.as_ref(), "Camera", Some(&node.children)),
            SceneNode::Character(node) => (node.id.as_ref(), "Character", Some(&node.children)),
            SceneNode::Puppet(node) => (node.id.as_ref(), "Puppet", Some(&node.children)),
            SceneNode::Pin(node) => (node.id.as_ref(), "Pin", None),
            SceneNode::Simulation(binding) => (simulation_binding_id(binding), "Simulation", None),
            _ => (None, "", None),
        };
        collect_optional_id(node_kinds, id, kind);
        if let Some(children) = children {
            collect_scene_node_kinds(children, node_kinds);
        }
    }
}

fn simulation_binding_id(
    binding: &crate::simulation::model::SimulationBindingNode,
) -> Option<&String> {
    use crate::simulation::model::SimulationBindingNode;
    match binding {
        SimulationBindingNode::SpringChain(node) => node.id.as_ref(),
        SimulationBindingNode::DynamicCurve(node) => node.id.as_ref(),
        SimulationBindingNode::DistanceConstraint(node) => node.id.as_ref(),
        SimulationBindingNode::Hinge(node) => node.id.as_ref(),
        SimulationBindingNode::RigidBody(node) => Some(&node.id),
        SimulationBindingNode::ParticleEmitter(node) => Some(&node.id),
        SimulationBindingNode::Cloth(node) => Some(&node.id),
        SimulationBindingNode::HairStrandField(node) => Some(&node.id),
        SimulationBindingNode::CacheBake(node) => Some(&node.id),
    }
}

#[cfg(test)]
mod tests {
    use crate::parse_graph_script;

    use super::{
        AnimationDiagnosticSeverity, AnimationValueType, animation_property_descriptor,
        animation_property_schema_json, inspect_animation_targets, validate_animation_key_value,
    };

    #[test]
    fn bone_property_paths_are_typed_numeric_model_channels() {
        let descriptor =
            animation_property_descriptor("bones.forearm_r.rotationZ").expect("bone descriptor");
        assert_eq!(descriptor.value_type, AnimationValueType::Number);
        assert!(descriptor.node_kinds.contains(&"Model"));
        assert!(animation_property_descriptor("bones.forearm_r.unknown").is_none());
    }

    #[test]
    fn registry_keeps_legacy_and_dynamic_process_properties() {
        assert_eq!(
            animation_property_descriptor("rotation")
                .unwrap()
                .value_type,
            AnimationValueType::Number
        );
        assert_eq!(
            animation_property_descriptor("params.sigma")
                .unwrap()
                .value_type,
            AnimationValueType::Number
        );
    }

    #[test]
    fn registry_exposes_animatable_scene_lighting_channels() {
        for property in [
            "intensity",
            "rotationY",
            "backgroundIntensity",
            "specularIntensity",
            "range",
            "innerCone",
            "outerCone",
            "width",
            "radius",
            "whiteBalance",
            "contrast",
        ] {
            assert!(
                animation_property_descriptor(property).is_some(),
                "missing lighting animation property {property}"
            );
        }
    }

    #[test]
    fn typed_values_are_validated_by_property_metadata() {
        let vector = animation_property_descriptor("position").unwrap();
        assert!(validate_animation_key_value(vector, "[0, 1.5, -2]").is_ok());
        assert!(validate_animation_key_value(vector, "0, 1").is_err());
    }

    #[test]
    fn property_registry_has_machine_readable_json() {
        let json = animation_property_schema_json();
        assert!(json.contains("\"params.*\""));
        assert!(json.contains("\"vector3\""));
    }

    #[test]
    fn inspector_reports_missing_and_duplicate_channels() {
        let script = r##"<Graph fps={30} duration="1s" size={[100,100]}>
  <Scene id="main">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect id="card" x="0" y="0" width="10" height="10" color="#fff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <AnimationTarget node="card" property="x">
    <Key time="0s" value="0" />
  </AnimationTarget>
  <AnimationTarget node="card" property="x">
    <Key time="1s" value="1" />
  </AnimationTarget>
  <AnimationTarget node="missing" property="opacity">
    <Key time="0s" value="1" />
  </AnimationTarget>
  <Present from="main" />
</Graph>"##;
        let graph = parse_graph_script(script).unwrap();
        let report = inspect_animation_targets(&graph);
        assert!(report.has_errors());
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.severity == AnimationDiagnosticSeverity::Error
                && diagnostic.code == "duplicate_channel"
        }));
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "missing_target_node")
        );
    }
}
