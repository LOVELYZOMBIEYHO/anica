// =========================================
// =========================================
// crates/motionloom/src/scene/editor_actions.rs

use crate::dsl::{ActionBoneNode, ActionNode, GraphScript, parse_graph_script};
use crate::error::GraphParseError;
use crate::scene::model::{Scene3DNode, SceneNode};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt;

const BONE_CHANNELS: [&str; 14] = [
    "x",
    "y",
    "z",
    "rotation",
    "rotationX",
    "rotationY",
    "rotationZ",
    "forward",
    "side",
    "twist",
    "bend",
    "turn",
    "scale",
    "opacity",
];

/// Complete editor-facing Action view extracted from one graph revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableActionDocument {
    pub fps: f32,
    pub duration_ms: u64,
    pub actions: Vec<EditableAction>,
    pub bindings: Vec<EditableActionBinding>,
    pub skeletons: Vec<EditableSkeleton>,
    pub model_targets: Vec<EditableModelTarget>,
}

/// One executable Action represented without exposing renderer-only fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableAction {
    pub id: String,
    pub external: bool,
    pub source: Option<String>,
    pub clip: Option<String>,
    pub source_profile: Option<String>,
    pub skeleton: Option<String>,
    pub duration_ms: u64,
    pub poses: Vec<EditableActionPose>,
    pub contacts: Vec<EditableActionContact>,
    pub iks: Vec<EditableActionIk>,
    pub diagnostics: Vec<EditableActionDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableActionPose {
    pub time_ms: u64,
    pub bones: Vec<EditableActionBone>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableActionBone {
    pub id: String,
    pub channels: BTreeMap<String, String>,
    pub interpolation: String,
    pub in_tangent: String,
    pub out_tangent: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableActionContact {
    pub id: String,
    pub effector: String,
    pub target: String,
    pub from: f32,
    pub to: f32,
    pub mode: String,
    pub weight: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableActionIk {
    pub id: String,
    pub root: String,
    pub mid: String,
    pub end: String,
    pub target_x: String,
    pub target_y: String,
    pub target_z: String,
    pub bend: String,
    pub weight: String,
    pub iterations: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableActionBinding {
    pub target: String,
    pub action: String,
    pub at_ms: u64,
    pub duration_ms: Option<u64>,
    pub looped: bool,
    pub speed: String,
    pub weight: String,
    pub root_motion: Option<String>,
    pub ground: Option<String>,
    pub contact_targets: BTreeMap<String, String>,
    pub destination: Option<String>,
    pub face: Option<String>,
    pub contact_correction: Option<String>,
    pub foot_lock: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableSkeleton {
    pub id: String,
    pub profile: Option<String>,
    pub bones: Vec<EditableSkeletonBone>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableSkeletonBone {
    pub id: String,
    pub parent: Option<String>,
    pub role: Option<String>,
    pub side: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableModelTarget {
    pub id: String,
    pub profile: Option<String>,
    pub rig: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableActionDiagnostic {
    pub severity: String,
    pub code: String,
    pub message: String,
}

/// Typed commands keep browser edits narrow and make every mutation testable in Rust.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum ActionEditCommand {
    CreateAction {
        id: String,
        #[serde(default = "default_humanoid_skeleton")]
        skeleton: String,
        duration_ms: u64,
    },
    DuplicateAction {
        action_id: String,
        new_id: String,
    },
    SetActionMetadata {
        action_id: String,
        #[serde(default)]
        skeleton: Option<String>,
        #[serde(default)]
        duration_ms: Option<u64>,
    },
    AddPose {
        action_id: String,
        time_ms: u64,
        #[serde(default)]
        copy_from_ms: Option<u64>,
    },
    RemovePose {
        action_id: String,
        time_ms: u64,
    },
    MovePose {
        action_id: String,
        from_ms: u64,
        to_ms: u64,
    },
    SetBoneChannel {
        action_id: String,
        time_ms: u64,
        bone_id: String,
        channel: String,
        #[serde(default)]
        value: Option<String>,
    },
    SetBoneKeyMetadata {
        action_id: String,
        time_ms: u64,
        bone_id: String,
        interpolation: String,
        #[serde(default)]
        in_tangent: Option<String>,
        #[serde(default)]
        out_tangent: Option<String>,
    },
    MirrorPose {
        action_id: String,
        time_ms: u64,
        #[serde(default)]
        direction: Option<String>,
    },
    UpsertContact {
        action_id: String,
        contact: EditableActionContact,
    },
    RemoveContact {
        action_id: String,
        contact_id: String,
    },
    UpsertIk {
        action_id: String,
        ik: EditableActionIk,
    },
    RemoveIk {
        action_id: String,
        ik_id: String,
    },
    SetBinding {
        target: String,
        action: String,
        attribute: String,
        #[serde(default)]
        value: Option<String>,
    },
}

fn default_humanoid_skeleton() -> String {
    "humanoid_v1".to_string()
}

/// Typed failures distinguish malformed commands from invalid generated DSL.
#[derive(Debug)]
pub enum ActionEditError {
    Parse(GraphParseError),
    InvalidCommand(String),
    ActionNotFound(String),
    PoseNotFound { action: String, time_ms: u64 },
    ContactNotFound { action: String, contact: String },
    BindingNotFound { target: String, action: String },
    SourceSpanNotFound(String),
}

impl fmt::Display for ActionEditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(err) => write!(f, "{err}"),
            Self::InvalidCommand(message) => write!(f, "Invalid Action edit: {message}"),
            Self::ActionNotFound(id) => write!(f, "Action not found: {id}"),
            Self::PoseNotFound { action, time_ms } => {
                write!(f, "Pose not found in Action {action} at {time_ms}ms")
            }
            Self::ContactNotFound { action, contact } => {
                write!(f, "Contact not found in Action {action}: {contact}")
            }
            Self::BindingNotFound { target, action } => {
                write!(
                    f,
                    "ApplyAction binding not found: target={target}, action={action}"
                )
            }
            Self::SourceSpanNotFound(label) => write!(f, "DSL source span not found: {label}"),
        }
    }
}

impl Error for ActionEditError {}

impl From<GraphParseError> for ActionEditError {
    fn from(value: GraphParseError) -> Self {
        Self::Parse(value)
    }
}

fn bone_channels(bone: &ActionBoneNode) -> BTreeMap<String, String> {
    let pairs = [
        ("x", bone.x.as_ref()),
        ("y", bone.y.as_ref()),
        ("z", bone.z.as_ref()),
        ("rotation", bone.rotation.as_ref()),
        ("rotationX", bone.rotation_x.as_ref()),
        ("rotationY", bone.rotation_y.as_ref()),
        ("rotationZ", bone.rotation_z.as_ref()),
        ("forward", bone.forward.as_ref()),
        ("side", bone.side.as_ref()),
        ("twist", bone.twist.as_ref()),
        ("bend", bone.bend.as_ref()),
        ("turn", bone.turn.as_ref()),
        ("scale", bone.scale.as_ref()),
        ("opacity", bone.opacity.as_ref()),
    ];
    pairs
        .into_iter()
        .filter_map(|(key, value)| value.map(|value| (key.to_string(), value.clone())))
        .collect()
}

fn action_diagnostics(action: &ActionNode) -> Vec<EditableActionDiagnostic> {
    let mut diagnostics = Vec::new();
    if action.source.is_none() && action.poses.len() < 2 {
        diagnostics.push(EditableActionDiagnostic {
            severity: "warning".to_string(),
            code: "action.pose_count".to_string(),
            message: "A looping authored Action normally needs at least two poses.".to_string(),
        });
    }
    if let (Some(first), Some(last)) = (action.poses.first(), action.poses.last()) {
        let first_bones = first
            .bones
            .iter()
            .map(|bone| (bone.id.as_str(), bone_channels(bone)))
            .collect::<HashMap<_, _>>();
        let mut largest_delta = 0.0_f32;
        for bone in &last.bones {
            let Some(channels) = first_bones.get(bone.id.as_str()) else {
                continue;
            };
            for (channel, value) in bone_channels(bone) {
                let Some(first_value) = channels.get(&channel) else {
                    continue;
                };
                let (Ok(left), Ok(right)) = (first_value.parse::<f32>(), value.parse::<f32>())
                else {
                    continue;
                };
                largest_delta = largest_delta.max((left - right).abs());
            }
        }
        if largest_delta > 8.0 {
            diagnostics.push(EditableActionDiagnostic {
                severity: "warning".to_string(),
                code: "action.loop_seam".to_string(),
                message: format!(
                    "First and last poses differ by up to {largest_delta:.1}; inspect the loop seam."
                ),
            });
        }
    }
    for left in 0..action.contacts.len() {
        for right in left + 1..action.contacts.len() {
            let a = &action.contacts[left];
            let b = &action.contacts[right];
            if a.effector == b.effector && a.from < b.to && b.from < a.to {
                diagnostics.push(EditableActionDiagnostic {
                    severity: "warning".to_string(),
                    code: "action.contact_overlap".to_string(),
                    message: format!("Contacts {} and {} overlap on {}.", a.id, b.id, a.effector),
                });
            }
        }
    }
    diagnostics
}

fn editable_action(action: &ActionNode) -> EditableAction {
    EditableAction {
        id: action.id.clone(),
        external: action.source.is_some(),
        source: action.source.clone(),
        clip: action.clip.clone(),
        source_profile: action.source_profile.clone(),
        skeleton: action.skeleton.clone(),
        duration_ms: action.duration_ms,
        poses: action
            .poses
            .iter()
            .map(|pose| EditableActionPose {
                time_ms: (pose.t.max(0.0) * 1000.0).round() as u64,
                bones: pose
                    .bones
                    .iter()
                    .map(|bone| EditableActionBone {
                        id: bone.id.clone(),
                        channels: bone_channels(bone),
                        interpolation: bone
                            .interpolation
                            .clone()
                            .unwrap_or_else(|| "linear".to_string()),
                        in_tangent: bone.in_tangent.clone().unwrap_or_else(|| "0".to_string()),
                        out_tangent: bone.out_tangent.clone().unwrap_or_else(|| "0".to_string()),
                    })
                    .collect(),
            })
            .collect(),
        contacts: action
            .contacts
            .iter()
            .map(|contact| EditableActionContact {
                id: contact.id.clone(),
                effector: contact.effector.clone(),
                target: contact.target.clone(),
                from: contact.from,
                to: contact.to,
                mode: contact.mode.clone(),
                weight: contact.weight.clone(),
            })
            .collect(),
        iks: action
            .iks
            .iter()
            .enumerate()
            .map(|(index, ik)| EditableActionIk {
                id: ik
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("{}_{}_{}_{index}", ik.root, ik.mid, ik.end)),
                root: ik.root.clone(),
                mid: ik.mid.clone(),
                end: ik.end.clone(),
                target_x: ik.target_x.clone(),
                target_y: ik.target_y.clone(),
                target_z: ik.target_z.clone(),
                bend: ik.bend.clone(),
                weight: ik.weight.clone(),
                iterations: ik.iterations.clone(),
            })
            .collect(),
        diagnostics: action_diagnostics(action),
    }
}

fn collect_model_targets(nodes: &[SceneNode], targets: &mut Vec<EditableModelTarget>) {
    for node in nodes {
        let children = match node {
            SceneNode::Timeline(node) => Some(node.children.as_slice()),
            SceneNode::Track(node) => Some(node.children.as_slice()),
            SceneNode::Sequence(node) => Some(node.children.as_slice()),
            SceneNode::Chain(node) => Some(node.children.as_slice()),
            SceneNode::Group(node) => {
                if let Some(composite) = node.composite.as_ref() {
                    for node_3d in &composite.nodes_3d {
                        if let Scene3DNode::Model(model) = node_3d
                            && let Some(id) = model.id.as_ref()
                        {
                            targets.push(EditableModelTarget {
                                id: id.clone(),
                                profile: model.profile.clone(),
                                rig: model.rig.clone(),
                            });
                        }
                    }
                }
                Some(node.children.as_slice())
            }
            SceneNode::Part(node) => Some(node.children.as_slice()),
            SceneNode::Repeat(node) => Some(node.children.as_slice()),
            SceneNode::Mask(node) => Some(node.children.as_slice()),
            SceneNode::Precompose(node) => Some(node.children.as_slice()),
            SceneNode::Layer(node) => Some(node.children.as_slice()),
            SceneNode::Camera(node) => Some(node.children.as_slice()),
            SceneNode::Character(node) => Some(node.children.as_slice()),
            SceneNode::Puppet(node) => Some(node.children.as_slice()),
            _ => None,
        };
        if let Some(children) = children {
            collect_model_targets(children, targets);
        }
    }
}

/// Extract Action, binding, rig, and model data needed by visual editors.
pub fn extract_editable_action_document(
    script: &str,
) -> Result<EditableActionDocument, ActionEditError> {
    let graph = parse_graph_script(script)?;
    let mut model_targets = Vec::new();
    for scene in &graph.scenes {
        collect_model_targets(&scene.children, &mut model_targets);
    }
    collect_model_targets(&graph.scene_nodes, &mut model_targets);
    Ok(EditableActionDocument {
        fps: graph.fps,
        duration_ms: graph.duration_ms,
        actions: graph.actions.iter().map(editable_action).collect(),
        bindings: graph
            .apply_actions
            .iter()
            .map(|binding| EditableActionBinding {
                target: binding.target.clone(),
                action: binding.action.clone(),
                at_ms: binding.at_ms,
                duration_ms: binding.duration_ms,
                looped: binding.r#loop,
                speed: binding.speed.clone(),
                weight: binding.weight.clone(),
                root_motion: binding.root_motion.clone(),
                ground: binding.ground.clone(),
                contact_targets: binding
                    .contact_targets
                    .iter()
                    .map(|(slot, target)| (slot.clone(), target.clone()))
                    .collect(),
                destination: binding.destination.clone(),
                face: binding.face.clone(),
                contact_correction: binding.contact_correction.clone(),
                foot_lock: binding.foot_lock.clone(),
            })
            .collect(),
        skeletons: graph
            .skeletons
            .iter()
            .map(|skeleton| EditableSkeleton {
                id: skeleton.id.clone(),
                profile: skeleton.profile.clone(),
                bones: skeleton
                    .bones
                    .iter()
                    .map(|bone| EditableSkeletonBone {
                        id: bone.id.clone(),
                        parent: bone.parent.clone(),
                        role: bone.role.clone(),
                        side: bone.side.clone(),
                    })
                    .collect(),
            })
            .collect(),
        model_targets,
    })
}

#[derive(Debug, Clone, Copy)]
struct SourceSpan {
    start: usize,
    end: usize,
}

fn line_ranges(script: &str) -> Vec<(usize, usize, &str)> {
    let mut output = Vec::new();
    let mut start = 0;
    for line in script.split_inclusive('\n') {
        let end = start + line.len();
        output.push((start, end, line));
        start = end;
    }
    if start < script.len() || script.is_empty() {
        output.push((start, script.len(), &script[start..]));
    }
    output
}

fn tag_attr(tag: &str, attribute: &str) -> Option<String> {
    let needle = format!("{attribute}=");
    let start = tag.find(&needle)? + needle.len();
    let tail = tag[start..].trim_start();
    let quote = tail.chars().next()?;
    if quote == '"' || quote == '\'' {
        let value = &tail[1..];
        return value.find(quote).map(|end| value[..end].to_string());
    }
    if let Some(value) = tail.strip_prefix('{') {
        return value.find('}').map(|end| value[..end].trim().to_string());
    }
    Some(
        tail.split(|character: char| character.is_whitespace() || character == '>')
            .next()?
            .trim_end_matches('/')
            .to_string(),
    )
}

fn opening_tag_end(script: &str, start: usize) -> Option<usize> {
    script[start..].find('>').map(|offset| start + offset + 1)
}

fn action_spans(script: &str) -> Vec<(String, SourceSpan)> {
    let ranges = line_ranges(script);
    let mut output = Vec::new();
    let mut i = 0;
    let mut in_comment = false;
    while i < ranges.len() {
        let (start, _, line) = ranges[i];
        let trimmed = line.trim_start();
        if trimmed.starts_with("<!--") {
            in_comment = !trimmed.contains("-->");
            i += 1;
            continue;
        }
        if in_comment {
            if trimmed.contains("-->") {
                in_comment = false;
            }
            i += 1;
            continue;
        }
        if !trimmed.starts_with("<Action")
            || trimmed
                .chars()
                .nth("<Action".len())
                .is_some_and(|character| character.is_alphanumeric())
        {
            i += 1;
            continue;
        }
        let tag_start = start + line.len() - line.trim_start().len();
        let Some(tag_end) = opening_tag_end(script, tag_start) else {
            break;
        };
        let tag = &script[tag_start..tag_end];
        let Some(id) = tag_attr(tag, "id") else {
            i += 1;
            continue;
        };
        if tag.trim_end().ends_with("/>") {
            output.push((
                id,
                SourceSpan {
                    start: tag_start,
                    end: tag_end,
                },
            ));
            i += 1;
            continue;
        }
        let Some(close_offset) = script[tag_end..].find("</Action>") else {
            break;
        };
        let end = tag_end + close_offset + "</Action>".len();
        output.push((
            id,
            SourceSpan {
                start: tag_start,
                end,
            },
        ));
        while i < ranges.len() && ranges[i].0 < end {
            i += 1;
        }
    }
    output
}

fn action_span(script: &str, action_id: &str) -> Result<SourceSpan, ActionEditError> {
    action_spans(script)
        .into_iter()
        .find_map(|(id, span)| (id == action_id).then_some(span))
        .ok_or_else(|| ActionEditError::ActionNotFound(action_id.to_string()))
}

fn child_spans(block: &str, tag_name: &str) -> Vec<SourceSpan> {
    let ranges = line_ranges(block);
    let mut output = Vec::new();
    let open = format!("<{tag_name}");
    let close = format!("</{tag_name}>");
    let mut i = 0;
    while i < ranges.len() {
        let (line_start, _, line) = ranges[i];
        let trimmed = line.trim_start();
        if !trimmed.starts_with(&open) {
            i += 1;
            continue;
        }
        let start = line_start + line.len() - trimmed.len();
        let Some(tag_end) = opening_tag_end(block, start) else {
            break;
        };
        if block[start..tag_end].trim_end().ends_with("/>") {
            output.push(SourceSpan {
                start,
                end: tag_end,
            });
            i += 1;
            continue;
        }
        let Some(close_offset) = block[tag_end..].find(&close) else {
            break;
        };
        output.push(SourceSpan {
            start,
            end: tag_end + close_offset + close.len(),
        });
        i += 1;
    }
    output
}

fn pose_span(block: &str, time_ms: u64) -> Option<SourceSpan> {
    child_spans(block, "Pose").into_iter().find(|span| {
        let Some(tag_end) = opening_tag_end(block, span.start) else {
            return false;
        };
        let tag = &block[span.start..tag_end];
        let raw = tag_attr(tag, "t").or_else(|| tag_attr(tag, "time"));
        raw.and_then(|value| parse_time_ms_literal(&value)) == Some(time_ms)
    })
}

fn parse_time_ms_literal(value: &str) -> Option<u64> {
    let value = value.trim();
    if let Some(ms) = value.strip_suffix("ms") {
        return ms
            .trim()
            .parse::<f64>()
            .ok()
            .map(|value| value.round() as u64);
    }
    value
        .strip_suffix('s')
        .unwrap_or(value)
        .trim()
        .parse::<f64>()
        .ok()
        .map(|value| (value * 1000.0).round() as u64)
}

fn format_seconds(time_ms: u64) -> String {
    let seconds = time_ms as f64 / 1000.0;
    let formatted = format!("{seconds:.3}");
    format!("{}s", formatted.trim_end_matches('0').trim_end_matches('.'))
}

fn replace_span(script: &str, span: SourceSpan, replacement: &str) -> String {
    format!(
        "{}{}{}",
        &script[..span.start],
        replacement,
        &script[span.end..]
    )
}

fn indent_before(script: &str, position: usize) -> String {
    let line_start = script[..position].rfind('\n').map_or(0, |index| index + 1);
    script[line_start..position]
        .chars()
        .take_while(|character| character.is_whitespace())
        .collect()
}

fn upsert_tag_attr(tag: &str, attribute: &str, value: Option<&str>) -> String {
    let mut search = 0;
    let needle = format!("{attribute}=");
    while let Some(offset) = tag[search..].find(&needle) {
        let start = search + offset;
        let boundary_ok = start == 0
            || tag[..start]
                .chars()
                .next_back()
                .is_some_and(char::is_whitespace);
        if !boundary_ok {
            search = start + needle.len();
            continue;
        }
        let mut end = start + needle.len();
        let rest = &tag[end..];
        let leading = rest.len() - rest.trim_start().len();
        end += leading;
        let rest = &tag[end..];
        if let Some(quote) = rest
            .chars()
            .next()
            .filter(|value| *value == '"' || *value == '\'')
        {
            end += 1;
            if let Some(offset) = tag[end..].find(quote) {
                end += offset + 1;
            }
        } else if rest.starts_with('{') {
            if let Some(offset) = rest.find('}') {
                end += offset + 1;
            }
        } else {
            end += rest
                .find(|character: char| character.is_whitespace() || character == '>')
                .unwrap_or(rest.len());
        }
        return match value {
            Some(value) => format!(
                "{}{}=\"{}\"{}",
                &tag[..start],
                attribute,
                value,
                &tag[end..]
            ),
            None => {
                let mut remove_start = start;
                while remove_start > 0 && tag.as_bytes()[remove_start - 1].is_ascii_whitespace() {
                    remove_start -= 1;
                    if tag.as_bytes()[remove_start] == b'\n' {
                        remove_start += 1;
                        break;
                    }
                }
                format!("{}{}", &tag[..remove_start], &tag[end..])
            }
        };
    }
    let Some(value) = value else {
        return tag.to_string();
    };
    let insert = tag
        .rfind("/>")
        .or_else(|| tag.rfind('>'))
        .unwrap_or(tag.len());
    format!(
        "{} {}=\"{}\"{}",
        &tag[..insert].trim_end(),
        attribute,
        value,
        &tag[insert..]
    )
}

fn validate_channel(channel: &str) -> Result<(), ActionEditError> {
    if BONE_CHANNELS.contains(&channel) {
        Ok(())
    } else {
        Err(ActionEditError::InvalidCommand(format!(
            "unsupported Bone channel {channel}"
        )))
    }
}

fn ensure_action_authored(graph: &GraphScript, action_id: &str) -> Result<(), ActionEditError> {
    let action = graph
        .actions
        .iter()
        .find(|action| action.id == action_id)
        .ok_or_else(|| ActionEditError::ActionNotFound(action_id.to_string()))?;
    if action.source.is_some() {
        return Err(ActionEditError::InvalidCommand(format!(
            "external Action {action_id} cannot contain authored poses"
        )));
    }
    Ok(())
}

fn set_bone_channel(
    script: &str,
    action_id: &str,
    time_ms: u64,
    bone_id: &str,
    channel: &str,
    value: Option<&str>,
) -> Result<String, ActionEditError> {
    validate_channel(channel)?;
    let graph = parse_graph_script(script)?;
    ensure_action_authored(&graph, action_id)?;
    if let Some(value) = value
        && value.trim().parse::<f32>().is_err()
    {
        return Err(ActionEditError::InvalidCommand(format!(
            "Bone {bone_id}.{channel} must be numeric"
        )));
    }
    let action_span = action_span(script, action_id)?;
    let action_block = &script[action_span.start..action_span.end];
    let pose = pose_span(action_block, time_ms).ok_or_else(|| ActionEditError::PoseNotFound {
        action: action_id.to_string(),
        time_ms,
    })?;
    let pose_block = &action_block[pose.start..pose.end];
    let existing_bone = child_spans(pose_block, "Bone").into_iter().find(|span| {
        opening_tag_end(pose_block, span.start)
            .and_then(|end| tag_attr(&pose_block[span.start..end], "id"))
            .as_deref()
            == Some(bone_id)
    });
    let next_action = if let Some(span) = existing_bone {
        let tag = &pose_block[span.start..span.end];
        let replacement = upsert_tag_attr(tag, channel, value);
        let next_pose = replace_span(pose_block, span, &replacement);
        replace_span(action_block, pose, &next_pose)
    } else {
        let Some(value) = value else {
            return Ok(script.to_string());
        };
        let close = pose_block
            .rfind("</Pose>")
            .ok_or_else(|| ActionEditError::SourceSpanNotFound("</Pose>".to_string()))?;
        let pose_indent = indent_before(pose_block, close);
        let bone_indent = format!("{pose_indent}  ");
        let insertion = format!("{bone_indent}<Bone id=\"{bone_id}\" {channel}=\"{value}\" />\n");
        let next_pose = format!(
            "{}{}{}",
            &pose_block[..close],
            insertion,
            &pose_block[close..]
        );
        replace_span(action_block, pose, &next_pose)
    };
    let output = replace_span(script, action_span, &next_action);
    parse_graph_script(&output)?;
    Ok(output)
}

fn set_bone_key_metadata(
    script: &str,
    action_id: &str,
    time_ms: u64,
    bone_id: &str,
    interpolation: &str,
    in_tangent: Option<&str>,
    out_tangent: Option<&str>,
) -> Result<String, ActionEditError> {
    let interpolation = interpolation.to_ascii_lowercase();
    if !matches!(
        interpolation.as_str(),
        "linear" | "bezier" | "ease" | "hold"
    ) {
        return Err(ActionEditError::InvalidCommand(format!(
            "unsupported interpolation {interpolation}"
        )));
    }
    for value in [in_tangent, out_tangent].into_iter().flatten() {
        if value.parse::<f32>().is_err() {
            return Err(ActionEditError::InvalidCommand(
                "Bezier tangents must be numeric".to_string(),
            ));
        }
    }
    let action = action_span(script, action_id)?;
    let action_block = &script[action.start..action.end];
    let pose = pose_span(action_block, time_ms).ok_or_else(|| ActionEditError::PoseNotFound {
        action: action_id.to_string(),
        time_ms,
    })?;
    let pose_block = &action_block[pose.start..pose.end];
    let bone = child_spans(pose_block, "Bone")
        .into_iter()
        .find(|span| {
            opening_tag_end(pose_block, span.start)
                .and_then(|end| tag_attr(&pose_block[span.start..end], "id"))
                .as_deref()
                == Some(bone_id)
        })
        .ok_or_else(|| {
            ActionEditError::InvalidCommand(format!("Bone {bone_id} has no key at {time_ms}ms"))
        })?;
    let mut tag = pose_block[bone.start..bone.end].to_string();
    tag = upsert_tag_attr(&tag, "interpolation", Some(&interpolation));
    tag = upsert_tag_attr(&tag, "inTangent", in_tangent);
    tag = upsert_tag_attr(&tag, "outTangent", out_tangent);
    let next_pose = replace_span(pose_block, bone, &tag);
    let next_action = replace_span(action_block, pose, &next_pose);
    let output = replace_span(script, action, &next_action);
    parse_graph_script(&output)?;
    Ok(output)
}

fn add_pose(
    script: &str,
    action_id: &str,
    time_ms: u64,
    copy_from_ms: Option<u64>,
) -> Result<String, ActionEditError> {
    let graph = parse_graph_script(script)?;
    ensure_action_authored(&graph, action_id)?;
    let span = action_span(script, action_id)?;
    let block = &script[span.start..span.end];
    if pose_span(block, time_ms).is_some() {
        return Err(ActionEditError::InvalidCommand(format!(
            "Action {action_id} already has a pose at {time_ms}ms"
        )));
    }
    let body = if let Some(source_time) = copy_from_ms {
        let source =
            pose_span(block, source_time).ok_or_else(|| ActionEditError::PoseNotFound {
                action: action_id.to_string(),
                time_ms: source_time,
            })?;
        let source_block = &block[source.start..source.end];
        let tag_end = opening_tag_end(source_block, 0)
            .ok_or_else(|| ActionEditError::SourceSpanNotFound("Pose opening tag".to_string()))?;
        let next_tag = upsert_tag_attr(
            &source_block[..tag_end],
            "t",
            Some(&format_seconds(time_ms)),
        );
        format!("{}{}", next_tag, &source_block[tag_end..])
    } else {
        let close = block
            .rfind("</Action>")
            .ok_or_else(|| ActionEditError::SourceSpanNotFound("</Action>".to_string()))?;
        let action_indent = indent_before(block, close);
        format!(
            "{action_indent}  <Pose t=\"{}\">\n{action_indent}  </Pose>",
            format_seconds(time_ms)
        )
    };
    let close = block
        .rfind("</Action>")
        .ok_or_else(|| ActionEditError::SourceSpanNotFound("</Action>".to_string()))?;
    let next = format!("{}{}\n{}", &block[..close], body, &block[close..]);
    let output = replace_span(script, span, &next);
    parse_graph_script(&output)?;
    Ok(output)
}

fn remove_pose(script: &str, action_id: &str, time_ms: u64) -> Result<String, ActionEditError> {
    let graph = parse_graph_script(script)?;
    ensure_action_authored(&graph, action_id)?;
    let action = graph
        .actions
        .iter()
        .find(|action| action.id == action_id)
        .unwrap();
    if action.poses.len() <= 1 && action.iks.is_empty() {
        return Err(ActionEditError::InvalidCommand(
            "an authored Action must retain at least one Pose or IK".to_string(),
        ));
    }
    let span = action_span(script, action_id)?;
    let block = &script[span.start..span.end];
    let mut pose = pose_span(block, time_ms).ok_or_else(|| ActionEditError::PoseNotFound {
        action: action_id.to_string(),
        time_ms,
    })?;
    if block[pose.end..].starts_with('\n') {
        pose.end += 1;
    }
    let next = replace_span(block, pose, "");
    let output = replace_span(script, span, &next);
    parse_graph_script(&output)?;
    Ok(output)
}

fn move_pose(
    script: &str,
    action_id: &str,
    from_ms: u64,
    to_ms: u64,
) -> Result<String, ActionEditError> {
    let span = action_span(script, action_id)?;
    let block = &script[span.start..span.end];
    if pose_span(block, to_ms).is_some() {
        return Err(ActionEditError::InvalidCommand(format!(
            "Action {action_id} already has a pose at {to_ms}ms"
        )));
    }
    let pose = pose_span(block, from_ms).ok_or_else(|| ActionEditError::PoseNotFound {
        action: action_id.to_string(),
        time_ms: from_ms,
    })?;
    let tag_end = opening_tag_end(block, pose.start)
        .ok_or_else(|| ActionEditError::SourceSpanNotFound("Pose opening tag".to_string()))?;
    let replacement = upsert_tag_attr(
        &block[pose.start..tag_end],
        "t",
        Some(&format_seconds(to_ms)),
    );
    let next = replace_span(
        block,
        SourceSpan {
            start: pose.start,
            end: tag_end,
        },
        &replacement,
    );
    let output = replace_span(script, span, &next);
    parse_graph_script(&output)?;
    Ok(output)
}

fn render_contact(
    contact: &EditableActionContact,
    indent: &str,
) -> Result<String, ActionEditError> {
    if contact.id.trim().is_empty() || contact.effector.trim().is_empty() {
        return Err(ActionEditError::InvalidCommand(
            "Contact id and effector are required".to_string(),
        ));
    }
    Ok(format!(
        "{indent}<Contact id=\"{}\" effector=\"{}\" target=\"{}\" from=\"{:.4}\" to=\"{:.4}\" mode=\"{}\" weight=\"{}\" />",
        contact.id,
        contact.effector,
        contact.target,
        contact.from,
        contact.to,
        contact.mode,
        contact.weight
    ))
}

fn upsert_contact(
    script: &str,
    action_id: &str,
    contact: &EditableActionContact,
) -> Result<String, ActionEditError> {
    let span = action_span(script, action_id)?;
    let block = &script[span.start..span.end];
    let existing = child_spans(block, "Contact").into_iter().find(|span| {
        opening_tag_end(block, span.start)
            .and_then(|end| tag_attr(&block[span.start..end], "id"))
            .as_deref()
            == Some(contact.id.as_str())
    });
    let next = if let Some(contact_span) = existing {
        let indent = indent_before(block, contact_span.start);
        replace_span(block, contact_span, &render_contact(contact, &indent)?)
    } else {
        let close = block
            .rfind("</Action>")
            .ok_or_else(|| ActionEditError::SourceSpanNotFound("</Action>".to_string()))?;
        let indent = format!("{}  ", indent_before(block, close));
        format!(
            "{}{}\n{}",
            &block[..close],
            render_contact(contact, &indent)?,
            &block[close..]
        )
    };
    let output = replace_span(script, span, &next);
    parse_graph_script(&output)?;
    Ok(output)
}

fn remove_contact(
    script: &str,
    action_id: &str,
    contact_id: &str,
) -> Result<String, ActionEditError> {
    let span = action_span(script, action_id)?;
    let block = &script[span.start..span.end];
    let mut contact = child_spans(block, "Contact")
        .into_iter()
        .find(|span| {
            opening_tag_end(block, span.start)
                .and_then(|end| tag_attr(&block[span.start..end], "id"))
                .as_deref()
                == Some(contact_id)
        })
        .ok_or_else(|| ActionEditError::ContactNotFound {
            action: action_id.to_string(),
            contact: contact_id.to_string(),
        })?;
    if block[contact.end..].starts_with('\n') {
        contact.end += 1;
    }
    let output = replace_span(script, span, &replace_span(block, contact, ""));
    parse_graph_script(&output)?;
    Ok(output)
}

fn render_ik(ik: &EditableActionIk, indent: &str) -> Result<String, ActionEditError> {
    for value in [
        &ik.target_x,
        &ik.target_y,
        &ik.target_z,
        &ik.bend,
        &ik.weight,
        &ik.iterations,
    ] {
        if value.parse::<f32>().is_err() {
            return Err(ActionEditError::InvalidCommand(
                "IK numeric fields must be numeric".to_string(),
            ));
        }
    }
    Ok(format!(
        "{indent}<IK id=\"{}\" root=\"{}\" mid=\"{}\" end=\"{}\" targetX=\"{}\" targetY=\"{}\" targetZ=\"{}\" bend=\"{}\" weight=\"{}\" iterations=\"{}\" />",
        ik.id,
        ik.root,
        ik.mid,
        ik.end,
        ik.target_x,
        ik.target_y,
        ik.target_z,
        ik.bend,
        ik.weight,
        ik.iterations
    ))
}

fn upsert_ik(
    script: &str,
    action_id: &str,
    ik: &EditableActionIk,
) -> Result<String, ActionEditError> {
    let span = action_span(script, action_id)?;
    let block = &script[span.start..span.end];
    let existing = child_spans(block, "IK").into_iter().find(|candidate| {
        opening_tag_end(block, candidate.start)
            .and_then(|end| tag_attr(&block[candidate.start..end], "id"))
            .as_deref()
            == Some(ik.id.as_str())
    });
    let next = if let Some(existing) = existing {
        let indent = indent_before(block, existing.start);
        replace_span(block, existing, &render_ik(ik, &indent)?)
    } else {
        let close = block
            .rfind("</Action>")
            .ok_or_else(|| ActionEditError::SourceSpanNotFound("</Action>".to_string()))?;
        let indent = format!("{}  ", indent_before(block, close));
        format!(
            "{}{}\n{}",
            &block[..close],
            render_ik(ik, &indent)?,
            &block[close..]
        )
    };
    let output = replace_span(script, span, &next);
    parse_graph_script(&output)?;
    Ok(output)
}

fn remove_ik(script: &str, action_id: &str, ik_id: &str) -> Result<String, ActionEditError> {
    let span = action_span(script, action_id)?;
    let block = &script[span.start..span.end];
    let mut existing = child_spans(block, "IK")
        .into_iter()
        .find(|candidate| {
            opening_tag_end(block, candidate.start)
                .and_then(|end| tag_attr(&block[candidate.start..end], "id"))
                .as_deref()
                == Some(ik_id)
        })
        .ok_or_else(|| ActionEditError::InvalidCommand(format!("IK {ik_id} not found")))?;
    if block[existing.end..].starts_with('\n') {
        existing.end += 1;
    }
    let output = replace_span(script, span, &replace_span(block, existing, ""));
    parse_graph_script(&output)?;
    Ok(output)
}

fn mirror_bone_id(id: &str) -> String {
    if let Some(stem) = id.strip_suffix("_l") {
        format!("{stem}_r")
    } else if let Some(stem) = id.strip_suffix("_r") {
        format!("{stem}_l")
    } else if let Some(stem) = id.strip_suffix(".L") {
        format!("{stem}.R")
    } else if let Some(stem) = id.strip_suffix(".R") {
        format!("{stem}.L")
    } else {
        id.to_string()
    }
}

fn mirrored_value(channel: &str, value: &str) -> String {
    if matches!(
        channel,
        "side" | "twist" | "turn" | "rotationY" | "rotationZ" | "x"
    ) {
        if let Ok(number) = value.parse::<f32>() {
            return format_number(-number);
        }
    }
    value.to_string()
}

fn format_number(value: f32) -> String {
    let formatted = format!("{value:.4}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn mirror_pose(
    script: &str,
    action_id: &str,
    time_ms: u64,
    direction: Option<&str>,
) -> Result<String, ActionEditError> {
    let graph = parse_graph_script(script)?;
    ensure_action_authored(&graph, action_id)?;
    let action = graph
        .actions
        .iter()
        .find(|action| action.id == action_id)
        .unwrap();
    let pose = action
        .poses
        .iter()
        .find(|pose| (pose.t * 1000.0).round() as u64 == time_ms)
        .ok_or_else(|| ActionEditError::PoseNotFound {
            action: action_id.to_string(),
            time_ms,
        })?;
    let source = pose
        .bones
        .iter()
        .map(|bone| (bone.id.clone(), bone_channels(bone)))
        .collect::<HashMap<_, _>>();
    let mut output = script.to_string();
    for (bone_id, channels) in &source {
        let source_is_left = bone_id.ends_with("_l") || bone_id.ends_with(".L");
        let source_is_right = bone_id.ends_with("_r") || bone_id.ends_with(".R");
        if direction == Some("leftToRight") && !source_is_left {
            continue;
        }
        if direction == Some("rightToLeft") && !source_is_right {
            continue;
        }
        let target = mirror_bone_id(bone_id);
        for (channel, value) in channels {
            output = set_bone_channel(
                &output,
                action_id,
                time_ms,
                &target,
                channel,
                Some(&mirrored_value(channel, value)),
            )?;
        }
    }
    Ok(output)
}

fn create_action(
    script: &str,
    id: &str,
    skeleton: &str,
    duration_ms: u64,
) -> Result<String, ActionEditError> {
    let graph = parse_graph_script(script)?;
    if id.trim().is_empty() || duration_ms == 0 {
        return Err(ActionEditError::InvalidCommand(
            "Action id and positive duration are required".to_string(),
        ));
    }
    if graph.actions.iter().any(|action| action.id == id) {
        return Err(ActionEditError::InvalidCommand(format!(
            "Action {id} already exists"
        )));
    }
    let present = script
        .rfind("<Present")
        .ok_or_else(|| ActionEditError::SourceSpanNotFound("<Present>".to_string()))?;
    let indent = indent_before(script, present);
    let block = format!(
        "{indent}<Action id=\"{id}\" skeleton=\"{skeleton}\" duration=\"{}\">\n{indent}  <Pose t=\"0s\">\n{indent}  </Pose>\n{indent}</Action>\n\n",
        format_seconds(duration_ms)
    );
    let output = format!("{}{}{}", &script[..present], block, &script[present..]);
    parse_graph_script(&output)?;
    Ok(output)
}

fn duplicate_action(
    script: &str,
    action_id: &str,
    new_id: &str,
) -> Result<String, ActionEditError> {
    let graph = parse_graph_script(script)?;
    if graph.actions.iter().any(|action| action.id == new_id) {
        return Err(ActionEditError::InvalidCommand(format!(
            "Action {new_id} already exists"
        )));
    }
    let span = action_span(script, action_id)?;
    let block = &script[span.start..span.end];
    let tag_end = opening_tag_end(block, 0)
        .ok_or_else(|| ActionEditError::SourceSpanNotFound("Action opening tag".to_string()))?;
    let next_tag = upsert_tag_attr(&block[..tag_end], "id", Some(new_id));
    let duplicate = format!("{}{}", next_tag, &block[tag_end..]);
    let insert = span.end;
    let output = format!(
        "{}\n\n{}{}",
        &script[..insert],
        duplicate,
        &script[insert..]
    );
    parse_graph_script(&output)?;
    Ok(output)
}

fn set_action_metadata(
    script: &str,
    action_id: &str,
    skeleton: Option<&str>,
    duration_ms: Option<u64>,
) -> Result<String, ActionEditError> {
    let span = action_span(script, action_id)?;
    let block = &script[span.start..span.end];
    let tag_end = opening_tag_end(block, 0)
        .ok_or_else(|| ActionEditError::SourceSpanNotFound("Action opening tag".to_string()))?;
    let mut tag = block[..tag_end].to_string();
    if let Some(skeleton) = skeleton {
        tag = upsert_tag_attr(&tag, "skeleton", Some(skeleton));
    }
    if let Some(duration_ms) = duration_ms {
        if duration_ms == 0 {
            return Err(ActionEditError::InvalidCommand(
                "Action duration must be positive".to_string(),
            ));
        }
        tag = upsert_tag_attr(&tag, "duration", Some(&format_seconds(duration_ms)));
    }
    let next = format!("{}{}", tag, &block[tag_end..]);
    let output = replace_span(script, span, &next);
    parse_graph_script(&output)?;
    Ok(output)
}

fn apply_action_spans(script: &str) -> Vec<SourceSpan> {
    child_spans(script, "ApplyAction")
}

fn set_binding(
    script: &str,
    target: &str,
    action: &str,
    attribute: &str,
    value: Option<&str>,
) -> Result<String, ActionEditError> {
    let allowed = [
        "loop",
        "speed",
        "weight",
        "rootMotion",
        "ground",
        "destination",
        "face",
        "contactCorrection",
        "footLock",
        "groundOffset",
        "colliderProfile",
        "safeMargin",
        "floorSnap",
        "maxSlides",
        "sweepStep",
    ];
    if !allowed.contains(&attribute) {
        return Err(ActionEditError::InvalidCommand(format!(
            "unsupported ApplyAction attribute {attribute}"
        )));
    }
    let span = apply_action_spans(script)
        .into_iter()
        .find(|span| {
            let tag = &script[span.start..span.end];
            tag_attr(tag, "target").as_deref() == Some(target)
                && tag_attr(tag, "action").as_deref() == Some(action)
        })
        .ok_or_else(|| ActionEditError::BindingNotFound {
            target: target.to_string(),
            action: action.to_string(),
        })?;
    let replacement = upsert_tag_attr(&script[span.start..span.end], attribute, value);
    let output = replace_span(script, span, &replacement);
    parse_graph_script(&output)?;
    Ok(output)
}

/// Apply one typed Action editor command and validate the resulting graph.
pub fn apply_action_edit(
    script: &str,
    command: ActionEditCommand,
) -> Result<String, ActionEditError> {
    match command {
        ActionEditCommand::CreateAction {
            id,
            skeleton,
            duration_ms,
        } => create_action(script, &id, &skeleton, duration_ms),
        ActionEditCommand::DuplicateAction { action_id, new_id } => {
            duplicate_action(script, &action_id, &new_id)
        }
        ActionEditCommand::SetActionMetadata {
            action_id,
            skeleton,
            duration_ms,
        } => set_action_metadata(script, &action_id, skeleton.as_deref(), duration_ms),
        ActionEditCommand::AddPose {
            action_id,
            time_ms,
            copy_from_ms,
        } => add_pose(script, &action_id, time_ms, copy_from_ms),
        ActionEditCommand::RemovePose { action_id, time_ms } => {
            remove_pose(script, &action_id, time_ms)
        }
        ActionEditCommand::MovePose {
            action_id,
            from_ms,
            to_ms,
        } => move_pose(script, &action_id, from_ms, to_ms),
        ActionEditCommand::SetBoneChannel {
            action_id,
            time_ms,
            bone_id,
            channel,
            value,
        } => set_bone_channel(
            script,
            &action_id,
            time_ms,
            &bone_id,
            &channel,
            value.as_deref(),
        ),
        ActionEditCommand::SetBoneKeyMetadata {
            action_id,
            time_ms,
            bone_id,
            interpolation,
            in_tangent,
            out_tangent,
        } => set_bone_key_metadata(
            script,
            &action_id,
            time_ms,
            &bone_id,
            &interpolation,
            in_tangent.as_deref(),
            out_tangent.as_deref(),
        ),
        ActionEditCommand::MirrorPose {
            action_id,
            time_ms,
            direction,
        } => mirror_pose(script, &action_id, time_ms, direction.as_deref()),
        ActionEditCommand::UpsertContact { action_id, contact } => {
            upsert_contact(script, &action_id, &contact)
        }
        ActionEditCommand::RemoveContact {
            action_id,
            contact_id,
        } => remove_contact(script, &action_id, &contact_id),
        ActionEditCommand::UpsertIk { action_id, ik } => upsert_ik(script, &action_id, &ik),
        ActionEditCommand::RemoveIk { action_id, ik_id } => remove_ik(script, &action_id, &ik_id),
        ActionEditCommand::SetBinding {
            target,
            action,
            attribute,
            value,
        } => set_binding(script, &target, &action, &attribute, value.as_deref()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn script() -> &'static str {
        r#"<Graph fps={30} duration="2s" size={[320,240]}>
  <Action id="walk" skeleton="humanoid_v1" duration="1s">
    // keep this authored note
    <Pose t="0s">
      <Bone id="hips" y="0.01" />
      <Bone id="upper_leg_l" forward="25" side="2" />
    </Pose>
    <Pose t="0.5s">
      <Bone id="hips" y="-0.02" />
      <Bone id="upper_leg_r" forward="25" side="-2" />
    </Pose>
    <Contact id="left" effector="foot_l" target="ground" from="0" to="0.4" mode="lock" weight="1" />
  </Action>
  <ApplyAction target="hero" action="walk" loop="true" rootMotion="in_place" />
  <Scene id="main_scene">
    <Timeline>
      <Track id="track" space="world">
        <Sequence from="0s" duration="2s">
          <CompositeGroup id="world" space="3d">
          </CompositeGroup>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="main_scene" />
</Graph>"#
    }

    #[test]
    fn extracts_actions_bindings_and_diagnostics() {
        let document = extract_editable_action_document(script()).expect("extract Action document");
        assert_eq!(document.actions[0].poses.len(), 2);
        assert_eq!(
            document.actions[0].poses[0].bones[1].channels["forward"],
            "25"
        );
        assert_eq!(
            document.bindings[0].root_motion.as_deref(),
            Some("in_place")
        );
    }

    #[test]
    fn patches_one_bone_channel_without_reformatting_action_comments() {
        let output = apply_action_edit(
            script(),
            ActionEditCommand::SetBoneChannel {
                action_id: "walk".to_string(),
                time_ms: 0,
                bone_id: "upper_leg_l".to_string(),
                channel: "forward".to_string(),
                value: Some("31".to_string()),
            },
        )
        .expect("patch Bone channel");
        assert!(output.contains("// keep this authored note"));
        assert!(output.contains("forward=\"31\" side=\"2\""));
    }

    #[test]
    fn adds_moves_and_removes_a_pose() {
        let added = apply_action_edit(
            script(),
            ActionEditCommand::AddPose {
                action_id: "walk".to_string(),
                time_ms: 250,
                copy_from_ms: Some(0),
            },
        )
        .expect("add Pose");
        let moved = apply_action_edit(
            &added,
            ActionEditCommand::MovePose {
                action_id: "walk".to_string(),
                from_ms: 250,
                to_ms: 300,
            },
        )
        .expect("move Pose");
        let removed = apply_action_edit(
            &moved,
            ActionEditCommand::RemovePose {
                action_id: "walk".to_string(),
                time_ms: 300,
            },
        )
        .expect("remove Pose");
        assert_eq!(
            extract_editable_action_document(&removed).unwrap().actions[0]
                .poses
                .len(),
            2
        );
    }

    #[test]
    fn mirrors_left_pose_channels_to_right() {
        let output = apply_action_edit(
            script(),
            ActionEditCommand::MirrorPose {
                action_id: "walk".to_string(),
                time_ms: 0,
                direction: Some("leftToRight".to_string()),
            },
        )
        .expect("mirror Pose");
        assert!(output.contains("id=\"upper_leg_r\""));
        assert!(output.contains("forward=\"25\""));
        assert!(output.contains("side=\"-2\""));
    }

    #[test]
    fn updates_contacts_and_apply_action_settings() {
        let contact = EditableActionContact {
            id: "left".to_string(),
            effector: "foot_l".to_string(),
            target: "ground".to_string(),
            from: 0.05,
            to: 0.45,
            mode: "lock".to_string(),
            weight: "0.9".to_string(),
        };
        let output = apply_action_edit(
            script(),
            ActionEditCommand::UpsertContact {
                action_id: "walk".to_string(),
                contact,
            },
        )
        .expect("update Contact");
        let output = apply_action_edit(
            &output,
            ActionEditCommand::SetBinding {
                target: "hero".to_string(),
                action: "walk".to_string(),
                attribute: "footLock".to_string(),
                value: Some("auto".to_string()),
            },
        )
        .expect("update ApplyAction");
        let output = apply_action_edit(
            &output,
            ActionEditCommand::SetBinding {
                target: "hero".to_string(),
                action: "walk".to_string(),
                attribute: "face".to_string(),
                value: Some("exit".to_string()),
            },
        )
        .expect("update ApplyAction facing");
        assert!(output.contains("from=\"0.0500\" to=\"0.4500\""));
        assert!(output.contains("footLock=\"auto\""));
        assert!(output.contains("face=\"exit\""));
    }

    #[test]
    fn deserializes_browser_camel_case_commands() {
        let command = serde_json::from_str::<ActionEditCommand>(
            r#"{"type":"setBoneChannel","actionId":"walk","timeMs":0,"boneId":"hips","channel":"y","value":"0.02"}"#,
        )
        .expect("browser command should deserialize");
        assert!(matches!(
            command,
            ActionEditCommand::SetBoneChannel { action_id, time_ms: 0, bone_id, .. }
                if action_id == "walk" && bone_id == "hips"
        ));
    }

    #[test]
    fn key_interpolation_and_tangents_round_trip_through_editor_document() {
        let output = apply_action_edit(
            script(),
            ActionEditCommand::SetBoneKeyMetadata {
                action_id: "walk".to_string(),
                time_ms: 0,
                bone_id: "hips".to_string(),
                interpolation: "bezier".to_string(),
                in_tangent: Some("-0.25".to_string()),
                out_tangent: Some("0.5".to_string()),
            },
        )
        .expect("set key metadata");
        assert!(output.contains("interpolation=\"bezier\""));
        let document = extract_editable_action_document(&output).expect("extract metadata");
        let bone = &document.actions[0].poses[0].bones[0];
        assert_eq!(bone.interpolation, "bezier");
        assert_eq!(bone.in_tangent, "-0.25");
        assert_eq!(bone.out_tangent, "0.5");
    }

    #[test]
    fn ik_and_contact_can_be_added_and_removed_independently() {
        let ik = EditableActionIk {
            id: "foot_l_ik".into(),
            root: "upper_leg_l".into(),
            mid: "lower_leg_l".into(),
            end: "foot_l".into(),
            target_x: "0".into(),
            target_y: "0".into(),
            target_z: "0".into(),
            bend: "1".into(),
            weight: "1".into(),
            iterations: "8".into(),
        };
        let with_ik = apply_action_edit(
            script(),
            ActionEditCommand::UpsertIk {
                action_id: "walk".into(),
                ik,
            },
        )
        .unwrap();
        let without_contact = apply_action_edit(
            &with_ik,
            ActionEditCommand::RemoveContact {
                action_id: "walk".into(),
                contact_id: "left".into(),
            },
        )
        .unwrap();
        let clean = apply_action_edit(
            &without_contact,
            ActionEditCommand::RemoveIk {
                action_id: "walk".into(),
                ik_id: "foot_l_ik".into(),
            },
        )
        .unwrap();
        assert!(!clean.contains("<Contact"));
        assert!(!clean.contains("<IK"));
    }
}
