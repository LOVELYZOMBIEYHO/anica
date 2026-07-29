// =========================================
// =========================================
// crates/motionloom/src/scene/dsl.rs

use crate::dsl::{
    attr_value, collect_self_closing_block, collect_tag_block, find_matching_close_tag,
    is_self_closing_tag, parse_duration_ms, parse_signed_time_ms, parse_size, parse_time_seconds,
    required_attr_value, required_attr_value_any, starts_open_tag, strip_wrappers,
};
use crate::error::GraphParseError;
use crate::scene::model::*;
use crate::scene::text::{
    TextAnimatorNode, TextEffectNode, TextGlowEffectNode, TextLayoutNode, TextNode,
    TextSelectorKind, TextStyleOverrideNode, TextTransformNode,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProfileNode {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub model: Option<String>,
    pub preset: String,
    #[serde(default)]
    pub retarget: Option<ModelProfileRetargetNode>,
    #[serde(default)]
    pub bone_axis_map: Option<ModelProfileBoneAxisMapNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProfileRetargetNode {
    pub preset: String,
    #[serde(default)]
    pub maps: Vec<ModelProfileRetargetMapNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProfileRetargetMapNode {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProfileBoneAxisMapNode {
    #[serde(default)]
    pub axes: Vec<ModelProfileBoneAxisNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProfileBoneAxisNode {
    pub bone: String,
    #[serde(default)]
    pub forward: Option<String>,
    #[serde(default)]
    pub side: Option<String>,
    #[serde(default)]
    pub twist: Option<String>,
    #[serde(default)]
    pub bend: Option<String>,
    #[serde(default)]
    pub turn: Option<String>,
    #[serde(default)]
    pub rest_forward: Option<String>,
    #[serde(default)]
    pub rest_side: Option<String>,
    #[serde(default)]
    pub rest_twist: Option<String>,
    #[serde(default)]
    pub rest_bend: Option<String>,
    #[serde(default)]
    pub rest_turn: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonNode {
    pub id: String,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub height: Option<String>,
    #[serde(default = "default_skeleton_facing")]
    pub facing: String,
    #[serde(default)]
    pub symmetry_axis: Option<String>,
    #[serde(default = "default_skeleton_validation")]
    pub validation: String,
    #[serde(default)]
    pub auto_correct: Option<String>,
    #[serde(default)]
    pub overlay: bool,
    pub bones: Vec<SkeletonBoneNode>,
    #[serde(default)]
    pub landmarks: Vec<SkeletonLandmarkNode>,
    #[serde(default)]
    pub measures: Vec<SkeletonMeasureNode>,
    #[serde(default)]
    pub ratios: Vec<SkeletonRatioNode>,
    #[serde(default)]
    pub regions: Vec<SkeletonRegionNode>,
    #[serde(default)]
    pub constraints: Vec<SkeletonConstraintNode>,
    #[serde(default)]
    pub guides: Vec<SkeletonGuideNode>,
    #[serde(default)]
    pub controls: Vec<SkeletonControlNode>,
}

fn default_skeleton_facing() -> String {
    "front".to_string()
}

fn default_skeleton_validation() -> String {
    "warn".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonBoneNode {
    pub id: String,
    pub parent: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub side: Option<String>,
    pub x: String,
    pub y: String,
    pub rotation: String,
    pub scale: String,
    pub length: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonLandmarkNode {
    pub id: String,
    pub bone: String,
    pub offset: (String, String),
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonMeasureNode {
    pub id: String,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonRatioNode {
    pub measure: String,
    pub relative_to: String,
    pub value: String,
    #[serde(default)]
    pub tolerance: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonRegionNode {
    pub id: String,
    pub role: String,
    pub kind: String,
    #[serde(default)]
    pub center: Option<String>,
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
    #[serde(default)]
    pub radius_x: Option<String>,
    #[serde(default)]
    pub radius_y: Option<String>,
    #[serde(default)]
    pub width: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonConstraintNode {
    pub kind: String,
    #[serde(default)]
    pub left: Option<String>,
    #[serde(default)]
    pub right: Option<String>,
    #[serde(default)]
    pub axis: Option<String>,
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
    #[serde(default)]
    pub bone: Option<String>,
    #[serde(default)]
    pub relative_to: Option<String>,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub min: Option<String>,
    #[serde(default)]
    pub max: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonGuideNode {
    pub id: String,
    pub kind: String,
    pub through: String,
    pub angle: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkeletonControlNode {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub targets: Vec<String>,
    #[serde(default)]
    pub chain_length: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionNode {
    pub id: String,
    pub skeleton: Option<String>,
    pub duration_ms: u64,
    pub poses: Vec<ActionPoseNode>,
    #[serde(default)]
    pub iks: Vec<ActionIkNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionPoseNode {
    pub t: f32,
    pub bones: Vec<ActionBoneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionBoneNode {
    pub id: String,
    pub x: Option<String>,
    pub y: Option<String>,
    pub rotation: Option<String>,
    pub scale: Option<String>,
    pub opacity: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionIkNode {
    pub root: String,
    pub mid: String,
    pub end: String,
    #[serde(default)]
    pub chain: Vec<String>,
    pub target_x: String,
    pub target_y: String,
    pub bend: String,
    pub weight: String,
    pub iterations: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplyActionNode {
    pub target: String,
    pub action: String,
    pub at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BackgroundNode {
    pub id: Option<String>,
    pub color: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageNode {
    pub id: Option<String>,
    #[serde(default)]
    pub material: Option<String>,
    pub src: String,
    pub x: String,
    pub y: String,
    pub scale: String,
    pub opacity: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SvgNode {
    pub id: Option<String>,
    #[serde(default)]
    pub material: Option<String>,
    pub src: String,
    pub x: String,
    pub y: String,
    pub scale: String,
    pub opacity: String,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct BrushParseContext {
    brushes: HashMap<String, BrushDef>,
    inherited_brush: Option<String>,
}

impl BrushParseContext {
    fn define_brushes(&mut self, brushes: &[BrushDef]) {
        for brush in brushes {
            self.brushes.insert(brush.id.clone(), brush.clone());
        }
    }

    fn with_inherited_brush(&self, brush: Option<String>) -> Self {
        let mut next = self.clone();
        if let Some(brush) = brush {
            next.inherited_brush = Some(brush);
        }
        next
    }

    fn validate_brush_ref(&self, brush: Option<&str>, line: usize) -> Result<(), GraphParseError> {
        let Some(brush) = brush else {
            return Ok(());
        };
        if self.brushes.contains_key(brush) {
            return Ok(());
        }
        Err(GraphParseError {
            line,
            message: format!("brush reference not found: {brush}"),
        })
    }

    fn brush_for_path<'a>(
        &'a self,
        block: &str,
        line: usize,
    ) -> Result<(Option<String>, Option<&'a BrushDef>), GraphParseError> {
        let brush_id = attr_value(block, "brush")
            .map(|v| strip_wrappers(&v).to_string())
            .or_else(|| self.inherited_brush.clone());
        self.validate_brush_ref(brush_id.as_deref(), line)?;
        let brush = brush_id.as_ref().and_then(|id| self.brushes.get(id));
        Ok((brush_id, brush))
    }
}

pub(crate) fn validate_scene_camera_structure(
    scenes: &[SceneRootNode],
    scene_nodes: &[SceneNode],
    line: usize,
) -> Result<(), GraphParseError> {
    for scene in scenes {
        validate_scene_camera_structure_in_nodes(&scene.children, false, line)?;
    }
    validate_scene_camera_structure_in_nodes(scene_nodes, false, line)
}

fn validate_scene_camera_structure_in_nodes(
    nodes: &[SceneNode],
    in_camera_track: bool,
    line: usize,
) -> Result<(), GraphParseError> {
    for node in nodes {
        match node {
            SceneNode::Defs(defs) => {
                for mask in &defs.masks {
                    validate_scene_camera_structure_in_nodes(&mask.children, false, line)?;
                }
                for precompose in &defs.precomposes {
                    validate_scene_camera_structure_in_nodes(&precompose.children, false, line)?;
                }
                for component in &defs.components {
                    validate_scene_camera_structure_in_nodes(&component.children, false, line)?;
                }
            }
            SceneNode::Timeline(timeline) => {
                validate_scene_camera_structure_in_nodes(&timeline.children, false, line)?;
            }
            SceneNode::Track(track) => {
                if is_scene_camera_track(track) {
                    validate_scene_camera_track(track, line)?;
                } else {
                    validate_scene_camera_structure_in_nodes(&track.children, false, line)?;
                }
            }
            SceneNode::Sequence(sequence) => {
                validate_scene_camera_structure_in_nodes(
                    &sequence.children,
                    in_camera_track,
                    line,
                )?;
            }
            SceneNode::Chain(chain) => {
                validate_scene_camera_structure_in_nodes(&chain.children, in_camera_track, line)?;
            }
            SceneNode::Camera(_) if !in_camera_track => {
                return Err(GraphParseError {
                    line,
                    message: "<Camera> must be inside <Track role=\"camera\"><Sequence><Camera ... /></Sequence></Track>. Put visual content in <Track space=\"world\"> or <Track space=\"screen\">.".to_string(),
                });
            }
            SceneNode::Camera(camera) => {
                if !camera.children.is_empty() {
                    return Err(GraphParseError {
                        line,
                        message: "<Scene> Camera must be self-closing and cannot contain visual children.".to_string(),
                    });
                }
            }
            SceneNode::Group(group) => {
                validate_scene_camera_structure_in_nodes(&group.children, false, line)?;
            }
            SceneNode::Layer(layer) => {
                validate_scene_camera_structure_in_nodes(&layer.children, false, line)?;
            }
            SceneNode::Character(character) => {
                validate_scene_camera_structure_in_nodes(&character.children, false, line)?;
            }
            SceneNode::Part(part) => {
                validate_scene_camera_structure_in_nodes(&part.children, false, line)?;
            }
            SceneNode::Repeat(repeat) => {
                validate_scene_camera_structure_in_nodes(&repeat.children, false, line)?;
            }
            SceneNode::Mask(mask) => {
                validate_scene_camera_structure_in_nodes(&mask.children, false, line)?;
            }
            SceneNode::Precompose(precompose) => {
                validate_scene_camera_structure_in_nodes(&precompose.children, false, line)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_scene_camera_track(track: &SceneTrackNode, line: usize) -> Result<(), GraphParseError> {
    if track.children.is_empty() {
        return Err(GraphParseError {
            line,
            message: "<Track role=\"camera\"> requires at least one <Sequence> containing a single <Camera />.".to_string(),
        });
    }
    for child in &track.children {
        let SceneNode::Sequence(sequence) = child else {
            return Err(GraphParseError {
                line,
                message: "<Track role=\"camera\"> only accepts <Sequence> children. Each sequence must contain a single self-closing <Camera />.".to_string(),
            });
        };
        if sequence.children.len() != 1 {
            return Err(GraphParseError {
                line,
                message: "<Track role=\"camera\"><Sequence> must contain exactly one self-closing <Camera />.".to_string(),
            });
        }
        let SceneNode::Camera(camera) = &sequence.children[0] else {
            return Err(GraphParseError {
                line,
                message: "<Track role=\"camera\"><Sequence> must contain exactly one self-closing <Camera />.".to_string(),
            });
        };
        if !camera.children.is_empty() {
            return Err(GraphParseError {
                line,
                message: "<Scene> Camera must be self-closing and cannot contain visual children."
                    .to_string(),
            });
        }
    }
    Ok(())
}

fn is_scene_camera_track(track: &SceneTrackNode) -> bool {
    track
        .role
        .as_deref()
        .is_some_and(|role| role.eq_ignore_ascii_case("camera"))
}

pub(crate) fn validate_scene_model_profile_refs(
    scenes: &[SceneRootNode],
    scene_nodes: &[SceneNode],
    model_profile_ids: &HashSet<String>,
    line: usize,
) -> Result<(), GraphParseError> {
    for scene in scenes {
        validate_scene_model_profile_refs_in_nodes(&scene.children, model_profile_ids, line)?;
    }
    validate_scene_model_profile_refs_in_nodes(scene_nodes, model_profile_ids, line)
}

fn validate_scene_model_profile_refs_in_nodes(
    nodes: &[SceneNode],
    model_profile_ids: &HashSet<String>,
    line: usize,
) -> Result<(), GraphParseError> {
    for node in nodes {
        match node {
            SceneNode::Defs(defs) => {
                for mask in &defs.masks {
                    validate_scene_model_profile_refs_in_nodes(
                        &mask.children,
                        model_profile_ids,
                        line,
                    )?;
                }
                for precompose in &defs.precomposes {
                    validate_scene_model_profile_refs_in_nodes(
                        &precompose.children,
                        model_profile_ids,
                        line,
                    )?;
                }
                for component in &defs.components {
                    validate_scene_model_profile_refs_in_nodes(
                        &component.children,
                        model_profile_ids,
                        line,
                    )?;
                }
            }
            SceneNode::Timeline(timeline) => {
                validate_scene_model_profile_refs_in_nodes(
                    &timeline.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Track(track) => {
                validate_scene_model_profile_refs_in_nodes(
                    &track.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Sequence(sequence) => {
                validate_scene_model_profile_refs_in_nodes(
                    &sequence.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Chain(chain) => {
                validate_scene_model_profile_refs_in_nodes(
                    &chain.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Character(character) => {
                if let Some(model_profile) = character.model_profile.as_deref()
                    && !model_profile_ids.contains(model_profile)
                {
                    return Err(GraphParseError {
                        line,
                        message: format!("Character modelProfile not found: {model_profile}"),
                    });
                }
                validate_scene_model_profile_refs_in_nodes(
                    &character.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Group(group) => {
                validate_scene_model_profile_refs_in_nodes(
                    &group.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Part(part) => {
                validate_scene_model_profile_refs_in_nodes(
                    &part.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Repeat(repeat) => {
                validate_scene_model_profile_refs_in_nodes(
                    &repeat.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Mask(mask) => {
                validate_scene_model_profile_refs_in_nodes(
                    &mask.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Precompose(precompose) => {
                validate_scene_model_profile_refs_in_nodes(
                    &precompose.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Layer(layer) => {
                validate_scene_model_profile_refs_in_nodes(
                    &layer.children,
                    model_profile_ids,
                    line,
                )?;
            }
            SceneNode::Camera(camera) => {
                validate_scene_model_profile_refs_in_nodes(
                    &camera.children,
                    model_profile_ids,
                    line,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn parse_scene_root_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(SceneRootNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Scene")?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let size = attr_value(&open_tag, "size")
        .as_deref()
        .map(|v| parse_size(v, start + 1, "size"))
        .transpose()?;
    let mut child_ctx = brush_ctx.clone();
    let mut children = parse_scene_root_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    resolve_puppet_targets(&mut children)?;
    Ok((SceneRootNode { id, size, children }, close_ix))
}

// PuppetWarp is authored beside its target so the DSL remains easy to edit.
// The renderer's stable Puppet AST owns visual children, so resolve the target
// into that internal shape after parsing without changing the source DSL.
fn resolve_puppet_targets(nodes: &mut Vec<SceneNode>) -> Result<(), GraphParseError> {
    resolve_puppet_targets_in_scope(nodes, false)
}

fn resolve_puppet_targets_in_scope(
    nodes: &mut Vec<SceneNode>,
    is_layer_scope: bool,
) -> Result<(), GraphParseError> {
    for node in nodes.iter_mut() {
        let (children, child_is_layer_scope) = match node {
            SceneNode::Timeline(node) => (Some(&mut node.children), false),
            SceneNode::Track(node) => (Some(&mut node.children), false),
            SceneNode::Sequence(node) => (Some(&mut node.children), false),
            SceneNode::Chain(node) => (Some(&mut node.children), false),
            SceneNode::Group(node) => (Some(&mut node.children), false),
            SceneNode::Part(node) => (Some(&mut node.children), false),
            SceneNode::Repeat(node) => (Some(&mut node.children), false),
            SceneNode::Mask(node) => (Some(&mut node.children), false),
            SceneNode::Precompose(node) => (Some(&mut node.children), false),
            SceneNode::Layer(node) => (Some(&mut node.children), true),
            SceneNode::Camera(node) => (Some(&mut node.children), false),
            SceneNode::Character(node) => (Some(&mut node.children), false),
            SceneNode::Puppet(node) => (Some(&mut node.children), false),
            _ => (None, false),
        };
        if let Some(children) = children {
            resolve_puppet_targets_in_scope(children, child_is_layer_scope)?;
        }
    }

    resolve_group_puppet_targets(nodes);
    if is_layer_scope {
        resolve_layer_puppet_targets(nodes)?;
    } else if nodes.iter().any(is_unresolved_layer_target_puppet) {
        return Err(GraphParseError {
            line: 1,
            message: "PuppetWarp target=\"@layer\" must be a direct child of <Layer>.".to_string(),
        });
    }
    Ok(())
}

// Existing Group-id targets retain their original sibling-binding behavior.
// Reserved selectors are handled separately so Group mode remains unchanged.
fn resolve_group_puppet_targets(nodes: &mut Vec<SceneNode>) {
    let mut target_bindings = std::collections::HashMap::<usize, (usize, PuppetNode)>::new();
    let mut consumed_puppets = std::collections::HashSet::<usize>::new();
    for (puppet_index, node) in nodes.iter().enumerate() {
        let SceneNode::Puppet(puppet) = node else {
            continue;
        };
        let Some(target) = puppet
            .target
            .as_deref()
            .filter(|target| !target.trim().is_empty())
        else {
            continue;
        };
        if target.starts_with('@') {
            continue;
        }
        if puppet
            .children
            .iter()
            .any(|child| scene_node_id(child) == Some(target))
        {
            continue;
        }
        if let Some((target_index, _)) = nodes
            .iter()
            .enumerate()
            .find(|(index, child)| *index != puppet_index && scene_node_id(child) == Some(target))
        {
            target_bindings.insert(target_index, (puppet_index, puppet.clone()));
            consumed_puppets.insert(puppet_index);
        }
    }
    if target_bindings.is_empty() {
        return;
    }

    let original = std::mem::take(nodes);
    for (index, node) in original.into_iter().enumerate() {
        if consumed_puppets.contains(&index) {
            continue;
        }
        if let Some((_, mut puppet)) = target_bindings.remove(&index) {
            puppet.children.insert(0, node);
            nodes.push(SceneNode::Puppet(puppet));
        } else {
            nodes.push(node);
        }
    }
}

// Universal Layer mode captures all earlier siblings into one Puppet surface.
// Moving those nodes rather than cloning them prevents duplicate bind-pose art.
fn resolve_layer_puppet_targets(nodes: &mut Vec<SceneNode>) -> Result<(), GraphParseError> {
    if !nodes.iter().any(is_layer_target_puppet) {
        return Ok(());
    }

    let original = std::mem::take(nodes);
    let mut resolved = Vec::<SceneNode>::with_capacity(original.len());
    for node in original {
        let SceneNode::Puppet(mut puppet) = node else {
            resolved.push(node);
            continue;
        };
        if !puppet
            .target
            .as_deref()
            .is_some_and(|target| target.eq_ignore_ascii_case("@layer"))
        {
            resolved.push(SceneNode::Puppet(puppet));
            continue;
        }
        if puppet.children.iter().any(is_puppet_visual_child) {
            resolved.push(SceneNode::Puppet(puppet));
            continue;
        }
        if resolved.is_empty() {
            return Err(GraphParseError {
                line: 1,
                message: "PuppetWarp target=\"@layer\" found no drawable nodes before it."
                    .to_string(),
            });
        }

        let captured = std::mem::take(&mut resolved);
        puppet.children.splice(0..0, captured);
        resolved.push(SceneNode::Puppet(puppet));
    }
    *nodes = resolved;
    Ok(())
}

fn is_puppet_visual_child(node: &SceneNode) -> bool {
    !matches!(
        node,
        SceneNode::Pin(_)
            | SceneNode::LimbEnvelope(_)
            | SceneNode::LimbRegion(_)
            | SceneNode::MeshTopology(_)
            | SceneNode::Vertex(_)
            | SceneNode::Triangle(_)
            | SceneNode::Edge(_)
            | SceneNode::Region(_)
    )
}

fn is_layer_target_puppet(node: &SceneNode) -> bool {
    matches!(
        node,
        SceneNode::Puppet(puppet)
            if puppet
                .target
                .as_deref()
                .is_some_and(|target| target.eq_ignore_ascii_case("@layer"))
    )
}

// A later universal rig wraps earlier rigs, so subsequent lowering passes must
// accept the already captured internal Puppet tree while rejecting raw nesting.
fn is_unresolved_layer_target_puppet(node: &SceneNode) -> bool {
    matches!(
        node,
        SceneNode::Puppet(puppet)
            if puppet
                .target
                .as_deref()
                .is_some_and(|target| target.eq_ignore_ascii_case("@layer"))
                && !puppet.children.iter().any(is_puppet_visual_child)
    )
}

/// Parametric Components are lowered after the first scene parse pass. Resolve
/// PuppetWarp targets again so a reusable deformation preset can bind artwork
/// authored beside its `<Use>` instance.
pub(crate) fn resolve_lowered_puppet_targets(
    scene_nodes: &mut Vec<SceneNode>,
    scenes: &mut [SceneRootNode],
) -> Result<(), GraphParseError> {
    resolve_puppet_targets(scene_nodes)?;
    for scene in scenes {
        resolve_puppet_targets(&mut scene.children)?;
    }
    Ok(())
}

fn scene_node_id(node: &SceneNode) -> Option<&str> {
    match node {
        SceneNode::Text(node) => node.id.as_deref(),
        SceneNode::Image(node) => node.id.as_deref(),
        SceneNode::Svg(node) => node.id.as_deref(),
        SceneNode::Rect(node) => node.id.as_deref(),
        SceneNode::Circle(node) => node.id.as_deref(),
        SceneNode::Ellipse(node) => node.id.as_deref(),
        SceneNode::Line(node) => node.id.as_deref(),
        SceneNode::Polyline(node) => node.id.as_deref(),
        SceneNode::Path(node) => node.id.as_deref(),
        SceneNode::Group(node) => node.id.as_deref(),
        SceneNode::Part(node) => node.id.as_deref(),
        SceneNode::Repeat(node) => node.id.as_deref(),
        SceneNode::Mask(node) => node.id.as_deref(),
        SceneNode::Precompose(node) => Some(node.id.as_str()),
        SceneNode::Layer(node) => node.id.as_deref(),
        SceneNode::Camera(node) => node.id.as_deref(),
        SceneNode::Character(node) => node.id.as_deref(),
        SceneNode::Puppet(node) => node.id.as_deref(),
        _ => None,
    }
}

fn parse_scene_root_nodes(
    lines: &[&str],
    start: usize,
    end: usize,
    brush_ctx: &mut BrushParseContext,
) -> Result<Vec<SceneNode>, GraphParseError> {
    let mut nodes = Vec::<SceneNode>::new();
    let mut i = start;
    while i < end {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with('{')
            || line.starts_with("<!--")
        {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Defs") {
            let (defs, end_ix) = parse_defs_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Defs(defs));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Timeline") {
            let (timeline, end_ix) = parse_timeline_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Timeline(timeline));
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!(
                "<Scene> root only accepts <Defs> and <Timeline>. Visual nodes must be wrapped in <Timeline><Track><Sequence>..., got: {line}"
            ),
        });
    }
    Ok(nodes)
}

pub(crate) fn parse_model_profile_block(
    lines: &[&str],
    start: usize,
) -> Result<(ModelProfileNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Ok((
            parse_model_profile_node(&open_tag, None, None, start + 1)?,
            open_end_ix,
        ));
    }

    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "ModelProfile")?;
    let mut retarget = None;
    let mut bone_axis_map = None;
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with('{')
            || line.starts_with("<!--")
        {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Retarget") {
            let (node, end_ix) = parse_model_profile_retarget_block(lines, i)?;
            retarget = Some(node);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "BoneAxisMap") {
            let (node, end_ix) = parse_model_profile_bone_axis_map_block(lines, i)?;
            bone_axis_map = Some(node);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!(
                "<ModelProfile> only accepts <Retarget> or <BoneAxisMap> children, got: {line}"
            ),
        });
    }

    Ok((
        parse_model_profile_node(&open_tag, retarget, bone_axis_map, start + 1)?,
        close_ix,
    ))
}

fn parse_model_profile_node(
    block: &str,
    retarget: Option<ModelProfileRetargetNode>,
    bone_axis_map: Option<ModelProfileBoneAxisMapNode>,
    line: usize,
) -> Result<ModelProfileNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let kind = attr_value(block, "kind")
        .map(|v| strip_wrappers(&v).to_ascii_lowercase())
        .unwrap_or_else(|| "2d".to_string());
    if !matches!(kind.as_str(), "2d" | "3d") {
        return Err(GraphParseError {
            line,
            message: format!("ModelProfile {id} kind must be \"2d\" or \"3d\", got: {kind}"),
        });
    }
    let model = attr_value(block, "model")
        .or_else(|| attr_value(block, "src"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.is_empty());
    let preset = attr_value(block, "preset")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "humanoid_v1".to_string());

    Ok(ModelProfileNode {
        id,
        kind,
        model,
        preset,
        retarget,
        bone_axis_map,
    })
}

fn parse_model_profile_retarget_block(
    lines: &[&str],
    start: usize,
) -> Result<(ModelProfileRetargetNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let preset = attr_value(&open_tag, "preset")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "humanoid_v1".to_string());
    if is_self_closing_tag(&open_tag) {
        return Ok((
            ModelProfileRetargetNode {
                preset,
                maps: Vec::new(),
            },
            open_end_ix,
        ));
    }

    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Retarget")?;
    let mut maps = Vec::<ModelProfileRetargetMapNode>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Map") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            maps.push(ModelProfileRetargetMapNode {
                from: strip_wrappers(&required_attr_value(&tag, "from", i + 1)?).to_string(),
                to: strip_wrappers(&required_attr_value(&tag, "to", i + 1)?).to_string(),
            });
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<Retarget> only accepts <Map /> children, got: {line}"),
        });
    }

    Ok((ModelProfileRetargetNode { preset, maps }, close_ix))
}

fn parse_model_profile_bone_axis_map_block(
    lines: &[&str],
    start: usize,
) -> Result<(ModelProfileBoneAxisMapNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Ok((
            ModelProfileBoneAxisMapNode { axes: Vec::new() },
            open_end_ix,
        ));
    }

    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "BoneAxisMap")?;
    let mut axes = Vec::<ModelProfileBoneAxisNode>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Axis") || starts_open_tag(line, "Bone") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            axes.push(parse_model_profile_bone_axis_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!(
                "<BoneAxisMap> only accepts <Axis /> or <Bone /> children, got: {line}"
            ),
        });
    }

    Ok((ModelProfileBoneAxisMapNode { axes }, close_ix))
}

fn parse_model_profile_bone_axis_node(
    block: &str,
    line: usize,
) -> Result<ModelProfileBoneAxisNode, GraphParseError> {
    let bone = strip_wrappers(&required_attr_value_any(block, &["bone", "id"], line)?).to_string();
    let attr = |keys: &[&str]| {
        keys.iter()
            .find_map(|key| attr_value(block, key))
            .map(|v| strip_wrappers(&v).to_string())
    };

    Ok(ModelProfileBoneAxisNode {
        bone,
        forward: attr(&["forward"]),
        side: attr(&["side"]),
        twist: attr(&["twist"]),
        bend: attr(&["bend"]),
        turn: attr(&["turn"]),
        rest_forward: attr(&["restForward", "rest_forward"]),
        rest_side: attr(&["restSide", "rest_side"]),
        rest_twist: attr(&["restTwist", "rest_twist"]),
        rest_bend: attr(&["restBend", "rest_bend"]),
        rest_turn: attr(&["restTurn", "rest_turn"]),
    })
}

pub(crate) fn parse_skeleton_block(
    lines: &[&str],
    start: usize,
) -> Result<(SkeletonNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Skeleton")?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let profile = skeleton_optional_attr(&open_tag, &["profile"]);
    let height = skeleton_optional_attr(&open_tag, &["height"]);
    let facing =
        skeleton_optional_attr(&open_tag, &["facing"]).unwrap_or_else(default_skeleton_facing);
    let symmetry_axis = skeleton_optional_attr(&open_tag, &["symmetryAxis", "symmetry_axis"]);
    let validation = skeleton_optional_attr(&open_tag, &["validation"])
        .unwrap_or_else(default_skeleton_validation);
    let auto_correct = skeleton_optional_attr(&open_tag, &["autoCorrect", "auto_correct"]);
    let overlay = skeleton_optional_attr(&open_tag, &["overlay"])
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "true" | "1" | "yes"))
        .unwrap_or(false);
    let mut bones = Vec::<SkeletonBoneNode>::new();
    let mut landmarks = Vec::<SkeletonLandmarkNode>::new();
    let mut measures = Vec::<SkeletonMeasureNode>::new();
    let mut ratios = Vec::<SkeletonRatioNode>::new();
    let mut regions = Vec::<SkeletonRegionNode>::new();
    let mut constraints = Vec::<SkeletonConstraintNode>::new();
    let mut guides = Vec::<SkeletonGuideNode>::new();
    let mut controls = Vec::<SkeletonControlNode>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with('{')
            || line.starts_with("<!--")
        {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Bone") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            bones.push(parse_skeleton_bone_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        macro_rules! parse_skeleton_child {
            ($tag_name:literal, $target:ident, $parser:ident) => {
                if starts_open_tag(line, $tag_name) {
                    let (tag, end_ix) = collect_self_closing_block(lines, i)?;
                    $target.push($parser(&tag, i + 1)?);
                    i = end_ix + 1;
                    continue;
                }
            };
        }
        parse_skeleton_child!("Landmark", landmarks, parse_skeleton_landmark_node);
        parse_skeleton_child!("Measure", measures, parse_skeleton_measure_node);
        parse_skeleton_child!("Ratio", ratios, parse_skeleton_ratio_node);
        parse_skeleton_child!("Region", regions, parse_skeleton_region_node);
        parse_skeleton_child!("Constraint", constraints, parse_skeleton_constraint_node);
        parse_skeleton_child!("Guide", guides, parse_skeleton_guide_node);
        parse_skeleton_child!("Control", controls, parse_skeleton_control_node);
        return Err(GraphParseError {
            line: i + 1,
            message: format!(
                "<Skeleton> only accepts Bone, Landmark, Measure, Ratio, Region, Constraint, Guide, or Control children, got: {line}"
            ),
        });
    }

    let mut skeleton = SkeletonNode {
        id,
        profile,
        height,
        facing,
        symmetry_axis,
        validation,
        auto_correct,
        overlay,
        bones,
        landmarks,
        measures,
        ratios,
        regions,
        constraints,
        guides,
        controls,
    };
    crate::scene::domain::prepare_skeleton(&mut skeleton);
    Ok((skeleton, close_ix))
}

fn parse_skeleton_bone_node(block: &str, line: usize) -> Result<SkeletonBoneNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let parent = attr_value(block, "parent")
        .or_else(|| attr_value(block, "parentId"))
        .or_else(|| attr_value(block, "parent_id"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.is_empty());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let rotation = attr_value(block, "rotation")
        .or_else(|| attr_value(block, "rotate"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let scale = attr_value(block, "scale")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let length = attr_value(block, "length")
        .or_else(|| attr_value(block, "len"))
        .map(|v| strip_wrappers(&v).to_string());

    Ok(SkeletonBoneNode {
        id,
        parent,
        role: skeleton_optional_attr(block, &["role"]),
        side: skeleton_optional_attr(block, &["side"]),
        x,
        y,
        rotation,
        scale,
        length,
    })
}

fn skeleton_optional_attr(block: &str, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| attr_value(block, key))
        .map(|value| strip_wrappers(&value).trim().to_string())
        .filter(|value| !value.is_empty())
}

fn skeleton_required_attr(block: &str, key: &str, line: usize) -> Result<String, GraphParseError> {
    Ok(strip_wrappers(&required_attr_value(block, key, line)?)
        .trim()
        .to_string())
}

fn parse_skeleton_pair(
    raw: &str,
    line: usize,
    name: &str,
) -> Result<(String, String), GraphParseError> {
    let clean = strip_wrappers(raw)
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_string();
    let values = clean
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if values.len() != 2 {
        return Err(GraphParseError {
            line,
            message: format!("{name} must contain exactly two values, got: {raw}"),
        });
    }
    Ok((values[0].to_string(), values[1].to_string()))
}

fn parse_skeleton_landmark_node(
    block: &str,
    line: usize,
) -> Result<SkeletonLandmarkNode, GraphParseError> {
    let offset_raw = required_attr_value(block, "offset", line)?;
    Ok(SkeletonLandmarkNode {
        id: skeleton_required_attr(block, "id", line)?,
        bone: skeleton_required_attr(block, "bone", line)?,
        offset: parse_skeleton_pair(&offset_raw, line, "Landmark.offset")?,
    })
}

fn parse_skeleton_measure_node(
    block: &str,
    line: usize,
) -> Result<SkeletonMeasureNode, GraphParseError> {
    Ok(SkeletonMeasureNode {
        id: skeleton_required_attr(block, "id", line)?,
        from: skeleton_required_attr(block, "from", line)?,
        to: skeleton_required_attr(block, "to", line)?,
    })
}

fn parse_skeleton_ratio_node(
    block: &str,
    line: usize,
) -> Result<SkeletonRatioNode, GraphParseError> {
    Ok(SkeletonRatioNode {
        measure: skeleton_required_attr(block, "measure", line)?,
        relative_to: skeleton_required_attr(block, "relativeTo", line)?,
        value: skeleton_required_attr(block, "value", line)?,
        tolerance: skeleton_optional_attr(block, &["tolerance"]),
    })
}

fn parse_skeleton_region_node(
    block: &str,
    line: usize,
) -> Result<SkeletonRegionNode, GraphParseError> {
    Ok(SkeletonRegionNode {
        id: skeleton_required_attr(block, "id", line)?,
        role: skeleton_required_attr(block, "role", line)?,
        kind: skeleton_required_attr(block, "type", line)?.to_ascii_lowercase(),
        center: skeleton_optional_attr(block, &["center"]),
        from: skeleton_optional_attr(block, &["from"]),
        to: skeleton_optional_attr(block, &["to"]),
        radius_x: skeleton_optional_attr(block, &["radiusX", "radius_x"]),
        radius_y: skeleton_optional_attr(block, &["radiusY", "radius_y"]),
        width: skeleton_optional_attr(block, &["width"]),
    })
}

fn parse_skeleton_constraint_node(
    block: &str,
    line: usize,
) -> Result<SkeletonConstraintNode, GraphParseError> {
    Ok(SkeletonConstraintNode {
        kind: skeleton_required_attr(block, "type", line)?.to_ascii_lowercase(),
        left: skeleton_optional_attr(block, &["left"]),
        right: skeleton_optional_attr(block, &["right"]),
        axis: skeleton_optional_attr(block, &["axis"]),
        from: skeleton_optional_attr(block, &["from"]),
        to: skeleton_optional_attr(block, &["to"]),
        bone: skeleton_optional_attr(block, &["bone"]),
        relative_to: skeleton_optional_attr(block, &["relativeTo", "relative_to"]),
        value: skeleton_optional_attr(block, &["value"]),
        min: skeleton_optional_attr(block, &["min"]),
        max: skeleton_optional_attr(block, &["max"]),
    })
}

fn parse_skeleton_guide_node(
    block: &str,
    line: usize,
) -> Result<SkeletonGuideNode, GraphParseError> {
    Ok(SkeletonGuideNode {
        id: skeleton_required_attr(block, "id", line)?,
        kind: skeleton_required_attr(block, "type", line)?.to_ascii_lowercase(),
        through: skeleton_required_attr(block, "through", line)?,
        angle: skeleton_optional_attr(block, &["angle"]).unwrap_or_else(|| "0".to_string()),
    })
}

fn parse_skeleton_control_node(
    block: &str,
    line: usize,
) -> Result<SkeletonControlNode, GraphParseError> {
    let targets = skeleton_optional_attr(block, &["targets"])
        .map(|value| parse_scene_string_list(&value))
        .unwrap_or_default();
    Ok(SkeletonControlNode {
        id: skeleton_required_attr(block, "id", line)?,
        kind: skeleton_required_attr(block, "type", line)?.to_ascii_lowercase(),
        target: skeleton_optional_attr(block, &["target"]),
        targets,
        chain_length: skeleton_optional_attr(block, &["chainLength", "chain_length"]),
    })
}

pub(crate) fn parse_action_block(
    lines: &[&str],
    start: usize,
) -> Result<(ActionNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Action")?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let skeleton = attr_value(&open_tag, "skeleton")
        .or_else(|| attr_value(&open_tag, "rig"))
        .map(|v| strip_wrappers(&v).to_string());
    let duration_explicit = attr_value(&open_tag, "duration").is_some();
    let mut poses = Vec::<ActionPoseNode>::new();
    let mut iks = Vec::<ActionIkNode>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Pose") {
            let (pose, end_ix) = parse_action_pose_block(lines, i)?;
            poses.push(pose);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "IK") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            iks.push(parse_action_ik_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<Action> only accepts <Pose> or <IK /> children, got: {line}"),
        });
    }

    poses.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    let duration_ms = if duration_explicit {
        parse_duration_ms(&open_tag, start + 1, 0)?
    } else {
        poses
            .iter()
            .map(|pose| (pose.t.max(0.0) * 1000.0).round() as u64)
            .max()
            .unwrap_or(0)
    };
    if duration_ms == 0 {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("Action {id} duration must be greater than zero."),
        });
    }

    Ok((
        ActionNode {
            id,
            skeleton,
            duration_ms,
            poses,
            iks,
        },
        close_ix,
    ))
}

fn parse_action_pose_block(
    lines: &[&str],
    start: usize,
) -> Result<(ActionPoseNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Pose")?;
    let t_raw = required_attr_value(&open_tag, "t", start + 1)
        .or_else(|_| required_attr_value(&open_tag, "time", start + 1))?;
    let t = parse_time_seconds(&t_raw, start + 1, "t")?;
    let mut bones = Vec::<ActionBoneNode>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Bone") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            bones.push(parse_action_bone_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<Pose> only accepts <Bone /> children, got: {line}"),
        });
    }

    Ok((ActionPoseNode { t, bones }, close_ix))
}

fn parse_action_bone_node(block: &str, line: usize) -> Result<ActionBoneNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    Ok(ActionBoneNode {
        id,
        x: attr_value(block, "x").map(|v| strip_wrappers(&v).to_string()),
        y: attr_value(block, "y").map(|v| strip_wrappers(&v).to_string()),
        rotation: attr_value(block, "rotation")
            .or_else(|| attr_value(block, "rotate"))
            .map(|v| strip_wrappers(&v).to_string()),
        scale: attr_value(block, "scale").map(|v| strip_wrappers(&v).to_string()),
        opacity: attr_value(block, "opacity").map(|v| strip_wrappers(&v).to_string()),
    })
}

fn parse_action_ik_node(block: &str, line: usize) -> Result<ActionIkNode, GraphParseError> {
    let chain = attr_value(block, "chain")
        .map(|v| {
            strip_wrappers(&v)
                .split(',')
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !chain.is_empty() && chain.len() < 3 {
        return Err(GraphParseError {
            line,
            message: "<IK chain=\"...\"> requires at least three bone ids.".to_string(),
        });
    }
    let root = if let Some(root) = chain.first() {
        root.clone()
    } else {
        required_attr_value(block, "root", line)
            .or_else(|_| required_attr_value(block, "start", line))
            .map(|v| strip_wrappers(&v).to_string())?
    };
    let mid = if chain.len() >= 3 {
        chain[1].clone()
    } else {
        required_attr_value(block, "mid", line)
            .or_else(|_| required_attr_value(block, "joint", line))
            .map(|v| strip_wrappers(&v).to_string())?
    };
    let end = if let Some(end) = chain.last() {
        end.clone()
    } else {
        required_attr_value(block, "end", line)
            .or_else(|_| required_attr_value(block, "tip", line))
            .map(|v| strip_wrappers(&v).to_string())?
    };
    let target_x = required_attr_value_any(block, &["targetX", "target_x", "x"], line)
        .map(|v| strip_wrappers(&v).to_string())?;
    let target_y = required_attr_value_any(block, &["targetY", "target_y", "y"], line)
        .map(|v| strip_wrappers(&v).to_string())?;
    let bend = attr_value(block, "bend")
        .or_else(|| attr_value(block, "pole"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1".to_string());
    let weight = attr_value(block, "weight")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1".to_string());
    let iterations = attr_value(block, "iterations")
        .or_else(|| attr_value(block, "iters"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "8".to_string());

    Ok(ActionIkNode {
        root,
        mid,
        end,
        chain,
        target_x,
        target_y,
        bend,
        weight,
        iterations,
    })
}

pub(crate) fn parse_apply_action_node(
    block: &str,
    line: usize,
) -> Result<ApplyActionNode, GraphParseError> {
    let target = strip_wrappers(&required_attr_value(block, "target", line)?).to_string();
    let action = strip_wrappers(&required_attr_value(block, "action", line)?).to_string();
    let at_ms = attr_value(block, "at")
        .as_deref()
        .map(|value| parse_time_seconds(value, line, "at"))
        .transpose()?
        .map(|seconds| (seconds.max(0.0) * 1000.0).round() as u64)
        .unwrap_or(0);
    Ok(ApplyActionNode {
        target,
        action,
        at_ms,
    })
}

fn parse_timeline_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &mut BrushParseContext,
) -> Result<(SceneTimelineNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Timeline")?;
    let id = attr_value(&open_tag, "id").map(|v| strip_wrappers(&v).to_string());
    let mut children = Vec::<SceneNode>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Track") {
            let (track, end_ix) = parse_track_block(lines, i, brush_ctx)?;
            children.push(SceneNode::Track(track));
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<Timeline> only accepts <Track> children, got: {line}"),
        });
    }

    Ok((SceneTimelineNode { id, children }, close_ix))
}

fn parse_track_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &mut BrushParseContext,
) -> Result<(SceneTrackNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Track")?;
    let id = attr_value(&open_tag, "id").map(|v| strip_wrappers(&v).to_string());
    let role = attr_value(&open_tag, "role")
        .map(|v| strip_wrappers(&v).to_ascii_lowercase())
        .filter(|v| !v.trim().is_empty());
    if let Some(role) = role.as_deref()
        && role != "camera"
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "Invalid Track role=\"{role}\". Expected role=\"camera\" or omit role."
            ),
        });
    }
    let space_attr = attr_value(&open_tag, "space")
        .map(|v| strip_wrappers(&v).to_ascii_lowercase())
        .filter(|v| !v.trim().is_empty());
    if role.as_deref() == Some("camera") && space_attr.is_some() {
        return Err(GraphParseError {
            line: start + 1,
            message: "<Track role=\"camera\"> must not set space. Use space=\"world\" or space=\"screen\" only on visual tracks.".to_string(),
        });
    }
    let space = space_attr.unwrap_or_else(|| "world".to_string());
    if !matches!(space.as_str(), "world" | "screen") {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "Invalid Track space=\"{space}\". Expected space=\"world\" or space=\"screen\"."
            ),
        });
    }
    let z = attr_value(&open_tag, "z")
        .map(|v| {
            let text = strip_wrappers(&v);
            text.parse::<i32>().map_err(|_| GraphParseError {
                line: start + 1,
                message: format!("Invalid Track z value: {text}"),
            })
        })
        .transpose()?
        .unwrap_or(0);
    let z_depth = attr_value(&open_tag, "zDepth")
        .or_else(|| attr_value(&open_tag, "z_depth"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let children = parse_timeline_item_nodes(lines, open_end_ix + 1, close_ix, brush_ctx)?;
    Ok((
        SceneTrackNode {
            id,
            role,
            space,
            z,
            z_depth,
            children,
        },
        close_ix,
    ))
}

fn parse_timeline_item_nodes(
    lines: &[&str],
    start: usize,
    end: usize,
    brush_ctx: &mut BrushParseContext,
) -> Result<Vec<SceneNode>, GraphParseError> {
    let mut nodes = Vec::<SceneNode>::new();
    let mut i = start;
    while i < end {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Sequence") {
            let (sequence, end_ix) = parse_sequence_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Sequence(sequence));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Chain") {
            let (chain, end_ix) = parse_chain_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Chain(chain));
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<Track> only accepts <Sequence> or <Chain> children, got: {line}"),
        });
    }
    Ok(nodes)
}

fn parse_sequence_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &mut BrushParseContext,
) -> Result<(SceneSequenceNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Sequence")?;
    let id = attr_value(&open_tag, "id").map(|v| strip_wrappers(&v).to_string());
    let from_ms = attr_value(&open_tag, "from")
        .or_else(|| attr_value(&open_tag, "at"))
        .as_deref()
        .map(|value| parse_time_seconds(value, start + 1, "from"))
        .transpose()?
        .map(|seconds| (seconds * 1000.0).round() as u64)
        .unwrap_or(0);
    let duration_ms = parse_duration_ms(&open_tag, start + 1, 0)?;
    if duration_ms == 0 {
        return Err(GraphParseError {
            line: start + 1,
            message: "<Sequence> requires duration greater than zero.".to_string(),
        });
    }
    let out = attr_value(&open_tag, "out")
        .map(|v| strip_wrappers(&v).to_ascii_lowercase())
        .unwrap_or_else(|| "hide".to_string());
    if !matches!(out.as_str(), "hide" | "hold") {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("Sequence out must be \"hide\" or \"hold\", got: {out}"),
        });
    }
    let mut child_ctx = brush_ctx.clone();
    let children = parse_scene_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    Ok((
        SceneSequenceNode {
            id,
            from_ms,
            duration_ms,
            out,
            children,
        },
        close_ix,
    ))
}

fn parse_chain_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &mut BrushParseContext,
) -> Result<(SceneChainNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Chain")?;
    let id = attr_value(&open_tag, "id").map(|v| strip_wrappers(&v).to_string());
    let from_ms = attr_value(&open_tag, "from")
        .or_else(|| attr_value(&open_tag, "at"))
        .as_deref()
        .map(|value| parse_time_seconds(value, start + 1, "from"))
        .transpose()?
        .map(|seconds| (seconds * 1000.0).round() as u64)
        .unwrap_or(0);
    let gap_ms = attr_value(&open_tag, "gap")
        .as_deref()
        .map(|value| parse_signed_time_ms(value, start + 1, "gap"))
        .transpose()?
        .unwrap_or(0);
    let mut children = Vec::<SceneNode>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Sequence") {
            let (sequence, end_ix) = parse_sequence_block(lines, i, brush_ctx)?;
            children.push(SceneNode::Sequence(sequence));
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<Chain> only accepts <Sequence> children, got: {line}"),
        });
    }

    Ok((
        SceneChainNode {
            id,
            from_ms,
            gap_ms,
            children,
        },
        close_ix,
    ))
}

pub(crate) fn parse_group_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(GroupNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Group")?;
    let brush = attr_value(&open_tag, "brush").map(|v| strip_wrappers(&v).to_string());
    brush_ctx.validate_brush_ref(brush.as_deref(), start + 1)?;
    let mut child_ctx = brush_ctx.with_inherited_brush(brush);
    let children = parse_scene_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    Ok((parse_group_node(&open_tag, start + 1, children)?, close_ix))
}

pub(crate) fn parse_puppet_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(PuppetNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Ok((
            parse_puppet_node(&open_tag, start + 1, Vec::new())?,
            open_end_ix,
        ));
    }
    let tag_name = if starts_open_tag(open_tag.trim(), "PuppetWarp") {
        "PuppetWarp"
    } else {
        "Puppet"
    };
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, tag_name)?;
    let mut child_ctx = brush_ctx.clone();
    let children = parse_scene_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    Ok((parse_puppet_node(&open_tag, start + 1, children)?, close_ix))
}

pub(crate) fn parse_mesh_topology_block(
    lines: &[&str],
    start: usize,
) -> Result<(MeshTopologyNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Ok((
            parse_mesh_topology_node(&open_tag, start + 1, Vec::new())?,
            open_end_ix,
        ));
    }
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "MeshTopology")?;
    let mut children = Vec::<SceneNode>::new();
    let mut i = open_end_ix + 1;
    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Vertex") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            children.push(SceneNode::Vertex(parse_vertex_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Triangle") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            children.push(SceneNode::Triangle(parse_triangle_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Edge") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            children.push(SceneNode::Edge(parse_edge_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Region") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            children.push(SceneNode::Region(parse_region_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("Unsupported <MeshTopology> child: {line}"),
        });
    }
    Ok((
        parse_mesh_topology_node(&open_tag, start + 1, children)?,
        close_ix,
    ))
}

pub(crate) fn parse_part_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(PartNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Part")?;
    let brush = attr_value(&open_tag, "brush").map(|v| strip_wrappers(&v).to_string());
    brush_ctx.validate_brush_ref(brush.as_deref(), start + 1)?;
    let mut child_ctx = brush_ctx.with_inherited_brush(brush);
    let children = parse_scene_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    Ok((parse_part_node(&open_tag, start + 1, children)?, close_ix))
}

pub(crate) fn parse_layout_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(GroupNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Layout")?;
    let mut child_ctx = brush_ctx.clone();
    let raw_children = parse_scene_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    let spans = parse_layout_child_spans(lines, open_end_ix + 1, close_ix)?;
    if spans.len() != raw_children.len() {
        return Err(GraphParseError {
            line: start + 1,
            message: "Could not map Layout children to their layoutSpan values.".to_string(),
        });
    }
    let mode = attr_value(&open_tag, "mode")
        .map(|value| strip_wrappers(&value).trim().to_ascii_lowercase())
        .unwrap_or_else(|| "row".to_string());
    if !matches!(mode.as_str(), "row" | "column" | "grid") {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("Invalid Layout mode=\"{mode}\". Expected row, column, or grid."),
        });
    }
    let item_width = literal_f32_attr(&open_tag, &["itemWidth", "item_width"], 240.0).max(0.0);
    let item_height = literal_f32_attr(&open_tag, &["itemHeight", "item_height"], 160.0).max(0.0);
    let gap = literal_f32_attr(&open_tag, &["gap"], 24.0);
    let row_gap = literal_f32_attr(&open_tag, &["rowGap", "row_gap"], gap);
    let column_gap = literal_f32_attr(&open_tag, &["columnGap", "column_gap"], gap);
    let columns = literal_f32_attr(&open_tag, &["columns"], 1.0)
        .round()
        .clamp(1.0, 1024.0) as usize;
    let align = normalize_layout_keyword(
        attr_value(&open_tag, "align")
            .as_deref()
            .map(strip_wrappers)
            .unwrap_or("start"),
    );
    let justify = normalize_layout_keyword(
        attr_value(&open_tag, "justify")
            .as_deref()
            .map(strip_wrappers)
            .unwrap_or("start"),
    );
    if !matches!(align.as_str(), "start" | "center" | "end") {
        return Err(GraphParseError {
            line: start + 1,
            message: "Layout align must be start, center, or end.".to_string(),
        });
    }
    if !matches!(
        justify.as_str(),
        "start" | "center" | "end" | "spacebetween" | "spacearound" | "spaceevenly"
    ) {
        return Err(GraphParseError {
            line: start + 1,
            message: "Layout justify must be start, center, end, spaceBetween, spaceAround, or spaceEvenly."
                .to_string(),
        });
    }
    let padding = parse_layout_padding(&open_tag, start + 1)?;
    let layout_id = attr_value(&open_tag, "id")
        .map(|value| strip_wrappers(&value).to_string())
        .unwrap_or_else(|| "layout".to_string());
    // Placement records reserve grid cells before transforms are lowered to Groups.
    let mut placements = Vec::with_capacity(raw_children.len());
    let mut cursor_column = 0_usize;
    let mut cursor_row = 0_usize;
    let mut cursor_linear = 0_usize;
    for span in spans.iter().copied() {
        let span = span.max(1).min(columns);
        match mode.as_str() {
            "grid" => {
                if cursor_column + span > columns {
                    cursor_column = 0;
                    cursor_row += 1;
                }
                placements.push((cursor_column, cursor_row, span));
                cursor_column += span;
                if cursor_column == columns {
                    cursor_column = 0;
                    cursor_row += 1;
                }
            }
            "column" => {
                placements.push((0, cursor_linear, span));
                cursor_linear += span;
            }
            _ => {
                placements.push((cursor_linear, 0, span));
                cursor_linear += span;
            }
        }
    }
    let occupied_columns = match mode.as_str() {
        "grid" => columns,
        "row" => spans.iter().sum::<usize>().max(1),
        _ => 1,
    };
    let occupied_rows = match mode.as_str() {
        "grid" => placements
            .iter()
            .map(|(_, row, _)| row + 1)
            .max()
            .unwrap_or(1),
        "column" => spans.iter().sum::<usize>().max(1),
        _ => 1,
    };
    let natural_width = occupied_columns as f32 * item_width
        + occupied_columns.saturating_sub(1) as f32 * column_gap;
    let natural_height =
        occupied_rows as f32 * item_height + occupied_rows.saturating_sub(1) as f32 * row_gap;
    let width = literal_f32_attr(
        &open_tag,
        &["width"],
        natural_width + padding[1] + padding[3],
    )
    .max(padding[1] + padding[3]);
    let height = literal_f32_attr(
        &open_tag,
        &["height"],
        natural_height + padding[0] + padding[2],
    )
    .max(padding[0] + padding[2]);
    let inner_width = (width - padding[1] - padding[3]).max(0.0);
    let inner_height = (height - padding[0] - padding[2]).max(0.0);
    let main_count = if mode == "column" {
        occupied_rows
    } else {
        occupied_columns
    };
    let natural_main = if mode == "column" {
        natural_height
    } else {
        natural_width
    };
    let available_main = if mode == "column" {
        inner_height
    } else {
        inner_width
    };
    let base_gap = if mode == "column" {
        row_gap
    } else {
        column_gap
    };
    let (main_offset, distributed_gap) =
        resolve_layout_justify(&justify, available_main, natural_main, main_count, base_gap);
    let cross_offset = if mode == "column" {
        resolve_layout_align(&align, inner_width, item_width)
    } else {
        resolve_layout_align(&align, inner_height, natural_height)
    };

    let mut children = Vec::with_capacity(raw_children.len());
    for (index, (child, (column, row, _span))) in raw_children
        .into_iter()
        .zip(placements.into_iter())
        .enumerate()
    {
        let (x, y) = match mode.as_str() {
            "column" => (
                padding[3] + cross_offset,
                padding[0] + main_offset + row as f32 * (item_height + distributed_gap),
            ),
            "grid" => (
                padding[3] + main_offset + column as f32 * (item_width + distributed_gap),
                padding[0] + cross_offset + row as f32 * (item_height + row_gap),
            ),
            _ => (
                padding[3] + main_offset + column as f32 * (item_width + distributed_gap),
                padding[0] + cross_offset,
            ),
        };
        let item_tag = format!("<Group id=\"{layout_id}__item_{index:03}\" x=\"{x}\" y=\"{y}\">");
        children.push(SceneNode::Group(parse_group_node(
            &item_tag,
            start + 1,
            vec![child],
        )?));
    }
    Ok((
        parse_group_node(
            &procedural_group_tag(&open_tag, &layout_id),
            start + 1,
            children,
        )?,
        close_ix,
    ))
}

fn parse_layout_child_spans(
    lines: &[&str],
    start: usize,
    end: usize,
) -> Result<Vec<usize>, GraphParseError> {
    let mut spans = Vec::new();
    let mut i = start;
    while i < end {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") {
            i += 1;
            continue;
        }
        let (tag, _, child_end) = scene_element_bounds(lines, i, end)?;
        let span = attr_value(&tag, "layoutSpan")
            .or_else(|| attr_value(&tag, "layout_span"))
            .map(|value| strip_wrappers(&value).trim().parse::<usize>())
            .transpose()
            .map_err(|_| GraphParseError {
                line: i + 1,
                message: "layoutSpan must be a positive literal integer.".to_string(),
            })?
            .unwrap_or(1);
        if span == 0 {
            return Err(GraphParseError {
                line: i + 1,
                message: "layoutSpan must be greater than zero.".to_string(),
            });
        }
        spans.push(span);
        i = child_end + 1;
    }
    Ok(spans)
}

fn parse_layout_padding(block: &str, line: usize) -> Result<[f32; 4], GraphParseError> {
    let Some(raw) = attr_value(block, "padding") else {
        return Ok([0.0; 4]);
    };
    let body = strip_wrappers(&raw);
    let body = body.trim().trim_start_matches('[').trim_end_matches(']');
    let values = split_scene_top_level_csv(body)
        .into_iter()
        .map(|value| value.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GraphParseError {
            line,
            message: "Layout padding requires literal numeric values.".to_string(),
        })?;
    match values.as_slice() {
        [all] => Ok([*all; 4]),
        [vertical, horizontal] => Ok([*vertical, *horizontal, *vertical, *horizontal]),
        [top, right, bottom, left] => Ok([*top, *right, *bottom, *left]),
        _ => Err(GraphParseError {
            line,
            message: "Layout padding accepts one, two, or four values.".to_string(),
        }),
    }
}

fn normalize_layout_keyword(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn resolve_layout_align(align: &str, available: f32, content: f32) -> f32 {
    let free = (available - content).max(0.0);
    match align {
        "center" => free * 0.5,
        "end" => free,
        _ => 0.0,
    }
}

fn resolve_layout_justify(
    justify: &str,
    available: f32,
    natural: f32,
    count: usize,
    base_gap: f32,
) -> (f32, f32) {
    let free = (available - natural).max(0.0);
    match justify {
        "center" => (free * 0.5, base_gap),
        "end" => (free, base_gap),
        "spacebetween" if count > 1 => (0.0, base_gap + free / (count - 1) as f32),
        "spacearound" if count > 0 => {
            let share = free / count as f32;
            (share * 0.5, base_gap + share)
        }
        "spaceevenly" => {
            let share = free / (count + 1).max(1) as f32;
            (share, base_gap + share)
        }
        _ => (0.0, base_gap),
    }
}

pub(crate) fn parse_repeat_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(SceneNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Repeat")?;
    let (children, variants, varies, variant_seed) =
        parse_repeat_contents(lines, open_end_ix + 1, close_ix, brush_ctx)?;
    let repeat = parse_repeat_node(&open_tag, start + 1, children)?;
    let distribution = attr_value(&open_tag, "distribution")
        .map(|value| strip_wrappers(&value).trim().to_ascii_lowercase())
        .unwrap_or_else(|| "linear".to_string());
    let has_advanced_variation = !variants.is_empty() || !varies.is_empty();
    let node = match distribution.as_str() {
        "linear" | "grid" if !has_advanced_variation => SceneNode::Repeat(repeat),
        "linear" | "grid" | "scatter" => SceneNode::Group(lower_advanced_repeat(
            &open_tag,
            repeat,
            &distribution,
            variants,
            varies,
            variant_seed,
            start + 1,
        )?),
        _ => {
            return Err(GraphParseError {
                line: start + 1,
                message: format!(
                    "Invalid Repeat distribution=\"{distribution}\". Expected linear, grid, or scatter."
                ),
            });
        }
    };
    Ok((node, close_ix))
}

#[derive(Debug, Clone)]
struct RepeatVariantDef {
    weight: f32,
    children: Vec<SceneNode>,
}

#[derive(Debug, Clone)]
struct RepeatVaryDef {
    property: String,
    values: Vec<String>,
    range: Option<[f32; 2]>,
}

type RepeatContents = (
    Vec<SceneNode>,
    Vec<RepeatVariantDef>,
    Vec<RepeatVaryDef>,
    Option<u32>,
);

fn parse_repeat_contents(
    lines: &[&str],
    start: usize,
    end: usize,
    brush_ctx: &BrushParseContext,
) -> Result<RepeatContents, GraphParseError> {
    let mut children = Vec::new();
    let mut variants = Vec::new();
    let mut varies = Vec::new();
    let mut variant_seed = None;
    let mut i = start;
    while i < end {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Variants") {
            if !variants.is_empty() {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "<Repeat> accepts at most one <Variants> block.".to_string(),
                });
            }
            let (tag, open_end) = collect_tag_block(lines, i, '>', false)?;
            let close = find_matching_close_tag(lines, open_end + 1, "Variants")?;
            let choose = attr_value(&tag, "choose")
                .map(|value| strip_wrappers(&value).trim().to_ascii_lowercase())
                .unwrap_or_else(|| "weighted".to_string());
            if choose != "weighted" {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "<Variants choose> currently supports only weighted.".to_string(),
                });
            }
            variant_seed = attr_value(&tag, "seed")
                .map(|value| strip_wrappers(&value).trim().parse::<u32>())
                .transpose()
                .map_err(|_| GraphParseError {
                    line: i + 1,
                    message: "<Variants seed> must be a literal unsigned integer.".to_string(),
                })?;
            let mut j = open_end + 1;
            while j < close {
                let variant_line = lines[j].trim();
                if variant_line.is_empty() || variant_line.starts_with("//") {
                    j += 1;
                    continue;
                }
                let (variant_tag, _, variant_end) = scene_element_bounds(lines, j, close)?;
                let weight = attr_value(&variant_tag, "weight")
                    .map(|value| strip_wrappers(&value).trim().parse::<f32>())
                    .transpose()
                    .map_err(|_| GraphParseError {
                        line: j + 1,
                        message: "Variant weight must be a literal number.".to_string(),
                    })?
                    .unwrap_or(1.0);
                if !weight.is_finite() || weight <= 0.0 {
                    return Err(GraphParseError {
                        line: j + 1,
                        message: "Variant weight must be greater than zero.".to_string(),
                    });
                }
                let mut child_ctx = brush_ctx.clone();
                let parsed = parse_scene_nodes(lines, j, variant_end + 1, &mut child_ctx)?;
                if parsed.len() != 1 {
                    return Err(GraphParseError {
                        line: j + 1,
                        message:
                            "Each direct <Variants> child must produce exactly one scene node."
                                .to_string(),
                    });
                }
                variants.push(RepeatVariantDef {
                    weight,
                    children: parsed,
                });
                j = variant_end + 1;
            }
            if variants.is_empty() {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "<Variants> requires at least one weighted child.".to_string(),
                });
            }
            i = close + 1;
            continue;
        }
        if starts_open_tag(line, "Vary") {
            let (tag, tag_end) = collect_self_closing_block(lines, i)?;
            let property = strip_wrappers(&required_attr_value(&tag, "property", i + 1)?)
                .trim()
                .to_string();
            let values = parse_literal_string_array(&tag, "values", i + 1)?.unwrap_or_default();
            let range = parse_literal_float_array(&tag, "range", 2, i + 1)?
                .map(|values| [values[0], values[1]]);
            if property.is_empty() || (values.is_empty() == range.is_none()) {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "<Vary> requires property and exactly one of values or range."
                        .to_string(),
                });
            }
            varies.push(RepeatVaryDef {
                property,
                values,
                range,
            });
            i = tag_end + 1;
            continue;
        }
        let (_, _, child_end) = scene_element_bounds(lines, i, end)?;
        let mut child_ctx = brush_ctx.clone();
        children.extend(parse_scene_nodes(lines, i, child_end + 1, &mut child_ctx)?);
        i = child_end + 1;
    }
    if !variants.is_empty() && !children.is_empty() {
        return Err(GraphParseError {
            line: start + 1,
            message: "<Repeat> cannot mix direct artwork with a <Variants> block.".to_string(),
        });
    }
    Ok((children, variants, varies, variant_seed))
}

fn scene_element_bounds(
    lines: &[&str],
    start: usize,
    end: usize,
) -> Result<(String, usize, usize), GraphParseError> {
    let (tag, open_end) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&tag) {
        return Ok((tag, open_end, open_end));
    }
    let name = tag
        .trim_start()
        .strip_prefix('<')
        .and_then(|rest| {
            rest.split(|ch: char| ch.is_whitespace() || ch == '>')
                .next()
        })
        .filter(|name| !name.is_empty())
        .ok_or_else(|| GraphParseError {
            line: start + 1,
            message: "Could not determine scene child tag name.".to_string(),
        })?;
    let close = find_matching_close_tag(lines, open_end + 1, name)?;
    if close >= end {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("Scene child <{name}> extends beyond its parent."),
        });
    }
    Ok((tag, open_end, close))
}

pub(crate) fn parse_mask_any(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(MaskNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Ok((
            parse_mask_node(&open_tag, start + 1, Vec::new())?,
            open_end_ix,
        ));
    }
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Mask")?;
    let mut child_ctx = brush_ctx.clone();
    let children = parse_scene_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    Ok((parse_mask_node(&open_tag, start + 1, children)?, close_ix))
}

pub(crate) fn parse_precompose_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(PrecomposeNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
        let duration_ms = if attr_value(&open_tag, "duration").is_some() {
            Some(parse_duration_ms(&open_tag, start + 1, 0)?)
        } else {
            None
        };
        let size = attr_value(&open_tag, "size")
            .as_deref()
            .map(|value| parse_size(value, start + 1, "size"))
            .transpose()?;
        return Ok((
            PrecomposeNode {
                id,
                duration_ms,
                size,
                children: Vec::new(),
            },
            open_end_ix,
        ));
    }
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Precompose")?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let duration_ms = if attr_value(&open_tag, "duration").is_some() {
        Some(parse_duration_ms(&open_tag, start + 1, 0)?)
    } else {
        None
    };
    let size = attr_value(&open_tag, "size")
        .as_deref()
        .map(|value| parse_size(value, start + 1, "size"))
        .transpose()?;
    let mut child_ctx = brush_ctx.clone();
    let children = parse_scene_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    Ok((
        PrecomposeNode {
            id,
            duration_ms,
            size,
            children,
        },
        close_ix,
    ))
}

fn parse_use_node(block: &str, line: usize) -> Result<UseNode, GraphParseError> {
    let ref_id = attr_value(block, "ref")
        .map(|v| strip_wrappers(&v).trim_start_matches('#').to_string())
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| GraphParseError {
            line,
            message: "<Use> requires ref=\"component_id\".".to_string(),
        })?;
    Ok(UseNode {
        id: attr_value(block, "id").map(|v| strip_wrappers(&v).to_string()),
        ref_id,
        x: scene_attr_or_default(block, &["x"], "0"),
        y: scene_attr_or_default(block, &["y"], "0"),
        rotation: scene_attr_or_default(block, &["rotation"], "0"),
        scale: scene_attr_or_default(block, &["scale"], "1"),
        scale_x: scene_attr_or_default(block, &["scaleX", "scale_x"], "1"),
        scale_y: scene_attr_or_default(block, &["scaleY", "scale_y"], "1"),
        skew_x: scene_attr_or_default(block, &["skewX", "skew_x"], "0"),
        skew_y: scene_attr_or_default(block, &["skewY", "skew_y"], "0"),
        transform_origin_x: scene_attr_or_default(
            block,
            &["transformOriginX", "transform_origin_x"],
            "0",
        ),
        transform_origin_y: scene_attr_or_default(
            block,
            &["transformOriginY", "transform_origin_y"],
            "0",
        ),
        opacity: scene_attr_or_default(block, &["opacity"], "1"),
        blend: scene_attr_or_default(block, &["blend"], "normal"),
        params: parse_component_param_values(block, line)?,
        slots: Vec::new(),
    })
}

fn parse_use_any(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(UseNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let mut use_node = parse_use_node(&open_tag, start + 1)?;
    if is_self_closing_tag(&open_tag) {
        return Ok((use_node, open_end_ix));
    }

    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Use")?;
    let mut seen = HashSet::new();
    let mut i = open_end_ix + 1;
    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") {
            i += 1;
            continue;
        }
        if !starts_open_tag(line, "Fill") {
            return Err(GraphParseError {
                line: i + 1,
                message: format!("<Use> only accepts <Fill slot=\"...\"> children, got: {line}"),
            });
        }
        let (fill_tag, fill_open_end) = collect_tag_block(lines, i, '>', false)?;
        let fill_close = find_matching_close_tag(lines, fill_open_end + 1, "Fill")?;
        let name = strip_wrappers(&required_attr_value(&fill_tag, "slot", i + 1)?)
            .trim()
            .to_string();
        if name.is_empty() || !seen.insert(name.clone()) {
            return Err(GraphParseError {
                line: i + 1,
                message: format!("Duplicate or empty <Fill slot=\"{name}\">."),
            });
        }
        let mut child_ctx = brush_ctx.clone();
        let children = parse_scene_nodes(lines, fill_open_end + 1, fill_close, &mut child_ctx)?;
        use_node.slots.push(ComponentSlotValue { name, children });
        i = fill_close + 1;
    }
    Ok((use_node, close_ix))
}

fn parse_component_param_values(
    block: &str,
    line: usize,
) -> Result<Vec<ComponentParamValue>, GraphParseError> {
    let Some(raw) = attr_value(block, "params") else {
        return Ok(Vec::new());
    };
    let body = strip_wrappers(&raw);
    let mut values = Vec::new();
    let mut seen = HashSet::new();
    for entry in split_scene_top_level_csv(body) {
        let Some((name, value)) = entry.split_once(':') else {
            return Err(GraphParseError {
                line,
                message: format!("Invalid <Use params> entry '{entry}'. Expected name: value."),
            });
        };
        let name = strip_wrappers(name).trim().to_string();
        let value = strip_wrappers(value).trim().to_string();
        if name.is_empty() || value.is_empty() {
            return Err(GraphParseError {
                line,
                message: "<Use params> names and values must not be empty.".to_string(),
            });
        }
        if !seen.insert(name.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate <Use params> value '{name}'."),
            });
        }
        values.push(ComponentParamValue { name, value });
    }
    Ok(values)
}

fn split_scene_top_level_csv(input: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut depth = 0_i32;
    for ch in input.chars() {
        if let Some(active) = quote {
            current.push(ch);
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
            }
            '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                if !current.trim().is_empty() {
                    entries.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        entries.push(current.trim().to_string());
    }
    entries
}

pub(crate) fn parse_camera_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(CameraNode, usize), GraphParseError> {
    let _ = brush_ctx;
    let (_open_tag, _open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    Err(GraphParseError {
        line: start + 1,
        message: "<Scene> Camera is an active Scene Camera controller and must be self-closing. Use <Track role=\"camera\"><Sequence><Camera ... /></Sequence></Track>. Put visuals in <Track space=\"world\"> or <Track space=\"screen\">.".to_string(),
    })
}

pub(crate) fn parse_character_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(CharacterNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        // Image-only characters need no child scene nodes.
        return Ok((
            parse_character_node(&open_tag, start + 1, Vec::new())?,
            open_end_ix,
        ));
    }
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Character")?;
    let mut child_ctx = brush_ctx.clone();
    let children = parse_scene_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    Ok((
        parse_character_node(&open_tag, start + 1, children)?,
        close_ix,
    ))
}

fn parse_scene_nodes(
    lines: &[&str],
    start: usize,
    end: usize,
    brush_ctx: &mut BrushParseContext,
) -> Result<Vec<SceneNode>, GraphParseError> {
    let mut nodes = Vec::<SceneNode>::new();
    let mut i = start;
    while i < end {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Defs") {
            let (defs, end_ix) = parse_defs_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Defs(defs));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Timeline") {
            let (timeline, end_ix) = parse_timeline_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Timeline(timeline));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "PixelGrid") {
            let (grid, end_ix) = parse_pixel_grid_block(lines, i)?;
            nodes.push(SceneNode::PixelGrid(grid));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Solid") {
            return Err(GraphParseError {
                line: i + 1,
                message:
                    "<Solid> has been removed. Use top-level <Background color=\"...\" /> instead."
                        .to_string(),
            });
        }
        if starts_open_tag(line, "Text") {
            let (text, end_ix) = parse_text_any(lines, i)?;
            nodes.push(SceneNode::Text(Box::new(text)));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "RadialRays") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Group(parse_radial_rays_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "ParticleField") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Group(parse_particle_field_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Image") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Image(parse_image_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Svg") || starts_open_tag(line, "SVG") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Svg(parse_svg_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Rect") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Rect(parse_rect_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Circle") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Circle(parse_circle_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Ellipse") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Ellipse(parse_ellipse_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Line") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Line(parse_line_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Polyline") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Polyline(parse_polyline_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Curve") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Polyline(parse_polyline_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if [
            "SpringChain",
            "DynamicCurve",
            "DistanceConstraint",
            "Hinge",
            "RigidBody2D",
            "ParticleEmitter",
            "Cloth",
            "HairStrandField",
            "CacheBake",
        ]
        .iter()
        .any(|name| starts_open_tag(line, name))
        {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Simulation(
                crate::simulation::dsl::parse_binding(&tag, i + 1)?,
            ));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Path") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Path(parse_path_node(&tag, i + 1, brush_ctx)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "FaceJaw") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::FaceJaw(parse_face_jaw_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Shadow") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Shadow(parse_shadow_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Group") {
            let (group, end_ix) = parse_group_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Group(group));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Layout") {
            let (layout, end_ix) = parse_layout_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Group(layout));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Puppet") || starts_open_tag(line, "PuppetWarp") {
            let (puppet, end_ix) = parse_puppet_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Puppet(puppet));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Pin") || starts_open_tag(line, "PuppetPin") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::Pin(parse_pin_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "LimbEnvelope") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::LimbEnvelope(parse_limb_envelope_node(
                &tag,
                i + 1,
            )?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "LimbRegion") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            nodes.push(SceneNode::LimbRegion(parse_limb_region_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "MeshTopology") {
            let (topology, end_ix) = parse_mesh_topology_block(lines, i)?;
            nodes.push(SceneNode::MeshTopology(topology));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Part") {
            let (part, end_ix) = parse_part_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Part(part));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Repeat") {
            let (repeat, end_ix) = parse_repeat_block(lines, i, brush_ctx)?;
            nodes.push(repeat);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Mask") {
            let (mask, end_ix) = parse_mask_any(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Mask(mask));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Precompose") {
            let (precompose, end_ix) = parse_precompose_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Precompose(precompose));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Use") {
            let (use_node, end_ix) = parse_use_any(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Use(use_node));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Layer3D") {
            let (tag, tag_end_ix) = collect_tag_block(lines, i, '>', false)?;
            if is_self_closing_tag(&tag) {
                nodes.push(SceneNode::Layer(parse_scene_layer_node(
                    &tag,
                    i + 1,
                    Vec::new(),
                    true,
                    true,
                )?));
                i = tag_end_ix + 1;
            } else {
                let (layer, end_ix) =
                    parse_scene_layer_block(lines, i, brush_ctx, "Layer3D", true)?;
                nodes.push(SceneNode::Layer(layer));
                i = end_ix + 1;
            }
            continue;
        }
        if starts_open_tag(line, "Layer") {
            let (tag, tag_end_ix) = collect_tag_block(lines, i, '>', false)?;
            if is_self_closing_tag(&tag) {
                nodes.push(SceneNode::Layer(parse_scene_layer_node(
                    &tag,
                    i + 1,
                    Vec::new(),
                    true,
                    false,
                )?));
                i = tag_end_ix + 1;
            } else {
                let (layer, end_ix) = parse_scene_layer_block(lines, i, brush_ctx, "Layer", false)?;
                nodes.push(SceneNode::Layer(layer));
                i = end_ix + 1;
            }
            continue;
        }
        if starts_open_tag(line, "Character") {
            let (character, end_ix) = parse_character_block(lines, i, brush_ctx)?;
            nodes.push(SceneNode::Character(character));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Camera") {
            let (tag, tag_end_ix) = collect_tag_block(lines, i, '>', false)?;
            if is_self_closing_tag(&tag) {
                nodes.push(SceneNode::Camera(parse_camera_node(
                    &tag,
                    i + 1,
                    Vec::new(),
                )?));
                i = tag_end_ix + 1;
            } else {
                let (camera, end_ix) = parse_camera_block(lines, i, brush_ctx)?;
                nodes.push(SceneNode::Camera(camera));
                i = end_ix + 1;
            }
            continue;
        }
        i += 1;
    }
    Ok(nodes)
}

pub(crate) fn parse_background_node(
    block: &str,
    line: usize,
) -> Result<BackgroundNode, GraphParseError> {
    let id = attr_value(block, "id")
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| Some("background".to_string()));
    let color =
        strip_wrappers(&attr_value(block, "color").unwrap_or_else(|| "#000000".to_string()))
            .to_string();
    if color.is_empty() {
        return Err(GraphParseError {
            line,
            message: "Background color must not be empty.".to_string(),
        });
    }
    Ok(BackgroundNode { id, color })
}

fn scene_attr_or_default(block: &str, names: &[&str], default_value: &str) -> String {
    names
        .iter()
        .find_map(|name| attr_value(block, name))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| default_value.to_string())
}

pub(crate) fn parse_text_node(
    block: &str,
    line: usize,
    layout: Option<TextLayoutNode>,
    animators: Vec<TextAnimatorNode>,
) -> Result<TextNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let value = strip_wrappers(&required_attr_value(block, "value", line)?).to_string();
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "center".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "center".to_string());
    let width = attr_value(block, "width").map(|v| strip_wrappers(&v).to_string());
    let max_width = attr_value(block, "maxWidth")
        .or_else(|| attr_value(block, "max_width"))
        .map(|v| strip_wrappers(&v).to_string());
    let align = attr_value(block, "align").map(|v| strip_wrappers(&v).to_string());
    if let Some(align) = align.as_deref()
        && crate::scene::text::TextAlignMode::parse(align).is_none()
    {
        return Err(GraphParseError {
            line,
            message: format!("Invalid Text align=\"{align}\". Expected left, center, or right."),
        });
    }
    let tracking = attr_value(block, "textGap")
        .or_else(|| attr_value(block, "text_gap"))
        .or_else(|| attr_value(block, "tracking"))
        .map(|v| strip_wrappers(&v).to_string());
    let font_size = attr_value(block, "fontSize")
        .or_else(|| attr_value(block, "font_size"))
        .or_else(|| attr_value(block, "size"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "96".to_string());
    let render_scale = attr_value(block, "renderScale")
        .or_else(|| attr_value(block, "render_scale"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1x".to_string());
    let antialias = attr_value(block, "antialias")
        .or_else(|| attr_value(block, "antiAlias"))
        .or_else(|| attr_value(block, "aa"))
        .map(|v| strip_wrappers(&v).to_string());
    let edge_smoothing = attr_value(block, "edgeSmoothing")
        .or_else(|| attr_value(block, "edge_smoothing"))
        .map(|v| strip_wrappers(&v).to_string());
    let soft_edge = attr_value(block, "softEdge")
        .or_else(|| attr_value(block, "soft_edge"))
        .map(|v| strip_wrappers(&v).to_string());
    let blur = attr_value(block, "blur").map(|v| strip_wrappers(&v).to_string());
    let line_height = attr_value(block, "lineHeight")
        .or_else(|| attr_value(block, "line_height"))
        .map(|v| strip_wrappers(&v).to_string());
    let color = attr_value(block, "color")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "#ffffff".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let box_style = attr_value(block, "box").map(|v| strip_wrappers(&v).to_string());
    let box_color = attr_value(block, "boxColor")
        .or_else(|| attr_value(block, "box_color"))
        .map(|v| strip_wrappers(&v).to_string());
    let box_padding = attr_value(block, "boxPadding")
        .or_else(|| attr_value(block, "box_padding"))
        .map(|v| strip_wrappers(&v).to_string());
    let box_padding_x = attr_value(block, "boxPaddingX")
        .or_else(|| attr_value(block, "box_padding_x"))
        .map(|v| strip_wrappers(&v).to_string());
    let box_padding_y = attr_value(block, "boxPaddingY")
        .or_else(|| attr_value(block, "box_padding_y"))
        .map(|v| strip_wrappers(&v).to_string());
    let box_radius = attr_value(block, "boxRadius")
        .or_else(|| attr_value(block, "box_radius"))
        .map(|v| strip_wrappers(&v).to_string());
    let stroke = attr_value(block, "stroke").map(|v| strip_wrappers(&v).to_string());
    let stroke_width = attr_value(block, "strokeWidth")
        .or_else(|| attr_value(block, "stroke_width"))
        .map(|v| strip_wrappers(&v).to_string());
    let stroke_join = attr_value(block, "strokeJoin")
        .or_else(|| attr_value(block, "stroke_join"))
        .map(|v| strip_wrappers(&v).to_string());
    let stroke_position = attr_value(block, "strokePosition")
        .or_else(|| attr_value(block, "stroke_position"))
        .map(|v| strip_wrappers(&v).to_string());
    let font_family = attr_value(block, "fontFamily")
        .or_else(|| attr_value(block, "font_family"))
        .map(|v| strip_wrappers(&v).to_string());
    let font_weight = attr_value(block, "fontWeight")
        .or_else(|| attr_value(block, "font_weight"))
        .map(|v| strip_wrappers(&v).to_string());
    let font = attr_value(block, "font").map(|v| strip_wrappers(&v).to_string());
    let font_path = attr_value(block, "fontPath")
        .or_else(|| attr_value(block, "font_path"))
        .map(|v| strip_wrappers(&v).to_string());
    let visible_chars = attr_value(block, "visibleChars")
        .or_else(|| attr_value(block, "visible_chars"))
        .map(|v| strip_wrappers(&v).to_string());
    let max_lines = attr_value(block, "maxLines")
        .or_else(|| attr_value(block, "max_lines"))
        .map(|v| strip_wrappers(&v).to_string());

    Ok(TextNode {
        id,
        value,
        x,
        y,
        rotation: scene_attr_or_default(block, &["rotation"], "0"),
        scale: scene_attr_or_default(block, &["scale"], "1"),
        scale_x: scene_attr_or_default(block, &["scaleX", "scale_x"], "1"),
        scale_y: scene_attr_or_default(block, &["scaleY", "scale_y"], "1"),
        skew_x: scene_attr_or_default(block, &["skewX", "skew_x"], "0"),
        skew_y: scene_attr_or_default(block, &["skewY", "skew_y"], "0"),
        transform_origin_x: scene_attr_or_default(
            block,
            &["transformOriginX", "transform_origin_x"],
            "0",
        ),
        transform_origin_y: scene_attr_or_default(
            block,
            &["transformOriginY", "transform_origin_y"],
            "0",
        ),
        width,
        max_width,
        align,
        tracking,
        font_size,
        render_scale,
        antialias,
        edge_smoothing,
        blur,
        soft_edge,
        line_height,
        color,
        opacity,
        box_style,
        box_color,
        box_padding,
        box_padding_x,
        box_padding_y,
        box_radius,
        stroke,
        stroke_width,
        stroke_join,
        stroke_position,
        visible_chars,
        max_lines,
        font,
        font_family,
        font_weight,
        font_path,
        layout,
        animators,
    })
}

fn parse_text_any(lines: &[&str], start: usize) -> Result<(TextNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Ok((
            parse_text_node(&open_tag, start + 1, None, Vec::new())?,
            open_end_ix,
        ));
    }

    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Text")?;
    let (layout, animators) = parse_text_children(lines, open_end_ix + 1, close_ix)?;
    Ok((
        parse_text_node(&open_tag, start + 1, layout, animators)?,
        close_ix,
    ))
}

fn parse_text_children(
    lines: &[&str],
    start: usize,
    end: usize,
) -> Result<(Option<TextLayoutNode>, Vec<TextAnimatorNode>), GraphParseError> {
    let mut layout = None;
    let mut animators = Vec::new();
    let mut i = start;
    while i < end {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "TextLayout") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            if layout.is_some() {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "<Text> accepts at most one <TextLayout /> child.".to_string(),
                });
            }
            layout = Some(parse_text_layout_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "TextAnimator") {
            let (animator, end_ix) = parse_text_animator_any(lines, i)?;
            animators.push(animator);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!(
                "<Text> only accepts <TextLayout /> and <TextAnimator> children, got: {line}"
            ),
        });
    }
    Ok((layout, animators))
}

fn parse_text_layout_node(block: &str, line: usize) -> Result<TextLayoutNode, GraphParseError> {
    let wrap = attr_value(block, "wrap")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "normal".to_string());
    if crate::scene::text::TextWrapMode::parse(&wrap).is_none() {
        return Err(GraphParseError {
            line,
            message: format!(
                "Invalid TextLayout wrap=\"{wrap}\". Expected none, normal, or balance."
            ),
        });
    }
    let overflow = attr_value(block, "overflow")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "clip".to_string());
    if crate::scene::text::TextOverflowMode::parse(&overflow).is_none() {
        return Err(GraphParseError {
            line,
            message: format!(
                "Invalid TextLayout overflow=\"{overflow}\". Expected clip, fit, or ellipsis."
            ),
        });
    }
    let safe_area = attr_value(block, "safeArea")
        .or_else(|| attr_value(block, "safe_area"))
        .map(|v| strip_wrappers(&v).to_string());
    if let Some(safe_area) = safe_area.as_deref() {
        crate::scene::text::parse_safe_area(safe_area)
            .map_err(|message| GraphParseError { line, message })?;
    }
    let max_lines = attr_value(block, "maxLines")
        .or_else(|| attr_value(block, "max_lines"))
        .map(|v| strip_wrappers(&v).to_string());
    let align = attr_value(block, "align").map(|v| strip_wrappers(&v).to_string());
    if let Some(align) = align.as_deref()
        && crate::scene::text::TextAlignMode::parse(align).is_none()
    {
        return Err(GraphParseError {
            line,
            message: format!(
                "Invalid TextLayout align=\"{align}\". Expected left, center, or right."
            ),
        });
    }

    Ok(TextLayoutNode {
        wrap,
        overflow,
        safe_area,
        max_lines,
        align,
    })
}

fn parse_text_animator_any(
    lines: &[&str],
    start: usize,
) -> Result<(TextAnimatorNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Ok((
            parse_text_animator_node(&open_tag, start + 1, None, None, Vec::new())?,
            open_end_ix,
        ));
    }

    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "TextAnimator")?;
    let (transform, style, effects) =
        parse_text_animator_children(lines, open_end_ix + 1, close_ix)?;
    Ok((
        parse_text_animator_node(&open_tag, start + 1, transform, style, effects)?,
        close_ix,
    ))
}

type TextAnimatorChildren = (
    Option<TextTransformNode>,
    Option<TextStyleOverrideNode>,
    Vec<TextEffectNode>,
);

fn parse_text_animator_children(
    lines: &[&str],
    start: usize,
    end: usize,
) -> Result<TextAnimatorChildren, GraphParseError> {
    let mut transform = None;
    let mut style = None;
    let mut effects = Vec::new();
    let mut i = start;
    while i < end {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Transform") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            if transform.is_some() {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "<TextAnimator> accepts at most one <Transform /> child.".to_string(),
                });
            }
            transform = Some(parse_text_transform_node(&tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Style") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            if style.is_some() {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "<TextAnimator> accepts at most one <Style /> child.".to_string(),
                });
            }
            style = Some(parse_text_style_override_node(&tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Effects") {
            let (mut parsed_effects, end_ix) = parse_text_effects_block(lines, i)?;
            effects.append(&mut parsed_effects);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!(
                "<TextAnimator> only accepts <Transform />, <Style />, and <Effects> children, got: {line}"
            ),
        });
    }
    Ok((transform, style, effects))
}

fn parse_text_animator_node(
    block: &str,
    line: usize,
    transform: Option<TextTransformNode>,
    style: Option<TextStyleOverrideNode>,
    effects: Vec<TextEffectNode>,
) -> Result<TextAnimatorNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let selector_raw = attr_value(block, "selector")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "word".to_string());
    let selector = TextSelectorKind::parse(&selector_raw).ok_or_else(|| GraphParseError {
        line,
        message: format!(
            "Invalid TextAnimator selector=\"{selector_raw}\". Expected char, word, line, or range."
        ),
    })?;
    let mode = attr_value(block, "mode")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "normal".to_string());
    if !matches!(mode.as_str(), "normal" | "karaoke") {
        return Err(GraphParseError {
            line,
            message: format!("Invalid TextAnimator mode=\"{mode}\". Expected normal or karaoke."),
        });
    }
    let from_ms = attr_value(block, "from")
        .map(|v| parse_signed_time_ms(&v, line, "TextAnimator.from"))
        .transpose()?
        .unwrap_or(0);
    let duration_ms = attr_value(block, "duration")
        .map(|v| parse_signed_time_ms(&v, line, "TextAnimator.duration"))
        .transpose()?
        .map(|value| value.max(0) as u64);
    let stagger_ms = attr_value(block, "stagger")
        .map(|v| parse_signed_time_ms(&v, line, "TextAnimator.stagger"))
        .transpose()?
        .unwrap_or(0);
    let order = attr_value(block, "order")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "forward".to_string());
    if !matches!(order.as_str(), "forward" | "reverse" | "random") {
        return Err(GraphParseError {
            line,
            message: format!(
                "Invalid TextAnimator order=\"{order}\". Expected forward, reverse, or random."
            ),
        });
    }
    let pre_roll_ms = attr_value(block, "preRoll")
        .or_else(|| attr_value(block, "pre_roll"))
        .map(|v| parse_signed_time_ms(&v, line, "TextAnimator.preRoll"))
        .transpose()?
        .unwrap_or(0);
    let post_roll_ms = attr_value(block, "postRoll")
        .or_else(|| attr_value(block, "post_roll"))
        .map(|v| parse_signed_time_ms(&v, line, "TextAnimator.postRoll"))
        .transpose()?
        .unwrap_or(0);
    let active_word = attr_value(block, "activeWord")
        .or_else(|| attr_value(block, "active_word"))
        .map(|v| strip_wrappers(&v).to_string());
    let random_seed = attr_value(block, "randomSeed")
        .or_else(|| attr_value(block, "random_seed"))
        .map(|v| {
            let text = strip_wrappers(&v);
            text.parse::<u64>().map_err(|_| GraphParseError {
                line,
                message: format!("Invalid TextAnimator randomSeed value: {text}"),
            })
        })
        .transpose()?;
    let range = attr_value(block, "range").map(|v| strip_wrappers(&v).to_string());

    Ok(TextAnimatorNode {
        id,
        selector,
        mode,
        from_ms,
        duration_ms,
        stagger_ms,
        order,
        pre_roll_ms,
        post_roll_ms,
        active_word,
        random_seed,
        range,
        transform,
        style,
        effects,
    })
}

fn parse_text_transform_node(block: &str) -> TextTransformNode {
    TextTransformNode {
        x: attr_value(block, "x").map(|v| strip_wrappers(&v).to_string()),
        y: attr_value(block, "y").map(|v| strip_wrappers(&v).to_string()),
        rotation: attr_value(block, "rotation").map(|v| strip_wrappers(&v).to_string()),
        scale: attr_value(block, "scale").map(|v| strip_wrappers(&v).to_string()),
        scale_x: attr_value(block, "scaleX")
            .or_else(|| attr_value(block, "scale_x"))
            .map(|v| strip_wrappers(&v).to_string()),
        scale_y: attr_value(block, "scaleY")
            .or_else(|| attr_value(block, "scale_y"))
            .map(|v| strip_wrappers(&v).to_string()),
        skew_x: attr_value(block, "skewX")
            .or_else(|| attr_value(block, "skew_x"))
            .map(|v| strip_wrappers(&v).to_string()),
        skew_y: attr_value(block, "skewY")
            .or_else(|| attr_value(block, "skew_y"))
            .map(|v| strip_wrappers(&v).to_string()),
    }
}

fn parse_text_style_override_node(block: &str) -> TextStyleOverrideNode {
    TextStyleOverrideNode {
        color: attr_value(block, "color").map(|v| strip_wrappers(&v).to_string()),
        opacity: attr_value(block, "opacity").map(|v| strip_wrappers(&v).to_string()),
        blur: attr_value(block, "blur").map(|v| strip_wrappers(&v).to_string()),
        stroke: attr_value(block, "stroke").map(|v| strip_wrappers(&v).to_string()),
        stroke_width: attr_value(block, "strokeWidth")
            .or_else(|| attr_value(block, "stroke_width"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_join: attr_value(block, "strokeJoin")
            .or_else(|| attr_value(block, "stroke_join"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_position: attr_value(block, "strokePosition")
            .or_else(|| attr_value(block, "stroke_position"))
            .map(|v| strip_wrappers(&v).to_string()),
        shadow_color: attr_value(block, "shadowColor")
            .or_else(|| attr_value(block, "shadow_color"))
            .map(|v| strip_wrappers(&v).to_string()),
        shadow_x: attr_value(block, "shadowX")
            .or_else(|| attr_value(block, "shadow_x"))
            .map(|v| strip_wrappers(&v).to_string()),
        shadow_y: attr_value(block, "shadowY")
            .or_else(|| attr_value(block, "shadow_y"))
            .map(|v| strip_wrappers(&v).to_string()),
        shadow_blur: attr_value(block, "shadowBlur")
            .or_else(|| attr_value(block, "shadow_blur"))
            .map(|v| strip_wrappers(&v).to_string()),
    }
}

fn parse_text_effects_block(
    lines: &[&str],
    start: usize,
) -> Result<(Vec<TextEffectNode>, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Ok((Vec::new(), open_end_ix));
    }
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Effects")?;
    let mut effects = Vec::new();
    let mut i = open_end_ix + 1;
    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Glow") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            effects.push(TextEffectNode::Glow(parse_text_glow_effect_node(
                &tag,
                i + 1,
            )?));
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<Effects> only accepts <Glow /> children for Text, got: {line}"),
        });
    }
    Ok((effects, close_ix))
}

fn parse_text_glow_effect_node(
    block: &str,
    line: usize,
) -> Result<TextGlowEffectNode, GraphParseError> {
    let radius = attr_value(block, "radius")
        .map(|v| strip_wrappers(&v).to_string())
        .ok_or_else(|| GraphParseError {
            line,
            message: "<Glow> requires radius=\"...\".".to_string(),
        })?;
    let intensity = attr_value(block, "intensity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1".to_string());
    let color = attr_value(block, "color").map(|v| strip_wrappers(&v).to_string());
    Ok(TextGlowEffectNode {
        radius,
        intensity,
        color,
    })
}

pub(crate) fn parse_image_node(block: &str, line: usize) -> Result<ImageNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let src = strip_wrappers(&required_attr_value_any(block, &["src", "path"], line)?).to_string();
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "center".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "center".to_string());
    let scale = attr_value(block, "scale")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());

    Ok(ImageNode {
        id,
        material: attr_value(block, "material").map(|v| strip_wrappers(&v).to_string()),
        src,
        x,
        y,
        scale,
        opacity,
    })
}

pub(crate) fn parse_svg_node(block: &str, line: usize) -> Result<SvgNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let src = strip_wrappers(&required_attr_value_any(block, &["src", "path"], line)?).to_string();
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "center".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "center".to_string());
    let scale = attr_value(block, "scale")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());

    Ok(SvgNode {
        id,
        material: attr_value(block, "material").map(|v| strip_wrappers(&v).to_string()),
        src,
        x,
        y,
        scale,
        opacity,
    })
}

pub(crate) fn parse_defs_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &mut BrushParseContext,
) -> Result<(DefsNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Defs")?;
    let id = attr_value(&open_tag, "id").map(|v| strip_wrappers(&v).to_string());
    let mut gradients = Vec::<GradientDef>::new();
    let mut textures = Vec::<TextureDef>::new();
    let mut noises = Vec::<NoiseDef>::new();
    let mut materials = Vec::<MaterialDef>::new();
    let mut brushes = Vec::<BrushDef>::new();
    let mut masks = Vec::<MaskNode>::new();
    let mut precomposes = Vec::<PrecomposeNode>::new();
    let mut components = Vec::<ComponentNode>::new();
    let mut filters = Vec::<FilterDef>::new();
    let mut fonts = Vec::<FontDef>::new();
    let mut palettes = Vec::<PaletteNode>::new();
    let mut simulation = Vec::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with('{')
            || line.starts_with("<!--")
        {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "LinearGradient") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            gradients.push(GradientDef::Linear(parse_linear_gradient_def(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "RadialGradient") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            gradients.push(GradientDef::Radial(parse_radial_gradient_def(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Texture") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            textures.push(parse_texture_def(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Noise") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            noises.push(parse_noise_def(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Material") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            materials.push(parse_material_def(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Brush") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            let brush = parse_brush_def(&tag, i + 1)?;
            brush_ctx.define_brushes(std::slice::from_ref(&brush));
            brushes.push(brush);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Mask") {
            let (mask, end_ix) = parse_mask_any(lines, i, brush_ctx)?;
            masks.push(mask);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Precompose") {
            let (precompose, end_ix) = parse_precompose_block(lines, i, brush_ctx)?;
            precomposes.push(precompose);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Component") {
            let (component, end_ix) = parse_component_block(lines, i, brush_ctx)?;
            components.push(component);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Filter") {
            let (filter, end_ix) = parse_filter_block(lines, i)?;
            filters.push(filter);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Font") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            fonts.push(parse_font_def(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Palette") {
            let (palette, end_ix) = parse_palette_block(lines, i)?;
            palettes.push(palette);
            i = end_ix + 1;
            continue;
        }
        if ["Gravity", "Wind", "Attraction", "Collider"]
            .iter()
            .any(|name| starts_open_tag(line, name))
        {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            simulation.push(crate::simulation::dsl::parse_resource(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!(
                "<Defs> only accepts resource tags: <LinearGradient />, <RadialGradient />, <Texture />, <Noise />, <Material />, <Brush />, <Mask>, <Precompose>, <Component>, <Filter>, <Font />, <Palette>, <Gravity />, <Wind />, <Attraction />, or <Collider />, got: {line}"
            ),
        });
    }

    Ok((
        DefsNode {
            id,
            gradients,
            textures,
            noises,
            materials,
            brushes,
            masks,
            precomposes,
            components,
            filters,
            fonts,
            palettes,
            simulation,
        },
        close_ix,
    ))
}

fn parse_noise_def(block: &str, line: usize) -> Result<NoiseDef, GraphParseError> {
    Ok(NoiseDef {
        id: strip_wrappers(&required_attr_value(block, "id", line)?).to_string(),
        kind: scene_attr_or_default(block, &["type", "kind"], "fbm"),
        scale: scene_attr_or_default(block, &["scale", "frequency"], "42"),
        octaves: scene_attr_or_default(block, &["octaves"], "4"),
        seed: scene_attr_or_default(block, &["seed"], "0"),
        contrast: scene_attr_or_default(block, &["contrast"], "1"),
        evolution: scene_attr_or_default(block, &["evolution", "time"], "0"),
    })
}

fn parse_material_def(block: &str, line: usize) -> Result<MaterialDef, GraphParseError> {
    Ok(MaterialDef {
        id: strip_wrappers(&required_attr_value(block, "id", line)?).to_string(),
        texture: attr_value(block, "texture").map(|v| strip_wrappers(&v).to_string()),
        texture_amount: scene_attr_or_default(block, &["textureAmount", "texture_amount"], "0.25"),
        displacement: attr_value(block, "displacement").map(|v| strip_wrappers(&v).to_string()),
        displacement_amount: scene_attr_or_default(
            block,
            &["displacementAmount", "displacement_amount"],
            "0",
        ),
        roughness: scene_attr_or_default(block, &["roughness"], "0.5"),
        specular: scene_attr_or_default(block, &["specular"], "0"),
        opacity: scene_attr_or_default(block, &["opacity"], "1"),
        refraction: scene_attr_or_default(block, &["refraction", "ior"], "0"),
        dispersion: scene_attr_or_default(
            block,
            &["dispersion", "chromaticDispersion", "chromatic_dispersion"],
            "0",
        ),
        glass: scene_attr_or_default(block, &["glass", "glassAmount", "glass_amount"], "0"),
    })
}

fn parse_texture_def(block: &str, line: usize) -> Result<TextureDef, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let kind = attr_value(block, "kind")
        .or_else(|| attr_value(block, "type"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "paper".to_string());
    Ok(TextureDef {
        id,
        src: scene_attr_or_default(block, &["src", "source", "href"], ""),
        kind,
        scale: scene_attr_or_default(block, &["scale"], "42"),
        strength: scene_attr_or_default(block, &["strength", "amount"], "0.25"),
        contrast: scene_attr_or_default(block, &["contrast"], "0.5"),
        seed: scene_attr_or_default(block, &["seed"], "0"),
        brush_angle: scene_attr_or_default(block, &["brushAngle", "brush_angle", "angle"], "-8"),
        bump_strength: scene_attr_or_default(
            block,
            &["bumpStrength", "bump_strength", "bump", "impastoStrength"],
            "0.35",
        ),
        relief: scene_attr_or_default(block, &["relief"], "0.45"),
    })
}

fn parse_component_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
) -> Result<(ComponentNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Component")?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let mut params = Vec::new();
    let mut derived = Vec::new();
    let mut slots = Vec::new();
    let mut param_names = HashSet::new();
    let mut derived_names = HashSet::new();
    let mut slot_names = HashSet::new();
    let mut child_lines = Vec::<String>::new();
    let mut i = open_end_ix + 1;
    while i < close_ix {
        let line = lines[i].trim();
        if starts_open_tag(line, "Param") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            let name = strip_wrappers(&required_attr_value(&tag, "name", i + 1)?)
                .trim()
                .to_string();
            if name.is_empty() {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "<Param name> must not be empty.".to_string(),
                });
            }
            if !param_names.insert(name.clone()) {
                return Err(GraphParseError {
                    line: i + 1,
                    message: format!("Duplicate <Param name=\"{name}\"> in Component '{id}'."),
                });
            }
            let value_type = attr_value(&tag, "type")
                .map(|value| strip_wrappers(&value).trim().to_ascii_lowercase())
                .unwrap_or_else(|| "number".to_string());
            if !matches!(
                value_type.as_str(),
                "number" | "color" | "text" | "path" | "boolean" | "enum"
            ) {
                return Err(GraphParseError {
                    line: i + 1,
                    message: format!(
                        "Invalid <Param type=\"{value_type}\">. Expected number, color, text, path, boolean, or enum."
                    ),
                });
            }
            let default = attr_value(&tag, "default")
                .map(|value| strip_wrappers(&value).to_string())
                .unwrap_or_default();
            let values = parse_literal_string_array(&tag, "values", i + 1)?.unwrap_or_default();
            if value_type == "enum" && values.is_empty() {
                return Err(GraphParseError {
                    line: i + 1,
                    message: format!(
                        "Enum parameter '{name}' requires non-empty values={{[...]}}."
                    ),
                });
            }
            params.push(ComponentParamDef {
                name,
                value_type,
                default,
                values,
            });
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Derived") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            let name = strip_wrappers(&required_attr_value(&tag, "name", i + 1)?)
                .trim()
                .to_string();
            if name.is_empty() || param_names.contains(&name) || !derived_names.insert(name.clone())
            {
                return Err(GraphParseError {
                    line: i + 1,
                    message: format!(
                        "Duplicate or empty <Derived name=\"{name}\"> in Component '{id}'."
                    ),
                });
            }
            let value = strip_wrappers(&required_attr_value(&tag, "value", i + 1)?).to_string();
            derived.push(ComponentDerivedDef { name, value });
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Slot") {
            let (tag, tag_end_ix) = collect_tag_block(lines, i, '>', false)?;
            let name = strip_wrappers(&required_attr_value(&tag, "name", i + 1)?)
                .trim()
                .to_string();
            if name.is_empty() || !slot_names.insert(name.clone()) {
                return Err(GraphParseError {
                    line: i + 1,
                    message: format!(
                        "Duplicate or empty <Slot name=\"{name}\"> in Component '{id}'."
                    ),
                });
            }
            slots.push(ComponentSlotDef { name: name.clone() });
            child_lines.push(format!("<Group id=\"__motionloom_slot__{name}\">"));
            if is_self_closing_tag(&tag) {
                child_lines.push("</Group>".to_string());
                i = tag_end_ix + 1;
                continue;
            }
            let slot_close = find_matching_close_tag(lines, tag_end_ix + 1, "Slot")?;
            child_lines.extend(
                lines[tag_end_ix + 1..slot_close]
                    .iter()
                    .map(|line| (*line).to_string()),
            );
            child_lines.push("</Group>".to_string());
            i = slot_close + 1;
            continue;
        }
        child_lines.push(lines[i].to_string());
        i += 1;
    }
    let mut child_ctx = brush_ctx.clone();
    let child_refs = child_lines.iter().map(String::as_str).collect::<Vec<_>>();
    let children = parse_scene_nodes(&child_refs, 0, child_refs.len(), &mut child_ctx)?;
    Ok((
        ComponentNode {
            id,
            params,
            derived,
            slots,
            children,
        },
        close_ix,
    ))
}

pub(crate) fn lower_parametric_component_uses(
    scene_nodes: &mut [SceneNode],
    scenes: &mut [SceneRootNode],
) -> Result<(), GraphParseError> {
    let mut components = HashMap::<String, ComponentNode>::new();
    collect_parametric_components(scene_nodes, &mut components);
    for scene in scenes.iter() {
        collect_parametric_components(&scene.children, &mut components);
    }
    lower_parametric_uses_in_nodes(scene_nodes, &components)?;
    for scene in scenes {
        lower_parametric_uses_in_nodes(&mut scene.children, &components)?;
    }
    Ok(())
}

fn collect_parametric_components(
    nodes: &[SceneNode],
    components: &mut HashMap<String, ComponentNode>,
) {
    for node in nodes {
        match node {
            SceneNode::Defs(defs) => {
                for component in &defs.components {
                    components.insert(component.id.clone(), component.clone());
                }
            }
            SceneNode::Timeline(node) => collect_parametric_components(&node.children, components),
            SceneNode::Track(node) => collect_parametric_components(&node.children, components),
            SceneNode::Sequence(node) => collect_parametric_components(&node.children, components),
            SceneNode::Chain(node) => collect_parametric_components(&node.children, components),
            SceneNode::Group(node) => collect_parametric_components(&node.children, components),
            SceneNode::Puppet(node) => collect_parametric_components(&node.children, components),
            SceneNode::Part(node) => collect_parametric_components(&node.children, components),
            SceneNode::Repeat(node) => collect_parametric_components(&node.children, components),
            SceneNode::Mask(node) => collect_parametric_components(&node.children, components),
            SceneNode::Precompose(node) => {
                collect_parametric_components(&node.children, components)
            }
            SceneNode::Layer(node) => collect_parametric_components(&node.children, components),
            SceneNode::Camera(node) => collect_parametric_components(&node.children, components),
            SceneNode::Character(node) => collect_parametric_components(&node.children, components),
            _ => {}
        }
    }
}

fn lower_parametric_uses_in_nodes(
    nodes: &mut [SceneNode],
    components: &HashMap<String, ComponentNode>,
) -> Result<(), GraphParseError> {
    for node in nodes {
        if let SceneNode::Use(use_node) = node {
            let Some(component) = components.get(&use_node.ref_id) else {
                if !use_node.params.is_empty() {
                    return Err(GraphParseError {
                        line: 0,
                        message: format!(
                            "Parameterized <Use ref=\"{}\"> references an unknown Component.",
                            use_node.ref_id
                        ),
                    });
                }
                continue;
            };
            if component.params.is_empty()
                && component.derived.is_empty()
                && component.slots.is_empty()
            {
                if !use_node.params.is_empty() {
                    return Err(GraphParseError {
                        line: 0,
                        message: format!(
                            "Component '{}' declares no <Param> values.",
                            component.id
                        ),
                    });
                }
                continue;
            }
            if !use_node.blend.trim().eq_ignore_ascii_case("normal") {
                return Err(GraphParseError {
                    line: 0,
                    message: "Parameterized <Use> currently requires blend=\"normal\".".to_string(),
                });
            }
            let mut bindings = HashMap::new();
            for param in &component.params {
                let normalized =
                    validate_component_param_value(param, &param.default, &component.id)?;
                bindings.insert(param.name.clone(), normalized);
            }
            for value in &use_node.params {
                let Some(param) = component
                    .params
                    .iter()
                    .find(|param| param.name == value.name)
                else {
                    return Err(GraphParseError {
                        line: 0,
                        message: format!(
                            "Unknown parameter '{}' for Component '{}'.",
                            value.name, component.id
                        ),
                    });
                };
                let normalized =
                    validate_component_param_value(param, &value.value, &component.id)?;
                bindings.insert(value.name.clone(), normalized);
            }
            if let Some(missing) = bindings.iter().find_map(|(name, value)| {
                if value.trim().is_empty() {
                    Some(name.clone())
                } else {
                    None
                }
            }) {
                return Err(GraphParseError {
                    line: 0,
                    message: format!(
                        "Component '{}' parameter '{}' has no default or Use value.",
                        component.id, missing
                    ),
                });
            }
            // Derived values are resolved in declaration order so later values can depend on earlier ones.
            for derived in &component.derived {
                let value = substitute_component_binding_text(&derived.value, &bindings);
                if value.contains("param(") || value.contains("derived(") {
                    return Err(GraphParseError {
                        line: 0,
                        message: format!(
                            "Component '{}' Derived '{}' references an unknown or later binding.",
                            component.id, derived.name
                        ),
                    });
                }
                bindings.insert(derived.name.clone(), value);
            }
            let slot_names = component
                .slots
                .iter()
                .map(|slot| slot.name.as_str())
                .collect::<HashSet<_>>();
            for fill in &use_node.slots {
                if !slot_names.contains(fill.name.as_str()) {
                    return Err(GraphParseError {
                        line: 0,
                        message: format!(
                            "Unknown slot '{}' for Component '{}'.",
                            fill.name, component.id
                        ),
                    });
                }
            }
            let mut children = substitute_component_params(&component.children, &bindings)?;
            replace_component_slots(&mut children, &use_node.slots);
            lower_parametric_uses_in_nodes(&mut children, components)?;
            let use_node = use_node.clone();

            // A reusable PuppetWarp must remain beside the artwork named by
            // `target`. An identity <Use> wrapper would otherwise hide it one
            // Group level deeper, preventing target resolution. Flatten only
            // this semantics-preserving special case; transformed Uses retain
            // the ordinary component Group wrapper.
            if use_has_identity_wrapper(&use_node) && children.len() == 1 {
                if let SceneNode::Puppet(mut puppet) = children.remove(0) {
                    if puppet.id.is_none() {
                        puppet.id = use_node.id;
                    }
                    *node = SceneNode::Puppet(puppet);
                    continue;
                }
            }

            *node = SceneNode::Group(GroupNode {
                id: use_node.id,
                brush: None,
                material: None,
                x: use_node.x,
                y: use_node.y,
                rotation: use_node.rotation,
                scale: use_node.scale,
                scale_x: use_node.scale_x,
                scale_y: use_node.scale_y,
                skew_x: use_node.skew_x,
                skew_y: use_node.skew_y,
                transform_origin_x: use_node.transform_origin_x,
                transform_origin_y: use_node.transform_origin_y,
                deform_grid: None,
                grid_from: None,
                grid_to: None,
                deform_amount: "0".to_string(),
                mask: None,
                mask_from: None,
                mask_mode: "alpha".to_string(),
                mask_feather: "0".to_string(),
                mask_expansion: "0".to_string(),
                effects: Vec::new(),
                opacity: use_node.opacity,
                children,
            });
            continue;
        }

        match node {
            SceneNode::Defs(defs) => {
                for mask in &mut defs.masks {
                    lower_parametric_uses_in_nodes(&mut mask.children, components)?;
                }
                for precompose in &mut defs.precomposes {
                    lower_parametric_uses_in_nodes(&mut precompose.children, components)?;
                }
            }
            SceneNode::Timeline(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Track(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Sequence(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Chain(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Group(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Puppet(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Part(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Repeat(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Mask(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Precompose(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Layer(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Camera(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            SceneNode::Character(node) => {
                lower_parametric_uses_in_nodes(&mut node.children, components)?
            }
            _ => {}
        }
    }
    Ok(())
}

fn use_has_identity_wrapper(node: &UseNode) -> bool {
    fn is_zero(value: &str) -> bool {
        value
            .trim()
            .parse::<f64>()
            .is_ok_and(|number| number.abs() <= f64::EPSILON)
    }
    fn is_one(value: &str) -> bool {
        value
            .trim()
            .parse::<f64>()
            .is_ok_and(|number| (number - 1.0).abs() <= f64::EPSILON)
    }

    is_zero(&node.x)
        && is_zero(&node.y)
        && is_zero(&node.rotation)
        && is_one(&node.scale)
        && is_one(&node.scale_x)
        && is_one(&node.scale_y)
        && is_zero(&node.skew_x)
        && is_zero(&node.skew_y)
        && is_zero(&node.transform_origin_x)
        && is_zero(&node.transform_origin_y)
        && is_one(&node.opacity)
        && node.blend.trim().eq_ignore_ascii_case("normal")
}

fn substitute_component_params(
    nodes: &[SceneNode],
    bindings: &HashMap<String, String>,
) -> Result<Vec<SceneNode>, GraphParseError> {
    let mut value = serde_json::to_value(nodes).map_err(|error| GraphParseError {
        line: 0,
        message: format!("Could not serialize Component for parameter substitution: {error}"),
    })?;
    substitute_component_value(&mut value, bindings);
    serde_json::from_value(value).map_err(|error| GraphParseError {
        line: 0,
        message: format!("Could not resolve Component parameters: {error}"),
    })
}

fn substitute_component_value(value: &mut serde_json::Value, bindings: &HashMap<String, String>) {
    match value {
        serde_json::Value::String(text) => {
            *text = substitute_component_binding_text(text, bindings);
        }
        serde_json::Value::Array(values) => {
            for value in values {
                substitute_component_value(value, bindings);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values_mut() {
                substitute_component_value(value, bindings);
            }
        }
        _ => {}
    }
}

fn substitute_component_binding_text(text: &str, bindings: &HashMap<String, String>) -> String {
    let mut resolved = text.to_string();
    for (name, replacement) in bindings {
        for function in ["param", "derived"] {
            resolved = resolved.replace(&format!("{function}(\"{name}\")"), replacement);
            resolved = resolved.replace(&format!("{function}('{name}')"), replacement);
        }
    }
    resolved
}

fn validate_component_param_value(
    param: &ComponentParamDef,
    value: &str,
    component_id: &str,
) -> Result<String, GraphParseError> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(String::new());
    }
    let valid = match param.value_type.as_str() {
        "number" => value.parse::<f64>().is_ok(),
        "color" => {
            let hex = value.strip_prefix('#');
            value == "none"
                || value == "transparent"
                || hex.is_some_and(|digits| {
                    matches!(digits.len(), 3 | 4 | 6 | 8)
                        && digits.chars().all(|digit| digit.is_ascii_hexdigit())
                })
        }
        "path" => matches!(value.chars().next(), Some('M' | 'm')),
        "boolean" => matches!(
            value.to_ascii_lowercase().as_str(),
            "true" | "false" | "1" | "0"
        ),
        "enum" => param.values.iter().any(|allowed| allowed == value),
        "text" => true,
        _ => false,
    };
    if !valid {
        return Err(GraphParseError {
            line: 0,
            message: format!(
                "Component '{component_id}' parameter '{}' expected {}; got '{value}'.",
                param.name, param.value_type
            ),
        });
    }
    if param.value_type == "boolean" {
        return Ok(
            if matches!(value.to_ascii_lowercase().as_str(), "true" | "1") {
                "1".to_string()
            } else {
                "0".to_string()
            },
        );
    }
    Ok(value.to_string())
}

fn replace_component_slots(nodes: &mut [SceneNode], fills: &[ComponentSlotValue]) {
    for node in nodes {
        match node {
            SceneNode::Group(group) => {
                if let Some(name) = group
                    .id
                    .as_deref()
                    .and_then(|id| id.strip_prefix("__motionloom_slot__"))
                {
                    if let Some(fill) = fills.iter().find(|fill| fill.name == name) {
                        group.children = fill.children.clone();
                    }
                    // Placeholder ids are parser internals and must not collide across Uses.
                    group.id = None;
                }
                replace_component_slots(&mut group.children, fills);
            }
            SceneNode::Puppet(node) => replace_component_slots(&mut node.children, fills),
            SceneNode::Part(node) => replace_component_slots(&mut node.children, fills),
            SceneNode::Repeat(node) => replace_component_slots(&mut node.children, fills),
            SceneNode::Mask(node) => replace_component_slots(&mut node.children, fills),
            SceneNode::Precompose(node) => replace_component_slots(&mut node.children, fills),
            SceneNode::Layer(node) => replace_component_slots(&mut node.children, fills),
            SceneNode::Camera(node) => replace_component_slots(&mut node.children, fills),
            SceneNode::Character(node) => replace_component_slots(&mut node.children, fills),
            _ => {}
        }
    }
}

fn parse_filter_block(lines: &[&str], start: usize) -> Result<(FilterDef, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Filter")?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let mut steps = Vec::<FilterStepDef>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Blur") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            steps.push(parse_filter_step_def("blur", &tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "EdgeSoftness") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            steps.push(parse_filter_step_def("edgeSoftness", &tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "EdgeRoughness") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            steps.push(parse_filter_step_def("edgeRoughness", &tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "ColorBleed") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            steps.push(parse_filter_step_def("colorBleed", &tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "LightStreak") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            steps.push(parse_filter_step_def("lightStreak", &tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "ChromaticAberration") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            steps.push(parse_filter_step_def("chromaticAberration", &tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "HighlightCompression") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            steps.push(parse_filter_step_def("highlightCompression", &tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "ColorMatrix") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            steps.push(parse_filter_step_def("colorMatrix", &tag));
            i = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Effect") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            let kind = attr_value(&tag, "type")
                .or_else(|| attr_value(&tag, "effect"))
                .map(|v| strip_wrappers(&v).to_string())
                .unwrap_or_else(|| "effect".to_string());
            steps.push(parse_filter_step_def(&kind, &tag));
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("unsupported <Filter> child: {line}"),
        });
    }

    Ok((FilterDef { id, steps }, close_ix))
}

fn parse_filter_step_def(kind: &str, block: &str) -> FilterStepDef {
    FilterStepDef {
        kind: kind.to_string(),
        radius: attr_value(block, "radius")
            .or_else(|| attr_value(block, "sigma"))
            .map(|v| strip_wrappers(&v).to_string()),
        amount: attr_value(block, "amount")
            .or_else(|| attr_value(block, "strength"))
            .map(|v| strip_wrappers(&v).to_string()),
        scale: attr_value(block, "scale").map(|v| strip_wrappers(&v).to_string()),
        seed: attr_value(block, "seed").map(|v| strip_wrappers(&v).to_string()),
        preserve_interior: attr_value(block, "preserveInterior")
            .or_else(|| attr_value(block, "preserve_interior"))
            .map(|v| strip_wrappers(&v).to_string()),
        saturation: attr_value(block, "saturation").map(|v| strip_wrappers(&v).to_string()),
        brightness: attr_value(block, "brightness").map(|v| strip_wrappers(&v).to_string()),
        contrast: attr_value(block, "contrast").map(|v| strip_wrappers(&v).to_string()),
        opacity: attr_value(block, "opacity").map(|v| strip_wrappers(&v).to_string()),
    }
}

fn parse_font_def(block: &str, line: usize) -> Result<FontDef, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    Ok(FontDef {
        id,
        family: attr_value(block, "family")
            .or_else(|| attr_value(block, "fontFamily"))
            .or_else(|| attr_value(block, "font_family"))
            .map(|v| strip_wrappers(&v).to_string()),
        path: attr_value(block, "path")
            .or_else(|| attr_value(block, "fontPath"))
            .or_else(|| attr_value(block, "font_path"))
            .map(|v| strip_wrappers(&v).to_string()),
        fallback: attr_value(block, "fallback").map(|v| strip_wrappers(&v).to_string()),
    })
}

fn parse_palette_block(
    lines: &[&str],
    start: usize,
) -> Result<(PaletteNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Palette")?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let mut colors = Vec::<PaletteColorDef>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Color") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            colors.push(parse_palette_color_def(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<Palette> only accepts <Color />, got: {line}"),
        });
    }

    Ok((PaletteNode { id, colors }, close_ix))
}

fn parse_palette_color_def(block: &str, line: usize) -> Result<PaletteColorDef, GraphParseError> {
    Ok(PaletteColorDef {
        key: strip_wrappers(&required_attr_value(block, "key", line)?).to_string(),
        value: strip_wrappers(&required_attr_value(block, "value", line)?).to_string(),
    })
}

pub(crate) fn parse_pixel_grid_block(
    lines: &[&str],
    start: usize,
) -> Result<(PixelGridNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "PixelGrid")?;
    let data = parse_pixel_grid_data(&lines[open_end_ix + 1..close_ix]);

    Ok((
        PixelGridNode {
            id: attr_value(&open_tag, "id").map(|v| strip_wrappers(&v).to_string()),
            x: attr_value(&open_tag, "x")
                .map(|v| strip_wrappers(&v).to_string())
                .unwrap_or_else(|| "0".to_string()),
            y: attr_value(&open_tag, "y")
                .map(|v| strip_wrappers(&v).to_string())
                .unwrap_or_else(|| "0".to_string()),
            pixel_size: attr_value(&open_tag, "pixelSize")
                .or_else(|| attr_value(&open_tag, "pixel_size"))
                .map(|v| strip_wrappers(&v).to_string())
                .unwrap_or_else(|| "1".to_string()),
            palette: strip_wrappers(&required_attr_value(&open_tag, "palette", start + 1)?)
                .to_string(),
            opacity: attr_value(&open_tag, "opacity")
                .map(|v| strip_wrappers(&v).to_string())
                .unwrap_or_else(|| "1".to_string()),
            blend: attr_value(&open_tag, "blend")
                .map(|v| strip_wrappers(&v).to_string())
                .unwrap_or_else(|| "normal".to_string()),
            data,
        },
        close_ix,
    ))
}

fn parse_pixel_grid_data(lines: &[&str]) -> String {
    let mut body = lines.join("\n");
    if let Some(start) = body.find("<![CDATA[") {
        body = body[start + "<![CDATA[".len()..].to_string();
    }
    if let Some(end) = body.rfind("]]>") {
        body.truncate(end);
    }
    body.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_brush_def(block: &str, line: usize) -> Result<BrushDef, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    Ok(BrushDef {
        id,
        stroke: attr_value(block, "stroke")
            .or_else(|| attr_value(block, "color"))
            .map(|v| strip_wrappers(&v).to_string()),
        fill: attr_value(block, "fill").map(|v| strip_wrappers(&v).to_string()),
        stroke_width: attr_value(block, "strokeWidth")
            .or_else(|| attr_value(block, "stroke_width"))
            .or_else(|| attr_value(block, "width"))
            .map(|v| strip_wrappers(&v).to_string()),
        opacity: attr_value(block, "opacity").map(|v| strip_wrappers(&v).to_string()),
        line_cap: attr_value(block, "lineCap")
            .or_else(|| attr_value(block, "line_cap"))
            .map(|v| strip_wrappers(&v).to_string()),
        line_join: attr_value(block, "lineJoin")
            .or_else(|| attr_value(block, "line_join"))
            .map(|v| strip_wrappers(&v).to_string()),
        taper_start: attr_value(block, "taperStart")
            .or_else(|| attr_value(block, "taper_start"))
            .map(|v| strip_wrappers(&v).to_string()),
        taper_end: attr_value(block, "taperEnd")
            .or_else(|| attr_value(block, "taper_end"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_style: attr_value(block, "strokeStyle")
            .or_else(|| attr_value(block, "stroke_style"))
            .or_else(|| attr_value(block, "style"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_roughness: attr_value(block, "strokeRoughness")
            .or_else(|| attr_value(block, "stroke_roughness"))
            .or_else(|| attr_value(block, "roughness"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_copies: attr_value(block, "strokeCopies")
            .or_else(|| attr_value(block, "stroke_copies"))
            .or_else(|| attr_value(block, "copies"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_texture: attr_value(block, "strokeTexture")
            .or_else(|| attr_value(block, "stroke_texture"))
            .or_else(|| attr_value(block, "texture"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_bristles: attr_value(block, "strokeBristles")
            .or_else(|| attr_value(block, "stroke_bristles"))
            .or_else(|| attr_value(block, "bristles"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_pressure: attr_value(block, "strokePressure")
            .or_else(|| attr_value(block, "stroke_pressure"))
            .or_else(|| attr_value(block, "pressure"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_pressure_min: attr_value(block, "strokePressureMin")
            .or_else(|| attr_value(block, "stroke_pressure_min"))
            .or_else(|| attr_value(block, "pressureMin"))
            .or_else(|| attr_value(block, "pressure_min"))
            .map(|v| strip_wrappers(&v).to_string()),
        stroke_pressure_curve: attr_value(block, "strokePressureCurve")
            .or_else(|| attr_value(block, "stroke_pressure_curve"))
            .or_else(|| attr_value(block, "pressureCurve"))
            .or_else(|| attr_value(block, "pressure_curve"))
            .map(|v| strip_wrappers(&v).to_string()),
        blend: attr_value(block, "blend").map(|v| strip_wrappers(&v).to_string()),
    })
}

fn parse_linear_gradient_def(
    block: &str,
    line: usize,
) -> Result<LinearGradientDef, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let x1 = attr_value(block, "x1")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y1 = attr_value(block, "y1")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let x2 = attr_value(block, "x2")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1".to_string());
    let y2 = attr_value(block, "y2")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let stops_raw = strip_wrappers(&required_attr_value(block, "stops", line)?).to_string();
    let stops = parse_gradient_stops(&stops_raw, line)?;
    let units = attr_value(block, "units")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "objectBoundingBox".to_string());
    Ok(LinearGradientDef {
        id,
        x1,
        y1,
        x2,
        y2,
        stops,
        units,
    })
}

fn parse_radial_gradient_def(
    block: &str,
    line: usize,
) -> Result<RadialGradientDef, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let cx = attr_value(block, "cx")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.5".to_string());
    let cy = attr_value(block, "cy")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.5".to_string());
    let r = attr_value(block, "r")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.5".to_string());
    let fx = attr_value(block, "fx").map(|v| strip_wrappers(&v).to_string());
    let fy = attr_value(block, "fy").map(|v| strip_wrappers(&v).to_string());
    let stops_raw = strip_wrappers(&required_attr_value(block, "stops", line)?).to_string();
    let stops = parse_gradient_stops(&stops_raw, line)?;
    let units = attr_value(block, "units")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "objectBoundingBox".to_string());
    Ok(RadialGradientDef {
        id,
        cx,
        cy,
        r,
        fx,
        fy,
        stops,
        units,
    })
}

fn parse_gradient_stops(raw: &str, line: usize) -> Result<Vec<GradientStop>, GraphParseError> {
    let mut stops = Vec::<GradientStop>::new();
    for token in raw.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let Some((offset_raw, color_raw)) = token.split_once(':') else {
            return Err(GraphParseError {
                line,
                message: format!("gradient stop must be 'offset:color', got: {token}"),
            });
        };
        let offset = offset_raw
            .trim()
            .parse::<f32>()
            .map_err(|_| GraphParseError {
                line,
                message: format!("invalid gradient stop offset: {}", offset_raw.trim()),
            })?
            .clamp(0.0, 1.0);
        let color = color_raw.trim();
        if color.is_empty() {
            return Err(GraphParseError {
                line,
                message: "gradient stop color cannot be empty".to_string(),
            });
        }
        stops.push(GradientStop {
            offset,
            color: color.to_string(),
        });
    }
    if stops.len() < 2 {
        return Err(GraphParseError {
            line,
            message: "gradient requires at least two stops".to_string(),
        });
    }
    stops.sort_by(|a, b| {
        a.offset
            .partial_cmp(&b.offset)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(stops)
}

pub(crate) fn parse_rect_node(block: &str, line: usize) -> Result<RectNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let width = strip_wrappers(&required_attr_value(block, "width", line)?).to_string();
    let height = strip_wrappers(&required_attr_value(block, "height", line)?).to_string();
    let radius = attr_value(block, "radius")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let color = attr_value(block, "color")
        .or_else(|| attr_value(block, "fill"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "#ffffff".to_string());
    let stroke = attr_value(block, "stroke").map(|v| strip_wrappers(&v).to_string());
    let stroke_width = attr_value(block, "strokeWidth")
        .or_else(|| attr_value(block, "stroke_width"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let rotation = attr_value(block, "rotation")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());

    Ok(RectNode {
        id,
        x,
        y,
        width,
        height,
        radius,
        color,
        stroke,
        stroke_width,
        opacity,
        rotation,
        scale: scene_attr_or_default(block, &["scale"], "1"),
        scale_x: scene_attr_or_default(block, &["scaleX", "scale_x"], "1"),
        scale_y: scene_attr_or_default(block, &["scaleY", "scale_y"], "1"),
        skew_x: scene_attr_or_default(block, &["skewX", "skew_x"], "0"),
        skew_y: scene_attr_or_default(block, &["skewY", "skew_y"], "0"),
        transform_origin_x: scene_attr_or_default(
            block,
            &["transformOriginX", "transform_origin_x"],
            "0",
        ),
        transform_origin_y: scene_attr_or_default(
            block,
            &["transformOriginY", "transform_origin_y"],
            "0",
        ),
        blend: attr_value(block, "blend")
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "normal".to_string()),
        texture: attr_value(block, "texture").map(|v| strip_wrappers(&v).to_string()),
        texture_opacity: scene_attr_or_default(block, &["textureOpacity", "texture_opacity"], "1"),
        texture_scale: scene_attr_or_default(block, &["textureScale", "texture_scale"], "1"),
        texture_mask: scene_attr_or_default(block, &["textureMask", "texture_mask"], "0"),
    })
}

pub(crate) fn parse_circle_node(block: &str, line: usize) -> Result<CircleNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let radius = strip_wrappers(&required_attr_value(block, "radius", line)?).to_string();
    let color = attr_value(block, "color")
        .or_else(|| attr_value(block, "fill"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "#ffffff".to_string());
    let stroke = attr_value(block, "stroke").map(|v| strip_wrappers(&v).to_string());
    let stroke_width = attr_value(block, "strokeWidth")
        .or_else(|| attr_value(block, "stroke_width"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    Ok(CircleNode {
        id,
        x,
        y,
        radius,
        color,
        stroke,
        stroke_width,
        opacity,
        rotation: scene_attr_or_default(block, &["rotation"], "0"),
        scale: scene_attr_or_default(block, &["scale"], "1"),
        scale_x: scene_attr_or_default(block, &["scaleX", "scale_x"], "1"),
        scale_y: scene_attr_or_default(block, &["scaleY", "scale_y"], "1"),
        skew_x: scene_attr_or_default(block, &["skewX", "skew_x"], "0"),
        skew_y: scene_attr_or_default(block, &["skewY", "skew_y"], "0"),
        transform_origin_x: scene_attr_or_default(
            block,
            &["transformOriginX", "transform_origin_x"],
            "0",
        ),
        transform_origin_y: scene_attr_or_default(
            block,
            &["transformOriginY", "transform_origin_y"],
            "0",
        ),
        blend: attr_value(block, "blend")
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "normal".to_string()),
        texture: attr_value(block, "texture").map(|v| strip_wrappers(&v).to_string()),
        texture_opacity: scene_attr_or_default(block, &["textureOpacity", "texture_opacity"], "1"),
        texture_scale: scene_attr_or_default(block, &["textureScale", "texture_scale"], "1"),
        texture_mask: scene_attr_or_default(block, &["textureMask", "texture_mask"], "0"),
    })
}

pub(crate) fn parse_ellipse_node(block: &str, line: usize) -> Result<EllipseNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let x = attr_value(block, "x")
        .or_else(|| attr_value(block, "cx"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .or_else(|| attr_value(block, "cy"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let radius_x = attr_value(block, "radiusX")
        .or_else(|| attr_value(block, "radius_x"))
        .or_else(|| attr_value(block, "rx"))
        .map(|v| strip_wrappers(&v).to_string())
        .ok_or_else(|| GraphParseError {
            line,
            message: "<Ellipse> requires radiusX/rx".to_string(),
        })?;
    let radius_y = attr_value(block, "radiusY")
        .or_else(|| attr_value(block, "radius_y"))
        .or_else(|| attr_value(block, "ry"))
        .map(|v| strip_wrappers(&v).to_string())
        .ok_or_else(|| GraphParseError {
            line,
            message: "<Ellipse> requires radiusY/ry".to_string(),
        })?;
    let color = attr_value(block, "color")
        .or_else(|| attr_value(block, "fill"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "#ffffff".to_string());
    let stroke = attr_value(block, "stroke").map(|v| strip_wrappers(&v).to_string());
    let stroke_width = attr_value(block, "strokeWidth")
        .or_else(|| attr_value(block, "stroke_width"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    Ok(EllipseNode {
        id,
        x,
        y,
        radius_x,
        radius_y,
        color,
        stroke,
        stroke_width,
        opacity,
        rotation: scene_attr_or_default(block, &["rotation"], "0"),
        scale: scene_attr_or_default(block, &["scale"], "1"),
        scale_x: scene_attr_or_default(block, &["scaleX", "scale_x"], "1"),
        scale_y: scene_attr_or_default(block, &["scaleY", "scale_y"], "1"),
        skew_x: scene_attr_or_default(block, &["skewX", "skew_x"], "0"),
        skew_y: scene_attr_or_default(block, &["skewY", "skew_y"], "0"),
        transform_origin_x: scene_attr_or_default(
            block,
            &["transformOriginX", "transform_origin_x"],
            "0",
        ),
        transform_origin_y: scene_attr_or_default(
            block,
            &["transformOriginY", "transform_origin_y"],
            "0",
        ),
        blend: attr_value(block, "blend")
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "normal".to_string()),
    })
}

#[derive(Debug, Clone)]
struct StrokeAttrs {
    style: String,
    roughness: String,
    copies: String,
    texture: String,
    bristles: String,
    pressure: String,
    pressure_min: String,
    pressure_curve: String,
}

fn stroke_style_attrs(block: &str) -> StrokeAttrs {
    stroke_style_attrs_with_brush(block, None)
}

fn attr_string(block: &str, names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| attr_value(block, name))
        .map(|v| strip_wrappers(&v).to_string())
}

fn attr_or_brush(
    block: &str,
    names: &[&str],
    brush_value: Option<&String>,
    default_value: &str,
) -> String {
    attr_string(block, names)
        .or_else(|| brush_value.cloned())
        .unwrap_or_else(|| default_value.to_string())
}

fn stroke_style_attrs_with_brush(block: &str, brush: Option<&BrushDef>) -> StrokeAttrs {
    let stroke_style = attr_value(block, "strokeStyle")
        .or_else(|| attr_value(block, "stroke_style"))
        .or_else(|| attr_value(block, "style"))
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.stroke_style.clone()))
        .unwrap_or_else(|| "solid".to_string());
    let stroke_roughness = attr_value(block, "strokeRoughness")
        .or_else(|| attr_value(block, "stroke_roughness"))
        .or_else(|| attr_value(block, "roughness"))
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.stroke_roughness.clone()))
        .unwrap_or_else(|| "0".to_string());
    let stroke_copies = attr_value(block, "strokeCopies")
        .or_else(|| attr_value(block, "stroke_copies"))
        .or_else(|| attr_value(block, "copies"))
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.stroke_copies.clone()))
        .unwrap_or_else(|| "1".to_string());
    let stroke_texture = attr_value(block, "strokeTexture")
        .or_else(|| attr_value(block, "stroke_texture"))
        .or_else(|| attr_value(block, "texture"))
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.stroke_texture.clone()))
        .unwrap_or_else(|| "0".to_string());
    let stroke_bristles = attr_value(block, "strokeBristles")
        .or_else(|| attr_value(block, "stroke_bristles"))
        .or_else(|| attr_value(block, "bristles"))
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.stroke_bristles.clone()))
        .unwrap_or_else(|| "0".to_string());
    let stroke_pressure = attr_value(block, "strokePressure")
        .or_else(|| attr_value(block, "stroke_pressure"))
        .or_else(|| attr_value(block, "pressure"))
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.stroke_pressure.clone()))
        .unwrap_or_else(|| "none".to_string());
    let stroke_pressure_min = attr_value(block, "strokePressureMin")
        .or_else(|| attr_value(block, "stroke_pressure_min"))
        .or_else(|| attr_value(block, "pressureMin"))
        .or_else(|| attr_value(block, "pressure_min"))
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.stroke_pressure_min.clone()))
        .unwrap_or_else(|| "1".to_string());
    let stroke_pressure_curve = attr_value(block, "strokePressureCurve")
        .or_else(|| attr_value(block, "stroke_pressure_curve"))
        .or_else(|| attr_value(block, "pressureCurve"))
        .or_else(|| attr_value(block, "pressure_curve"))
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.stroke_pressure_curve.clone()))
        .unwrap_or_else(|| "1".to_string());
    StrokeAttrs {
        style: stroke_style,
        roughness: stroke_roughness,
        copies: stroke_copies,
        texture: stroke_texture,
        bristles: stroke_bristles,
        pressure: stroke_pressure,
        pressure_min: stroke_pressure_min,
        pressure_curve: stroke_pressure_curve,
    }
}

pub(crate) fn parse_line_node(block: &str, line: usize) -> Result<LineNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let x1 = strip_wrappers(&required_attr_value(block, "x1", line)?).to_string();
    let y1 = strip_wrappers(&required_attr_value(block, "y1", line)?).to_string();
    let x2 = strip_wrappers(&required_attr_value(block, "x2", line)?).to_string();
    let y2 = strip_wrappers(&required_attr_value(block, "y2", line)?).to_string();
    let width = attr_value(block, "width")
        .or_else(|| attr_value(block, "strokeWidth"))
        .or_else(|| attr_value(block, "stroke_width"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "4".to_string());
    let color = attr_value(block, "color")
        .or_else(|| attr_value(block, "stroke"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "#ffffff".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let stroke_attrs = stroke_style_attrs(block);
    Ok(LineNode {
        id,
        x: scene_attr_or_default(block, &["x"], "0"),
        y: scene_attr_or_default(block, &["y"], "0"),
        rotation: scene_attr_or_default(block, &["rotation"], "0"),
        scale: scene_attr_or_default(block, &["scale"], "1"),
        scale_x: scene_attr_or_default(block, &["scaleX", "scale_x"], "1"),
        scale_y: scene_attr_or_default(block, &["scaleY", "scale_y"], "1"),
        skew_x: scene_attr_or_default(block, &["skewX", "skew_x"], "0"),
        skew_y: scene_attr_or_default(block, &["skewY", "skew_y"], "0"),
        transform_origin_x: scene_attr_or_default(
            block,
            &["transformOriginX", "transform_origin_x"],
            "0",
        ),
        transform_origin_y: scene_attr_or_default(
            block,
            &["transformOriginY", "transform_origin_y"],
            "0",
        ),
        x1,
        y1,
        x2,
        y2,
        width,
        color,
        opacity,
        line_cap: attr_value(block, "lineCap")
            .or_else(|| attr_value(block, "line_cap"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "round".to_string()),
        taper_start: attr_value(block, "taperStart")
            .or_else(|| attr_value(block, "taper_start"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "0".to_string()),
        taper_end: attr_value(block, "taperEnd")
            .or_else(|| attr_value(block, "taper_end"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "0".to_string()),
        stroke_style: stroke_attrs.style,
        stroke_roughness: stroke_attrs.roughness,
        stroke_copies: stroke_attrs.copies,
        stroke_texture: stroke_attrs.texture,
        stroke_bristles: stroke_attrs.bristles,
        stroke_pressure: stroke_attrs.pressure,
        stroke_pressure_min: stroke_attrs.pressure_min,
        stroke_pressure_curve: stroke_attrs.pressure_curve,
        blend: attr_value(block, "blend")
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "normal".to_string()),
    })
}

pub(crate) fn parse_polyline_node(
    block: &str,
    line: usize,
) -> Result<PolylineNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let points = strip_wrappers(&required_attr_value(block, "points", line)?).to_string();
    let stroke = attr_value(block, "stroke")
        .or_else(|| attr_value(block, "color"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "#ffffff".to_string());
    let stroke_width = attr_value(block, "strokeWidth")
        .or_else(|| attr_value(block, "stroke_width"))
        .or_else(|| attr_value(block, "width"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "4".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let trim_start = attr_value(block, "trimStart")
        .or_else(|| attr_value(block, "trim_start"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.0".to_string());
    let trim_end = attr_value(block, "trimEnd")
        .or_else(|| attr_value(block, "trim_end"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let stroke_attrs = stroke_style_attrs(block);

    Ok(PolylineNode {
        id,
        x: scene_attr_or_default(block, &["x"], "0"),
        y: scene_attr_or_default(block, &["y"], "0"),
        rotation: scene_attr_or_default(block, &["rotation"], "0"),
        scale: scene_attr_or_default(block, &["scale"], "1"),
        scale_x: scene_attr_or_default(block, &["scaleX", "scale_x"], "1"),
        scale_y: scene_attr_or_default(block, &["scaleY", "scale_y"], "1"),
        skew_x: scene_attr_or_default(block, &["skewX", "skew_x"], "0"),
        skew_y: scene_attr_or_default(block, &["skewY", "skew_y"], "0"),
        transform_origin_x: scene_attr_or_default(
            block,
            &["transformOriginX", "transform_origin_x"],
            "0",
        ),
        transform_origin_y: scene_attr_or_default(
            block,
            &["transformOriginY", "transform_origin_y"],
            "0",
        ),
        points,
        stroke,
        stroke_width,
        opacity,
        trim_start,
        trim_end,
        line_cap: attr_value(block, "lineCap")
            .or_else(|| attr_value(block, "line_cap"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "round".to_string()),
        line_join: attr_value(block, "lineJoin")
            .or_else(|| attr_value(block, "line_join"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "round".to_string()),
        taper_start: attr_value(block, "taperStart")
            .or_else(|| attr_value(block, "taper_start"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "0".to_string()),
        taper_end: attr_value(block, "taperEnd")
            .or_else(|| attr_value(block, "taper_end"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "0".to_string()),
        stroke_style: stroke_attrs.style,
        stroke_roughness: stroke_attrs.roughness,
        stroke_copies: stroke_attrs.copies,
        stroke_texture: stroke_attrs.texture,
        stroke_bristles: stroke_attrs.bristles,
        stroke_pressure: stroke_attrs.pressure,
        stroke_pressure_min: stroke_attrs.pressure_min,
        stroke_pressure_curve: stroke_attrs.pressure_curve,
        blend: attr_value(block, "blend")
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "normal".to_string()),
    })
}

pub(crate) fn parse_path_node(
    block: &str,
    line: usize,
    brush_ctx: &BrushParseContext,
) -> Result<PathNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let (brush_id, brush) = brush_ctx.brush_for_path(block, line)?;
    let d = strip_wrappers(&required_attr_value(block, "d", line)?).to_string();
    let fill = attr_value(block, "fill")
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.fill.clone()));
    let stroke = attr_value(block, "stroke")
        .or_else(|| attr_value(block, "color"))
        .map(|v| strip_wrappers(&v).to_string())
        .or_else(|| brush.and_then(|brush| brush.stroke.clone()))
        .unwrap_or_else(|| {
            if fill.is_some() {
                "none".to_string()
            } else {
                "#ffffff".to_string()
            }
        });
    let stroke_width = attr_or_brush(
        block,
        &["strokeWidth", "stroke_width", "width"],
        brush.and_then(|brush| brush.stroke_width.as_ref()),
        "4",
    );
    let opacity = attr_or_brush(
        block,
        &["opacity"],
        brush.and_then(|brush| brush.opacity.as_ref()),
        "1.0",
    );
    let trim_start = attr_value(block, "trimStart")
        .or_else(|| attr_value(block, "trim_start"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.0".to_string());
    let trim_end = attr_value(block, "trimEnd")
        .or_else(|| attr_value(block, "trim_end"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let stroke_attrs = stroke_style_attrs_with_brush(block, brush);

    Ok(PathNode {
        id,
        brush: brush_id,
        material: attr_value(block, "material").map(|v| strip_wrappers(&v).to_string()),
        x: scene_attr_or_default(block, &["x"], "0"),
        y: scene_attr_or_default(block, &["y"], "0"),
        rotation: scene_attr_or_default(block, &["rotation"], "0"),
        scale: scene_attr_or_default(block, &["scale"], "1"),
        scale_x: scene_attr_or_default(block, &["scaleX", "scale_x"], "1"),
        scale_y: scene_attr_or_default(block, &["scaleY", "scale_y"], "1"),
        skew_x: scene_attr_or_default(block, &["skewX", "skew_x"], "0"),
        skew_y: scene_attr_or_default(block, &["skewY", "skew_y"], "0"),
        transform_origin_x: scene_attr_or_default(
            block,
            &["transformOriginX", "transform_origin_x"],
            "0",
        ),
        transform_origin_y: scene_attr_or_default(
            block,
            &["transformOriginY", "transform_origin_y"],
            "0",
        ),
        d,
        stroke,
        fill,
        fill_rule: attr_value(block, "fillRule")
            .or_else(|| attr_value(block, "fill_rule"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "nonzero".to_string()),
        boolean_op: scene_attr_or_default(
            block,
            &["booleanOp", "boolean_op", "compoundOp", "compound_op"],
            "none",
        ),
        offset_path: scene_attr_or_default(block, &["offsetPath", "offset_path", "offset"], "0"),
        round_corners: scene_attr_or_default(
            block,
            &[
                "roundCorners",
                "round_corners",
                "cornerRadius",
                "corner_radius",
            ],
            "0",
        ),
        normalize: scene_attr_or_default(block, &["normalize", "normalizePath"], "false"),
        stroke_width,
        stroke_width_start: scene_attr_or_default(
            block,
            &[
                "strokeWidthStart",
                "stroke_width_start",
                "widthStart",
                "width_start",
            ],
            "1",
        ),
        stroke_width_end: scene_attr_or_default(
            block,
            &[
                "strokeWidthEnd",
                "stroke_width_end",
                "widthEnd",
                "width_end",
            ],
            "1",
        ),
        stroke_width_profile: attr_value(block, "strokeWidthProfile")
            .or_else(|| attr_value(block, "stroke_width_profile"))
            .map(|value| strip_wrappers(&value).to_string())
            .unwrap_or_default(),
        opacity,
        trim_start,
        trim_end,
        line_cap: attr_or_brush(
            block,
            &["lineCap", "line_cap"],
            brush.and_then(|brush| brush.line_cap.as_ref()),
            "round",
        ),
        line_join: attr_or_brush(
            block,
            &["lineJoin", "line_join"],
            brush.and_then(|brush| brush.line_join.as_ref()),
            "round",
        ),
        taper_start: attr_or_brush(
            block,
            &["taperStart", "taper_start"],
            brush.and_then(|brush| brush.taper_start.as_ref()),
            "0",
        ),
        taper_end: attr_or_brush(
            block,
            &["taperEnd", "taper_end"],
            brush.and_then(|brush| brush.taper_end.as_ref()),
            "0",
        ),
        stroke_style: stroke_attrs.style,
        stroke_roughness: stroke_attrs.roughness,
        stroke_copies: stroke_attrs.copies,
        stroke_texture: stroke_attrs.texture,
        stroke_bristles: stroke_attrs.bristles,
        stroke_pressure: stroke_attrs.pressure,
        stroke_pressure_min: stroke_attrs.pressure_min,
        stroke_pressure_curve: stroke_attrs.pressure_curve,
        blend: attr_or_brush(
            block,
            &["blend"],
            brush.and_then(|brush| brush.blend.as_ref()),
            "normal",
        ),
        texture: attr_value(block, "texture").map(|v| strip_wrappers(&v).to_string()),
        texture_opacity: scene_attr_or_default(block, &["textureOpacity", "texture_opacity"], "1"),
        texture_scale: scene_attr_or_default(block, &["textureScale", "texture_scale"], "1"),
        texture_mask: scene_attr_or_default(block, &["textureMask", "texture_mask"], "0"),
    })
}

pub(crate) fn parse_face_jaw_node(
    block: &str,
    _line: usize,
) -> Result<FaceJawNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let width = attr_value(block, "width")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "100".to_string());
    let height = attr_value(block, "height")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "100".to_string());
    let cheek_width = attr_value(block, "cheekWidth")
        .or_else(|| attr_value(block, "cheek_width"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| width.clone());
    let chin_width = attr_value(block, "chinWidth")
        .or_else(|| attr_value(block, "chin_width"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "24".to_string());
    let chin_sharpness = attr_value(block, "chinSharpness")
        .or_else(|| attr_value(block, "chin_sharpness"))
        .or_else(|| attr_value(block, "sharpness"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.45".to_string());
    let jaw_ease = attr_value(block, "jawEase")
        .or_else(|| attr_value(block, "jaw_ease"))
        .or_else(|| attr_value(block, "ease"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.55".to_string());
    let scale = attr_value(block, "scale")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1".to_string());
    let closed = attr_value(block, "closed")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "true".to_string());
    let fill = attr_value(block, "fill").map(|v| strip_wrappers(&v).to_string());
    let stroke = attr_value(block, "stroke")
        .or_else(|| attr_value(block, "color"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| {
            if fill.is_some() {
                "none".to_string()
            } else {
                "#ffffff".to_string()
            }
        });
    let stroke_width = attr_value(block, "strokeWidth")
        .or_else(|| attr_value(block, "stroke_width"))
        .or_else(|| attr_value(block, "widthStroke"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "4".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let trim_start = attr_value(block, "trimStart")
        .or_else(|| attr_value(block, "trim_start"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.0".to_string());
    let trim_end = attr_value(block, "trimEnd")
        .or_else(|| attr_value(block, "trim_end"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let stroke_attrs = stroke_style_attrs(block);
    Ok(FaceJawNode {
        id,
        x,
        y,
        width,
        height,
        cheek_width,
        chin_width,
        chin_sharpness,
        jaw_ease,
        scale,
        closed,
        stroke,
        fill,
        stroke_width,
        opacity,
        trim_start,
        trim_end,
        line_cap: attr_value(block, "lineCap")
            .or_else(|| attr_value(block, "line_cap"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "round".to_string()),
        line_join: attr_value(block, "lineJoin")
            .or_else(|| attr_value(block, "line_join"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "round".to_string()),
        taper_start: attr_value(block, "taperStart")
            .or_else(|| attr_value(block, "taper_start"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "0".to_string()),
        taper_end: attr_value(block, "taperEnd")
            .or_else(|| attr_value(block, "taper_end"))
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "0".to_string()),
        stroke_style: stroke_attrs.style,
        stroke_roughness: stroke_attrs.roughness,
        stroke_copies: stroke_attrs.copies,
        stroke_texture: stroke_attrs.texture,
        stroke_bristles: stroke_attrs.bristles,
        stroke_pressure: stroke_attrs.pressure,
        stroke_pressure_min: stroke_attrs.pressure_min,
        stroke_pressure_curve: stroke_attrs.pressure_curve,
        blend: attr_value(block, "blend")
            .map(|v| strip_wrappers(&v).to_string())
            .unwrap_or_else(|| "normal".to_string()),
    })
}

pub(crate) fn parse_shadow_node(block: &str, _line: usize) -> Result<ShadowNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "18".to_string());
    let blur = attr_value(block, "blur")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "36".to_string());
    let color = attr_value(block, "color")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "[0,0,0,0.18]".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    Ok(ShadowNode {
        id,
        x,
        y,
        blur,
        color,
        opacity,
    })
}

fn literal_f32_attr(block: &str, names: &[&str], default: f32) -> f32 {
    names
        .iter()
        .find_map(|name| attr_value(block, name))
        .map(|value| {
            strip_wrappers(&value)
                .trim()
                .parse::<f32>()
                .unwrap_or(default)
        })
        .unwrap_or(default)
}

fn procedural_group_tag(block: &str, fallback_id: &str) -> String {
    let id = scene_attr_or_default(block, &["id"], fallback_id);
    let x = scene_attr_or_default(block, &["x", "cx"], "0");
    let y = scene_attr_or_default(block, &["y", "cy"], "0");
    let rotation = scene_attr_or_default(block, &["rotation"], "0");
    let scale = scene_attr_or_default(block, &["scale"], "1");
    let opacity = scene_attr_or_default(block, &["opacity"], "1");
    format!(
        "<Group id=\"{id}\" x=\"{x}\" y=\"{y}\" rotation=\"{rotation}\" scale=\"{scale}\" opacity=\"{opacity}\">"
    )
}

fn procedural_random(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) as f32) / 16_777_215.0
}

fn parse_radial_rays_node(block: &str, line: usize) -> Result<GroupNode, GraphParseError> {
    let count = literal_f32_attr(block, &["count"], 24.0)
        .round()
        .clamp(1.0, 512.0) as usize;
    let inner = literal_f32_attr(block, &["innerRadius", "inner_radius"], 24.0);
    let length = literal_f32_attr(block, &["length"], 160.0);
    let variation = literal_f32_attr(block, &["lengthVariation", "length_variation"], 0.0);
    let start = literal_f32_attr(block, &["startAngle", "start_angle"], -90.0);
    let spread = literal_f32_attr(block, &["spread"], 360.0);
    let width = scene_attr_or_default(block, &["strokeWidth", "stroke_width"], "2");
    let color = scene_attr_or_default(block, &["stroke", "color"], "#ffffff");
    let opacity = scene_attr_or_default(block, &["rayOpacity", "ray_opacity"], "1");
    let mut seed = literal_f32_attr(block, &["seed"], 1.0).to_bits();
    let mut children = Vec::with_capacity(count);
    for index in 0..count {
        let fraction = if count == 1 {
            0.0
        } else {
            index as f32 / count as f32
        };
        let angle = (start + spread * fraction).to_radians();
        let ray_length = length * (1.0 + (procedural_random(&mut seed) * 2.0 - 1.0) * variation);
        let x1 = angle.cos() * inner;
        let y1 = angle.sin() * inner;
        let x2 = angle.cos() * (inner + ray_length.max(0.0));
        let y2 = angle.sin() * (inner + ray_length.max(0.0));
        let tag = format!(
            "<Line id=\"ray_{index:03}\" x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" stroke=\"{color}\" strokeWidth=\"{width}\" opacity=\"{opacity}\" />"
        );
        children.push(SceneNode::Line(parse_line_node(&tag, line)?));
    }
    parse_group_node(&procedural_group_tag(block, "radial_rays"), line, children)
}

fn parse_particle_field_node(block: &str, line: usize) -> Result<GroupNode, GraphParseError> {
    let count = literal_f32_attr(block, &["count"], 80.0)
        .round()
        .clamp(1.0, 2_048.0) as usize;
    let width = literal_f32_attr(block, &["width"], 400.0).max(0.0);
    let height = literal_f32_attr(block, &["height"], 240.0).max(0.0);
    let min_size = literal_f32_attr(block, &["minSize", "min_size"], 1.0).max(0.01);
    let max_size = literal_f32_attr(block, &["maxSize", "max_size"], 5.0).max(min_size);
    let min_opacity = literal_f32_attr(block, &["minOpacity", "min_opacity"], 0.2).clamp(0.0, 1.0);
    let max_opacity =
        literal_f32_attr(block, &["maxOpacity", "max_opacity"], 1.0).clamp(min_opacity, 1.0);
    let color = scene_attr_or_default(block, &["color", "fill"], "#ffffff");
    let mut seed = literal_f32_attr(block, &["seed"], 1.0).to_bits();
    let mut children = Vec::with_capacity(count);
    for index in 0..count {
        let x = (procedural_random(&mut seed) - 0.5) * width;
        let y = (procedural_random(&mut seed) - 0.5) * height;
        let radius = min_size + procedural_random(&mut seed) * (max_size - min_size);
        let opacity = min_opacity + procedural_random(&mut seed) * (max_opacity - min_opacity);
        let tag = format!(
            "<Circle id=\"particle_{index:04}\" x=\"{x}\" y=\"{y}\" radius=\"{radius}\" color=\"{color}\" opacity=\"{opacity}\" />"
        );
        children.push(SceneNode::Circle(parse_circle_node(&tag, line)?));
    }
    parse_group_node(
        &procedural_group_tag(block, "particle_field"),
        line,
        children,
    )
}

fn parse_group_node(
    block: &str,
    _line: usize,
    children: Vec<SceneNode>,
) -> Result<GroupNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let brush = attr_value(block, "brush").map(|v| strip_wrappers(&v).to_string());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let rotation = attr_value(block, "rotation")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let scale = attr_value(block, "scale")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let scale_x = attr_value(block, "scaleX")
        .or_else(|| attr_value(block, "scale_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let scale_y = attr_value(block, "scaleY")
        .or_else(|| attr_value(block, "scale_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let skew_x = attr_value(block, "skewX")
        .or_else(|| attr_value(block, "skew_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let skew_y = attr_value(block, "skewY")
        .or_else(|| attr_value(block, "skew_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let transform_origin_x = attr_value(block, "transformOriginX")
        .or_else(|| attr_value(block, "transform_origin_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let transform_origin_y = attr_value(block, "transformOriginY")
        .or_else(|| attr_value(block, "transform_origin_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let deform_grid = attr_value(block, "deformGrid")
        .or_else(|| attr_value(block, "deform_grid"))
        .map(|v| strip_wrappers(&v).to_string());
    let grid_from = attr_value(block, "gridFrom")
        .or_else(|| attr_value(block, "grid_from"))
        .map(|v| strip_wrappers(&v).to_string());
    let grid_to = attr_value(block, "gridTo")
        .or_else(|| attr_value(block, "grid_to"))
        .map(|v| strip_wrappers(&v).to_string());
    let deform_amount = attr_value(block, "deformAmount")
        .or_else(|| attr_value(block, "deform_amount"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let mask = attr_value(block, "mask")
        .or_else(|| attr_value(block, "maskId"))
        .or_else(|| attr_value(block, "mask_id"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    let mask_from = attr_value(block, "maskFrom")
        .or_else(|| attr_value(block, "mask_from"))
        .or_else(|| attr_value(block, "matteFrom"))
        .or_else(|| attr_value(block, "matte_from"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    let mask_mode = attr_value(block, "maskMode")
        .or_else(|| attr_value(block, "mask_mode"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "alpha".to_string());
    let mask_feather =
        scene_attr_or_default(block, &["maskFeather", "mask_feather", "feather"], "0");
    let mask_expansion = scene_attr_or_default(
        block,
        &["maskExpansion", "mask_expansion", "expansion"],
        "0",
    );
    let effects = attr_value(block, "effects")
        .or_else(|| attr_value(block, "effectStack"))
        .or_else(|| attr_value(block, "effect_stack"))
        .map(|value| parse_scene_string_list(&value))
        .unwrap_or_default();
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    Ok(GroupNode {
        id,
        brush,
        material: attr_value(block, "material").map(|v| strip_wrappers(&v).to_string()),
        x,
        y,
        rotation,
        scale,
        scale_x,
        scale_y,
        skew_x,
        skew_y,
        transform_origin_x,
        transform_origin_y,
        deform_grid,
        grid_from,
        grid_to,
        deform_amount,
        mask,
        mask_from,
        mask_mode,
        mask_feather,
        mask_expansion,
        effects,
        opacity,
        children,
    })
}

fn parse_scene_string_list(raw: &str) -> Vec<String> {
    strip_wrappers(raw)
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(str::trim)
        .map(|value| value.trim_matches(['"', '\'']).trim().to_string())
        .filter(|value| !value.is_empty())
        .collect()
}

fn parse_puppet_node(
    block: &str,
    line: usize,
    children: Vec<SceneNode>,
) -> Result<PuppetNode, GraphParseError> {
    let target = attr_value(block, "target")
        .or_else(|| attr_value(block, "targetId"))
        .or_else(|| attr_value(block, "target_id"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    let mut capture = attr_value(block, "capture")
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    if let Some(selector) = target.as_deref().filter(|value| value.starts_with('@')) {
        if !selector.eq_ignore_ascii_case("@layer") {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Unknown PuppetWarp target selector \"{selector}\". Use \"@layer\" or a Group id."
                ),
            });
        }
        if capture.is_none() {
            capture = Some("before".to_string());
        } else if !capture
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("before"))
        {
            return Err(GraphParseError {
                line,
                message: "PuppetWarp target=\"@layer\" requires capture=\"before\".".to_string(),
            });
        }
    }
    Ok(PuppetNode {
        id: attr_value(block, "id").map(|v| strip_wrappers(&v).to_string()),
        target,
        capture,
        solver: scene_attr_or_default(block, &["solver"], "soft"),
        mesh: scene_attr_or_default(block, &["mesh"], "auto"),
        density: scene_attr_or_default(block, &["density"], "medium"),
        bend: scene_attr_or_default(block, &["bend", "bendDirection", "bend_direction"], "auto"),
        stretch: scene_attr_or_default(block, &["stretch"], "0"),
        joint_softness: scene_attr_or_default(block, &["jointSoftness", "joint_softness"], "32"),
        preserve_volume: scene_attr_or_default(
            block,
            &["preserveVolume", "preserve_volume"],
            "true",
        ),
        preserve_outside: scene_attr_or_default(
            block,
            &["preserveOutside", "preserve_outside"],
            "false",
        ),
        preserve_length: scene_attr_or_default(
            block,
            &["preserveLength", "preserve_length"],
            "true",
        ),
        stiffness: scene_attr_or_default(block, &["stiffness"], "0.72"),
        damping: scene_attr_or_default(block, &["damping"], "0.84"),
        drag: scene_attr_or_default(block, &["drag"], "0.18"),
        overlap: scene_attr_or_default(block, &["overlap"], "0.12"),
        x: scene_attr_or_default(block, &["x"], "0"),
        y: scene_attr_or_default(block, &["y"], "0"),
        rotation: scene_attr_or_default(block, &["rotation"], "0"),
        scale: scene_attr_or_default(block, &["scale"], "1"),
        scale_x: scene_attr_or_default(block, &["scaleX", "scale_x"], "1"),
        scale_y: scene_attr_or_default(block, &["scaleY", "scale_y"], "1"),
        skew_x: scene_attr_or_default(block, &["skewX", "skew_x"], "0"),
        skew_y: scene_attr_or_default(block, &["skewY", "skew_y"], "0"),
        transform_origin_x: scene_attr_or_default(
            block,
            &["transformOriginX", "transform_origin_x"],
            "0",
        ),
        transform_origin_y: scene_attr_or_default(
            block,
            &["transformOriginY", "transform_origin_y"],
            "0",
        ),
        width: scene_attr_or_default(block, &["width", "w"], "512"),
        height: scene_attr_or_default(block, &["height", "h"], "512"),
        amount: scene_attr_or_default(block, &["amount", "deformAmount", "deform_amount"], "1"),
        opacity: scene_attr_or_default(block, &["opacity"], "1"),
        children,
    })
}

fn parse_limb_envelope_node(block: &str, line: usize) -> Result<LimbEnvelopeNode, GraphParseError> {
    let d = required_attr_value(block, "d", line)?;
    if !d.trim_end().ends_with(['Z', 'z']) {
        return Err(GraphParseError {
            line,
            message: "<LimbEnvelope d=\"...\"> must be a closed path ending in Z.".to_string(),
        });
    }
    Ok(LimbEnvelopeNode {
        id: attr_value(block, "id").map(|value| strip_wrappers(&value).to_string()),
        d,
        alpha_clip: scene_attr_or_default(block, &["alphaClip", "alpha_clip"], "true"),
        hand_from: attr_value(block, "handFrom")
            .or_else(|| attr_value(block, "hand_from"))
            .map(|value| strip_wrappers(&value).to_string())
            .filter(|value| !value.trim().is_empty()),
    })
}

fn parse_limb_region_node(block: &str, line: usize) -> Result<LimbRegionNode, GraphParseError> {
    let d = required_attr_value(block, "d", line)?;
    if !d.trim_end().ends_with(['Z', 'z']) {
        return Err(GraphParseError {
            line,
            message: "<LimbRegion d=\"...\"> must be a closed path ending in Z.".to_string(),
        });
    }
    let role = required_attr_value(block, "role", line)?;
    let normalized_role = strip_wrappers(&role).trim().to_ascii_lowercase();
    if !matches!(
        normalized_role.as_str(),
        "anchor"
            | "upper"
            | "shoulder"
            | "joint"
            | "elbow"
            | "control"
            | "lower"
            | "forearm"
            | "wrist"
            | "hand"
    ) {
        return Err(GraphParseError {
            line,
            message: "<LimbRegion role=\"...\"> expected anchor, joint, or control.".to_string(),
        });
    }
    Ok(LimbRegionNode {
        id: attr_value(block, "id").map(|value| strip_wrappers(&value).to_string()),
        role: normalized_role,
        d,
        alpha_clip: scene_attr_or_default(block, &["alphaClip", "alpha_clip"], "true"),
    })
}

pub(crate) fn parse_pin_node(block: &str, _line: usize) -> Result<PinNode, GraphParseError> {
    Ok(PinNode {
        id: attr_value(block, "id").map(|v| strip_wrappers(&v).to_string()),
        role: attr_value(block, "role")
            .map(|v| strip_wrappers(&v).to_string())
            .filter(|v| !v.trim().is_empty()),
        bind_to: attr_value(block, "bindTo")
            .or_else(|| attr_value(block, "bind_to"))
            .map(|v| strip_wrappers(&v).to_string())
            .filter(|v| !v.trim().is_empty()),
        vertex: attr_value(block, "vertex")
            .or_else(|| attr_value(block, "vertexId"))
            .or_else(|| attr_value(block, "vertex_id"))
            .map(|v| strip_wrappers(&v).to_string())
            .filter(|v| !v.trim().is_empty()),
        parent: attr_value(block, "parent")
            .or_else(|| attr_value(block, "parentId"))
            .or_else(|| attr_value(block, "parent_id"))
            .map(|v| strip_wrappers(&v).to_string())
            .filter(|v| !v.trim().is_empty()),
        x: attr_value(block, "x").map(|v| strip_wrappers(&v).to_string()),
        y: attr_value(block, "y").map(|v| strip_wrappers(&v).to_string()),
        target_x: attr_value(block, "targetX")
            .or_else(|| attr_value(block, "target_x"))
            .map(|v| strip_wrappers(&v).to_string()),
        target_y: attr_value(block, "targetY")
            .or_else(|| attr_value(block, "target_y"))
            .map(|v| strip_wrappers(&v).to_string()),
        radius: scene_attr_or_default(block, &["radius", "r"], "120"),
        strength: scene_attr_or_default(block, &["strength", "weight"], "1"),
        rotation: scene_attr_or_default(block, &["rotation", "rotate", "angle"], "0"),
        scale: scene_attr_or_default(block, &["scale"], "1"),
        falloff: scene_attr_or_default(block, &["falloff"], "smooth"),
        fixed: scene_attr_or_default(block, &["fixed", "lock", "locked"], "false"),
    })
}

fn parse_mesh_topology_node(
    block: &str,
    _line: usize,
    children: Vec<SceneNode>,
) -> Result<MeshTopologyNode, GraphParseError> {
    Ok(MeshTopologyNode {
        id: attr_value(block, "id").map(|v| strip_wrappers(&v).to_string()),
        mode: attr_value(block, "mode")
            .or_else(|| attr_value(block, "kind"))
            .map(|v| strip_wrappers(&v).to_string()),
        children,
    })
}

fn parse_vertex_node(block: &str, line: usize) -> Result<VertexNode, GraphParseError> {
    Ok(VertexNode {
        id: required_attr_value(block, "id", line)?,
        x: required_attr_value(block, "x", line)?,
        y: required_attr_value(block, "y", line)?,
        sample_x: attr_value(block, "sampleX")
            .or_else(|| attr_value(block, "sample_x"))
            .map(|value| strip_wrappers(&value).to_string()),
        sample_y: attr_value(block, "sampleY")
            .or_else(|| attr_value(block, "sample_y"))
            .map(|value| strip_wrappers(&value).to_string()),
        bone: attr_value(block, "bone")
            .or_else(|| attr_value(block, "bindTo"))
            .or_else(|| attr_value(block, "bind_to"))
            .map(|value| strip_wrappers(&value).to_string())
            .filter(|value| !value.trim().is_empty()),
    })
}

fn parse_triangle_node(block: &str, line: usize) -> Result<TriangleNode, GraphParseError> {
    Ok(TriangleNode {
        id: attr_value(block, "id").map(|v| strip_wrappers(&v).to_string()),
        a: required_attr_value(block, "a", line)?,
        b: required_attr_value(block, "b", line)?,
        c: required_attr_value(block, "c", line)?,
    })
}

fn parse_edge_node(block: &str, line: usize) -> Result<EdgeNode, GraphParseError> {
    Ok(EdgeNode {
        id: attr_value(block, "id").map(|v| strip_wrappers(&v).to_string()),
        a: required_attr_value(block, "a", line)?,
        b: required_attr_value(block, "b", line)?,
        boundary: scene_attr_or_default(block, &["boundary"], "false"),
    })
}

fn parse_region_node(block: &str, line: usize) -> Result<RegionNode, GraphParseError> {
    Ok(RegionNode {
        id: required_attr_value(block, "id", line)?,
        vertices: scene_attr_or_default(block, &["vertices", "verts"], ""),
        triangles: scene_attr_or_default(block, &["triangles"], ""),
        weight: scene_attr_or_default(block, &["weight"], "1"),
    })
}

fn parse_part_node(
    block: &str,
    _line: usize,
    children: Vec<SceneNode>,
) -> Result<PartNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let label = attr_value(block, "label").map(|v| strip_wrappers(&v).to_string());
    let role = attr_value(block, "role").map(|v| strip_wrappers(&v).to_string());
    let attach_to = attr_value(block, "attachTo")
        .or_else(|| attr_value(block, "attach_to"))
        .or_else(|| attr_value(block, "bone"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.is_empty());
    let brush = attr_value(block, "brush").map(|v| strip_wrappers(&v).to_string());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let rotation = attr_value(block, "rotation")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let scale = attr_value(block, "scale")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let anchor_x = attr_value(block, "anchorX")
        .or_else(|| attr_value(block, "anchor_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let anchor_y = attr_value(block, "anchorY")
        .or_else(|| attr_value(block, "anchor_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    Ok(PartNode {
        id,
        label,
        role,
        attach_to,
        brush,
        x,
        y,
        rotation,
        scale,
        opacity,
        anchor_x,
        anchor_y,
        children,
    })
}

fn parse_repeat_node(
    block: &str,
    _line: usize,
    children: Vec<SceneNode>,
) -> Result<RepeatNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let count = attr_value(block, "count")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1".to_string());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let rotation = attr_value(block, "rotation")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let scale = attr_value(block, "scale")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let x_step = attr_value(block, "xStep")
        .or_else(|| attr_value(block, "x_step"))
        .or_else(|| attr_value(block, "dx"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y_step = attr_value(block, "yStep")
        .or_else(|| attr_value(block, "y_step"))
        .or_else(|| attr_value(block, "dy"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let rotation_step = attr_value(block, "rotationStep")
        .or_else(|| attr_value(block, "rotation_step"))
        .or_else(|| attr_value(block, "dRotation"))
        .or_else(|| attr_value(block, "d_rotation"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let scale_step = attr_value(block, "scaleStep")
        .or_else(|| attr_value(block, "scale_step"))
        .or_else(|| attr_value(block, "dScale"))
        .or_else(|| attr_value(block, "d_scale"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let opacity_step = attr_value(block, "opacityStep")
        .or_else(|| attr_value(block, "opacity_step"))
        .or_else(|| attr_value(block, "dOpacity"))
        .or_else(|| attr_value(block, "d_opacity"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());

    Ok(RepeatNode {
        id,
        count,
        x,
        y,
        rotation,
        scale,
        opacity,
        x_step,
        y_step,
        rotation_step,
        scale_step,
        opacity_step,
        children,
    })
}

fn lower_advanced_repeat(
    block: &str,
    repeat: RepeatNode,
    distribution: &str,
    mut variants: Vec<RepeatVariantDef>,
    varies: Vec<RepeatVaryDef>,
    variant_seed: Option<u32>,
    line: usize,
) -> Result<GroupNode, GraphParseError> {
    let count = repeat
        .count
        .trim()
        .parse::<usize>()
        .map_err(|_| GraphParseError {
            line,
            message: "Advanced Repeat variation requires a literal integer count.".to_string(),
        })?;
    if count > 10_000 {
        return Err(GraphParseError {
            line,
            message: "Advanced Repeat variation supports at most 10000 instances.".to_string(),
        });
    }

    let bounds = parse_literal_float_array(block, "bounds", 4, line)?
        .unwrap_or_else(|| vec![0.0, 0.0, 400.0, 240.0]);
    let scale_range =
        parse_literal_float_array(block, "scaleRange", 2, line)?.unwrap_or_else(|| vec![1.0, 1.0]);
    let rotation_range = parse_literal_float_array(block, "rotationRange", 2, line)?
        .unwrap_or_else(|| vec![0.0, 0.0]);
    let opacity_range = parse_literal_float_array(block, "opacityRange", 2, line)?
        .unwrap_or_else(|| vec![1.0, 1.0]);
    let literal = |name: &str, value: &str| {
        value.trim().parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("Advanced Repeat variation requires literal {name}; got '{value}'."),
        })
    };
    let x_step = literal("xStep", &repeat.x_step)?;
    let y_step = literal("yStep", &repeat.y_step)?;
    let rotation_step = literal("rotationStep", &repeat.rotation_step)?;
    let scale_step = literal("scaleStep", &repeat.scale_step)?;
    let opacity_step = literal("opacityStep", &repeat.opacity_step)?;
    let repeat_seed = attr_value(block, "seed")
        .and_then(|value| strip_wrappers(&value).trim().parse::<u32>().ok())
        .unwrap_or(1);
    let mut placement_state = repeat_seed;
    let mut variation_state = variant_seed.unwrap_or(repeat_seed);
    let base_id = repeat
        .id
        .clone()
        .unwrap_or_else(|| "scatter_repeat".to_string());
    if variants.is_empty() {
        variants.push(RepeatVariantDef {
            weight: 1.0,
            children: repeat.children.clone(),
        });
    }
    let columns = literal_f32_attr(block, &["columns"], 1.0)
        .round()
        .clamp(1.0, 1024.0) as usize;
    let mut children = Vec::with_capacity(count);
    for index in 0..count {
        let fraction = |state: &mut u32, range: &[f32]| {
            range[0] + procedural_random(state) * (range[1] - range[0])
        };
        let (column, row) = (index % columns, index / columns);
        let (mut x, mut y) = match distribution {
            "scatter" => (
                bounds[0]
                    + procedural_random(&mut placement_state) * bounds[2]
                    + x_step * index as f32,
                bounds[1]
                    + procedural_random(&mut placement_state) * bounds[3]
                    + y_step * index as f32,
            ),
            "grid" => (x_step * column as f32, y_step * row as f32),
            _ => (x_step * index as f32, y_step * index as f32),
        };
        let mut rotation = rotation_step * index as f32
            + if distribution == "scatter" {
                fraction(&mut placement_state, &rotation_range)
            } else {
                0.0
            };
        let mut scale = ((1.0 + scale_step * index as f32)
            * if distribution == "scatter" {
                fraction(&mut placement_state, &scale_range)
            } else {
                1.0
            })
        .clamp(0.001, 64.0);
        let mut opacity = ((1.0 + opacity_step * index as f32)
            * if distribution == "scatter" {
                fraction(&mut placement_state, &opacity_range)
            } else {
                1.0
            })
        .clamp(0.0, 1.0);
        let variant_index = weighted_repeat_variant_index(&variants, &mut variation_state);
        let mut artwork = variants[variant_index].children.clone();
        for vary in &varies {
            let value = if let Some(range) = vary.range {
                let value =
                    range[0] + procedural_random(&mut variation_state) * (range[1] - range[0]);
                value.to_string()
            } else {
                let choice = (procedural_random(&mut variation_state) * vary.values.len() as f32)
                    .floor() as usize;
                vary.values[choice.min(vary.values.len() - 1)].clone()
            };
            match vary.property.as_str() {
                "x" => {
                    x += value
                        .parse::<f32>()
                        .map_err(|_| vary_numeric_error(vary, line))?
                }
                "y" => {
                    y += value
                        .parse::<f32>()
                        .map_err(|_| vary_numeric_error(vary, line))?
                }
                "rotation" => {
                    rotation += value
                        .parse::<f32>()
                        .map_err(|_| vary_numeric_error(vary, line))?
                }
                "scale" => {
                    scale *= value
                        .parse::<f32>()
                        .map_err(|_| vary_numeric_error(vary, line))?
                }
                "opacity" => {
                    opacity *= value
                        .parse::<f32>()
                        .map_err(|_| vary_numeric_error(vary, line))?
                }
                property => apply_repeat_vary_property(&mut artwork, property, &value)?,
            }
        }
        scale = scale.clamp(0.001, 64.0);
        opacity = opacity.clamp(0.0, 1.0);
        let item_tag = format!(
            "<Group id=\"{base_id}__item_{index:04}\" x=\"{x}\" y=\"{y}\" rotation=\"{rotation}\" scale=\"{scale}\" opacity=\"{opacity}\">"
        );
        children.push(SceneNode::Group(parse_group_node(
            &item_tag, line, artwork,
        )?));
    }

    let outer_tag = format!(
        "<Group id=\"{base_id}\" x=\"{}\" y=\"{}\" rotation=\"{}\" scale=\"{}\" opacity=\"{}\">",
        repeat.x, repeat.y, repeat.rotation, repeat.scale, repeat.opacity
    );
    parse_group_node(&outer_tag, line, children)
}

fn weighted_repeat_variant_index(variants: &[RepeatVariantDef], state: &mut u32) -> usize {
    if variants.len() == 1 {
        return 0;
    }
    let total = variants.iter().map(|variant| variant.weight).sum::<f32>();
    let mut choice = procedural_random(state) * total;
    for (index, variant) in variants.iter().enumerate() {
        if choice < variant.weight {
            return index;
        }
        choice -= variant.weight;
    }
    variants.len() - 1
}

fn vary_numeric_error(vary: &RepeatVaryDef, line: usize) -> GraphParseError {
    GraphParseError {
        line,
        message: format!(
            "<Vary property=\"{}\"> requires numeric values.",
            vary.property
        ),
    }
}

fn apply_repeat_vary_property(
    nodes: &mut Vec<SceneNode>,
    property: &str,
    value: &str,
) -> Result<(), GraphParseError> {
    let mut json = serde_json::to_value(&*nodes).map_err(|error| GraphParseError {
        line: 0,
        message: format!("Could not serialize Repeat variant: {error}"),
    })?;
    let mut matches = 0;
    apply_repeat_vary_json(&mut json, property, value, &mut matches);
    if matches == 0 {
        return Err(GraphParseError {
            line: 0,
            message: format!("<Vary property=\"{property}\"> matched no variant attributes."),
        });
    }
    *nodes = serde_json::from_value(json).map_err(|error| GraphParseError {
        line: 0,
        message: format!("Could not resolve Repeat variation: {error}"),
    })?;
    Ok(())
}

fn apply_repeat_vary_json(
    value: &mut serde_json::Value,
    property: &str,
    replacement: &str,
    matches: &mut usize,
) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                apply_repeat_vary_json(value, property, replacement, matches);
            }
        }
        serde_json::Value::Object(values) => {
            for (name, value) in values.iter_mut() {
                let is_color_alias =
                    property == "color" && matches!(name.as_str(), "color" | "fill");
                if (name == property || is_color_alias) && value.is_string() {
                    *value = serde_json::Value::String(replacement.to_string());
                    *matches += 1;
                } else {
                    apply_repeat_vary_json(value, property, replacement, matches);
                }
            }
        }
        _ => {}
    }
}

fn parse_literal_float_array(
    block: &str,
    name: &str,
    expected_len: usize,
    line: usize,
) -> Result<Option<Vec<f32>>, GraphParseError> {
    let Some(raw) = attr_value(block, name) else {
        return Ok(None);
    };
    let body = strip_wrappers(&raw)
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');
    let values = split_scene_top_level_csv(body)
        .into_iter()
        .map(|value| {
            value.trim().parse::<f32>().map_err(|_| GraphParseError {
                line,
                message: format!("{name} requires literal numeric values; got '{value}'."),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != expected_len {
        return Err(GraphParseError {
            line,
            message: format!(
                "{name} requires {expected_len} numeric values; got {}.",
                values.len()
            ),
        });
    }
    Ok(Some(values))
}

fn parse_literal_string_array(
    block: &str,
    name: &str,
    line: usize,
) -> Result<Option<Vec<String>>, GraphParseError> {
    let Some(raw) = attr_value(block, name) else {
        return Ok(None);
    };
    let body = strip_wrappers(&raw);
    let body = body.trim().trim_start_matches('[').trim_end_matches(']');
    let values = split_scene_top_level_csv(body)
        .into_iter()
        .map(|value| strip_wrappers(&value).trim().to_string())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Err(GraphParseError {
            line,
            message: format!("{name} requires at least one literal value."),
        });
    }
    Ok(Some(values))
}

fn parse_mask_node(
    block: &str,
    _line: usize,
    children: Vec<SceneNode>,
) -> Result<MaskNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let follow = attr_value(block, "follow")
        .or_else(|| attr_value(block, "target"))
        .or_else(|| attr_value(block, "followTarget"))
        .or_else(|| attr_value(block, "follow_target"))
        .map(|v| strip_wrappers(&v).to_string());
    let shape = attr_value(block, "shape")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "rect".to_string());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let width = attr_value(block, "width")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1920".to_string());
    let height = attr_value(block, "height")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1080".to_string());
    let radius = attr_value(block, "radius")
        .or_else(|| attr_value(block, "r"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let d = attr_value(block, "d").map(|v| strip_wrappers(&v).to_string());
    let feather = attr_value(block, "feather")
        .or_else(|| attr_value(block, "softness"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    Ok(MaskNode {
        id,
        follow,
        shape,
        x,
        y,
        width,
        height,
        radius,
        d,
        feather,
        opacity,
        children,
    })
}

fn parse_scene_layer_block(
    lines: &[&str],
    start: usize,
    brush_ctx: &BrushParseContext,
    tag_name: &str,
    is_3d: bool,
) -> Result<(SceneLayerNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, tag_name)?;
    let mut child_ctx = brush_ctx.clone();
    let children = parse_scene_nodes(lines, open_end_ix + 1, close_ix, &mut child_ctx)?;
    Ok((
        parse_scene_layer_node(&open_tag, start + 1, children, false, is_3d)?,
        close_ix,
    ))
}

fn parse_scene_layer_node(
    block: &str,
    line: usize,
    children: Vec<SceneNode>,
    require_source: bool,
    is_3d: bool,
) -> Result<SceneLayerNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let source = attr_value(block, "source")
        .or_else(|| attr_value(block, "src"))
        .or_else(|| attr_value(block, "from"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    if require_source && source.is_none() {
        return Err(GraphParseError {
            line,
            message: "Scene <Layer> requires source=\"precompose_id\".".to_string(),
        });
    }
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let z = attr_value(block, "z")
        .or_else(|| attr_value(block, "translateZ"))
        .or_else(|| attr_value(block, "translate_z"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let rotation_x = attr_value(block, "rotationX")
        .or_else(|| attr_value(block, "rotation_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let rotation_y = attr_value(block, "rotationY")
        .or_else(|| attr_value(block, "rotation_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let rotation = attr_value(block, "rotation")
        .or_else(|| attr_value(block, "rotationZ"))
        .or_else(|| attr_value(block, "rotation_z"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let perspective = attr_value(block, "perspective")
        .or_else(|| attr_value(block, "cameraDistance"))
        .or_else(|| attr_value(block, "camera_distance"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "900".to_string());
    let scale = attr_value(block, "scale")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let scale_x = attr_value(block, "scaleX")
        .or_else(|| attr_value(block, "scale_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let scale_y = attr_value(block, "scaleY")
        .or_else(|| attr_value(block, "scale_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let skew_x = attr_value(block, "skewX")
        .or_else(|| attr_value(block, "skew_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let skew_y = attr_value(block, "skewY")
        .or_else(|| attr_value(block, "skew_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let transform_origin_x = attr_value(block, "transformOriginX")
        .or_else(|| attr_value(block, "transform_origin_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let transform_origin_y = attr_value(block, "transformOriginY")
        .or_else(|| attr_value(block, "transform_origin_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let z_depth = attr_value(block, "zDepth")
        .or_else(|| attr_value(block, "z_depth"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let blend = attr_value(block, "blend")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "normal".to_string());
    let effect = attr_value(block, "effect")
        .or_else(|| attr_value(block, "filter"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    let source_time = attr_value(block, "sourceTime")
        .or_else(|| attr_value(block, "source_time"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "local".to_string());
    let time_offset_ms = attr_value(block, "timeOffset")
        .or_else(|| attr_value(block, "time_offset"))
        .or_else(|| attr_value(block, "sourceTimeOffset"))
        .or_else(|| attr_value(block, "source_time_offset"))
        .as_deref()
        .map(|v| parse_signed_time_ms(v, line, "Layer.timeOffset"))
        .transpose()?
        .unwrap_or(0);
    let playback_rate = attr_value(block, "playbackRate")
        .or_else(|| attr_value(block, "playback_rate"))
        .or_else(|| attr_value(block, "speed"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1".to_string());
    let out = attr_value(block, "out")
        .or_else(|| attr_value(block, "sourceOut"))
        .or_else(|| attr_value(block, "source_out"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "hold".to_string());
    let mask = attr_value(block, "mask")
        .or_else(|| attr_value(block, "maskId"))
        .or_else(|| attr_value(block, "mask_id"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    let mask_from = attr_value(block, "maskFrom")
        .or_else(|| attr_value(block, "mask_from"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    let mask_mode = attr_value(block, "maskMode")
        .or_else(|| attr_value(block, "mask_mode"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "alpha".to_string());
    let mask_feather =
        scene_attr_or_default(block, &["maskFeather", "mask_feather", "feather"], "0");
    let mask_expansion = scene_attr_or_default(
        block,
        &["maskExpansion", "mask_expansion", "expansion"],
        "0",
    );
    let matte = attr_value(block, "matte")
        .or_else(|| attr_value(block, "trackMatte"))
        .or_else(|| attr_value(block, "track_matte"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    let matte_from = attr_value(block, "matteFrom")
        .or_else(|| attr_value(block, "matte_from"))
        .map(|v| strip_wrappers(&v).to_string())
        .filter(|v| !v.trim().is_empty());
    let matte_mode = attr_value(block, "matteMode")
        .or_else(|| attr_value(block, "matte_mode"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "alpha".to_string());
    let invert_matte = attr_value(block, "invertMatte")
        .or_else(|| attr_value(block, "invert_matte"))
        .or_else(|| attr_value(block, "matteInvert"))
        .or_else(|| attr_value(block, "matte_invert"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "false".to_string());

    Ok(SceneLayerNode {
        id,
        source,
        is_3d,
        x,
        y,
        z,
        rotation_x,
        rotation_y,
        rotation,
        perspective,
        scale,
        scale_x,
        scale_y,
        skew_x,
        skew_y,
        transform_origin_x,
        transform_origin_y,
        z_depth,
        opacity,
        blend,
        effect,
        source_time,
        time_offset_ms,
        playback_rate,
        out,
        mask,
        mask_from,
        mask_mode,
        mask_feather,
        mask_expansion,
        matte,
        matte_from,
        matte_mode,
        invert_matte,
        children,
    })
}

pub(crate) fn parse_camera_node(
    block: &str,
    line: usize,
    children: Vec<SceneNode>,
) -> Result<CameraNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    if attr_value(block, "mode").is_some() {
        return Err(GraphParseError {
            line,
            message: "<Scene> Camera is the 2D Scene Camera; remove mode=\"...\". Use <World><Camera> for 3D/world cameras.".to_string(),
        });
    }
    let x = attr_value(block, "x")
        .or_else(|| attr_value(block, "positionX"))
        .or_else(|| attr_value(block, "centerX"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "center".to_string());
    let y = attr_value(block, "y")
        .or_else(|| attr_value(block, "positionY"))
        .or_else(|| attr_value(block, "centerY"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "center".to_string());
    let target_x = attr_value(block, "targetX")
        .or_else(|| attr_value(block, "target_x"))
        .or_else(|| attr_value(block, "pointOfInterestX"))
        .or_else(|| attr_value(block, "focusX"))
        .map(|v| strip_wrappers(&v).to_string());
    let target_y = attr_value(block, "targetY")
        .or_else(|| attr_value(block, "target_y"))
        .or_else(|| attr_value(block, "pointOfInterestY"))
        .or_else(|| attr_value(block, "focusY"))
        .map(|v| strip_wrappers(&v).to_string());
    let anchor_x = attr_value(block, "anchorX")
        .or_else(|| attr_value(block, "anchor_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.5".to_string());
    let anchor_y = attr_value(block, "anchorY")
        .or_else(|| attr_value(block, "anchor_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0.5".to_string());
    let offset_x = attr_value(block, "offsetX")
        .or_else(|| attr_value(block, "offset_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let offset_y = attr_value(block, "offsetY")
        .or_else(|| attr_value(block, "offset_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let shake_x = attr_value(block, "shakeX")
        .or_else(|| attr_value(block, "shake_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let shake_y = attr_value(block, "shakeY")
        .or_else(|| attr_value(block, "shake_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let zoom = attr_value(block, "zoom")
        .or_else(|| attr_value(block, "scale"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let rotation = attr_value(block, "rotation")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let follow = attr_value(block, "follow")
        .or_else(|| attr_value(block, "target"))
        .or_else(|| attr_value(block, "followTarget"))
        .map(|v| strip_wrappers(&v).to_string());
    let dead_zone = attr_value(block, "deadZone")
        .or_else(|| attr_value(block, "dead_zone"))
        .or_else(|| attr_value(block, "dragMargin"))
        .map(|v| strip_wrappers(&v).to_string());
    let viewport = attr_value(block, "viewport")
        .or_else(|| attr_value(block, "crop"))
        .map(|v| strip_wrappers(&v).to_string());
    let world_bounds = attr_value(block, "worldBounds")
        .or_else(|| attr_value(block, "world_bounds"))
        .or_else(|| attr_value(block, "limit"))
        .or_else(|| attr_value(block, "limits"))
        .map(|v| strip_wrappers(&v).to_string());

    Ok(CameraNode {
        id,
        x,
        y,
        target_x,
        target_y,
        anchor_x,
        anchor_y,
        offset_x,
        offset_y,
        shake_x,
        shake_y,
        zoom,
        rotation,
        opacity,
        follow,
        dead_zone,
        viewport,
        world_bounds,
        children,
    })
}

fn parse_character_node(
    block: &str,
    _line: usize,
    children: Vec<SceneNode>,
) -> Result<CharacterNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let src = attr_value(block, "src")
        .or_else(|| attr_value(block, "image"))
        .or_else(|| attr_value(block, "path"))
        .map(|v| strip_wrappers(&v).to_string());
    let rig = attr_value(block, "rig")
        .or_else(|| attr_value(block, "skeleton"))
        .map(|v| strip_wrappers(&v).to_string());
    let model_profile = attr_value(block, "modelProfile")
        .or_else(|| attr_value(block, "model_profile"))
        .or_else(|| attr_value(block, "profile"))
        .map(|v| strip_wrappers(&v).to_string());
    let x = attr_value(block, "x")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let y = attr_value(block, "y")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let rotation = attr_value(block, "rotation")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let scale = attr_value(block, "scale")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let scale_x = attr_value(block, "scaleX")
        .or_else(|| attr_value(block, "scale_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let scale_y = attr_value(block, "scaleY")
        .or_else(|| attr_value(block, "scale_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    let skew_x = attr_value(block, "skewX")
        .or_else(|| attr_value(block, "skew_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let skew_y = attr_value(block, "skewY")
        .or_else(|| attr_value(block, "skew_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let transform_origin_x = attr_value(block, "transformOriginX")
        .or_else(|| attr_value(block, "transform_origin_x"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let transform_origin_y = attr_value(block, "transformOriginY")
        .or_else(|| attr_value(block, "transform_origin_y"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "0".to_string());
    let opacity = attr_value(block, "opacity")
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "1.0".to_string());
    Ok(CharacterNode {
        id,
        src,
        rig,
        model_profile,
        x,
        y,
        rotation,
        scale,
        scale_x,
        scale_y,
        skew_x,
        skew_y,
        transform_origin_x,
        transform_origin_y,
        opacity,
        children,
    })
}
