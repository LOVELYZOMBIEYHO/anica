// =========================================
// =========================================
// crates/motionloom/src/scene/editor_keyframes.rs

use crate::dsl::{AnimationKeyNode, AnimationTargetNode, parse_graph_script, parse_time_seconds};
use crate::error::GraphParseError;
use crate::scene::animation::{animation_property_descriptor, validate_animation_key_value};
use std::error::Error;
use std::fmt;

/// Editor-facing keyframe timeline extracted from MotionLoom graph DSL.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableAnimationTimeline {
    pub fps: f32,
    pub targets: Vec<EditableAnimationTarget>,
}

/// One editable `AnimationTarget` channel for a node/property pair.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableAnimationTarget {
    pub node: String,
    pub property: String,
    pub keys: Vec<EditableAnimationKey>,
}

/// One timed key for an editable `AnimationTarget` channel.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditableAnimationKey {
    pub frame: u32,
    #[serde(default)]
    pub time: Option<String>,
    pub value: String,
    pub ease: String,
}

/// Typed errors for editor keyframe extraction and write-back.
#[derive(Debug, Clone, PartialEq)]
pub enum AnimationKeyframeEditError {
    Parse(GraphParseError),
    MissingGraphClose,
    MissingGraphPresent,
    InvalidTarget {
        node: String,
        property: String,
        reason: &'static str,
    },
    InvalidKey {
        node: String,
        property: String,
        frame: u32,
        reason: &'static str,
    },
}

impl fmt::Display for AnimationKeyframeEditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(err) => write!(f, "{err}"),
            Self::MissingGraphClose => write!(f, "MotionLoom graph is missing </Graph>."),
            Self::MissingGraphPresent => {
                write!(
                    f,
                    "MotionLoom graph is missing a final <Present ... /> node."
                )
            }
            Self::InvalidTarget {
                node,
                property,
                reason,
            } => write!(
                f,
                "Invalid AnimationTarget node={node:?} property={property:?}: {reason}"
            ),
            Self::InvalidKey {
                node,
                property,
                frame,
                reason,
            } => write!(
                f,
                "Invalid AnimationTarget key node={node:?} property={property:?} frame={frame}: {reason}"
            ),
        }
    }
}

impl Error for AnimationKeyframeEditError {}

impl From<GraphParseError> for AnimationKeyframeEditError {
    fn from(value: GraphParseError) -> Self {
        Self::Parse(value)
    }
}

impl From<&AnimationTargetNode> for EditableAnimationTarget {
    fn from(value: &AnimationTargetNode) -> Self {
        Self {
            node: value.node.clone(),
            property: value.property.clone(),
            keys: value.keys.iter().map(EditableAnimationKey::from).collect(),
        }
    }
}

impl From<&AnimationKeyNode> for EditableAnimationKey {
    fn from(value: &AnimationKeyNode) -> Self {
        Self {
            frame: value.frame,
            time: value.time.clone(),
            value: value.value.clone(),
            ease: value.ease.clone(),
        }
    }
}

/// Parse a MotionLoom script and return the UI-editable AnimationTarget model.
pub fn extract_editable_animation_timeline(
    script: &str,
) -> Result<EditableAnimationTimeline, AnimationKeyframeEditError> {
    let graph = parse_graph_script(script)?;
    Ok(EditableAnimationTimeline {
        fps: graph.fps,
        targets: graph
            .animation_targets
            .iter()
            .map(EditableAnimationTarget::from)
            .collect(),
    })
}

/// Replace every existing AnimationTarget block with the supplied UI model.
pub fn replace_editable_animation_targets(
    script: &str,
    targets: &[EditableAnimationTarget],
) -> Result<String, AnimationKeyframeEditError> {
    for target in targets {
        validate_editable_target(target)?;
    }

    let stripped = strip_animation_target_blocks(script);
    let insertion = render_animation_target_blocks(targets);
    let output = insert_before_final_present(&stripped, &insertion)?;

    // Re-parse the generated script so UI write-back cannot emit invalid DSL.
    parse_graph_script(&output)?;
    Ok(output)
}

/// Replace or add a single node/property channel while preserving other channels.
pub fn upsert_editable_animation_target(
    script: &str,
    target: EditableAnimationTarget,
) -> Result<String, AnimationKeyframeEditError> {
    validate_editable_target(&target)?;
    // Patch one channel in place so comments and unrelated author formatting
    // survive a visual-editor keyframe edit.
    let replacement = render_animation_target_block(&target);
    let output = if let Some((start, end)) =
        find_animation_target_block_span(script, &target.node, &target.property)
    {
        format!(
            "{}{}{}",
            &script[..start],
            replacement.trim_end(),
            &script[end..]
        )
    } else {
        insert_before_final_present(script, &format!("{replacement}\n"))?
    };
    parse_graph_script(&output)?;
    Ok(output)
}

/// Return one editor channel without forcing callers to scan the whole timeline.
pub fn editable_animation_target(
    script: &str,
    node: &str,
    property: &str,
) -> Result<Option<EditableAnimationTarget>, AnimationKeyframeEditError> {
    Ok(extract_editable_animation_timeline(script)?
        .targets
        .into_iter()
        .find(|target| target.node == node && target.property == property))
}

/// Remove one channel while preserving all unrelated script text.
pub fn remove_editable_animation_target(
    script: &str,
    node: &str,
    property: &str,
) -> Result<String, AnimationKeyframeEditError> {
    let Some((start, end)) = find_animation_target_block_span(script, node, property) else {
        parse_graph_script(script)?;
        return Ok(script.to_string());
    };
    let mut output = String::with_capacity(script.len().saturating_sub(end - start));
    output.push_str(&script[..start]);
    output.push_str(&script[end..]);
    parse_graph_script(&output)?;
    Ok(output)
}

/// Promote an existing inline numeric `curve(...)` attribute into an editable channel.
///
/// The original attribute stays in place as the authored fallback;
/// `AnimationTarget` has explicit editor-override precedence at render time.
pub fn promote_inline_curve_to_animation_target(
    script: &str,
    node: &str,
    property: &str,
) -> Result<String, AnimationKeyframeEditError> {
    parse_graph_script(script)?;
    let expression = find_inline_property_expression(script, node, property).ok_or_else(|| {
        AnimationKeyframeEditError::InvalidTarget {
            node: node.to_string(),
            property: property.to_string(),
            reason: "inline property expression was not found",
        }
    })?;
    let keys = editable_keys_from_curve(expression).map_err(|reason| {
        AnimationKeyframeEditError::InvalidTarget {
            node: node.to_string(),
            property: property.to_string(),
            reason,
        }
    })?;
    upsert_editable_animation_target(
        script,
        EditableAnimationTarget {
            node: node.to_string(),
            property: property.to_string(),
            keys,
        },
    )
}

/// Render one numeric editor channel as a compact inline `curve(...)` expression.
pub fn animation_target_inline_curve_expression(
    target: &EditableAnimationTarget,
    fps: f32,
) -> Result<String, AnimationKeyframeEditError> {
    validate_editable_target(target)?;
    let descriptor = animation_property_descriptor(&target.property).ok_or_else(|| {
        AnimationKeyframeEditError::InvalidTarget {
            node: target.node.clone(),
            property: target.property.clone(),
            reason: "unsupported property",
        }
    })?;
    if descriptor.value_type != crate::scene::animation::AnimationValueType::Number {
        return Err(AnimationKeyframeEditError::InvalidTarget {
            node: target.node.clone(),
            property: target.property.clone(),
            reason: "only numeric channels can be converted to inline curve",
        });
    }
    let mut keys = target.keys.clone();
    keys.sort_by(|a, b| editable_key_seconds(a, fps).total_cmp(&editable_key_seconds(b, fps)));
    let points = keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            let ease = keys.get(index + 1).unwrap_or(key).ease.as_str();
            format!(
                "{}:{}:{}",
                format_editor_number(editable_key_seconds(key, fps)),
                key.value,
                ease
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!("curve(\"{points}\")"))
}

fn validate_editable_target(
    target: &EditableAnimationTarget,
) -> Result<(), AnimationKeyframeEditError> {
    if target.node.trim().is_empty() {
        return Err(AnimationKeyframeEditError::InvalidTarget {
            node: target.node.clone(),
            property: target.property.clone(),
            reason: "node id is empty",
        });
    }
    if animation_property_descriptor(&target.property).is_none() {
        return Err(AnimationKeyframeEditError::InvalidTarget {
            node: target.node.clone(),
            property: target.property.clone(),
            reason: "unsupported property",
        });
    }
    if contains_unsafe_attr_text(&target.node) || contains_unsafe_attr_text(&target.property) {
        return Err(AnimationKeyframeEditError::InvalidTarget {
            node: target.node.clone(),
            property: target.property.clone(),
            reason: "node or property contains unsupported attribute characters",
        });
    }
    if target.keys.is_empty() {
        return Err(AnimationKeyframeEditError::InvalidTarget {
            node: target.node.clone(),
            property: target.property.clone(),
            reason: "target requires at least one key",
        });
    }
    for key in &target.keys {
        validate_editable_key(target, key)?;
    }
    Ok(())
}

fn validate_editable_key(
    target: &EditableAnimationTarget,
    key: &EditableAnimationKey,
) -> Result<(), AnimationKeyframeEditError> {
    if key.ease.trim().is_empty()
        || contains_unsafe_attr_text(&key.ease)
        || contains_unsafe_attr_text(&key.value)
        || key
            .time
            .as_ref()
            .is_some_and(|time| contains_unsafe_attr_text(time))
    {
        return Err(AnimationKeyframeEditError::InvalidKey {
            node: target.node.clone(),
            property: target.property.clone(),
            frame: key.frame,
            reason: "value or ease contains unsupported attribute characters",
        });
    }
    if let Some(time) = key.time.as_ref() {
        parse_time_seconds(time, 0, "Key.time").map_err(|_| {
            AnimationKeyframeEditError::InvalidKey {
                node: target.node.clone(),
                property: target.property.clone(),
                frame: key.frame,
                reason: "time must be a valid non-negative time value",
            }
        })?;
    }
    let descriptor = animation_property_descriptor(&target.property).ok_or_else(|| {
        AnimationKeyframeEditError::InvalidTarget {
            node: target.node.clone(),
            property: target.property.clone(),
            reason: "unsupported property",
        }
    })?;
    if let Err(reason) = validate_animation_key_value(descriptor, &key.value) {
        return Err(AnimationKeyframeEditError::InvalidKey {
            node: target.node.clone(),
            property: target.property.clone(),
            frame: key.frame,
            reason,
        });
    }
    Ok(())
}

fn contains_unsafe_attr_text(value: &str) -> bool {
    value.contains('"') || value.contains('\n') || value.contains('\r')
}

fn strip_animation_target_blocks(script: &str) -> String {
    let mut output = Vec::<String>::new();
    let mut skipping = false;
    for line in script.lines() {
        let trimmed = line.trim_start();
        if !skipping && trimmed.starts_with("<AnimationTarget") {
            skipping = !trimmed.contains("</AnimationTarget>");
            continue;
        }
        if skipping {
            if trimmed.contains("</AnimationTarget>") {
                skipping = false;
            }
            continue;
        }
        output.push(line.to_string());
    }

    let mut result = output.join("\n");
    if script.ends_with('\n') {
        result.push('\n');
    }
    result
}

fn insert_before_final_present(
    script: &str,
    insertion: &str,
) -> Result<String, AnimationKeyframeEditError> {
    if !script.contains("</Graph>") {
        return Err(AnimationKeyframeEditError::MissingGraphClose);
    }
    if insertion.trim().is_empty() {
        return Ok(script.to_string());
    }

    let Some(index) = script.rfind("<Present") else {
        return Err(AnimationKeyframeEditError::MissingGraphPresent);
    };
    let before = script[..index].trim_end();
    let after = &script[index..];
    Ok(format!("{before}\n\n{insertion}{after}"))
}

fn render_animation_target_blocks(targets: &[EditableAnimationTarget]) -> String {
    let mut sorted = targets.to_vec();
    sorted.sort_by(|a, b| {
        a.node
            .cmp(&b.node)
            .then_with(|| a.property.cmp(&b.property))
    });

    let mut output = String::new();
    for target in sorted {
        output.push_str(&render_animation_target_block(&target));
    }
    output.push('\n');
    output
}

fn render_animation_target_block(target: &EditableAnimationTarget) -> String {
    let mut output = String::new();
    let mut keys = target.keys.clone();
    keys.sort_by(|a, b| {
        editable_key_sort_seconds(a)
            .partial_cmp(&editable_key_sort_seconds(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.frame.cmp(&b.frame))
    });
    output.push_str(&format!(
        "  <AnimationTarget node=\"{}\" property=\"{}\">\n",
        target.node, target.property
    ));
    for key in keys {
        if let Some(time) = key.time.as_ref() {
            output.push_str(&format!(
                "    <Key time=\"{}\" value=\"{}\" ease=\"{}\" />\n",
                time, key.value, key.ease
            ));
        } else {
            output.push_str(&format!(
                "    <Key frame=\"{}\" value=\"{}\" ease=\"{}\" />\n",
                key.frame, key.value, key.ease
            ));
        }
    }
    output.push_str("  </AnimationTarget>\n");
    output
}

fn find_animation_target_block_span(
    script: &str,
    node: &str,
    property: &str,
) -> Option<(usize, usize)> {
    let mut search_from = 0usize;
    while let Some(relative_start) = script[search_from..].find("<AnimationTarget") {
        let start = search_from + relative_start;
        let open_end = start + script[start..].find('>')? + 1;
        let open = &script[start..open_end];
        let close_relative = script[open_end..].find("</AnimationTarget>")?;
        let end = open_end + close_relative + "</AnimationTarget>".len();
        if animation_attr(open, "node") == Some(node)
            && animation_attr(open, "property") == Some(property)
        {
            return Some((start, end));
        }
        search_from = end;
    }
    None
}

fn animation_attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = start + tag[start..].find('"')?;
    Some(&tag[start..end])
}

fn find_inline_property_expression<'a>(
    script: &'a str,
    node: &str,
    property: &str,
) -> Option<&'a str> {
    let id_needle = format!("id=\"{node}\"");
    let id_index = script.find(&id_needle)?;
    let tag_start = script[..id_index].rfind('<')?;
    let tag_end = id_index + script[id_index..].find('>')?;
    let tag = &script[tag_start..tag_end];
    let property_start = tag.find(&format!("{property}="))? + property.len() + 1;
    let rest = &tag[property_start..];
    if let Some(rest) = rest.strip_prefix('{') {
        let end = matching_delimiter_end(rest, '{', '}')?;
        return Some(rest[..end].trim());
    }
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].trim())
}

fn matching_delimiter_end(value: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 1usize;
    let mut quote = None;
    for (index, ch) in value.char_indices() {
        if let Some(active) = quote {
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => quote = Some(ch),
            value if value == open => depth += 1,
            value if value == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn editable_keys_from_curve(expression: &str) -> Result<Vec<EditableAnimationKey>, &'static str> {
    let expression = expression.trim();
    let inner = expression
        .strip_prefix("curve(")
        .and_then(|value| value.strip_suffix(')'))
        .ok_or("property is not an inline curve(...) expression")?
        .trim()
        .trim_matches('"');
    let tokens = split_editor_curve_tokens(inner)?;
    let mut points = Vec::<(f32, String, String)>::new();
    for token in tokens {
        let parts = token.splitn(3, ':').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 2 {
            return Err("curve point must use time:value[:ease]");
        }
        let seconds = parts[0]
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or("curve time must be a non-negative number")?;
        parts[1]
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite())
            .ok_or("curve value must be numeric")?;
        points.push((
            seconds,
            parts[1].to_string(),
            parts.get(2).copied().unwrap_or("linear").to_string(),
        ));
    }
    if points.is_empty() {
        return Err("curve requires at least one point");
    }
    Ok(points
        .iter()
        .enumerate()
        .map(|(index, (seconds, value, _))| EditableAnimationKey {
            frame: 0,
            time: Some(format!("{}s", format_editor_number(*seconds))),
            value: value.clone(),
            // Inline curves store easing on the source point; editor channels
            // store it on the destination key for the same rendered segment.
            ease: index
                .checked_sub(1)
                .and_then(|previous| points.get(previous))
                .map(|point| point.2.clone())
                .unwrap_or_else(|| "linear".to_string()),
        })
        .collect())
}

fn split_editor_curve_tokens(value: &str) -> Result<Vec<&str>, &'static str> {
    let mut tokens = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    for (index, ch) in value.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or("curve has unmatched parentheses")?
            }
            ',' if depth == 0 => {
                tokens.push(value[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err("curve has unmatched parentheses");
    }
    tokens.push(value[start..].trim());
    Ok(tokens)
}

fn editable_key_seconds(key: &EditableAnimationKey, fps: f32) -> f32 {
    key.time
        .as_ref()
        .and_then(|time| parse_time_seconds(time, 0, "Key.time").ok())
        .unwrap_or(key.frame as f32 / fps.max(1.0))
}

fn format_editor_number(value: f32) -> String {
    let mut output = format!("{value:.4}");
    while output.contains('.') && output.ends_with('0') {
        output.pop();
    }
    if output.ends_with('.') {
        output.pop();
    }
    output
}

fn editable_key_sort_seconds(key: &EditableAnimationKey) -> f32 {
    key.time
        .as_ref()
        .and_then(|time| parse_time_seconds(time, 0, "Key.time").ok())
        .unwrap_or(key.frame as f32)
}

#[cfg(test)]
mod tests {
    use super::{
        EditableAnimationKey, EditableAnimationTarget, animation_target_inline_curve_expression,
        editable_animation_target, extract_editable_animation_timeline,
        promote_inline_curve_to_animation_target, remove_editable_animation_target,
        replace_editable_animation_targets, upsert_editable_animation_target,
    };

    const SCRIPT: &str = r##"<Graph fps={30} duration="1s" size={[100,100]}>
  <Scene id="scene0">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Group id="card" x="0" y="0">
              <Rect x="0" y="0" width="10" height="10" color="#fff" />
            </Group>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <AnimationTarget node="card" property="rotation">
    <Key frame="0" value="0" ease="linear" />
    <Key frame="15" value="18" ease="ease_in_out" />
  </AnimationTarget>

  <Present from="scene0" />
</Graph>
"##;

    #[test]
    fn extracts_editable_animation_timeline() {
        let timeline = extract_editable_animation_timeline(SCRIPT).unwrap();
        assert_eq!(timeline.fps, 30.0);
        assert_eq!(timeline.targets.len(), 1);
        assert_eq!(timeline.targets[0].node, "card");
        assert_eq!(timeline.targets[0].property, "rotation");
        assert_eq!(timeline.targets[0].keys[1].frame, 15);
    }

    #[test]
    fn replaces_animation_targets_and_keeps_script_parseable() {
        let output = replace_editable_animation_targets(
            SCRIPT,
            &[EditableAnimationTarget {
                node: "card".to_string(),
                property: "x".to_string(),
                keys: vec![
                    EditableAnimationKey {
                        frame: 20,
                        time: None,
                        value: "50".to_string(),
                        ease: "ease_out".to_string(),
                    },
                    EditableAnimationKey {
                        frame: 0,
                        time: None,
                        value: "0".to_string(),
                        ease: "linear".to_string(),
                    },
                ],
            }],
        )
        .unwrap();
        let timeline = extract_editable_animation_timeline(&output).unwrap();
        assert_eq!(timeline.targets.len(), 1);
        assert_eq!(timeline.targets[0].property, "x");
        assert_eq!(timeline.targets[0].keys[0].frame, 0);
        assert!(output.contains("<Present from=\"scene0\" />"));
    }

    #[test]
    fn upserts_one_target_and_preserves_other_channels() {
        let output = upsert_editable_animation_target(
            SCRIPT,
            EditableAnimationTarget {
                node: "card".to_string(),
                property: "x".to_string(),
                keys: vec![EditableAnimationKey {
                    frame: 0,
                    time: None,
                    value: "12".to_string(),
                    ease: "linear".to_string(),
                }],
            },
        )
        .unwrap();
        let timeline = extract_editable_animation_timeline(&output).unwrap();
        assert_eq!(timeline.targets.len(), 2);
        assert!(
            timeline
                .targets
                .iter()
                .any(|target| target.property == "rotation")
        );
        assert!(timeline.targets.iter().any(|target| target.property == "x"));
    }

    #[test]
    fn accepts_extended_transform_properties() {
        let output = upsert_editable_animation_target(
            SCRIPT,
            EditableAnimationTarget {
                node: "card".to_string(),
                property: "skewX".to_string(),
                keys: vec![EditableAnimationKey {
                    frame: 0,
                    time: Some("0s".to_string()),
                    value: "-30".to_string(),
                    ease: "linear".to_string(),
                }],
            },
        )
        .unwrap();
        assert!(output.contains("<Key time=\"0s\" value=\"-30\" ease=\"linear\" />"));
        let timeline = extract_editable_animation_timeline(&output).unwrap();
        assert!(
            timeline
                .targets
                .iter()
                .any(|target| target.property == "skewX")
        );
    }

    #[test]
    fn extracts_time_keys_with_compat_frame() {
        let script = r##"<Graph fps={60} duration="1s" size={[100,100]}>
  <Scene id="scene0">
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
    <Key time="0.5s" value="20" ease="linear" />
  </AnimationTarget>

  <Present from="scene0" />
</Graph>
"##;
        let timeline = extract_editable_animation_timeline(script).unwrap();
        assert_eq!(timeline.targets[0].keys[0].time.as_deref(), Some("0.5s"));
        assert_eq!(timeline.targets[0].keys[0].frame, 30);
    }

    #[test]
    fn upsert_patches_only_the_selected_channel() {
        let script = SCRIPT.replace(
            "  <AnimationTarget node=\"card\" property=\"rotation\">",
            "  <!-- keep-channel-comment -->\n  <AnimationTarget node=\"card\" property=\"rotation\">",
        );
        let output = upsert_editable_animation_target(
            &script,
            EditableAnimationTarget {
                node: "card".to_string(),
                property: "rotation".to_string(),
                keys: vec![EditableAnimationKey {
                    frame: 12,
                    time: None,
                    value: "9".to_string(),
                    ease: "linear".to_string(),
                }],
            },
        )
        .unwrap();
        assert!(output.contains("<!-- keep-channel-comment -->"));
        assert!(output.contains("frame=\"12\" value=\"9\""));
        assert_eq!(
            editable_animation_target(&output, "card", "rotation")
                .unwrap()
                .unwrap()
                .keys
                .len(),
            1
        );
    }

    #[test]
    fn removes_one_channel_without_rewriting_the_scene() {
        let output = remove_editable_animation_target(SCRIPT, "card", "rotation").unwrap();
        assert!(!output.contains("<AnimationTarget"));
        assert!(output.contains("<Group id=\"card\" x=\"0\" y=\"0\">"));
    }

    #[test]
    fn accepts_typed_color_and_effect_parameter_channels() {
        let color = EditableAnimationTarget {
            node: "card".to_string(),
            property: "color".to_string(),
            keys: vec![EditableAnimationKey {
                frame: 0,
                time: None,
                value: "#ff00aa".to_string(),
                ease: "linear".to_string(),
            }],
        };
        let output = upsert_editable_animation_target(SCRIPT, color).unwrap();
        assert!(output.contains("property=\"color\""));
    }

    #[test]
    fn promotes_inline_curve_and_can_emit_compact_expression() {
        let script = SCRIPT.replace(
            "<Group id=\"card\" x=\"0\" y=\"0\">",
            "<Group id=\"card\" x={curve(\"0:0:ease_in, 1:20:linear\")} y=\"0\">",
        );
        let output = promote_inline_curve_to_animation_target(&script, "card", "x").unwrap();
        let target = editable_animation_target(&output, "card", "x")
            .unwrap()
            .unwrap();
        assert_eq!(target.keys.len(), 2);
        assert_eq!(target.keys[1].ease, "ease_in");
        assert_eq!(
            animation_target_inline_curve_expression(&target, 30.0).unwrap(),
            "curve(\"0:0:ease_in, 1:20:ease_in\")"
        );
    }
}
