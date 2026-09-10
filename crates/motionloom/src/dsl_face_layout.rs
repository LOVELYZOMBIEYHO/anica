// =========================================
// =========================================
// crates/motionloom/src/dsl_face_layout.rs

use super::*;

/// Components are the only source of facial parameters; flat attributes are rejected.
pub(super) fn parse(
    lines: &[&str],
    start: usize,
    asset: &str,
) -> Result<(FaceLayoutNode, usize), GraphParseError> {
    let (tag, end) = collect_tag_block(lines, start, '>', false)?;
    validate_head_attributes(&tag, &[], "FaceLayout", start + 1)?;
    let mut layout = FaceLayoutNode {
        eyebrows: vec![],
        eyes: vec![],
        noses: vec![],
        mouths: vec![],
        ears: vec![],
    };
    if is_self_closing_tag(&tag) {
        return Ok((layout, end));
    }
    let close = find_matching_close_tag(lines, end + 1, "FaceLayout")?;
    let mut index = end + 1;
    let mut ids = HashSet::new();
    while index < close {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("<!--") || line.starts_with("//") {
            index += 1;
            continue;
        }
        let (tag, end) = collect_tag_block(lines, index, '>', false)?;
        let kind = ["Eye", "Eyebrow", "Nose", "Mouth", "Ear"]
            .into_iter()
            .find(|kind| starts_open_tag(&tag, kind))
            .ok_or_else(|| GraphParseError {
                line: index + 1,
                message: "FaceLayout accepts Eye, Eyebrow, Nose, Mouth, and Ear children.".into(),
            })?;
        let specific: &[&str] = match kind {
            "Eyebrow" => &["width", "thickness", "arch", "tilt"],
            "Eye" => &[
                "width",
                "opening",
                "tilt",
                "socketWidth",
                "socketHeight",
                "socketDepth",
            ],
            "Nose" => &["length", "width", "projection"],
            "Mouth" => &[
                "width",
                "opening",
                "upperLip",
                "lowerLip",
                "muzzleLength",
                "muzzleWidth",
            ],
            _ => &["width", "height", "depth"],
        };
        let mut allowed = vec!["id", "position"];
        allowed.extend_from_slice(specific);
        validate_head_attributes(&tag, &allowed, kind, index + 1)?;
        let id = strip_wrappers(&required_attr_value(&tag, "id", index + 1)?).to_string();
        if id.trim().is_empty() || !ids.insert(id.clone()) {
            return Err(GraphParseError {
                line: index + 1,
                message: format!("FaceLayout has empty or duplicate component id: {id}"),
            });
        }
        let position =
            parse_optional_primitive_vec::<3>(&tag, "position", asset, index + 1, false)?
                .unwrap_or([0.0; 3]);
        let number = |key, default| parse_head_number(&tag, key, asset, index + 1, default);
        // Positive dimensions are checked before any mesh allocation.
        for key in specific.iter().filter(|key| {
            matches!(
                **key,
                "width" | "height" | "length" | "thickness" | "socketWidth" | "socketHeight"
            )
        }) {
            if number(key, 1.0)? <= 0.0 {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("{kind} \"{id}\" {key} must be positive."),
                });
            }
        }
        for key in specific
            .iter()
            .filter(|key| !matches!(**key, "tilt" | "arch"))
        {
            if number(key, 0.0)? < 0.0 {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("{kind} \"{id}\" {key} must not be negative."),
                });
            }
        }
        let component_end = if is_self_closing_tag(&tag) {
            end
        } else {
            find_matching_close_tag(lines, end + 1, kind)?
        };
        let (texture, iris, eyeliners) = if kind == "Eye" {
            parse_eye_children(lines, end + 1, component_end, asset, &id, &mut ids)?
        } else {
            (
                parse_texture_only(lines, end + 1, component_end, asset, kind, &id)?,
                None,
                vec![],
            )
        };
        match kind {
            "Eyebrow" => layout.eyebrows.push(EyebrowNode {
                id,
                position,
                texture,
                width: number("width", 0.300)?,
                thickness: number("thickness", 0.018)?,
                arch: number("arch", 0.035)?,
                tilt: number("tilt", 0.0)?,
            }),
            "Eye" => layout.eyes.push(EyeNode {
                id,
                position,
                texture,
                width: number("width", 0.34)?,
                opening: number("opening", 0.14)?,
                tilt: number("tilt", 0.0)?,
                socket_width: number("socketWidth", 0.44)?,
                socket_height: number("socketHeight", 0.28)?,
                socket_depth: number("socketDepth", 0.035)?,
                iris,
                eyeliners,
            }),
            "Nose" => layout.noses.push(NoseNode {
                id,
                position,
                texture,
                length: number("length", 0.22)?,
                width: number("width", 0.13)?,
                projection: number("projection", 0.045)?,
            }),
            "Mouth" => layout.mouths.push(MouthNode {
                id,
                position,
                texture,
                width: number("width", 0.34)?,
                opening: number("opening", 0.035)?,
                upper_lip: number("upperLip", 0.025)?,
                lower_lip: number("lowerLip", 0.03)?,
                muzzle_length: number("muzzleLength", 0.0)?,
                muzzle_width: number("muzzleWidth", 0.30)?,
            }),
            _ => layout.ears.push(EarNode {
                id,
                position,
                texture,
                width: number("width", 0.10)?,
                height: number("height", 0.24)?,
                depth: number("depth", 0.045)?,
            }),
        }
        index = component_end + 1;
        if ids.len() > 64 {
            return Err(GraphParseError {
                line: start + 1,
                message: "FaceLayout supports at most 64 components.".into(),
            });
        }
    }
    Ok((layout, close))
}

/// Parse the shared image binding without giving it geometry semantics.
fn parse_texture(
    lines: &[&str],
    start: usize,
    asset: &str,
) -> Result<(FaceTextureNode, usize), GraphParseError> {
    let (tag, last) = collect_self_closing_block(lines, start)?;
    if !starts_open_tag(&tag, "Texture") {
        return Err(GraphParseError {
            line: start + 1,
            message: "Expected a self-closing Texture child.".into(),
        });
    }
    validate_head_attributes(
        &tag,
        &["asset", "offset", "scale", "rotation"],
        "Texture",
        start + 1,
    )?;
    let scale = parse_optional_primitive_vec::<2>(&tag, "scale", asset, start + 1, false)?
        .unwrap_or([1.0; 2]);
    if scale.contains(&0.0) {
        return Err(GraphParseError {
            line: start + 1,
            message: "Texture scale must not be zero.".into(),
        });
    }
    Ok((
        FaceTextureNode {
            source: None,
            asset: strip_wrappers(&required_attr_value(&tag, "asset", start + 1)?).to_string(),
            offset: parse_optional_primitive_vec::<2>(&tag, "offset", asset, start + 1, false)?
                .unwrap_or([0.0; 2]),
            scale,
            rotation: parse_head_number(&tag, "rotation", asset, start + 1, 0.0)?,
        },
        last,
    ))
}

fn ignorable(line: &str) -> bool {
    let line = line.trim();
    line.is_empty() || line.starts_with("<!--") || line.starts_with("//")
}

/// Non-eye components keep the single optional Texture rule.
fn parse_texture_only(
    lines: &[&str],
    start: usize,
    close: usize,
    asset: &str,
    kind: &str,
    id: &str,
) -> Result<Option<FaceTextureNode>, GraphParseError> {
    let mut texture = None;
    let mut child = start;
    while child < close {
        if ignorable(lines[child]) {
            child += 1;
            continue;
        }
        if texture.is_some() {
            return Err(GraphParseError {
                line: child + 1,
                message: format!("{kind} \"{id}\" accepts at most one Texture child."),
            });
        }
        let (value, last) = parse_texture(lines, child, asset)?;
        texture = Some(value);
        child = last + 1;
    }
    Ok(texture)
}

fn nested_id(
    tag: &str,
    kind: &str,
    line: usize,
    ids: &mut HashSet<String>,
) -> Result<String, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(tag, "id", line)?).to_string();
    if id.trim().is_empty() || !ids.insert(id.clone()) {
        return Err(GraphParseError {
            line,
            message: format!("FaceLayout has empty or duplicate {kind} id: {id}"),
        });
    }
    Ok(id)
}

type EyeChildren = (Option<FaceTextureNode>, Option<IrisNode>, Vec<EyelinerNode>);

/// Eye owns one optional Iris, any number of Eyeliner ribbons, and one sclera Texture.
fn parse_eye_children(
    lines: &[&str],
    start: usize,
    close: usize,
    asset: &str,
    eye_id: &str,
    ids: &mut HashSet<String>,
) -> Result<EyeChildren, GraphParseError> {
    let mut texture = None;
    let mut iris = None;
    let mut eyeliners = vec![];
    let mut child = start;
    while child < close {
        if ignorable(lines[child]) {
            child += 1;
            continue;
        }
        let (tag, open_end) = collect_tag_block(lines, child, '>', false)?;
        if starts_open_tag(&tag, "Texture") {
            if texture.is_some() {
                return Err(GraphParseError {
                    line: child + 1,
                    message: format!("Eye \"{eye_id}\" accepts at most one direct Texture child."),
                });
            }
            let (value, last) = parse_texture(lines, child, asset)?;
            texture = Some(value);
            child = last + 1;
            continue;
        }
        if starts_open_tag(&tag, "Iris") {
            if iris.is_some() {
                return Err(GraphParseError {
                    line: child + 1,
                    message: format!("Eye \"{eye_id}\" accepts at most one Iris child."),
                });
            }
            validate_head_attributes(
                &tag,
                &["id", "position", "shape", "scale", "radius", "pupilRadius"],
                "Iris",
                child + 1,
            )?;
            let id = nested_id(&tag, "Iris", child + 1, ids)?;
            let position =
                parse_optional_primitive_vec::<3>(&tag, "position", asset, child + 1, false)?
                    .unwrap_or([0.0; 3]);
            let shape = attr_value(&tag, "shape")
                .map(|value| strip_wrappers(&value).to_ascii_lowercase())
                .unwrap_or_else(|| "circle".into());
            if !matches!(shape.as_str(), "circle" | "ellipse" | "square") {
                return Err(GraphParseError {
                    line: child + 1,
                    message: format!("Iris \"{id}\" shape must be circle, ellipse, or square."),
                });
            }
            let scale = parse_optional_primitive_vec::<2>(&tag, "scale", asset, child + 1, false)?
                .unwrap_or(if shape == "ellipse" {
                    [1.0, 1.25]
                } else {
                    [1.0; 2]
                });
            let radius = parse_head_number(&tag, "radius", asset, child + 1, 0.075)?;
            let pupil_radius = parse_head_number(&tag, "pupilRadius", asset, child + 1, 0.03)?;
            if radius <= 0.0
                || pupil_radius < 0.0
                || pupil_radius > radius
                || scale.iter().any(|value| *value <= 0.0)
            {
                return Err(GraphParseError {
                    line: child + 1,
                    message: format!(
                        "Iris \"{id}\" requires positive radius/scale and 0 <= pupilRadius <= radius."
                    ),
                });
            }
            let component_end = if is_self_closing_tag(&tag) {
                open_end
            } else {
                find_matching_close_tag(lines, open_end + 1, "Iris")?
            };
            let iris_texture =
                parse_texture_only(lines, open_end + 1, component_end, asset, "Iris", &id)?;
            iris = Some(IrisNode {
                id,
                position,
                shape,
                scale,
                radius,
                pupil_radius,
                texture: iris_texture,
            });
            child = component_end + 1;
            continue;
        }
        if starts_open_tag(&tag, "Eyeliner") {
            validate_head_attributes(
                &tag,
                &[
                    "id",
                    "edge",
                    "thickness",
                    "span",
                    "taper",
                    "extension",
                    "tipLift",
                ],
                "Eyeliner",
                child + 1,
            )?;
            let id = nested_id(&tag, "Eyeliner", child + 1, ids)?;
            let edge = strip_wrappers(&required_attr_value(&tag, "edge", child + 1)?).to_string();
            if !matches!(edge.as_str(), "upper" | "lower") {
                return Err(GraphParseError {
                    line: child + 1,
                    message: format!("Eyeliner \"{id}\" edge must be upper or lower."),
                });
            }
            let thickness = parse_head_number(&tag, "thickness", asset, child + 1, 0.01)?;
            let span = parse_optional_primitive_vec::<2>(&tag, "span", asset, child + 1, false)?
                .unwrap_or([0.0, 1.0]);
            let taper = parse_optional_primitive_vec::<2>(&tag, "taper", asset, child + 1, false)?
                .unwrap_or([0.0; 2]);
            let extension =
                parse_optional_primitive_vec::<2>(&tag, "extension", asset, child + 1, false)?
                    .unwrap_or([0.0; 2]);
            let tip_lift =
                parse_optional_primitive_vec::<2>(&tag, "tipLift", asset, child + 1, false)?
                    .unwrap_or([0.0; 2]);
            if thickness <= 0.0
                || span[0] < 0.0
                || span[0] >= span[1]
                || span[1] > 1.0
                || taper.iter().any(|v| !(0.0..=1.0).contains(v))
                || extension.iter().any(|v| *v < 0.0)
            {
                return Err(GraphParseError {
                    line: child + 1,
                    message: format!(
                        "Eyeliner \"{id}\" has invalid thickness, span, taper, or extension."
                    ),
                });
            }
            let component_end = if is_self_closing_tag(&tag) {
                open_end
            } else {
                find_matching_close_tag(lines, open_end + 1, "Eyeliner")?
            };
            let liner_texture =
                parse_texture_only(lines, open_end + 1, component_end, asset, "Eyeliner", &id)?;
            eyeliners.push(EyelinerNode {
                id,
                edge,
                thickness,
                span,
                taper,
                extension,
                tip_lift,
                texture: liner_texture,
            });
            child = component_end + 1;
            continue;
        }
        return Err(GraphParseError {
            line: child + 1,
            message: format!(
                "Eye \"{eye_id}\" accepts Texture, optional Iris, and Eyeliner children."
            ),
        });
    }
    Ok((texture, iris, eyeliners))
}
