// =========================================
// =========================================
// crates/motionloom/src/render_style/parser.rs

//! Strict parsing for render-style resources and material-level cel controls.

use super::model::CelMaterialSettings;
use super::validation::{color, error, one_of, range};
use crate::dsl::{GraphParseError, attr_value, strip_wrappers};

pub(crate) fn parse_cel_material(
    tag: &str,
    line: usize,
) -> Result<CelMaterialSettings, GraphParseError> {
    let scalar = |key: &str| -> Result<Option<f32>, GraphParseError> {
        attr_value(tag, key)
            .map(|v| {
                strip_wrappers(&v)
                    .parse::<f32>()
                    .map_err(|_| GraphParseError {
                        line,
                        message: format!("{key} must be a number"),
                    })
            })
            .transpose()
    };
    let settings = CelMaterialSettings {
        control_map: attr_value(tag, "celControlMap").map(|v| strip_wrappers(&v).to_owned()),
        role: attr_value(tag, "celRole").map(|v| strip_wrappers(&v).to_owned()),
        shadow_color: attr_value(tag, "celShadowColor").map(|v| strip_wrappers(&v).to_owned()),
        outline_width: scalar("outlineWidth")?,
        hair_highlight: scalar("hairHighlight")?,
    };
    one_of(
        settings.role.as_deref(),
        &["body", "skin", "hair", "face"],
        "celRole",
    )?;
    range(settings.outline_width, 0.0, 12.0, "outlineWidth")?;
    range(settings.hair_highlight, 0.0, 2.0, "hairHighlight")?;
    if let Some(c) = &settings.shadow_color {
        color(c)?;
    }
    if settings.role.as_deref() == Some("face") && settings.control_map.is_none() {
        return Err(GraphParseError {
            line,
            message: "celRole=face requires celControlMap (ImageAsset id)".into(),
        });
    }
    Ok(settings)
}
// Reuse the DSL attribute lexer, then deserialize into strict typed children.
fn attributes<T: serde::de::DeserializeOwned>(
    tag: &str,
    line: usize,
) -> Result<T, GraphParseError> {
    let mut values = serde_json::Map::new();
    for key in crate::dsl::tag_attribute_names(tag) {
        if values.contains_key(&key) {
            return Err(error(format!("Duplicate style attribute {key}")));
        }
        let raw = attr_value(tag, &key).unwrap_or_default();
        let raw = strip_wrappers(&raw);
        let value = raw
            .parse::<serde_json::Number>()
            .map(serde_json::Value::Number)
            .unwrap_or_else(|_| match raw {
                "true" if key == "enabled" => serde_json::Value::Bool(true),
                "false" if key == "enabled" => serde_json::Value::Bool(false),
                _ => serde_json::Value::String(raw.to_string()),
            });
        values.insert(key, value);
    }
    serde_json::from_value(serde_json::Value::Object(values)).map_err(|e| GraphParseError {
        line,
        message: format!("Render style: {e}"),
    })
}

pub(crate) fn parse_resource(
    lines: &[&str],
    start: usize,
) -> Result<(serde_json::Value, usize), GraphParseError> {
    let name = "RenderStyle";
    let (open, end) = crate::dsl::collect_tag_block(lines, start, '>', false)?;
    let mut root: serde_json::Value = attributes(&open, start + 1)?;
    let id = root.get("id").and_then(|v| v.as_str()).unwrap_or("");
    if id.is_empty() || id.starts_with("__ml_style_") {
        return Err(error(
            "Style id must be nonempty and not use reserved __ml_style_ prefix",
        ));
    }
    let mut i = end + 1;
    while i < lines.len() {
        let text = lines[i].trim();
        if text == format!("</{name}>") {
            return Ok((root, i));
        }
        if text.is_empty() || text.starts_with("//") || text.starts_with("<!--") {
            i += 1;
            continue;
        }
        let (tag, last) = crate::dsl::collect_tag_block(lines, i, '>', false)?;
        if !tag.trim_end().ends_with("/>") {
            return Err(error("Style children must be self-closing"));
        }
        let child = tag
            .trim_start_matches('<')
            .split_whitespace()
            .next()
            .unwrap_or("");
        let field = match child {
            "SurfaceStyle" => "surface",
            "OutlineStyle" => "outline",
            "LightingStyle" => "lighting",
            "PostStyle" => "post",
            "ColorStyle" => "color",
            "ToneStyle" => "tone",
            "DepthOfFieldStyle" => "depthOfField",
            _ => return Err(error(format!("Unsupported {name} child {child}"))),
        };
        if root.get(field).is_some() {
            return Err(error(format!("Duplicate {child}")));
        }
        root[field] = attributes(&tag, i + 1)?;
        i = last + 1;
    }
    Err(error(format!("Missing </{name}>")))
}
