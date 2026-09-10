// =========================================
// =========================================
// crates/motionloom/src/head_fitting/source.rs

use super::*;
use std::ops::Range;

// Existing action editing helpers are Action-specific and line-based. This scanner
// preserves byte spans and skips comments, quoted text and expression braces.
struct Tag {
    range: Range<usize>,
    name: String,
    attrs: Vec<(String, Range<usize>)>,
}
fn tags(s: &str) -> Result<Vec<Tag>, HeadFitError> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = vec![];
    while i < b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        if s[i..].starts_with("<!--") {
            i += s[i..]
                .find("-->")
                .ok_or_else(|| HeadFitError::Source("unclosed comment".into()))?
                + 3;
            continue;
        }
        let start = i;
        i += 1;
        let n = i;
        while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'>' {
            i += 1
        }
        let name = s[n..i].trim_end_matches('/').to_string();
        let mut attrs = vec![];
        while i < b.len() && b[i] != b'>' {
            if b[i].is_ascii_whitespace() || b[i] == b'/' {
                i += 1;
                continue;
            }
            let n = i;
            while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'=' {
                i += 1
            }
            let key = s[n..i].to_string();
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1
            }
            if i == b.len() || b[i] != b'=' {
                return Err(HeadFitError::Source("unsupported attribute syntax".into()));
            }
            i += 1;
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1
            }
            let v = i;
            if i >= b.len() {
                return Err(HeadFitError::Source("missing value".into()));
            }
            if b[i] == b'\'' || b[i] == b'"' {
                let q = b[i];
                i += 1;
                while i < b.len() && b[i] != q {
                    if b[i] == b'\\' {
                        i += 1
                    }
                    i += 1
                }
                i += 1;
            } else if b[i] == b'{' {
                let mut depth = 1;
                i += 1;
                let mut quote = 0;
                while i < b.len() && depth > 0 {
                    let c = b[i];
                    if quote != 0 {
                        if c == b'\\' {
                            i += 1
                        } else if c == quote {
                            quote = 0
                        }
                    } else if c == b'\'' || c == b'"' {
                        quote = c
                    } else if c == b'{' {
                        depth += 1
                    } else if c == b'}' {
                        depth -= 1
                    }
                    i += 1;
                }
            } else {
                while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'>' {
                    i += 1
                }
            }
            if i > b.len() {
                return Err(HeadFitError::Source("unterminated value".into()));
            }
            attrs.push((key, v..i));
        }
        if i == b.len() {
            return Err(HeadFitError::Source("unterminated tag".into()));
        }
        i += 1;
        out.push(Tag {
            range: start..i,
            name,
            attrs,
        });
    }
    Ok(out)
}
pub(super) fn target_range(s: &str, id: &str) -> Result<Range<usize>, HeadFitError> {
    let tags = tags(s)?;
    let mut ranges = vec![];
    for (i, t) in tags.iter().enumerate() {
        if t.name == "HeadAsset"
            && t.attrs
                .iter()
                .any(|(n, r)| n == "id" && s[r.clone()].trim_matches(['\'', '"']) == id)
        {
            let end = tags[i + 1..]
                .iter()
                .find(|t| t.name == "/HeadAsset")
                .ok_or_else(|| HeadFitError::Source("missing HeadAsset close".into()))?;
            ranges.push(t.range.start..end.range.end);
        }
    }
    if ranges.len() != 1 {
        return Err(HeadFitError::Source(
            "target must have one unambiguous source span".into(),
        ));
    }
    Ok(ranges.remove(0))
}
pub(super) fn changes(
    a: &PrimitiveAssetNode,
    b: &PrimitiveAssetNode,
    id: &str,
) -> Vec<HeadFitChange> {
    let (
        PrimitiveGeometry::HeadSurface {
            head_shape: sa,
            face_layout: fa,
            ..
        },
        PrimitiveGeometry::HeadSurface {
            head_shape: sb,
            face_layout: fb,
            ..
        },
    ) = (&a.geometry, &b.geometry)
    else {
        return vec![];
    };
    let mut out = vec![];
    let mut compare = |tag: &str, node: &str, a: serde_json::Value, b: serde_json::Value| {
        if let (Some(a), Some(b)) = (a.as_object(), b.as_object()) {
            for (key, value) in b {
                if a.get(key) != Some(value) && value.is_number()
                    || key == "position" && a.get(key) != Some(value)
                    || key == "size" && a.get(key) != Some(value)
                {
                    out.push(HeadFitChange {
                        node_id: node.into(),
                        child_tag: tag.into(),
                        attribute: key.clone(),
                        before: a[key].clone(),
                        after: value.clone(),
                    });
                }
            }
        }
    };
    compare(
        "HeadShape",
        id,
        serde_json::to_value(sa).unwrap(),
        serde_json::to_value(sb).unwrap(),
    );
    if let (Some(a), Some(b)) = (fa, fb) {
        for (a, b) in a.eyes.iter().zip(&b.eyes) {
            compare(
                "Eye",
                &a.id,
                serde_json::to_value(a).unwrap(),
                serde_json::to_value(b).unwrap(),
            );
            if let (Some(a), Some(b)) = (&a.iris, &b.iris) {
                compare(
                    "Iris",
                    &a.id,
                    serde_json::to_value(a).unwrap(),
                    serde_json::to_value(b).unwrap(),
                );
            }
        }
        for (a, b) in a.noses.iter().zip(&b.noses) {
            compare(
                "Nose",
                &a.id,
                serde_json::to_value(a).unwrap(),
                serde_json::to_value(b).unwrap(),
            );
        }
        for (a, b) in a.mouths.iter().zip(&b.mouths) {
            compare(
                "Mouth",
                &a.id,
                serde_json::to_value(a).unwrap(),
                serde_json::to_value(b).unwrap(),
            );
        }
    }
    out
}
pub(super) fn patch(s: &str, id: &str, changes: &[HeadFitChange]) -> Result<String, HeadFitError> {
    let range = target_range(s, id)?;
    let block = &s[range.clone()];
    let mut edits = vec![];
    let scanned = tags(block)?;
    let mut seen = std::collections::BTreeSet::new();
    for c in changes {
        if !seen.insert((&c.node_id, &c.child_tag, &c.attribute)) {
            return Err(HeadFitError::Source("duplicate or foreign edit".into()));
        }
        let allowed = if c.child_tag == "HeadShape" {
            [
                "size",
                "forehead",
                "cheekWidth",
                "jawWidth",
                "chinLength",
                "chinRoundness",
            ]
            .contains(&c.attribute.as_str())
        } else if c.child_tag == "Eye" {
            [
                "position",
                "width",
                "opening",
                "tilt",
                "socketWidth",
                "socketHeight",
                "socketDepth",
            ]
            .contains(&c.attribute.as_str())
        } else if c.child_tag == "Iris" {
            ["radius", "pupilRadius"].contains(&c.attribute.as_str())
        } else if c.child_tag == "Nose" {
            ["position", "length", "projection"].contains(&c.attribute.as_str())
        } else if c.child_tag == "Mouth" {
            ["position", "width", "opening", "upperLip", "lowerLip"].contains(&c.attribute.as_str())
        } else {
            false
        };
        if !allowed {
            return Err(HeadFitError::Source(
                "edit outside fitting parameters".into(),
            ));
        }
        let matches: Vec<_> = scanned
            .iter()
            .filter(|t| {
                t.name == c.child_tag
                    && (c.child_tag == "HeadShape" && c.node_id == id
                        || t.attrs.iter().any(|(name, range)| {
                            name == "id"
                                && block[range.clone()].trim_matches(['\'', '"']) == c.node_id
                        }))
            })
            .collect();
        if matches.len() != 1 {
            return Err(HeadFitError::Source(
                "editable child must exist exactly once".into(),
            ));
        }
        let t = matches[0];
        let attrs: Vec<_> = t.attrs.iter().filter(|(n, _)| n == &c.attribute).collect();
        let raw = serde_json::to_string(&c.after)?;
        if !(c.after.is_number()
            || matches!(c.attribute.as_str(), "size" | "position")
                && c.after
                    .as_array()
                    .is_some_and(|a| a.len() == 3 && a.iter().all(|v| v.is_number())))
        {
            return Err(HeadFitError::Source("expected numeric parameter".into()));
        }
        let value = if c.after.is_array() {
            format!("{{{raw}}}")
        } else {
            format!("\"{raw}\"")
        };
        if attrs.len() > 1 {
            return Err(HeadFitError::Source("duplicate attribute".into()));
        }
        if let Some((_, r)) = attrs.first() {
            edits.push((range.start + r.start..range.start + r.end, value));
        } else {
            let end = t.range.end - 1;
            let insertion = if block.as_bytes()[end - 1] == b'/' {
                end - 1
            } else {
                end
            };
            edits.push((
                range.start + insertion..range.start + insertion,
                format!(" {}={} ", c.attribute, value),
            ));
        }
    }
    edits.sort_by_key(|(r, _)| std::cmp::Reverse(r.start));
    let mut output = s.to_string();
    for (r, v) in edits {
        output.replace_range(r, &v)
    }
    Ok(output)
}
