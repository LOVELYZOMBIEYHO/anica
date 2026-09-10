// =========================================
// =========================================
// crates/motionloom/src/audio/dsl.rs

// Audio has independent sample-accurate timing while retaining the existing Key attributes.
use super::*;
use crate::audio::{AudioClipNode, AudioKeyNode, AudioTargetNode};

fn error(line: usize, message: impl Into<String>) -> GraphParseError {
    GraphParseError {
        line,
        message: message.into(),
    }
}
fn number(raw: &str, line: usize) -> Result<f64, GraphParseError> {
    strip_wrappers(raw)
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| error(line, format!("Audio requires a finite number, got {raw}")))
}
fn time(raw: &str, line: usize) -> Result<f64, GraphParseError> {
    let raw = strip_wrappers(raw);
    let result = if let Some(ms) = raw.strip_suffix("ms") {
        number(ms, line)? / 1000.0
    } else {
        number(raw.strip_suffix('s').unwrap_or(raw), line)?
    };
    if !(0.0..=86_400.0).contains(&result) {
        return Err(error(
            line,
            "Audio time must be between 0 and 86400 seconds",
        ));
    }
    Ok(result)
}
fn bounded(raw: &str, min: f64, max: f64, line: usize) -> Result<f64, GraphParseError> {
    let value = number(raw, line)?;
    if value < min || value > max {
        return Err(error(line, format!("Audio value must be {min}..{max}")));
    }
    Ok(value)
}
pub(super) fn clip(tag: &str, line: usize) -> Result<AudioClipNode, GraphParseError> {
    let get = |name| required_attr_value(tag, name, line);
    let optional = |name, fallback: &str| attr_value(tag, name).unwrap_or_else(|| fallback.into());
    let from = time(&get("from")?, line)?;
    let duration = time(&get("duration")?, line)?;
    let source_in = time(&optional("sourceIn", "0s"), line)?;
    let source_out = attr_value(tag, "sourceOut")
        .map(|raw| time(&raw, line))
        .transpose()?;
    if duration <= 0.0 || from + duration > 86_400.0 {
        return Err(error(
            line,
            "AudioClip duration must be positive; end must be within 86400s",
        ));
    }
    if source_out.is_some_and(|end| end <= source_in) {
        return Err(error(line, "AudioClip sourceOut must be after sourceIn"));
    }
    let looped = match strip_wrappers(&optional("loop", "false")) {
        "true" => true,
        "false" => false,
        _ => return Err(error(line, "AudioClip loop must be true or false")),
    };
    let id = strip_wrappers(&get("id")?).to_string();
    let asset = strip_wrappers(&get("asset")?).to_string();
    if id.trim().is_empty() || asset.trim().is_empty() {
        return Err(error(line, "AudioClip id and asset cannot be empty"));
    }
    Ok(AudioClipNode {
        id,
        asset,
        from,
        duration,
        source_in,
        source_out,
        looped,
        gain_db: bounded(&optional("gainDb", "0"), -120.0, 24.0, line)?,
        pan: bounded(&optional("pan", "0"), -1.0, 1.0, line)?,
        playback_rate: bounded(&optional("playbackRate", "1"), 0.125, 8.0, line)?,
    })
}
pub(super) fn target(
    lines: &[&str],
    start: usize,
    fps: f32,
) -> Result<(AudioTargetNode, usize), GraphParseError> {
    let (tag, open_end) = collect_tag_block(lines, start, '>', false)?;
    let close = find_matching_close_tag(lines, open_end + 1, "AudioTarget")?;
    let node = strip_wrappers(&required_attr_value(&tag, "node", start + 1)?).to_string();
    let property = strip_wrappers(&required_attr_value(&tag, "property", start + 1)?).to_string();
    let (min, max) = match property.as_str() {
        "gainDb" => (-120.0, 24.0),
        "pan" => (-1.0, 1.0),
        "playbackRate" => (0.125, 8.0),
        _ => {
            return Err(error(
                start + 1,
                "AudioTarget property must be gainDb, pan or playbackRate",
            ));
        }
    };
    let mut keys = Vec::new();
    let mut i = open_end + 1;
    while i < close {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            i += 1;
            continue;
        }
        if !starts_open_tag(line, "Key") {
            return Err(error(i + 1, "AudioTarget only accepts Key children"));
        }
        let (key, end) = collect_self_closing_block(lines, i)?;
        let seconds = match (attr_value(&key, "time"), attr_value(&key, "frame")) {
            (Some(raw), None) => time(&raw, i + 1)?,
            (None, Some(raw)) => {
                let frame = strip_wrappers(&raw)
                    .parse::<u32>()
                    .map_err(|_| error(i + 1, "Key frame must be a non-negative integer"))?;
                frame as f64 / fps.max(1.0) as f64
            }
            _ => return Err(error(i + 1, "Key requires exactly one of time or frame")),
        };
        let value = bounded(&required_attr_value(&key, "value", i + 1)?, min, max, i + 1)?;
        let ease = strip_wrappers(&attr_value(&key, "ease").unwrap_or_else(|| "linear".into()))
            .to_string();
        if !matches!(
            ease.as_str(),
            "linear" | "step" | "ease_in" | "ease_out" | "ease_in_out"
        ) {
            return Err(error(
                i + 1,
                "Audio Key ease must be linear, step, ease_in, ease_out or ease_in_out",
            ));
        }
        keys.push(AudioKeyNode {
            seconds,
            value,
            ease,
        });
        i = end + 1;
    }
    keys.sort_by(|a, b| a.seconds.total_cmp(&b.seconds));
    if keys.is_empty() || keys.windows(2).any(|p| p[0].seconds == p[1].seconds) {
        return Err(error(
            start + 1,
            "AudioTarget requires non-empty keys with unique times",
        ));
    }
    Ok((
        AudioTargetNode {
            node,
            property,
            keys,
        },
        close,
    ))
}
