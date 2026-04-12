pub const DEFAULT_LANDSCAPE_EXPORT_RESOLUTION: &str = "1920x1080";
pub const DEFAULT_PORTRAIT_EXPORT_RESOLUTION: &str = "1080x1920";
pub const DEFAULT_SQUARE_EXPORT_RESOLUTION: &str = "1080x1080";

pub const UI_EXPORT_RESOLUTION_CHOICES: [(&str, &str); 25] = [
    ("canvas", "Match Canvas"),
    ("7680x4320", "8K UHD 7680x4320"),
    ("4320x7680", "Vertical 4320x7680 (8K)"),
    ("5120x2880", "5K UHD 5120x2880"),
    ("2880x5120", "Vertical 2880x5120 (5K)"),
    ("4096x2160", "DCI 4K 4096x2160"),
    ("3840x2160", "4K UHD 3840x2160"),
    ("2560x1440", "QHD 2560x1440"),
    ("1920x1080", "Full HD 1920x1080"),
    ("1080x1920", "Vertical 1080x1920"),
    ("1600x1200", "UXGA 1600x1200 (4:3)"),
    ("1200x1600", "Vertical 1200x1600 (4:3)"),
    ("1440x1080", "HDV 1440x1080 (4:3)"),
    ("1080x1440", "Vertical 1080x1440 (4:3)"),
    ("1024x768", "XGA 1024x768 (4:3)"),
    ("768x1024", "Vertical 768x1024 (4:3)"),
    ("854x480", "SD 480p 854x480"),
    ("480x854", "Vertical 480x854"),
    ("1280x720", "HD 1280x720"),
    ("720x1280", "Vertical 720x1280"),
    ("640x360", "SD 360p 640x360"),
    ("360x640", "Vertical 360x640"),
    ("256x144", "Low 144p 256x144"),
    ("144x256", "Vertical 144x256"),
    ("1080x1080", "Square 1080x1080"),
];

const PORTRAIT_KEYWORDS: [&str; 3] = ["portrait", "vertical", "tall"];

const LANDSCAPE_KEYWORDS: [&str; 3] = ["landscape", "horizontal", "wide"];

const SQUARE_KEYWORDS: [&str; 1] = ["square"];

// Add/update aliases here when introducing new preset labels.
const NAMED_EXPORT_RESOLUTION_ALIASES: [(&str, &str); 22] = [
    ("vertical 8k", "4320x7680"),
    ("portrait 8k", "4320x7680"),
    ("vertical 5k", "2880x5120"),
    ("portrait 5k", "2880x5120"),
    ("dci 4k", "4096x2160"),
    ("4k dci", "4096x2160"),
    ("8k uhd", "7680x4320"),
    ("uhd 8k", "7680x4320"),
    ("5k uhd", "5120x2880"),
    ("uhd 5k", "5120x2880"),
    ("4k uhd", "3840x2160"),
    ("uhd 4k", "3840x2160"),
    ("full hd", "1920x1080"),
    ("fhd", "1920x1080"),
    ("qhd", "2560x1440"),
    ("uxga", "1600x1200"),
    ("hdv", "1440x1080"),
    ("xga", "1024x768"),
    ("720p", "1280x720"),
    ("480p", "854x480"),
    ("360p", "640x360"),
    ("144p", "256x144"),
];

pub const fn export_resolution_choices_for_ui() -> &'static [(&'static str, &'static str)] {
    &UI_EXPORT_RESOLUTION_CHOICES
}

fn normalize_for_match(raw: &str) -> String {
    raw.chars()
        .flat_map(|ch| ch.to_lowercase())
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

fn contains_keyword(raw: &str, lower_ascii: &str, keyword: &str) -> bool {
    if keyword.is_ascii() {
        lower_ascii.contains(keyword)
    } else {
        raw.contains(keyword)
    }
}

fn parse_resolution_dims_from_text(raw: &str) -> Option<(u32, u32)> {
    if let Some((w, h)) = parse_resolution_dims(raw) {
        return Some((w, h));
    }

    let mut token = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_digit() || matches!(ch, 'x' | 'X' | ' ') {
            token.push(ch);
        } else {
            if let Some((w, h)) = parse_resolution_dims(&token) {
                return Some((w, h));
            }
            token.clear();
        }
    }

    parse_resolution_dims(&token)
}

pub fn parse_resolution_dims(raw: &str) -> Option<(u32, u32)> {
    let token = raw.trim();
    let (w, h) = token.split_once(['x', 'X'])?;
    let width = w.trim().parse::<u32>().ok()?.max(2);
    let height = h.trim().parse::<u32>().ok()?.max(2);
    Some((width, height))
}

pub fn format_resolution_label(width: u32, height: u32) -> String {
    format!("{width}x{height}")
}

fn find_resolution_from_ui_presets(raw: &str) -> Option<&'static str> {
    let normalized = normalize_for_match(raw);
    if normalized.is_empty() {
        return None;
    }

    for (id, label) in export_resolution_choices_for_ui().iter().copied() {
        if id == "canvas" {
            continue;
        }
        let id_key = normalize_for_match(id);
        if normalized == id_key || normalized.contains(&id_key) {
            return Some(id);
        }

        let label_key = normalize_for_match(label);
        if normalized == label_key || normalized.contains(&label_key) {
            return Some(id);
        }
    }

    None
}

fn find_resolution_from_aliases(raw: &str) -> Option<&'static str> {
    let normalized = normalize_for_match(raw);
    if normalized.is_empty() {
        return None;
    }

    if normalized.contains("2k") {
        if normalized.contains("vertical") || normalized.contains("portrait") {
            return Some("1440x2560");
        }
        return Some("2560x1440");
    }
    if normalized.contains("dci4k") {
        return Some("4096x2160");
    }
    if normalized.contains("4k") {
        if normalized.contains("vertical") || normalized.contains("portrait") {
            return Some("2160x3840");
        }
        return Some("3840x2160");
    }
    if normalized.contains("5k") {
        if normalized.contains("vertical") || normalized.contains("portrait") {
            return Some("2880x5120");
        }
        return Some("5120x2880");
    }
    if normalized.contains("8k") {
        if normalized.contains("vertical") || normalized.contains("portrait") {
            return Some("4320x7680");
        }
        return Some("7680x4320");
    }

    for (alias, id) in NAMED_EXPORT_RESOLUTION_ALIASES {
        let alias_key = normalize_for_match(alias);
        if normalized == alias_key || normalized.contains(&alias_key) {
            return Some(id);
        }
    }

    None
}

fn find_orientation_default_resolution(raw: &str) -> Option<&'static str> {
    let token = raw.trim();
    if token.is_empty() {
        return None;
    }

    let lower = token.to_ascii_lowercase();
    let portrait = PORTRAIT_KEYWORDS
        .iter()
        .any(|key| contains_keyword(token, &lower, key));
    let landscape = LANDSCAPE_KEYWORDS
        .iter()
        .any(|key| contains_keyword(token, &lower, key));
    let square = SQUARE_KEYWORDS
        .iter()
        .any(|key| contains_keyword(token, &lower, key));

    match (portrait, landscape, square) {
        (true, false, _) => Some(DEFAULT_PORTRAIT_EXPORT_RESOLUTION),
        (false, true, _) => Some(DEFAULT_LANDSCAPE_EXPORT_RESOLUTION),
        (false, false, true) => Some(DEFAULT_SQUARE_EXPORT_RESOLUTION),
        _ => None,
    }
}

pub fn normalize_export_resolution_hint(raw: &str) -> Option<String> {
    let token = raw.trim();
    if token.is_empty() {
        return None;
    }

    if let Some((w, h)) = parse_resolution_dims_from_text(token) {
        return Some(format_resolution_label(w, h));
    }
    if let Some(id) = find_resolution_from_ui_presets(token) {
        return Some(id.to_string());
    }
    if let Some(id) = find_resolution_from_aliases(token) {
        return Some(id.to_string());
    }
    find_orientation_default_resolution(token).map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::{export_resolution_choices_for_ui, normalize_export_resolution_hint};

    #[test]
    fn normalize_export_resolution_hint_supports_display_setting_labels() {
        assert_eq!(
            normalize_export_resolution_hint("Vertical (720x1280) 9:16"),
            Some("720x1280".to_string())
        );
        assert_eq!(
            normalize_export_resolution_hint("QHD"),
            Some("2560x1440".to_string())
        );
        assert_eq!(
            normalize_export_resolution_hint("DCI 4K"),
            Some("4096x2160".to_string())
        );
        assert_eq!(
            normalize_export_resolution_hint("2K portrait"),
            Some("1440x2560".to_string())
        );
        assert_eq!(
            normalize_export_resolution_hint("2K landscape"),
            Some("2560x1440".to_string())
        );
        assert_eq!(
            normalize_export_resolution_hint("4K portrait"),
            Some("2160x3840".to_string())
        );
        assert_eq!(
            normalize_export_resolution_hint("4K landscape"),
            Some("3840x2160".to_string())
        );
    }

    #[test]
    fn normalize_export_resolution_hint_covers_all_ui_presets() {
        for (id, label) in export_resolution_choices_for_ui() {
            if *id == "canvas" {
                continue;
            }
            assert_eq!(
                normalize_export_resolution_hint(id),
                Some((*id).to_string()),
                "failed to parse preset id: {id}"
            );
            assert_eq!(
                normalize_export_resolution_hint(label),
                Some((*id).to_string()),
                "failed to parse preset label: {label}"
            );
        }
    }
}
