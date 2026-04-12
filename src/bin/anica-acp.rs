use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::{
    Agent, AgentSideConnection, AuthenticateRequest, AuthenticateResponse, CancelNotification,
    Client, ContentBlock, ContentChunk, Error, ExtRequest, Implementation, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse,
    SessionId, SessionNotification, SessionUpdate, StopReason,
};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value, json};
use tokio::process::Command as TokioCommand;
use tokio::task::LocalSet;
use tokio::time::sleep;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

#[path = "../export_resolution.rs"]
mod export_resolution;
#[path = "../runtime_paths.rs"]
mod runtime_paths;

#[derive(Debug)]
struct SessionState {
    cwd: PathBuf,
    cancelled: bool,
    running_pid: Option<u32>,
    last_audio_silence_args: Option<Value>,
    last_validated_operations: Option<Value>,
    last_export_resolution_preference: Option<String>,
}

#[derive(Debug)]
struct AgentState {
    conn: RefCell<Option<Rc<AgentSideConnection>>>,
    sessions: RefCell<HashMap<SessionId, SessionState>>,
    next_id: Cell<u64>,
    resources: ResourceBundle,
}

#[derive(Debug, Clone)]
struct AnicaAcpAgent {
    inner: Rc<AgentState>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ToolPlannerOutput {
    #[serde(default)]
    intent: Option<String>,
    #[serde(default)]
    use_tool: Option<String>,
    #[serde(default)]
    arguments: Option<Value>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    next_step: Option<String>,
    #[serde(default)]
    tool_call: Option<ToolPlannerCall>,
    #[serde(default, rename = "final")]
    r#final: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolPlannerCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectCutKind {
    Silence,
    SubtitleGap,
}

#[derive(Debug, Clone)]
struct HybridSecondCheckRange {
    start_ms: u64,
    end_ms: u64,
    category: String,
    confidence: f32,
    reason: String,
    source_clip_ids: Vec<u64>,
}

#[derive(Debug, Clone)]
struct HybridSecondCheckParse {
    ranges: Vec<HybridSecondCheckRange>,
    categories_reported: Vec<String>,
    missing_categories: Vec<String>,
    hard_rule_ok: bool,
}

#[derive(Debug, Clone)]
struct SubtitleTranslationRow {
    clip_id: u64,
    track_index: usize,
    start_ms: u64,
    duration_ms: u64,
    text: String,
}

const HYBRID_SECOND_CHECK_REQUIRED_CATEGORIES: [&str; 4] = [
    "exact_repeat",
    "same_topic_consecutive_restart",
    "near_synonym_semantic_repeat",
    "prefix_or_continuation_restart",
];
const SUBTITLE_TRANSLATION_LLM_CHUNK_SIZE: usize = 80;
const SUBTITLE_TRANSLATION_APPLY_CHUNK_SIZE: usize = 220;

#[derive(Debug, Clone)]
struct ResourceBundle {
    catalog: HashMap<String, String>,
    tool_router_prompt: String,
    llm_similarity_intent_phrases: Vec<String>,
}

impl ResourceBundle {
    fn load() -> Self {
        let locale = std::env::var("ANICA_LOCALE")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "en-US".to_string());

        let mut catalog = parse_i18n_map(include_str!("../../assets/i18n/en-US.json"));

        if let Some(runtime_en) = read_asset_text("i18n/en-US.json") {
            for (k, v) in parse_i18n_map(&runtime_en) {
                catalog.insert(k, v);
            }
        }

        if !locale.eq_ignore_ascii_case("en-US") {
            let rel = format!("i18n/{locale}.json");
            let locale_raw = read_asset_text(&rel).or_else(|| embedded_locale_text(&locale));
            if let Some(raw) = locale_raw {
                for (k, v) in parse_i18n_map(&raw) {
                    catalog.insert(k, v);
                }
            }
        }

        let tool_router_prompt = read_asset_text("prompts/tool_router.md")
            .unwrap_or_else(|| include_str!("../../assets/prompts/tool_router.md").to_string());
        let llm_similarity_intent_phrases = load_llm_similarity_intent_phrases();

        Self {
            catalog,
            tool_router_prompt,
            llm_similarity_intent_phrases,
        }
    }

    fn tr(&self, key: &str) -> String {
        self.catalog
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string())
    }

    fn tr_args(&self, key: &str, vars: &[(&str, String)]) -> String {
        let mut out = self.tr(key);
        for (name, value) in vars {
            out = out.replace(&format!("{{{name}}}"), value);
        }
        out
    }

    fn render_tool_router_prompt(&self, user_prompt: &str, tool_results: &[String]) -> String {
        let tool_results_section = if tool_results.is_empty() {
            String::new()
        } else {
            let mut block = String::from("<tool_results>\n");
            for (idx, result) in tool_results.iter().enumerate() {
                block.push_str(&format!("<result index=\"{}\">\n", idx + 1));
                block.push_str(result);
                block.push_str("\n</result>\n");
            }
            block.push_str("</tool_results>\n");
            block
        };

        self.tool_router_prompt
            .replace("{{USER_PROMPT}}", user_prompt)
            .replace("{{TOOL_RESULTS_SECTION}}", &tool_results_section)
    }
}

fn parse_i18n_map(raw: &str) -> HashMap<String, String> {
    serde_json::from_str::<HashMap<String, String>>(raw).unwrap_or_default()
}

fn parse_llm_similarity_intent_phrases_md(raw: &str) -> Vec<String> {
    let mut phrases = Vec::new();
    let mut seen = HashSet::new();
    let mut in_block = false;

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed == "<!-- ACP_INTENT_PHRASES_START -->" {
            in_block = true;
            continue;
        }
        if trimmed == "<!-- ACP_INTENT_PHRASES_END -->" {
            break;
        }
        if !in_block {
            continue;
        }

        let Some(rest) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        else {
            continue;
        };

        let mut phrase = rest.trim();
        if let Some(inner) = phrase.strip_prefix('`').and_then(|v| v.strip_suffix('`')) {
            phrase = inner.trim();
        }
        if phrase.is_empty() {
            continue;
        }

        let normalized = phrase.to_lowercase();
        if seen.insert(normalized.clone()) {
            phrases.push(normalized);
        }
    }

    phrases
}

fn candidate_acp_doc_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();

    if let Some(dir) = std::env::var_os("ANICA_ACP_DOCS_DIR")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
    {
        roots.push(dir);
    }

    for root in runtime_paths::candidate_docs_roots() {
        let acp_root = root.join("acp");
        if !roots.iter().any(|existing| existing == &acp_root) {
            roots.push(acp_root);
        }
    }

    roots
}

fn read_acp_doc_text(relative_path: &str) -> Option<String> {
    for root in candidate_acp_doc_roots() {
        let path = root.join(relative_path);
        if let Ok(raw) = fs::read_to_string(&path) {
            return Some(raw);
        }
    }
    None
}

fn default_llm_similarity_intent_phrases() -> Vec<String> {
    vec![
        "cut similar sentences".to_string(),
        "delete similar sentences".to_string(),
        "remove similar sentences".to_string(),
        "delete duplicate subtitles".to_string(),
        "remove duplicate subtitles".to_string(),
        "llm cut similar sentences".to_string(),
        "use llm cut similar sentences".to_string(),
    ]
}

fn load_llm_similarity_intent_phrases() -> Vec<String> {
    let raw = read_acp_doc_text("intent-phrases/INTENT-0001.md").unwrap_or_else(|| {
        include_str!("../../docs/acp/intent-phrases/INTENT-0001.md").to_string()
    });
    let parsed = parse_llm_similarity_intent_phrases_md(&raw);
    if parsed.is_empty() {
        default_llm_similarity_intent_phrases()
    } else {
        parsed
    }
}

fn embedded_locale_text(locale: &str) -> Option<String> {
    if locale.eq_ignore_ascii_case("zh-HK") || locale.eq_ignore_ascii_case("zh-TW") {
        Some(include_str!("../../assets/i18n/zh-HK.json").to_string())
    } else if locale.eq_ignore_ascii_case("zh-CN") || locale.eq_ignore_ascii_case("zh-SG") {
        Some(include_str!("../../assets/i18n/zh-CN.json").to_string())
    } else if locale.eq_ignore_ascii_case("ja-JP") {
        Some(include_str!("../../assets/i18n/ja-JP.json").to_string())
    } else if locale.eq_ignore_ascii_case("ko-KR") {
        Some(include_str!("../../assets/i18n/ko-KR.json").to_string())
    } else if locale.eq_ignore_ascii_case("de-DE") {
        Some(include_str!("../../assets/i18n/de-DE.json").to_string())
    } else if locale.eq_ignore_ascii_case("fr-FR") {
        Some(include_str!("../../assets/i18n/fr-FR.json").to_string())
    } else if locale.eq_ignore_ascii_case("it-IT") {
        Some(include_str!("../../assets/i18n/it-IT.json").to_string())
    } else {
        None
    }
}

fn read_asset_text(relative_path: &str) -> Option<String> {
    for root in candidate_asset_roots() {
        let path = root.join(relative_path);
        if let Ok(raw) = fs::read_to_string(&path) {
            return Some(raw);
        }
    }
    None
}

fn candidate_asset_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();

    if let Some(dir) = std::env::var_os("ANICA_ASSETS_DIR")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
    {
        roots.push(dir);
    }

    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        roots.push(exe_dir.join("assets"));
        if let Some(contents_dir) = exe_dir.parent() {
            roots.push(contents_dir.join("Resources").join("assets"));
        }
    }

    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets"));

    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join("assets"));
        roots.push(cwd.join("anica").join("assets"));
    }

    let mut unique: Vec<PathBuf> = Vec::new();
    for root in roots {
        if !unique.iter().any(|p| p == &root) {
            unique.push(root);
        }
    }
    unique
}

impl Default for AnicaAcpAgent {
    fn default() -> Self {
        Self {
            inner: Rc::new(AgentState {
                conn: RefCell::new(None),
                sessions: RefCell::new(HashMap::new()),
                next_id: Cell::new(1),
                resources: ResourceBundle::load(),
            }),
        }
    }
}

impl AnicaAcpAgent {
    fn has_any_keyword(haystack: &str, needles: &[&str]) -> bool {
        needles.iter().any(|needle| haystack.contains(needle))
    }

    fn detect_direct_cut_kind(user_prompt: &str) -> Option<DirectCutKind> {
        let normalized = user_prompt.trim().to_lowercase();
        if normalized.is_empty() {
            return None;
        }

        let has_cut_intent =
            Self::has_any_keyword(&normalized, &["cut", "trim", "delete", "remove", "ripple"]);
        if !has_cut_intent {
            return None;
        }

        if Self::has_any_keyword(
            &normalized,
            &[
                "no subtitle",
                "no-subtitle",
                "subtitle gap",
                "without subtitle",
            ],
        ) {
            return Some(DirectCutKind::SubtitleGap);
        }

        if Self::has_any_keyword(&normalized, &["silence", "silent", "pause", "low energy"]) {
            return Some(DirectCutKind::Silence);
        }

        None
    }

    fn is_low_confidence_cut_intent(user_prompt: &str) -> bool {
        let normalized = user_prompt.trim().to_lowercase();
        if normalized.is_empty() {
            return false;
        }

        let has_cut_intent =
            Self::has_any_keyword(&normalized, &["cut", "trim", "delete", "remove", "ripple"]);
        let has_low_confidence_phrase = Self::has_any_keyword(
            &normalized,
            &[
                "low confidence",
                "low-confidence",
                "no confident",
                "not confident",
                "unconfident",
                "uncertain speech",
                "confidence speech",
                "no-confidence",
            ],
        );

        has_cut_intent && has_low_confidence_phrase
    }

    fn is_llm_similarity_cut_intent(&self, user_prompt: &str) -> bool {
        let normalized = user_prompt.trim().to_lowercase();
        if normalized.is_empty() {
            return false;
        }

        if self
            .inner
            .resources
            .llm_similarity_intent_phrases
            .iter()
            .any(|phrase| !phrase.is_empty() && normalized.contains(phrase))
        {
            return true;
        }

        let has_cut_intent =
            Self::has_any_keyword(&normalized, &["cut", "trim", "delete", "remove", "ripple"]);
        let has_similarity_phrase = Self::has_any_keyword(
            &normalized,
            &[
                "similar sentence",
                "similar sentences",
                "similar subtitle",
                "similar subtitles",
                "similar line",
                "similar lines",
                "duplicate sentence",
                "duplicate subtitles",
                "semantic repeat",
                "repeat sentence",
                "repeated sentence",
                "similar text",
                "duplicate line",
                "duplicate lines",
            ],
        );
        let has_llm_hint = Self::has_any_keyword(
            &normalized,
            &[
                "llm",
                "decision_making_srt_similar_serach",
                "srt similar",
                "similarity check",
            ],
        );
        let has_subtitle_hint = Self::has_any_keyword(
            &normalized,
            &[
                "subtitle",
                "subtitles",
                "line",
                "lines",
                "sentence",
                "sentences",
            ],
        );
        (has_llm_hint || has_cut_intent) && has_similarity_phrase
            || (has_llm_hint && has_subtitle_hint)
    }

    fn parse_json_value(raw: &str) -> Option<Value> {
        serde_json::from_str::<Value>(raw).ok()
    }

    fn parse_json_value_relaxed(raw: &str) -> Option<Value> {
        if let Some(value) = Self::parse_json_value(raw) {
            return Some(value);
        }
        let start = raw.find('{')?;
        let end = raw.rfind('}')?;
        if end <= start {
            return None;
        }
        serde_json::from_str::<Value>(&raw[start..=end]).ok()
    }

    fn parse_u64_from_value(value: &Value) -> Option<u64> {
        if let Some(v) = value.as_u64() {
            return Some(v);
        }
        if let Some(v) = value.as_i64()
            && v >= 0
        {
            return Some(v as u64);
        }
        if let Some(v) = value.as_f64()
            && v.is_finite()
            && v >= 0.0
        {
            return Some(v.round() as u64);
        }
        value.as_str().and_then(|s| s.trim().parse::<u64>().ok())
    }

    fn parse_f32_from_value(value: &Value) -> Option<f32> {
        if let Some(v) = value.as_f64() {
            return Some(v as f32);
        }
        if let Some(v) = value.as_i64() {
            return Some(v as f32);
        }
        value.as_str().and_then(|s| s.trim().parse::<f32>().ok())
    }

    fn extract_target_language_from_tool_args(tool_args: &Value) -> Option<String> {
        let obj = tool_args.as_object()?;
        for key in [
            "target_language",
            "language",
            "to_language",
            "translate_to",
            "target_lang",
        ] {
            let value = obj.get(key).and_then(|v| v.as_str()).map(str::trim);
            if let Some(value) = value
                && !value.is_empty()
            {
                return Some(value.to_string());
            }
        }
        None
    }

    fn infer_target_language_from_prompt(user_prompt: &str) -> Option<String> {
        let lower = user_prompt.to_lowercase();
        let hints = [
            ("english", "English"),
            ("japanese", "Japanese"),
            ("korean", "Korean"),
            ("french", "French"),
            ("german", "German"),
            ("spanish", "Spanish"),
            ("italian", "Italian"),
            ("chinese", "Chinese"),
        ];
        for (needle, language) in hints {
            if lower.contains(needle) {
                return Some(language.to_string());
            }
        }
        None
    }

    fn extract_requested_track_indices_from_tool_args(tool_args: &Value) -> Option<Vec<usize>> {
        let obj = tool_args.as_object()?;
        let mut out: Vec<usize> = Vec::new();
        for key in ["track_indices", "subtitle_track_indices", "track_numbers"] {
            if let Some(items) = obj.get(key).and_then(|v| v.as_array()) {
                for item in items {
                    if let Some(v) = Self::parse_u64_from_value(item) {
                        out.push(v as usize);
                    }
                }
            }
        }
        for key in ["track_index", "subtitle_track_index", "track_number"] {
            if let Some(v) = obj.get(key).and_then(Self::parse_u64_from_value) {
                out.push(v as usize);
            }
        }
        if out.is_empty() {
            None
        } else {
            out.sort_unstable();
            out.dedup();
            Some(out)
        }
    }

    fn infer_requested_track_indices_from_prompt(user_prompt: &str) -> Option<Vec<usize>> {
        let lower = user_prompt.to_lowercase();
        let mut out: Vec<usize> = Vec::new();
        let mut cursor = 0usize;
        while let Some(pos) = lower[cursor..].find("track") {
            let mut idx = cursor + pos + "track".len();
            while idx < lower.len() {
                let b = lower.as_bytes()[idx];
                if b == b' ' || b == b':' || b == b'#' || b == b'_' || b == b'-' {
                    idx += 1;
                    continue;
                }
                break;
            }
            let start = idx;
            while idx < lower.len() && lower.as_bytes()[idx].is_ascii_digit() {
                idx += 1;
            }
            if idx > start
                && let Ok(num) = lower[start..idx].parse::<usize>()
            {
                if num == 0 {
                    out.push(0);
                } else {
                    // Prompt-side track numbers are usually 1-based for humans.
                    out.push(num.saturating_sub(1));
                }
            }
            cursor = cursor + pos + 1;
        }
        if out.is_empty() {
            None
        } else {
            out.sort_unstable();
            out.dedup();
            Some(out)
        }
    }

    fn collect_subtitle_rows_for_translation(
        snapshot: &Value,
        requested_track_indices: Option<Vec<usize>>,
        warnings: &mut Vec<String>,
    ) -> (Vec<SubtitleTranslationRow>, Vec<usize>) {
        let Some(tracks) = snapshot.get("subtitle_tracks").and_then(|v| v.as_array()) else {
            return (Vec::new(), Vec::new());
        };

        let mut available = HashSet::new();
        for (fallback_idx, track) in tracks.iter().enumerate() {
            let idx = track
                .get("index")
                .and_then(Self::parse_u64_from_value)
                .map(|v| v as usize)
                .unwrap_or(fallback_idx);
            available.insert(idx);
        }

        let selected_tracks = if let Some(requested) = requested_track_indices {
            let mut normalized = Vec::new();
            for requested_idx in requested {
                if available.contains(&requested_idx) {
                    normalized.push(requested_idx);
                    continue;
                }
                if requested_idx > 0 && available.contains(&requested_idx.saturating_sub(1)) {
                    normalized.push(requested_idx.saturating_sub(1));
                    warnings.push(format!(
                        "subtitle track index {requested_idx} interpreted as 1-based index {}.",
                        requested_idx.saturating_sub(1)
                    ));
                    continue;
                }
                warnings.push(format!(
                    "subtitle track index {requested_idx} is not available and was ignored."
                ));
            }
            normalized.sort_unstable();
            normalized.dedup();
            normalized
        } else {
            let mut all = available.into_iter().collect::<Vec<_>>();
            all.sort_unstable();
            all
        };

        if selected_tracks.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let selected_set = selected_tracks.iter().copied().collect::<HashSet<_>>();

        let mut rows = Vec::new();
        for (fallback_idx, track) in tracks.iter().enumerate() {
            let track_index = track
                .get("index")
                .and_then(Self::parse_u64_from_value)
                .map(|v| v as usize)
                .unwrap_or(fallback_idx);
            if !selected_set.contains(&track_index) {
                continue;
            }
            let Some(clips) = track.get("clips").and_then(|v| v.as_array()) else {
                continue;
            };
            for clip in clips {
                let Some(clip_id) = clip.get("clip_id").and_then(Self::parse_u64_from_value) else {
                    continue;
                };
                let Some(start_ms) = clip.get("start_ms").and_then(Self::parse_u64_from_value)
                else {
                    continue;
                };
                let duration_ms = clip
                    .get("duration_ms")
                    .and_then(Self::parse_u64_from_value)
                    .unwrap_or(0)
                    .max(1);
                let text = clip
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if text.is_empty() {
                    continue;
                }
                rows.push(SubtitleTranslationRow {
                    clip_id,
                    track_index,
                    start_ms,
                    duration_ms,
                    text,
                });
            }
        }
        rows.sort_by_key(|row| (row.start_ms, row.clip_id));
        (rows, selected_tracks)
    }

    fn build_subtitle_translation_prompt(
        user_prompt: &str,
        target_language: &str,
        rows: &[SubtitleTranslationRow],
        extra_instruction: Option<&str>,
    ) -> String {
        let row_json = json!(
            rows.iter()
                .map(|row| {
                    json!({
                        "clip_id": row.clip_id,
                        "track_index": row.track_index,
                        "start_ms": row.start_ms,
                        "duration_ms": row.duration_ms,
                        "text": row.text,
                    })
                })
                .collect::<Vec<_>>()
        );
        let row_json_text =
            serde_json::to_string_pretty(&row_json).unwrap_or_else(|_| row_json.to_string());
        let extra_instruction = extra_instruction
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or("");

        format!(
            "You are a subtitle translator.\n\
            Task: translate each subtitle text into {target_language}.\n\
            Keep timeline metadata unchanged.\n\
            \n\
            HARD RULES:\n\
            - Return STRICT JSON only.\n\
            - Keep exactly one translation for every input clip_id.\n\
            - Do not skip any clip_id.\n\
            - Do not output timestamps or explanations.\n\
            - Preserve line-break intent where possible.\n\
            \n\
            Output schema:\n\
            {{\n\
              \"translations\": [\n\
                {{\"clip_id\": 123, \"text\": \"translated subtitle\"}}\n\
              ]\n\
            }}\n\
            \n\
            User request context:\n\
            {user_prompt}\n\
            \n\
            Extra instruction:\n\
            {extra_instruction}\n\
            \n\
            Input subtitle rows JSON:\n\
            {row_json_text}\n"
        )
    }

    fn parse_subtitle_translation_output(raw: &str) -> HashMap<u64, String> {
        let mut out = HashMap::new();
        let Some(value) = Self::parse_json_value_relaxed(raw) else {
            return out;
        };

        let mut ingest_item = |item: &Value| {
            let Some(clip_id) = item.get("clip_id").and_then(Self::parse_u64_from_value) else {
                return;
            };
            let text = item
                .get("text")
                .or_else(|| item.get("translation"))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .unwrap_or("");
            if text.is_empty() {
                return;
            }
            out.insert(clip_id, text.to_string());
        };

        if let Some(items) = value.get("translations").and_then(|v| v.as_array()) {
            for item in items {
                ingest_item(item);
            }
            return out;
        }
        if let Some(items) = value.get("items").and_then(|v| v.as_array()) {
            for item in items {
                ingest_item(item);
            }
            return out;
        }
        if let Some(obj) = value.get("by_clip_id").and_then(|v| v.as_object()) {
            for (clip_id_raw, text_value) in obj {
                let Ok(clip_id) = clip_id_raw.parse::<u64>() else {
                    continue;
                };
                let text = text_value
                    .as_str()
                    .map(str::trim)
                    .unwrap_or_default()
                    .to_string();
                if !text.is_empty() {
                    out.insert(clip_id, text);
                }
            }
            return out;
        }
        if let Some(items) = value.as_array() {
            for item in items {
                ingest_item(item);
            }
        }
        out
    }

    fn parse_plan_tool_ok_errors_and_after_revision(
        raw_json: &str,
    ) -> (bool, Vec<String>, Option<String>) {
        let Some(value) = Self::parse_json_value(raw_json) else {
            return (
                false,
                vec!["invalid JSON response from timeline tool".to_string()],
                None,
            );
        };
        let ok = value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
        let errors = value
            .get("errors")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_str().map(ToString::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let after_revision = value
            .get("after_revision")
            .or_else(|| value.get("timeline_revision"))
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        (ok, errors, after_revision)
    }

    fn is_same_validated_operations_placeholder(value: &Value) -> bool {
        let Some(raw) = value.as_str() else {
            return false;
        };
        let normalized = raw.trim().to_ascii_lowercase();
        matches!(
            normalized.as_str(),
            "use_same_validated_operations"
                | "use_same_operations"
                | "same_validated_operations"
                | "same_operations"
        )
    }

    fn normalize_timeline_edit_tool_args(tool_name: &str, tool_args: Value) -> Value {
        if tool_name != "anica.timeline/validate_edit_plan"
            && tool_name != "anica.timeline/apply_edit_plan"
        {
            return tool_args;
        }

        let Some(mut obj) = tool_args.as_object().cloned() else {
            return tool_args;
        };

        let Some(ops_value) = obj.get("operations") else {
            return Value::Object(obj);
        };
        let Some(ops) = ops_value.as_array() else {
            return Value::Object(obj);
        };

        let mut normalized_ops: Vec<Value> = Vec::with_capacity(ops.len());
        for op in ops {
            let Some(op_obj) = op.as_object() else {
                normalized_ops.push(op.clone());
                continue;
            };

            let op_name = op_obj
                .get("op")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();

            if matches!(
                op_name.as_str(),
                "delete_clips" | "remove_clips" | "delete_clip_ids"
            ) {
                let mut ids: Vec<u64> = Vec::new();
                if let Some(items) = op_obj.get("clip_ids").and_then(|v| v.as_array()) {
                    ids.extend(items.iter().filter_map(Self::parse_u64_from_value));
                } else if let Some(items) = op_obj.get("ids").and_then(|v| v.as_array()) {
                    ids.extend(items.iter().filter_map(Self::parse_u64_from_value));
                } else if let Some(id) = op_obj.get("clip_id").and_then(Self::parse_u64_from_value)
                {
                    ids.push(id);
                }

                if ids.is_empty() {
                    normalized_ops.push(op.clone());
                    continue;
                }

                let ripple_opt = op_obj.get("ripple").and_then(|v| v.as_bool());
                for clip_id in ids {
                    let mut mapped = Map::new();
                    mapped.insert("op".to_string(), Value::String("delete_clip".to_string()));
                    mapped.insert("clip_id".to_string(), Value::Number(clip_id.into()));
                    if let Some(ripple) = ripple_opt {
                        mapped.insert("ripple".to_string(), Value::Bool(ripple));
                    }
                    normalized_ops.push(Value::Object(mapped));
                }
                continue;
            }

            if matches!(op_name.as_str(), "delete_clip" | "remove_clip") {
                let mut mapped = op_obj.clone();
                if let Some(id_val) = mapped.get("clip_id").and_then(Self::parse_u64_from_value) {
                    mapped.insert("clip_id".to_string(), Value::Number(id_val.into()));
                }
                normalized_ops.push(Value::Object(mapped));
                continue;
            }

            if matches!(op_name.as_str(), "ripple_delete_range" | "ripple_delete") {
                let mut mapped = op_obj.clone();
                if mapped.get("start_ms").is_none()
                    && let Some(start_ms) = mapped
                        .get("start")
                        .or_else(|| mapped.get("from_ms"))
                        .or_else(|| mapped.get("startTimeMs"))
                        .and_then(Self::parse_u64_from_value)
                {
                    mapped.insert("start_ms".to_string(), Value::Number(start_ms.into()));
                }
                if mapped.get("end_ms").is_none() {
                    if let Some(end_ms) = mapped
                        .get("end")
                        .or_else(|| mapped.get("to_ms"))
                        .or_else(|| mapped.get("endTimeMs"))
                        .and_then(Self::parse_u64_from_value)
                    {
                        mapped.insert("end_ms".to_string(), Value::Number(end_ms.into()));
                    } else if let (Some(start_ms), Some(duration_ms)) = (
                        mapped.get("start_ms").and_then(Self::parse_u64_from_value),
                        mapped
                            .get("duration_ms")
                            .or_else(|| mapped.get("duration"))
                            .and_then(Self::parse_u64_from_value),
                    ) {
                        let end_ms = start_ms.saturating_add(duration_ms);
                        mapped.insert("end_ms".to_string(), Value::Number(end_ms.into()));
                    }
                }
                if mapped.get("mode").and_then(|v| v.as_str()).is_none() {
                    mapped.insert("mode".to_string(), Value::String("all_tracks".to_string()));
                }
                normalized_ops.push(Value::Object(mapped));
                continue;
            }

            normalized_ops.push(op.clone());
        }

        obj.insert("operations".to_string(), Value::Array(normalized_ops));
        Value::Object(obj)
    }

    fn normalize_export_resolution_hint(raw: &str) -> Option<String> {
        export_resolution::normalize_export_resolution_hint(raw)
    }

    fn normalize_export_target_resolution_value(raw: &str) -> Option<String> {
        let token = raw.trim();
        if token.eq_ignore_ascii_case("canvas") {
            return Some("canvas".to_string());
        }
        Self::normalize_export_resolution_hint(token)
    }

    fn prompt_export_orientations(raw: &str) -> (bool, bool) {
        let lower = raw.to_ascii_lowercase();
        let has_portrait =
            lower.contains("portrait") || lower.contains("vertical") || lower.contains("tall");
        let has_landscape =
            lower.contains("landscape") || lower.contains("horizontal") || lower.contains("wide");
        (has_portrait, has_landscape)
    }

    fn normalize_export_tool_args(
        user_prompt: &str,
        tool_args: Value,
        export_call_index: usize,
    ) -> Value {
        let Some(mut obj) = tool_args.as_object().cloned() else {
            return tool_args;
        };

        // Prefer explicit tool args so batch export calls can carry different sizes safely.
        let target_resolution = obj
            .get("target_resolution")
            .and_then(|v| v.as_str())
            .and_then(Self::normalize_export_resolution_hint)
            .or_else(|| {
                obj.get("output_name")
                    .and_then(|v| v.as_str())
                    .and_then(Self::normalize_export_resolution_hint)
            })
            .or_else(|| {
                obj.get("output_path")
                    .and_then(|v| v.as_str())
                    .and_then(Self::normalize_export_resolution_hint)
            })
            .or_else(|| Self::normalize_export_resolution_hint(user_prompt))
            .or_else(|| {
                // When one prompt asks for both portrait and landscape batch outputs and
                // no per-job hint is present, alternate deterministically by call order.
                let (has_portrait, has_landscape) = Self::prompt_export_orientations(user_prompt);
                match (has_portrait, has_landscape) {
                    (true, true) => {
                        if export_call_index % 2 == 0 {
                            Some("1080x1920".to_string())
                        } else {
                            Some("1920x1080".to_string())
                        }
                    }
                    _ => None,
                }
            });

        if let Some(target_resolution) = target_resolution {
            obj.insert(
                "target_resolution".to_string(),
                Value::String(target_resolution),
            );
        } else if obj
            .get("target_resolution")
            .and_then(|v| v.as_str())
            .is_some()
        {
            // Drop invalid target resolution hints so later fallback can apply cleanly.
            obj.remove("target_resolution");
        }

        Value::Object(obj)
    }

    fn session_last_export_resolution_preference(&self, session_id: &SessionId) -> Option<String> {
        self.inner
            .sessions
            .borrow()
            .get(session_id)
            .and_then(|s| s.last_export_resolution_preference.clone())
    }

    fn set_session_last_export_resolution_preference(
        &self,
        session_id: &SessionId,
        resolution: Option<String>,
    ) {
        if let Some(s) = self.inner.sessions.borrow_mut().get_mut(session_id) {
            s.last_export_resolution_preference = resolution;
        }
    }

    fn update_export_resolution_preference_from_prompt(
        &self,
        session_id: &SessionId,
        user_prompt: &str,
    ) {
        let trimmed = user_prompt.trim();
        if trimmed.is_empty() {
            return;
        }

        if let Some(normalized) = Self::normalize_export_resolution_hint(trimmed) {
            self.set_session_last_export_resolution_preference(session_id, Some(normalized));
            return;
        }

        let lower = trimmed.to_ascii_lowercase();
        if lower.contains("match canvas")
            || lower.contains("use canvas")
            || lower.contains("canvas resolution")
            || lower == "canvas"
        {
            self.set_session_last_export_resolution_preference(
                session_id,
                Some("canvas".to_string()),
            );
        }
    }

    fn resolve_export_tool_args(&self, session_id: &SessionId, tool_args: Value) -> Value {
        let Some(mut obj) = tool_args.as_object().cloned() else {
            return tool_args;
        };

        if let Some(explicit) = obj
            .get("target_resolution")
            .and_then(|v| v.as_str())
            .and_then(Self::normalize_export_target_resolution_value)
        {
            obj.insert(
                "target_resolution".to_string(),
                Value::String(explicit.clone()),
            );
            self.set_session_last_export_resolution_preference(session_id, Some(explicit));
            return Value::Object(obj);
        }

        if obj.get("target_resolution").is_some() {
            obj.remove("target_resolution");
        }

        if let Some(preferred) = self.session_last_export_resolution_preference(session_id) {
            obj.insert("target_resolution".to_string(), Value::String(preferred));
        }

        Value::Object(obj)
    }

    fn trace_has_successful_apply(tool_trace: &[(String, String)]) -> bool {
        tool_trace.iter().rev().any(|(tool, raw)| {
            if tool != "anica.timeline/apply_edit_plan" {
                return false;
            }
            let Some(value) = Self::parse_json_value(raw) else {
                return false;
            };
            value.get("ok").and_then(|v| v.as_bool()) == Some(true)
                && value
                    .get("after_revision")
                    .and_then(|v| v.as_str())
                    .is_some()
        })
    }

    fn trace_last_snapshot_revision(tool_trace: &[(String, String)]) -> Option<String> {
        tool_trace
            .iter()
            .rev()
            .find(|(tool, _)| tool == "anica.timeline/get_snapshot")
            .and_then(|(_, raw)| Self::parse_json_value(raw))
            .and_then(|v| {
                v.get("timeline_revision")
                    .and_then(|r| r.as_str())
                    .map(|s| s.to_string())
            })
    }

    fn trace_last_build_operations(
        tool_trace: &[(String, String)],
        build_tool_name: &str,
    ) -> Option<Value> {
        tool_trace
            .iter()
            .rev()
            .find(|(tool, _)| tool == build_tool_name)
            .and_then(|(_, raw)| Self::parse_json_value(raw))
            .and_then(|v| v.get("operations").cloned())
    }

    fn trace_last_ok_and_errors(
        tool_trace: &[(String, String)],
        tool_name: &str,
    ) -> Option<(bool, Vec<String>)> {
        let value = tool_trace
            .iter()
            .rev()
            .find(|(tool, _)| tool == tool_name)
            .and_then(|(_, raw)| Self::parse_json_value(raw))?;

        let ok = value.get("ok").and_then(|v| v.as_bool())?;
        let errors = value
            .get("errors")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|e| e.as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default();
        Some((ok, errors))
    }

    fn trace_last_tool_index(tool_trace: &[(String, String)], tool_name: &str) -> Option<usize> {
        tool_trace
            .iter()
            .enumerate()
            .rev()
            .find(|(_, (tool, _))| tool == tool_name)
            .map(|(idx, _)| idx)
    }

    fn trace_last_semantic_anchor_index(tool_trace: &[(String, String)]) -> Option<usize> {
        tool_trace
            .iter()
            .enumerate()
            .rev()
            .find(|(_, (tool, _))| {
                tool == "anica.timeline/get_transcript_low_confidence_map"
                    || tool == "anica.timeline/build_transcript_low_confidence_cut_plan"
            })
            .map(|(idx, _)| idx)
    }

    fn trace_has_hybrid_second_check(tool_trace: &[(String, String)]) -> bool {
        tool_trace
            .iter()
            .any(|(tool, _)| tool == "anica.internal/hybrid_second_check")
    }

    fn trace_last_tool_payload(tool_trace: &[(String, String)], tool_name: &str) -> Option<Value> {
        tool_trace
            .iter()
            .rev()
            .find(|(tool, _)| tool == tool_name)
            .and_then(|(_, raw)| Self::parse_json_value(raw))
    }

    fn trace_last_semantic_payload(tool_trace: &[(String, String)]) -> Option<Value> {
        for (tool, raw) in tool_trace.iter().rev() {
            if (tool == "anica.timeline/get_transcript_low_confidence_map"
                || tool == "anica.timeline/build_transcript_low_confidence_cut_plan")
                && let Some(value) = Self::parse_json_value(raw)
            {
                return Some(value);
            }
        }
        None
    }

    fn compact_semantic_payload_for_second_check(payload: &Value) -> Value {
        let cut_candidates = payload
            .get("cut_candidates")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().take(128).cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let repeat_groups = payload
            .get("repeat_groups")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .take(64)
                    .map(|group| {
                        json!({
                            "group_id": group.get("group_id").cloned().unwrap_or(Value::Null),
                            "keep_clip_id": group.get("keep_clip_id").cloned().unwrap_or(Value::Null),
                            "confidence": group.get("confidence").cloned().unwrap_or(Value::Null),
                            "members": group.get("members").cloned().unwrap_or(Value::Array(Vec::new())),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        json!({
            "analysis_source": payload.get("analysis_source").cloned().unwrap_or(Value::Null),
            "window_ms": payload.get("window_ms").cloned().unwrap_or(Value::Null),
            "similarity_threshold": payload.get("similarity_threshold").cloned().unwrap_or(Value::Null),
            "cut_candidates": cut_candidates,
            "repeat_groups": repeat_groups,
            "warnings": payload.get("warnings").cloned().unwrap_or(Value::Array(Vec::new())),
        })
    }

    fn extract_subtitle_clips_for_second_check(snapshot: &Value) -> Vec<Value> {
        let mut rows: Vec<Value> = Vec::new();
        let Some(tracks) = snapshot.get("subtitle_tracks").and_then(|v| v.as_array()) else {
            return rows;
        };
        for (track_index, track) in tracks.iter().enumerate() {
            let Some(clips) = track.get("clips").and_then(|v| v.as_array()) else {
                continue;
            };
            for clip in clips {
                let Some(start_ms) = clip.get("start_ms").and_then(Self::parse_u64_from_value)
                else {
                    continue;
                };
                let duration_ms = clip
                    .get("duration_ms")
                    .and_then(Self::parse_u64_from_value)
                    .unwrap_or(0);
                let end_ms = start_ms.saturating_add(duration_ms);
                let text = clip
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let clip_id = clip
                    .get("clip_id")
                    .and_then(Self::parse_u64_from_value)
                    .unwrap_or(0);
                if clip_id == 0 || end_ms <= start_ms || text.trim().is_empty() {
                    continue;
                }
                rows.push(json!({
                    "clip_id": clip_id,
                    "track_index": track_index,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                }));
            }
        }
        rows.sort_by(|a, b| {
            let sa = a
                .get("start_ms")
                .and_then(Self::parse_u64_from_value)
                .unwrap_or(0);
            let sb = b
                .get("start_ms")
                .and_then(Self::parse_u64_from_value)
                .unwrap_or(0);
            sa.cmp(&sb)
        });
        if rows.len() > 420 {
            rows.truncate(420);
        }
        rows
    }

    fn build_hybrid_second_check_prompt(
        user_prompt: &str,
        semantic_payload: &Value,
        subtitle_rows: &[Value],
    ) -> String {
        let semantic_json = serde_json::to_string_pretty(semantic_payload)
            .unwrap_or_else(|_| semantic_payload.to_string());
        let subtitle_json = serde_json::to_string_pretty(subtitle_rows)
            .unwrap_or_else(|_| Value::Array(subtitle_rows.to_vec()).to_string());

        format!(
            "You are an editing second-check analyzer.\n\
            First round was rule-based. Now do a conservative LLM second-check for missed repeated speech.\n\
            Analyze repeated lines by these categories only:\n\
            - exact_repeat\n\
            - same_topic_consecutive_restart\n\
            - near_synonym_semantic_repeat\n\
            - prefix_or_continuation_restart\n\
            \n\
            Requirements:\n\
            - HARD RULE: you must analyze ALL FOUR categories above. Do not skip any category.\n\
            - HARD RULE: return a `category_assessments` array containing exactly these 4 category keys.\n\
            - Keep the final/best phrasing, prefer deleting earlier duplicate attempts.\n\
            - Use first-round analysis as context only; decide cuts from current subtitle clips.\n\
            - Ignore ranges shorter than 400ms.\n\
            - Be conservative; avoid over-cutting emphasis that is likely intentional.\n\
            - Return STRICT JSON only.\n\
            \n\
            JSON schema:\n\
            {{\n\
              \"category_assessments\": [\n\
                {{\"category\": \"exact_repeat\", \"decision\": \"has_candidate|no_candidate\", \"reason\": \"...\", \"candidate_count\": 0}},\n\
                {{\"category\": \"same_topic_consecutive_restart\", \"decision\": \"has_candidate|no_candidate\", \"reason\": \"...\", \"candidate_count\": 0}},\n\
                {{\"category\": \"near_synonym_semantic_repeat\", \"decision\": \"has_candidate|no_candidate\", \"reason\": \"...\", \"candidate_count\": 0}},\n\
                {{\"category\": \"prefix_or_continuation_restart\", \"decision\": \"has_candidate|no_candidate\", \"reason\": \"...\", \"candidate_count\": 0}}\n\
              ],\n\
              \"missed_cut_ranges\": [\n\
                {{\n\
                  \"start_ms\": 0,\n\
                  \"end_ms\": 0,\n\
                  \"category\": \"exact_repeat\",\n\
                  \"confidence\": 0.0,\n\
                  \"reason\": \"short explanation\",\n\
                  \"source_clip_ids\": [1,2]\n\
                }}\n\
              ],\n\
              \"notes\": [\"optional notes\"]\n\
            }}\n\
            \n\
            User request:\n\
            {user_prompt}\n\
            \n\
            First-round rule analysis JSON:\n\
            {semantic_json}\n\
            \n\
            Subtitle clips JSON:\n\
            {subtitle_json}\n"
        )
    }

    fn normalize_hybrid_second_check_category(raw: &str) -> Option<String> {
        let normalized = raw.trim().to_lowercase();
        if normalized.is_empty() {
            return None;
        }
        HYBRID_SECOND_CHECK_REQUIRED_CATEGORIES
            .iter()
            .find(|candidate| **candidate == normalized)
            .map(|candidate| (*candidate).to_string())
    }

    fn parse_hybrid_second_check_output(raw: &str) -> HybridSecondCheckParse {
        let Some(root) = Self::parse_json_value_relaxed(raw) else {
            return HybridSecondCheckParse {
                ranges: Vec::new(),
                categories_reported: Vec::new(),
                missing_categories: HYBRID_SECOND_CHECK_REQUIRED_CATEGORIES
                    .iter()
                    .map(|c| (*c).to_string())
                    .collect(),
                hard_rule_ok: false,
            };
        };

        let mut categories_reported: Vec<String> = root
            .get("category_assessments")
            .or_else(|| root.get("category_reports"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.get("category").and_then(|v| v.as_str()))
                    .filter_map(Self::normalize_hybrid_second_check_category)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        categories_reported.sort();
        categories_reported.dedup();

        let missing_categories = HYBRID_SECOND_CHECK_REQUIRED_CATEGORIES
            .iter()
            .filter(|required| {
                !categories_reported
                    .iter()
                    .any(|reported| reported == **required)
            })
            .map(|required| (*required).to_string())
            .collect::<Vec<_>>();

        let candidates = root
            .get("missed_cut_ranges")
            .or_else(|| root.get("candidate_ranges"))
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut out = Vec::new();
        for item in candidates {
            let Some(start_ms) = item.get("start_ms").and_then(Self::parse_u64_from_value) else {
                continue;
            };
            let Some(end_ms) = item.get("end_ms").and_then(Self::parse_u64_from_value) else {
                continue;
            };
            if end_ms <= start_ms || end_ms.saturating_sub(start_ms) < 400 {
                continue;
            }
            let category = item
                .get("category")
                .and_then(|v| v.as_str())
                .and_then(Self::normalize_hybrid_second_check_category)
                .unwrap_or_else(|| "exact_repeat".to_string());
            let confidence = item
                .get("confidence")
                .and_then(Self::parse_f32_from_value)
                .unwrap_or(0.75)
                .clamp(0.0, 1.0);
            let reason = item
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("llm_second_check")
                .to_string();
            let source_clip_ids = item
                .get("source_clip_ids")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(Self::parse_u64_from_value)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            out.push(HybridSecondCheckRange {
                start_ms,
                end_ms,
                category,
                confidence,
                reason,
                source_clip_ids,
            });
        }

        out.sort_by_key(|r| (r.start_ms, r.end_ms));
        if out.len() > 96 {
            out.truncate(96);
        }
        HybridSecondCheckParse {
            ranges: out,
            categories_reported,
            hard_rule_ok: missing_categories.is_empty(),
            missing_categories,
        }
    }

    fn merge_hybrid_ranges(ranges: &[HybridSecondCheckRange]) -> Vec<(u64, u64)> {
        if ranges.is_empty() {
            return Vec::new();
        }
        let mut sorted = ranges
            .iter()
            .map(|r| (r.start_ms, r.end_ms))
            .collect::<Vec<_>>();
        sorted.sort_by_key(|(start, end)| (*start, *end));

        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (start, end) in sorted {
            if let Some((_, last_end)) = merged.last_mut()
                && start <= last_end.saturating_add(40)
            {
                if end > *last_end {
                    *last_end = end;
                }
                continue;
            }
            merged.push((start, end));
        }
        merged
    }

    fn build_ripple_delete_operations(ranges: &[(u64, u64)]) -> Value {
        Value::Array(
            ranges
                .iter()
                .map(|(start_ms, end_ms)| {
                    json!({
                        "op": "ripple_delete_range",
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "mode": "all_tracks",
                    })
                })
                .collect(),
        )
    }

    async fn run_llm_similarity_only_flow(
        &self,
        session_id: &SessionId,
        cwd: &Path,
        user_prompt: &str,
    ) -> anyhow::Result<String> {
        self.emit_status("acp.status.llm_similarity_only_flow_start", &[]);
        let mut tool_results: Vec<String> = Vec::new();
        let mut tool_trace: Vec<(String, String)> = Vec::new();

        let snapshot_name = "anica.timeline/get_snapshot".to_string();
        let snapshot_json = self
            .execute_router_tool(
                session_id,
                &snapshot_name,
                json!({
                    "include_subtitles": true,
                }),
            )
            .await?;
        tool_trace.push((snapshot_name, snapshot_json.clone()));
        tool_results.push(snapshot_json.clone());

        let snapshot_payload = Self::parse_json_value(&snapshot_json).unwrap_or(Value::Null);
        let subtitle_rows = Self::extract_subtitle_clips_for_second_check(&snapshot_payload);
        if subtitle_rows.len() < 2 {
            let report = json!({
                "analysis_source": "llm_similarity_only_second_check",
                "status": "skipped",
                "reason": "not_enough_subtitles",
                "missed_cut_ranges": [],
            });
            let raw = report.to_string();
            tool_trace.push((
                "anica.internal/llm_similarity_only_check".to_string(),
                raw.clone(),
            ));
            tool_results.push(raw);
            if let Ok(synthesized) = self
                .synthesize_final_from_tool_results(session_id, cwd, user_prompt, &tool_results)
                .await
            {
                return Ok(synthesized);
            }
            return Ok(tool_results
                .last()
                .cloned()
                .unwrap_or_else(|| "No subtitle rows.".to_string()));
        }

        let llm_only_context = json!({
            "analysis_source": "llm_only",
            "cut_candidates": [],
            "repeat_groups": [],
            "warnings": [],
        });
        let prompt_tool_name = "anica.llm/decision_making_srt_similar_serach".to_string();
        let prompt_tool_json = self
            .execute_router_tool(
                session_id,
                &prompt_tool_name,
                json!({
                    "user_goal": user_prompt,
                    "first_pass_rule_analysis": llm_only_context,
                    "subtitle_rows": subtitle_rows.clone(),
                    "llm_only": true,
                }),
            )
            .await?;
        tool_trace.push((prompt_tool_name, prompt_tool_json.clone()));
        tool_results.push(prompt_tool_json.clone());

        let prompt = Self::parse_json_value(&prompt_tool_json)
            .and_then(|v| {
                v.get("prompt")
                    .and_then(|p| p.as_str())
                    .map(|s| s.to_string())
            })
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| {
                Self::build_hybrid_second_check_prompt(
                    user_prompt,
                    &json!({
                        "analysis_source": "llm_only",
                        "cut_candidates": [],
                        "repeat_groups": [],
                        "warnings": [],
                    }),
                    &subtitle_rows,
                )
            });
        self.emit_status("acp.status.llm_similarity_only_llm_check_start", &[]);
        let mut llm_raw = self.run_codex_prompt(session_id, cwd, &prompt).await?;
        self.emit_status("acp.status.llm_similarity_only_llm_check_done", &[]);
        let mut parsed = Self::parse_hybrid_second_check_output(&llm_raw);
        let mut retry_count = 0usize;
        while !parsed.hard_rule_ok && retry_count < 1 {
            let missing = serde_json::to_string(&parsed.missing_categories)
                .unwrap_or_else(|_| "[]".to_string());
            let repair_prompt = format!(
                "Your previous output violated a hard validator rule.\n\
                Missing category_assessments categories: {missing}\n\
                You MUST return STRICT JSON and include `category_assessments` with exactly these categories:\n\
                - exact_repeat\n\
                - same_topic_consecutive_restart\n\
                - near_synonym_semantic_repeat\n\
                - prefix_or_continuation_restart\n\
                Keep the same schema and be conservative.\n\
                Previous invalid output:\n\
                {previous}\n",
                previous = llm_raw
            );
            llm_raw = self
                .run_codex_prompt(session_id, cwd, &repair_prompt)
                .await?;
            parsed = Self::parse_hybrid_second_check_output(&llm_raw);
            retry_count += 1;
        }

        let parsed_ranges = parsed.ranges.clone();
        let merged_ranges = Self::merge_hybrid_ranges(&parsed_ranges);
        let hard_rule_ok = parsed.hard_rule_ok;
        let categories_reported = parsed.categories_reported.clone();
        let missing_categories = parsed.missing_categories.clone();
        let report = json!({
            "analysis_source": "llm_similarity_only_second_check",
            "status": if hard_rule_ok { "ok" } else { "invalid_missing_categories" },
            "hard_rule_ok": hard_rule_ok,
            "retry_count": retry_count,
            "categories_reported": categories_reported,
            "missing_categories": missing_categories,
            "missed_cut_ranges": parsed_ranges.iter().map(|r| {
                json!({
                    "start_ms": r.start_ms,
                    "end_ms": r.end_ms,
                    "category": r.category,
                    "confidence": r.confidence,
                    "reason": r.reason,
                    "source_clip_ids": r.source_clip_ids,
                })
            }).collect::<Vec<_>>(),
            "merged_ranges": merged_ranges.iter().map(|(start_ms, end_ms)| {
                json!({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                })
            }).collect::<Vec<_>>(),
        });
        let report_raw = report.to_string();
        tool_trace.push((
            "anica.internal/llm_similarity_only_check".to_string(),
            report_raw.clone(),
        ));
        tool_results.push(report_raw);

        if hard_rule_ok
            && !merged_ranges.is_empty()
            && let Some(revision) = Self::trace_last_snapshot_revision(&tool_trace)
        {
            let operations = Self::build_ripple_delete_operations(&merged_ranges);
            let validate_name = "anica.timeline/validate_edit_plan".to_string();
            let validate_json = self
                .execute_router_tool(
                    session_id,
                    &validate_name,
                    json!({
                        "based_on_revision": revision,
                        "operations": operations,
                    }),
                )
                .await?;
            tool_trace.push((validate_name, validate_json.clone()));
            tool_results.push(validate_json);

            let (ok, errors) =
                Self::trace_last_ok_and_errors(&tool_trace, "anica.timeline/validate_edit_plan")
                    .unwrap_or((false, vec!["validate_failed".to_string()]));
            if ok {
                let apply_revision =
                    Self::trace_last_snapshot_revision(&tool_trace).unwrap_or(revision);
                let apply_name = "anica.timeline/apply_edit_plan".to_string();
                let apply_json = self
                    .execute_router_tool(
                        session_id,
                        &apply_name,
                        json!({
                            "based_on_revision": apply_revision,
                            "operations": operations,
                        }),
                    )
                    .await?;
                tool_trace.push((apply_name, apply_json.clone()));
                tool_results.push(apply_json);
            } else if Self::errors_indicate_revision_mismatch(&errors) {
                let refresh_name = "anica.timeline/get_snapshot".to_string();
                let refresh_json = self
                    .execute_router_tool(
                        session_id,
                        &refresh_name,
                        json!({
                            "include_subtitles": true,
                        }),
                    )
                    .await?;
                tool_trace.push((refresh_name, refresh_json.clone()));
                tool_results.push(refresh_json);
            }
        }

        if let Ok(synthesized) = self
            .synthesize_final_from_tool_results(session_id, cwd, user_prompt, &tool_results)
            .await
        {
            return Ok(synthesized);
        }

        Ok(tool_results
            .last()
            .cloned()
            .unwrap_or_else(|| "Done.".to_string()))
    }

    async fn maybe_run_hybrid_second_check(
        &self,
        session_id: &SessionId,
        cwd: &Path,
        user_prompt: &str,
        tool_trace: &mut Vec<(String, String)>,
        tool_results: &mut Vec<String>,
    ) -> anyhow::Result<bool> {
        if Self::trace_has_hybrid_second_check(tool_trace) {
            return Ok(false);
        }

        let Some(semantic_anchor_idx) = Self::trace_last_semantic_anchor_index(tool_trace) else {
            return Ok(false);
        };
        let latest_snapshot_idx =
            Self::trace_last_tool_index(tool_trace, "anica.timeline/get_snapshot");
        let latest_apply_idx =
            Self::trace_last_tool_index(tool_trace, "anica.timeline/apply_edit_plan");

        let needs_fresh_snapshot = match (latest_snapshot_idx, latest_apply_idx) {
            (Some(snapshot_idx), Some(apply_idx)) => snapshot_idx <= apply_idx,
            (Some(snapshot_idx), None) => snapshot_idx <= semantic_anchor_idx,
            (None, _) => true,
        };

        // Round 2 (LLM second check) always needs subtitle snapshot newer than first-pass apply (if any).
        if needs_fresh_snapshot {
            let tool_name = "anica.timeline/get_snapshot".to_string();
            let tool_json = self
                .execute_router_tool(
                    session_id,
                    &tool_name,
                    json!({
                        "include_subtitles": true,
                    }),
                )
                .await?;
            tool_trace.push((tool_name, tool_json.clone()));
            tool_results.push(tool_json);
            return Ok(true);
        }

        let semantic_payload = match Self::trace_last_semantic_payload(tool_trace) {
            Some(v) => v,
            None => return Ok(false),
        };

        let Some(snapshot_payload) =
            Self::trace_last_tool_payload(tool_trace, "anica.timeline/get_snapshot")
        else {
            return Ok(false);
        };
        let subtitle_rows = Self::extract_subtitle_clips_for_second_check(&snapshot_payload);
        if subtitle_rows.len() < 2 {
            let report = json!({
                "analysis_source": "hybrid_second_check",
                "status": "skipped",
                "reason": "not_enough_subtitles",
                "missed_cut_ranges": [],
            });
            let raw = report.to_string();
            tool_trace.push((
                "anica.internal/hybrid_second_check".to_string(),
                raw.clone(),
            ));
            tool_results.push(raw);
            return Ok(true);
        }

        let compact_semantic = Self::compact_semantic_payload_for_second_check(&semantic_payload);
        let prompt_tool_name = "anica.llm/decision_making_srt_similar_serach".to_string();
        let prompt_tool_json = self
            .execute_router_tool(
                session_id,
                &prompt_tool_name,
                json!({
                    "user_goal": user_prompt,
                    "first_pass_rule_analysis": compact_semantic,
                    "subtitle_rows": subtitle_rows.clone(),
                }),
            )
            .await?;
        tool_trace.push((prompt_tool_name, prompt_tool_json.clone()));
        tool_results.push(prompt_tool_json.clone());

        let prompt = Self::parse_json_value(&prompt_tool_json)
            .and_then(|v| {
                v.get("prompt")
                    .and_then(|p| p.as_str())
                    .map(|s| s.to_string())
            })
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| {
                Self::build_hybrid_second_check_prompt(
                    user_prompt,
                    &compact_semantic,
                    &subtitle_rows,
                )
            });
        let mut llm_raw = self.run_codex_prompt(session_id, cwd, &prompt).await?;
        let mut parsed = Self::parse_hybrid_second_check_output(&llm_raw);
        let mut retry_count = 0usize;

        while !parsed.hard_rule_ok && retry_count < 1 {
            let missing = serde_json::to_string(&parsed.missing_categories)
                .unwrap_or_else(|_| "[]".to_string());
            let repair_prompt = format!(
                "Your previous output violated a hard validator rule.\n\
                Missing category_assessments categories: {missing}\n\
                You MUST return STRICT JSON and include `category_assessments` with exactly these categories:\n\
                - exact_repeat\n\
                - same_topic_consecutive_restart\n\
                - near_synonym_semantic_repeat\n\
                - prefix_or_continuation_restart\n\
                Keep the same schema and be conservative.\n\
                Previous invalid output:\n\
                {previous}\n",
                previous = llm_raw
            );
            llm_raw = self
                .run_codex_prompt(session_id, cwd, &repair_prompt)
                .await?;
            parsed = Self::parse_hybrid_second_check_output(&llm_raw);
            retry_count += 1;
        }

        let parsed_ranges = parsed.ranges.clone();
        let merged_ranges = Self::merge_hybrid_ranges(&parsed_ranges);
        let hard_rule_ok = parsed.hard_rule_ok;
        let categories_reported = parsed.categories_reported.clone();
        let missing_categories = parsed.missing_categories.clone();

        let second_check_report = json!({
            "analysis_source": "hybrid_second_check",
            "status": if hard_rule_ok { "ok" } else { "invalid_missing_categories" },
            "hard_rule_ok": hard_rule_ok,
            "retry_count": retry_count,
            "categories_reported": categories_reported,
            "missing_categories": missing_categories,
            "missed_cut_ranges": parsed_ranges.iter().map(|r| {
                json!({
                    "start_ms": r.start_ms,
                    "end_ms": r.end_ms,
                    "category": r.category,
                    "confidence": r.confidence,
                    "reason": r.reason,
                    "source_clip_ids": r.source_clip_ids,
                })
            }).collect::<Vec<_>>(),
            "merged_ranges": merged_ranges.iter().map(|(start_ms, end_ms)| {
                json!({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                })
            }).collect::<Vec<_>>(),
        });
        let second_check_raw = second_check_report.to_string();
        tool_trace.push((
            "anica.internal/hybrid_second_check".to_string(),
            second_check_raw.clone(),
        ));
        tool_results.push(second_check_raw);

        if !hard_rule_ok {
            return Ok(true);
        }

        if merged_ranges.is_empty() {
            return Ok(true);
        }

        let Some(revision) = Self::trace_last_snapshot_revision(tool_trace) else {
            return Ok(true);
        };
        let operations = Self::build_ripple_delete_operations(&merged_ranges);

        let validate_name = "anica.timeline/validate_edit_plan".to_string();
        let validate_json = self
            .execute_router_tool(
                session_id,
                &validate_name,
                json!({
                    "based_on_revision": revision,
                    "operations": operations,
                }),
            )
            .await?;
        tool_trace.push((validate_name, validate_json.clone()));
        tool_results.push(validate_json);

        let (ok, errors) =
            Self::trace_last_ok_and_errors(tool_trace, "anica.timeline/validate_edit_plan")
                .unwrap_or((false, vec!["validate_failed".to_string()]));
        if !ok {
            if Self::errors_indicate_revision_mismatch(&errors) {
                let snapshot_name = "anica.timeline/get_snapshot".to_string();
                let snapshot_json = self
                    .execute_router_tool(
                        session_id,
                        &snapshot_name,
                        json!({
                            "include_subtitles": true,
                        }),
                    )
                    .await?;
                tool_trace.push((snapshot_name, snapshot_json.clone()));
                tool_results.push(snapshot_json);
            }
            return Ok(true);
        }

        let apply_revision = Self::trace_last_snapshot_revision(tool_trace).unwrap_or(revision);
        let apply_name = "anica.timeline/apply_edit_plan".to_string();
        let apply_json = self
            .execute_router_tool(
                session_id,
                &apply_name,
                json!({
                    "based_on_revision": apply_revision,
                    "operations": operations,
                }),
            )
            .await?;
        tool_trace.push((apply_name, apply_json.clone()));
        tool_results.push(apply_json);

        Ok(true)
    }

    fn errors_indicate_revision_mismatch(errors: &[String]) -> bool {
        errors.iter().any(|err| {
            let s = err.to_lowercase();
            s.contains("revision")
                || s.contains("stale")
                || s.contains("based_on_revision")
                || s.contains("snapshot")
        })
    }

    fn promote_subtitle_gap_query_to_build_args(args: &Value) -> Value {
        let mut out = match args.as_object() {
            Some(obj) => obj.clone(),
            None => Map::new(),
        };
        out.entry("mode".to_string())
            .or_insert_with(|| Value::String("balanced".to_string()));
        out.entry("include_head_tail".to_string())
            .or_insert(Value::Bool(true));
        out.insert(
            "cut_strategy".to_string(),
            Value::String("subtitle_only".to_string()),
        );
        Value::Object(out)
    }

    fn forced_direct_cut_next_tool(
        &self,
        session_id: &SessionId,
        kind: DirectCutKind,
        tool_trace: &[(String, String)],
    ) -> Option<(String, Value)> {
        if Self::trace_has_successful_apply(tool_trace) {
            return None;
        }

        if tool_trace
            .last()
            .is_some_and(|(tool, _)| tool == "anica.timeline/apply_edit_plan")
            && let Some((ok, errors)) =
                Self::trace_last_ok_and_errors(tool_trace, "anica.timeline/apply_edit_plan")
            && !ok
            && !Self::errors_indicate_revision_mismatch(&errors)
        {
            return None;
        }

        let build_tool_name = match kind {
            DirectCutKind::Silence => "anica.timeline/build_audio_silence_cut_plan",
            DirectCutKind::SubtitleGap => "anica.timeline/build_subtitle_gap_cut_plan",
        };

        if let Some(ops) = Self::trace_last_build_operations(tool_trace, build_tool_name) {
            if ops.as_array().is_some_and(|arr| arr.is_empty()) {
                return None;
            }

            if tool_trace
                .last()
                .is_some_and(|(tool, _)| tool == "anica.timeline/validate_edit_plan")
                && let Some((ok, errors)) =
                    Self::trace_last_ok_and_errors(tool_trace, "anica.timeline/validate_edit_plan")
            {
                if ok {
                    if let Some(rev) = Self::trace_last_snapshot_revision(tool_trace) {
                        return Some((
                            "anica.timeline/apply_edit_plan".to_string(),
                            json!({
                                "based_on_revision": rev,
                                "operations": ops,
                            }),
                        ));
                    }
                    return Some((
                        "anica.timeline/get_snapshot".to_string(),
                        json!({
                            "include_subtitles": true,
                        }),
                    ));
                }
                if !Self::errors_indicate_revision_mismatch(&errors) {
                    return None;
                }
            }

            if let Some(rev) = Self::trace_last_snapshot_revision(tool_trace) {
                return Some((
                    "anica.timeline/validate_edit_plan".to_string(),
                    json!({
                        "based_on_revision": rev,
                        "operations": ops,
                    }),
                ));
            }

            return Some((
                "anica.timeline/get_snapshot".to_string(),
                json!({
                    "include_subtitles": true,
                }),
            ));
        }

        let build_args = match kind {
            DirectCutKind::Silence => {
                self.resolve_audio_silence_tool_args(session_id, &Value::Object(Map::new()))
            }
            DirectCutKind::SubtitleGap => json!({
                "mode": "balanced",
                "cut_strategy": "subtitle_only",
                "include_head_tail": true,
            }),
        };
        Some((build_tool_name.to_string(), build_args))
    }

    async fn execute_router_tool(
        &self,
        session_id: &SessionId,
        tool_name: &str,
        mut tool_args: Value,
    ) -> anyhow::Result<String> {
        if tool_name == "anica.timeline/get_audio_silence_map"
            || tool_name == "anica.timeline/build_audio_silence_cut_plan"
        {
            tool_args = self.resolve_audio_silence_tool_args(session_id, &tool_args);
        }
        tool_args = Self::normalize_timeline_edit_tool_args(tool_name, tool_args);
        if tool_name == "anica.timeline/validate_edit_plan"
            || tool_name == "anica.timeline/apply_edit_plan"
        {
            tool_args = self.resolve_same_operations_placeholder(session_id, tool_name, tool_args);
            tool_args = Self::normalize_timeline_edit_tool_args(tool_name, tool_args);
        }

        self.emit_status(
            "acp.status.calling_tool",
            &[("tool", tool_name.to_string())],
        );
        self.emit_status(
            "acp.status.tool_args",
            &[
                ("tool", tool_name.to_string()),
                ("args", Self::compact_json_for_status(&tool_args, 260)),
            ],
        );

        let tool_json = self
            .call_ext_tool(tool_name, &tool_args.to_string())
            .await?;
        eprintln!("[ACP SYSTEM][TOOL_RAW][{tool_name}] {tool_json}");
        self.emit_status("acp.status.tool_done", &[("tool", tool_name.to_string())]);
        self.emit_status(
            "acp.status.tool_result_summary",
            &[(
                "summary",
                Self::summarize_tool_result(tool_name, &tool_json),
            )],
        );

        if (tool_name == "anica.timeline/validate_edit_plan"
            || tool_name == "anica.timeline/apply_edit_plan")
            && let Some(ops) = tool_args.get("operations")
            && ops.as_array().is_some()
        {
            if tool_name == "anica.timeline/apply_edit_plan" {
                self.set_session_last_validated_operations(session_id, Some(ops.clone()));
            } else {
                let ok = Self::parse_json_value(&tool_json)
                    .and_then(|v| v.get("ok").and_then(|v| v.as_bool()))
                    .unwrap_or(false);
                if ok {
                    self.set_session_last_validated_operations(session_id, Some(ops.clone()));
                }
            }
        }
        Ok(tool_json)
    }

    async fn execute_subtitle_translation_tool(
        &self,
        session_id: &SessionId,
        cwd: &Path,
        user_prompt: &str,
        tool_args: &Value,
    ) -> anyhow::Result<String> {
        let mut warnings: Vec<String> = Vec::new();
        let mut target_language = Self::extract_target_language_from_tool_args(tool_args);
        if target_language.is_none() {
            target_language = Self::infer_target_language_from_prompt(user_prompt);
        }
        let Some(target_language) = target_language else {
            return Ok(json!({
                "ok": false,
                "needs_target_language": true,
                "question": "Which target language should I translate the subtitles into?",
                "hint": "Provide `target_language` (for example: English, Japanese, French).",
            })
            .to_string());
        };

        let mut requested_track_indices =
            Self::extract_requested_track_indices_from_tool_args(tool_args);
        if requested_track_indices.is_none() {
            requested_track_indices = Self::infer_requested_track_indices_from_prompt(user_prompt);
        }
        let extra_instruction = tool_args
            .get("instruction")
            .or_else(|| tool_args.get("style_instruction"))
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(ToString::to_string);

        let snapshot_json = self
            .execute_router_tool(
                session_id,
                "anica.timeline/get_snapshot",
                json!({ "include_subtitles": true }),
            )
            .await?;
        let Some(snapshot_payload) = Self::parse_json_value(&snapshot_json) else {
            return Ok(json!({
                "ok": false,
                "stage": "snapshot_before",
                "errors": ["Invalid timeline snapshot JSON."],
            })
            .to_string());
        };
        let Some(base_revision) = snapshot_payload
            .get("timeline_revision")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
        else {
            return Ok(json!({
                "ok": false,
                "stage": "snapshot_before",
                "errors": ["Timeline snapshot missing revision."],
            })
            .to_string());
        };

        let (rows, selected_tracks) = Self::collect_subtitle_rows_for_translation(
            &snapshot_payload,
            requested_track_indices,
            &mut warnings,
        );
        if rows.is_empty() {
            return Ok(json!({
                "ok": false,
                "stage": "collect_source_subtitles",
                "errors": ["No subtitle clips found for translation in the selected tracks."],
                "warnings": warnings,
            })
            .to_string());
        }

        let mut translated_text_by_clip: HashMap<u64, String> = HashMap::new();
        let mut llm_call_count = 0usize;
        let mut fallback_clip_count = 0usize;
        for chunk in rows.chunks(SUBTITLE_TRANSLATION_LLM_CHUNK_SIZE) {
            llm_call_count = llm_call_count.saturating_add(1);
            let prompt = Self::build_subtitle_translation_prompt(
                user_prompt,
                target_language.as_str(),
                chunk,
                extra_instruction.as_deref(),
            );
            let mut llm_raw = self.run_codex_prompt(session_id, cwd, &prompt).await?;
            let mut parsed = Self::parse_subtitle_translation_output(&llm_raw);

            let mut retry = 0usize;
            while parsed.len() < chunk.len() && retry < 1 {
                let missing_ids = chunk
                    .iter()
                    .filter(|row| !parsed.contains_key(&row.clip_id))
                    .map(|row| row.clip_id)
                    .collect::<Vec<_>>();
                if missing_ids.is_empty() {
                    break;
                }
                let repair_prompt = format!(
                    "Your previous output is missing some clip_id translations.\n\
                    Return STRICT JSON only with schema {{\"translations\":[{{\"clip_id\":123,\"text\":\"...\"}}]}}.\n\
                    Missing clip_ids: {missing_ids}\n\
                    Previous output:\n{previous_output}\n",
                    missing_ids =
                        serde_json::to_string(&missing_ids).unwrap_or_else(|_| "[]".to_string()),
                    previous_output = llm_raw
                );
                llm_raw = self
                    .run_codex_prompt(session_id, cwd, &repair_prompt)
                    .await?;
                parsed = Self::parse_subtitle_translation_output(&llm_raw);
                retry = retry.saturating_add(1);
            }

            for row in chunk {
                if let Some(translated) = parsed.get(&row.clip_id) {
                    translated_text_by_clip.insert(row.clip_id, translated.clone());
                } else {
                    translated_text_by_clip.insert(row.clip_id, row.text.clone());
                    fallback_clip_count = fallback_clip_count.saturating_add(1);
                }
            }
        }
        if fallback_clip_count > 0 {
            warnings.push(format!(
                "Used source text fallback for {fallback_clip_count} clip(s) because translation output missed those clip_ids."
            ));
        }

        let add_track_name = format!("Translated ({})", target_language.trim());
        let add_track_ops = json!([{
            "op": "add_subtitle_track",
            "name": add_track_name,
        }]);
        let validate_add_json = self
            .execute_router_tool(
                session_id,
                "anica.timeline/validate_edit_plan",
                json!({
                    "based_on_revision": base_revision,
                    "operations": add_track_ops,
                }),
            )
            .await?;
        let (validate_add_ok, validate_add_errors, _) =
            Self::parse_plan_tool_ok_errors_and_after_revision(&validate_add_json);
        if !validate_add_ok {
            return Ok(json!({
                "ok": false,
                "stage": "validate_add_subtitle_track",
                "errors": validate_add_errors,
                "warnings": warnings,
            })
            .to_string());
        }

        let apply_add_json = self
            .execute_router_tool(
                session_id,
                "anica.timeline/apply_edit_plan",
                json!({
                    "based_on_revision": base_revision,
                    // Let the API auto-generate "S{n}" based on existing tracks,
                    // same behavior as the "+S" button.
                    "operations": json!([{
                        "op": "add_subtitle_track",
                    }]),
                }),
            )
            .await?;
        let (apply_add_ok, apply_add_errors, _) =
            Self::parse_plan_tool_ok_errors_and_after_revision(&apply_add_json);
        if !apply_add_ok {
            return Ok(json!({
                "ok": false,
                "stage": "apply_add_subtitle_track",
                "errors": apply_add_errors,
                "warnings": warnings,
            })
            .to_string());
        }

        let after_add_snapshot_json = self
            .execute_router_tool(
                session_id,
                "anica.timeline/get_snapshot",
                json!({ "include_subtitles": true }),
            )
            .await?;
        let Some(after_add_snapshot) = Self::parse_json_value(&after_add_snapshot_json) else {
            return Ok(json!({
                "ok": false,
                "stage": "snapshot_after_add_subtitle_track",
                "errors": ["Invalid timeline snapshot JSON after adding subtitle track."],
                "warnings": warnings,
            })
            .to_string());
        };
        let Some(mut current_revision) = after_add_snapshot
            .get("timeline_revision")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
        else {
            return Ok(json!({
                "ok": false,
                "stage": "snapshot_after_add_subtitle_track",
                "errors": ["Timeline snapshot missing revision after add track."],
                "warnings": warnings,
            })
            .to_string());
        };
        let subtitle_tracks = after_add_snapshot
            .get("subtitle_tracks")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if subtitle_tracks.is_empty() {
            return Ok(json!({
                "ok": false,
                "stage": "snapshot_after_add_subtitle_track",
                "errors": ["No subtitle tracks available after adding translated track."],
                "warnings": warnings,
            })
            .to_string());
        }
        let output_track_index = subtitle_tracks
            .last()
            .and_then(|track| track.get("index"))
            .and_then(Self::parse_u64_from_value)
            .map(|v| v as usize)
            .unwrap_or_else(|| subtitle_tracks.len().saturating_sub(1));
        let output_track_name = subtitle_tracks
            .last()
            .and_then(|track| track.get("name"))
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("S{}", output_track_index.saturating_add(1)));

        let generated_entries = rows
            .iter()
            .map(|row| {
                json!({
                    "text": translated_text_by_clip
                        .get(&row.clip_id)
                        .cloned()
                        .unwrap_or_else(|| row.text.clone()),
                    "start_ms": row.start_ms,
                    "duration_ms": row.duration_ms.max(1),
                })
            })
            .collect::<Vec<_>>();

        let mut translated_clip_count = 0usize;
        for chunk in generated_entries.chunks(SUBTITLE_TRANSLATION_APPLY_CHUNK_SIZE) {
            let chunk_entries = chunk.to_vec();
            let validate_generate_json = self
                .execute_router_tool(
                    session_id,
                    "anica.timeline/validate_edit_plan",
                    json!({
                        "based_on_revision": current_revision,
                        "operations": [{
                            "op": "generate_subtitles",
                            "track_index": output_track_index,
                            "entries": chunk_entries,
                        }],
                    }),
                )
                .await?;
            let (validate_generate_ok, validate_generate_errors, _) =
                Self::parse_plan_tool_ok_errors_and_after_revision(&validate_generate_json);
            if !validate_generate_ok {
                return Ok(json!({
                    "ok": false,
                    "stage": "validate_generate_subtitles",
                    "errors": validate_generate_errors,
                    "warnings": warnings,
                    "output_track_index": output_track_index,
                })
                .to_string());
            }

            let apply_generate_json = self
                .execute_router_tool(
                    session_id,
                    "anica.timeline/apply_edit_plan",
                    json!({
                        "based_on_revision": current_revision,
                        "operations": [{
                            "op": "generate_subtitles",
                            "track_index": output_track_index,
                            "entries": chunk.to_vec(),
                        }],
                    }),
                )
                .await?;
            let (apply_generate_ok, apply_generate_errors, apply_after_revision) =
                Self::parse_plan_tool_ok_errors_and_after_revision(&apply_generate_json);
            if !apply_generate_ok {
                return Ok(json!({
                    "ok": false,
                    "stage": "apply_generate_subtitles",
                    "errors": apply_generate_errors,
                    "warnings": warnings,
                    "output_track_index": output_track_index,
                })
                .to_string());
            }
            if let Some(after_revision) = apply_after_revision {
                current_revision = after_revision;
            }
            translated_clip_count = translated_clip_count.saturating_add(chunk.len());
        }

        Ok(json!({
            "ok": true,
            "target_language": target_language,
            "selected_track_indices": selected_tracks,
            "translated_clip_count": translated_clip_count,
            "source_clip_count": rows.len(),
            "fallback_clip_count": fallback_clip_count,
            "output_track_index": output_track_index,
            "output_track_name": output_track_name,
            "llm_call_count": llm_call_count,
            "before_revision": base_revision,
            "after_revision": current_revision,
            "warnings": warnings,
        })
        .to_string())
    }

    fn audio_silence_arg_keys() -> &'static [&'static str] {
        &[
            "rms_threshold_db",
            "min_silence_ms",
            "pad_ms",
            "detect_low_energy_repeats",
            "repeat_similarity_threshold",
            "repeat_window_ms",
        ]
    }

    fn extract_audio_silence_args(value: &Value) -> Map<String, Value> {
        let mut out = Map::new();
        let Some(obj) = value.as_object() else {
            return out;
        };
        for key in Self::audio_silence_arg_keys() {
            if let Some(v) = obj.get(*key) {
                out.insert((*key).to_string(), v.clone());
            }
        }
        out
    }

    fn session_last_audio_silence_args(&self, session_id: &SessionId) -> Option<Value> {
        self.inner
            .sessions
            .borrow()
            .get(session_id)
            .and_then(|s| s.last_audio_silence_args.clone())
    }

    fn set_session_last_audio_silence_args(&self, session_id: &SessionId, args: Option<Value>) {
        if let Some(s) = self.inner.sessions.borrow_mut().get_mut(session_id) {
            s.last_audio_silence_args = args;
        }
    }

    fn session_last_validated_operations(&self, session_id: &SessionId) -> Option<Value> {
        self.inner
            .sessions
            .borrow()
            .get(session_id)
            .and_then(|s| s.last_validated_operations.clone())
    }

    fn set_session_last_validated_operations(&self, session_id: &SessionId, ops: Option<Value>) {
        if let Some(s) = self.inner.sessions.borrow_mut().get_mut(session_id) {
            s.last_validated_operations = ops;
        }
    }

    fn resolve_same_operations_placeholder(
        &self,
        session_id: &SessionId,
        tool_name: &str,
        tool_args: Value,
    ) -> Value {
        let Some(mut obj) = tool_args.as_object().cloned() else {
            return tool_args;
        };

        let cached_ops = self
            .session_last_validated_operations(session_id)
            .and_then(|ops| {
                ops.as_array()
                    .is_some_and(|arr| !arr.is_empty())
                    .then_some(ops)
            });

        if !obj.contains_key("operations") {
            if tool_name == "anica.timeline/apply_edit_plan"
                && let Some(ops) = cached_ops
            {
                obj.insert("operations".to_string(), ops);
                self.emit_status("acp.status.reuse_last_validated_operations", &[]);
            }
            return Value::Object(obj);
        }

        let Some(ops_value) = obj.get("operations") else {
            return Value::Object(obj);
        };

        if ops_value.as_array().is_some_and(|arr| arr.is_empty())
            && tool_name == "anica.timeline/apply_edit_plan"
            && let Some(ops) = cached_ops.clone()
        {
            obj.insert("operations".to_string(), ops);
            self.emit_status("acp.status.reuse_last_validated_operations", &[]);
            return Value::Object(obj);
        }

        if !Self::is_same_validated_operations_placeholder(ops_value) {
            return Value::Object(obj);
        }

        if let Some(cached_ops) = cached_ops {
            obj.insert("operations".to_string(), cached_ops);
            self.emit_status("acp.status.reuse_last_validated_operations", &[]);
        } else {
            obj.insert("operations".to_string(), Value::Array(Vec::new()));
        }
        Value::Object(obj)
    }

    fn resolve_audio_silence_tool_args(&self, session_id: &SessionId, incoming: &Value) -> Value {
        let incoming_args = Self::extract_audio_silence_args(incoming);
        if !incoming_args.is_empty() {
            let mut merged = self
                .session_last_audio_silence_args(session_id)
                .as_ref()
                .map(Self::extract_audio_silence_args)
                .unwrap_or_default();
            for (k, v) in incoming_args {
                merged.insert(k, v);
            }
            let merged_value = Value::Object(merged);
            self.set_session_last_audio_silence_args(session_id, Some(merged_value.clone()));
            return merged_value;
        }

        if let Some(cached) = self.session_last_audio_silence_args(session_id) {
            return cached;
        }

        Value::Object(Map::new())
    }

    fn bind_connection(&self, conn: Rc<AgentSideConnection>) {
        self.inner.conn.replace(Some(conn));
    }

    fn connection(&self) -> agent_client_protocol::Result<Rc<AgentSideConnection>> {
        self.inner
            .conn
            .borrow()
            .as_ref()
            .cloned()
            .ok_or_else(Error::internal_error)
    }

    fn alloc_session_id(&self) -> SessionId {
        let id = self.inner.next_id.get();
        self.inner.next_id.set(id.saturating_add(1));
        SessionId::new(format!("anica-session-{id}"))
    }

    fn set_cancelled(&self, session_id: &SessionId, cancelled: bool) -> bool {
        if let Some(s) = self.inner.sessions.borrow_mut().get_mut(session_id) {
            s.cancelled = cancelled;
            true
        } else {
            false
        }
    }

    fn is_cancelled(&self, session_id: &SessionId) -> bool {
        self.inner
            .sessions
            .borrow()
            .get(session_id)
            .map(|s| s.cancelled)
            .unwrap_or(false)
    }

    fn session_exists(&self, session_id: &SessionId) -> bool {
        self.inner.sessions.borrow().contains_key(session_id)
    }

    fn session_cwd(&self, session_id: &SessionId) -> Option<PathBuf> {
        self.inner
            .sessions
            .borrow()
            .get(session_id)
            .map(|s| s.cwd.clone())
    }

    fn set_running_pid(&self, session_id: &SessionId, pid: Option<u32>) {
        if let Some(s) = self.inner.sessions.borrow_mut().get_mut(session_id) {
            s.running_pid = pid;
        }
    }

    fn running_pid(&self, session_id: &SessionId) -> Option<u32> {
        self.inner
            .sessions
            .borrow()
            .get(session_id)
            .and_then(|s| s.running_pid)
    }

    async fn send_agent_chunk(
        &self,
        session_id: &SessionId,
        text: String,
    ) -> agent_client_protocol::Result<()> {
        eprintln!("[ACP AGENT] {text}");
        let conn = self.connection()?;
        conn.session_notification(SessionNotification::new(
            session_id.clone(),
            SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::from(text))),
        ))
        .await
    }

    async fn run_codex_prompt(
        &self,
        session_id: &SessionId,
        cwd: &Path,
        prompt: &str,
    ) -> anyhow::Result<String> {
        let provider = std::env::var("ANICA_ACP_PROVIDER")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "codex".to_string());
        if provider == "gemini" {
            let gemini_bin = runtime_paths::resolve_cli_bin("ANICA_GEMINI_CLI_BIN", "gemini")
                .unwrap_or_else(|| PathBuf::from("gemini"))
                .to_string_lossy()
                .to_string();
            if !command_exists(&gemini_bin) {
                anyhow::bail!(
                    "Gemini CLI not found: `{}`. Install Gemini CLI or set ANICA_GEMINI_CLI_BIN to a valid path.",
                    gemini_bin
                );
            }

            let mut cmd = TokioCommand::new(&gemini_bin);
            cmd.arg("--prompt")
                .arg(prompt)
                .current_dir(cwd)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            let child = cmd.spawn().map_err(|err| {
                anyhow::anyhow!("Failed to spawn `{}`. Is it installed? ({err})", gemini_bin)
            })?;
            self.set_running_pid(session_id, child.id());

            let output = child
                .wait_with_output()
                .await
                .map_err(|err| anyhow::anyhow!("Failed waiting for gemini process: {err}"))?;
            self.set_running_pid(session_id, None);

            if self.is_cancelled(session_id) {
                anyhow::bail!("cancelled");
            }

            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if !output.status.success() {
                let detail = if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status: {}", output.status)
                };
                anyhow::bail!("Gemini CLI failed: {detail}");
            }

            if stdout.is_empty() {
                anyhow::bail!("Gemini CLI returned empty output.");
            }

            return Ok(stdout);
        }

        if provider == "claude" {
            let claude_bin = runtime_paths::resolve_cli_bin("ANICA_CLAUDE_CLI_BIN", "claude")
                .unwrap_or_else(|| PathBuf::from("claude"))
                .to_string_lossy()
                .to_string();
            if !command_exists(&claude_bin) {
                anyhow::bail!(
                    "Claude CLI not found: `{}`. Install Claude CLI or set ANICA_CLAUDE_CLI_BIN to a valid path.",
                    claude_bin
                );
            }

            let mut cmd = TokioCommand::new(&claude_bin);
            cmd.arg("-p")
                .arg(prompt)
                .arg("--output-format")
                .arg("text")
                .current_dir(cwd)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            let child = cmd.spawn().map_err(|err| {
                anyhow::anyhow!("Failed to spawn `{}`. Is it installed? ({err})", claude_bin)
            })?;
            self.set_running_pid(session_id, child.id());

            let mut output = child
                .wait_with_output()
                .await
                .map_err(|err| anyhow::anyhow!("Failed waiting for claude process: {err}"))?;
            self.set_running_pid(session_id, None);

            if self.is_cancelled(session_id) {
                anyhow::bail!("cancelled");
            }

            if !output.status.success() {
                let stderr_lc = String::from_utf8_lossy(&output.stderr).to_lowercase();
                let stdout_lc = String::from_utf8_lossy(&output.stdout).to_lowercase();
                let unknown_output_format = stderr_lc.contains("unknown option '--output-format'")
                    || stderr_lc.contains("unknown option: --output-format")
                    || stderr_lc.contains("unrecognized option '--output-format'")
                    || stdout_lc.contains("unknown option '--output-format'")
                    || stdout_lc.contains("unknown option: --output-format")
                    || stdout_lc.contains("unrecognized option '--output-format'");
                if unknown_output_format {
                    let mut fallback_cmd = TokioCommand::new(&claude_bin);
                    fallback_cmd
                        .arg("-p")
                        .arg(prompt)
                        .current_dir(cwd)
                        .stdin(Stdio::null())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .kill_on_drop(true);
                    let fallback_child = fallback_cmd.spawn().map_err(|err| {
                        anyhow::anyhow!(
                            "Failed to spawn `{}`. Is it installed? ({err})",
                            claude_bin
                        )
                    })?;
                    self.set_running_pid(session_id, fallback_child.id());
                    output = fallback_child.wait_with_output().await.map_err(|err| {
                        anyhow::anyhow!("Failed waiting for claude process: {err}")
                    })?;
                    self.set_running_pid(session_id, None);
                    if self.is_cancelled(session_id) {
                        anyhow::bail!("cancelled");
                    }
                }
            }

            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if !output.status.success() {
                let detail = if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status: {}", output.status)
                };
                anyhow::bail!("Claude CLI failed: {detail}");
            }

            if stdout.is_empty() {
                anyhow::bail!("Claude CLI returned empty output.");
            }

            return Ok(stdout);
        }

        let codex_bin = runtime_paths::resolve_cli_bin("ANICA_CODEX_CLI_BIN", "codex")
            .unwrap_or_else(|| PathBuf::from("codex"))
            .to_string_lossy()
            .to_string();

        if !command_exists(&codex_bin) {
            anyhow::bail!(
                "Codex CLI not found: `{}`. Login status only confirms ~/.codex/auth.json. Install Codex CLI or set ANICA_CODEX_CLI_BIN to a valid path.",
                codex_bin
            );
        }

        let mut cmd = TokioCommand::new(&codex_bin);
        cmd.arg("exec");
        if let Some(effort) = resolve_reasoning_effort() {
            cmd.arg("-c")
                .arg(format!("model_reasoning_effort={effort}"));
        }
        cmd.arg(prompt)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let child = cmd.spawn().map_err(|err| {
            anyhow::anyhow!("Failed to spawn `{}`. Is it installed? ({err})", codex_bin)
        })?;
        self.set_running_pid(session_id, child.id());

        let output = child
            .wait_with_output()
            .await
            .map_err(|err| anyhow::anyhow!("Failed waiting for codex process: {err}"))?;
        self.set_running_pid(session_id, None);

        if self.is_cancelled(session_id) {
            anyhow::bail!("cancelled");
        }

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if !output.status.success() {
            let detail = if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status: {}", output.status)
            };
            anyhow::bail!("Codex CLI failed: {detail}");
        }

        if stdout.is_empty() {
            anyhow::bail!("Codex CLI returned empty output.");
        }

        Ok(stdout)
    }

    fn extract_json_blob(raw: &str) -> Option<&str> {
        let start = raw.find('{')?;
        let end = raw.rfind('}')?;
        if end <= start {
            return None;
        }
        Some(&raw[start..=end])
    }

    fn parse_tool_planner_output(raw: &str) -> Option<ToolPlannerOutput> {
        let json_blob = Self::extract_json_blob(raw)?;
        serde_json::from_str::<ToolPlannerOutput>(json_blob).ok()
    }

    async fn call_ext_tool(&self, method: &str, params_json: &str) -> anyhow::Result<String> {
        let conn = self.connection().map_err(|err| {
            anyhow::anyhow!("failed to access ACP connection for tool call: {err}")
        })?;

        let params_raw = RawValue::from_string(params_json.to_string())
            .map_err(|err| anyhow::anyhow!("failed to encode tool params: {err}"))?;
        let params_arc: Arc<RawValue> = params_raw.into();

        let response = conn
            .ext_method(ExtRequest::new(method, params_arc))
            .await
            .map_err(|err| anyhow::anyhow!("tool ext_method failed: {err}"))?;

        Ok(response.0.get().to_string())
    }

    fn router_max_turns() -> usize {
        std::env::var("ANICA_ACP_ROUTER_MAX_TURNS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0 && *v <= 12)
            .unwrap_or(8)
    }

    fn normalize_tool_name(name: &str) -> Option<&'static str> {
        match name.trim() {
            "anica.media_pool/list_metadata" | "list_media_metadata_for_ai" => {
                Some("anica.media_pool/list_metadata")
            }
            "anica.media_pool/remove_by_id"
            | "media_pool_remove_by_id"
            | "remove_media_pool_item" => Some("anica.media_pool/remove_by_id"),
            "anica.media_pool/clear_all" | "media_pool_clear_all" | "clear_media_pool" => {
                Some("anica.media_pool/clear_all")
            }
            "anica.llm/decision_making_srt_similar_serach"
            | "llm_decision_making_srt_similar_serach"
            | "decision_making_srt_similar_serach" => {
                Some("anica.llm/decision_making_srt_similar_serach")
            }
            "anica.timeline/get_snapshot" | "timeline_get_snapshot" => {
                Some("anica.timeline/get_snapshot")
            }
            "anica.timeline/build_autonomous_edit_plan"
            | "timeline_build_autonomous_edit_plan"
            | "build_autonomous_edit_plan" => Some("anica.timeline/build_autonomous_edit_plan"),
            "anica.timeline/get_audio_silence_map" | "timeline_get_audio_silence_map" => {
                Some("anica.timeline/get_audio_silence_map")
            }
            "anica.timeline/build_audio_silence_cut_plan"
            | "timeline_build_audio_silence_cut_plan" => {
                Some("anica.timeline/build_audio_silence_cut_plan")
            }
            "anica.timeline/get_transcript_low_confidence_map"
            | "timeline_get_transcript_low_confidence_map" => {
                Some("anica.timeline/get_transcript_low_confidence_map")
            }
            "anica.timeline/build_transcript_low_confidence_cut_plan"
            | "timeline_build_transcript_low_confidence_cut_plan" => {
                Some("anica.timeline/build_transcript_low_confidence_cut_plan")
            }
            "anica.timeline/get_subtitle_gap_map" | "timeline_get_subtitle_gap_map" => {
                Some("anica.timeline/get_subtitle_gap_map")
            }
            "anica.timeline/build_subtitle_gap_cut_plan"
            | "timeline_build_subtitle_gap_cut_plan" => {
                Some("anica.timeline/build_subtitle_gap_cut_plan")
            }
            "anica.timeline/translate_subtitles_to_new_track"
            | "timeline_translate_subtitles_to_new_track"
            | "translate_subtitles_to_new_track" => {
                Some("anica.timeline/translate_subtitles_to_new_track")
            }
            "anica.timeline/validate_edit_plan" | "timeline_validate_edit_plan" => {
                Some("anica.timeline/validate_edit_plan")
            }
            "anica.timeline/apply_edit_plan" | "timeline_apply_edit_plan" => {
                Some("anica.timeline/apply_edit_plan")
            }
            "anica.docs/list_files" | "docs_list_files" => Some("anica.docs/list_files"),
            "anica.docs/read_file" | "docs_read_file" => Some("anica.docs/read_file"),
            "anica.export/run" | "export_run" | "run_export" => Some("anica.export/run"),
            _ => None,
        }
    }

    fn normalize_docs_read_path(path: &str) -> String {
        let normalized = path.trim().trim_start_matches("./");
        match normalized {
            "export/export-choices.md" | "acp/export/export-choices.md" | "export/EXP-0001.md" => {
                "acp/export/EXP-0001.md".to_string()
            }
            "export/index.md" | "acp/export/index.md" => "acp/export/catalog.md".to_string(),
            "limitation.md" | "acp/limitation.md" => "acp/limitation/LIM-0001.md".to_string(),
            "limitation/index.md" | "acp/limitation/index.md" => {
                "acp/limitation/catalog.md".to_string()
            }
            other => other.to_string(),
        }
    }

    fn build_tool_synthesis_prompt(user_prompt: &str, tool_results: &[String]) -> String {
        let mut tool_block = String::new();
        if tool_results.is_empty() {
            tool_block.push_str("(none)");
        } else {
            for (idx, result) in tool_results.iter().enumerate() {
                tool_block.push_str(&format!(
                    "<tool_result index=\"{}\">\n{}\n</tool_result>\n",
                    idx + 1,
                    result
                ));
            }
        }

        format!(
            "You are the final answer synthesizer for Anica ACP.\n\
Task:\n\
- Answer the user directly from the tool results.\n\
\n\
Rules:\n\
- Respond in the same language as the user.\n\
- Output Markdown only.\n\
- Never output JSON.\n\
- If docs are missing/unavailable (e.g. \"file not found\" or \"docs subdir does not exist\"), begin with a short apology, then provide a best-effort inferred answer from available context.\n\
- If uncertain, explicitly mark assumptions.\n\
\n\
<user_prompt>\n\
{user_prompt}\n\
</user_prompt>\n\
\n\
<tool_results>\n\
{tool_block}\
</tool_results>\n"
        )
    }

    async fn synthesize_final_from_tool_results(
        &self,
        session_id: &SessionId,
        cwd: &Path,
        user_prompt: &str,
        tool_results: &[String],
    ) -> anyhow::Result<String> {
        if tool_results.is_empty() {
            anyhow::bail!("no tool results to synthesize");
        }

        let prompt = Self::build_tool_synthesis_prompt(user_prompt, tool_results);
        let raw = self.run_codex_prompt(session_id, cwd, &prompt).await?;

        if let Some(parsed) = Self::parse_tool_planner_output(&raw) {
            if let Some(final_text) = parsed.r#final {
                return Ok(final_text);
            }
            if Self::planner_tool_request(&parsed).is_some() {
                anyhow::bail!("synthesizer returned another tool decision");
            }
        }

        Ok(raw)
    }

    fn planner_tool_request(parsed: &ToolPlannerOutput) -> Option<(String, Value)> {
        if let Some(call) = parsed.tool_call.as_ref() {
            return Some((call.name.clone(), call.arguments.clone()));
        }

        if let Some(use_tool) = parsed.use_tool.as_ref() {
            return Some((
                use_tool.clone(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("query_media_pool"))
            .unwrap_or(false)
        {
            return Some((
                "anica.media_pool/list_metadata".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| {
                v.eq_ignore_ascii_case("remove_media_pool_item")
                    || v.eq_ignore_ascii_case("delete_media_pool_item")
            })
            .unwrap_or(false)
        {
            return Some((
                "anica.media_pool/remove_by_id".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| {
                v.eq_ignore_ascii_case("clear_media_pool")
                    || v.eq_ignore_ascii_case("delete_all_media_pool_items")
            })
            .unwrap_or(false)
        {
            return Some((
                "anica.media_pool/clear_all".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("query_timeline"))
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/get_snapshot".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| {
                v.eq_ignore_ascii_case("build_autonomous_edit_plan")
                    || v.eq_ignore_ascii_case("query_autonomous_edit_plan")
                    || v.eq_ignore_ascii_case("plan_edit_autonomous")
            })
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/build_autonomous_edit_plan".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("query_silence"))
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/get_audio_silence_map".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| {
                v.eq_ignore_ascii_case("build_audio_silence_cut_plan")
                    || v.eq_ignore_ascii_case("build_silence_cut_plan")
            })
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/build_audio_silence_cut_plan".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| {
                v.eq_ignore_ascii_case("query_transcript_low_confidence")
                    || v.eq_ignore_ascii_case("query_low_confidence_speech")
            })
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/get_transcript_low_confidence_map".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| {
                v.eq_ignore_ascii_case("build_transcript_low_confidence_cut_plan")
                    || v.eq_ignore_ascii_case("build_low_confidence_speech_cut_plan")
            })
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/build_transcript_low_confidence_cut_plan".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("query_subtitle_gaps"))
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/get_subtitle_gap_map".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("build_subtitle_gap_cut_plan"))
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/build_subtitle_gap_cut_plan".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| {
                v.eq_ignore_ascii_case("translate_subtitles")
                    || v.eq_ignore_ascii_case("translate_timeline_subtitles")
                    || v.eq_ignore_ascii_case("translate_subtitles_to_new_track")
            })
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/translate_subtitles_to_new_track".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("validate_plan"))
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/validate_edit_plan".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("apply_plan"))
            .unwrap_or(false)
        {
            return Some((
                "anica.timeline/apply_edit_plan".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| {
                v.eq_ignore_ascii_case("query_docs")
                    || v.eq_ignore_ascii_case("list_docs")
                    || v.eq_ignore_ascii_case("read_docs")
            })
            .unwrap_or(false)
        {
            let args = parsed.arguments.clone().unwrap_or(Value::Null);
            let method = if args.get("path").and_then(|v| v.as_str()).is_some() {
                "anica.docs/read_file"
            } else {
                "anica.docs/list_files"
            };
            return Some((method.to_string(), args));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("query_export_choices"))
            .unwrap_or(false)
        {
            let args = parsed.arguments.clone().unwrap_or_else(|| {
                json!({
                    "path": "acp/export/catalog.md",
                    "max_chars": 60000
                })
            });
            return Some(("anica.docs/read_file".to_string(), args));
        }

        if parsed
            .intent
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("run_export"))
            .unwrap_or(false)
        {
            return Some((
                "anica.export/run".to_string(),
                parsed.arguments.clone().unwrap_or(Value::Null),
            ));
        }

        None
    }

    fn emit_status(&self, key: &str, vars: &[(&str, String)]) {
        let line = self.inner.resources.tr_args(key, vars);
        eprintln!("ACP_STATUS: {line}");
        eprintln!("[ACP SYSTEM] {line}");
    }

    fn truncate_status_text(input: &str, max_chars: usize) -> String {
        if max_chars == 0 {
            return String::new();
        }
        let mut out = String::new();
        for (count, ch) in input.chars().enumerate() {
            if count >= max_chars {
                break;
            }
            out.push(ch);
        }
        if input.chars().count() > max_chars {
            out.push('…');
        }
        out
    }

    fn compact_json_for_status(value: &Value, max_chars: usize) -> String {
        let compact = serde_json::to_string(value).unwrap_or_else(|_| value.to_string());
        Self::truncate_status_text(&compact, max_chars)
    }

    fn summarize_tool_result(tool_name: &str, raw: &str) -> String {
        let Ok(value) = serde_json::from_str::<Value>(raw) else {
            return format!(
                "{tool_name}: non-json result ({} chars)",
                raw.chars().count()
            );
        };

        if let Some(obj) = value.as_object() {
            let mut parts: Vec<String> = Vec::new();

            if let Some(ok) = obj.get("ok").and_then(|v| v.as_bool()) {
                parts.push(format!("ok={ok}"));
            }
            if let Some(started) = obj.get("started").and_then(|v| v.as_bool()) {
                parts.push(format!("started={started}"));
            }
            if let Some(mode) = obj.get("mode").and_then(|v| v.as_str()) {
                parts.push(format!("mode={mode}"));
            }
            if let Some(preset) = obj.get("preset").and_then(|v| v.as_str()) {
                parts.push(format!("preset={preset}"));
            }
            if let Some(out_path) = obj.get("out_path").and_then(|v| v.as_str()) {
                parts.push(format!("out_path={out_path}"));
            }
            if let Some(export_resolution) = obj.get("export_resolution").and_then(|v| v.as_str()) {
                parts.push(format!("export_resolution={export_resolution}"));
            }
            if let Some(layout_resolution) = obj.get("layout_resolution").and_then(|v| v.as_str()) {
                parts.push(format!("layout_resolution={layout_resolution}"));
            }
            if let Some(resolution_source) = obj.get("resolution_source").and_then(|v| v.as_str()) {
                parts.push(format!("resolution_source={resolution_source}"));
            }
            if let Some(rev) = obj.get("timeline_revision").and_then(|v| v.as_str()) {
                parts.push(format!("revision={rev}"));
            }
            if let Some(rev) = obj.get("before_revision").and_then(|v| v.as_str()) {
                parts.push(format!("before={rev}"));
            }
            if let Some(rev) = obj.get("after_revision").and_then(|v| v.as_str()) {
                parts.push(format!("after={rev}"));
            }
            if let Some(applied) = obj.get("applied_ops").and_then(|v| v.as_u64()) {
                parts.push(format!("applied_ops={applied}"));
            }
            if let Some(errors) = obj.get("errors").and_then(|v| v.as_array()) {
                parts.push(format!("errors={}", errors.len()));
            }
            if let Some(warnings) = obj.get("warnings").and_then(|v| v.as_array()) {
                parts.push(format!("warnings={}", warnings.len()));
            }
            if let Some(files) = obj.get("files").and_then(|v| v.as_array()) {
                parts.push(format!("files={}", files.len()));
            }
            if let Some(total_files) = obj.get("total_files").and_then(|v| v.as_u64()) {
                parts.push(format!("total_files={total_files}"));
            }
            if let Some(items) = obj.get("items").and_then(|v| v.as_array()) {
                parts.push(format!("items={}", items.len()));
            }
            if let Some(error) = obj.get("error").and_then(|v| v.as_str()) {
                parts.push(format!("error={error}"));
            }

            if parts.is_empty() {
                parts.push(format!("keys={}", obj.len()));
            }

            return Self::truncate_status_text(&format!("{tool_name}: {}", parts.join(", ")), 260);
        }

        if let Some(arr) = value.as_array() {
            return format!("{tool_name}: array_len={}", arr.len());
        }

        Self::truncate_status_text(&format!("{tool_name}: {value}"), 260)
    }

    async fn run_codex_prompt_with_tool_bridge(
        &self,
        session_id: &SessionId,
        cwd: &Path,
        user_prompt: &str,
    ) -> anyhow::Result<String> {
        let mut tool_results: Vec<String> = Vec::new();
        let mut tool_trace: Vec<(String, String)> = Vec::new();
        let mut last_raw = String::new();
        let mut export_call_index = 0usize;
        let max_turns = Self::router_max_turns();
        let direct_cut_kind = Self::detect_direct_cut_kind(user_prompt);
        let low_confidence_cut_intent = Self::is_low_confidence_cut_intent(user_prompt);
        let llm_similarity_cut_intent = self.is_llm_similarity_cut_intent(user_prompt);
        self.update_export_resolution_preference_from_prompt(session_id, user_prompt);

        if llm_similarity_cut_intent {
            self.emit_status("acp.status.llm_similarity_only_mode", &[]);
            return self
                .run_llm_similarity_only_flow(session_id, cwd, user_prompt)
                .await;
        }

        for turn in 0..max_turns {
            self.emit_status(
                "acp.status.router_turn",
                &[
                    ("turn", (turn + 1).to_string()),
                    ("max_turns", max_turns.to_string()),
                ],
            );
            let routing_prompt = self
                .inner
                .resources
                .render_tool_router_prompt(user_prompt, &tool_results);
            let raw = self
                .run_codex_prompt(session_id, cwd, &routing_prompt)
                .await?;
            last_raw = raw.clone();

            let Some(parsed) = Self::parse_tool_planner_output(&raw) else {
                self.emit_status("acp.status.model_non_json", &[]);
                return Ok(raw);
            };

            if let Some(final_text) = parsed.r#final {
                if let Some(kind) = direct_cut_kind
                    && !Self::trace_has_successful_apply(&tool_trace)
                    && let Some((forced_tool, forced_args)) =
                        self.forced_direct_cut_next_tool(session_id, kind, &tool_trace)
                {
                    self.emit_status(
                        "acp.status.defer_final_until_apply",
                        &[("tool", forced_tool.clone())],
                    );
                    let tool_json = self
                        .execute_router_tool(session_id, &forced_tool, forced_args)
                        .await?;
                    tool_trace.push((forced_tool, tool_json.clone()));
                    tool_results.push(tool_json);
                    continue;
                }
                if self
                    .maybe_run_hybrid_second_check(
                        session_id,
                        cwd,
                        user_prompt,
                        &mut tool_trace,
                        &mut tool_results,
                    )
                    .await?
                {
                    self.emit_status("acp.status.defer_final_until_apply", &[]);
                    continue;
                }
                return Ok(final_text);
            }

            if let Some(confidence) = parsed.confidence {
                self.emit_status(
                    "acp.status.router_confidence",
                    &[("confidence", format!("{confidence:.2}"))],
                );
            }
            if let Some(reason) = parsed.reason.as_deref().map(str::trim)
                && !reason.is_empty()
            {
                self.emit_status(
                    "acp.status.router_reason",
                    &[("reason", Self::truncate_status_text(reason, 260))],
                );
            }
            if let Some(next_step) = parsed.next_step.as_deref().map(str::trim)
                && !next_step.is_empty()
            {
                self.emit_status(
                    "acp.status.router_next_step",
                    &[("next_step", Self::truncate_status_text(next_step, 260))],
                );
            }

            let Some((requested_tool, arguments)) = Self::planner_tool_request(&parsed) else {
                if let Some(kind) = direct_cut_kind
                    && !Self::trace_has_successful_apply(&tool_trace)
                    && let Some((forced_tool, forced_args)) =
                        self.forced_direct_cut_next_tool(session_id, kind, &tool_trace)
                {
                    self.emit_status(
                        "acp.status.force_direct_cut_flow",
                        &[("tool", forced_tool.clone())],
                    );
                    let tool_json = self
                        .execute_router_tool(session_id, &forced_tool, forced_args)
                        .await?;
                    tool_trace.push((forced_tool, tool_json.clone()));
                    tool_results.push(tool_json);
                    continue;
                }
                if self
                    .maybe_run_hybrid_second_check(
                        session_id,
                        cwd,
                        user_prompt,
                        &mut tool_trace,
                        &mut tool_results,
                    )
                    .await?
                {
                    self.emit_status("acp.status.force_direct_cut_flow", &[]);
                    continue;
                }

                if !tool_results.is_empty()
                    && let Ok(synthesized) = self
                        .synthesize_final_from_tool_results(
                            session_id,
                            cwd,
                            user_prompt,
                            &tool_results,
                        )
                        .await
                {
                    return Ok(synthesized);
                }
                return Ok(raw);
            };

            self.emit_status(
                "acp.status.model_requested_tool",
                &[("tool", requested_tool.clone())],
            );

            let requested_tool_lower = requested_tool.trim().to_ascii_lowercase();
            let legacy_semantic_tool_requested = requested_tool_lower
                == "anica.timeline/get_subtitle_semantic_repeats"
                || requested_tool_lower == "timeline_get_subtitle_semantic_repeats";
            let legacy_semantic_intent_requested = parsed
                .intent
                .as_deref()
                .map(|v| v.eq_ignore_ascii_case("query_subtitle_semantic_repeats"))
                .unwrap_or(false);
            if legacy_semantic_tool_requested || legacy_semantic_intent_requested {
                self.emit_status(
                    "acp.status.remap_legacy_semantic_to_llm_only",
                    &[("tool", requested_tool.clone())],
                );
                return self
                    .run_llm_similarity_only_flow(session_id, cwd, user_prompt)
                    .await;
            }

            let Some(normalized_tool_name) = Self::normalize_tool_name(&requested_tool) else {
                tool_results.push(
                    json!({
                        "error": "unsupported_tool",
                        "tool": requested_tool
                    })
                    .to_string(),
                );
                continue;
            };

            let include_missing_files = arguments
                .get("include_missing_files")
                .and_then(|v| v.as_bool());
            let mut tool_name = normalized_tool_name.to_string();
            let mut tool_args = if tool_name == "anica.media_pool/list_metadata" {
                json!({
                    "include_missing_files": include_missing_files.unwrap_or(true),
                    "include_file_stats": true,
                    "include_media_probe": true
                })
            } else if tool_name == "anica.docs/read_file" {
                if let Some(path) = arguments.get("path").and_then(|v| v.as_str()) {
                    let max_chars = arguments
                        .get("max_chars")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(60_000);
                    json!({
                        "path": Self::normalize_docs_read_path(path),
                        "max_chars": max_chars,
                    })
                } else {
                    json!({
                        "path": "acp/index.md",
                        "max_chars": 30000
                    })
                }
            } else if arguments.is_null() {
                json!({})
            } else {
                arguments
            };

            if direct_cut_kind == Some(DirectCutKind::SubtitleGap)
                && tool_name == "anica.timeline/get_subtitle_gap_map"
            {
                tool_name = "anica.timeline/build_subtitle_gap_cut_plan".to_string();
                tool_args = Self::promote_subtitle_gap_query_to_build_args(&tool_args);
                self.emit_status(
                    "acp.status.upgrade_subtitle_gap_query_to_build",
                    &[("tool", tool_name.clone())],
                );
            }

            if low_confidence_cut_intent
                && (tool_name == "anica.timeline/get_audio_silence_map"
                    || tool_name == "anica.timeline/build_audio_silence_cut_plan")
            {
                let mut promoted = match tool_args.as_object() {
                    Some(obj) => obj.clone(),
                    None => Map::new(),
                };
                promoted
                    .entry("enable_semantic_fallback".to_string())
                    .or_insert(Value::Bool(true));
                tool_name = "anica.timeline/build_transcript_low_confidence_cut_plan".to_string();
                tool_args = Value::Object(promoted);
                self.emit_status(
                    "acp.status.upgrade_low_confidence_to_transcript_tool",
                    &[("tool", tool_name.clone())],
                );
            }

            if tool_name == "anica.timeline/translate_subtitles_to_new_track" {
                let tool_json = self
                    .execute_subtitle_translation_tool(session_id, cwd, user_prompt, &tool_args)
                    .await?;
                tool_trace.push((tool_name, tool_json.clone()));
                tool_results.push(tool_json.clone());

                if let Some(question) = Self::parse_json_value(&tool_json).and_then(|value| {
                    if value
                        .get("needs_target_language")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        value
                            .get("question")
                            .and_then(|v| v.as_str())
                            .map(ToString::to_string)
                    } else {
                        None
                    }
                }) {
                    return Ok(question);
                }
                continue;
            }

            if tool_name == "anica.export/run" {
                tool_args =
                    Self::normalize_export_tool_args(user_prompt, tool_args, export_call_index);
                tool_args = self.resolve_export_tool_args(session_id, tool_args);
                export_call_index = export_call_index.saturating_add(1);
            }
            tool_args = Self::normalize_timeline_edit_tool_args(&tool_name, tool_args);
            let tool_json = self
                .execute_router_tool(session_id, &tool_name, tool_args)
                .await?;
            tool_trace.push((tool_name, tool_json.clone()));
            tool_results.push(tool_json);

            if self
                .maybe_run_hybrid_second_check(
                    session_id,
                    cwd,
                    user_prompt,
                    &mut tool_trace,
                    &mut tool_results,
                )
                .await?
            {
                continue;
            }
        }

        self.emit_status(
            "acp.status.router_max_turns",
            &[("max_turns", max_turns.to_string())],
        );
        if direct_cut_kind.is_some() && !Self::trace_has_successful_apply(&tool_trace) {
            self.emit_status("acp.status.direct_cut_incomplete", &[]);
        }
        if !tool_results.is_empty()
            && let Ok(synthesized) = self
                .synthesize_final_from_tool_results(session_id, cwd, user_prompt, &tool_results)
                .await
        {
            return Ok(synthesized);
        }
        Ok(last_raw)
    }
}

fn collect_prompt_text(blocks: &[ContentBlock]) -> String {
    let mut lines = Vec::new();
    let mut image_count = 0usize;
    let mut audio_count = 0usize;
    let mut resource_count = 0usize;

    for block in blocks {
        match block {
            ContentBlock::Text(t) => {
                let txt = t.text.trim();
                if !txt.is_empty() {
                    lines.push(txt.to_string());
                }
            }
            ContentBlock::Image(_) => image_count += 1,
            ContentBlock::Audio(_) => audio_count += 1,
            ContentBlock::Resource(_) | ContentBlock::ResourceLink(_) => resource_count += 1,
            _ => {}
        }
    }

    if image_count > 0 || audio_count > 0 || resource_count > 0 {
        lines.push(format!(
            "[non-text context] images={image_count}, audio={audio_count}, resources={resource_count}"
        ));
    }

    lines.join("\n")
}

fn chunk_text(input: &str, max_chars: usize) -> Vec<String> {
    if input.trim().is_empty() {
        return vec!["(empty response)".to_string()];
    }

    if max_chars == 0 {
        return vec![input.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0usize;

    // Preserve original Markdown/layout exactly; only split transport chunks by character count.
    for ch in input.chars() {
        current.push(ch);
        current_chars += 1;
        if current_chars >= max_chars {
            chunks.push(std::mem::take(&mut current));
            current_chars = 0;
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

fn command_exists(bin: &str) -> bool {
    if bin.is_empty() {
        return false;
    }

    if bin.contains('/') || bin.contains('\\') {
        return Path::new(bin).is_file();
    }

    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };

    std::env::split_paths(&path_var).any(|dir| {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            return true;
        }

        if cfg!(windows) {
            for ext in [".exe", ".bat", ".cmd"] {
                if dir.join(format!("{bin}{ext}")).is_file() {
                    return true;
                }
            }
        }

        false
    })
}

fn resolve_reasoning_effort() -> Option<&'static str> {
    let raw = std::env::var("ANICA_CODEX_REASONING_EFFORT").ok()?;
    let normalized = raw.trim().to_lowercase();
    match normalized.as_str() {
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" | "extra-high" | "extra_high" | "extrahigh" => Some("xhigh"),
        _ => None,
    }
}

#[async_trait::async_trait(?Send)]
impl Agent for AnicaAcpAgent {
    async fn initialize(
        &self,
        args: InitializeRequest,
    ) -> agent_client_protocol::Result<InitializeResponse> {
        Ok(InitializeResponse::new(args.protocol_version).agent_info(
            Implementation::new("anica-acp", env!("CARGO_PKG_VERSION")).title("Anica ACP"),
        ))
    }

    async fn authenticate(
        &self,
        _args: AuthenticateRequest,
    ) -> agent_client_protocol::Result<AuthenticateResponse> {
        Ok(AuthenticateResponse::default())
    }

    async fn new_session(
        &self,
        args: NewSessionRequest,
    ) -> agent_client_protocol::Result<NewSessionResponse> {
        let session_id = self.alloc_session_id();
        self.inner.sessions.borrow_mut().insert(
            session_id.clone(),
            SessionState {
                cwd: args.cwd,
                cancelled: false,
                running_pid: None,
                last_audio_silence_args: None,
                last_validated_operations: None,
                last_export_resolution_preference: None,
            },
        );
        Ok(NewSessionResponse::new(session_id))
    }

    async fn prompt(&self, args: PromptRequest) -> agent_client_protocol::Result<PromptResponse> {
        if !self.session_exists(&args.session_id) {
            return Err(
                Error::invalid_params().data(format!("unknown session id: {}", args.session_id.0))
            );
        }
        let _ = self.set_cancelled(&args.session_id, false);

        let user_text = collect_prompt_text(&args.prompt);
        let reply = if user_text.trim().is_empty() {
            self.inner.resources.tr("acp.empty_prompt")
        } else {
            let cwd = self
                .session_cwd(&args.session_id)
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            match self
                .run_codex_prompt_with_tool_bridge(&args.session_id, &cwd, &user_text)
                .await
            {
                Ok(text) => text,
                Err(err) => self
                    .inner
                    .resources
                    .tr_args("acp.codex_error", &[("error", err.to_string())]),
            }
        };

        for chunk in chunk_text(&reply, 72) {
            if self.is_cancelled(&args.session_id) {
                let _ = self
                    .send_agent_chunk(&args.session_id, self.inner.resources.tr("acp.cancelled"))
                    .await;
                return Ok(PromptResponse::new(StopReason::Cancelled));
            }

            self.send_agent_chunk(&args.session_id, chunk).await?;
            sleep(Duration::from_millis(24)).await;
        }

        Ok(PromptResponse::new(StopReason::EndTurn))
    }

    async fn cancel(&self, args: CancelNotification) -> agent_client_protocol::Result<()> {
        let _ = self.set_cancelled(&args.session_id, true);
        if let Some(pid) = self.running_pid(&args.session_id) {
            #[cfg(unix)]
            {
                let _ = std::process::Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .status();
            }
        }
        Ok(())
    }
}

async fn async_main() -> anyhow::Result<()> {
    let agent = AnicaAcpAgent::default();
    let (conn, io_task) = AgentSideConnection::new(
        agent.clone(),
        tokio::io::stdout().compat_write(),
        tokio::io::stdin().compat(),
        |fut| {
            tokio::task::spawn_local(fut);
        },
    );

    agent.bind_connection(Rc::new(conn));

    io_task
        .await
        .map_err(|err| anyhow::anyhow!("anica-acp io task failed: {err}"))?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| anyhow::anyhow!("failed to build runtime: {err}"))?;
    let local = LocalSet::new();
    runtime.block_on(local.run_until(async_main()))
}

#[cfg(test)]
mod tests {
    use super::AnicaAcpAgent;
    use serde_json::json;

    #[test]
    fn normalize_export_tool_args_maps_single_orientation_prompt() {
        let args = json!({});
        let normalized =
            AnicaAcpAgent::normalize_export_tool_args("please export portrait video", args, 0);
        assert_eq!(
            normalized
                .get("target_resolution")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "1080x1920"
        );
    }

    #[test]
    fn normalize_export_tool_args_alternates_for_dual_orientation_batch_prompt() {
        let prompt = "please batch export portrait and landscape versions";
        let first = AnicaAcpAgent::normalize_export_tool_args(prompt, json!({}), 0);
        let second = AnicaAcpAgent::normalize_export_tool_args(prompt, json!({}), 1);
        assert_eq!(
            first
                .get("target_resolution")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "1080x1920"
        );
        assert_eq!(
            second
                .get("target_resolution")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "1920x1080"
        );
    }

    #[test]
    fn normalize_export_tool_args_maps_named_resolution_hint() {
        let normalized =
            AnicaAcpAgent::normalize_export_tool_args("please export DCI 4K version", json!({}), 0);
        assert_eq!(
            normalized
                .get("target_resolution")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "4096x2160"
        );
    }

    #[test]
    fn normalize_export_tool_args_maps_2k_portrait_hint() {
        let normalized =
            AnicaAcpAgent::normalize_export_tool_args("please export 2K portrait", json!({}), 0);
        assert_eq!(
            normalized
                .get("target_resolution")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "1440x2560"
        );
    }

    #[test]
    fn normalize_export_tool_args_maps_4k_portrait_hint() {
        let normalized =
            AnicaAcpAgent::normalize_export_tool_args("please export 4K portrait", json!({}), 0);
        assert_eq!(
            normalized
                .get("target_resolution")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "2160x3840"
        );
    }
}
