use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, atomic::AtomicBool, mpsc};
use std::time::Duration;

use gpui::{
    ClipboardItem, Context, Entity, MouseButton, Render, SharedString, Subscription, Timer, Window,
    div, prelude::*, px, rgb,
};
use gpui_component::{
    input::{Input, InputEvent, InputState},
    scroll::ScrollableElement,
    text::TextView,
    white,
};
use serde_json::Value;

use crate::api::export::{
    AcpExportRunRequest, AcpExportRunResponse, resolve_acp_export_run_request,
};
use crate::api::llm::llm_decision_making_srt_similar_serach;
use crate::api::media_pool::{clear_media_pool, remove_media_pool_by_id};
use crate::api::timeline::{
    apply_edit_plan, build_audio_silence_cut_plan, build_autonomous_edit_plan,
    build_subtitle_gap_cut_plan, build_transcript_low_confidence_cut_plan, get_audio_silence_map,
    get_subtitle_gap_map, get_subtitle_semantic_repeats, get_timeline_snapshot,
    get_transcript_low_confidence_map, validate_edit_plan,
};
use crate::api::transport_acp::{AcpToolBridgeRequest, AcpUiEvent, AcpWorker};
use crate::core::export::{FfmpegExporter, is_cancelled_export_error};
use crate::core::global_state::{
    AiChatMessage, AiChatRole, GlobalState, MediaPoolItem, MediaPoolUiEvent,
    SilencePreviewCandidate, SilencePreviewModalState,
};
use crate::core::user_settings::{
    SettingsScope, load_settings, resolve_workspace_root, save_auto_connect,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AcpAgentProvider {
    Codex,
    Gemini,
    Claude,
}

impl AcpAgentProvider {
    fn label(self) -> &'static str {
        match self {
            AcpAgentProvider::Codex => "Codex",
            AcpAgentProvider::Gemini => "Gemini",
            AcpAgentProvider::Claude => "Claude",
        }
    }

    fn default_command(self) -> String {
        match self {
            AcpAgentProvider::Codex => resolve_default_agent_command(),
            AcpAgentProvider::Gemini => resolve_default_agent_command(),
            AcpAgentProvider::Claude => resolve_default_agent_command(),
        }
    }

    fn infer_from_command(command: &str) -> Self {
        let lower = command.trim().to_ascii_lowercase();
        if lower.starts_with("claude ")
            || lower == "claude"
            || lower.contains("/claude ")
            || lower.contains("claude-agent-acp")
        {
            AcpAgentProvider::Claude
        } else if lower.starts_with("gemini ") || lower == "gemini" || lower.contains("/gemini ") {
            AcpAgentProvider::Gemini
        } else {
            AcpAgentProvider::Codex
        }
    }
}

const AI_CHAT_MAX_CONVERSATION_MESSAGES: usize = 1000;

fn is_counted_ai_chat_role(role: AiChatRole) -> bool {
    !matches!(role, AiChatRole::System)
}

fn rebase_ai_chat_index(
    idx: Option<usize>,
    removed_prefix: usize,
    remaining_len: usize,
) -> Option<usize> {
    idx.and_then(|old| old.checked_sub(removed_prefix))
        .filter(|new_idx| *new_idx < remaining_len)
}

fn prune_ai_chat_history(messages: &mut Vec<AiChatMessage>) -> usize {
    let mut counted = messages
        .iter()
        .filter(|msg| is_counted_ai_chat_role(msg.role))
        .count();
    if counted <= AI_CHAT_MAX_CONVERSATION_MESSAGES {
        return 0;
    }

    let mut remove_upto = 0usize;
    while counted > AI_CHAT_MAX_CONVERSATION_MESSAGES && remove_upto < messages.len() {
        while remove_upto < messages.len()
            && matches!(messages[remove_upto].role, AiChatRole::System)
        {
            remove_upto += 1;
        }
        if remove_upto >= messages.len() {
            break;
        }

        let first_role = messages[remove_upto].role;
        remove_upto += 1;
        counted = counted.saturating_sub(1);

        while remove_upto < messages.len()
            && matches!(messages[remove_upto].role, AiChatRole::System)
        {
            remove_upto += 1;
        }

        if matches!(first_role, AiChatRole::User)
            && remove_upto < messages.len()
            && matches!(messages[remove_upto].role, AiChatRole::Assistant)
        {
            remove_upto += 1;
            counted = counted.saturating_sub(1);

            while remove_upto < messages.len()
                && matches!(messages[remove_upto].role, AiChatRole::System)
            {
                remove_upto += 1;
            }
        }
    }

    if remove_upto > 0 {
        messages.drain(0..remove_upto);
    }
    remove_upto
}

#[derive(Clone, Debug)]
enum AgentLoginStatus {
    LoggedIn {
        detail: String,
        source: &'static str,
    },
    LoggedOut {
        reason: String,
        source: &'static str,
    },
    CliMissing {
        detail: String,
    },
    Error(String),
}

impl AgentLoginStatus {
    fn title(&self, provider: AcpAgentProvider) -> String {
        match self {
            AgentLoginStatus::LoggedIn { .. } => format!("{}: Ready", provider.label()),
            AgentLoginStatus::LoggedOut { .. } => format!("{}: Setup Needed", provider.label()),
            AgentLoginStatus::CliMissing { .. } => format!("{} CLI Not Found", provider.label()),
            AgentLoginStatus::Error(_) => format!("{} Status Check Error", provider.label()),
        }
    }

    fn color(&self) -> gpui::Hsla {
        match self {
            AgentLoginStatus::LoggedIn { .. } => rgb(0x22c55e).into(),
            AgentLoginStatus::LoggedOut { .. } => rgb(0xf59e0b).into(),
            AgentLoginStatus::CliMissing { .. } => rgb(0xef4444).into(),
            AgentLoginStatus::Error(_) => rgb(0xf97316).into(),
        }
    }

    fn detail_text(&self) -> String {
        match self {
            AgentLoginStatus::LoggedIn { detail, source } => {
                format!("{detail} (source: {source})")
            }
            AgentLoginStatus::LoggedOut { reason, source } => {
                format!("{reason} (source: {source})")
            }
            AgentLoginStatus::CliMissing { detail } => detail.clone(),
            AgentLoginStatus::Error(err) => err.clone(),
        }
    }

    fn action_hint(&self, provider: AcpAgentProvider) -> &'static str {
        match provider {
            AcpAgentProvider::Codex => match self {
                AgentLoginStatus::LoggedIn { .. } => "Ready for ACP chat.",
                AgentLoginStatus::LoggedOut { .. } => "Run: codex login",
                AgentLoginStatus::CliMissing { .. } => {
                    "Install Codex CLI (or codex-acp), then run: codex login"
                }
                AgentLoginStatus::Error(_) => {
                    "Check terminal logs and verify Codex CLI auth state."
                }
            },
            AcpAgentProvider::Gemini => match self {
                AgentLoginStatus::LoggedIn { .. } => {
                    "Ready for ACP chat. API key is optional; CLI login is supported."
                }
                AgentLoginStatus::LoggedOut { .. } => {
                    "Set GEMINI_API_KEY or login in Gemini CLI (`gemini`, then `/auth`)."
                }
                AgentLoginStatus::CliMissing { .. } => {
                    "Install Gemini CLI or point agent command to a Gemini ACP-capable command."
                }
                AgentLoginStatus::Error(_) => {
                    "Check Gemini CLI install path and auth state (`gemini`, then `/auth`)."
                }
            },
            AcpAgentProvider::Claude => match self {
                AgentLoginStatus::LoggedIn { .. } => {
                    "Ready for ACP chat. CLI login or ANTHROPIC_API_KEY is supported."
                }
                AgentLoginStatus::LoggedOut { .. } => {
                    "Run `claude auth login` (or set ANTHROPIC_API_KEY)."
                }
                AgentLoginStatus::CliMissing { .. } => {
                    "Install Claude Code CLI, then run `claude auth login`."
                }
                AgentLoginStatus::Error(_) => {
                    "Check Claude CLI install path and auth state (`claude auth status`)."
                }
            },
        }
    }
}

fn first_non_empty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToString::to_string)
}

fn cli_command_exists(bin: &str) -> bool {
    if bin.is_empty() {
        return false;
    }

    if bin.contains('/') || bin.contains('\\') {
        return PathBuf::from(bin).is_file();
    }

    let Some(path_var) = env::var_os("PATH") else {
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

fn codex_install_hint() -> String {
    "Codex CLI not found.\nInstall first, then restart Anica:\n1) npm i -g @openai/codex\n2) codex login\n3) codex --version\nOptional: set ANICA_CODEX_CLI_BIN=/absolute/path/to/codex if PATH is not visible to GUI app."
        .to_string()
}

fn gemini_install_hint() -> String {
    "Gemini CLI not found.\nInstall Gemini CLI (`npm install -g @google/gemini-cli`) so bundled `anica-acp` can route prompts through Gemini when Gemini provider is selected.\nYou can authenticate with GEMINI_API_KEY, or use CLI login (`gemini`, then `/auth`)."
        .to_string()
}

fn claude_install_hint() -> String {
    "Claude CLI not found.\nInstall Claude Code CLI, then login:\n1) curl -fsSL https://claude.ai/install.sh | bash\n2) claude auth login\n3) claude auth status\nAlternative npm path (legacy): npm install -g @anthropic-ai/claude-code"
        .to_string()
}

fn has_gemini_api_key(gemini_api_key: &str) -> bool {
    if !gemini_api_key.trim().is_empty() {
        return true;
    }

    env::var("GEMINI_API_KEY")
        .ok()
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
        || env::var("GOOGLE_API_KEY")
            .ok()
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
}

fn detect_gemini_cli_auth_from_files() -> Option<AgentLoginStatus> {
    let home = env::var_os("HOME")?;
    let gemini_dir = PathBuf::from(home).join(".gemini");
    let settings_path = gemini_dir.join("settings.json");
    let oauth_path = gemini_dir.join("oauth_creds.json");

    let settings_json = fs::read_to_string(&settings_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());

    let selected_auth_type = settings_json
        .as_ref()
        .and_then(|json| json.get("security"))
        .and_then(|v| v.get("auth"))
        .and_then(|v| v.get("selectedType"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string);

    let oauth_has_token = || -> bool {
        let Ok(raw) = fs::read_to_string(&oauth_path) else {
            return false;
        };
        let Ok(json) = serde_json::from_str::<Value>(&raw) else {
            return false;
        };

        let access = json
            .get("access_token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let refresh = json
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();

        !access.is_empty() || !refresh.is_empty()
    };

    if let Some(auth_type) = selected_auth_type {
        return Some(match auth_type.as_str() {
            "oauth-personal" => {
                if oauth_has_token() {
                    AgentLoginStatus::LoggedIn {
                        detail: "Gemini CLI OAuth credentials detected".to_string(),
                        source: "~/.gemini/oauth_creds.json",
                    }
                } else {
                    AgentLoginStatus::LoggedOut {
                        reason: "Gemini CLI auth is `oauth-personal`, but OAuth credentials are missing or invalid".to_string(),
                        source: "~/.gemini/oauth_creds.json",
                    }
                }
            }
            "compute-default-credentials" | "cloud-shell" => AgentLoginStatus::LoggedIn {
                detail: format!("Gemini CLI auth type is `{auth_type}`"),
                source: "~/.gemini/settings.json",
            },
            "vertex-ai" => {
                let has_google_api_key = env::var("GOOGLE_API_KEY")
                    .ok()
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false);
                let has_cloud_project = env::var("GOOGLE_CLOUD_PROJECT")
                    .ok()
                    .or_else(|| env::var("GOOGLE_CLOUD_PROJECT_ID").ok())
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false);
                let has_cloud_location = env::var("GOOGLE_CLOUD_LOCATION")
                    .ok()
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false);

                if has_google_api_key || (has_cloud_project && has_cloud_location) {
                    AgentLoginStatus::LoggedIn {
                        detail: "Gemini CLI Vertex auth config detected".to_string(),
                        source: "~/.gemini/settings.json",
                    }
                } else {
                    AgentLoginStatus::LoggedOut {
                        reason: "Gemini CLI auth is `vertex-ai`, but required GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION is missing".to_string(),
                        source: "~/.gemini/settings.json",
                    }
                }
            }
            "gemini-api-key" => AgentLoginStatus::LoggedOut {
                reason: "Gemini CLI auth is `gemini-api-key`, but no key is set in Anica/env"
                    .to_string(),
                source: "~/.gemini/settings.json",
            },
            _ => AgentLoginStatus::LoggedOut {
                reason: format!(
                    "Gemini CLI auth type `{auth_type}` is configured but could not be verified from Anica"
                ),
                source: "~/.gemini/settings.json",
            },
        });
    }

    if oauth_has_token() {
        return Some(AgentLoginStatus::LoggedIn {
            detail: "Gemini CLI OAuth credentials detected".to_string(),
            source: "~/.gemini/oauth_creds.json",
        });
    }

    None
}

fn detect_from_auth_file() -> Option<AgentLoginStatus> {
    let home = env::var_os("HOME")?;
    let auth_path = PathBuf::from(home).join(".codex").join("auth.json");

    if !auth_path.exists() {
        return Some(AgentLoginStatus::LoggedOut {
            reason: format!("Auth file not found: {}", auth_path.display()),
            source: "~/.codex/auth.json",
        });
    }

    let raw = match fs::read_to_string(&auth_path) {
        Ok(v) => v,
        Err(err) => {
            return Some(AgentLoginStatus::Error(format!(
                "Failed to read {}: {err}",
                auth_path.display()
            )));
        }
    };

    let json: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(err) => {
            return Some(AgentLoginStatus::Error(format!(
                "Failed to parse {}: {err}",
                auth_path.display()
            )));
        }
    };

    let token = json
        .get("tokens")
        .and_then(|v| v.get("access_token"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();

    let api_key = json
        .get("OPENAI_API_KEY")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();

    let auth_mode = json
        .get("auth_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    if !token.is_empty() || !api_key.is_empty() {
        let account_hint = json
            .get("tokens")
            .and_then(|v| v.get("account_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown-account");

        return Some(AgentLoginStatus::LoggedIn {
            detail: format!("Credentials detected (auth_mode={auth_mode}, account={account_hint})"),
            source: "~/.codex/auth.json",
        });
    }

    Some(AgentLoginStatus::LoggedOut {
        reason: "Auth file exists but no usable access token/API key found".to_string(),
        source: "~/.codex/auth.json",
    })
}

fn detect_codex_login_status() -> AgentLoginStatus {
    match Command::new("codex").args(["login", "status"]).output() {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let combined_lc = format!("{}\n{}", stdout, stderr).to_lowercase();

            if out.status.success() {
                if combined_lc.contains("not logged")
                    || combined_lc.contains("login required")
                    || combined_lc.contains("unauthorized")
                {
                    return AgentLoginStatus::LoggedOut {
                        reason: first_non_empty_line(&stdout)
                            .or_else(|| first_non_empty_line(&stderr))
                            .unwrap_or_else(|| "CLI reports not logged in".to_string()),
                        source: "codex login status",
                    };
                }

                return AgentLoginStatus::LoggedIn {
                    detail: first_non_empty_line(&stdout)
                        .or_else(|| first_non_empty_line(&stderr))
                        .unwrap_or_else(|| "codex login status succeeded".to_string()),
                    source: "codex login status",
                };
            }

            if combined_lc.contains("not logged")
                || combined_lc.contains("login required")
                || combined_lc.contains("unauthorized")
            {
                return AgentLoginStatus::LoggedOut {
                    reason: first_non_empty_line(&stderr)
                        .or_else(|| first_non_empty_line(&stdout))
                        .unwrap_or_else(|| "CLI reports not logged in".to_string()),
                    source: "codex login status",
                };
            }

            detect_from_auth_file().unwrap_or_else(|| {
                AgentLoginStatus::Error(format!(
                    "`codex login status` failed with code {:?}",
                    out.status.code()
                ))
            })
        }
        Err(err) if err.kind() == ErrorKind::NotFound => {
            if let Some(status) = detect_from_auth_file() {
                status
            } else {
                AgentLoginStatus::CliMissing {
                    detail: "Cannot find `codex` in PATH. Install Codex CLI or launch app from a shell where `codex` is available.".to_string(),
                }
            }
        }
        Err(err) => detect_from_auth_file().unwrap_or_else(|| {
            AgentLoginStatus::Error(format!("Failed to execute `codex login status`: {err}"))
        }),
    }
}

fn detect_gemini_login_status(gemini_api_key: &str) -> AgentLoginStatus {
    if !cli_command_exists("gemini") {
        return AgentLoginStatus::CliMissing {
            detail: "Cannot find `gemini` in PATH. Install Gemini CLI or set ACP Agent Command to a Gemini ACP-capable command.".to_string(),
        };
    }

    if has_gemini_api_key(gemini_api_key) {
        let source = if !gemini_api_key.trim().is_empty() {
            "AI Agents page"
        } else if env::var("GEMINI_API_KEY")
            .ok()
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
        {
            "GEMINI_API_KEY"
        } else {
            "GOOGLE_API_KEY"
        };

        return AgentLoginStatus::LoggedIn {
            detail: "Gemini API key present".to_string(),
            source,
        };
    }

    if let Some(status) = detect_gemini_cli_auth_from_files() {
        return status;
    }

    AgentLoginStatus::LoggedOut {
        reason: "Gemini CLI is available, but no API key and no CLI auth credentials were detected"
            .to_string(),
        source: "environment + ~/.gemini",
    }
}

fn detect_claude_login_status() -> AgentLoginStatus {
    if !cli_command_exists("claude") {
        return AgentLoginStatus::CliMissing {
            detail: "Cannot find `claude` in PATH. Install Claude Code CLI or set ACP Agent Command to a Claude-capable command.".to_string(),
        };
    }

    let parse_status_output = |out: std::process::Output, source: &'static str| {
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        let combined_lc = format!("{}\n{}", stdout, stderr).to_lowercase();
        if out.status.success() {
            return AgentLoginStatus::LoggedIn {
                detail: first_non_empty_line(&stdout)
                    .or_else(|| first_non_empty_line(&stderr))
                    .unwrap_or_else(|| "claude auth status succeeded".to_string()),
                source,
            };
        }

        if combined_lc.contains("not logged")
            || combined_lc.contains("login required")
            || combined_lc.contains("auth required")
            || combined_lc.contains("unauthorized")
        {
            return AgentLoginStatus::LoggedOut {
                reason: first_non_empty_line(&stderr)
                    .or_else(|| first_non_empty_line(&stdout))
                    .unwrap_or_else(|| "Claude CLI reports not logged in".to_string()),
                source,
            };
        }

        AgentLoginStatus::LoggedOut {
            reason: first_non_empty_line(&stderr)
                .or_else(|| first_non_empty_line(&stdout))
                .unwrap_or_else(|| format!("`{source}` failed with code {:?}", out.status.code())),
            source,
        }
    };

    match Command::new("claude")
        .args(["auth", "status", "--text"])
        .output()
    {
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).to_lowercase();
            let stdout = String::from_utf8_lossy(&out.stdout).to_lowercase();
            let unknown_text_flag = (!out.status.success())
                && (stderr.contains("unknown option '--text'")
                    || stderr.contains("unknown option: --text")
                    || stderr.contains("unrecognized option '--text'")
                    || stdout.contains("unknown option '--text'")
                    || stdout.contains("unknown option: --text")
                    || stdout.contains("unrecognized option '--text'"));
            if unknown_text_flag {
                return match Command::new("claude").args(["auth", "status"]).output() {
                    Ok(fallback) => parse_status_output(fallback, "claude auth status"),
                    Err(err) => AgentLoginStatus::Error(format!(
                        "Failed to execute `claude auth status`: {err}"
                    )),
                };
            }
            parse_status_output(out, "claude auth status --text")
        }
        Err(err) if err.kind() == ErrorKind::NotFound => AgentLoginStatus::CliMissing {
            detail: "Cannot find `claude` in PATH. Install Claude Code CLI or launch app from a shell where `claude` is available.".to_string(),
        },
        Err(err) => AgentLoginStatus::Error(format!(
            "Failed to execute `claude auth status --text`: {err}"
        )),
    }
}

fn bundled_acp_command() -> Option<String> {
    if let Ok(exe) = env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        // Dev/bundled sidecar next to main executable.
        for name in ["anica-acp", "codex-acp"] {
            let candidate = exe_dir.join(name);
            if candidate.exists() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
    }

    // macOS app bundle: .../Contents/MacOS/anica -> .../Contents/Resources/codex-acp
    if let Ok(exe) = env::current_exe()
        && let Some(macos_dir) = exe.parent()
        && let Some(contents_dir) = macos_dir.parent()
    {
        for name in ["anica-acp", "codex-acp"] {
            let candidate = contents_dir.join("Resources").join(name);
            if candidate.exists() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
    }

    // Dev fallback: local bundled helper in repo.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for rel in [
        "assets/bin/anica-acp",
        "assets/bin/codex-acp",
        "bin/anica-acp",
        "bin/codex-acp",
    ] {
        let p = manifest.join(rel);
        if p.exists() {
            return Some(p.to_string_lossy().to_string());
        }
    }

    None
}

fn split_first_token(command: &str) -> Option<(&str, &str)> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return None;
    }
    let idx = trimmed
        .find(|c: char| c.is_whitespace())
        .unwrap_or(trimmed.len());
    Some((&trimmed[..idx], &trimmed[idx..]))
}

fn sidecar_candidates_for_name(name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(exe) = env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        out.push(exe_dir.join(name));
        // macOS app bundle: .../Contents/MacOS/anica -> .../Contents/Resources/anica-acp
        if let Some(contents_dir) = exe_dir.parent()
            && contents_dir
                .file_name()
                .and_then(|v| v.to_str())
                .is_some_and(|v| v == "Contents")
        {
            out.push(contents_dir.join("Resources").join(name));
        }
    }
    out
}

fn sidecar_candidate_for_token(token: &str) -> Option<String> {
    let path = PathBuf::from(token);
    let sidecar_name = path.file_name().and_then(|v| v.to_str()).and_then(|name| {
        if name == "anica-acp" || name == "codex-acp" {
            Some(name)
        } else {
            None
        }
    });

    if path.is_absolute() {
        if let Some(name) = sidecar_name {
            for candidate in sidecar_candidates_for_name(name) {
                if candidate.is_file() {
                    return Some(candidate.to_string_lossy().to_string());
                }
            }
        }
        if path.is_file() {
            return Some(path.to_string_lossy().to_string());
        }
        return None;
    }

    if token.contains('/') || token.contains('\\') {
        if let Some(name) = sidecar_name {
            for candidate in sidecar_candidates_for_name(name) {
                if candidate.is_file() {
                    return Some(candidate.to_string_lossy().to_string());
                }
            }
        }
        if let Ok(exe) = env::current_exe()
            && let Some(exe_dir) = exe.parent()
        {
            let candidate = exe_dir.join(&path);
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
        return None;
    }

    if token == "anica-acp" || token == "codex-acp" {
        for candidate in sidecar_candidates_for_name(token) {
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
    }
    None
}

fn resolve_agent_command_for_spawn(command: &str) -> String {
    let trimmed = command.trim();
    if PathBuf::from(trimmed).is_file() {
        return shell_quote_token(trimmed);
    }
    let Some((token, rest)) = split_first_token(command) else {
        return trimmed.to_string();
    };
    if let Some(resolved_token) = sidecar_candidate_for_token(token) {
        format!("{}{}", shell_quote_token(&resolved_token), rest)
    } else {
        trimmed.to_string()
    }
}

fn shell_quote_token(token: &str) -> String {
    if cfg!(windows) {
        return token.to_string();
    }
    if !token.chars().any(char::is_whitespace) && !token.contains('\'') {
        return token.to_string();
    }
    format!("'{}'", token.replace('\'', "'\"'\"'"))
}

fn agent_command_env_override() -> Option<String> {
    env::var("ANICA_ACP_AGENT_CMD")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn resolve_default_agent_command() -> String {
    if let Some(cmd) = agent_command_env_override() {
        return cmd;
    }

    if bundled_acp_command().is_some() {
        // Keep UI command portable/user-friendly. Connect-time resolution maps this
        // short command to a sidecar binary next to the executable when available.
        return "anica-acp".to_string();
    }

    "anica-acp".to_string()
}

fn normalize_agent_command_for_ui(command: String) -> String {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return "anica-acp".to_string();
    }
    let path = PathBuf::from(trimmed);
    if path.is_absolute()
        && let Some(name) = path.file_name().and_then(|s| s.to_str())
        && (name == "anica-acp" || name == "codex-acp")
    {
        return name.to_string();
    }
    trimmed.to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CodexReasoningMode {
    Low,
    Medium,
    High,
    ExtraHigh,
}

impl CodexReasoningMode {
    fn label(self) -> &'static str {
        match self {
            CodexReasoningMode::Low => "Low",
            CodexReasoningMode::Medium => "Medium",
            CodexReasoningMode::High => "High",
            CodexReasoningMode::ExtraHigh => "Extra High",
        }
    }

    fn env_value(self) -> &'static str {
        match self {
            CodexReasoningMode::Low => "low",
            CodexReasoningMode::Medium => "medium",
            CodexReasoningMode::High => "high",
            CodexReasoningMode::ExtraHigh => "xhigh",
        }
    }

    fn from_setting_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "low" => Some(CodexReasoningMode::Low),
            "medium" => Some(CodexReasoningMode::Medium),
            "high" => Some(CodexReasoningMode::High),
            "xhigh" | "extra_high" | "extra-high" | "extra high" => {
                Some(CodexReasoningMode::ExtraHigh)
            }
            _ => None,
        }
    }
}

pub struct AiAgentsPage {
    pub global: Entity<GlobalState>,
    agent_provider: AcpAgentProvider,
    agent_status: AgentLoginStatus,

    worker: AcpWorker,
    connected: bool,
    // Tracks which provider is actually connected, so the UI can highlight it.
    connected_provider: Option<AcpAgentProvider>,
    busy: bool,
    codex_reasoning_mode: CodexReasoningMode,
    auto_connect_enabled: bool,
    auto_connect_scope: SettingsScope,

    agent_command: String,
    prompt_text: String,
    last_status: String,

    command_input: Option<Entity<InputState>>,
    command_input_sub: Option<Subscription>,

    gemini_api_key: String,
    gemini_key_input: Option<Entity<InputState>>,
    gemini_key_input_sub: Option<Subscription>,

    prompt_input: Option<Entity<InputState>>,
    prompt_input_sub: Option<Subscription>,

    active_assistant_idx: Option<usize>,
    media_pool_signature: u64,
    last_chat_status: String,
    prompt_send_on_next_render: bool,
    show_system_messages: bool,

    poll_running: bool,
    poll_token: u64,
}

impl AiAgentsPage {
    fn refresh_agent_status(&mut self) {
        self.agent_status = match self.agent_provider {
            AcpAgentProvider::Codex => detect_codex_login_status(),
            AcpAgentProvider::Gemini => detect_gemini_login_status(&self.gemini_api_key),
            AcpAgentProvider::Claude => detect_claude_login_status(),
        };
    }

    fn set_agent_command_value(
        &mut self,
        value: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.agent_command = value.clone();
        if let Some(input) = self.command_input.as_ref() {
            input.update(cx, |this, cx| {
                this.set_value(value.clone(), window, cx);
            });
        }
    }

    fn select_agent_provider(
        &mut self,
        provider: AcpAgentProvider,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.agent_provider == provider {
            return;
        }
        self.agent_provider = provider;
        self.set_agent_command_value(provider.default_command(), window, cx);
        self.refresh_agent_status();
        self.push_system_message(
            format!(
                "ACP provider switched to {}. Command reset to: {}",
                provider.label(),
                self.agent_command
            ),
            cx,
        );
    }

    pub fn new(global: Entity<GlobalState>, cx: &mut Context<Self>) -> Self {
        let workspace_root = resolve_workspace_root(global.read(cx).project_file_path.as_deref());
        let loaded_settings = load_settings(&workspace_root);

        let mut agent_command = resolve_default_agent_command();
        if agent_command_env_override().is_none()
            && let Some(saved_command) = loaded_settings.effective.acp_agent_command.clone()
        {
            agent_command = saved_command;
        }
        if agent_command_env_override().is_none() {
            agent_command = normalize_agent_command_for_ui(agent_command);
        }
        let agent_provider = AcpAgentProvider::infer_from_command(&agent_command);

        let reasoning_mode = loaded_settings
            .effective
            .acp_reasoning_mode
            .as_deref()
            .and_then(CodexReasoningMode::from_setting_value)
            .unwrap_or(CodexReasoningMode::Medium);
        let auto_connect_enabled = loaded_settings.effective.acp_auto_connect;
        let auto_connect_scope = loaded_settings.auto_connect_source.preferred_scope();
        let gemini_api_key = env::var("GEMINI_API_KEY")
            .ok()
            .or_else(|| env::var("GOOGLE_API_KEY").ok())
            .unwrap_or_default();

        cx.observe(&global, |this, _global, cx| {
            this.sync_worker_media_pool_snapshot(cx);
            cx.notify();
        })
        .detach();
        cx.subscribe(&global, |this, _global, evt: &MediaPoolUiEvent, cx| {
            if matches!(evt, MediaPoolUiEvent::StateChanged) {
                this.sync_worker_media_pool_snapshot(cx);
                cx.notify();
            }
        })
        .detach();

        let mut this = Self {
            global,
            agent_provider,
            agent_status: AgentLoginStatus::Error("Status not initialized".to_string()),

            worker: AcpWorker::spawn(),
            connected: false,
            connected_provider: None,
            busy: false,
            codex_reasoning_mode: reasoning_mode,
            auto_connect_enabled,
            auto_connect_scope,

            agent_command,
            prompt_text: String::new(),
            last_status: "Idle".to_string(),

            command_input: None,
            command_input_sub: None,

            gemini_api_key,
            gemini_key_input: None,
            gemini_key_input_sub: None,

            prompt_input: None,
            prompt_input_sub: None,

            active_assistant_idx: None,
            media_pool_signature: 0,
            last_chat_status: String::new(),
            prompt_send_on_next_render: false,
            show_system_messages: false,

            poll_running: false,
            poll_token: 0,
        };
        this.refresh_agent_status();

        this.sync_worker_media_pool_snapshot(cx);

        this.push_system_message(
            format!(
                "ACP chat ready. Provider: {}. Set agent command, click Connect, then send prompt.",
                this.agent_provider.label()
            ),
            cx,
        );
        match this.agent_provider {
            AcpAgentProvider::Codex if !cli_command_exists("codex") => {
                this.push_system_message(codex_install_hint(), cx);
            }
            AcpAgentProvider::Gemini if !cli_command_exists("gemini") => {
                this.push_system_message(gemini_install_hint(), cx);
            }
            AcpAgentProvider::Claude if !cli_command_exists("claude") => {
                this.push_system_message(claude_install_hint(), cx);
            }
            _ => {}
        }
        this.ensure_worker_poller(cx);
        if this.auto_connect_enabled {
            this.push_system_message(
                format!(
                    "Auto Connect is enabled ({} settings). Connecting...",
                    this.auto_connect_scope.label()
                ),
                cx,
            );
            this.on_connect(cx);
        }

        this
    }

    fn action_button(label: impl Into<String>) -> gpui::Div {
        div()
            .h(px(30.0))
            .px_3()
            .rounded_md()
            .border_1()
            .border_color(white().opacity(0.14))
            .bg(white().opacity(0.05))
            .text_color(white().opacity(0.9))
            .hover(|s| s.bg(white().opacity(0.1)))
            .cursor_pointer()
            .flex()
            .items_center()
            .justify_center()
            .child(label.into())
    }

    fn media_pool_signature(
        items: &[MediaPoolItem],
        ffmpeg_available: bool,
        ffprobe_available: bool,
    ) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        items.len().hash(&mut hasher);
        ffmpeg_available.hash(&mut hasher);
        ffprobe_available.hash(&mut hasher);
        for item in items {
            item.path.hash(&mut hasher);
            item.name.hash(&mut hasher);
            item.duration.as_millis().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn sync_worker_media_pool_snapshot(&mut self, cx: &mut Context<Self>) {
        let (media_pool, ffmpeg_available, ffprobe_available, ffprobe_command) = {
            let gs = self.global.read(cx);
            (
                gs.media_pool.clone(),
                gs.media_dependency.ffmpeg_available,
                gs.media_dependency.ffprobe_available,
                gs.ffprobe_path.clone(),
            )
        };
        let sig = Self::media_pool_signature(&media_pool, ffmpeg_available, ffprobe_available);
        if sig == self.media_pool_signature {
            return;
        }
        self.media_pool_signature = sig;
        self.worker.update_media_pool_snapshot(
            media_pool,
            ffmpeg_available,
            ffprobe_available,
            ffprobe_command,
        );
    }

    fn current_workspace_root(&self, cx: &mut Context<Self>) -> PathBuf {
        resolve_workspace_root(self.global.read(cx).project_file_path.as_deref())
    }

    fn acp_runtime_cwd(&self, cx: &mut Context<Self>) -> PathBuf {
        let workspace_root = self.current_workspace_root(cx);
        let api_root = workspace_root.join("src").join("api");
        let repo_api_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("api");

        // Keep ACP chat scoped to API sources so runtime guidance stays separate from coding-only folders.
        if api_root.is_dir() {
            api_root
        } else if repo_api_root.is_dir() {
            repo_api_root
        } else {
            workspace_root
        }
    }

    fn acp_docs_dir(&self, cx: &mut Context<Self>) -> PathBuf {
        let workspace_docs = self.current_workspace_root(cx).join("docs").join("acp");
        if workspace_docs.is_dir() {
            workspace_docs
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("docs")
                .join("acp")
        }
    }

    fn on_toggle_auto_connect(&mut self, cx: &mut Context<Self>) {
        let next_value = !self.auto_connect_enabled;
        self.auto_connect_enabled = next_value;

        let workspace_root = self.current_workspace_root(cx);
        match save_auto_connect(self.auto_connect_scope, &workspace_root, next_value) {
            Ok(path) => {
                self.push_system_message(
                    format!(
                        "Auto Connect {} (saved to {} settings: {}).",
                        if next_value { "enabled" } else { "disabled" },
                        self.auto_connect_scope.label(),
                        path.display()
                    ),
                    cx,
                );
            }
            Err(err) => {
                self.push_system_message(format!("Failed to save Auto Connect setting: {err}"), cx);
            }
        }

        cx.notify();
    }

    fn chat_bubble(
        msg: &AiChatMessage,
        bubble_index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let (title, border, bg): (&str, gpui::Hsla, gpui::Hsla) = match msg.role {
            AiChatRole::User => ("You", rgb(0x2563eb).into(), rgb(0x172554).into()),
            AiChatRole::Assistant => ("Agent", rgb(0x10b981).into(), rgb(0x052e2b).into()),
            AiChatRole::System => ("System", rgb(0xf59e0b).into(), rgb(0x3f2a05).into()),
        };

        let text = if msg.pending && msg.text.trim().is_empty() {
            "...".to_string()
        } else {
            msg.text.clone()
        };
        let copy_text = text.clone();

        div()
            .w_full()
            .rounded_md()
            .border_1()
            .border_color(border.opacity(0.55))
            .bg(bg.opacity(0.45))
            .px_3()
            .py_2()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(div().text_xs().text_color(border.opacity(0.9)).child(title))
                    .child(
                        div()
                            .h(px(20.0))
                            .px_2()
                            .rounded_md()
                            .border_1()
                            .border_color(white().opacity(0.18))
                            .bg(white().opacity(0.04))
                            .text_xs()
                            .text_color(white().opacity(0.72))
                            .hover(|s| s.bg(white().opacity(0.09)))
                            .cursor_pointer()
                            .flex()
                            .items_center()
                            .justify_center()
                            .child("Copy")
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |_this, _, _, cx| {
                                    cx.write_to_clipboard(ClipboardItem::new_string(
                                        copy_text.clone(),
                                    ));
                                }),
                            ),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(white().opacity(0.92))
                    .whitespace_normal()
                    .child(
                        TextView::markdown(("ai-agents-msg-body", bubble_index), text, window, cx)
                            .selectable(true)
                            .w_full(),
                    ),
            )
    }

    fn push_chat_message(
        &mut self,
        role: AiChatRole,
        text: impl Into<String>,
        pending: bool,
        cx: &mut Context<Self>,
    ) -> usize {
        let text = text.into();
        let mut removed_prefix = 0usize;
        let mut remaining_len = 0usize;
        let mut idx = 0usize;
        self.global.update(cx, |gs, cx| {
            gs.ai_chat_messages.push(AiChatMessage {
                role,
                text: text.clone(),
                pending,
            });
            removed_prefix = prune_ai_chat_history(&mut gs.ai_chat_messages);
            remaining_len = gs.ai_chat_messages.len();
            idx = gs.ai_chat_messages.len().saturating_sub(1);
            cx.notify();
        });
        self.active_assistant_idx =
            rebase_ai_chat_index(self.active_assistant_idx, removed_prefix, remaining_len);
        idx
    }

    fn push_system_message(&mut self, text: impl Into<String>, cx: &mut Context<Self>) {
        let text = text.into();
        let active_idx_before = self.active_assistant_idx;
        let mut inserted_before_assistant = false;
        let mut removed_prefix = 0usize;
        let mut remaining_len = 0usize;

        self.global.update(cx, |gs, cx| {
            if let Some(active_idx) = active_idx_before
                && let Some(active_msg) = gs.ai_chat_messages.get(active_idx)
                && matches!(active_msg.role, AiChatRole::Assistant)
                && active_msg.pending
            {
                gs.ai_chat_messages.insert(
                    active_idx,
                    AiChatMessage {
                        role: AiChatRole::System,
                        text: text.clone(),
                        pending: false,
                    },
                );
                inserted_before_assistant = true;
            } else {
                gs.ai_chat_messages.push(AiChatMessage {
                    role: AiChatRole::System,
                    text: text.clone(),
                    pending: false,
                });
            }

            removed_prefix = prune_ai_chat_history(&mut gs.ai_chat_messages);
            remaining_len = gs.ai_chat_messages.len();
            cx.notify();
        });

        let rebased_active = if inserted_before_assistant {
            active_idx_before.map(|active_idx| active_idx.saturating_add(1))
        } else {
            self.active_assistant_idx
        };
        self.active_assistant_idx =
            rebase_ai_chat_index(rebased_active, removed_prefix, remaining_len);
    }

    fn ensure_inputs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.command_input.is_none() {
            let input = cx.new(|cx| InputState::new(window, cx).placeholder("ACP agent command"));
            let current = self.agent_command.clone();
            input.update(cx, |this, cx| {
                this.set_value(current.clone(), window, cx);
            });

            let sub = cx.subscribe(&input, |this, input, ev, cx| {
                if !matches!(ev, InputEvent::Change) {
                    return;
                }
                this.agent_command = input.read(cx).value().to_string();
            });

            self.command_input = Some(input);
            self.command_input_sub = Some(sub);
        }

        if self.gemini_key_input.is_none() {
            let input = cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder("GEMINI_API_KEY (optional, forwarded on connect)")
            });
            let current = self.gemini_api_key.clone();
            input.update(cx, |this, cx| {
                this.set_value(current.clone(), window, cx);
            });

            let sub = cx.subscribe(&input, |this, input, ev, cx| {
                if !matches!(ev, InputEvent::Change) {
                    return;
                }
                this.gemini_api_key = input.read(cx).value().to_string();
                if this.agent_provider == AcpAgentProvider::Gemini {
                    this.refresh_agent_status();
                    cx.notify();
                }
            });

            self.gemini_key_input = Some(input);
            self.gemini_key_input_sub = Some(sub);
        }

        if self.prompt_input.is_none() {
            let input = cx.new(|cx| {
                InputState::new(window, cx).placeholder("Type a message to the AI agent")
            });
            let sub = cx.subscribe(&input, |this, input, ev, cx| match ev {
                InputEvent::Change => {
                    this.prompt_text = input.read(cx).value().to_string();
                }
                InputEvent::PressEnter { .. } => {
                    this.prompt_text = input.read(cx).value().to_string();
                    this.prompt_send_on_next_render = true;
                    cx.notify();
                }
                _ => {}
            });

            self.prompt_input = Some(input);
            self.prompt_input_sub = Some(sub);
        }
    }

    fn clear_prompt_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.prompt_text.clear();
        if let Some(input) = self.prompt_input.as_ref() {
            input.update(cx, |input, cx| {
                input.set_value("", window, cx);
            });
        }
    }

    fn ensure_worker_poller(&mut self, cx: &mut Context<Self>) {
        if self.poll_running {
            return;
        }

        self.poll_running = true;
        self.poll_token = self.poll_token.wrapping_add(1);
        let token = self.poll_token;

        cx.spawn(async move |view, cx| {
            loop {
                Timer::after(Duration::from_millis(120)).await;
                let keep = view
                    .update(cx, |this, cx| {
                        if this.poll_token != token {
                            this.poll_running = false;
                            return false;
                        }

                        if this.drain_worker_events(cx) {
                            cx.notify();
                        }

                        true
                    })
                    .unwrap_or(false);

                if !keep {
                    break;
                }
            }
        })
        .detach();
    }

    fn append_assistant_chunk(&mut self, chunk: String, cx: &mut Context<Self>) {
        if chunk.is_empty() {
            return;
        }

        if let Some(idx) = self.active_assistant_idx {
            let mut appended = false;
            self.global.update(cx, |gs, cx| {
                if let Some(msg) = gs.ai_chat_messages.get_mut(idx) {
                    msg.text.push_str(&chunk);
                    appended = true;
                    cx.notify();
                }
            });
            if appended {
                return;
            }
        }

        let idx = self.push_chat_message(AiChatRole::Assistant, chunk, true, cx);
        self.active_assistant_idx = Some(idx);
    }

    fn mark_assistant_done(&mut self, cx: &mut Context<Self>) {
        if let Some(idx) = self.active_assistant_idx.take() {
            self.global.update(cx, |gs, cx| {
                if let Some(msg) = gs.ai_chat_messages.get_mut(idx) {
                    msg.pending = false;
                    cx.notify();
                }
            });
        }
    }

    fn start_export_from_acp_request(
        &mut self,
        request: AcpExportRunRequest,
        cx: &mut Context<Self>,
    ) -> Result<AcpExportRunResponse, String> {
        let tools_ready = self.global.read(cx).media_tools_ready_for_export();
        if !tools_ready {
            self.global.update(cx, |gs, cx| {
                gs.show_media_dependency_modal();
                gs.ui_notice = Some(
                    "ACP export requires FFmpeg and FFprobe. Install tools first.".to_string(),
                );
                cx.notify();
            });
            self.push_system_message(
                "ACP export blocked: missing FFmpeg/FFprobe. Install tools and retry.".to_string(),
                cx,
            );
            return Err(
                "MISSING_FFMPEG_FFPROBE: anica.export/run requires ffmpeg and ffprobe. Install FFmpeg package and retry."
                    .to_string(),
            );
        }

        let resolved = {
            let gs = self.global.read(cx);
            resolve_acp_export_run_request(request, gs).map_err(|err| err.to_string())?
        };
        let out_path = resolved.out_path.clone();
        let mode_id = resolved.export_mode.id().to_string();
        let preset_id = resolved.export_preset.id().to_string();

        self.push_system_message(format!("ACP export started: {}", out_path), cx);
        self.busy = true;

        let cancel_signal = Arc::new(AtomicBool::new(false));
        self.global.update(cx, |gs, cx| {
            gs.export_begin(out_path.clone(), resolved.export_total);
            cx.notify();
        });

        enum AcpExportWorkerEvent {
            Progress(crate::core::export::ExportProgress),
            Finished(Result<String, String>),
        }

        let (tx, rx) = mpsc::channel::<AcpExportWorkerEvent>();
        let out_path_for_thread = resolved.out_path.clone();
        let ffmpeg_path = resolved.ffmpeg_path.clone();
        std::thread::spawn(move || {
            let tx_progress = tx.clone();
            let result = FfmpegExporter::export(
                &ffmpeg_path,
                &resolved.v1,
                &resolved.audio_tracks,
                &resolved.video_tracks,
                &resolved.subtitle_tracks,
                &resolved.subtitle_groups,
                &out_path_for_thread,
                resolved.layout_canvas_w,
                resolved.layout_canvas_h,
                resolved.export_w,
                resolved.export_h,
                resolved.layer_effects,
                &resolved.layer_effect_clips,
                resolved.export_color_mode,
                resolved.export_mode,
                resolved.export_preset,
                resolved.export_settings,
                resolved.export_range,
                cancel_signal,
                move |progress| {
                    let _ = tx_progress.send(AcpExportWorkerEvent::Progress(progress));
                },
            );
            let result = match result {
                Ok(_) => Ok(out_path_for_thread),
                Err(err) => {
                    eprintln!("[ACP Export] Failed: {err}");
                    Err(err.to_string())
                }
            };
            let _ = tx.send(AcpExportWorkerEvent::Finished(result));
        });

        let global_for_finish = self.global.clone();
        cx.spawn(async move |view, cx| {
            loop {
                Timer::after(Duration::from_millis(120)).await;

                let mut latest_progress = None;
                let mut finished = None;
                loop {
                    match rx.try_recv() {
                        Ok(AcpExportWorkerEvent::Progress(p)) => latest_progress = Some(p),
                        Ok(AcpExportWorkerEvent::Finished(result)) => {
                            finished = Some(result);
                            break;
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            finished = Some(Err("Export worker disconnected.".to_string()));
                            break;
                        }
                    }
                }

                let has_finished = finished.is_some();
                let updated = view.update(cx, |this, cx| {
                    let mut chat_notice: Option<String> = None;
                    global_for_finish.update(cx, |gs, cx| {
                        if let Some(p) = latest_progress {
                            gs.export_update_progress(p.rendered, p.total, p.speed);
                        }
                        if let Some(result) = finished {
                            match result {
                                Ok(path) => {
                                    gs.export_done();
                                    gs.export_last_out_path = Some(path.clone());
                                    gs.ui_notice = Some(format!("Export saved: {}", path));
                                    chat_notice = Some(format!("ACP export saved: {}", path));
                                }
                                Err(err) => {
                                    if is_cancelled_export_error(&err) {
                                        gs.export_cancelled();
                                        gs.ui_notice = Some("Export stopped.".to_string());
                                        chat_notice = Some("ACP export stopped.".to_string());
                                    } else {
                                        gs.export_fail(err.clone());
                                        chat_notice = Some(format!("ACP export failed: {err}"));
                                    }
                                }
                            }
                            this.busy = false;
                        }
                        cx.notify();
                    });
                    if let Some(msg) = chat_notice {
                        this.push_system_message(msg, cx);
                    }
                });

                if has_finished || updated.is_err() {
                    break;
                }
            }
        })
        .detach();

        Ok(AcpExportRunResponse {
            ok: true,
            started: true,
            mode: mode_id,
            preset: preset_id,
            out_path: Some(out_path),
            message: "Export job started.".to_string(),
        })
    }

    fn drain_worker_events(&mut self, cx: &mut Context<Self>) -> bool {
        let mut changed = false;

        while let Some(ev) = self.worker.try_recv() {
            changed = true;
            match ev {
                AcpUiEvent::Status(line) => {
                    self.last_status = line.clone();
                    let status_for_chat = if let Some(raw) = line.strip_prefix("ACP_STATUS:") {
                        Some(raw.trim().to_string())
                    } else if line.starts_with("Tool call:")
                        || line.contains("timed out")
                        || line.to_lowercase().contains("error")
                    {
                        Some(line.clone())
                    } else {
                        None
                    };
                    if let Some(status_text) = status_for_chat
                        && !status_text.is_empty()
                        && status_text != self.last_chat_status
                    {
                        self.last_chat_status = status_text.clone();
                        self.push_system_message(status_text, cx);
                    }
                }
                AcpUiEvent::Connected {
                    session_id,
                    agent_label,
                } => {
                    self.connected = true;
                    self.connected_provider = Some(self.agent_provider);
                    self.busy = false;
                    self.last_status = format!("Connected ({agent_label})");
                    self.push_system_message(
                        format!("Connected to agent. Session: {session_id}"),
                        cx,
                    );
                }
                AcpUiEvent::Disconnected { reason } => {
                    self.connected = false;
                    self.connected_provider = None;
                    self.busy = false;
                    self.mark_assistant_done(cx);
                    self.last_status = reason.clone();
                    self.push_system_message(reason, cx);
                }
                AcpUiEvent::AssistantChunk(chunk) => {
                    self.append_assistant_chunk(chunk, cx);
                }
                AcpUiEvent::PromptFinished { stop_reason } => {
                    self.busy = false;
                    self.mark_assistant_done(cx);
                    self.last_status = format!("Prompt finished: {stop_reason}");
                }
                AcpUiEvent::Error(err) => {
                    self.busy = false;
                    self.mark_assistant_done(cx);
                    self.last_status = format!("Error: {err}");
                    self.push_system_message(format!("Error: {err}"), cx);
                    if err.to_lowercase().contains("codex cli not found")
                        && self.agent_provider == AcpAgentProvider::Codex
                    {
                        self.push_system_message(codex_install_hint(), cx);
                    }
                    if err.to_lowercase().contains("gemini")
                        && self.agent_provider == AcpAgentProvider::Gemini
                    {
                        self.push_system_message(gemini_install_hint(), cx);
                    }
                    if err.to_lowercase().contains("claude")
                        && self.agent_provider == AcpAgentProvider::Claude
                    {
                        self.push_system_message(claude_install_hint(), cx);
                    }
                }
                AcpUiEvent::ToolBridgeRequest(request) => match request {
                    AcpToolBridgeRequest::LlmDecisionMakingSrtSimilarSerach {
                        request,
                        reply_tx,
                    } => {
                        let response = llm_decision_making_srt_similar_serach(request);
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::BuildAutonomousEditPlan { request, reply_tx } => {
                        let response = {
                            let gs = self.global.read(cx);
                            build_autonomous_edit_plan(gs, request)
                        };
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::GetTimelineSnapshot { request, reply_tx } => {
                        let response = {
                            let gs = self.global.read(cx);
                            get_timeline_snapshot(gs, request)
                        };
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::GetAudioSilenceMap { request, reply_tx } => {
                        let response = {
                            let gs = self.global.read(cx);
                            get_audio_silence_map(gs, request)
                        };
                        // Show silence preview modal so user can pick which ranges to cut.
                        // Candidates with confidence data are shown as checkbox rows.
                        if !response.cut_candidates.is_empty() {
                            let modal_state = SilencePreviewModalState {
                                timeline_revision: response.timeline_revision.clone(),
                                candidates: response
                                    .cut_candidates
                                    .iter()
                                    .map(|c| SilencePreviewCandidate {
                                        start_ms: c.start_ms,
                                        end_ms: c.end_ms,
                                        confidence: c.confidence,
                                        reason: c.reason.clone(),
                                        selected: true, // pre-select all by default
                                    })
                                    .collect(),
                            };
                            self.global.update(cx, |gs, cx| {
                                gs.show_silence_preview_modal(modal_state);
                                cx.notify();
                            });
                        }
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::BuildAudioSilenceCutPlan { request, reply_tx } => {
                        let response = {
                            let gs = self.global.read(cx);
                            build_audio_silence_cut_plan(gs, request)
                        };
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::GetTranscriptLowConfidenceMap { request, reply_tx } => {
                        let response = {
                            let gs = self.global.read(cx);
                            get_transcript_low_confidence_map(gs, request)
                        };
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::BuildTranscriptLowConfidenceCutPlan {
                        request,
                        reply_tx,
                    } => {
                        let response = {
                            let gs = self.global.read(cx);
                            build_transcript_low_confidence_cut_plan(gs, request)
                        };
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::GetSubtitleGapMap { request, reply_tx } => {
                        let response = {
                            let gs = self.global.read(cx);
                            get_subtitle_gap_map(gs, request)
                        };
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::BuildSubtitleGapCutPlan { request, reply_tx } => {
                        let response = {
                            let gs = self.global.read(cx);
                            build_subtitle_gap_cut_plan(gs, request)
                        };
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::GetSubtitleSemanticRepeats { request, reply_tx } => {
                        let response = {
                            let gs = self.global.read(cx);
                            get_subtitle_semantic_repeats(gs, request)
                        };
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::ValidateEditPlan { request, reply_tx } => {
                        let response = {
                            let gs = self.global.read(cx);
                            validate_edit_plan(gs, &request)
                        };
                        let _ = reply_tx.send(Ok(response));
                    }
                    AcpToolBridgeRequest::ApplyEditPlan { request, reply_tx } => {
                        let mut applied_ok = false;
                        let response = self.global.update(cx, |gs, cx| {
                            let result = apply_edit_plan(gs, &request);
                            if result.ok {
                                applied_ok = true;
                                cx.notify();
                            }
                            result
                        });
                        let _ = reply_tx.send(Ok(response));
                        if applied_ok {
                            self.sync_worker_media_pool_snapshot(cx);
                        }
                    }
                    AcpToolBridgeRequest::RunExport { request, reply_tx } => {
                        let response = self.start_export_from_acp_request(request, cx);
                        let _ = reply_tx.send(response);
                    }
                    AcpToolBridgeRequest::RemoveMediaPoolById { request, reply_tx } => {
                        let mut removed = false;
                        let response = self.global.update(cx, |gs, cx| {
                            let result = remove_media_pool_by_id(gs, request);
                            if result.removed {
                                removed = true;
                                cx.emit(MediaPoolUiEvent::StateChanged);
                                cx.notify();
                            }
                            result
                        });
                        let _ = reply_tx.send(Ok(response));
                        if removed {
                            self.sync_worker_media_pool_snapshot(cx);
                        }
                    }
                    AcpToolBridgeRequest::ClearMediaPool { request, reply_tx } => {
                        let mut removed_count = 0usize;
                        let response = self.global.update(cx, |gs, cx| {
                            let result = clear_media_pool(gs, request);
                            removed_count = result.removed_count;
                            if result.removed_count > 0 {
                                cx.emit(MediaPoolUiEvent::StateChanged);
                                cx.notify();
                            }
                            result
                        });
                        let _ = reply_tx.send(Ok(response));
                        if removed_count > 0 {
                            self.sync_worker_media_pool_snapshot(cx);
                        }
                    }
                },
            }
        }

        changed
    }

    // Pure disconnect — stops the agent, no reconnect.
    fn on_disconnect(&mut self, cx: &mut Context<Self>) {
        if !self.connected {
            return;
        }
        self.worker.disconnect();
        self.connected = false;
        self.connected_provider = None;
        self.busy = false;
        self.last_status = "Disconnecting...".to_string();
        cx.notify();
    }

    // Connect to the currently selected provider.
    // If already connected to a different provider, disconnect first then connect.
    fn on_connect(&mut self, cx: &mut Context<Self>) {
        if self.connected {
            self.worker.disconnect();
            self.connected = false;
            self.connected_provider = None;
            self.busy = false;
            self.push_system_message(
                format!(
                    "Disconnected previous provider. Connecting to {}...",
                    self.agent_provider.label()
                ),
                cx,
            );
        }

        let cmd = self.agent_command.trim().to_string();
        if cmd.is_empty() {
            self.push_system_message("Agent command is empty.", cx);
            cx.notify();
            return;
        }
        let resolved_cmd = resolve_agent_command_for_spawn(&cmd);
        if self.agent_provider == AcpAgentProvider::Gemini
            && !has_gemini_api_key(&self.gemini_api_key)
            && !matches!(self.agent_status, AgentLoginStatus::LoggedIn { .. })
        {
            self.push_system_message(
                "No API key detected. Trying Gemini CLI auth state. If connect fails, run `gemini` then `/auth`, or set GEMINI_API_KEY."
                    .to_string(),
                cx,
            );
        }

        // Force ACP session cwd to API folder so runtime AGENTS policy is applied first.
        let runtime_cwd = self.acp_runtime_cwd(cx);
        let acp_docs_dir = self.acp_docs_dir(cx);
        let api_scoped = runtime_cwd.ends_with("src/api");
        let docs_scoped = acp_docs_dir.is_dir();

        self.last_status = format!(
            "Connecting with: {resolved_cmd} (cwd: {})",
            runtime_cwd.display()
        );
        let mut acp_env = vec![(
            "ANICA_ACP_PROVIDER".to_string(),
            match self.agent_provider {
                AcpAgentProvider::Codex => "codex",
                AcpAgentProvider::Gemini => "gemini",
                AcpAgentProvider::Claude => "claude",
            }
            .to_string(),
        )];
        if self.agent_provider == AcpAgentProvider::Codex {
            acp_env.push((
                "ANICA_CODEX_REASONING_EFFORT".to_string(),
                self.codex_reasoning_mode.env_value().to_string(),
            ));
        }
        if self.agent_provider == AcpAgentProvider::Gemini && !self.gemini_api_key.trim().is_empty()
        {
            acp_env.push(("GEMINI_API_KEY".to_string(), self.gemini_api_key.clone()));
            acp_env.push(("GOOGLE_API_KEY".to_string(), self.gemini_api_key.clone()));
        }
        // Pin ACP docs lookups to docs/acp so runtime guidance stays isolated from development docs.
        if docs_scoped {
            acp_env.push((
                "ANICA_DOCS_DIR".to_string(),
                acp_docs_dir.to_string_lossy().to_string(),
            ));
        }
        self.worker
            .connect(resolved_cmd.clone(), Some(runtime_cwd.clone()), acp_env);
        self.busy = true;
        self.push_system_message(
            format!(
                "Connecting {} ACP agent: {resolved_cmd}{}",
                self.agent_provider.label(),
                if self.agent_provider == AcpAgentProvider::Codex {
                    format!(" (thinking: {})", self.codex_reasoning_mode.label())
                } else {
                    String::new()
                }
            ),
            cx,
        );
        if api_scoped {
            self.push_system_message(
                format!(
                    "ACP runtime scope locked to API folder: {}",
                    runtime_cwd.display()
                ),
                cx,
            );
        } else {
            self.push_system_message(
                format!(
                    "API folder not found at {}/src/api. Fallback scope: {}",
                    self.current_workspace_root(cx).display(),
                    runtime_cwd.display()
                ),
                cx,
            );
        }
        if docs_scoped {
            self.push_system_message(
                format!("ACP docs scope locked to: {}", acp_docs_dir.display()),
                cx,
            );
        } else {
            self.push_system_message(
                format!(
                    "ACP docs folder not found at {}. Agent will use default docs discovery.",
                    acp_docs_dir.display()
                ),
                cx,
            );
        }
        cx.notify();
    }

    fn submit_prompt(
        &mut self,
        prompt: String,
        clear_main_prompt: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if prompt.is_empty() {
            return false;
        }

        if !self.connected {
            self.push_system_message("Not connected. Connect an ACP agent first.", cx);
            cx.notify();
            return false;
        }

        let _ = self.push_chat_message(AiChatRole::User, prompt.clone(), false, cx);
        let pending_idx = self.push_chat_message(AiChatRole::Assistant, String::new(), true, cx);
        self.active_assistant_idx = Some(pending_idx);

        self.worker.send_prompt(prompt);
        self.busy = true;
        if clear_main_prompt {
            self.clear_prompt_input(window, cx);
        }
        cx.notify();
        true
    }

    fn on_send_prompt(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let prompt = self.prompt_text.trim().to_string();
        let _ = self.submit_prompt(prompt, true, window, cx);
    }

    pub fn send_prompt_from_external(
        &mut self,
        prompt: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        self.submit_prompt(prompt.trim().to_string(), false, window, cx)
    }
}

impl Render for AiAgentsPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let _ = self.global.read(cx).active_page;

        if self.prompt_send_on_next_render {
            self.prompt_send_on_next_render = false;
            if self.connected && !self.busy {
                self.on_send_prompt(window, cx);
            }
        }

        self.sync_worker_media_pool_snapshot(cx);
        self.ensure_inputs(window, cx);
        self.ensure_worker_poller(cx);

        let command_input_elem = if let Some(input) = self.command_input.as_ref() {
            Input::new(input).h(px(32.0)).w_full().into_any_element()
        } else {
            div()
                .h(px(32.0))
                .w_full()
                .rounded_sm()
                .bg(white().opacity(0.05))
                .into_any_element()
        };

        let gemini_key_input_elem = if let Some(input) = self.gemini_key_input.as_ref() {
            Input::new(input).h(px(32.0)).w_full().into_any_element()
        } else {
            div()
                .h(px(32.0))
                .w_full()
                .rounded_sm()
                .bg(white().opacity(0.05))
                .into_any_element()
        };

        let prompt_input_elem = if let Some(input) = self.prompt_input.as_ref() {
            Input::new(input).h(px(46.0)).w_full().into_any_element()
        } else {
            div()
                .h(px(46.0))
                .w_full()
                .rounded_sm()
                .bg(white().opacity(0.05))
                .into_any_element()
        };

        let chat_messages = self.global.read(cx).ai_chat_messages.clone();
        let system_count = chat_messages
            .iter()
            .filter(|m| matches!(m.role, AiChatRole::System))
            .count();
        let visible_chat_messages: Vec<AiChatMessage> = chat_messages
            .iter()
            .filter(|m| self.show_system_messages || !matches!(m.role, AiChatRole::System))
            .cloned()
            .collect();
        let mut message_list = div().flex().flex_col().gap_2();
        if visible_chat_messages.is_empty() {
            message_list =
                message_list.child(div().text_xs().text_color(white().opacity(0.55)).child(
                    if system_count > 0 {
                        format!(
                            "System logs are hidden ({}). Toggle to view them.",
                            system_count
                        )
                    } else {
                        "No messages yet.".to_string()
                    },
                ));
        } else {
            for (idx, msg) in visible_chat_messages.iter().enumerate() {
                message_list = message_list.child(Self::chat_bubble(msg, idx, window, cx));
            }
        }

        let is_logged_in = matches!(self.agent_status, AgentLoginStatus::LoggedIn { .. });
        let can_send = self.connected && !self.busy;
        let mut system_toggle_button = Self::action_button(if self.show_system_messages {
            format!("Hide System ({})", system_count)
        } else {
            format!("Show System ({})", system_count)
        })
        .h(px(26.0))
        .text_xs();
        if self.show_system_messages {
            system_toggle_button = system_toggle_button
                .bg(gpui::Hsla::from(rgb(0x7c2d12)).opacity(0.35))
                .border_color(gpui::Hsla::from(rgb(0xf59e0b)).opacity(0.45));
        }
        let mut auto_connect_button = Self::action_button(if self.auto_connect_enabled {
            "Auto Connect: On"
        } else {
            "Auto Connect: Off"
        });
        if self.auto_connect_enabled {
            auto_connect_button = auto_connect_button
                .bg(gpui::Hsla::from(rgb(0x14532d)).opacity(0.45))
                .border_color(gpui::Hsla::from(rgb(0x22c55e)).opacity(0.45))
                .text_color(white().opacity(0.95));
        }
        let mode_button = |label: &'static str, mode: CodexReasoningMode, active: bool| {
            let mut chip = div()
                .h(px(28.0))
                .px_2()
                .rounded_md()
                .border_1()
                .border_color(if active {
                    gpui::Hsla::from(rgb(0x3b82f6)).opacity(0.55)
                } else {
                    white().opacity(0.14)
                })
                .bg(if active {
                    gpui::Hsla::from(rgb(0x1e3a8a)).opacity(0.35)
                } else {
                    white().opacity(0.05)
                })
                .text_xs()
                .text_color(white().opacity(if active { 0.95 } else { 0.75 }))
                .flex()
                .items_center()
                .justify_center()
                .child(label);
            if !active {
                chip = chip.hover(|s| s.bg(white().opacity(0.1))).cursor_pointer();
            }
            chip.on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    this.codex_reasoning_mode = mode;
                    this.push_system_message(
                        format!(
                            "Thinking mode set to {} (applies on next connect).",
                            mode.label()
                        ),
                        cx,
                    );
                    cx.notify();
                }),
            )
        };
        let connected_prov = self.connected_provider;
        let provider_button = |label: &'static str, provider: AcpAgentProvider, active: bool| {
            let is_connected = connected_prov == Some(provider);
            // Show a green dot next to the label of the provider that is actually connected.
            let display_label: SharedString = if is_connected {
                format!("{label} \u{25cf}").into()
            } else {
                label.into()
            };
            let mut chip = div()
                .h(px(30.0))
                .px_3()
                .rounded_md()
                .border_1()
                .border_color(if active {
                    gpui::Hsla::from(rgb(0x10b981)).opacity(0.55)
                } else {
                    white().opacity(0.14)
                })
                .bg(if active {
                    gpui::Hsla::from(rgb(0x064e3b)).opacity(0.45)
                } else {
                    white().opacity(0.05)
                })
                .text_xs()
                .text_color(if is_connected {
                    gpui::Hsla::from(rgb(0x22c55e)).opacity(0.95)
                } else {
                    white().opacity(if active { 0.95 } else { 0.75 })
                })
                .flex()
                .items_center()
                .justify_center()
                .child(display_label);
            if !active {
                chip = chip.hover(|s| s.bg(white().opacity(0.1))).cursor_pointer();
            }
            chip.on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, window, cx| {
                    this.select_agent_provider(provider, window, cx);
                    cx.notify();
                }),
            )
        };
        let provider_name = self.agent_provider.label();
        let status_title = self.agent_status.title(self.agent_provider);
        let status_detail = self.agent_status.detail_text();
        let status_hint = self.agent_status.action_hint(self.agent_provider);
        let mut provider_specific_help = div().flex().flex_col().gap_2();
        if self.agent_provider == AcpAgentProvider::Codex {
            provider_specific_help = provider_specific_help
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .text_xs()
                                .text_color(white().opacity(0.55))
                                .child("Codex Thinking Mode"),
                        )
                        .child(
                            div()
                                .flex()
                                .gap_2()
                                .child(mode_button(
                                    "Low",
                                    CodexReasoningMode::Low,
                                    self.codex_reasoning_mode == CodexReasoningMode::Low,
                                ))
                                .child(mode_button(
                                    "Medium",
                                    CodexReasoningMode::Medium,
                                    self.codex_reasoning_mode == CodexReasoningMode::Medium,
                                ))
                                .child(mode_button(
                                    "High",
                                    CodexReasoningMode::High,
                                    self.codex_reasoning_mode == CodexReasoningMode::High,
                                ))
                                .child(mode_button(
                                    "Extra High",
                                    CodexReasoningMode::ExtraHigh,
                                    self.codex_reasoning_mode == CodexReasoningMode::ExtraHigh,
                                )),
                        ),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.55))
                        .child("If Codex reply fails with \"not found\": install CLI (`npm i -g @openai/codex`), then run `codex login`."),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.5))
                        .child(
                            "Thinking mode is passed as ANICA_CODEX_REASONING_EFFORT to anica-acp at connect time.",
                        ),
                );
        } else if self.agent_provider == AcpAgentProvider::Gemini {
            provider_specific_help = provider_specific_help
                .child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.55))
                        .child("Gemini API Key (optional)"),
                )
                .child(gemini_key_input_elem)
                .child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.5))
                        .child(
                            "Gemini mode uses the same bundled anica-acp tool bridge as Codex. Backend model routing is selected via provider (Codex/Gemini/Claude). You can use API key or CLI login (`gemini`, then `/auth`). If provided, GEMINI_API_KEY / GOOGLE_API_KEY is forwarded on connect.",
                        ),
                );
        } else {
            provider_specific_help = provider_specific_help
                .child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.55))
                        .child("Claude Login"),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(white().opacity(0.5))
                        .child(
                            "Claude mode uses the same bundled anica-acp tool bridge as Codex/Gemini. Login in terminal: `claude auth login`, verify with `claude auth status`. You can also use ANTHROPIC_API_KEY.",
                        ),
                );
        }

        div()
            .size_full()
            .bg(gpui::rgb(0x09090b))
            .p_5()
            .flex()
            .flex_col()
            .gap_4()
            .child(
                div()
                    .text_lg()
                    .text_color(white().opacity(0.92))
                    .child("AI Agents (ACP Chat)"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(white().opacity(0.62))
                    .child("Connect an ACP agent command and chat directly from Anica."),
            )
            .child(
                div()
                    .rounded_lg()
                    .border_1()
                    .border_color(self.agent_status.color().opacity(0.5))
                    .bg(self.agent_status.color().opacity(0.08))
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_sm()
                            .text_color(self.agent_status.color())
                            .child(status_title),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.72))
                            .child(status_detail),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.6))
                            .child(status_hint),
                    ),
            )
            .child(
                div()
                    .rounded_lg()
                    .border_1()
                    .border_color(white().opacity(0.14))
                    .bg(white().opacity(0.03))
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.55))
                            .child("ACP Provider"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(provider_button(
                                "Codex",
                                AcpAgentProvider::Codex,
                                self.agent_provider == AcpAgentProvider::Codex,
                            ))
                            .child(provider_button(
                                "Gemini",
                                AcpAgentProvider::Gemini,
                                self.agent_provider == AcpAgentProvider::Gemini,
                            ))
                            .child(provider_button(
                                "Claude",
                                AcpAgentProvider::Claude,
                                self.agent_provider == AcpAgentProvider::Claude,
                            )),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.55))
                            .child("ACP Agent Command"),
                    )
                    .child(command_input_elem)
                    .child(provider_specific_help)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                Self::action_button("Refresh Login").on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        this.refresh_agent_status();
                                        cx.notify();
                                    }),
                                ),
                            )
                            .child(
                                Self::action_button("Connect")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        this.on_connect(cx);
                                    }),
                                ),
                            )
                            .child(
                                Self::action_button("Disconnect")
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        this.on_disconnect(cx);
                                    }),
                                ),
                            )
                            .child(
                                auto_connect_button.on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, _, cx| {
                                        this.on_toggle_auto_connect(cx);
                                    }),
                                ),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(if self.connected {
                                        gpui::Hsla::from(rgb(0x22c55e)).opacity(0.9)
                                    } else if is_logged_in {
                                        white().opacity(0.65)
                                    } else {
                                        gpui::Hsla::from(rgb(0xf59e0b)).opacity(0.9)
                                    })
                                    .child(if self.connected {
                                        format!("Connected ({provider_name}) | {}", self.last_status)
                                    } else {
                                        format!("Disconnected ({provider_name}) | {}", self.last_status)
                                    }),
                            ),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(white().opacity(0.45))
                            .child("Tip: Codex, Gemini, and Claude default to bundled anica-acp (same tool-bridge pipeline). You can still override ACP Agent Command manually."),
                    )
            )
            .child(
                div().flex().items_center().justify_between().child(system_toggle_button.on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        this.show_system_messages = !this.show_system_messages;
                        cx.notify();
                    }),
                )),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .rounded_lg()
                    .border_1()
                    .border_color(white().opacity(0.14))
                    .bg(white().opacity(0.02))
                    .p_3()
                    .overflow_y_scrollbar()
                    .child(message_list),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().flex_1().min_w_0().child(prompt_input_elem))
                    .child(
                        div()
                            .w(px(86.0))
                            .flex_shrink_0()
                            .child(
                                if can_send {
                                    Self::action_button("Send")
                                } else {
                                    Self::action_button(if self.busy { "Sending..." } else { "Send" })
                                        .opacity(0.55)
                                }
                                .h(px(46.0))
                                .w_full()
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, window, cx| {
                                        if this.connected && !this.busy {
                                            this.on_send_prompt(window, cx);
                                        }
                                    }),
                                ),
                        ),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::{AI_CHAT_MAX_CONVERSATION_MESSAGES, prune_ai_chat_history};
    use crate::core::global_state::{AiChatMessage, AiChatRole};

    fn msg(role: AiChatRole, text: &str) -> AiChatMessage {
        AiChatMessage {
            role,
            text: text.to_string(),
            pending: false,
        }
    }

    #[test]
    fn prune_history_ignores_system_messages_for_limit() {
        let mut messages = vec![
            msg(AiChatRole::System, "boot"),
            msg(AiChatRole::User, "u1"),
            msg(AiChatRole::System, "tool"),
            msg(AiChatRole::Assistant, "a1"),
        ];

        let removed = prune_ai_chat_history(&mut messages);

        assert_eq!(removed, 0);
        assert_eq!(messages.len(), 4);
    }

    #[test]
    fn prune_history_removes_oldest_user_assistant_turn() {
        let mut messages = Vec::new();
        for idx in 0..(AI_CHAT_MAX_CONVERSATION_MESSAGES / 2 + 1) {
            messages.push(msg(AiChatRole::User, &format!("u{idx}")));
            messages.push(msg(AiChatRole::System, &format!("s{idx}")));
            messages.push(msg(AiChatRole::Assistant, &format!("a{idx}")));
        }

        let removed = prune_ai_chat_history(&mut messages);

        assert!(removed >= 3);
        assert_eq!(
            messages
                .iter()
                .filter(|msg| !matches!(msg.role, AiChatRole::System))
                .count(),
            AI_CHAT_MAX_CONVERSATION_MESSAGES
        );
        assert_eq!(messages.first().map(|msg| msg.text.as_str()), Some("u1"));
    }
}
