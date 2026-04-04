// =========================================
// =========================================
// src/core/user_settings.rs
use serde::Deserialize;
use serde_json::{Map, Value};
use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

const SETTINGS_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum UserSettingsError {
    #[error("failed to read {path}: {source}")]
    ReadFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    ParseFile {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("failed to create settings dir {path}: {source}")]
    CreateSettingsDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to encode settings JSON: {source}")]
    EncodeSettingsJson { source: serde_json::Error },
    #[error("failed to write temp settings {path}: {source}")]
    WriteTempSettings {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to replace settings {path}: {rename_error}; retry failed: {retry_error}")]
    ReplaceSettings {
        path: PathBuf,
        rename_error: String,
        retry_error: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsScope {
    User,
    Workspace,
}

impl SettingsScope {
    pub fn label(self) -> &'static str {
        match self {
            SettingsScope::User => "User",
            SettingsScope::Workspace => "Workspace",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingSource {
    Default,
    User,
    Workspace,
}

impl SettingSource {
    pub fn preferred_scope(self) -> SettingsScope {
        match self {
            SettingSource::Workspace => SettingsScope::Workspace,
            SettingSource::Default | SettingSource::User => SettingsScope::User,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EffectiveSettings {
    pub acp_auto_connect: bool,
    pub acp_agent_command: Option<String>,
    pub acp_reasoning_mode: Option<String>,
    pub acp_codex_cli_bin: Option<String>,
    pub acp_gemini_cli_bin: Option<String>,
    pub acp_claude_cli_bin: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LoadedSettings {
    pub effective: EffectiveSettings,
    pub auto_connect_source: SettingSource,
    pub agent_command_source: SettingSource,
    pub reasoning_mode_source: SettingSource,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct SettingsLayer {
    #[serde(default)]
    acp: Option<AcpLayer>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct AcpLayer {
    #[serde(default)]
    auto_connect: Option<bool>,
    #[serde(default)]
    agent_command: Option<String>,
    #[serde(default)]
    reasoning_mode: Option<String>,
    #[serde(default)]
    codex_cli_bin: Option<String>,
    #[serde(default)]
    gemini_cli_bin: Option<String>,
    #[serde(default)]
    claude_cli_bin: Option<String>,
}

pub fn resolve_workspace_root(project_file_path: Option<&Path>) -> PathBuf {
    if let Some(parent) = project_file_path.and_then(Path::parent) {
        return parent.to_path_buf();
    }

    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn workspace_settings_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(".anica").join("settings.json")
}

pub fn user_settings_path() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
            return home
                .join("Library")
                .join("Application Support")
                .join("Anica")
                .join("User")
                .join("settings.json");
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = env::var_os("APPDATA").map(PathBuf::from) {
            return appdata.join("Anica").join("User").join("settings.json");
        }

        if let Some(home) = env::var_os("USERPROFILE").map(PathBuf::from) {
            return home
                .join("AppData")
                .join("Roaming")
                .join("Anica")
                .join("User")
                .join("settings.json");
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        if let Some(config_home) = env::var_os("XDG_CONFIG_HOME").map(PathBuf::from) {
            return config_home.join("anica").join("User").join("settings.json");
        }

        if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
            return home
                .join(".config")
                .join("anica")
                .join("User")
                .join("settings.json");
        }
    }

    PathBuf::from(".anica").join("user").join("settings.json")
}

pub fn load_settings(workspace_root: &Path) -> LoadedSettings {
    let user_path = user_settings_path();
    let workspace_path = workspace_settings_path(workspace_root);

    let user_layer = read_settings_layer(&user_path, "user");
    let workspace_layer = read_settings_layer(&workspace_path, "workspace");

    let mut loaded = LoadedSettings {
        effective: EffectiveSettings::default(),
        auto_connect_source: SettingSource::Default,
        agent_command_source: SettingSource::Default,
        reasoning_mode_source: SettingSource::Default,
    };

    if let Some(layer) = user_layer.as_ref() {
        apply_layer(&mut loaded, layer, SettingSource::User);
    }
    if let Some(layer) = workspace_layer.as_ref() {
        apply_layer(&mut loaded, layer, SettingSource::Workspace);
    }

    loaded
}

pub fn save_auto_connect(
    scope: SettingsScope,
    workspace_root: &Path,
    enabled: bool,
) -> Result<PathBuf, UserSettingsError> {
    let path = match scope {
        SettingsScope::User => user_settings_path(),
        SettingsScope::Workspace => workspace_settings_path(workspace_root),
    };

    update_settings_file(&path, |root| {
        root.insert(
            "schema_version".to_string(),
            Value::from(SETTINGS_SCHEMA_VERSION),
        );

        let acp = ensure_child_object(root, "acp");
        acp.insert("auto_connect".to_string(), Value::Bool(enabled));
    })?;

    Ok(path)
}

pub fn save_acp_cli_paths(
    scope: SettingsScope,
    workspace_root: &Path,
    codex_cli_bin: Option<&str>,
    gemini_cli_bin: Option<&str>,
    claude_cli_bin: Option<&str>,
) -> Result<PathBuf, UserSettingsError> {
    let path = match scope {
        SettingsScope::User => user_settings_path(),
        SettingsScope::Workspace => workspace_settings_path(workspace_root),
    };

    let codex_cli_bin = normalize_non_empty(codex_cli_bin);
    let gemini_cli_bin = normalize_non_empty(gemini_cli_bin);
    let claude_cli_bin = normalize_non_empty(claude_cli_bin);

    update_settings_file(&path, |root| {
        root.insert(
            "schema_version".to_string(),
            Value::from(SETTINGS_SCHEMA_VERSION),
        );

        let acp = ensure_child_object(root, "acp");

        match &codex_cli_bin {
            Some(value) => {
                acp.insert("codex_cli_bin".to_string(), Value::String(value.clone()));
            }
            None => {
                acp.remove("codex_cli_bin");
            }
        }

        match &gemini_cli_bin {
            Some(value) => {
                acp.insert("gemini_cli_bin".to_string(), Value::String(value.clone()));
            }
            None => {
                acp.remove("gemini_cli_bin");
            }
        }

        match &claude_cli_bin {
            Some(value) => {
                acp.insert("claude_cli_bin".to_string(), Value::String(value.clone()));
            }
            None => {
                acp.remove("claude_cli_bin");
            }
        }
    })?;

    Ok(path)
}

fn apply_layer(loaded: &mut LoadedSettings, layer: &SettingsLayer, source: SettingSource) {
    let Some(acp) = layer.acp.as_ref() else {
        return;
    };

    if let Some(v) = acp.auto_connect {
        loaded.effective.acp_auto_connect = v;
        loaded.auto_connect_source = source;
    }

    if let Some(cmd) = normalize_non_empty(acp.agent_command.as_deref()) {
        loaded.effective.acp_agent_command = Some(cmd);
        loaded.agent_command_source = source;
    }

    if let Some(mode) = normalize_non_empty(acp.reasoning_mode.as_deref()) {
        loaded.effective.acp_reasoning_mode = Some(mode.to_ascii_lowercase());
        loaded.reasoning_mode_source = source;
    }

    if let Some(path) = normalize_non_empty(acp.codex_cli_bin.as_deref()) {
        loaded.effective.acp_codex_cli_bin = Some(path);
    }
    if let Some(path) = normalize_non_empty(acp.gemini_cli_bin.as_deref()) {
        loaded.effective.acp_gemini_cli_bin = Some(path);
    }
    if let Some(path) = normalize_non_empty(acp.claude_cli_bin.as_deref()) {
        loaded.effective.acp_claude_cli_bin = Some(path);
    }
}

fn normalize_non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string)
}

fn read_settings_layer(path: &Path, label: &str) -> Option<SettingsLayer> {
    let raw = match fs::read_to_string(path) {
        Ok(v) => v,
        Err(err) if err.kind() == ErrorKind::NotFound => return None,
        Err(err) => {
            eprintln!(
                "[settings] failed to read {label} settings {}: {err}",
                path.display()
            );
            return None;
        }
    };

    if raw.trim().is_empty() {
        return None;
    }

    match serde_json::from_str::<SettingsLayer>(&raw) {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!(
                "[settings] failed to parse {label} settings {}: {err}",
                path.display()
            );
            None
        }
    }
}

fn ensure_child_object<'a>(
    root: &'a mut Map<String, Value>,
    key: &str,
) -> &'a mut Map<String, Value> {
    if !matches!(root.get(key), Some(Value::Object(_))) {
        root.insert(key.to_string(), Value::Object(Map::new()));
    }

    root.get_mut(key)
        .and_then(Value::as_object_mut)
        .expect("object just inserted")
}

fn read_settings_object(path: &Path) -> Result<Map<String, Value>, UserSettingsError> {
    let raw = match fs::read_to_string(path) {
        Ok(v) => v,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(Map::new()),
        Err(source) => {
            return Err(UserSettingsError::ReadFile {
                path: path.to_path_buf(),
                source,
            });
        }
    };

    if raw.trim().is_empty() {
        return Ok(Map::new());
    }

    let parsed: Value =
        serde_json::from_str(&raw).map_err(|source| UserSettingsError::ParseFile {
            path: path.to_path_buf(),
            source,
        })?;

    Ok(match parsed {
        Value::Object(map) => map,
        _ => Map::new(),
    })
}

fn update_settings_file(
    path: &Path,
    mutator: impl FnOnce(&mut Map<String, Value>),
) -> Result<(), UserSettingsError> {
    let mut root = read_settings_object(path)?;
    mutator(&mut root);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| UserSettingsError::CreateSettingsDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let json = serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|source| UserSettingsError::EncodeSettingsJson { source })?;

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tmp_name = format!(
        "{}.tmp.{stamp}",
        path.file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("settings.json")
    );
    let tmp_path = path.with_file_name(tmp_name);

    fs::write(&tmp_path, json.as_bytes()).map_err(|source| {
        UserSettingsError::WriteTempSettings {
            path: tmp_path.clone(),
            source,
        }
    })?;

    if let Err(rename_err) = fs::rename(&tmp_path, path) {
        if path.exists() {
            let _ = fs::remove_file(path);
        }
        fs::rename(&tmp_path, path).map_err(|retry_err| UserSettingsError::ReplaceSettings {
            path: path.to_path_buf(),
            rename_error: rename_err.to_string(),
            retry_error: retry_err.to_string(),
        })?;
    }

    Ok(())
}
