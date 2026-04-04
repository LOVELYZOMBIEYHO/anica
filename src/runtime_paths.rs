#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if path.as_os_str().is_empty() || paths.iter().any(|existing| existing == &path) {
        return;
    }
    paths.push(path);
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn search_path(bin: &str) -> Option<PathBuf> {
    if bin.is_empty() {
        return None;
    }

    let path = PathBuf::from(bin);
    if path.components().count() > 1 {
        return path.is_file().then_some(path);
    }

    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            return Some(candidate);
        }

        if cfg!(windows) {
            for ext in [".exe", ".bat", ".cmd"] {
                let with_ext = dir.join(format!("{bin}{ext}"));
                if with_ext.is_file() {
                    return Some(with_ext);
                }
            }
        }
    }

    None
}

fn candidate_nvm_bins(bin: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Some(home) = home_dir() else {
        return out;
    };

    let versions_dir = home.join(".nvm").join("versions").join("node");
    let Ok(entries) = fs::read_dir(versions_dir) else {
        return out;
    };

    let mut version_dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    version_dirs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    for version_dir in version_dirs {
        let candidate = version_dir.join("bin").join(bin);
        if candidate.is_file() {
            push_unique(&mut out, candidate);
        }
    }

    out
}

pub fn candidate_cli_bins(bin: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();

    if let Some(path_hit) = search_path(bin) {
        push_unique(&mut out, path_hit);
    }

    if let Some(home) = home_dir() {
        for candidate in [
            home.join(".local").join("bin").join(bin),
            home.join(".npm-global").join("bin").join(bin),
            home.join(".volta").join("bin").join(bin),
            home.join("bin").join(bin),
        ] {
            if candidate.is_file() {
                push_unique(&mut out, candidate);
            }
        }
    }

    for candidate in [
        PathBuf::from("/opt/homebrew/bin").join(bin),
        PathBuf::from("/usr/local/bin").join(bin),
    ] {
        if candidate.is_file() {
            push_unique(&mut out, candidate);
        }
    }

    for candidate in candidate_nvm_bins(bin) {
        push_unique(&mut out, candidate);
    }

    out
}

pub fn resolve_cli_bin(env_var: &str, bin: &str) -> Option<PathBuf> {
    if let Some(from_env) = std::env::var_os(env_var)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return Some(from_env);
    }

    candidate_cli_bins(bin)
        .into_iter()
        .find(|path| path.is_file())
}

pub fn apply_common_agent_cli_env_overrides() {
    for (env_var, bin) in [
        ("ANICA_CODEX_CLI_BIN", "codex"),
        ("ANICA_GEMINI_CLI_BIN", "gemini"),
        ("ANICA_CLAUDE_CLI_BIN", "claude"),
    ] {
        if std::env::var_os(env_var).is_some() {
            continue;
        }
        if let Some(path) = resolve_cli_bin(env_var, bin) {
            // Startup config runs before worker threads are spawned, so this
            // environment update stays within the documented safety boundary.
            unsafe {
                std::env::set_var(env_var, path);
            }
        }
    }
}

pub fn current_exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(PathBuf::from))
}

pub fn bundle_resources_dir() -> Option<PathBuf> {
    let exe_dir = current_exe_dir()?;
    let exe_name = exe_dir.file_name().and_then(|v| v.to_str());
    if exe_name == Some("Resources") {
        let parent = exe_dir.parent()?;
        if parent.file_name().and_then(|v| v.to_str()) == Some("Contents") {
            return Some(exe_dir);
        }
    }
    if let Some(contents_dir) = exe_dir.parent()
        && contents_dir.file_name().and_then(|v| v.to_str()) == Some("Contents")
    {
        return Some(contents_dir.join("Resources"));
    }
    None
}

pub fn bundle_runtime_root() -> Option<PathBuf> {
    let resources = bundle_resources_dir()?;
    let os = std::env::consts::OS;
    let candidate = resources.join("runtime").join("current").join(os);
    if candidate.is_dir() {
        Some(candidate)
    } else {
        None
    }
}

pub fn candidate_asset_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(from_env) = std::env::var_os("ANICA_ASSETS_DIR")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
    {
        push_unique(&mut roots, from_env);
    }

    if let Some(exe_dir) = current_exe_dir() {
        push_unique(&mut roots, exe_dir.join("assets"));
    }
    if let Some(resources_dir) = bundle_resources_dir() {
        push_unique(&mut roots, resources_dir.join("assets"));
    }

    push_unique(
        &mut roots,
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets"),
    );

    if let Ok(cwd) = std::env::current_dir() {
        push_unique(&mut roots, cwd.join("assets"));
        push_unique(&mut roots, cwd.join("anica").join("assets"));
    }

    roots
}

pub fn resolve_asset_root() -> PathBuf {
    candidate_asset_roots()
        .into_iter()
        .find(|path| path.is_dir())
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets"))
}

pub fn candidate_font_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(from_env) = std::env::var_os("ANICA_FONTS_DIR")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
    {
        push_unique(&mut dirs, from_env);
    }

    for root in candidate_asset_roots() {
        push_unique(&mut dirs, root.join("fonts"));
    }

    dirs
}

pub fn candidate_twemoji_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(from_env) = std::env::var_os("ANICA_TWEMOJI_DIR")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
    {
        push_unique(&mut dirs, from_env);
    }

    for root in candidate_asset_roots() {
        push_unique(&mut dirs, root.join("twemoji").join("72x72"));
    }

    push_unique(
        &mut dirs,
        std::env::temp_dir().join("anica_twemoji").join("72x72"),
    );
    dirs
}

pub fn candidate_docs_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(from_env) = std::env::var_os("ANICA_DOCS_DIR")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
    {
        push_unique(&mut roots, from_env);
    }

    if let Some(exe_dir) = current_exe_dir() {
        push_unique(&mut roots, exe_dir.join("docs"));
    }
    if let Some(resources_dir) = bundle_resources_dir() {
        push_unique(&mut roots, resources_dir.join("docs"));
    }

    push_unique(
        &mut roots,
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs"),
    );
    push_unique(
        &mut roots,
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs"),
    );

    if let Ok(cwd) = std::env::current_dir() {
        push_unique(&mut roots, cwd.join("docs"));
        push_unique(&mut roots, cwd.join("anica").join("docs"));
        if let Some(parent) = cwd.parent() {
            push_unique(&mut roots, parent.join("docs"));
        }
    }

    roots
}
