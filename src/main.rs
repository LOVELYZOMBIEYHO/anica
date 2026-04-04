// =========================================
// =========================================
// src/main.rs
use gpui::{App, Application, AssetSource, SharedString};
use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::fs;
use std::path::PathBuf;

mod api;
mod app;
mod core;
mod runtime_paths;
mod ui;

use crate::core::media_tools::{
    configure_bundled_media_runtime_environment, detect_gstreamer_cli,
    detect_or_bootstrap_media_dependencies,
};

/// Install a panic hook so unexpected crashes always print stack traces to terminal logs.
fn install_panic_logging() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("[Panic] {info}");
        eprintln!("[Panic] backtrace:\n{}", Backtrace::force_capture());
    }));
}

/// Check GStreamer CLI tool (As an indicator of library presence)
fn check_gstreamer() -> Option<String> {
    if let Some(candidate) = detect_gstreamer_cli(None) {
        println!("[System Check] ✅ Found GStreamer CLI: {candidate}");
        return Some(candidate);
    }

    eprintln!(
        "[System Check] ⚠️ GStreamer CLI not found. Video playback might fail if libraries are missing."
    );
    None
}

fn main() {
    install_panic_logging();
    env_logger::init();
    configure_bundled_media_runtime_environment();
    runtime_paths::apply_common_agent_cli_env_overrides();

    println!("--- Starting Anica Editor ---");

    // Run environment checks
    let media_tools = detect_or_bootstrap_media_dependencies(None);
    let gst_cli = check_gstreamer(); // Warning only, does not block startup

    if media_tools.ffmpeg_available {
        println!(
            "[System Check] Found ffmpeg: {}",
            media_tools.ffmpeg_command
        );
    } else {
        eprintln!("[System Check] ffmpeg not found.");
    }
    if media_tools.ffprobe_available {
        println!(
            "[System Check] Found ffprobe: {}",
            media_tools.ffprobe_command
        );
    } else {
        eprintln!("[System Check] ffprobe not found.");
    }
    let media_tools_for_app = media_tools.clone();

    Application::new()
        .with_assets(Assets {
            base: runtime_paths::resolve_asset_root(),
        })
        .run(move |cx: &mut App| {
            gpui_component::init(cx);
            load_asset_fonts(cx);
            let global = app::editor_window::open_editor_window(cx);
            global.update(cx, |gs, cx| {
                gs.apply_gstreamer_dependency_status(gst_cli.clone());
                gs.apply_media_dependency_status(media_tools_for_app.clone(), true);
                cx.notify();
            });
            app::menu::init_app_menus(cx, global);
        });
}

struct Assets {
    base: PathBuf,
}

impl AssetSource for Assets {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        match fs::read(self.base.join(path)) {
            Ok(data) => Ok(Some(Cow::Owned(data))),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<SharedString>> {
        let mut entries = Vec::new();
        let Ok(dir) = fs::read_dir(self.base.join(path)) else {
            return Ok(entries);
        };
        for entry in dir.flatten() {
            if let Ok(name) = entry.file_name().into_string() {
                entries.push(SharedString::from(name));
            }
        }
        Ok(entries)
    }
}

fn load_asset_fonts(cx: &mut App) {
    let mut fonts: Vec<Cow<'static, [u8]>> = Vec::new();
    for dir in runtime_paths::candidate_font_dirs() {
        if !dir.exists() {
            continue;
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            let ext = ext.to_ascii_lowercase();
            if ext != "otf" && ext != "ttf" && ext != "ttc" {
                continue;
            }
            if let Ok(bytes) = fs::read(&path) {
                fonts.push(Cow::Owned(bytes));
            }
        }
    }

    if !fonts.is_empty()
        && let Err(err) = cx.text_system().add_fonts(fonts)
    {
        eprintln!("[Fonts] Failed to load asset fonts: {err}");
    }
}
