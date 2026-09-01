// =========================================
// =========================================
// crates/motionloom-action-tool/src/blender_source.rs

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::ActionToolError;
use crate::source::AnimationSource;

pub(crate) fn load(path: &Path) -> Result<AnimationSource, ActionToolError> {
    let blender = find_blender().ok_or(ActionToolError::BlenderNotFound)?;
    let temp = tempfile::tempdir().map_err(ActionToolError::TemporaryDirectory)?;
    let output = temp.path().join("motionloom-action-import.glb");
    // Pass file names as argv, never executable Python source. Factory startup
    // avoids user startup files and auto-run scripts in this offline subprocess.
    let expression = "import bpy, sys; paths=sys.argv[sys.argv.index('--')+1:]; bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False); bpy.context.scene.render.fps=120; bpy.ops.import_scene.fbx(filepath=paths[0], anim_offset=0); bpy.ops.export_scene.gltf(filepath=paths[1], export_format='GLB', export_animations=True, export_force_sampling=True, export_frame_range=False)";
    let result = Command::new(&blender)
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python-expr",
            expression,
            "--",
        ])
        .arg(path)
        .arg(&output)
        .output()
        .map_err(|source| ActionToolError::BlenderLaunch {
            executable: blender.clone(),
            source,
        })?;
    if !result.status.success() {
        return Err(ActionToolError::BlenderConversion {
            status: result.status.code(),
            stderr: format!(
                "{}\n{}",
                String::from_utf8_lossy(&result.stderr),
                String::from_utf8_lossy(&result.stdout)
            )
            .trim()
            .to_string(),
        });
    }
    let mut source = crate::gltf_source::load(&output)?;
    source.path = path.to_path_buf();
    source.backend = "fbx-blender".to_string();
    source.diagnostics.push(format!(
        "FBX was converted through Blender {} before canonical Action sampling",
        blender.display()
    ));
    Ok(source)
}

fn find_blender() -> Option<PathBuf> {
    if let Some(path) = env::var_os("BLENDER") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let candidates = if cfg!(target_os = "macos") {
        vec![PathBuf::from(
            "/Applications/Blender.app/Contents/MacOS/Blender",
        )]
    } else if cfg!(target_os = "windows") {
        vec![PathBuf::from(
            r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        )]
    } else {
        vec![
            PathBuf::from("/usr/bin/blender"),
            PathBuf::from("/usr/local/bin/blender"),
        ]
    };
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .or_else(|| {
            env::var_os("PATH").and_then(|paths| {
                env::split_paths(&paths)
                    .map(|directory| {
                        directory.join(if cfg!(target_os = "windows") {
                            "blender.exe"
                        } else {
                            "blender"
                        })
                    })
                    .find(|path| path.is_file())
            })
        })
}
