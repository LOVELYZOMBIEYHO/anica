// =========================================
// =========================================
// crates/motionloom-action-tool/src/main.rs

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use motionloom_action_tool::target::TargetOptions;
use motionloom_action_tool::{
    ConvertOptions, FbxBackend, convert_animation_file, inspect_animation_file,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("motionloom-action-tool: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        return Ok(());
    };
    match command.as_str() {
        "certify" => {
            let report = PathBuf::from(args.next().ok_or("certify requires report.json")?);
            let wasm = PathBuf::from(args.next().ok_or("certify requires wasm-evidence.json")?);
            let output = PathBuf::from(args.next().ok_or("certify requires output.json")?);
            if args.next().is_some() {
                return Err("unexpected certify argument".into());
            }
            if output.exists() {
                return Err("certified report already exists".into());
            }
            for input in [&report, &wasm] {
                if fs::canonicalize(input)? == resolve_destination(&output)? {
                    return Err("certified report must not overwrite evidence".into());
                }
            }
            let certified = motionloom_action_tool::target::certify_report(
                &fs::read(report)?,
                &fs::read(wasm)?,
            )?;
            atomic_write(&output, &serde_json::to_vec_pretty(&certified)?, false)?;
            if !certified.strict_pass {
                return Err("certification failed; inspect the output report".into());
            }
            println!(
                "Certified source/native/WASM Action fidelity: {}",
                output.display()
            );
        }
        "audit-source" => {
            let path = args.next().ok_or("audit-source requires an FBX path")?;
            if args.next().is_some() {
                return Err("unexpected audit-source argument".into());
            }
            let audit =
                motionloom_action_tool::target::audit_fbx_source(std::path::Path::new(&path))?;
            println!("{}", serde_json::to_string_pretty(&audit)?);
        }
        "inspect" => {
            let Some(path) = args.next() else {
                return Err("inspect requires an animated .fbx/.glb/.gltf path".into());
            };
            let mut backend = FbxBackend::Auto;
            let remaining = args.collect::<Vec<_>>();
            let mut index = 0;
            while index < remaining.len() {
                match remaining[index].as_str() {
                    "--fbx-backend" => {
                        backend = remaining
                            .get(index + 1)
                            .ok_or("--fbx-backend requires auto, native, or blender")?
                            .parse()?;
                        index += 2;
                    }
                    option => return Err(format!("unknown inspect option: {option}").into()),
                }
            }
            let report = inspect_animation_file(&path, backend)?;
            println!("Asset: {}", report.path.display());
            println!("Backend: {}", report.backend);
            for clip in report.clips {
                println!(
                    "Clip: {} · {:.3}s · {} channels",
                    clip.name, clip.duration_sec, clip.channel_count
                );
            }
            println!("Canonical mappings: {}", report.mapped_bones.len());
            for mapping in report.mapped_bones {
                println!("  {} -> {}", mapping.source, mapping.canonical);
            }
            if !report.unmapped_joints.is_empty() {
                println!("Unmapped joints: {}", report.unmapped_joints.join(", "));
            }
            for diagnostic in report.diagnostics {
                println!("Diagnostic: {diagnostic}");
            }
        }
        "convert" => {
            let Some(path) = args.next() else {
                return Err("convert requires an animated .fbx/.glb/.gltf path".into());
            };
            let mut options = ConvertOptions::default();
            let mut output = None::<PathBuf>;
            let mut force = false;
            let mut strict = false;
            let mut report_path = None::<PathBuf>;
            let mut bundle_path = None::<PathBuf>;
            let mut target_model = None::<PathBuf>;
            let mut target_profile = None::<PathBuf>;
            let mut target_profile_id = None::<String>;
            let mut actor_height = 1.82;
            let mut proportional = true;
            let mut max_position_mm = 1.0;
            let mut max_rotation_deg = 0.1;
            let mut editor_safe = false;
            let remaining = args.collect::<Vec<_>>();
            let mut index = 0usize;
            while index < remaining.len() {
                let flag = &remaining[index];
                if flag == "--strict-fidelity" {
                    strict = true;
                    index += 1;
                    continue;
                }
                if flag == "--force" {
                    force = true;
                    index += 1;
                    continue;
                }
                if flag == "--detect-contacts" {
                    options.detect_contacts = true;
                    index += 1;
                    continue;
                }
                let value = remaining
                    .get(index + 1)
                    .ok_or_else(|| format!("{flag} requires a value"))?;
                match flag.as_str() {
                    "--clip" => options.clip = Some(value.clone()),
                    "--source-profile" => options.source_profile = value.clone(),
                    "--action-id" => options.action_id = value.clone(),
                    "--fps" => options.fps = value.parse()?,
                    "--fbx-backend" => options.fbx_backend = value.parse()?,
                    "--key-reduction-tolerance" => {
                        options.key_reduction_tolerance = value.parse()?
                    }
                    "--output" => output = Some(PathBuf::from(value)),
                    "--report" => report_path = Some(PathBuf::from(value)),
                    "--validation-bundle" => bundle_path = Some(PathBuf::from(value)),
                    "--target-model" => target_model = Some(PathBuf::from(value)),
                    "--target-profile" => target_profile = Some(PathBuf::from(value)),
                    "--target-profile-id" => target_profile_id = Some(value.clone()),
                    "--target-height" => actor_height = value.parse()?,
                    "--time-grid" => {
                        editor_safe = match value.as_str() {
                            "milliseconds" => true,
                            "subframes" => false,
                            _ => return Err("--time-grid expects milliseconds or subframes".into()),
                        }
                    }
                    "--max-position-mm" => max_position_mm = value.parse()?,
                    "--max-rotation-deg" => max_rotation_deg = value.parse()?,
                    "--motion-scale" => {
                        proportional = match value.as_str() {
                            "proportional" => true,
                            "preserve" => false,
                            _ => {
                                return Err(
                                    "--motion-scale expects proportional or preserve".into()
                                );
                            }
                        }
                    }
                    _ => return Err(format!("unknown convert option: {flag}").into()),
                }
                index += 2;
            }
            let output = output.ok_or("convert requires --output <file.motionloom>")?;
            let target_requested =
                target_model.is_some() || target_profile.is_some() || target_profile_id.is_some();
            if target_requested {
                let mut target = TargetOptions::new(
                    target_model.ok_or("--target-model required")?,
                    target_profile.ok_or("--target-profile required")?,
                    target_profile_id.ok_or("--target-profile-id required")?,
                );
                target.actor_height = actor_height;
                target.proportional = proportional;
                target.max_position_mm = max_position_mm;
                target.max_rotation_deg = max_rotation_deg;
                target.editor_safe = editor_safe;
                options.target = Some(target);
            } else if strict || report_path.is_some() || bundle_path.is_some() {
                return Err("--strict-fidelity/--report require a target rig".into());
            }
            // Validate all destinations before conversion or opening any output file.
            let mut inputs = vec![PathBuf::from(&path)];
            if let Some(target) = &options.target {
                inputs.extend([target.model.clone(), target.profile.clone()]);
            }
            let output_resolved = resolve_destination(&output)?;
            for input in &inputs {
                if fs::canonicalize(input)? == output_resolved {
                    return Err("output must not overwrite an input file".into());
                }
            }
            if let Some(report) = &report_path {
                let report_resolved = resolve_destination(report)?;
                if report_resolved == output_resolved
                    || inputs
                        .iter()
                        .any(|p| fs::canonicalize(p).ok().as_ref() == Some(&report_resolved))
                {
                    return Err("report path collides with input/output".into());
                }
                if report.exists() && !force {
                    return Err("report exists; use --force to replace it".into());
                }
            }
            if output.exists() && !force {
                return Err("output exists; use --force to replace it".into());
            }
            if let Some(bundle) = &bundle_path {
                let dest = resolve_destination(bundle)?;
                if dest == output_resolved
                    || inputs
                        .iter()
                        .any(|p| fs::canonicalize(p).ok().as_ref() == Some(&dest))
                    || report_path
                        .as_ref()
                        .is_some_and(|p| resolve_destination(p).ok().as_ref() == Some(&dest))
                {
                    return Err("validation bundle collides with input/output/report".into());
                }
                if bundle.exists() && !force {
                    return Err("validation bundle exists; use --force to replace it".into());
                }
            }
            let converted = convert_animation_file(&path, &options)?;
            if let (Some(path), Some(target)) = (&bundle_path, &options.target) {
                let bundle = motionloom_action_tool::target::validation_bundle(target, &converted)?;
                atomic_write(path, bundle.as_bytes(), force)?;
            }
            if let Some(report) = &converted.fidelity {
                if let Some(path) = &report_path {
                    atomic_write(path, &serde_json::to_vec_pretty(report)?, force)?;
                }
                if strict && !report.strict_pass {
                    return Err("strict fidelity NOT verified: independent source/native/WASM evidence is missing or failed. Action output was not written; inspect --report.".into());
                }
            }
            atomic_write(&output, converted.dsl.as_bytes(), force)?;
            println!(
                "Wrote {} poses ({} sampled) from '{}' to {}",
                converted.pose_count,
                converted.sampled_pose_count,
                converted.clip_name,
                output.display()
            );
            for diagnostic in converted.diagnostics {
                println!("Diagnostic: {diagnostic}");
            }
        }
        "help" | "--help" | "-h" => print_help(),
        _ => return Err(format!("unknown command: {command}").into()),
    }
    Ok(())
}

fn print_help() {
    println!(
        r#"MotionLoom offline Action authoring tool

Usage:
  motionloom-action-tool audit-source <animation.fbx>

  motionloom-action-tool certify <report.json> <wasm-evidence.json> <output.json>

  motionloom-action-tool inspect <animation.fbx|glb|gltf>
    [--fbx-backend auto|native|blender]

  motionloom-action-tool convert <animation.fbx|glb|gltf>
    [--clip <name>] --source-profile fbx_humanoid
    --action-id <id> --fps 30
    [--fbx-backend auto|native|blender]
    [--key-reduction-tolerance <degrees-or-mm>] [--detect-contacts]
    [--target-model <character.glb> --target-profile <scene.motionloom>
     --target-profile-id <id> --target-height 1.82]
    [--motion-scale proportional|preserve]
    [--max-position-mm 1 --max-rotation-deg 0.1]
    [--time-grid milliseconds|subframes]
    [--report <report.json>] [--strict-fidelity]
    [--validation-bundle <native-snapshots.json>]
    [--force] --output <action.motionloom>

Auto/native never launch Blender. Select blender explicitly to opt in.
Target mode writes a target-bound candidate, not a universally retargeted Action.
Strict mode refuses output until independent source AND runtime fidelity are verified."#
    );
}

fn resolve_destination(path: &std::path::Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if path.exists() {
        return Ok(fs::canonicalize(path)?);
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    Ok(fs::canonicalize(parent)?.join(path.file_name().ok_or("output has no file name")?))
}

// Publish a complete file atomically; failed conversion never truncates a good Action.
fn atomic_write(
    path: &std::path::Path,
    bytes: &[u8],
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    if force {
        temp.persist(path)?;
    } else {
        temp.persist_noclobber(path)?;
    }
    Ok(())
}
