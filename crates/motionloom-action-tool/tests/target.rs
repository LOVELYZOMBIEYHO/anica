// =========================================
// =========================================
// crates/motionloom-action-tool/tests/target.rs

//! Analytic joint motion, not a redistributed motion-capture asset.
use motionloom_action_tool::{ConvertOptions, convert_animation_file, target::TargetOptions};
use serde_json::json;
use std::{fs, path::Path, process::Command};

const BONES: [(&str, &str); 7] = [
    ("source:Hips", "hips"),
    ("source:LeftUpLeg", "upper_leg_l"),
    ("source:LeftLeg", "lower_leg_l"),
    ("source:LeftFoot", "foot_l"),
    ("source:RightUpLeg", "upper_leg_r"),
    ("source:RightLeg", "lower_leg_r"),
    ("source:RightFoot", "foot_r"),
];

fn fixture(dir: &Path) -> TargetOptions {
    let r = 14.5_f32.to_radians() * 0.5;
    let mut nodes = vec![
        json!({"name":BONES[0].0,"translation":[0,1,0],"rotation":[r.sin(),0,0,r.cos()],"children":[1,4]}),
        json!({"name":BONES[1].0,"translation":[-0.15,0,0],"children":[2]}),
        json!({"name":BONES[2].0,"translation":[0,-0.4,0],"children":[3]}),
        json!({"name":BONES[3].0,"translation":[0,-0.4,0]}),
        json!({"name":BONES[4].0,"translation":[0.15,0,0],"children":[5]}),
        json!({"name":BONES[5].0,"translation":[0,-0.4,0],"children":[6]}),
        json!({"name":BONES[6].0,"translation":[0,-0.4,0]}),
    ];
    nodes.push(json!({"name":"mesh","mesh":0}));
    let mut values = vec![-0.5_f32, 0., 0., 0.5, 0., 0., 0., 1.8, 0.];
    values.extend([0., 1.]);
    values.extend([0., 1., 0., 0., 0.8, -2.]);
    values.extend([0., 0., 0., 1., -0.5, 0., 0., 3_f32.sqrt() / 2.]);
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let mut doc = json!({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,7]}],"nodes":nodes,
        "buffers":[{"byteLength":bytes.len()}],
        "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":8},{"buffer":0,"byteOffset":44,"byteLength":24},{"buffer":0,"byteOffset":68,"byteLength":32}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-0.5,0,0],"max":[0.5,1.8,0]},
            {"bufferView":1,"componentType":5126,"count":2,"type":"SCALAR","min":[0],"max":[1]},
            {"bufferView":2,"componentType":5126,"count":2,"type":"VEC3"},{"bufferView":3,"componentType":5126,"count":2,"type":"VEC4"}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]});
    write_glb(&dir.join("target.glb"), &doc, &bytes);
    doc["animations"] = json!([{"name":"Analytic","samplers":[{"input":1,"output":2,"interpolation":"LINEAR"},{"input":1,"output":3,"interpolation":"LINEAR"}],"channels":[{"sampler":0,"target":{"node":0,"path":"translation"}},{"sampler":1,"target":{"node":5,"path":"rotation"}}]}]);
    write_glb(&dir.join("source.glb"), &doc, &bytes);
    let maps = BONES
        .iter()
        .map(|(from, to)| format!("<Map from=\"{from}\" to=\"{to}\" />\n"))
        .collect::<String>();
    let profile = format!(
        "<!-- unrelated template need not have loaded its actions -->\n<ModelProfile id=\"test\" kind=\"3d\" preset=\"humanoid_v1\"><Retarget preset=\"humanoid_v1\">{maps}</Retarget><BoneAxisMap><Axis bone=\"lower_leg_r\" bend=\"rotationX:1\" restBend=\"5\" /></BoneAxisMap></ModelProfile>\n<ApplyAction action=\"not_loaded\" />"
    );
    fs::write(
        dir.join("profile.motionloom"),
        profile.replace("><", ">\n<"),
    )
    .unwrap();
    let mut target = TargetOptions::new(
        dir.join("target.glb"),
        dir.join("profile.motionloom"),
        "test".into(),
    );
    target.actor_height = 1.8;
    target.proportional = false;
    target
}

fn write_glb(path: &Path, doc: &serde_json::Value, bytes: &[u8]) {
    let mut json = serde_json::to_vec(doc).unwrap();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let mut file = b"glTF".to_vec();
    file.extend(2_u32.to_le_bytes());
    file.extend(((28 + json.len() + bytes.len()) as u32).to_le_bytes());
    file.extend((json.len() as u32).to_le_bytes());
    file.extend(b"JSON");
    file.extend(json);
    file.extend((bytes.len() as u32).to_le_bytes());
    file.extend(b"BIN\0");
    file.extend(bytes);
    fs::write(path, file).unwrap();
}

fn command(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_motionloom-action-tool"));
    cmd.arg("convert")
        .arg(dir.join("source.glb"))
        .arg("--target-model")
        .arg(dir.join("target.glb"))
        .arg("--target-profile")
        .arg(dir.join("profile.motionloom"))
        .args([
            "--target-profile-id",
            "test",
            "--target-height",
            "1.8",
            "--motion-scale",
            "preserve",
        ]);
    cmd
}

#[test]
fn analytic_same_rig_translation_and_knee_are_reconstructed() {
    let dir = tempfile::tempdir().unwrap();
    let target = fixture(dir.path());
    let out = convert_animation_file(
        dir.path().join("source.glb"),
        &ConvertOptions {
            target: Some(target),
            ..Default::default()
        },
    )
    .unwrap();
    let report = out.fidelity.unwrap();
    assert!(
        report.tool_reconstruction.max_position_mm < 0.01,
        "{report:?}"
    );
    assert!(report.tool_reconstruction.max_rotation_deg < 0.001);
    assert!(!report.strict_pass);
    assert_eq!(report.native_runtime, "passed", "{report:?}");
    assert!(
        report.native_reconstruction.max_position_mm < 0.02,
        "{report:?}"
    );
    let trajectory = report.joint_trajectory_audit.as_ref().unwrap();
    assert_eq!(trajectory.sample_hz, 120);
    assert!(trajectory.effectors.iter().any(|e| e.bone == "foot_l"));
    assert!(
        trajectory
            .interpretation
            .contains("not proof of floor contact")
    );
    let script = format!(
        "<Graph fps={{30}} duration=\"1s\" size={{[64,64]}}>\n{}\n<Tex id=\"out\" fmt=\"rgba8unorm\" size={{[64,64]}} /><Present from=\"out\" /></Graph>",
        out.dsl
    );
    let graph = motionloom::parse_graph_script(&script.replace("><", ">\n<")).unwrap();
    let pose = graph.actions[0].poses.last().unwrap();
    let hip = pose.bones.iter().find(|b| b.id == "hips").unwrap();
    let y: f32 = hip.y.as_ref().unwrap().parse().unwrap();
    let z: f32 = hip.z.as_ref().unwrap().parse().unwrap();
    let r = 14.5_f32.to_radians();
    // Analytic expectation: [0, 1, 0] -> [0, .8, -2], never upward.
    assert!((1. + r.cos() * y - r.sin() * z - 0.8).abs() < 1e-5);
    assert!((r.sin() * y + r.cos() * z + 2.).abs() < 1e-5);
    let knee = pose.bones.iter().find(|b| b.id == "lower_leg_r").unwrap();
    let bend: f32 = knee.bend.as_ref().unwrap().parse().unwrap();
    assert!((bend + 5. + 60.).abs() < 0.001, "{bend}");
}

#[test]
fn diagnostics_are_read_only_and_reject_invalid_sampling() {
    let dir = tempfile::tempdir().unwrap();
    let target = fixture(dir.path());
    let converted = convert_animation_file(
        dir.path().join("source.glb"),
        &ConvertOptions {
            target: Some(target.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let bundle: serde_json::Value = serde_json::from_str(
        &motionloom_action_tool::target::validation_bundle(&target, &converted).unwrap(),
    )
    .unwrap();
    let graph =
        motionloom::parse_world_graph_script(bundle["world_dsl"].as_str().unwrap()).unwrap();
    let before = graph.clone();
    let mesh = motionloom::load_glb_mesh_data(&target.model).unwrap();
    let time = motionloom::WorldTime {
        frame: 15,
        fps: 30.,
        duration_ms: 1000,
    };
    let a =
        motionloom::experimental::diagnose_world_actor_pose(&graph, &mesh, "actor", time).unwrap();
    let b =
        motionloom::experimental::diagnose_world_actor_pose(&graph, &mesh, "actor", time).unwrap();
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
    assert_eq!(graph, before);
    assert!(
        a.joints
            .iter()
            .any(|j| j.canonical_bone.as_deref() == Some("lower_leg_r"))
    );
    assert!(
        motionloom::experimental::diagnose_world_actor_pose(
            &graph,
            &mesh,
            "actor",
            motionloom::WorldTime {
                fps: f32::NAN,
                ..time
            }
        )
        .is_err()
    );
    assert!(
        motionloom::experimental::diagnose_world_actor_pose(&graph, &mesh, "absent", time).is_err()
    );
}

#[test]
fn target_default_times_survive_editor_millisecond_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let mut target = fixture(dir.path());
    target.editor_safe = true;
    let converted = convert_animation_file(
        dir.path().join("source.glb"),
        &ConvertOptions {
            target: Some(target),
            ..Default::default()
        },
    )
    .unwrap();
    let report = converted.fidelity.as_ref().unwrap();
    assert_eq!(report.time_grid, "milliseconds");
    assert_eq!(report.editor_millisecond_collisions, 0);
}

#[test]
fn certification_requires_hash_matched_wasm_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let target = fixture(dir.path());
    let converted = convert_animation_file(
        dir.path().join("source.glb"),
        &ConvertOptions {
            target: Some(target),
            ..Default::default()
        },
    )
    .unwrap();
    let mut source_report = converted.fidelity.clone().unwrap();
    source_report.source_reference = "passed".into();
    let report = serde_json::to_vec(&source_report).unwrap();
    let good=serde_json::to_vec(&json!({"passed":true,"action_sha256":converted.fidelity.as_ref().unwrap().action_sha256,"target_sha256":converted.fidelity.as_ref().unwrap().target_sha256})).unwrap();
    let certified = motionloom_action_tool::target::certify_report(&report, &good).unwrap();
    assert!(certified.strict_pass);
    let bad = serde_json::to_vec(
        &json!({"passed":true,"action_sha256":"wrong","target_sha256":certified.target_sha256}),
    )
    .unwrap();
    assert!(
        !motionloom_action_tool::target::certify_report(&report, &bad)
            .unwrap()
            .strict_pass
    );
}

#[test]
fn strict_failure_writes_report_but_preserves_existing_action() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let output = dir.path().join("good.motionloom");
    fs::write(&output, "KEEP THIS ACTION").unwrap();
    let report = dir.path().join("report.json");
    let result = command(dir.path())
        .args(["--strict-fidelity", "--force"])
        .arg("--output")
        .arg(&output)
        .arg("--report")
        .arg(&report)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("strict fidelity NOT verified"));
    assert_eq!(fs::read_to_string(output).unwrap(), "KEEP THIS ACTION");
    let report: serde_json::Value = serde_json::from_slice(&fs::read(report).unwrap()).unwrap();
    assert_eq!(report["strict_pass"], false);
}

#[test]
fn output_cannot_replace_source_even_with_force() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let source = dir.path().join("source.glb");
    let before = fs::read(&source).unwrap();
    let result = command(dir.path())
        .arg("--force")
        .arg("--output")
        .arg(&source)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(before, fs::read(source).unwrap());
}

#[test]
fn profile_fingerprint_changes_and_duplicate_mapping_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let target = fixture(dir.path());
    let options = ConvertOptions {
        target: Some(target.clone()),
        ..Default::default()
    };
    let first = convert_animation_file(dir.path().join("source.glb"), &options).unwrap();
    let text = fs::read_to_string(&target.profile).unwrap();
    fs::write(&target.profile, format!("{text}\n<!-- revision -->")).unwrap();
    let second = convert_animation_file(dir.path().join("source.glb"), &options).unwrap();
    assert_ne!(
        first.fidelity.unwrap().profile_sha256,
        second.fidelity.unwrap().profile_sha256
    );
    fs::write(
        &target.profile,
        text.replace(
            "</Retarget>",
            "<Map from=\"source:Hips\" to=\"hips\" /></Retarget>",
        ),
    )
    .unwrap();
    assert!(convert_animation_file(dir.path().join("source.glb"), &options).is_err());
}

#[test]
fn target_mode_does_not_silently_enable_foot_locks() {
    let dir = tempfile::tempdir().unwrap();
    let target = fixture(dir.path());
    let result = convert_animation_file(
        dir.path().join("source.glb"),
        &ConvertOptions {
            target: Some(target),
            detect_contacts: true,
            ..Default::default()
        },
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not automatically add contacts")
    );
}
