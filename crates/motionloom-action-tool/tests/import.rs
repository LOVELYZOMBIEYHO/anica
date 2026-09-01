// =========================================
// =========================================
// crates/motionloom-action-tool/tests/import.rs

//! Synthetic sources exercise importers without redistributing commercial clips.

use draco_io::fbx_reader::{FbxNode, FbxProperty};
use motionloom_action_tool::{
    ConvertOptions, FbxBackend, convert_animation_file, inspect_animation_file,
};
use std::{fs, path::Path};

const ASCII: &str = include_str!("fixtures/knee_ascii.fbx");

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/knee_ascii.fbx")
}

#[test]
fn ascii_fbx_without_mesh_exports_standalone_action() {
    let report = inspect_animation_file(fixture(), FbxBackend::Native).unwrap();
    assert_eq!(report.backend, "fbx-native");
    assert_eq!(report.clips[0].duration_sec, 1.0);
    assert_eq!(report.mapped_bones.len(), 3);
    let converted = convert_animation_file(
        fixture(),
        &ConvertOptions {
            fps: 2.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(converted.pose_count, 3);
    assert!(converted.dsl.contains("bend=\"60\""), "{}", converted.dsl);
    assert!(!converted.dsl.contains(".fbx"));
    assert!(!converted.dsl.contains("ModelAsset"));
}

#[test]
fn binary_fbx_and_ascii_produce_identical_actions() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("knee.fbx");
    let mut reader = draco_io::FbxMemoryReader::from_bytes(ASCII.as_bytes()).unwrap();
    let nodes = reader.read_nodes().unwrap();
    let mut bytes = b"Kaydara FBX Binary  \0\x1a\0".to_vec();
    bytes.extend(7400u32.to_le_bytes());
    for node in nodes {
        encode_node(&node, &mut bytes);
    }
    bytes.extend([0u8; 13]);
    fs::write(&path, bytes).unwrap();
    let options = ConvertOptions {
        fbx_backend: FbxBackend::Native,
        ..Default::default()
    };
    assert_eq!(
        convert_animation_file(&path, &options).unwrap().dsl,
        convert_animation_file(fixture(), &options).unwrap().dsl
    );
}

#[test]
fn unsupported_weighted_and_unaligned_axes_are_not_silently_linearized() {
    let temp = tempfile::tempdir().unwrap();
    for (text, expected) in [
        (ASCII.replace("a: 4", "a: 16777224"), "weighted"),
        (
            ASCII.replacen("0,46186158000", "0,40000000000", 1),
            "different timestamps",
        ),
    ] {
        let path = temp.path().join("unsupported.fbx");
        fs::write(&path, text).unwrap();
        let error = inspect_animation_file(path, FbxBackend::Native)
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn invalid_input_and_invalid_cli_options_fail() {
    assert!(inspect_animation_file("motion.obj", FbxBackend::Native).is_err());
    assert!("bogus".parse::<FbxBackend>().is_err());
    for fps in [0.0, -1.0, f32::NAN] {
        assert!(
            convert_animation_file(
                fixture(),
                &ConvertOptions {
                    fps,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    assert!(
        convert_animation_file(
            fixture(),
            &ConvertOptions {
                clip: Some("missing".into()),
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn automatic_backend_never_launches_blender_on_invalid_fbx() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("invalid.fbx");
    fs::write(&path, b"not an FBX").unwrap();
    assert!(matches!(
        inspect_animation_file(path, FbxBackend::Auto),
        Err(motionloom_action_tool::ActionToolError::FbxRead { .. })
    ));
}

#[test]
fn world_front_axis_does_not_reverse_local_knee_flexion() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("front.fbx");
    fs::write(
        &path,
        ASCII.replace(
            "\"FrontAxisSign\", \"int\", \"Integer\", \"\", -1",
            "\"FrontAxisSign\", \"int\", \"Integer\", \"\", 1",
        ),
    )
    .unwrap();
    let action = convert_animation_file(path, &ConvertOptions::default()).unwrap();
    assert!(action.dsl.contains("bend=\"60\""));
}

#[test]
fn full_turn_is_not_lost_during_quaternion_conversion() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("spin.fbx");
    fs::write(&path, ASCII.replace("a: 0,-60", "a: 0,-360")).unwrap();
    let action = convert_animation_file(path, &ConvertOptions::default()).unwrap();
    assert!(action.dsl.contains("bend=\"360\""), "{}", action.dsl);
}

#[test]
fn stationary_foot_contacts_and_reduced_action_round_trip() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("still.fbx");
    fs::write(&path, ASCII.replace("a: 0,-60", "a: 0,0")).unwrap();
    let action = convert_animation_file(
        path,
        &ConvertOptions {
            detect_contacts: true,
            key_reduction_tolerance: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(action.pose_count, 2);
    assert!(
        action
            .dsl
            .contains("effector=\"foot_r\" target=\"ground\" from=\"0\" to=\"1\""),
        "{}",
        action.dsl
    );
}

#[test]
fn glb_and_fbx_single_axis_animation_match() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("knee.glb");
    let mut json=r#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":40}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":8},{"buffer":0,"byteOffset":8,"byteLength":32}],"accessors":[{"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR"},{"bufferView":1,"componentType":5126,"count":2,"type":"VEC4"}],"nodes":[{"name":"source:Hips","translation":[0,1,0],"children":[1]},{"name":"source:RightLeg","translation":[0,-0.4,0],"children":[2]},{"name":"source:RightFoot","translation":[0,-0.4,0]}],"animations":[{"name":"Knee","samplers":[{"input":0,"output":1,"interpolation":"LINEAR"}],"channels":[{"sampler":0,"target":{"node":1,"path":"rotation"}}]}]}"#.as_bytes().to_vec();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let values = [
        0.0f32,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        -0.5,
        0.0,
        0.0,
        (3.0f32).sqrt() / 2.0,
    ];
    let mut glb = b"glTF".to_vec();
    glb.extend(2u32.to_le_bytes());
    glb.extend(((12 + 8 + json.len() + 8 + 40) as u32).to_le_bytes());
    glb.extend((json.len() as u32).to_le_bytes());
    glb.extend(b"JSON");
    glb.extend(json);
    glb.extend(40u32.to_le_bytes());
    glb.extend(b"BIN\0");
    for v in values {
        glb.extend(v.to_le_bytes());
    }
    fs::write(&path, glb).unwrap();
    let options = ConvertOptions {
        fps: 2.0,
        ..Default::default()
    };
    let a = convert_animation_file(&path, &options).unwrap();
    let b = convert_animation_file(fixture(), &options).unwrap();
    assert_eq!(a.dsl, b.dsl);
}

/// Minimal fixture encoder: tests the binary reader without enabling a writer
/// dependency, compressing data, or checking in an opaque binary asset.
fn encode_node(node: &FbxNode, bytes: &mut Vec<u8>) {
    let start = bytes.len();
    bytes.extend([0u8; 13]);
    bytes.extend(node.name.as_bytes());
    let property_start = bytes.len();
    for value in &node.properties {
        match value {
            FbxProperty::I32(v) => {
                bytes.push(b'I');
                bytes.extend(v.to_le_bytes());
            }
            FbxProperty::I64(v) => {
                bytes.push(b'L');
                bytes.extend(v.to_le_bytes());
            }
            FbxProperty::F32(v) => {
                bytes.push(b'F');
                bytes.extend(v.to_le_bytes());
            }
            FbxProperty::F64(v) => {
                bytes.push(b'D');
                bytes.extend(v.to_le_bytes());
            }
            FbxProperty::String(v) => {
                bytes.push(b'S');
                bytes.extend((v.len() as u32).to_le_bytes());
                bytes.extend(v.as_bytes());
            }
            FbxProperty::I64Array(v) => array(
                bytes,
                b'l',
                v.len(),
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            FbxProperty::I32Array(v) => array(
                bytes,
                b'i',
                v.len(),
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            FbxProperty::F32Array(v) => array(
                bytes,
                b'f',
                v.len(),
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            FbxProperty::F64Array(v) => array(
                bytes,
                b'd',
                v.len(),
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            other => panic!("unexpected synthetic property {other:?}"),
        }
    }
    let property_len = bytes.len() - property_start;
    for child in &node.children {
        encode_node(child, bytes);
    }
    if !node.children.is_empty() {
        bytes.extend([0u8; 13]);
    }
    let end = bytes.len() as u32;
    bytes[start..start + 4].copy_from_slice(&end.to_le_bytes());
    bytes[start + 4..start + 8].copy_from_slice(&(node.properties.len() as u32).to_le_bytes());
    bytes[start + 8..start + 12].copy_from_slice(&(property_len as u32).to_le_bytes());
    bytes[start + 12] = node.name.len() as u8;
}

fn array(bytes: &mut Vec<u8>, kind: u8, count: usize, data: Vec<u8>) {
    bytes.push(kind);
    bytes.extend((count as u32).to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend((data.len() as u32).to_le_bytes());
    bytes.extend(data);
}
