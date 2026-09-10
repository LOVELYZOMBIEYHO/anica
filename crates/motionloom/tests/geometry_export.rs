// =========================================
// =========================================
// crates/motionloom/tests/geometry_export.rs

#![cfg(not(target_arch = "wasm32"))]
use motionloom::{experimental::*, parse_graph_script};

fn source() -> String {
    r##"<Graph fps="30" duration="2s" size={[128,128]}>
<Assets>
<MaterialAsset id="paint" baseColor="#AF7050" roughness="0.6" />
<PrimitiveAsset id="box" shape="box" size={[1,2,3]} material="paint" />
</Assets>
<Scene id="main">
<Timeline>
<Track space="3d">
<Sequence duration="2s">
<CompositeGroup space="3d" depth="true">
<Camera3D id="camera" position={[0,0,4]} target={[0,0,0]} />
<Model id="subject" asset="box" position={[3,2,1]} />
</CompositeGroup>
</Sequence>
</Track>
</Timeline>
</Scene>
<AnimationTarget node="camera" property="position">
<Key time="0s" value={[0,0,4]} />
<Key time="2s" value={[20,20,-4]} />
</AnimationTarget>
<Present from="main" />
</Graph>"##
        .into()
}

fn mesh_source(subdivision: u32) -> String {
    format!(
        r##"<Graph fps="30" duration="1s" size={{[64,64]}}>
<Assets>
<MaterialAsset id="paint" baseColor="#AF7050" roughness="0.6" />
<MeshAsset id="mesh" material="paint" subdivision="{subdivision}" subdivisionScheme="catmullClark">
<Vertex position={{[-1,-1,0]}} uv={{[0,0]}} pinned="true" />
<Vertex position={{[1,-1,0]}} uv={{[1,0]}} pinned="true" />
<Vertex position={{[1,1,0]}} uv={{[1,1]}} />
<Vertex position={{[-1,1,0]}} uv={{[0,1]}} />
<Face indices={{[0,1,2,3]}} />
</MeshAsset>
</Assets>
<Scene id="main">
<Timeline>
<Track space="3d">
<Sequence duration="1s">
<CompositeGroup space="3d" depth="true">
<Camera3D position={{[0,0,4]}} target={{[0,0,0]}} />
<Model asset="mesh" />
</CompositeGroup>
</Sequence>
</Track>
</Timeline>
</Scene>
<Present from="main" />
</Graph>"##
    )
}

fn snapshot(source: &str, frame: u32) -> GeometrySnapshot {
    pollster::block_on(extract_scene_geometry(
        &parse_graph_script(source).unwrap(),
        &SceneGeometryOptions {
            scene_id: "main".into(),
            frame,
            include_hidden: false,
        },
    ))
    .unwrap()
}

#[test]
fn camera_animation_cannot_change_geometry_or_export() {
    let script = source();
    let a = snapshot(&script, 0);
    let b = snapshot(&script, 30);
    assert_eq!(a.topology_signature, b.topology_signature);
    assert_eq!(a.meshes[0].positions, b.meshes[0].positions);
    assert_eq!(export_scene_glb(&a).unwrap(), export_scene_glb(&b).unwrap());
    assert!(script.contains("<Camera3D"));
}

#[test]
fn glb_roundtrip_preserves_geometry_uvs_and_has_no_cameras() {
    let a = snapshot(&source(), 0);
    let bytes = export_scene_glb(&a).unwrap();
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json: serde_json::Value = serde_json::from_slice(&bytes[20..20 + json_len]).unwrap();
    assert!(json.get("cameras").is_none());
    assert!(json.get("animations").is_none());
    assert!(
        json["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|n| n.get("camera").is_none())
    );
    let roundtrip =
        load_glb_mesh_data_from_bytes(std::path::Path::new("roundtrip.glb"), &bytes).unwrap();
    let expected: Vec<_> = a.meshes.iter().flat_map(|m| m.positions.clone()).collect();
    assert_eq!(roundtrip.positions, expected);
    assert_eq!(
        roundtrip.texcoords,
        a.meshes
            .iter()
            .flat_map(|m| m.uvs.iter().copied().map(Some))
            .collect::<Vec<_>>()
    );
    assert!(
        roundtrip
            .positions
            .iter()
            .all(|p| p[0] >= 2.5 && p[0] <= 3.5)
    );
}

#[test]
fn mesh_asset_levels_roundtrip_positions_indices_and_uvs_through_glb() {
    for subdivision in 0..=2 {
        let scene = snapshot(&mesh_source(subdivision), 0);
        let bytes = export_scene_glb(&scene).unwrap();
        let roundtrip =
            load_glb_mesh_data_from_bytes(std::path::Path::new("mesh.glb"), &bytes).unwrap();
        assert_eq!(roundtrip.positions, scene.meshes[0].positions);
        assert_eq!(roundtrip.indices, scene.meshes[0].indices);
        assert_eq!(
            roundtrip.texcoords,
            scene.meshes[0]
                .uvs
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn uv_check_detects_overlap_and_degeneracy_without_rejecting_export() {
    let mut a = snapshot(&source(), 0);
    let m = &mut a.meshes[0];
    m.indices = vec![0, 1, 2, 0, 1, 2, 0, 0, 0];
    m.uvs[0] = [0.1, 0.1];
    m.uvs[1] = [0.9, 0.1];
    m.uvs[2] = [0.1, 0.9];
    let check = check_scene_uvs(
        &a,
        &UvCheckOptions {
            resolution: 64,
            padding: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(check.report.meshes[0].overlap_pixels > 100);
    assert_eq!(check.report.meshes[0].degenerate_uv_triangles, 1);
    assert!(export_scene_glb(&a).is_ok());
}

#[test]
fn model_animation_is_sampled() {
    let script = source().replace(
        "node=\"camera\" property=\"position\"",
        "node=\"subject\" property=\"position\"",
    );
    let a = snapshot(&script, 0);
    let b = snapshot(&script, 30);
    assert_ne!(a.meshes[0].positions, b.meshes[0].positions);
    assert_eq!(a.topology_signature, b.topology_signature);
}

#[test]
#[ignore = "requires a native GPU"]
fn glb_roundtrip_keeps_vertex_color_brightness() {
    use base64::Engine;
    let src = source().replace("position={[3,2,1]}", "position={[0,0,0]}");
    let data = export_scene_glb(&snapshot(&src, 0)).unwrap();
    let uri = format!(
        "data:model/gltf-binary;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(data)
    );
    let exported = src
        .replace(
            "<PrimitiveAsset id=\"box\" shape=\"box\" size={[1,2,3]} material=\"paint\" />",
            &format!("<ModelAsset id=\"box\" src=\"{uri}\" />"),
        )
        .replace(
            "<Model id=\"subject\" asset=\"box\"",
            "<Model scaleMode=\"none\" id=\"subject\" asset=\"box\"",
        );
    pollster::block_on(async {
        let mut renderer = motionloom::SceneRenderer::new(motionloom::SceneRenderProfile::Gpu)
            .await
            .unwrap();
        let a = renderer
            .render_frame_gpu_readback(&parse_graph_script(&src).unwrap(), 0)
            .await
            .unwrap();
        let b = renderer
            .render_frame_gpu_readback(&parse_graph_script(&exported).unwrap(), 0)
            .await
            .unwrap();
        let error = a
            .as_raw()
            .iter()
            .zip(b.as_raw())
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum::<u64>() as f64
            / a.as_raw().len() as f64;
        assert!(error < 1.0, "roundtrip color error: {error}");
    });
}

#[test]
fn explicit_atlas_domain_detects_cross_mesh_overlap() {
    let mut a = snapshot(&source(), 0);
    a.meshes[0].indices = vec![0, 1, 2];
    a.meshes[0].uvs[0] = [0.1, 0.1];
    a.meshes[0].uvs[1] = [0.9, 0.1];
    a.meshes[0].uvs[2] = [0.1, 0.9];
    let mut duplicate = a.meshes[0].clone();
    for p in &mut duplicate.positions {
        p[0] += 3.0;
    }
    a.meshes.push(duplicate);
    let check = check_scene_uvs(
        &a,
        &UvCheckOptions {
            resolution: 64,
            padding: 2,
            atlas_meshes: vec![0, 1],
        },
    )
    .unwrap();
    assert_eq!(check.report.meshes[0].overlap_pixels, 0);
    assert!(check.report.meshes.last().unwrap().overlap_pixels > 100);
}

#[test]
#[ignore = "requires sibling S86 showcase"]
fn s86_geometry_baseline_and_static_glb_roundtrip() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../motionloom-example/showcase/s-000086/main.motionloom");
    let script = std::fs::read_to_string(&path).unwrap();
    let graph = parse_graph_script(&script).unwrap();
    let get = |frame| {
        pollster::block_on(extract_scene_geometry(
            &graph,
            &SceneGeometryOptions {
                scene_id: "S86AnimeHead".into(),
                frame,
                include_hidden: false,
            },
        ))
        .unwrap()
    };
    let a = get(0);
    let b = get(180);
    assert_eq!(a.meshes.len(), 30);
    assert_eq!(
        a.meshes.iter().map(|m| m.indices.len() / 3).sum::<usize>(),
        348966
    );
    assert_eq!(
        a.topology_signature,
        "topology-v1:bedc0bbe25896e7378be810524fd9d30bb606a7e1a7e63fe823ab80b8df03726"
    );
    assert_eq!(
        a.uv_signature,
        "uv-v1:7e931c7cf33f1efdb9e4fecb4c7388a7728cb87b9dff72d1b26af6633eff8de1"
    );
    assert_eq!(a.topology_signature, b.topology_signature);
    for (a, b) in a.meshes.iter().zip(&b.meshes) {
        assert_eq!(a.positions.len(), b.positions.len());
        assert!(a.uvs == b.uvs);
    }
    let bytes = export_scene_glb(&a).unwrap();
    let loaded = load_glb_mesh_data_from_bytes(std::path::Path::new("s86.glb"), &bytes).unwrap();
    assert_eq!(
        loaded.indices.len(),
        a.meshes.iter().map(|m| m.indices.len()).sum::<usize>()
    );
    let expected: Vec<_> = a
        .meshes
        .iter()
        .flat_map(|m| m.positions.iter().copied())
        .collect();
    assert!(
        loaded.positions == expected,
        "GLB coordinates must exactly match S86 snapshot"
    );
    assert_eq!(std::fs::read_to_string(path).unwrap(), script);
}

#[test]
fn mesh_editor_snapshot_preserves_authored_cage_and_rejects_procedural_assets() {
    let source = mesh_source(2).replace("<Model asset=", "<Model id=\"editable\" asset=");
    let graph = parse_graph_script(&source).unwrap();
    let snapshot = pollster::block_on(motionloom::experimental::mesh_edit_snapshot(
        &graph, "editable", 0,
    ))
    .unwrap();
    assert_eq!(snapshot["positions"].as_array().unwrap().len(), 4);
    assert_eq!(snapshot["faces"], serde_json::json!([[0, 1, 2, 3]]));
    assert_eq!(snapshot["assetId"], "mesh");
    assert!(snapshot["focal"].as_f64().unwrap() > 0.0);
    let moved = parse_graph_script(&source.replace(
        "id=\"editable\" asset=",
        "id=\"editable\" position={[1,0,0]} rotation={[0,0,30]} asset=",
    ))
    .unwrap();
    let moved_snapshot = pollster::block_on(motionloom::experimental::mesh_edit_snapshot(
        &moved, "editable", 0,
    ))
    .unwrap();
    assert_eq!(snapshot["positions"], moved_snapshot["positions"]);
    assert_ne!(snapshot["origin"], moved_snapshot["origin"]);
    assert_ne!(snapshot["basis"], moved_snapshot["basis"]);
    let procedural =
        parse_graph_script(&source.replace("asset=\"mesh\"", "asset=\"box\"").replace(
            "</Assets>",
            "<PrimitiveAsset id=\"box\" shape=\"box\" size={[1,1,1]} material=\"paint\" />\n</Assets>",
        ))
        .unwrap();
    assert!(
        pollster::block_on(motionloom::experimental::mesh_edit_snapshot(
            &procedural,
            "editable",
            0
        ))
        .is_err()
    );
}
