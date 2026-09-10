// =========================================
// =========================================
// crates/motionloom/tests/control_cages.rs

use motionloom::{PrimitiveGeometry, parse_graph_script};

#[test]
fn eye_supports_optional_iris_and_multiple_eyeliners() {
    let old = "<Eye id=\"eye_a\" position={[-0.316,-0.113,0]} width=\"0.380\" opening=\"0.210\" tilt=\"0\" socketWidth=\"0.550\" socketHeight=\"0.510\" socketDepth=\"0.139\" />";
    let new = r#"<Eye id="eye_a" position={[-0.316,-0.113,0]} width="0.380" opening="0.210" tilt="-4" socketWidth="0.550" socketHeight="0.510" socketDepth="0.139">
  <Texture asset="sclera" />
  <Iris id="iris_a" position={[0.01,-0.02,0.003]} shape="ellipse" scale={[0.8,1.2]} radius="0.082" pupilRadius="0.032">
    <Texture asset="iris_image" />
  </Iris>
  <Eyeliner id="upper" edge="upper" thickness="0.014" span={[0,1]} taper={[0.85,0.2]} extension={[0,0.028]} tipLift={[0,0.012]}>
    <Texture asset="liner_image" />
  </Eyeliner>
  <Eyeliner id="lower" edge="lower" thickness="0.003" span={[0.12,0.88]} taper={[0.5,0.1]} extension={[0,0]} tipLift={[0,0]} />
</Eye>"#;
    let source = s86_head()
        .replace("<Assets>", "<Assets>\n<ImageAsset id=\"sclera\" src=\"sclera.png\" />\n<ImageAsset id=\"iris_image\" src=\"iris.png\" />\n<ImageAsset id=\"liner_image\" src=\"liner.png\" />")
        .replacen(old, new, 1);
    let graph = parse_graph_script(&source).unwrap();
    let asset = graph.assets.iter().find_map(|a| a.primitive()).unwrap();
    let PrimitiveGeometry::HeadSurface {
        face_layout: Some(layout),
        ..
    } = &asset.geometry
    else {
        panic!()
    };
    let iris = layout.eyes[0].iris.as_ref().unwrap();
    assert_eq!(iris.id, "iris_a");
    assert_eq!(iris.position, [0.01, -0.02, 0.003]);
    assert_eq!(iris.shape, "ellipse");
    assert_eq!(iris.scale, [0.8, 1.2]);
    assert_eq!(layout.eyes[0].eyeliners.len(), 2);
    assert_eq!(layout.eyes[1].iris, None);
    let mesh = motionloom::experimental::generate_primitive_mesh(asset);
    for id in ["eye_a", "iris_a", "iris_a_pupil", "upper", "lower"] {
        assert!(
            mesh.mesh_names
                .iter()
                .any(|name| name.as_deref() == Some(id))
        );
    }
    for id in ["iris_a_pupil", "lower"] {
        let part = mesh
            .mesh_names
            .iter()
            .position(|name| name.as_deref() == Some(id))
            .unwrap();
        let material = mesh
            .triangles
            .iter()
            .find(|triangle| triangle.mesh == Some(part))
            .and_then(|triangle| triangle.material)
            .unwrap();
        assert!(mesh.materials[material].base_color_texture.is_some());
    }
}

#[test]
fn removed_flat_iris_and_lid_attributes_are_rejected() {
    for attribute in [
        "irisRadius",
        "irisOcclusion",
        "upperLidFold",
        "cornerTaper",
        "eyelidRim",
    ] {
        let source = s86_head().replacen(
            "width=\"0.380\"",
            &format!("width=\"0.380\" {attribute}=\"0.1\""),
            1,
        );
        assert!(
            parse_graph_script(&source)
                .unwrap_err()
                .message
                .contains(attribute)
        );
    }
}

#[test]
fn iris_and_eyeliner_validation_is_strict() {
    let add = |child: &str| {
        s86_head().replacen(
            " />\n  <Eye id=\"eye_b\"",
            &format!(">\n{child}\n</Eye>\n  <Eye id=\"eye_b\""),
            1,
        )
    };
    assert!(
        parse_graph_script(&add(
            "<Iris id=\"iris\" radius=\"0.08\" pupilRadius=\"0.09\" />"
        ))
        .is_err()
    );
    assert!(parse_graph_script(&add("<Eyeliner id=\"line\" edge=\"side\" />")).is_err());
    assert!(
        parse_graph_script(&add(
            "<Eyeliner id=\"line\" edge=\"upper\" span={[0.8,0.2]} />"
        ))
        .is_err()
    );
    assert!(parse_graph_script(&add("<Iris id=\"iris\" shape=\"star\" />")).is_err());
    assert!(
        parse_graph_script(&add("<Iris id=\"iris\" shape=\"ellipse\" scale={[1,0]} />")).is_err()
    );
}

#[test]
fn eyebrows_generate_attached_ribbons_and_inherit_head_material() {
    let source = s86_head().replace("</FaceLayout>",
        "<Eyebrow id=\"brow\" position={[-0.316,0.08,0]} width=\"0.3\" thickness=\"0.018\" arch=\"0.035\" tilt=\"0\" />\n</FaceLayout>");
    let build = |s: &str| {
        let g = parse_graph_script(s).unwrap();
        motionloom::experimental::generate_primitive_mesh(
            g.assets.iter().find_map(|a| a.primitive()).unwrap(),
        )
    };
    let base = build(s86_head());
    let brow = build(&source);
    let start = base.positions.len();
    assert_eq!(brow.positions.len() - start, 75);
    assert_eq!(brow.materials.len(), base.materials.len());
    assert!(
        brow.triangles[base.triangles.len()..]
            .iter()
            .all(|t| t.material == Some(0))
    );
    assert!((brow.positions[start + 12][1] - brow.positions[start][1] - 0.035).abs() < 1e-5);
    assert!((brow.positions[start + 24][0] - brow.positions[start][0] - 0.3).abs() < 1e-5);
    let moved = build(&source.replace("[-0.316,0.08,0]", "[-0.216,0.08,0]"));
    assert!((moved.positions[start][0] - brow.positions[start][0] - 0.1).abs() < 1e-5);
    assert_ne!(moved.positions[start][2], brow.positions[start][2]);
    assert_eq!(moved.texcoords, brow.texcoords);
    assert_eq!(&base.positions[..], &brow.positions[..start]);
}

#[test]
fn eyebrow_texture_transforms_uv_without_changing_ribbon_shape() {
    let source = s86_head().replace("<Assets>", "<Assets>\n<ImageAsset id=\"brow_image\" src=\"brow.png\" />")
        .replace("</FaceLayout>", "<Eyebrow id=\"brow\" arch=\"-0.01\"><Texture asset=\"brow_image\" /></Eyebrow>\n</FaceLayout>");
    // Keep tags on separate lines, as required by the line-oriented parser.
    let source = source
        .replace("><Texture", ">\n<Texture")
        .replace("/></Eyebrow>", "/>\n</Eyebrow>");
    let build = |s: &str| {
        let g = parse_graph_script(s).unwrap();
        motionloom::experimental::generate_primitive_mesh(
            g.assets.iter().find_map(|a| a.primitive()).unwrap(),
        )
    };
    let a = build(&source);
    let b = build(&source.replace(
        "asset=\"brow_image\"",
        "asset=\"brow_image\" offset={[0.1,0]} scale={[1,1.2]} rotation=\"15\"",
    ));
    assert_eq!(a.positions, b.positions);
    assert_ne!(a.texcoords, b.texcoords);
    assert_eq!(a.materials.last().unwrap().name.as_deref(), Some("brow"));
    assert!(parse_graph_script(&source.replace("arch=\"-0.01\"", "thickness=\"0\"")).is_err());
    assert!(
        parse_graph_script(&source.replace("asset=\"brow_image\"", "asset=\"missing\"")).is_err()
    );
}

#[test]
fn face_layout_rejects_flat_attributes_and_duplicate_component_ids() {
    let old = s86_head().replace("<FaceLayout>", "<FaceLayout eyeWidth=\"0.35\">");
    assert!(
        parse_graph_script(&old)
            .unwrap_err()
            .message
            .contains("eyeWidth")
    );
    let duplicate = s86_head().replace("id=\"eye_b\"", "id=\"eye_a\"");
    assert!(
        parse_graph_script(&duplicate)
            .unwrap_err()
            .message
            .contains("duplicate")
    );
}

#[test]
fn third_eye_generates_a_closed_independent_socket() {
    let third = s86_head().replace("</FaceLayout>", "<Eye id=\"third\" position={[0,0.5,0]} width=\"0.12\" opening=\"0.08\" socketWidth=\"0.22\" socketHeight=\"0.18\" />\n</FaceLayout>");
    let graph = parse_graph_script(&third).unwrap();
    let asset = graph.assets[0].primitive().unwrap();
    let report = motionloom::inspect_control_cage(asset).unwrap();
    assert_eq!(report.open_edges, 0);
    assert_eq!(report.non_manifold_edges, 0);
    let PrimitiveGeometry::HeadSurface {
        face_layout: Some(layout),
        ..
    } = &asset.geometry
    else {
        panic!()
    };
    assert_eq!(layout.eyes.len(), 3);
    assert_eq!(layout.eyes[2].id, "third");
}

#[test]
fn nested_eye_texture_moves_with_its_component_and_keeps_uvs() {
    let mut source = s86_head().replace(
        "<Assets>",
        "<Assets>\n<ImageAsset id=\"eye_image\" src=\"eye.png\" />",
    );
    let begin = source.find("<Eye id=\"eye_a\"").unwrap();
    let end = begin + source[begin..].find("/>").unwrap();
    source.replace_range(end..end + 2, ">\n<Texture asset=\"eye_image\" />\n</Eye>");
    let moved = source.replacen("[-0.316,-0.113,0]", "[-0.326,-0.113,0]", 1);
    let mesh = |s: &str| {
        let g = parse_graph_script(s).unwrap();
        motionloom::experimental::generate_primitive_mesh(
            g.assets.iter().find_map(|a| a.primitive()).unwrap(),
        )
    };
    let a = mesh(&source);
    let b = mesh(&moved);
    let part_bounds = |m: &motionloom::GlbMeshData| {
        let part = m
            .mesh_names
            .iter()
            .position(|n| n.as_deref() == Some("eye_a"))
            .unwrap();
        let indices: Vec<_> = m
            .triangles
            .iter()
            .filter(|t| t.mesh == Some(part))
            .flat_map(|t| t.indices)
            .collect();
        let x = indices
            .iter()
            .map(|&i| m.positions[i as usize][0])
            .fold(f32::INFINITY, f32::min);
        let uv: Vec<_> = indices.iter().map(|&i| m.texcoords[i as usize]).collect();
        (x, uv)
    };
    let (ax, auv) = part_bounds(&a);
    let (bx, buv) = part_bounds(&b);
    assert!((bx - ax + 0.01).abs() < 1e-5);
    assert_eq!(auv, buv);
    let smaller = mesh(&source.replacen("width=\"0.380\"", "width=\"0.190\"", 1));
    let (sx, suv) = part_bounds(&smaller);
    assert!((sx - ax - 0.095).abs() < 1e-5);
    assert_eq!(auv, suv);
    assert!(
        parse_graph_script(&source.replace("asset=\"eye_image\"", "asset=\"missing\"")).is_err()
    );
}

#[test]
fn overlapping_eye_patches_are_rejected_before_meshing() {
    let overlapping = s86_head().replace("[0.316,-0.113,0]", "[-0.316,-0.113,0]");
    assert!(
        parse_graph_script(&overlapping)
            .unwrap_err()
            .message
            .contains("overlap")
    );
}

fn s86_head() -> &'static str {
    r##"<Graph fps={30} duration="1s" size={[64,64]}>
<Assets>
<MaterialAsset id="clay" baseColor="#aaaaaa" />
<HeadAsset id="s86_head" material="clay" archetype="humanoid" variant="anime" symmetry="x" topology="facialCage">
<HeadShape size={[1.50,1.833,1.67]} forehead="1" cheekWidth="1" jawWidth="1" chinLength="0" chinRoundness="0.6" />
<FacialCage generatorVersion="1" segments="192" profileSegments="96" samplesPerSection="6" subdivision="1" orbitalRings="10" mouthRings="8" preserveProfile="true" uvMode="fallbackXY" />
<HeadProfile>
<HeadSection id="chin_point" at="-0.833" width="0.018" frontDepth="0.462" backDepth="0.450" />
<HeadSection id="chin_side" at="-0.793" width="0.217" frontDepth="0.500" backDepth="-0.466" />
<HeadSection id="jawline" at="-0.714" width="0.491" frontDepth="0.537" backDepth="-0.454" />
<HeadSection id="jaw_angle" at="-0.596" width="0.810" frontDepth="0.609" backDepth="-0.434" />
<HeadSection id="cheek_transition" at="-0.438" width="1.051" frontDepth="0.690" backDepth="-0.538" />
<HeadSection id="cheekbone" at="-0.263" width="1.221" frontDepth="0.638" backDepth="-0.688" />
<HeadSection id="temple" at="-0.050" width="1.301" frontDepth="0.657" backDepth="-0.813" />
<HeadSection id="forehead" at="0.200" width="1.398" frontDepth="0.726" backDepth="-0.861" />
<HeadSection id="crown_transition" at="0.420" width="1.500" frontDepth="0.710" backDepth="-0.840" />
</HeadProfile>
<HeadDome start="0.200" top="1.0" centerDepth="-0.07" frontRadius="0.80" backRadius="0.80" samples="80" />
<FaceLayout>
  <Eye id="eye_a" position={[-0.316,-0.113,0]} width="0.380" opening="0.210" tilt="0" socketWidth="0.550" socketHeight="0.510" socketDepth="0.139" />
  <Eye id="eye_b" position={[0.316,-0.113,0]} width="0.380" opening="0.210" tilt="0" socketWidth="0.550" socketHeight="0.510" socketDepth="0.139" />
  <Nose id="nose" position={[0,-0.363,0]} length="0.180" width="0.100" projection="0.060" />
  <Mouth id="mouth" position={[0,-0.561,0]} width="0.086" opening="0.014" upperLip="0.025" lowerLip="0.03" muzzleLength="0" muzzleWidth="0.3" />
</FaceLayout>
<HeadMorph headWidth="1" headHeight="1" headDepth="1" faceWidth="1" faceHeight="1" jawWidth="1" muzzleLength="1" featureScale="1" />
</HeadAsset>
</Assets>
<Scene id="review">
<Timeline>
<Track id="world" space="3d">
<Sequence duration="1s">
<CompositeGroup id="stage" space="3d" depth="true">
<Camera3D position={[0,0,4]} target={[0,0,0]} />
<Model id="model" asset="s86_head" />
</CompositeGroup>
</Sequence>
</Track>
</Timeline>
</Scene>
<Present from="review" />
</Graph>"##
}

#[test]
fn facial_cage_v1_reproduces_s86_topology_baseline() {
    let graph = parse_graph_script(s86_head()).unwrap();
    let asset = graph.assets[0].primitive().unwrap();
    let cage = motionloom::experimental::generated_control_cage(asset).unwrap();
    assert_eq!(cage.positions.len(), 25_024);
    assert_eq!(cage.faces.len(), 25_336);
    assert_eq!(cage.pinned.iter().filter(|&&pin| pin).count(), 22_596);
    assert_eq!(cage.subdivision, 1);
    assert_eq!(cage.positions[24], [0.006364, -0.833, 0.460243]);
    assert!(cage.faces.iter().any(|f| f == &[7672, 7864, 7865, 7673]));
    let report = motionloom::inspect_control_cage(asset).unwrap();
    assert_eq!(report.control_vertices, 25_024);
    assert_eq!(report.control_faces, 25_336);
    assert_eq!(report.open_edges, 0);
    assert_eq!(report.non_manifold_edges, 0);
    assert_eq!(report.uv_source, "fallback_xy");
}

#[test]
fn explicit_nose_height_is_independent_from_eye_height() {
    let moved_eyes = s86_head().replace(",-0.113,0]", ",0.050,0]");
    let original = parse_graph_script(s86_head()).unwrap();
    let changed = parse_graph_script(&moved_eyes).unwrap();
    let original_asset = original.assets[0].primitive().unwrap();
    let changed_asset = changed.assets[0].primitive().unwrap();
    let original_cage = motionloom::experimental::generated_control_cage(original_asset).unwrap();
    let changed_cage = motionloom::experimental::generated_control_cage(changed_asset).unwrap();

    // Compare the central nasal surface, outside both orbital patches.
    let nose_peak = |positions: &[[f32; 3]]| {
        positions
            .iter()
            .filter(|p| p[0].abs() < 0.02 && (p[1] + 0.363).abs() < 0.03)
            .map(|p| p[2])
            .reduce(f32::max)
            .unwrap()
    };
    assert!(
        (nose_peak(&original_cage.positions) - nose_peak(&changed_cage.positions)).abs() < 1e-6
    );

    let PrimitiveGeometry::HeadSurface { face_layout, .. } = &changed_asset.geometry else {
        panic!("expected head surface");
    };
    let face = face_layout.as_ref().unwrap();
    assert_eq!(face.eyes[0].position[1], 0.050);
    assert_eq!(face.noses[0].position[1], -0.363);
}

#[test]
fn eye_socket_depth_scales_the_complete_orbital_profile() {
    let shallow_source = s86_head().replace("socketDepth=\"0.139\"", "socketDepth=\"0.015\"");
    let deep = parse_graph_script(s86_head()).unwrap();
    let shallow = parse_graph_script(&shallow_source).unwrap();
    let deep_cage =
        motionloom::experimental::generated_control_cage(deep.assets[0].primitive().unwrap())
            .unwrap();
    let shallow_cage =
        motionloom::experimental::generated_control_cage(shallow.assets[0].primitive().unwrap())
            .unwrap();

    assert_eq!(deep_cage.positions.len(), shallow_cage.positions.len());
    let largest_orbital_change = deep_cage
        .positions
        .iter()
        .zip(&shallow_cage.positions)
        .filter(|(deep, _)| (deep[0].abs() - 0.316).abs() < 0.3 && (deep[1] + 0.113).abs() < 0.3)
        .map(|(deep, shallow)| shallow[2] - deep[2])
        .reduce(f32::max)
        .unwrap();
    assert!(largest_orbital_change > 0.08);
}

#[test]
fn eye_z_moves_the_eye_component_without_extruding_the_skin_patch() {
    let moved_source = s86_head().replacen(
        "position={[-0.316,-0.113,0]}",
        "position={[-0.316,-0.113,0.25]}",
        1,
    );
    let original = parse_graph_script(s86_head()).unwrap();
    let moved = parse_graph_script(&moved_source).unwrap();
    let original_asset = original.assets[0].primitive().unwrap();
    let moved_asset = moved.assets[0].primitive().unwrap();
    let original_cage = motionloom::experimental::generated_control_cage(original_asset).unwrap();
    let moved_cage = motionloom::experimental::generated_control_cage(moved_asset).unwrap();
    assert_eq!(original_cage.positions, moved_cage.positions);
    assert_eq!(original_cage.faces, moved_cage.faces);

    let original_mesh = motionloom::experimental::generate_primitive_mesh(original_asset);
    let moved_mesh = motionloom::experimental::generate_primitive_mesh(moved_asset);
    let eye_center_z = |mesh: &motionloom::GlbMeshData| {
        let part = mesh
            .mesh_names
            .iter()
            .position(|name| name.as_deref() == Some("eye_a"))
            .unwrap();
        let vertex = mesh
            .triangles
            .iter()
            .find(|triangle| triangle.mesh == Some(part))
            .unwrap()
            .indices[0];
        mesh.positions[vertex as usize][2]
    };
    assert!((eye_center_z(&moved_mesh) - eye_center_z(&original_mesh) - 0.25).abs() < 1e-5);
}

#[test]
fn removed_eye_cage_names_are_not_accepted() {
    let old = r#"<Graph fps={30} duration="1s" size={[64,64]}>
<Assets><EyeAsset id="eye" material="clay" subdivision="0"></EyeAsset></Assets>
<Present from="scene" />
</Graph>"#;
    assert!(parse_graph_script(old).is_err());
}

#[test]
fn facial_cage_is_known_to_the_authoring_schema() {
    let source = s86_head().replace("</FaceLayout>", "<Eyebrow id=\"brow\" width=\"0.3\" thickness=\"0.018\" arch=\"0.035\" tilt=\"5\" />\n</FaceLayout>");
    let report: serde_json::Value =
        serde_json::from_str(&motionloom::motionloom_analyze_script_json(&source)).unwrap();
    assert_eq!(report["summary"]["unknownTags"], 0);
    assert_eq!(report["summary"]["ignoredAttributes"], 0);
    let schema: serde_json::Value =
        serde_json::from_str(&motionloom::motionloom_dsl_schema_json()).unwrap();
    for tag in [
        "MeshAsset",
        "Vertex",
        "Face",
        "HeadAsset",
        "FacialCage",
        "HeadProfile",
        "HeadSection",
        "HeadDome",
        "HeadCage",
        "Eyebrow",
        "Iris",
        "Eyeliner",
    ] {
        assert!(
            schema["tags"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["tag"] == tag)
        );
    }
}

#[test]
fn generic_and_explicit_head_cages_share_the_same_ir() {
    let generic = r##"<Graph fps={30} duration="1s" size={[64,64]}>
<Assets>
<MaterialAsset id="clay" baseColor="#aaaaaa" />
<MeshAsset id="patch" material="clay" subdivision="1" subdivisionScheme="catmullClark">
<Vertex position={[-1,0,0]} pinned="true" />
<Vertex position={[1,0,0]} uv={[1,0]} pinned="true" />
<Vertex position={[1,1,0]} />
<Vertex position={[-1,1,0]} />
<Face indices={[0,1,2,3]} />
</MeshAsset>
</Assets>
<Scene id="review">
<Timeline>
<Track id="world" space="3d">
<Sequence duration="1s">
<CompositeGroup id="stage" space="3d" depth="true">
<Camera3D position={[0,0,4]} target={[0,0,0]} />
<Model id="model" asset="patch" />
</CompositeGroup>
</Sequence>
</Track>
</Timeline>
</Scene>
<Present from="review" />
</Graph>"##;
    let graph = parse_graph_script(generic).unwrap();
    assert!(matches!(
        graph.assets[0].primitive().unwrap().geometry,
        PrimitiveGeometry::Mesh { .. }
    ));

    let explicit = r##"<Graph fps={30} duration="1s" size={[64,64]}>
<Assets>
<MaterialAsset id="clay" baseColor="#aaaaaa" />
<HeadAsset id="head" material="clay" archetype="humanoid" topology="explicit">
<HeadShape size={[1,1,1]} />
<HeadCage subdivision="1">
<Vertex position={[-1,0,0]} pinned="true" />
<Vertex position={[1,0,0]} uv={[1,0]} pinned="true" />
<Vertex position={[1,1,0]} />
<Vertex position={[-1,1,0]} />
<Face indices={[0,1,2,3]} />
</HeadCage>
</HeadAsset>
</Assets>
<Scene id="review">
<Timeline>
<Track id="world" space="3d">
<Sequence duration="1s">
<CompositeGroup id="stage" space="3d" depth="true">
<Camera3D position={[0,0,4]} target={[0,0,0]} />
<Model id="model" asset="head" />
</CompositeGroup>
</Sequence>
</Track>
</Timeline>
</Scene>
<Present from="review" />
</Graph>"##;
    let graph = parse_graph_script(explicit).unwrap();
    let cage =
        motionloom::experimental::generated_control_cage(graph.assets[0].primitive().unwrap())
            .unwrap();
    assert_eq!(cage.positions.len(), 4);
    assert_eq!(cage.faces, vec![vec![0, 1, 2, 3]]);
}
