// =========================================
// =========================================
// crates/motionloom/tests/hair_cards.rs

use motionloom::{PrimitiveAssetNode, PrimitiveGeometry, parse_graph_script};

// Exercise the public parser and mesh path together, including expanded mirrors.
fn script(group: &str) -> String {
    format!(
        r##"<Graph fps="30" duration="1s" size={{[64,64]}}>
<Assets>
<MaterialAsset id="hair" baseColor="#776677" />
<HairAsset id="test" material="hair" space="asset_local">
<HairGroom>
<HairGroup id="bangs">
{group}
</HairGroup>
</HairGroom>
<HairRepresentations>
<HairCards id="cards" lengthSegments="24" widthSegments="6" thickness="0.02" tipShape="point" />
</HairRepresentations>
</HairAsset>
</Assets>
<Background color="#FFFFFF" />
<Present from="scene" />
</Graph>"##
    )
}

fn asset(group: &str) -> PrimitiveAssetNode {
    parse_graph_script(&script(group)).unwrap().assets[0]
        .primitive()
        .unwrap()
        .clone()
}

const GUIDE: &str = r#"<HairGuide id="left" normal={[0,0,1]} width="0.2" roll="12">
<HairPoint position={[-0.3,1,0]} />
<HairPoint position={[-0.35,0.5,0.1]} width="0.12" roll="20" />
<HairPoint position={[-0.4,0,0.2]} />
</HairGuide>"#;

#[test]
fn defaults_overrides_and_mirrors_resolve_without_extra_resources() {
    let a = asset(&format!(
        r#"<HairDefaults width="0.3" camber="0.15" stiffness="0.4" />
{GUIDE}
<HairMirror id="right" source="left" axis="x" />"#
    ));
    let PrimitiveGeometry::HairCards { guides, .. } = &a.geometry else {
        panic!()
    };
    assert_eq!(guides.len(), 2);
    assert_eq!(guides[0].points[0].width, 0.2);
    assert_eq!(guides[0].points[1].width, 0.12);
    assert_eq!(guides[0].points[2].width, 0.2);
    assert_eq!(guides[0].points[0].camber, 0.15);
    assert_eq!(guides[0].points[0].stiffness, 0.4);
    assert_eq!(guides[1].points[1].position, [0.35, 0.5, 0.1]);
    assert_eq!(guides[1].points[1].roll, -20.0);
    assert_eq!(guides[1].normal, Some([0.0, 0.0, 1.0]));
    let PrimitiveGeometry::HairCards { guides, .. } =
        asset(&GUIDE.replace(" width=\"0.2\"", "")).geometry
    else {
        panic!()
    };
    assert_eq!(guides[0].points[0].width, 0.1);
}

#[test]
fn invalid_authoring_reports_the_actual_problem() {
    for (group, expected) in [
        (
            GUIDE.replace("normal={[0,0,1]}", "normal={[0,0,0]}"),
            "normal must",
        ),
        (
            GUIDE.replace("normal={[0,0,1]}", "normal={[-0.05,-0.5,0.1]}"),
            "parallel",
        ),
        (
            format!("{GUIDE}\n<HairMirror id=\"right\" source=\"missing\" axis=\"x\" />"),
            "earlier",
        ),
        (
            format!("{GUIDE}\n<HairMirror id=\"right\" source=\"left\" axis=\"xy\" />"),
            "axis",
        ),
        (
            format!("{GUIDE}\n<HairMirror id=\"left\" source=\"left\" axis=\"x\" />"),
            "duplicate",
        ),
        (format!("{GUIDE}\n<HairDefaults width=\"0.3\" />"), "before"),
        (format!("<HairDefaults camber=\"2\" />\n{GUIDE}"), "range"),
        (GUIDE.replace("width=\"0.2\"", "width=\"0\""), "range"),
    ] {
        let err = parse_graph_script(&script(&group)).unwrap_err();
        assert!(err.to_string().contains(expected), "{err}");
    }
}

#[test]
fn mirror_surface_is_a_geometric_reflection_with_consistent_winding() {
    for explicit in [true, false] {
        let guide = if explicit {
            GUIDE.to_string()
        } else {
            GUIDE.replace(" normal={[0,0,1]}", "")
        };
        let a = asset(&format!(
            "<HairDefaults camber=\"0.25\" />\n{guide}\n<HairMirror id=\"right\" source=\"left\" axis=\"x\" />"
        ));
        let mesh = motionloom::experimental::generate_primitive_mesh(&a);
        for p in &mesh.positions {
            assert!(
                mesh.positions.iter().any(|q| (p[0] + q[0]).abs() < 1e-5
                    && (p[1] - q[1]).abs() < 1e-5
                    && (p[2] - q[2]).abs() < 1e-5),
                "unmirrored {p:?}"
            );
        }
        for tri in mesh.indices.chunks_exact(3) {
            let p = tri
                .iter()
                .map(|&i| mesh.positions[i as usize])
                .collect::<Vec<_>>();
            let a: [f32; 3] = std::array::from_fn(|i| p[1][i] - p[0][i]);
            let b: [f32; 3] = std::array::from_fn(|i| p[2][i] - p[0][i]);
            let n = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            assert!(n.iter().map(|v| v * v).sum::<f32>() > 1e-20);
            let vn = mesh.normals[tri[0] as usize].unwrap();
            assert!(n.iter().zip(vn).map(|(a, b)| a * b).sum::<f32>() >= -1e-7);
        }
    }
}

#[test]
fn old_point_explicit_syntax_still_parses_and_roundtrips() {
    let a = asset(
        r#"<HairGuide id="old">
<HairPoint position={[0,1,0]} width="0.2" />
<HairPoint position={[0,0,0]} width="0.1" />
</HairGuide>"#,
    );
    let mut json = serde_json::to_value(&a).unwrap();
    // Old serialized guides predate the optional root normal field.
    json["geometry"]["guides"][0]
        .as_object_mut()
        .unwrap()
        .remove("normal");
    let b: PrimitiveAssetNode = serde_json::from_value(json).unwrap();
    assert_eq!(a, b);
}

#[test]
fn authoring_schema_accepts_defaults_and_optional_point_width() {
    let source = script(&format!(
        "<HairDefaults camber=\"0.2\" />\n{GUIDE}\n<HairMirror id=\"right\" source=\"left\" axis=\"x\" />"
    ));
    let report = motionloom::api::analyze_motionloom_script(&source);
    assert!(report.parse_succeeded);
    for diagnostic in &report.diagnostics {
        let text = format!("{diagnostic:?}");
        assert!(
            !text.contains("HairPoint")
                && !text.contains("HairMirror")
                && !text.contains("HairDefaults"),
            "{text}"
        );
    }
}
