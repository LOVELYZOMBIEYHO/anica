// =========================================
// =========================================
// crates/motionloom/src/head_fitting/tests.rs

use super::*;

#[test]
fn component_proposal_patches_only_the_selected_eye_id() {
    let s = source().replace("</HeadAsset>", "<FaceLayout>\n<Eye id=\"one\" position={[-0.23,0.16,0]} width=\"0.34\" />\n<Eye id=\"two\" position={[0.23,0.16,0]} width=\"0.34\" />\n</FaceLayout>\n</HeadAsset>");
    let a = asset(&s, "head").unwrap();
    let mut b = a.clone();
    let PrimitiveGeometry::HeadSurface {
        face_layout: Some(layout),
        ..
    } = &mut b.geometry
    else {
        panic!()
    };
    layout.eyes[0].width = 0.4;
    layout.eyes[0].position[1] = 0.18;
    let changes = source::changes(&a, &b, "head");
    assert_eq!(changes.len(), 2);
    assert!(
        changes
            .iter()
            .all(|c| c.node_id == "one" && c.child_tag == "Eye")
    );
    let output = source::patch(&s, "head", &changes).unwrap();
    let parsed = asset(&output, "head").unwrap();
    assert_eq!(parsed.geometry, b.geometry);
    assert!(output.contains("<Eye id=\"two\" position={[0.23,0.16,0]} width=\"0.34\" />"));
}

fn source() -> String {
    r##"<!-- preserve this <HeadAsset id="head"> verbatim -->
<Graph fps={30} duration="1s" size={[64,64]}>
<Assets><MaterialAsset id="clay" baseColor="#BBAAAA" /><HeadAsset material="clay" id="head" archetype="humanoid" symmetry="x" segments="32" rings="24">
<HeadShape size={[0.8,1.4,0.9]} forehead="1" cheekWidth="1" jawWidth="1" chinLength="0" chinRoundness="0.5" />
<HeadMorph headWidth="1" headHeight="1" headDepth="1" />
</HeadAsset></Assets><Background color="#121212" /><Present from="scene" /></Graph>
<!-- preserve suffix -->"##.replace("><", ">\n<")
}
fn request() -> HeadReferenceSet {
    HeadReferenceSet {
        schema_version: "1.0".into(),
        target_asset_id: "head".into(),
        options: FitOptions {
            max_iterations: 12,
            allowed_parameters: vec!["HeadShape.width".into(), "HeadShape.depth".into()],
            ..Default::default()
        },
        references: [View::Front, View::Left, View::Back]
            .into_iter()
            .enumerate()
            .map(|(i, view)| HeadReference {
                id: format!("v{i}"),
                image_id: "host-image".into(),
                view,
                projection: "orthographic".into(),
                image_size: [1000, 1000],
                alignment: Alignment {
                    top: [500.0, 100.0],
                    bottom: [500.0, 900.0],
                    centerline: [[500.0, 100.0], [500.0, 900.0]],
                },
                landmarks: vec![],
                contours: vec![Contour {
                    id: "head_outline".into(),
                    closed: true,
                    confidence: 1.0,
                    weight: 1.0,
                    enabled: true,
                    points: (0..64)
                        .map(|j| {
                            let t = j as f64 * std::f64::consts::TAU / 64.0;
                            [500.0 + 260.0 * t.cos(), 500.0 + 400.0 * t.sin()]
                        })
                        .collect(),
                }],
            })
            .collect(),
    }
}

#[test]
fn fit_improves_and_preserves_source_and_locks() {
    let s = source();
    let mut r = request();
    r.options.locked_parameters = vec!["HeadShape.depth".into()];
    let p = fit_head_asset_to_references(&s, &r, || false).unwrap();
    assert!(p.after_metrics.objective < p.before_metrics.objective);
    let old = asset(&s, "head").unwrap();
    let new = asset(&p.candidate_dsl, "head").unwrap();
    assert_eq!(parameter(&old, 1), parameter(&new, 1));
    assert!(p.candidate_dsl.starts_with("<!-- preserve this"));
    assert!(p.candidate_dsl.ends_with("<!-- preserve suffix -->"));
    assert_eq!(apply_head_fit_proposal(&s, &p).unwrap(), p.candidate_dsl);
    assert!(apply_head_fit_proposal(&(s.clone() + " "), &p).is_err());
    let mut forged = p.clone();
    forged.candidate_dsl.push(' ');
    assert!(apply_head_fit_proposal(&s, &forged).is_err());
    let mut forged = p;
    forged.changes[0].before = serde_json::json!([1, 2, 3]);
    assert!(apply_head_fit_proposal(&s, &forged).is_err());
}
#[test]
fn cancellation_returns_original() {
    let s = source();
    let p = fit_head_asset_to_references(&s, &request(), || true).unwrap();
    assert_eq!(p.stop_reason, "cancelled");
    assert_eq!(p.candidate_dsl, s);
}

#[test]
fn explicit_head_is_comparable_but_parameter_fitting_is_rejected() {
    let explicit = r##"<Graph fps={30} duration="1s" size={[64,64]}>
<Assets>
<MaterialAsset id="clay" baseColor="#BBAAAA" />
<HeadAsset id="head" material="clay" archetype="humanoid" topology="explicit">
<HeadShape size={[1,1,1]} />
<HeadCage subdivision="0">
<Vertex position={[-0.5,-0.5,-0.5]} />
<Vertex position={[0.5,-0.5,-0.5]} />
<Vertex position={[0.5,0.5,-0.5]} />
<Vertex position={[-0.5,0.5,-0.5]} />
<Vertex position={[-0.5,-0.5,0.5]} />
<Vertex position={[0.5,-0.5,0.5]} />
<Vertex position={[0.5,0.5,0.5]} />
<Vertex position={[-0.5,0.5,0.5]} />
<Face indices={[0,3,2,1]} />
<Face indices={[4,5,6,7]} />
<Face indices={[0,1,5,4]} />
<Face indices={[1,2,6,5]} />
<Face indices={[3,7,6,2]} />
<Face indices={[0,4,7,3]} />
</HeadCage>
</HeadAsset>
</Assets>
<Background color="#121212" />
<Present from="scene" />
</Graph>"##;
    let mut reference = request();
    reference.references.truncate(1);
    evaluate_head_reference_fit(explicit, &reference).unwrap();
    let error = fit_head_asset_to_references(explicit, &reference, || false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("topology=explicit"));
}
#[test]
fn rejects_invalid_and_conflicting_data() {
    let mut r = request();
    r.references[1].id = r.references[0].id.clone();
    assert!(!validate_head_reference_set(&r).valid);
    let mut r = request();
    r.references[0].alignment.bottom = r.references[0].alignment.top;
    assert!(!validate_head_reference_set(&r).valid);
    let mut r = request();
    r.references[0].contours[0].points = vec![[0.0, 0.0]; 3];
    assert!(!validate_head_reference_set(&r).valid);
    let mut r = request();
    r.references[0].contours[0].confidence = f64::NAN;
    assert!(!validate_head_reference_set(&r).valid);
    let mut r = request();
    for (i, v) in r.references.iter_mut().enumerate() {
        v.landmarks.push(Landmark {
            id: "nose_tip".into(),
            pixel: [500.0, 400.0 + i as f64 * 100.0],
            confidence: 1.0,
            weight: 1.0,
            enabled: true,
        })
    }
    assert!(!validate_head_reference_set(&r).valid);
}
#[test]
fn normalization_handles_translation_scale_and_roll() {
    let r = request();
    let original = &r.references[0];
    let p = [610.0, 340.0];
    let want = validation::normalize(original, p);
    let transform = |p: [f64; 2]| {
        let t = 0.2f64;
        [
            p[0] * 0.7 * t.cos() - p[1] * 0.7 * t.sin() + 200.0,
            p[0] * 0.7 * t.sin() + p[1] * 0.7 * t.cos() + 80.0,
        ]
    };
    let mut v = original.clone();
    v.alignment.top = transform(v.alignment.top);
    v.alignment.bottom = transform(v.alignment.bottom);
    v.alignment.centerline = v.alignment.centerline.map(transform);
    assert!(validation::distance(want, validation::normalize(&v, transform(p))) < 1e-12);
}
#[test]
fn json_matches_typed_and_unsupported_marks_are_explicit() {
    let s = source();
    let mut r = request();
    r.references[0].landmarks.push(Landmark {
        id: "mouth_corner_left".into(),
        pixel: [500.0, 500.0],
        confidence: 1.0,
        weight: 1.0,
        enabled: true,
    });
    let typed = evaluate_head_reference_fit(&s, &r).unwrap();
    let json: FitMetrics = serde_json::from_str(
        &evaluate_head_reference_fit_json(&s, &serde_json::to_string(&r).unwrap()).unwrap(),
    )
    .unwrap();
    assert!((typed.objective - json.objective).abs() < 1e-12);
    assert_eq!(typed.views[0].landmarks[0].candidate, None);
    let partial = r#"{"preserveHeight":true,"maxIterations":2}"#;
    assert_eq!(
        serde_json::from_str::<FitOptions>(partial)
            .unwrap()
            .symmetry,
        "x"
    );
}

#[test]
fn semantic_face_curves_are_measured_from_face_layout() {
    let s = source()
        .replace(
            "<HeadMorph headWidth=\"1\" headHeight=\"1\" headDepth=\"1\" />",
            "<HeadMorph headWidth=\"1\" headHeight=\"1\" headDepth=\"1\" />\n\
<FaceLayout>
  <Eye id=\"eye_a\" position={[-0.23,0.16,0]} width=\"0.34\" opening=\"0.14\" tilt=\"0.02\" socketWidth=\"0.44\" socketHeight=\"0.28\" socketDepth=\"0.035\">
    <Iris id=\"iris_a\" radius=\"0.075\" pupilRadius=\"0.03\" />
  </Eye>
  <Eye id=\"eye_b\" position={[0.23,0.16,0]} width=\"0.34\" opening=\"0.14\" tilt=\"0.02\" socketWidth=\"0.44\" socketHeight=\"0.28\" socketDepth=\"0.035\">
    <Iris id=\"iris_b\" radius=\"0.075\" pupilRadius=\"0.03\" />
  </Eye>
  <Nose id=\"nose\" position={[0,0.0456,0]} length=\"0.22\" width=\"0.13\" projection=\"0.045\" />
  <Mouth id=\"mouth\" position={[0,-0.25,0]} width=\"0.34\" opening=\"0.04\" upperLip=\"0.025\" lowerLip=\"0.03\" muzzleLength=\"0\" muzzleWidth=\"0.30\" />
  <Ear id=\"ear_a\" position={[-0.4408,0.14,0.38]} width=\"0.1\" height=\"0.24\" depth=\"0.045\" />
  <Ear id=\"ear_b\" position={[0.4408,0.14,0.38]} width=\"0.1\" height=\"0.24\" depth=\"0.045\" />
</FaceLayout>",
        );
    let mut r = request();
    r.references[0].contours.extend([
        Contour {
            id: "upper_eyelid_left".into(),
            closed: false,
            points: (0..=8)
                .map(|i| [600.0 + i as f64 * 12.0, 430.0 - (i as f64 - 4.0).powi(2)])
                .collect(),
            confidence: 1.0,
            weight: 1.0,
            enabled: true,
        },
        Contour {
            id: "iris_left".into(),
            closed: true,
            points: (0..16)
                .map(|i| {
                    let a = i as f64 * std::f64::consts::TAU / 16.0;
                    [650.0 + 35.0 * a.cos(), 440.0 + 28.0 * a.sin()]
                })
                .collect(),
            confidence: 1.0,
            weight: 1.0,
            enabled: true,
        },
        Contour {
            id: "upper_lip".into(),
            closed: false,
            points: (0..=8)
                .map(|i| [450.0 + i as f64 * 12.0, 690.0 - (i as f64 - 4.0).powi(2)])
                .collect(),
            confidence: 1.0,
            weight: 1.0,
            enabled: true,
        },
    ]);
    let report = validate_head_reference_set(&r);
    assert!(report.valid, "{report:?}");
    let metrics = evaluate_head_reference_fit(&s, &r).unwrap();
    assert_eq!(metrics.views[0].curves.len(), 3);
    assert!(
        metrics.views[0]
            .curves
            .iter()
            .all(|c| c.candidate.is_some())
    );
}
#[test]
fn rejects_duplicate_target_and_preserves_inline_comments() {
    let s = source();
    let range = source::target_range(&s, "head").unwrap();
    let duplicated = s.replace("</Assets>", &format!("{}</Assets>", &s[range]));
    assert!(evaluate_head_reference_fit(&duplicated, &request()).is_err());
    let s = s.replace(
        "<HeadShape",
        "<!-- > <HeadShape size={[9,9,9]} /> -->\n<HeadShape",
    );
    let p = fit_head_asset_to_references(&s, &request(), || false).unwrap();
    assert!(
        p.candidate_dsl
            .contains("<!-- > <HeadShape size={[9,9,9]} /> -->")
    );
}

// Synthetic references come from the actual known convex mesh, with perturbations only in DSL.
#[test]
fn synthetic_mesh_recovery() {
    let truth = source().replace("[0.8,1.4,0.9]", "[1.0,1.4,1.1]");
    let mut r = request();
    let measurement = evaluate_head_reference_fit(&truth, &r).unwrap();
    for (v, m) in r.references.iter_mut().zip(measurement.views) {
        let mut points = m.candidate_contour;
        points.sort_by(|a, b| a[1].atan2(a[0]).total_cmp(&b[1].atan2(b[0])));
        v.contours[0].points = points
            .into_iter()
            .map(|p| [500.0 + p[0] * 800.0, 500.0 - p[1] * 800.0])
            .collect();
    }
    let p = fit_head_asset_to_references(&source(), &r, || false).unwrap();
    assert!(
        p.after_metrics.objective < p.before_metrics.objective * 0.65,
        "{} -> {}",
        p.before_metrics.objective,
        p.after_metrics.objective
    );
}

#[test]
fn actual_morph_changes_measurements_and_nose_is_surface_anchored() {
    let s=source().replace("headWidth=\"1\"","headWidth=\"1.2\"").replace("</HeadAsset>","<HeadFeature id=\"nose\" kind=\"nose_tip\" center={[0,-0.25,0.96]} size={[0.2,0.2,0.2]} amount=\"0.3\" />\n</HeadAsset>");
    let r = request();
    let original = evaluate_head_reference_fit(&source(), &r).unwrap();
    let changed = evaluate_head_reference_fit(&s, &r).unwrap();
    assert!(changed.width_over_height > original.width_over_height * 1.15);
    let a = asset(&s, "head").unwrap();
    let anchor = projection::anchors(&a).unwrap();
    let mesh = crate::world::primitive::generate_primitive_mesh(&a);
    assert_ne!(mesh.positions[anchor], [0.0, -0.25, 0.96]);
}

#[test]
fn inactive_constraints_are_ignored_and_front_back_conflicts_block() {
    let mut r = request();
    r.references[0].landmarks.push(Landmark {
        id: "nose_tip".into(),
        pixel: [0.0, 0.0],
        confidence: 1.0,
        weight: 1.0,
        enabled: false,
    });
    assert!(validate_head_reference_set(&r).valid);
    for p in &mut r.references[2].contours[0].points {
        p[0] = 500.0 + (p[0] - 500.0) * 0.65;
    }
    assert!(
        validate_head_reference_set(&r)
            .diagnostics
            .iter()
            .any(|d| d.code == "CONFLICTING_SILHOUETTE")
    );
}
