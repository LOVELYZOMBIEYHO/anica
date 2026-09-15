// =========================================
// =========================================
// crates/motionloom/src/mesh_reference/tests/mod.rs

use super::*;
use crate::ControlCageNode;
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use std::io::Cursor;

fn png() -> Vec<u8> {
    let mut image = RgbaImage::from_pixel(64, 48, Rgba([210, 215, 220, 255]));
    for y in 8..40 {
        for x in 16..48 {
            image[(x, y)] = Rgba([70, 50, 35, 255]);
        }
    }
    let mut output = Cursor::new(vec![]);
    DynamicImage::ImageRgba8(image)
        .write_to(&mut output, ImageFormat::Png)
        .unwrap();
    output.into_inner()
}

fn request() -> AnalyzeImageReferenceRequest {
    AnalyzeImageReferenceRequest {
        schema_version: MESH_REFERENCE_SCHEMA_VERSION.into(),
        image_id: "fixture".into(),
        view: "front".into(),
        segmentation: SegmentationOptions {
            mode: SegmentationMode::BackgroundColor,
            threshold: 40.0,
            ..Default::default()
        },
        requested_landmarks: vec![],
        analysis_profile: None,
        feature_hints: vec![],
        depth_hints: vec![],
    }
}

fn source() -> String {
    r##"<Graph fps={24} duration="1s" size={[320,240]}>
<Assets>
<MaterialAsset id="clay" baseColor="#888888" />
<MeshAsset id="shape" material="clay">
<Vertex position={[-1,-1,0]} />
<Vertex position={[1,-1,0]} />
<Vertex position={[1,1,0]} />
<Vertex position={[-1,1,0]} />
<Face indices={[0,1,2,3]} />
</MeshAsset>
</Assets>
<Background color="#111111" />
<Present from="scene" />
</Graph>"##
        .into()
}

fn evaluation_source() -> String {
    r##"<Graph fps={24} duration="1s" size={[64,48]}>
<Assets>
<MaterialAsset id="clay" baseColor="#888888" />
<MeshAsset id="shape" material="clay">
<Vertex position={[-1,-1,0.5]} />
<Vertex position={[1,-1,0.5]} />
<Vertex position={[1,1,0.5]} />
<Vertex position={[-1,1,0.5]} />
<Vertex position={[-1,-1,-0.5]} />
<Vertex position={[1,-1,-0.5]} />
<Vertex position={[1,1,-0.5]} />
<Vertex position={[-1,1,-0.5]} />
<Face indices={[0,3,2,1]} />
<Face indices={[4,5,6,7]} />
<Face indices={[0,1,5,4]} />
<Face indices={[1,2,6,5]} />
<Face indices={[3,7,6,2]} />
<Face indices={[0,4,7,3]} />
</MeshAsset>
</Assets>
<Background color="#111111" />
<Scene id="mesh_reference_test_scene">
<Timeline>
<Track id="main" space="3d">
<Sequence duration="1s">
<CompositeGroup id="stage" space="3d" depth="true">
<Camera3D id="camera" position={[0,0,5]} target={[0,0,0]} fov="35" />
<Model id="shape_model" asset="shape" position={[0,0,0]} rotation={[0,0,0]} />
</CompositeGroup>
</Sequence>
</Track>
</Timeline>
</Scene>
<Present from="mesh_reference_test_scene" />
</Graph>"##
        .into()
}

fn feature_evaluation_source(center_x: f32) -> String {
    format!(
        r##"<Graph fps={{24}} duration="1s" size={{[64,48]}}>
<Assets>
<MaterialAsset id="clay" baseColor="#888888" />
<MeshAsset id="shape" material="clay">
<Vertex position={{[-1,-1,0.5]}} />
<Vertex position={{[1,-1,0.5]}} />
<Vertex position={{[1,1,0.5]}} />
<Vertex position={{[-1,1,0.5]}} />
<Vertex position={{[-1,-1,-0.5]}} />
<Vertex position={{[1,-1,-0.5]}} />
<Vertex position={{[1,1,-0.5]}} />
<Vertex position={{[-1,1,-0.5]}} />
<Vertex position={{[{center_x},0,0.5]}} />
<Face indices={{[0,3,8]}} />
<Face indices={{[3,2,8]}} />
<Face indices={{[2,1,8]}} />
<Face indices={{[1,0,8]}} />
<Face indices={{[4,5,6,7]}} />
<Face indices={{[0,1,5,4]}} />
<Face indices={{[1,2,6,5]}} />
<Face indices={{[3,7,6,2]}} />
<Face indices={{[0,4,7,3]}} />
</MeshAsset>
</Assets>
<Background color="#111111" />
<Scene id="mesh_reference_test_scene">
<Timeline>
<Track id="main" space="3d">
<Sequence duration="1s">
<CompositeGroup id="stage" space="3d" depth="true">
<Camera3D id="camera" position={{[0,0,5]}} target={{[0,0,0]}} fov="35" />
<Model id="shape_model" asset="shape" position={{[0,0,0]}} rotation={{[0,0,0]}} />
</CompositeGroup>
</Sequence>
</Track>
</Timeline>
</Scene>
<Present from="mesh_reference_test_scene" />
</Graph>"##
    )
}

#[test]
fn analysis_extracts_mask_contour_regions_and_landmarks() {
    let result = analyze_image_reference(&png(), &request()).unwrap();
    assert_eq!(result.image_size, [64, 48]);
    assert!(result.regions.iter().any(|region| region.area >= 32 * 32));
    assert!(result.contours.iter().any(|contour| contour.closed));
    assert_eq!(result.landmarks.len(), 4);
    assert_eq!(decode_mask(&result.foreground_mask).unwrap().len(), 64 * 48);
    assert!(!analysis_mask_png(&result).unwrap().is_empty());
    assert!(!analysis_overlay_png(&png(), &result).unwrap().is_empty());
}

#[test]
fn analysis_preserves_semantic_feature_hints_and_builds_internal_edges() {
    let mut request = request();
    request.analysis_profile = Some("human_head_v1".into());
    request.feature_hints.push(ReferenceFeatureHint {
        id: "nose_tip".into(),
        kind: ReferenceFeatureKind::Point,
        points: vec![[17.0, 20.0]],
        semantic_label: Some("nose_tip".into()),
        binding: Some(FeatureBinding::Vertex { vertex: 0 }),
        confidence: 0.9,
        snap_radius: 4,
    });
    let result = analyze_image_reference(&png(), &request).unwrap();
    assert_eq!(result.features.len(), 1);
    assert_eq!(result.features[0].id, "nose_tip");
    assert!(result.internal_edge_mask.is_some());
}

#[test]
fn mask_metrics_report_a_shift() {
    let mut a = vec![false; 20 * 20];
    let mut b = vec![false; 20 * 20];
    for y in 5..15 {
        for x in 5..15 {
            a[y * 20 + x] = true;
        }
        for x in 7..17 {
            b[y * 20 + x] = true;
        }
    }
    let (iou, edge) = evaluation::evaluate_masks_for_test(&a, &b, 20, 20);
    assert!((0.6..0.8).contains(&iou));
    assert!(edge > 0.0);
}

#[test]
fn proposal_is_fingerprint_guarded_and_source_preserving() {
    let source = source();
    let cage = proposal::mesh_asset(&source, "shape").unwrap();
    let proposal = MeshAssetProposal {
        schema_version: MESH_REFERENCE_SCHEMA_VERSION.into(),
        source_fingerprint: mesh_source_fingerprint(&source),
        topology_signature: mesh_topology_signature(&cage),
        camera_fingerprint: None,
        reference_set_fingerprint: None,
        metric_profile_fingerprint: None,
        evaluation_fingerprint: None,
        target_asset_id: "shape".into(),
        reason: "widen left edge".into(),
        changes: vec![MeshVertexChange {
            vertex: 0,
            before: [-1.0, -1.0, 0.0],
            after: [-1.02, -1.0, 0.0],
            confidence: 0.9,
            evidence_views: vec!["front".into()],
        }],
        validation: MeshProposalValidationOptions {
            allow_boundary: true,
            allow_multiple_components: false,
            max_move_relative_to_bounds: 0.08,
            ..Default::default()
        },
    };
    let result = apply_mesh_asset_proposal(&source, &proposal).unwrap();
    assert!(result.source.contains("position={[-1.02, -1, 0]}"));
    assert!(result.source.contains("<Face indices={[0,1,2,3]} />"));
    let mut stale = proposal;
    stale.source_fingerprint = "stale".into();
    assert!(apply_mesh_asset_proposal(&source, &stale).is_err());
}

#[test]
fn topology_rejects_degenerate_and_non_manifold_faces() {
    let cage = ControlCageNode {
        positions: vec![
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., -1., 0.],
        ],
        uvs: vec![[0., 0.]; 5],
        pinned: vec![false; 5],
        faces: vec![vec![0, 1, 2], vec![1, 0, 3], vec![0, 1, 4], vec![0, 0, 1]],
        subdivision: 0,
    };
    let report = validate_mesh_topology(&cage, &MeshProposalValidationOptions::default());
    assert!(!report.valid);
    assert_eq!(report.non_manifold_edges, 1);
    assert_eq!(report.degenerate_faces, vec![3]);
}

#[test]
fn topology_detects_winding_and_non_coplanar_self_intersection() {
    let winding = ControlCageNode {
        positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]],
        uvs: vec![[0., 0.]; 4],
        pinned: vec![false; 4],
        faces: vec![vec![0, 1, 2], vec![0, 1, 3]],
        subdivision: 0,
    };
    let options = MeshProposalValidationOptions {
        allow_boundary: true,
        allow_multiple_components: true,
        ..Default::default()
    };
    assert_eq!(
        validate_mesh_topology(&winding, &options).inconsistent_winding_edges,
        1
    );

    let crossing = ControlCageNode {
        positions: vec![
            [-1., -1., 0.],
            [1., -1., 0.],
            [0., 1., 0.],
            [0., 0., -1.],
            [0., 0.5, 1.],
            [0., -0.5, 1.],
        ],
        uvs: vec![[0., 0.]; 6],
        pinned: vec![false; 6],
        faces: vec![vec![0, 1, 2], vec![3, 4, 5]],
        subdivision: 0,
    };
    assert!(
        !validate_mesh_topology(&crossing, &options)
            .self_intersections
            .is_empty()
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn evaluation_projects_a_real_meshasset_runtime_snapshot() {
    let analysis = analyze_image_reference(&png(), &request()).unwrap();
    let request = MeshReferenceSet {
        schema_version: MESH_REFERENCE_SCHEMA_VERSION.into(),
        target_asset_id: "shape".into(),
        target_model_id: "shape_model".into(),
        references: vec![MeshReferenceView {
            id: "front".into(),
            frame: 0,
            analysis,
        }],
        options: MeshReferenceOptions::default(),
    };
    let evaluation = pollster::block_on(evaluate_mesh_asset_reference(
        &evaluation_source(),
        &request,
    ))
    .unwrap();
    assert_eq!(evaluation.views.len(), 1);
    assert!(evaluation.objective.is_finite());
    assert!((0.0..=1.0).contains(&evaluation.views[0].mask_iou));
    assert!(!evaluation.source_fingerprint.is_empty());
    assert!(!evaluation.camera_fingerprint.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn feature_metrics_reject_an_internal_deformation_with_the_same_silhouette() {
    let mut analysis = analyze_image_reference(&png(), &request()).unwrap();
    analysis.features = vec![ReferenceFeature {
        id: "internal_anchor".into(),
        kind: ReferenceFeatureKind::Point,
        points: vec![[32.0, 24.0]],
        semantic_label: Some("internal_anchor".into()),
        binding: Some(FeatureBinding::Vertex { vertex: 8 }),
        confidence: 1.0,
    }];
    let build_request = |analysis: ImageReferenceAnalysis| MeshReferenceSet {
        schema_version: MESH_REFERENCE_SCHEMA_VERSION.into(),
        target_asset_id: "shape".into(),
        target_model_id: "shape_model".into(),
        references: vec![MeshReferenceView {
            id: "front".into(),
            frame: 0,
            analysis,
        }],
        options: MeshReferenceOptions::default(),
    };
    let first = pollster::block_on(evaluate_mesh_asset_reference(
        &feature_evaluation_source(0.0),
        &build_request(analysis.clone()),
    ))
    .unwrap();
    analysis.features[0].points = first.views[0].features[0].candidate_points.clone();
    let good = pollster::block_on(evaluate_mesh_asset_reference(
        &feature_evaluation_source(0.0),
        &build_request(analysis.clone()),
    ))
    .unwrap();
    let bad = pollster::block_on(evaluate_mesh_asset_reference(
        &feature_evaluation_source(0.35),
        &build_request(analysis),
    ))
    .unwrap();
    assert!((good.views[0].mask_iou - bad.views[0].mask_iou).abs() < 1e-6);
    assert!(bad.views[0].features[0].mean_distance_px.unwrap() > 1.0);
    assert!(bad.objective > good.objective);
}

#[test]
fn json_contract_rejects_unknown_fields() {
    let mut value = serde_json::to_value(request()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("surprise".into(), true.into());
    assert!(serde_json::from_value::<AnalyzeImageReferenceRequest>(value).is_err());
}

#[test]
fn schema_catalog_names_all_llm_entry_points() {
    let schema = mesh_reference_schema_json();
    assert!(schema.contains("analyzeImageReference"));
    assert!(schema.contains("evaluateMeshAssetReference"));
    assert!(schema.contains("applyMeshAssetProposal"));
}
